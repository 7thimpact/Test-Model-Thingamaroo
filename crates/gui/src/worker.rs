//! The background thread that owns the brain.
//!
//! Every operation that touches weights runs here, so the window never
//! freezes. The window sends [`Request`]s and receives [`Notice`]s; each
//! request maps onto one method of the core library's `Brain`.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use thingamaroo_core::brain::{
    Brain, BrainSummary, ChatOptions, ChatSession, ChatTurnReport, EvaluationReport,
    FeedbackReport, HistoryRecord, PretrainingSummary, RoundTripReport, SleepReport,
    SnapshotInformation,
};
use thingamaroo_core::cancellation::CancellationFlag;
use thingamaroo_core::configuration::{
    GenerationSettings, ModelConfiguration, RuntimeSettings, TrainingOptions,
};
use thingamaroo_core::error::{BrainError, BrainResult};
use thingamaroo_core::learning::ProgressEvent;
use thingamaroo_core::memory::{MemoryEntry, MemoryMatch};
use thingamaroo_core::model::{DeviceChoice, Reply};
use thingamaroo_core::text::SampleCorpus;

/// Something the window asks the worker to do.
#[derive(Debug, Clone)]
pub enum Request {
    /// Create a new, untrained brain and open it.
    CreateBrain {
        directory: PathBuf,
        model_configuration: Box<ModelConfiguration>,
        device_choice: DeviceChoice,
        seed: u64,
    },
    /// Open an existing brain.
    OpenBrain {
        directory: PathBuf,
        device_choice: DeviceChoice,
    },
    /// Close the open brain.
    CloseBrain,
    /// Write a file of simple generated English.
    WriteSampleCorpus {
        path: PathBuf,
        target_bytes: u64,
        seed: u64,
    },
    /// Pretrain on a corpus.
    Pretrain(TrainingOptions),
    /// Measure the slow weights on text files.
    Evaluate {
        corpus_paths: Vec<PathBuf>,
        batch_count: usize,
        batch_sequence_count: usize,
        seed: u64,
    },
    /// Show how text is grouped into patches.
    SplitPatches(String),
    /// Send text through the encoder and back through the decoder.
    RoundTrip(String),
    /// Continue a prompt without memory or learning.
    Generate {
        prompt: String,
        generation: GenerationSettings,
        is_tracing_thoughts: bool,
    },
    /// Start a new conversation.
    StartChat(ChatOptions),
    /// Send a message in the current conversation.
    ChatTurn(String),
    /// Replace the last reply with the right answer.
    CorrectLastReply(String),
    /// Mark the last reply as good.
    PraiseLastReply,
    /// Mark the last reply as bad.
    RejectLastReply,
    /// Reload the list of memories.
    RefreshMemories,
    /// Find the memories closest in meaning to a text.
    SearchMemories { query: String, limit: usize },
    /// Store a fact by hand.
    AddNote { text: String, importance: f64 },
    /// Delete memories by identifier.
    DeleteMemories(Vec<u64>),
    /// Delete every memory.
    ClearMemories,
    /// Run one sleep cycle.
    Sleep,
    /// Sleep if the schedule says it is time.
    SleepIfDue { idle: Duration },
    /// Save the current weights.
    CreateSnapshot {
        label: String,
        includes_slow_weights: bool,
    },
    /// Restore the weights held by a snapshot.
    RollBack(u64),
    /// Delete a snapshot.
    DeleteSnapshot(u64),
    /// Discard what the fast weights have learned.
    ResetFastWeights,
    /// Replace the adjustable behaviour.
    ReplaceSettings(Box<RuntimeSettings>),
    /// Reload the learning history.
    RefreshHistory,
}

/// Which kind of feedback a [`Notice::Feedback`] reports on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackKind {
    Correction,
    Praise,
    Rejection,
}

/// Something the worker tells the window.
#[derive(Debug, Clone)]
pub enum Notice {
    /// A brain is now open.
    BrainOpened {
        summary: Box<BrainSummary>,
        settings: Box<RuntimeSettings>,
    },
    /// No brain is open any more.
    BrainClosed,
    /// The brain's summary changed.
    Summary(Box<BrainSummary>),
    /// The brain's settings changed.
    Settings(Box<RuntimeSettings>),
    /// A long operation made progress.
    Progress(ProgressEvent),
    /// Pretraining ended.
    PretrainingFinished(PretrainingSummary),
    /// A held-out measurement is ready.
    Evaluation(EvaluationReport),
    /// A patch split is ready.
    Patches(Vec<String>),
    /// A round trip is ready.
    RoundTrip(RoundTripReport),
    /// More bytes of a generation or reply were written.
    WrittenBytes(Vec<u8>),
    /// A prompt continuation ended.
    GenerationFinished(Reply),
    /// A conversation started.
    ChatStarted(ChatOptions),
    /// A conversation turn ended.
    ChatTurnFinished(Box<ChatTurnReport>),
    /// Feedback on the last reply was applied.
    Feedback {
        kind: FeedbackKind,
        report: FeedbackReport,
        corrected_reply: Option<String>,
    },
    /// The list of memories.
    Memories(Vec<MemoryEntry>),
    /// The result of a memory search.
    MemoryMatches(Vec<MemoryMatch>),
    /// A sleep ended.
    SleepFinished(SleepReport),
    /// The list of snapshots.
    Snapshots(Vec<SnapshotInformation>),
    /// The learning history.
    History(Vec<HistoryRecord>),
    /// Something worth telling the user.
    Information(String),
    /// A request failed.
    Failure(String),
    /// The worker finished a request and is ready for the next one.
    Ready,
}

/// The window's handle on the worker thread.
pub struct WorkerHandle {
    requests: Sender<Request>,
    notices: Receiver<Notice>,
    cancellation: CancellationFlag,
}

impl WorkerHandle {
    /// Starts the worker thread. `wake_window` is called whenever a notice is ready.
    pub fn spawn(wake_window: impl Fn() + Send + 'static) -> Self {
        let (request_sender, request_receiver) = channel();
        let (notice_sender, notice_receiver) = channel();
        let cancellation = CancellationFlag::new();
        let mut worker = Worker {
            brain: None,
            session: None,
            cancellation: cancellation.clone(),
            notices: NoticeSender {
                sender: notice_sender,
                wake_window: Box::new(wake_window),
            },
        };
        std::thread::spawn(move || worker.serve(request_receiver));
        Self {
            requests: request_sender,
            notices: notice_receiver,
            cancellation,
        }
    }

    /// Queues a request.
    pub fn send(&self, request: Request) {
        // The worker only stops when the window closes, so a failed send can be ignored.
        let _ = self.requests.send(request);
    }

    /// Takes every notice that has arrived since the last call.
    pub fn take_notices(&self) -> Vec<Notice> {
        self.notices.try_iter().collect()
    }

    /// Asks the running operation to stop at its next safe point.
    pub fn request_stop(&self) {
        self.cancellation.request();
    }
}

/// Sends notices and wakes the window so it shows them promptly.
struct NoticeSender {
    sender: Sender<Notice>,
    wake_window: Box<dyn Fn() + Send>,
}

impl NoticeSender {
    fn send(&self, notice: Notice) {
        let _ = self.sender.send(notice);
        (self.wake_window)();
    }
}

struct Worker {
    brain: Option<Brain>,
    session: Option<ChatSession>,
    cancellation: CancellationFlag,
    notices: NoticeSender,
}

impl Worker {
    fn serve(&mut self, requests: Receiver<Request>) {
        for request in requests {
            self.cancellation.reset();
            if let Err(error) = self.handle(request) {
                self.notices.send(Notice::Failure(error.to_string()));
            }
            self.notices.send(Notice::Ready);
        }
    }

    fn handle(&mut self, request: Request) -> BrainResult<()> {
        match request {
            Request::CreateBrain {
                directory,
                model_configuration,
                device_choice,
                seed,
            } => {
                let brain = Brain::create(directory, *model_configuration, device_choice, seed)?;
                self.install_brain(brain)
            }
            Request::OpenBrain {
                directory,
                device_choice,
            } => self.install_brain(Brain::open(directory, device_choice)?),
            Request::CloseBrain => self.close_brain(),
            Request::WriteSampleCorpus {
                path,
                target_bytes,
                seed,
            } => self.write_sample_corpus(path, target_bytes, seed),
            Request::Pretrain(options) => self.pretrain(&options),
            Request::Evaluate {
                corpus_paths,
                batch_count,
                batch_sequence_count,
                seed,
            } => self.evaluate(&corpus_paths, batch_count, batch_sequence_count, seed),
            Request::SplitPatches(text) => self.split_patches(&text),
            Request::RoundTrip(text) => self.round_trip(&text),
            Request::Generate {
                prompt,
                generation,
                is_tracing_thoughts,
            } => self.generate(&prompt, &generation, is_tracing_thoughts),
            Request::StartChat(options) => self.start_chat(options),
            Request::ChatTurn(user_text) => self.chat_turn(&user_text),
            Request::CorrectLastReply(corrected_reply) => self.correct_last_reply(corrected_reply),
            Request::PraiseLastReply => self.praise_last_reply(),
            Request::RejectLastReply => self.reject_last_reply(),
            Request::RefreshMemories => self.send_memories(),
            Request::SearchMemories { query, limit } => self.search_memories(&query, limit),
            Request::AddNote { text, importance } => self.add_note(&text, importance),
            Request::DeleteMemories(ids) => self.delete_memories(&ids),
            Request::ClearMemories => self.clear_memories(),
            Request::Sleep => self.sleep(),
            Request::SleepIfDue { idle } => self.sleep_if_due(idle),
            Request::CreateSnapshot {
                label,
                includes_slow_weights,
            } => self.create_snapshot(&label, includes_slow_weights),
            Request::RollBack(snapshot_id) => self.roll_back(snapshot_id),
            Request::DeleteSnapshot(snapshot_id) => self.delete_snapshot(snapshot_id),
            Request::ResetFastWeights => self.reset_fast_weights(),
            Request::ReplaceSettings(settings) => self.replace_settings(*settings),
            Request::RefreshHistory => self.send_history(),
        }
    }

    fn evaluate(
        &mut self,
        corpus_paths: &[PathBuf],
        batch_count: usize,
        batch_sequence_count: usize,
        seed: u64,
    ) -> BrainResult<()> {
        let report =
            self.brain()?
                .evaluate(corpus_paths, batch_count, batch_sequence_count, seed)?;
        self.notices.send(Notice::Evaluation(report));
        Ok(())
    }

    fn split_patches(&mut self, text: &str) -> BrainResult<()> {
        let patches = self.brain()?.split_patches(text);
        self.notices.send(Notice::Patches(patches));
        Ok(())
    }

    fn round_trip(&mut self, text: &str) -> BrainResult<()> {
        let report = self.brain()?.round_trip(text);
        self.notices.send(Notice::RoundTrip(report));
        Ok(())
    }

    fn search_memories(&mut self, query: &str, limit: usize) -> BrainResult<()> {
        let matches = self.brain()?.search_memories(query, limit)?;
        self.notices.send(Notice::MemoryMatches(matches));
        Ok(())
    }

    fn add_note(&mut self, text: &str, importance: f64) -> BrainResult<()> {
        let entry = self.brain()?.add_note(text, importance)?;
        self.inform(format!("Stored as memory #{}", entry.id));
        self.send_memories()
    }

    fn delete_memories(&mut self, ids: &[u64]) -> BrainResult<()> {
        let deleted_count = self.brain()?.delete_memories(ids)?;
        self.inform(format!("Deleted {deleted_count} memories"));
        self.send_memories()
    }

    fn clear_memories(&mut self) -> BrainResult<()> {
        let deleted_count = self.brain()?.clear_memories()?;
        self.inform(format!("Deleted all {deleted_count} memories"));
        self.send_memories()
    }

    fn create_snapshot(&mut self, label: &str, includes_slow_weights: bool) -> BrainResult<()> {
        let information = self
            .brain()?
            .create_snapshot(label, includes_slow_weights)?;
        self.inform(format!("Created snapshot #{}", information.id));
        self.send_snapshots()
    }

    fn roll_back(&mut self, snapshot_id: u64) -> BrainResult<()> {
        self.brain()?.roll_back(snapshot_id)?;
        self.inform(format!("Rolled back to snapshot #{snapshot_id}"));
        self.send_summary()
    }

    fn delete_snapshot(&mut self, snapshot_id: u64) -> BrainResult<()> {
        self.brain()?.delete_snapshot(snapshot_id)?;
        self.inform(format!("Deleted snapshot #{snapshot_id}"));
        self.send_snapshots()
    }

    fn reset_fast_weights(&mut self) -> BrainResult<()> {
        self.brain()?.reset_fast_weights()?;
        self.inform("Fast weights reset".to_string());
        Ok(())
    }

    fn replace_settings(&mut self, settings: RuntimeSettings) -> BrainResult<()> {
        self.brain()?.replace_settings(settings.clone())?;
        self.notices.send(Notice::Settings(Box::new(settings)));
        self.inform("Settings saved".to_string());
        Ok(())
    }

    fn send_history(&mut self) -> BrainResult<()> {
        let records = self.brain()?.history()?;
        self.notices.send(Notice::History(records));
        Ok(())
    }

    fn brain(&mut self) -> BrainResult<&mut Brain> {
        self.brain.as_mut().ok_or_else(|| {
            BrainError::NotFound("no brain is open; create or open one first".into())
        })
    }

    fn inform(&self, message: String) {
        self.notices.send(Notice::Information(message));
    }

    fn install_brain(&mut self, brain: Brain) -> BrainResult<()> {
        let summary = brain.summary()?;
        let settings = brain.settings().clone();
        self.brain = Some(brain);
        self.session = None;
        self.notices.send(Notice::BrainOpened {
            summary: Box::new(summary),
            settings: Box::new(settings),
        });
        self.send_memories()?;
        self.send_snapshots()?;
        self.send_history()
    }

    fn close_brain(&mut self) -> BrainResult<()> {
        self.brain = None;
        self.session = None;
        self.notices.send(Notice::BrainClosed);
        Ok(())
    }

    fn write_sample_corpus(&self, path: PathBuf, target_bytes: u64, seed: u64) -> BrainResult<()> {
        let written_bytes = SampleCorpus::built_in()?.write_file(&path, target_bytes, seed)?;
        self.inform(format!(
            "Wrote {:.2} MB of sample text to {}",
            written_bytes as f64 / 1e6,
            path.display()
        ));
        Ok(())
    }

    fn send_summary(&mut self) -> BrainResult<()> {
        let summary = self.brain()?.summary()?;
        self.notices.send(Notice::Summary(Box::new(summary)));
        self.send_snapshots()
    }

    fn send_memories(&mut self) -> BrainResult<()> {
        let memories = self.brain()?.memories()?;
        self.notices.send(Notice::Memories(memories));
        Ok(())
    }

    fn send_snapshots(&mut self) -> BrainResult<()> {
        let snapshots = self.brain()?.snapshots()?;
        self.notices.send(Notice::Snapshots(snapshots));
        Ok(())
    }

    fn pretrain(&mut self, options: &TrainingOptions) -> BrainResult<()> {
        let Self {
            brain,
            cancellation,
            notices,
            ..
        } = self;
        let brain = brain.as_mut().ok_or_else(|| {
            BrainError::NotFound("no brain is open; create or open one first".into())
        })?;
        let summary = brain.pretrain(options, cancellation, &mut |event| {
            notices.send(Notice::Progress(event));
        })?;
        self.session = None;
        self.notices.send(Notice::PretrainingFinished(summary));
        self.send_summary()
    }

    fn generate(
        &mut self,
        prompt: &str,
        generation: &GenerationSettings,
        is_tracing_thoughts: bool,
    ) -> BrainResult<()> {
        let Self {
            brain,
            cancellation,
            notices,
            ..
        } = self;
        let brain = brain.as_mut().ok_or_else(|| {
            BrainError::NotFound("no brain is open; create or open one first".into())
        })?;
        let reply = brain.generate(
            prompt,
            generation,
            is_tracing_thoughts,
            cancellation,
            &mut |bytes| {
                notices.send(Notice::WrittenBytes(bytes.to_vec()));
            },
        );
        self.notices.send(Notice::GenerationFinished(reply));
        Ok(())
    }

    fn start_chat(&mut self, options: ChatOptions) -> BrainResult<()> {
        self.session = Some(self.brain()?.start_chat(options));
        self.notices.send(Notice::ChatStarted(options));
        Ok(())
    }

    /// Borrows the brain and the conversation together.
    fn conversation(&mut self) -> BrainResult<(&mut Brain, &mut ChatSession)> {
        match (self.brain.as_mut(), self.session.as_mut()) {
            (Some(brain), Some(session)) => Ok((brain, session)),
            (None, _) => Err(BrainError::NotFound(
                "no brain is open; create or open one first".into(),
            )),
            (_, None) => Err(BrainError::NotFound(
                "no conversation has been started".into(),
            )),
        }
    }

    fn chat_turn(&mut self, user_text: &str) -> BrainResult<()> {
        let Self {
            brain,
            session,
            cancellation,
            notices,
        } = self;
        let (Some(brain), Some(session)) = (brain.as_mut(), session.as_mut()) else {
            return Err(BrainError::NotFound(
                "no conversation has been started".into(),
            ));
        };
        let turn = brain.chat_turn(session, user_text, cancellation, &mut |bytes| {
            notices.send(Notice::WrittenBytes(bytes.to_vec()));
        })?;
        self.notices.send(Notice::ChatTurnFinished(Box::new(turn)));
        // A turn can store a memory and always updates the recall counts of the ones it used.
        self.send_memories()?;
        self.send_summary()
    }

    fn correct_last_reply(&mut self, corrected_reply: String) -> BrainResult<()> {
        let (brain, session) = self.conversation()?;
        let report = brain.correct_last_reply(session, &corrected_reply)?;
        self.send_feedback(FeedbackKind::Correction, report, Some(corrected_reply))
    }

    fn praise_last_reply(&mut self) -> BrainResult<()> {
        let (brain, session) = self.conversation()?;
        let report = brain.praise_last_reply(session)?;
        self.send_feedback(FeedbackKind::Praise, report, None)
    }

    fn reject_last_reply(&mut self) -> BrainResult<()> {
        let (brain, session) = self.conversation()?;
        let report = brain.reject_last_reply(session)?;
        self.send_feedback(FeedbackKind::Rejection, report, None)
    }

    fn send_feedback(
        &mut self,
        kind: FeedbackKind,
        report: FeedbackReport,
        corrected_reply: Option<String>,
    ) -> BrainResult<()> {
        self.notices.send(Notice::Feedback {
            kind,
            report,
            corrected_reply,
        });
        self.send_memories()?;
        self.send_summary()
    }

    fn sleep(&mut self) -> BrainResult<()> {
        let Self {
            brain,
            cancellation,
            notices,
            ..
        } = self;
        let brain = brain.as_mut().ok_or_else(|| {
            BrainError::NotFound("no brain is open; create or open one first".into())
        })?;
        let report = brain.sleep(cancellation, &mut |event| {
            notices.send(Notice::Progress(event));
        })?;
        self.notices.send(Notice::SleepFinished(report));
        self.send_memories()?;
        self.send_summary()?;
        self.send_history()
    }

    /// Sleeps when the schedule says so; conversations that do not save never sleep.
    fn sleep_if_due(&mut self, idle: Duration) -> BrainResult<()> {
        let is_session_learning = self
            .session
            .as_ref()
            .is_some_and(|session| session.options().is_saving && !session.options().is_frozen);
        if !is_session_learning {
            return Ok(());
        }
        let Some(trigger) = self.brain()?.sleep_trigger(idle)? else {
            return Ok(());
        };
        self.inform(format!("Going to sleep: {}", trigger.description()));
        self.sleep()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use thingamaroo_core::configuration::ModelPreset;

    use super::*;

    /// Longest a miniature request may take before the test gives up.
    const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

    /// A model small enough to train in a few seconds on any machine.
    fn miniature_configuration() -> ModelConfiguration {
        let mut configuration = ModelPreset::Tiny.configuration();
        configuration.latent.width = 32;
        configuration.encoder.width = 32;
        configuration.encoder.layer_count = 1;
        configuration.decoder.width = 32;
        configuration.decoder.layer_count = 1;
        configuration.predictor.width = 48;
        configuration.predictor.layer_count = 2;
        configuration.predictor.maximum_context_patches = 16;
        configuration.adapter.rank = 2;
        configuration
    }

    /// Sends one request and returns every notice up to the worker's "ready".
    fn request(worker: &WorkerHandle, request: Request) -> Vec<Notice> {
        worker.send(request);
        let mut notices = Vec::new();
        loop {
            let notice = worker
                .notices
                .recv_timeout(REQUEST_TIMEOUT)
                .expect("the worker should answer every request");
            if matches!(notice, Notice::Ready) {
                return notices;
            }
            notices.push(notice);
        }
    }

    fn failures(notices: &[Notice]) -> Vec<&String> {
        notices
            .iter()
            .filter_map(|notice| match notice {
                Notice::Failure(message) => Some(message),
                _ => None,
            })
            .collect()
    }

    fn trained_worker(name: &str) -> (WorkerHandle, PathBuf) {
        let directory = std::env::temp_dir().join(format!("thingamaroo-gui-{name}"));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let corpus_path = directory.join("sample.txt");
        let worker = WorkerHandle::spawn(|| {});

        let written = request(
            &worker,
            Request::WriteSampleCorpus {
                path: corpus_path.clone(),
                target_bytes: 50_000,
                seed: 1,
            },
        );
        assert!(failures(&written).is_empty());
        let created = request(
            &worker,
            Request::CreateBrain {
                directory: directory.join("brain"),
                model_configuration: Box::new(miniature_configuration()),
                device_choice: DeviceChoice::Cpu,
                seed: 3,
            },
        );
        assert!(
            created
                .iter()
                .any(|notice| matches!(notice, Notice::BrainOpened { .. }))
        );

        let mut options = TrainingOptions::default();
        options.data.corpus_paths = vec![corpus_path];
        options.data.batch_sequence_count = 4;
        options.data.replay_window_count = 16;
        options.schedule.step_count = 12;
        options.schedule.warmup_step_count = 2;
        options.reporting.report_interval_steps = 6;
        options.reporting.validation_interval_steps = 12;
        let trained = request(&worker, Request::Pretrain(options));
        assert!(failures(&trained).is_empty(), "{:?}", failures(&trained));
        assert!(
            trained
                .iter()
                .any(|notice| matches!(notice, Notice::PretrainingFinished(_)))
        );
        (worker, directory)
    }

    #[test]
    fn a_request_without_a_brain_fails_politely() {
        let worker = WorkerHandle::spawn(|| {});

        let notices = request(&worker, Request::Sleep);

        assert_eq!(failures(&notices).len(), 1);
    }

    #[test]
    fn the_worker_serves_a_whole_session() {
        let (worker, _) = trained_worker("session");

        let generated = request(
            &worker,
            Request::Generate {
                prompt: "The ".to_string(),
                generation: GenerationSettings {
                    maximum_reply_bytes: 30,
                    ..GenerationSettings::default()
                },
                is_tracing_thoughts: true,
            },
        );
        assert!(
            generated
                .iter()
                .any(|notice| matches!(notice, Notice::WrittenBytes(_)))
        );
        assert!(
            generated
                .iter()
                .any(|notice| matches!(notice, Notice::GenerationFinished(_)))
        );

        request(&worker, Request::StartChat(ChatOptions::default()));
        let turn = request(
            &worker,
            Request::ChatTurn("What does the cat say?".to_string()),
        );
        assert!(
            turn.iter()
                .any(|notice| matches!(notice, Notice::ChatTurnFinished(_)))
        );
        let corrected = request(
            &worker,
            Request::CorrectLastReply("The cat says quack.".to_string()),
        );
        let has_learned = corrected.iter().any(|notice| {
            matches!(notice, Notice::Feedback { kind: FeedbackKind::Correction, report, .. } if report.live_update.is_some())
        });
        assert!(has_learned, "{:?}", failures(&corrected));

        let memories = request(&worker, Request::RefreshMemories);
        let stored_count = memories
            .iter()
            .find_map(|notice| match notice {
                Notice::Memories(entries) => Some(entries.len()),
                _ => None,
            })
            .unwrap();
        assert_eq!(stored_count, 1);

        let patches = request(
            &worker,
            Request::SplitPatches("The cat says meow.".to_string()),
        );
        assert!(
            patches
                .iter()
                .any(|notice| matches!(notice, Notice::Patches(found) if found.len() == 4))
        );
        let round_trip = request(
            &worker,
            Request::RoundTrip("The cat says meow.".to_string()),
        );
        assert!(
            round_trip
                .iter()
                .any(|notice| matches!(notice, Notice::RoundTrip(_)))
        );

        let snapshot = request(
            &worker,
            Request::CreateSnapshot {
                label: "test".to_string(),
                includes_slow_weights: true,
            },
        );
        assert!(failures(&snapshot).is_empty());
        let closed = request(&worker, Request::CloseBrain);
        assert!(
            closed
                .iter()
                .any(|notice| matches!(notice, Notice::BrainClosed))
        );
    }

    #[test]
    fn settings_changes_reach_the_brain_and_the_window() {
        let (worker, directory) = trained_worker("settings");
        let changed = RuntimeSettings::default()
            .with_entry_text("sleep.replay_step_count", "3")
            .unwrap();

        let notices = request(&worker, Request::ReplaceSettings(Box::new(changed.clone())));

        assert!(
            notices.iter().any(
                |notice| matches!(notice, Notice::Settings(settings) if **settings == changed)
            )
        );
        request(&worker, Request::CloseBrain);
        let reopened = request(
            &worker,
            Request::OpenBrain {
                directory: directory.join("brain"),
                device_choice: DeviceChoice::Cpu,
            },
        );
        let has_saved_setting = reopened.iter().any(|notice| {
            matches!(notice, Notice::BrainOpened { settings, .. } if settings.sleep.replay_step_count == 3)
        });
        assert!(has_saved_setting);
    }
}
