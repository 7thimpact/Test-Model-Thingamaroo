//! Conversation: the runtime loop of one turn, plus feedback on the last reply.
//!
//! Per turn: the input is measured for surprise, related memories are
//! retrieved, the predictor thinks and the decoder writes a reply. Afterwards
//! the exchange is stored in proportion to its importance, and a surprising or
//! corrective input triggers a small fast-weight update.

use crate::brain::history::{HistoryRecord, append_record};
use crate::brain::memory_access::ensure_line_end;
use crate::brain::sleep_schedule::SleepTrigger;
use crate::brain::{Brain, lesson_context};
use crate::cancellation::CancellationFlag;
use crate::configuration::{ChatSettings, GenerationSettings, SafetySettings};
use crate::error::{BrainError, BrainResult};
use crate::learning::{Lesson, LiveUpdateReport, learn_lesson};
use crate::memory::{
    MemoryDraft, MemoryKind, MemoryMatch, current_unix_seconds, opens_with_correction_marker,
};
use crate::model::{PredictionError, Reply, ReplyRequest, write_reply};

const LINE_END: &str = "\n";
const LIVE_UPDATE_SNAPSHOT_LABEL: &str = "before live update";

/// How many times the working-memory size the transcript may grow before it is trimmed.
const TRANSCRIPT_SLACK_FACTOR: usize = 2;

/// How a conversation treats learning and persistence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatOptions {
    /// Read-only conversation: no weight updates, no memory writes.
    pub is_frozen: bool,
    /// Whether anything learned is written to disk; when off it lasts only for the session.
    pub is_saving: bool,
    /// Whether to record what each thinking stage would have written.
    pub is_tracing_thoughts: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            is_frozen: false,
            is_saving: true,
            is_tracing_thoughts: false,
        }
    }
}

/// The most recent turn, kept so the user can correct or praise it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Exchange {
    /// The conversation before this turn.
    context: Vec<u8>,
    /// The user's message as the model saw it.
    prompt: Vec<u8>,
    /// What the model wrote back.
    reply: Vec<u8>,
    /// The memory this exchange was stored as, if it was stored.
    memory_id: Option<u64>,
}

/// One conversation: its working memory and its most recent exchange.
#[derive(Debug, Clone)]
pub struct ChatSession {
    options: ChatOptions,
    transcript: Vec<u8>,
    transcript_limit_bytes: usize,
    last_exchange: Option<Exchange>,
}

impl ChatSession {
    /// How this conversation treats learning and persistence.
    pub fn options(&self) -> ChatOptions {
        self.options
    }

    /// The conversation so far, as the model sees it.
    pub fn transcript(&self) -> String {
        String::from_utf8_lossy(&self.transcript).into_owned()
    }

    /// Whether there is a reply that can be corrected, praised or rejected.
    pub fn has_exchange(&self) -> bool {
        self.last_exchange.is_some()
    }

    /// Empties the working memory; stored memories and weights are unaffected.
    pub fn forget_transcript(&mut self) {
        self.transcript.clear();
        self.last_exchange = None;
    }

    fn record_exchange(&mut self, prompt: Vec<u8>, reply: Vec<u8>) {
        let context = std::mem::take(&mut self.transcript);
        self.transcript = [context.as_slice(), &prompt, &reply].concat();
        self.last_exchange = Some(Exchange {
            context,
            prompt,
            reply,
            memory_id: None,
        });
        self.trim_transcript();
    }

    /// Drops the oldest bytes once the transcript far exceeds what the predictor can attend to.
    fn trim_transcript(&mut self) {
        let excess_bytes = self
            .transcript
            .len()
            .saturating_sub(self.transcript_limit_bytes);
        if excess_bytes > 0 {
            self.transcript.drain(..excess_bytes);
        }
    }

    fn last_exchange(&self) -> BrainResult<&Exchange> {
        self.last_exchange
            .as_ref()
            .ok_or_else(|| BrainError::NotFound("there is no reply to give feedback on yet".into()))
    }
}

/// Everything that happened during one conversation turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurnReport {
    /// What the model wrote.
    pub reply: Reply,
    /// How badly the user's message was predicted; `None` when it was too short to measure.
    pub surprise: Option<PredictionError>,
    /// How much the exchange matters, in `[0, 1]`.
    pub importance: f64,
    /// Whether the message opened with a correction marker.
    pub is_correction: bool,
    /// The memories shown to the predictor, most similar first.
    pub retrieved: Vec<MemoryMatch>,
    /// The memory the exchange was stored as, if it was important enough.
    pub stored_memory_id: Option<u64>,
    /// The fast-weight update the input triggered, if any.
    pub live_update: Option<LiveUpdateReport>,
    /// Whether the safety filter kept the input from being stored or learned.
    pub is_input_filtered: bool,
    /// Set when the sleep schedule says it is time to consolidate.
    pub sleep_trigger: Option<SleepTrigger>,
}

/// What giving feedback on the last reply did.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackReport {
    /// The importance assigned to the exchange.
    pub importance: f64,
    /// The fast-weight update the feedback triggered, if any.
    pub live_update: Option<LiveUpdateReport>,
    /// The memory the exchange is now stored as, if any.
    pub memory_id: Option<u64>,
}

/// What the brain concluded about a user message before replying.
struct InputAssessment {
    lesson: Option<Lesson>,
    surprise: Option<PredictionError>,
    importance: f64,
    is_correction: bool,
    is_learnable: bool,
}

impl Brain {
    /// Starts a conversation with empty working memory.
    pub fn start_chat(&self, options: ChatOptions) -> ChatSession {
        let working_memory_bytes = (self.model_configuration.predictor.maximum_context_patches + 1)
            * self.model_configuration.patching.maximum_patch_bytes;
        ChatSession {
            options,
            transcript: Vec::new(),
            transcript_limit_bytes: working_memory_bytes * TRANSCRIPT_SLACK_FACTOR,
            last_exchange: None,
        }
    }

    /// Continues a prompt without memory or learning.
    pub fn generate(
        &mut self,
        prompt: &str,
        generation: &GenerationSettings,
        is_tracing_thoughts: bool,
        cancellation: &CancellationFlag,
        on_bytes: &mut dyn FnMut(&[u8]),
    ) -> Reply {
        let request = ReplyRequest {
            context: prompt.as_bytes(),
            memories: &[],
            generation,
            is_stopping_at_line_end: false,
            is_tracing_thoughts,
        };
        write_reply(&mut self.inference, request, cancellation, on_bytes)
    }

    /// Runs one conversation turn: assess, retrieve, reply, then store and learn.
    pub fn chat_turn(
        &mut self,
        session: &mut ChatSession,
        user_text: &str,
        cancellation: &CancellationFlag,
        on_bytes: &mut dyn FnMut(&[u8]),
    ) -> BrainResult<ChatTurnReport> {
        let prompt = frame_prompt(user_text, &self.settings.chat);
        let assessment = self.assess_input(session, user_text, &prompt);
        let retrieved = self.retrieve_memories(session, user_text)?;
        let reply = self.reply_to(session, &prompt, &retrieved, cancellation, on_bytes);
        session.record_exchange(prompt, reply.bytes.clone());

        let stored_memory_id = self.remember_exchange(session, &assessment)?;
        let live_update = self.learn_from_input(session, &assessment)?;
        self.finish_turn(session)?;

        Ok(ChatTurnReport {
            reply,
            surprise: assessment.surprise,
            importance: assessment.importance,
            is_correction: assessment.is_correction,
            retrieved,
            stored_memory_id,
            live_update,
            is_input_filtered: !passes_input_filter(user_text, &self.settings.safety),
            sleep_trigger: self.sleep_trigger(std::time::Duration::ZERO)?,
        })
    }

    /// Measures how surprising the message is and decides how much it matters.
    fn assess_input(
        &mut self,
        session: &ChatSession,
        user_text: &str,
        prompt: &[u8],
    ) -> InputAssessment {
        let maximum_patch_count = self.model_configuration.predictor.maximum_context_patches + 1;
        let lesson = Lesson::continuing(
            self.inference.patcher(),
            &session.transcript,
            prompt,
            maximum_patch_count,
        );
        let surprise = lesson
            .as_ref()
            .and_then(|lesson| self.measure_surprise(lesson));
        let surprise_importance = surprise.map_or(0.0, |measured| {
            self.state.surprise.importance_of(measured.bits_per_byte)
        });

        let is_correction = opens_with_correction_marker(user_text, &self.settings.surprise);
        let correction_importance = match is_correction {
            true => self.settings.surprise.correction_importance,
            false => 0.0,
        };
        let is_learnable =
            !session.options.is_frozen && passes_input_filter(user_text, &self.settings.safety);
        if let (true, Some(measured)) = (is_learnable, surprise) {
            self.state
                .surprise
                .observe(measured.bits_per_byte, &self.settings.surprise);
        }

        InputAssessment {
            lesson,
            surprise,
            importance: self.cap_importance(surprise_importance.max(correction_importance)),
            is_correction,
            is_learnable,
        }
    }

    fn measure_surprise(&mut self, lesson: &Lesson) -> Option<PredictionError> {
        let patches: Vec<&[u8]> = lesson.patches.iter().map(Vec::as_slice).collect();
        self.inference.measure_prediction_error(
            &patches,
            lesson.first_learned_patch,
            self.settings.generation.thinking_step_count,
        )
    }

    /// Finds stored exchanges related to the message and notes that they were used.
    fn retrieve_memories(
        &mut self,
        session: &ChatSession,
        user_text: &str,
    ) -> BrainResult<Vec<MemoryMatch>> {
        let memory = &self.settings.memory;
        if !memory.is_retrieval_enabled {
            return Ok(Vec::new());
        }
        let Some(query) = self.inference.gist(user_text.as_bytes()) else {
            return Ok(Vec::new());
        };
        let slot_count = memory
            .retrieval_count
            .min(self.model_configuration.predictor.maximum_memory_slots);
        let retrieved = self
            .memory
            .search(&query, slot_count, memory.minimum_similarity)?;

        let is_recording = session.options.is_saving && !session.options.is_frozen;
        if is_recording {
            let retrieved_ids: Vec<u64> = retrieved.iter().map(|found| found.entry.id).collect();
            self.memory.record_retrievals(&retrieved_ids)?;
        }
        Ok(retrieved)
    }

    fn reply_to(
        &mut self,
        session: &ChatSession,
        prompt: &[u8],
        retrieved: &[MemoryMatch],
        cancellation: &CancellationFlag,
        on_bytes: &mut dyn FnMut(&[u8]),
    ) -> Reply {
        let context = [session.transcript.as_slice(), prompt].concat();
        let memories: Vec<Vec<f32>> = retrieved.iter().map(|found| found.vector.clone()).collect();
        let request = ReplyRequest {
            context: &context,
            memories: &memories,
            generation: &self.settings.generation,
            is_stopping_at_line_end: self.settings.chat.is_stopping_at_line_end,
            is_tracing_thoughts: session.options.is_tracing_thoughts,
        };
        write_reply(&mut self.inference, request, cancellation, on_bytes)
    }

    /// Stores the exchange when it is important enough to keep.
    fn remember_exchange(
        &mut self,
        session: &mut ChatSession,
        assessment: &InputAssessment,
    ) -> BrainResult<Option<u64>> {
        let is_worth_storing = assessment.is_learnable
            && session.options.is_saving
            && self.settings.memory.is_writing_enabled
            && assessment.importance >= self.settings.memory.minimum_stored_importance;
        if !is_worth_storing {
            return Ok(None);
        }
        let kind = match assessment.is_correction {
            true => MemoryKind::Correction,
            false => MemoryKind::Exchange,
        };
        self.store_last_exchange(session, kind, assessment.importance)
    }

    /// Writes the session's last exchange to the memory store, replacing its earlier entry.
    fn store_last_exchange(
        &mut self,
        session: &mut ChatSession,
        kind: MemoryKind,
        importance: f64,
    ) -> BrainResult<Option<u64>> {
        let Some(exchange) = session.last_exchange.as_mut() else {
            return Ok(None);
        };
        let prompt = String::from_utf8_lossy(&exchange.prompt).into_owned();
        let reply = String::from_utf8_lossy(&exchange.reply).into_owned();
        let Some(vector) = self.inference.gist(format!("{prompt}{reply}").as_bytes()) else {
            return Ok(None);
        };

        if let Some(previous_id) = exchange.memory_id.take() {
            self.memory.delete(&[previous_id])?;
        }
        let entry = self.memory.insert(MemoryDraft {
            kind,
            prompt: &prompt,
            reply: &reply,
            importance,
            vector: &vector,
        })?;
        exchange.memory_id = Some(entry.id);
        Ok(Some(entry.id))
    }

    /// Updates the fast weights when the input was a correction or surprising enough.
    fn learn_from_input(
        &mut self,
        session: &ChatSession,
        assessment: &InputAssessment,
    ) -> BrainResult<Option<LiveUpdateReport>> {
        let learning = &self.settings.learning;
        let is_corrective = assessment.is_correction && learning.is_learning_from_corrections;
        let is_surprising = learning.is_learning_from_surprise
            && assessment.importance >= learning.surprise_importance_threshold;
        let is_worth_learning =
            assessment.is_learnable && learning.is_enabled && (is_corrective || is_surprising);
        let (true, Some(lesson)) = (is_worth_learning, assessment.lesson.as_ref()) else {
            return Ok(None);
        };
        self.apply_live_update(lesson, assessment.importance, session.options.is_saving)
            .map(Some)
    }

    /// Teaches one lesson to the fast weights, undoing the update if it fails.
    pub(crate) fn apply_live_update(
        &mut self,
        lesson: &Lesson,
        importance: f64,
        is_saving: bool,
    ) -> BrainResult<LiveUpdateReport> {
        let previous_fast_weights = self.fast_weight_record();
        if is_saving {
            self.create_automatic_snapshot(LIVE_UPDATE_SNAPSHOT_LABEL, false)?;
        }

        let context = lesson_context(
            &self.inference,
            &self.compute_device,
            &self.model_configuration,
            &self.settings,
        );
        let outcome = learn_lesson(
            &mut self.fast_weights,
            &context,
            lesson,
            &self.settings.learning,
            importance,
        );
        let report = match outcome {
            Ok(report) => report,
            Err(error) => {
                self.replace_fast_weights_from(previous_fast_weights)?;
                return Err(error);
            }
        };

        self.inference
            .replace_predictor(burn::module::Module::valid(self.fast_weights.network()));
        self.record_live_update(&report, importance, is_saving)?;
        Ok(report)
    }

    fn record_live_update(
        &mut self,
        report: &LiveUpdateReport,
        importance: f64,
        is_saving: bool,
    ) -> BrainResult<()> {
        self.state.conversation.live_update_count += 1;
        self.state.conversation.live_updates_since_sleep += 1;
        if !is_saving {
            return Ok(());
        }
        self.save_fast_weights()?;
        append_record(
            &self.directory.history_path(),
            &HistoryRecord::LiveUpdate {
                unix_seconds: current_unix_seconds(),
                importance,
                report: *report,
            },
        )
    }

    fn finish_turn(&mut self, session: &ChatSession) -> BrainResult<()> {
        self.state.conversation.turn_count += 1;
        if !session.options.is_frozen {
            self.state.conversation.turns_since_sleep += 1;
        }
        if !session.options.is_saving {
            return Ok(());
        }
        self.save_state()
    }

    /// Replaces the last reply with the right answer and learns it strongly.
    pub fn correct_last_reply(
        &mut self,
        session: &mut ChatSession,
        corrected_reply: &str,
    ) -> BrainResult<FeedbackReport> {
        let corrected_reply = ensure_line_end(corrected_reply).into_bytes();
        let exchange = session.last_exchange()?.clone();
        session.transcript = [
            exchange.context.as_slice(),
            &exchange.prompt,
            &corrected_reply,
        ]
        .concat();
        if let Some(last_exchange) = session.last_exchange.as_mut() {
            last_exchange.reply = corrected_reply;
        }

        let importance = self.cap_importance(self.settings.surprise.correction_importance);
        let is_learning = self.settings.learning.is_learning_from_corrections;
        self.learn_from_feedback(session, MemoryKind::Correction, importance, is_learning)
    }

    /// Marks the last reply as good and reinforces it.
    pub fn praise_last_reply(&mut self, session: &mut ChatSession) -> BrainResult<FeedbackReport> {
        session.last_exchange()?;
        let importance = self.cap_importance(self.settings.surprise.praise_importance);
        let is_learning = self.settings.learning.is_learning_from_praise;
        self.learn_from_feedback(session, MemoryKind::Praise, importance, is_learning)
    }

    /// Marks the last reply as bad: its memory is deleted so it is never replayed.
    pub fn reject_last_reply(&mut self, session: &mut ChatSession) -> BrainResult<FeedbackReport> {
        session.last_exchange()?;
        let stored_id = session
            .last_exchange
            .as_mut()
            .and_then(|exchange| exchange.memory_id.take());
        if let Some(memory_id) = stored_id {
            self.memory.delete(&[memory_id])?;
        }
        Ok(FeedbackReport {
            importance: 0.0,
            live_update: None,
            memory_id: None,
        })
    }

    /// Stores the last exchange as trusted and teaches its reply to the fast weights.
    fn learn_from_feedback(
        &mut self,
        session: &mut ChatSession,
        kind: MemoryKind,
        importance: f64,
        is_learning_from_this_feedback: bool,
    ) -> BrainResult<FeedbackReport> {
        if session.options.is_frozen {
            return Ok(FeedbackReport {
                importance,
                live_update: None,
                memory_id: None,
            });
        }
        let exchange = session.last_exchange()?.clone();
        let is_storing = session.options.is_saving && self.settings.memory.is_writing_enabled;
        let memory_id = match is_storing {
            true => self.store_last_exchange(session, kind, importance)?,
            false => None,
        };

        let lesson = Lesson::continuing(
            self.inference.patcher(),
            &[exchange.context.as_slice(), &exchange.prompt].concat(),
            &exchange.reply,
            self.model_configuration.predictor.maximum_context_patches + 1,
        );
        let is_learning = self.settings.learning.is_enabled && is_learning_from_this_feedback;
        let live_update = match (is_learning, lesson) {
            (true, Some(lesson)) => {
                Some(self.apply_live_update(&lesson, importance, session.options.is_saving)?)
            }
            _ => None,
        };
        Ok(FeedbackReport {
            importance,
            live_update,
            memory_id,
        })
    }
}

/// Wraps a user message the way the training data frames a turn.
fn frame_prompt(user_text: &str, chat: &ChatSettings) -> Vec<u8> {
    let message = user_text.trim_end_matches(['\r', '\n']);
    format!(
        "{}{message}{LINE_END}{}",
        chat.user_prefix, chat.model_prefix
    )
    .into_bytes()
}

/// Whether an input may be stored and learned from.
fn passes_input_filter(user_text: &str, safety: &SafetySettings) -> bool {
    let is_size_allowed =
        (safety.minimum_input_bytes..=safety.maximum_input_bytes).contains(&user_text.len());
    let lowercase_text = user_text.to_lowercase();
    let has_blocked_phrase = safety
        .blocked_phrases
        .iter()
        .any(|phrase| !phrase.is_empty() && lowercase_text.contains(&phrase.to_lowercase()));
    is_size_allowed && !has_blocked_phrase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_framed_with_prefixes_and_one_line_end() {
        let chat = ChatSettings {
            user_prefix: "User: ".into(),
            model_prefix: "Bot: ".into(),
            is_stopping_at_line_end: true,
        };

        assert_eq!(frame_prompt("hello\n", &chat), b"User: hello\nBot: ");
        assert_eq!(frame_prompt("hello", &ChatSettings::default()), b"hello\n");
    }

    #[test]
    fn the_input_filter_blocks_phrases_and_extreme_sizes() {
        let safety = SafetySettings {
            blocked_phrases: vec!["Forget Everything".into()],
            minimum_input_bytes: 2,
            maximum_input_bytes: 40,
            ..SafetySettings::default()
        };

        assert!(passes_input_filter("the cat says meow", &safety));
        assert!(!passes_input_filter(
            "please forget everything now",
            &safety
        ));
        assert!(!passes_input_filter("x", &safety));
        assert!(!passes_input_filter(&"long ".repeat(20), &safety));
    }
}
