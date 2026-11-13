//! Sleep: moving what the fast weights and memories hold into the slow weights.
//!
//! One cycle: snapshot, replay important memories mixed with old training
//! data, check that old skills survived, then reset the fast weights and prune
//! weak memories. A cycle that damages old skills is discarded.

use burn::module::Module;
use serde::{Deserialize, Serialize};

use crate::brain::history::{HistoryRecord, append_record};
use crate::brain::{Brain, lesson_context};
use crate::cancellation::CancellationFlag;
use crate::configuration::OptimizerOptions;
use crate::error::BrainResult;
use crate::learning::{
    ConsolidationInput, ConsolidationReport, ElasticAnchor, Lesson, LessonContext, ProgressEvent,
    ProgressObserver, TrainableModule, attach_fast_weights, build_optimizer, consolidate,
    lesson_loss, measure_lessons, shrink_fast_weights,
};
use crate::memory::{MemoryEntry, current_unix_seconds, plan_retention};
use crate::model::ConceptPredictor;
use crate::random::RandomSource;
use crate::text::ReplayReservoir;

const SLEEP_SNAPSHOT_LABEL: &str = "before sleep";

/// Old-data batches used to check that old skills survived a sleep.
const OLD_SKILL_PROBE_BATCH_COUNT: usize = 4;

/// Slow weights are regularised by the elastic penalty rather than by weight decay.
const CONSOLIDATION_WEIGHT_DECAY: f64 = 0.0;
const CONSOLIDATION_GRADIENT_CLIP_NORM: f64 = 1.0;

/// What one sleep did.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SleepReport {
    /// When the sleep finished.
    pub finished_unix_seconds: u64,
    /// Memories chosen for replay.
    pub replayed_episode_count: usize,
    /// What the replay loop did.
    pub consolidation: ConsolidationReport,
    /// Loss of the slow weights on old training data before the sleep.
    pub old_loss_before: Option<f64>,
    /// Loss of the slow weights on old training data after the sleep.
    pub old_loss_after: Option<f64>,
    /// Loss of the slow weights on the replayed memories before the sleep.
    pub episode_loss_before: Option<f64>,
    /// Loss of the slow weights on the replayed memories after the sleep.
    pub episode_loss_after: Option<f64>,
    /// Whether the new weights were discarded because old skills degraded or the user cancelled.
    pub was_discarded: bool,
    /// Memories deleted by pruning.
    pub pruned_memory_count: usize,
    /// Memories left in the store.
    pub remaining_memory_count: usize,
    /// The snapshot taken before the sleep, for manual rollback.
    pub snapshot_id: Option<u64>,
}

/// How well a predictor does on what must be kept and on what must be learned.
struct SleepLosses {
    /// Loss on samples of the original training data; `None` when there are none.
    old_skills: Option<f64>,
    /// Loss on the memories chosen for replay; `None` when there are none.
    episodes: Option<f64>,
}

/// The lessons a sleep works with.
struct SleepMaterial {
    episodes: Vec<Lesson>,
    old_skill_probe: Vec<Lesson>,
    reservoir: ReplayReservoir,
    anchor: Option<ElasticAnchor>,
}

impl Brain {
    /// Runs one sleep cycle.
    pub fn sleep(
        &mut self,
        cancellation: &CancellationFlag,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<SleepReport> {
        on_progress(ProgressEvent::Status(
            "Taking a snapshot before sleep".into(),
        ));
        let snapshot = self.create_automatic_snapshot(SLEEP_SNAPSHOT_LABEL, true)?;
        let mut random = RandomSource::from_seed(self.state.sleep.cycle_count);
        let material = self.gather_sleep_material(&mut random)?;
        let mut report = SleepReport {
            replayed_episode_count: material.episodes.len(),
            snapshot_id: Some(snapshot.id),
            ..SleepReport::default()
        };

        on_progress(ProgressEvent::Status(format!(
            "Replaying {} memories with {} windows of old data",
            material.episodes.len(),
            material.reservoir.window_count()
        )));
        let consolidated_predictor = self.replay_into_slow_weights(
            &material,
            &mut report,
            &mut random,
            cancellation,
            on_progress,
        )?;

        report.was_discarded =
            report.consolidation.was_cancelled || self.has_degraded_old_skills(&report);
        if !report.was_discarded {
            on_progress(ProgressEvent::Status(
                "Keeping the consolidated weights".into(),
            ));
            self.accept_consolidation(consolidated_predictor, &material.reservoir, &mut random)?;
            report.pruned_memory_count = self.prune_memories()?;
        }
        self.finish_sleep(&mut report)?;
        Ok(report)
    }

    fn gather_sleep_material(&self, random: &mut RandomSource) -> BrainResult<SleepMaterial> {
        let context = lesson_context(
            &self.inference,
            &self.compute_device,
            &self.model_configuration,
            &self.settings,
        );
        let reservoir = ReplayReservoir::load(&self.directory.replay_reservoir_path())?;
        let probe_window_count =
            OLD_SKILL_PROBE_BATCH_COUNT * self.settings.sleep.batch_sequence_count;
        let old_skill_probe = (0..probe_window_count)
            .filter_map(|_| reservoir.pick(random))
            .filter_map(|window| {
                Lesson::whole(context.patcher(), window, context.maximum_patch_count())
            })
            .collect();

        Ok(SleepMaterial {
            episodes: self.select_episodes(&context)?,
            old_skill_probe,
            anchor: ElasticAnchor::load(
                &self.slow_model.predictor,
                &self.directory.anchor_values_path(),
                &self.directory.anchor_importances_path(),
            )?,
            reservoir,
        })
    }

    /// Turns the most important memories into lessons.
    fn select_episodes(&self, context: &LessonContext<'_>) -> BrainResult<Vec<Lesson>> {
        let mut entries = self.memory.list()?;
        entries.sort_by(|left, right| right.importance.total_cmp(&left.importance));
        Ok(entries
            .iter()
            .take(self.settings.sleep.episode_count)
            .filter_map(|entry| lesson_from_memory(entry, context))
            .collect())
    }

    /// Trains a copy of the slow predictor on the replay mix and measures it before and after.
    fn replay_into_slow_weights(
        &self,
        material: &SleepMaterial,
        report: &mut SleepReport,
        random: &mut RandomSource,
        cancellation: &CancellationFlag,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<ConceptPredictor> {
        let context = lesson_context(
            &self.inference,
            &self.compute_device,
            &self.model_configuration,
            &self.settings,
        );
        let slow_predictor = &self.slow_model.predictor;
        let losses_before = self.measure_sleep_losses(&slow_predictor.valid(), &context, material);
        report.old_loss_before = losses_before.old_skills;
        report.episode_loss_before = losses_before.episodes;

        let optimizer = build_optimizer(&OptimizerOptions {
            weight_decay: CONSOLIDATION_WEIGHT_DECAY,
            gradient_clip_norm: CONSOLIDATION_GRADIENT_CLIP_NORM,
        });
        let mut trainable = TrainableModule::new(slow_predictor.clone(), optimizer);
        let input = ConsolidationInput {
            episodes: &material.episodes,
            reservoir: &material.reservoir,
            anchor: material.anchor.as_ref(),
            sleep: &self.settings.sleep,
        };
        report.consolidation = consolidate(
            &mut trainable,
            &context,
            &input,
            random,
            cancellation,
            on_progress,
        )?;

        let (consolidated_predictor, _) = trainable.into_parts();
        let losses_after =
            self.measure_sleep_losses(&consolidated_predictor.valid(), &context, material);
        report.old_loss_after = losses_after.old_skills;
        report.episode_loss_after = losses_after.episodes;
        Ok(consolidated_predictor)
    }

    /// Measures a gradient-free predictor on old training data and on the replayed memories.
    fn measure_sleep_losses(
        &self,
        frozen_predictor: &ConceptPredictor,
        context: &LessonContext<'_>,
        material: &SleepMaterial,
    ) -> SleepLosses {
        let batch_sequence_count = self.settings.sleep.batch_sequence_count;
        SleepLosses {
            old_skills: measure_lessons(
                frozen_predictor,
                context,
                &material.old_skill_probe,
                batch_sequence_count,
            ),
            episodes: measure_lessons(
                frozen_predictor,
                context,
                &material.episodes,
                batch_sequence_count,
            ),
        }
    }

    /// Whether the loss on old data rose by more than the configured tolerance.
    fn has_degraded_old_skills(&self, report: &SleepReport) -> bool {
        let (Some(loss_before), Some(loss_after)) = (report.old_loss_before, report.old_loss_after)
        else {
            return false;
        };
        loss_after > loss_before * (1.0 + self.settings.sleep.maximum_old_loss_increase)
    }

    /// Installs the consolidated predictor, resets or shrinks the fast weights, and re-anchors.
    fn accept_consolidation(
        &mut self,
        consolidated_predictor: ConceptPredictor,
        reservoir: &ReplayReservoir,
        random: &mut RandomSource,
    ) -> BrainResult<()> {
        let fast_weight_retention = self.settings.sleep.fast_weight_retention.clamp(0.0, 1.0);
        let previous_fast_weights = self.fast_weight_record();
        self.slow_model.predictor = consolidated_predictor;

        if fast_weight_retention > 0.0 {
            self.replace_fast_weights_from(previous_fast_weights)?;
            let shrunk =
                shrink_fast_weights(self.fast_weights.network().clone(), fast_weight_retention);
            self.install_fast_weights(shrunk);
        } else {
            let fresh = attach_fast_weights(
                &self.slow_model.predictor,
                &self.model_configuration.adapter,
            );
            self.install_fast_weights(fresh);
        }
        self.refresh_inference();
        self.renew_elastic_anchor(reservoir, random)
    }

    /// Records the current slow predictor as the weights later sleeps must stay close to.
    pub(crate) fn renew_elastic_anchor(
        &self,
        reservoir: &ReplayReservoir,
        random: &mut RandomSource,
    ) -> BrainResult<()> {
        if reservoir.is_empty() {
            return Ok(());
        }
        let context = lesson_context(
            &self.inference,
            &self.compute_device,
            &self.model_configuration,
            &self.settings,
        );
        let sleep = &self.settings.sleep;
        let anchor = ElasticAnchor::estimate(
            &self.slow_model.predictor,
            sleep.importance_batch_count,
            |predictor| {
                let lessons: Vec<Lesson> = (0..sleep.batch_sequence_count)
                    .filter_map(|_| reservoir.pick(random))
                    .filter_map(|window| {
                        Lesson::whole(context.patcher(), window, context.maximum_patch_count())
                    })
                    .collect();
                Ok(lesson_loss(predictor, &context, &lessons))
            },
        )?;
        anchor.save(
            &self.slow_model.predictor,
            &self.directory.anchor_values_path(),
            &self.directory.anchor_importances_path(),
        )
    }

    /// Decays every memory's importance and deletes the ones that have faded.
    fn prune_memories(&mut self) -> BrainResult<usize> {
        let plan = plan_retention(
            self.memory.list()?,
            &self.settings.sleep,
            &self.settings.memory,
        );
        self.memory.update(&plan.kept_entries)?;
        self.memory.delete(&plan.pruned_ids)
    }

    fn finish_sleep(&mut self, report: &mut SleepReport) -> BrainResult<()> {
        report.finished_unix_seconds = current_unix_seconds();
        report.remaining_memory_count = self.memory.entry_count()?;
        self.state.sleep.last_sleep_unix_seconds = report.finished_unix_seconds;
        self.state.sleep.awake_since_unix_seconds = report.finished_unix_seconds;
        // Reset even when the result was discarded, so the schedule does not retry at once.
        self.state.conversation.turns_since_sleep = 0;
        self.state.conversation.live_updates_since_sleep = 0;
        if !report.was_discarded {
            self.state.sleep.cycle_count += 1;
        }
        self.save()?;
        append_record(
            &self.directory.history_path(),
            &HistoryRecord::Sleep(report.clone()),
        )
    }
}

/// Builds the lesson a memory teaches during replay.
///
/// A trusted reply is learned as the continuation of its prompt. An ordinary
/// exchange holds the model's own unverified reply, so only the user's side
/// is learned.
fn lesson_from_memory(entry: &MemoryEntry, context: &LessonContext<'_>) -> Option<Lesson> {
    let maximum_patch_count = context.maximum_patch_count();
    match entry.kind.is_reply_trusted() {
        true => Lesson::continuing(
            context.patcher(),
            entry.prompt.as_bytes(),
            entry.reply.as_bytes(),
            maximum_patch_count,
        ),
        false => Lesson::whole(
            context.patcher(),
            entry.prompt.as_bytes(),
            maximum_patch_count,
        ),
    }
}
