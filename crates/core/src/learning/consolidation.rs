//! Sleep consolidation: replaying memories into the slow weights.
//!
//! Important episodes are replayed mixed with samples of the original
//! training data, while elastic weight consolidation holds the weights that
//! mattered for old skills in place. Only the predictor's slow weights change.

use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::cancellation::CancellationFlag;
use crate::configuration::SleepSettings;
use crate::error::{BrainError, BrainResult};
use crate::learning::elastic_anchor::ElasticAnchor;
use crate::learning::lesson::{Lesson, LessonContext, lesson_loss};
use crate::learning::progress::{ProgressEvent, ProgressObserver, SleepStepReport};
use crate::learning::trainable_module::TrainableModule;
use crate::model::{ConceptPredictor, read_scalar};
use crate::random::RandomSource;
use crate::text::ReplayReservoir;

/// What is replayed and what holds the old weights in place.
#[derive(Debug, Clone, Copy)]
pub struct ConsolidationInput<'sleep> {
    /// The memories chosen for replay.
    pub episodes: &'sleep [Lesson],
    /// Samples of the original training data.
    pub reservoir: &'sleep ReplayReservoir,
    /// Recorded weights and their importance; `None` disables the elastic penalty.
    pub anchor: Option<&'sleep ElasticAnchor>,
    /// How the replay behaves.
    pub sleep: &'sleep SleepSettings,
}

/// What the replay loop did.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Gradient steps taken.
    pub step_count: usize,
    /// Replay loss at the first step.
    pub first_replay_loss: f64,
    /// Replay loss at the last step.
    pub last_replay_loss: f64,
    /// Elastic penalty at the last step.
    pub last_elastic_penalty: f64,
    /// Whether the user stopped the replay early.
    pub was_cancelled: bool,
}

/// Replays episodes and old data into the predictor's slow weights.
pub fn consolidate(
    trainable: &mut TrainableModule<ConceptPredictor>,
    context: &LessonContext<'_>,
    input: &ConsolidationInput<'_>,
    random: &mut RandomSource,
    cancellation: &CancellationFlag,
    on_progress: ProgressObserver<'_>,
) -> BrainResult<ConsolidationReport> {
    let mut report = ConsolidationReport::default();
    for step in 0..input.sleep.replay_step_count {
        if cancellation.is_requested() {
            report.was_cancelled = true;
            break;
        }
        let lessons = sample_replay_batch(context, input, random);
        if lessons.is_empty() {
            break;
        }
        let step_report = replay_step(trainable, context, input, &lessons, step)?;
        record_step(&mut report, &step_report);
        on_progress(ProgressEvent::SleepStep(step_report));
    }
    Ok(report)
}

fn replay_step(
    trainable: &mut TrainableModule<ConceptPredictor>,
    context: &LessonContext<'_>,
    input: &ConsolidationInput<'_>,
    lessons: &[Lesson],
    step: usize,
) -> BrainResult<SleepStepReport> {
    let predictor = trainable.network();
    let replay_loss = lesson_loss(predictor, context, lessons);
    let elastic_penalty = elastic_penalty(predictor, input);

    let step_report = SleepStepReport {
        step: step + 1,
        step_count: input.sleep.replay_step_count,
        replay_loss: read_scalar(replay_loss.clone()),
        elastic_penalty: read_scalar(elastic_penalty.clone()),
    };
    if !step_report.replay_loss.is_finite() || !step_report.elastic_penalty.is_finite() {
        return Err(BrainError::Diverged(
            "sleep replay produced a non-finite loss".into(),
        ));
    }

    trainable.descend(replay_loss + elastic_penalty, input.sleep.learning_rate);
    Ok(step_report)
}

fn elastic_penalty(predictor: &ConceptPredictor, input: &ConsolidationInput<'_>) -> Tensor<1> {
    match input.anchor {
        Some(anchor) if input.sleep.elastic_strength > 0.0 => anchor
            .penalty(predictor)
            .mul_scalar(input.sleep.elastic_strength),
        _ => Tensor::zeros([1], &burn::module::Module::devices(predictor)[0]),
    }
}

fn record_step(report: &mut ConsolidationReport, step_report: &SleepStepReport) {
    if report.step_count == 0 {
        report.first_replay_loss = step_report.replay_loss;
    }
    report.step_count = step_report.step;
    report.last_replay_loss = step_report.replay_loss;
    report.last_elastic_penalty = step_report.elastic_penalty;
}

fn sample_replay_batch(
    context: &LessonContext<'_>,
    input: &ConsolidationInput<'_>,
    random: &mut RandomSource,
) -> Vec<Lesson> {
    (0..input.sleep.batch_sequence_count)
        .filter_map(|_| sample_replay_lesson(context, input, random))
        .collect()
}

/// Picks one lesson: old training data with the configured probability, otherwise a memory.
fn sample_replay_lesson(
    context: &LessonContext<'_>,
    input: &ConsolidationInput<'_>,
    random: &mut RandomSource,
) -> Option<Lesson> {
    let has_old_data = !input.reservoir.is_empty();
    let is_replaying_old_data = input.episodes.is_empty()
        || (has_old_data && random.next_chance(input.sleep.old_data_fraction));
    if !is_replaying_old_data {
        return random.pick(input.episodes).cloned();
    }
    let window = input.reservoir.pick(random)?;
    Lesson::whole(context.patcher(), window, context.maximum_patch_count())
}

/// Average lesson loss of a predictor, measured without changing it.
///
/// Returns `None` when there are no lessons to measure.
pub fn measure_lessons(
    predictor: &ConceptPredictor,
    context: &LessonContext<'_>,
    lessons: &[Lesson],
    batch_sequence_count: usize,
) -> Option<f64> {
    if lessons.is_empty() {
        return None;
    }
    let batch_losses: Vec<f64> = lessons
        .chunks(batch_sequence_count.max(1))
        .map(|batch| read_scalar(lesson_loss(predictor, context, batch)))
        .collect();
    Some(batch_losses.iter().sum::<f64>() / batch_losses.len() as f64)
}
