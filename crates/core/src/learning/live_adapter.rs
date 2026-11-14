//! Live learning: small updates to the fast weights during conversation.
//!
//! The predictor's slow weights are frozen here. Only the low-rank adapter
//! attached to them changes, so a bad update can be undone by restoring an
//! adapter snapshot and the core model is never touched live.

use burn::module::{Lora, Module, ModuleMapper, Param, ParamGroup};
use burn::optim::ModuleOptimizer;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::configuration::{AdapterConfiguration, LiveLearningSettings, OptimizerOptions};
use crate::error::{BrainError, BrainResult};
use crate::learning::lesson::{Lesson, LessonContext, lesson_loss};
use crate::learning::trainable_module::{TrainableModule, build_optimizer};
use crate::model::{ADAPTED_WEIGHT_PATTERN, ConceptPredictor, FAST_WEIGHT_PATTERN, read_scalar};

/// Fast weights are not decayed: they are reset or shrunk explicitly during sleep.
const FAST_WEIGHT_DECAY: f64 = 0.0;
const FAST_WEIGHT_GRADIENT_CLIP_NORM: f64 = 1.0;

/// Path segments the framework gives a low-rank adapter and its output factor.
const ADAPTER_PATH_NAME: &str = "lora";
const OUTPUT_FACTOR_PATH_NAME: &str = "b";

/// What one live update did.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LiveUpdateReport {
    /// Gradient steps taken.
    pub step_count: usize,
    /// Lesson loss before the first step.
    pub loss_before: f64,
    /// Lesson loss after the last step.
    pub loss_after: f64,
}

/// The parameters that receive fast weights.
fn adapted_weight_group() -> ParamGroup {
    ParamGroup::from_regex(ADAPTED_WEIGHT_PATTERN)
        .expect("the adapted-weight pattern is a valid regular expression")
}

/// The fast weights themselves.
pub fn fast_weight_group() -> ParamGroup {
    ParamGroup::from_regex(FAST_WEIGHT_PATTERN)
        .expect("the fast-weight pattern is a valid regular expression")
}

/// Attaches fresh, zero-effect fast weights to a copy of the slow predictor.
///
/// The slow weights are shared, not copied, and are frozen in the result.
pub fn attach_fast_weights(
    slow_predictor: &ConceptPredictor,
    adapter: &AdapterConfiguration,
) -> ConceptPredictor {
    let low_rank = Lora::new(adapter.rank, adapter.scale).set_param_group(adapted_weight_group());
    slow_predictor.clone().apply_lora(low_rank)
}

/// Multiplies the fast weights' effect by `retention`, keeping their learned directions.
///
/// Only each adapter's output factor is scaled, which scales the whole
/// low-rank update. Use [`attach_fast_weights`] instead for a full reset.
pub fn shrink_fast_weights(
    adapted_predictor: ConceptPredictor,
    retention: f64,
) -> ConceptPredictor {
    adapted_predictor.map(&mut OutputFactorScaler {
        path: Vec::new(),
        retention,
    })
}

/// Scales the parameters whose path ends in the adapter's output factor.
struct OutputFactorScaler {
    path: Vec<String>,
    retention: f64,
}

impl OutputFactorScaler {
    fn is_at_output_factor(&self) -> bool {
        let [.., adapter_name, factor_name] = self.path.as_slice() else {
            return false;
        };
        adapter_name == ADAPTER_PATH_NAME && factor_name == OUTPUT_FACTOR_PATH_NAME
    }
}

impl ModuleMapper for OutputFactorScaler {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const RANK: usize>(
        &mut self,
        parameter: Param<Tensor<RANK>>,
    ) -> Param<Tensor<RANK>> {
        if !self.is_at_output_factor() {
            return parameter;
        }
        let retention = self.retention;
        parameter.map(|output_factor| output_factor.mul_scalar(retention))
    }
}

/// Builds the optimiser for fast-weight updates.
pub fn build_fast_weight_optimizer() -> ModuleOptimizer {
    build_optimizer(&OptimizerOptions {
        weight_decay: FAST_WEIGHT_DECAY,
        gradient_clip_norm: FAST_WEIGHT_GRADIENT_CLIP_NORM,
    })
}

/// Teaches one lesson to the fast weights.
///
/// `strength` in `[0, 1]` scales the number of gradient steps, so corrections
/// are learned strongly and mildly surprising inputs only weakly.
pub fn learn_lesson(
    trainable: &mut TrainableModule<ConceptPredictor>,
    context: &LessonContext<'_>,
    lesson: &Lesson,
    learning: &LiveLearningSettings,
    strength: f64,
) -> BrainResult<LiveUpdateReport> {
    let lessons = std::slice::from_ref(lesson);
    let step_count = scaled_step_count(learning.maximum_step_count, strength);
    let mut loss_before = None;

    for _ in 0..step_count {
        let loss = lesson_loss(trainable.network(), context, lessons);
        let loss_value = read_scalar(loss.clone());
        if !loss_value.is_finite() {
            return Err(BrainError::Diverged(
                "a live update produced a non-finite loss".into(),
            ));
        }
        loss_before.get_or_insert(loss_value);
        trainable.descend(loss, learning.learning_rate);
    }

    let loss_after = read_scalar(lesson_loss(&trainable.network().valid(), context, lessons));
    Ok(LiveUpdateReport {
        step_count,
        loss_before: loss_before.unwrap_or(loss_after),
        loss_after,
    })
}

fn scaled_step_count(maximum_step_count: usize, strength: f64) -> usize {
    let scaled = (maximum_step_count as f64 * strength.clamp(0.0, 1.0)).ceil() as usize;
    scaled.max(1)
}
