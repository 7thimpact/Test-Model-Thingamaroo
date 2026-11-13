//! Learning on three timescales: pretraining, live fast-weight updates, and sleep.

mod consolidation;
mod elastic_anchor;
mod learning_rate;
mod lesson;
mod live_adapter;
mod parameter_tensors;
mod pretraining;
mod progress;
mod target_encoder;
mod trainable_module;

pub use consolidation::{ConsolidationInput, ConsolidationReport, consolidate, measure_lessons};
pub use elastic_anchor::ElasticAnchor;
pub use learning_rate::LearningRateSchedule;
pub use lesson::{Lesson, LessonContext, lesson_loss};
pub use live_adapter::{
    LiveUpdateReport, attach_fast_weights, build_fast_weight_optimizer, fast_weight_group,
    learn_lesson, shrink_fast_weights,
};
pub use parameter_tensors::ParameterTensors;
pub use pretraining::{
    Pretrainer, PretrainerParts, TrainingBatch, measure_batch, pretraining_objective,
};
pub use progress::{
    LossBreakdown, ProgressEvent, ProgressObserver, SleepStepReport, TrainingStepReport,
    ValidationReport,
};
pub use target_encoder::TargetEncoder;
pub use trainable_module::{TrainableModule, build_optimizer};
