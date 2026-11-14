//! Everything that configures a brain.
//!
//! Three separate objects, each defined once and each at most three levels deep:
//!
//! - [`ModelConfiguration`]: the architecture, fixed when a brain is created.
//! - [`RuntimeSettings`]: behaviour that can change at any time.
//! - [`TrainingOptions`]: the parameters of one pretraining run.

mod model;
mod setting_entries;
mod settings;
mod training;

pub use model::{
    AdapterConfiguration, BYTE_VALUE_COUNT, ByteCodecConfiguration, DECODER_SYMBOL_COUNT,
    END_OF_PATCH_SYMBOL, FEED_FORWARD_EXPANSION, LatentConfiguration, ModelConfiguration,
    ModelPreset, ParameterCounts, PatchingConfiguration, PredictorConfiguration,
};
pub use setting_entries::{SettingEntry, describe_setting};
pub use settings::{
    ChatSettings, GenerationSettings, LiveLearningSettings, MemorySettings, ObjectiveSettings,
    RuntimeSettings, SafetySettings, SleepScheduleSettings, SleepSettings, SurpriseSettings,
};
pub use training::{
    OptimizerOptions, ReportingOptions, TrainingDataOptions, TrainingOptions,
    TrainingScheduleOptions,
};
