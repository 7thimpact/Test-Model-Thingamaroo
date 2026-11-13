//! Options for one pretraining run.
//!
//! Path convention: `training_options.<group>.<field>`, for example
//! `training_options.schedule.step_count`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{BrainError, BrainResult};

/// Everything a pretraining run needs to know.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingOptions {
    /// Where the training text comes from.
    pub data: TrainingDataOptions,
    /// How long to train and how the step size changes.
    pub schedule: TrainingScheduleOptions,
    /// How the optimiser is regularised.
    pub optimizer: OptimizerOptions,
    /// How often progress is reported and saved.
    pub reporting: ReportingOptions,
}

/// Where the training text comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingDataOptions {
    /// Text files, or directories searched recursively for text files.
    pub corpus_paths: Vec<PathBuf>,
    /// Share of the corpus held out to measure progress on unseen text.
    pub validation_fraction: f64,
    /// Sequences per batch.
    pub batch_sequence_count: usize,
    /// Windows of training text kept for replay during sleep.
    pub replay_window_count: usize,
    /// Fixes batch sampling and weight initialisation.
    pub seed: u64,
}

/// How long to train and how the step size changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingScheduleOptions {
    /// Gradient steps in this run.
    pub step_count: usize,
    /// Steps over which the learning rate rises from zero.
    pub warmup_step_count: usize,
    /// Largest learning rate, reached at the end of warm-up.
    pub peak_learning_rate: f64,
    /// Learning rate at the last step, as a fraction of the peak.
    pub final_learning_rate_fraction: f64,
}

/// How the optimiser is regularised.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizerOptions {
    /// Decoupled weight decay applied at every step.
    pub weight_decay: f64,
    /// Gradients of each weight tensor are rescaled to at most this norm.
    pub gradient_clip_norm: f64,
}

/// How often progress is reported and saved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReportingOptions {
    /// Steps between progress reports.
    pub report_interval_steps: usize,
    /// Steps between validation measurements and sample generations.
    pub validation_interval_steps: usize,
    /// Steps between saving weights to disk.
    pub checkpoint_interval_steps: usize,
    /// Batches of held-out text used for each validation measurement.
    pub validation_batch_count: usize,
    /// Prompt continued at each validation to show what the model writes.
    pub sample_prompt: String,
}

impl Default for TrainingDataOptions {
    fn default() -> Self {
        Self {
            corpus_paths: Vec::new(),
            validation_fraction: 0.02,
            batch_sequence_count: 16,
            replay_window_count: 2_000,
            seed: 42,
        }
    }
}

impl Default for TrainingScheduleOptions {
    fn default() -> Self {
        Self {
            step_count: 2_000,
            warmup_step_count: 100,
            peak_learning_rate: 1e-3,
            final_learning_rate_fraction: 0.1,
        }
    }
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        Self {
            weight_decay: 0.01,
            gradient_clip_norm: 1.0,
        }
    }
}

impl Default for ReportingOptions {
    fn default() -> Self {
        Self {
            report_interval_steps: 10,
            validation_interval_steps: 200,
            checkpoint_interval_steps: 500,
            validation_batch_count: 4,
            sample_prompt: String::from("The "),
        }
    }
}

impl TrainingOptions {
    /// Checks that the run can start, returning the first problem found.
    pub fn validate(&self) -> BrainResult<()> {
        if self.data.corpus_paths.is_empty() {
            return invalid("at least one corpus path is required");
        }
        if self.data.batch_sequence_count == 0 {
            return invalid("data.batch_sequence_count must be greater than zero");
        }
        if !(0.0..0.5).contains(&self.data.validation_fraction) {
            return invalid("data.validation_fraction must be in [0, 0.5)");
        }
        if self.schedule.step_count == 0 {
            return invalid("schedule.step_count must be greater than zero");
        }
        if self.schedule.peak_learning_rate <= 0.0 {
            return invalid("schedule.peak_learning_rate must be greater than zero");
        }
        Ok(())
    }
}

fn invalid(reason: &str) -> BrainResult<()> {
    Err(BrainError::InvalidConfiguration(reason.to_string()))
}
