//! A one-glance description of a brain.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::brain::Brain;
use crate::brain::state::BrainState;
use crate::configuration::ModelConfiguration;
use crate::error::BrainResult;
use crate::text::ReplayReservoir;

/// What a brain is, how big it is, and what it has been through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainSummary {
    /// Where the brain is stored.
    pub root: PathBuf,
    /// The compute backend in use.
    pub device_name: String,
    /// The architecture.
    pub model_configuration: ModelConfiguration,
    /// Slow parameters in the byte encoder.
    pub encoder_parameter_count: usize,
    /// Slow parameters in the concept predictor.
    pub predictor_parameter_count: usize,
    /// Slow parameters in the byte decoder.
    pub decoder_parameter_count: usize,
    /// Fast-weight parameters attached to the predictor.
    pub adapter_parameter_count: usize,
    /// Memories in the episodic store.
    pub memory_count: usize,
    /// Snapshots available for rollback.
    pub snapshot_count: usize,
    /// Windows of original training data kept for replay.
    pub replay_window_count: usize,
    /// Whether weight importances have been recorded for sleep.
    pub has_elastic_anchor: bool,
    /// Counters and running statistics.
    pub state: BrainState,
}

impl BrainSummary {
    /// Slow parameters across all three components.
    pub fn slow_parameter_count(&self) -> usize {
        self.encoder_parameter_count + self.predictor_parameter_count + self.decoder_parameter_count
    }

    /// Whether the brain has been pretrained at all.
    pub fn is_pretrained(&self) -> bool {
        self.state.training.step_count > 0
    }
}

impl Brain {
    /// Describes the brain as it is right now.
    pub fn summary(&self) -> BrainResult<BrainSummary> {
        let counts = self.model_configuration.parameter_counts();
        let reservoir = ReplayReservoir::load(&self.directory.replay_reservoir_path())?;
        Ok(BrainSummary {
            root: self.directory.root().to_path_buf(),
            device_name: self.compute_device.choice.name().to_string(),
            model_configuration: self.model_configuration.clone(),
            encoder_parameter_count: counts.encoder,
            predictor_parameter_count: counts.predictor,
            decoder_parameter_count: counts.decoder,
            adapter_parameter_count: counts.adapter,
            memory_count: self.memory.entry_count()?,
            snapshot_count: self.snapshots()?.len(),
            replay_window_count: reservoir.window_count(),
            has_elastic_anchor: self.directory.anchor_values_path().exists(),
            state: self.state.clone(),
        })
    }

    /// Every record of how this brain has learned, oldest first.
    pub fn history(&self) -> BrainResult<Vec<crate::brain::HistoryRecord>> {
        crate::brain::history::read_records(&self.directory.history_path())
    }
}
