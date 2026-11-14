//! What long-running operations report while they work.

use serde::{Deserialize, Serialize};

use crate::model::NATS_PER_BIT;

/// The value of every training loss at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct LossBreakdown {
    /// The weighted sum that was minimised.
    pub total: f64,
    /// Rebuilding patches from their own meaning vectors, in nats per symbol.
    pub reconstruction: f64,
    /// Writing the next patch from the predicted meaning, in nats per symbol.
    pub generation: f64,
    /// Predicting the next meaning vector.
    pub latent: f64,
    /// The penalty that keeps meaning vectors spread out.
    pub variance: f64,
}

impl LossBreakdown {
    /// The generation loss as bits per byte, the usual way to quote byte-level models.
    pub fn generation_bits_per_byte(&self) -> f64 {
        self.generation / NATS_PER_BIT
    }

    /// Whether every term is a finite number.
    pub fn is_finite(&self) -> bool {
        [
            self.total,
            self.reconstruction,
            self.generation,
            self.latent,
            self.variance,
        ]
        .iter()
        .all(|term| term.is_finite())
    }

    /// Adds another breakdown term by term, for averaging.
    pub fn accumulate(&mut self, other: &LossBreakdown) {
        self.total += other.total;
        self.reconstruction += other.reconstruction;
        self.generation += other.generation;
        self.latent += other.latent;
        self.variance += other.variance;
    }

    /// Divides every term by `count`, turning a sum into a mean.
    pub fn divided_by(&self, count: usize) -> LossBreakdown {
        let divisor = count.max(1) as f64;
        LossBreakdown {
            total: self.total / divisor,
            reconstruction: self.reconstruction / divisor,
            generation: self.generation / divisor,
            latent: self.latent / divisor,
            variance: self.variance / divisor,
        }
    }
}

/// One gradient step of pretraining.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingStepReport {
    /// Steps taken over the brain's whole life, including this one.
    pub lifetime_step: u64,
    /// Steps taken in this run, including this one.
    pub run_step: usize,
    /// Steps this run will take in total.
    pub run_step_count: usize,
    /// Learning rate used for this step.
    pub learning_rate: f64,
    /// Losses on this step's batch.
    pub losses: LossBreakdown,
    /// Training speed over the recent steps.
    pub patches_per_second: f64,
}

/// A measurement on held-out text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Lifetime step at which the measurement was taken.
    pub lifetime_step: u64,
    /// Average losses over the validation batches.
    pub losses: LossBreakdown,
    /// What the model wrote when asked to continue the sample prompt.
    pub sample: String,
}

/// One gradient step of sleep consolidation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SleepStepReport {
    /// Replay steps taken so far, including this one.
    pub step: usize,
    /// Replay steps this sleep will take in total.
    pub step_count: usize,
    /// Loss on this step's mix of memories and old data, in nats per symbol.
    pub replay_loss: f64,
    /// The elastic penalty holding important weights in place.
    pub elastic_penalty: f64,
}

/// Something a long-running operation wants the user to know.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgressEvent {
    /// A plain status line.
    Status(String),
    /// A pretraining step finished.
    TrainingStep(TrainingStepReport),
    /// Held-out text was measured.
    Validation(ValidationReport),
    /// Weights were written to disk.
    CheckpointSaved {
        /// Lifetime step the checkpoint holds.
        lifetime_step: u64,
    },
    /// A sleep replay step finished.
    SleepStep(SleepStepReport),
}

/// Receives progress events; front ends pass a closure that prints or plots them.
pub type ProgressObserver<'observer> = &'observer mut dyn FnMut(ProgressEvent);
