//! Counters and running statistics that change as a brain is used.
//!
//! Path convention: `state.<group>.<field>`, for example
//! `state.conversation.turn_count`.

use serde::{Deserialize, Serialize};

use crate::memory::SurpriseBaseline;

/// Everything a brain counts about its own life.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BrainState {
    /// Progress of pretraining.
    pub training: TrainingState,
    /// The model's usual prediction error.
    pub surprise: SurpriseBaseline,
    /// Progress of conversation and live learning.
    pub conversation: ConversationState,
    /// Progress of sleep consolidation.
    pub sleep: SleepState,
}

/// Progress of pretraining.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingState {
    /// Pretraining steps taken over the brain's whole life.
    pub step_count: u64,
    /// Bits per byte on held-out text at the last validation, if any.
    pub validation_bits_per_byte: Option<f64>,
}

/// Progress of conversation and live learning.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ConversationState {
    /// Conversation turns over the brain's whole life.
    pub turn_count: u64,
    /// Conversation turns since the last sleep.
    pub turns_since_sleep: u64,
    /// Fast-weight updates over the brain's whole life.
    pub live_update_count: u64,
    /// Fast-weight updates since the last sleep.
    pub live_updates_since_sleep: u64,
}

/// Progress of sleep consolidation.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SleepState {
    /// Completed sleeps.
    pub cycle_count: u64,
    /// When the last sleep finished; zero when the brain has never slept.
    pub last_sleep_unix_seconds: u64,
    /// When the brain was created or last woke up; the hourly schedule counts from here.
    pub awake_since_unix_seconds: u64,
}
