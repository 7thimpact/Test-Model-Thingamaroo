//! Behaviour that can be changed at any time without retraining.
//!
//! Path convention: `settings.<group>.<field>`, for example
//! `settings.sleep.replay_step_count`. Every field has a default, so a
//! settings file may list only the values it wants to change.

use serde::{Deserialize, Serialize};

/// Every adjustable behaviour of a brain, grouped by purpose.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeSettings {
    /// How replies are produced.
    pub generation: GenerationSettings,
    /// How conversation turns are framed.
    pub chat: ChatSettings,
    /// How the episodic memory is written and read.
    pub memory: MemorySettings,
    /// How prediction error becomes an importance score.
    pub surprise: SurpriseSettings,
    /// How the fast weights learn during conversation.
    pub learning: LiveLearningSettings,
    /// How one sleep consolidation run behaves.
    pub sleep: SleepSettings,
    /// When sleep happens on its own.
    pub schedule: SleepScheduleSettings,
    /// Guards against poisoning and runaway updates.
    pub safety: SafetySettings,
    /// How much each training loss counts.
    pub objective: ObjectiveSettings,
}

/// How replies are produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationSettings {
    /// Sampling temperature; zero always picks the most likely byte.
    pub temperature: f64,
    /// Only this many most-likely symbols are considered for each byte.
    pub candidate_count: usize,
    /// Internal refinement steps before a meaning vector is decoded.
    pub thinking_step_count: usize,
    /// Longest reply the model may write.
    pub maximum_reply_bytes: usize,
    /// Fixes the sampling sequence when set; otherwise the clock seeds it.
    pub seed: Option<u64>,
}

/// How conversation turns are framed as bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatSettings {
    /// Text placed before each user message, matching the training data's format.
    pub user_prefix: String,
    /// Text placed before each model reply, matching the training data's format.
    pub model_prefix: String,
    /// Whether a reply ends at the first line break the model writes.
    pub is_stopping_at_line_end: bool,
}

/// How the episodic memory is written and read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemorySettings {
    /// Whether retrieved memories are shown to the predictor.
    pub is_retrieval_enabled: bool,
    /// Whether new exchanges are written to the store.
    pub is_writing_enabled: bool,
    /// How many memories are retrieved per turn.
    pub retrieval_count: usize,
    /// Cosine similarity a memory needs before it is retrieved.
    pub minimum_similarity: f64,
    /// Importance an exchange needs before it is stored at all.
    pub minimum_stored_importance: f64,
    /// Hard cap on stored memories; the least valuable are pruned beyond it.
    pub maximum_entry_count: usize,
}

/// How prediction error becomes an importance score in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SurpriseSettings {
    /// How slowly the running average of surprise adapts (closer to 1 is slower).
    pub baseline_momentum: f64,
    /// Importance given to an exchange the user corrected.
    pub correction_importance: f64,
    /// Importance given to an exchange the user explicitly praised.
    pub praise_importance: f64,
    /// Message openings that mark a correction, compared without letter case.
    pub correction_markers: Vec<String>,
}

/// How the fast weights learn during conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LiveLearningSettings {
    /// Master switch for weight updates during conversation.
    pub is_enabled: bool,
    /// Learn when the user corrects the model.
    pub is_learning_from_corrections: bool,
    /// Learn when an input was predicted badly.
    pub is_learning_from_surprise: bool,
    /// Learn when the user explicitly praises a reply.
    pub is_learning_from_praise: bool,
    /// Importance a surprising input needs before it triggers an update.
    pub surprise_importance_threshold: f64,
    /// Step size of fast-weight updates at full importance.
    pub learning_rate: f64,
    /// Gradient steps taken at full importance; weaker signals take fewer.
    pub maximum_step_count: usize,
}

/// How one sleep consolidation run behaves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SleepSettings {
    /// Gradient steps on the slow weights per sleep.
    pub replay_step_count: usize,
    /// Sequences per replay batch.
    pub batch_sequence_count: usize,
    /// How many of the most important memories are replayed.
    pub episode_count: usize,
    /// Share of each replay batch drawn from the original training data.
    pub old_data_fraction: f64,
    /// Step size of slow-weight updates.
    pub learning_rate: f64,
    /// Strength of elastic weight consolidation; zero disables it.
    ///
    /// Weight importances are squared gradients of a batch-averaged loss, which
    /// are tiny numbers, so useful strengths are large. Raise it until the
    /// "old skills" loss stops rising during sleep; lower it if memories stop
    /// being learned.
    pub elastic_strength: f64,
    /// Batches of old data used to estimate which weights matter.
    pub importance_batch_count: usize,
    /// Share of the fast weights kept after sleep; zero resets them.
    pub fast_weight_retention: f64,
    /// Every memory's importance is multiplied by this at each sleep.
    pub importance_decay: f64,
    /// Memories scoring below this after decay are deleted.
    pub minimum_retention_score: f64,
    /// Sleep is undone when old-data loss rises by more than this fraction.
    pub maximum_old_loss_increase: f64,
}

/// When sleep happens on its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SleepScheduleSettings {
    /// Master switch for automatic sleep.
    pub is_automatic: bool,
    /// Sleep after this many conversation turns.
    pub turn_interval: usize,
    /// Sleep after this long without input while a session is open.
    pub idle_seconds: u64,
    /// Sleep at least this often while a session is open.
    pub hour_interval: u64,
    /// Sleep when the store holds this many memories.
    pub memory_count_limit: usize,
}

/// Guards against poisoning and runaway updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SafetySettings {
    /// Inputs containing any of these phrases are neither stored nor learned from.
    pub blocked_phrases: Vec<String>,
    /// Inputs shorter than this are neither stored nor learned from.
    pub minimum_input_bytes: usize,
    /// Inputs longer than this are neither stored nor learned from.
    pub maximum_input_bytes: usize,
    /// No single exchange can be more important than this.
    pub maximum_importance: f64,
    /// Automatic snapshots kept for rollback; older ones are deleted.
    pub maximum_automatic_snapshots: usize,
}

/// How much each training loss counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectiveSettings {
    /// Weight of rebuilding a patch's bytes from its own meaning vector.
    pub reconstruction_weight: f64,
    /// Weight of writing the next patch's bytes from the predicted meaning vector.
    pub generation_weight: f64,
    /// Weight of predicting the next meaning vector in latent space.
    pub latent_weight: f64,
    /// Weight of the penalty that keeps meaning vectors spread out.
    pub variance_weight: f64,
    /// How slowly the target encoder follows the trained encoder.
    pub target_encoder_momentum: f64,
    /// Chance that a pretraining batch is shown gist vectors of preceding text.
    pub memory_slot_probability: f64,
    /// Patches summarised into each gist vector during pretraining.
    pub memory_span_patches: usize,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            candidate_count: 40,
            thinking_step_count: 2,
            maximum_reply_bytes: 240,
            seed: None,
        }
    }
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            user_prefix: String::new(),
            model_prefix: String::new(),
            is_stopping_at_line_end: true,
        }
    }
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            is_retrieval_enabled: true,
            is_writing_enabled: true,
            retrieval_count: 3,
            minimum_similarity: 0.35,
            minimum_stored_importance: 0.55,
            maximum_entry_count: 100_000,
        }
    }
}

impl Default for SurpriseSettings {
    fn default() -> Self {
        Self {
            baseline_momentum: 0.95,
            correction_importance: 0.95,
            praise_importance: 0.8,
            correction_markers: default_correction_markers(),
        }
    }
}

fn default_correction_markers() -> Vec<String> {
    [
        "no,",
        "no.",
        "wrong",
        "incorrect",
        "actually",
        "that's not",
        "that is not",
    ]
    .map(String::from)
    .to_vec()
}

impl Default for LiveLearningSettings {
    fn default() -> Self {
        Self {
            is_enabled: true,
            is_learning_from_corrections: true,
            is_learning_from_surprise: true,
            is_learning_from_praise: true,
            surprise_importance_threshold: 0.8,
            learning_rate: 2e-3,
            maximum_step_count: 12,
        }
    }
}

impl Default for SleepSettings {
    fn default() -> Self {
        Self {
            replay_step_count: 60,
            batch_sequence_count: 8,
            episode_count: 64,
            old_data_fraction: 0.5,
            learning_rate: 2e-4,
            elastic_strength: 1_000_000.0,
            importance_batch_count: 8,
            fast_weight_retention: 0.0,
            importance_decay: 0.9,
            minimum_retention_score: 0.15,
            maximum_old_loss_increase: 0.1,
        }
    }
}

impl Default for SleepScheduleSettings {
    fn default() -> Self {
        Self {
            is_automatic: true,
            turn_interval: 50,
            idle_seconds: 600,
            hour_interval: 24,
            memory_count_limit: 5_000,
        }
    }
}

impl Default for SafetySettings {
    fn default() -> Self {
        Self {
            blocked_phrases: Vec::new(),
            minimum_input_bytes: 2,
            maximum_input_bytes: 2_000,
            maximum_importance: 0.95,
            maximum_automatic_snapshots: 10,
        }
    }
}

impl Default for ObjectiveSettings {
    fn default() -> Self {
        Self {
            reconstruction_weight: 1.0,
            generation_weight: 1.0,
            latent_weight: 1.0,
            variance_weight: 1.0,
            target_encoder_momentum: 0.99,
            memory_slot_probability: 0.5,
            memory_span_patches: 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partial_document_keeps_defaults_for_missing_fields() {
        let document = r#"{ "sleep": { "replay_step_count": 7 } }"#;

        let settings: RuntimeSettings = serde_json::from_str(document).unwrap();

        assert_eq!(settings.sleep.replay_step_count, 7);
        assert_eq!(
            settings.sleep.importance_decay,
            SleepSettings::default().importance_decay
        );
        assert_eq!(settings.generation, GenerationSettings::default());
    }

    #[test]
    fn settings_round_trip_through_json() {
        let settings = RuntimeSettings::default();

        let document = serde_json::to_string(&settings).unwrap();
        let restored: RuntimeSettings = serde_json::from_str(&document).unwrap();

        assert_eq!(restored, settings);
    }
}
