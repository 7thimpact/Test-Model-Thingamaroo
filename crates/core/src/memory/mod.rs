//! Episodic memory: what the brain experienced, weighted by how much it mattered.

mod entry;
mod retention;
mod similarity;
mod store;
mod surprise;

pub use entry::{MemoryDraft, MemoryEntry, MemoryKind, MemoryMatch, current_unix_seconds};
pub use retention::{RetentionPlan, plan_retention, retention_score};
pub use similarity::{BestMatches, cosine_similarity};
pub use store::EpisodicStore;
pub use surprise::{SurpriseBaseline, opens_with_correction_marker};
