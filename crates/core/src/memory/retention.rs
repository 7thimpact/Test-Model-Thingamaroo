//! Decides which memories fade: "store gist, prune the rest".

use crate::configuration::{MemorySettings, SleepSettings};
use crate::memory::entry::MemoryEntry;

/// How much each (logarithmic) retrieval adds to a memory's retention score.
const RETRIEVAL_BONUS: f64 = 0.25;

/// What sleep decided about the stored memories.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RetentionPlan {
    /// Memories that stay, with their importance already decayed.
    pub kept_entries: Vec<MemoryEntry>,
    /// Identifiers of memories to delete.
    pub pruned_ids: Vec<u64>,
}

/// How strongly a memory resists pruning: importance, raised by being retrieved.
pub fn retention_score(entry: &MemoryEntry) -> f64 {
    let retrieval_factor = 1.0 + RETRIEVAL_BONUS * (1.0 + entry.retrieval_count as f64).ln();
    entry.importance * retrieval_factor
}

/// Decays every memory's importance, then prunes the weak and the excess.
pub fn plan_retention(
    entries: Vec<MemoryEntry>,
    sleep: &SleepSettings,
    memory: &MemorySettings,
) -> RetentionPlan {
    let mut decayed_entries: Vec<MemoryEntry> = entries
        .into_iter()
        .map(|entry| decay_importance(entry, sleep.importance_decay))
        .collect();
    // Strongest first, so the entry-count limit drops the weakest memories.
    decayed_entries.sort_by(|left, right| retention_score(right).total_cmp(&retention_score(left)));

    let mut plan = RetentionPlan::default();
    for entry in decayed_entries {
        let is_strong_enough = retention_score(&entry) >= sleep.minimum_retention_score;
        let has_room = plan.kept_entries.len() < memory.maximum_entry_count;
        if is_strong_enough && has_room {
            plan.kept_entries.push(entry);
            continue;
        }
        plan.pruned_ids.push(entry.id);
    }
    plan
}

fn decay_importance(mut entry: MemoryEntry, importance_decay: f64) -> MemoryEntry {
    entry.importance *= importance_decay.clamp(0.0, 1.0);
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::entry::MemoryKind;

    fn entry(id: u64, importance: f64, retrieval_count: u64) -> MemoryEntry {
        MemoryEntry {
            id,
            kind: MemoryKind::Exchange,
            prompt: String::new(),
            reply: String::new(),
            created_unix_seconds: 0,
            importance,
            retrieval_count,
            last_retrieved_unix_seconds: 0,
        }
    }

    #[test]
    fn weak_unused_memories_are_pruned_and_the_rest_decay() {
        let sleep = SleepSettings {
            importance_decay: 0.5,
            minimum_retention_score: 0.2,
            ..SleepSettings::default()
        };
        let entries = vec![entry(1, 0.9, 0), entry(2, 0.3, 0), entry(3, 0.3, 20)];

        let plan = plan_retention(entries, &sleep, &MemorySettings::default());

        let kept_ids: Vec<u64> = plan.kept_entries.iter().map(|kept| kept.id).collect();
        assert_eq!(kept_ids, [1, 3]);
        assert_eq!(plan.pruned_ids, [2]);
        assert!((plan.kept_entries[0].importance - 0.45).abs() < 1e-9);
    }

    #[test]
    fn the_entry_limit_drops_the_weakest_memories() {
        let sleep = SleepSettings {
            importance_decay: 1.0,
            minimum_retention_score: 0.0,
            ..SleepSettings::default()
        };
        let memory = MemorySettings {
            maximum_entry_count: 2,
            ..MemorySettings::default()
        };
        let entries = vec![entry(1, 0.2, 0), entry(2, 0.8, 0), entry(3, 0.5, 0)];

        let plan = plan_retention(entries, &sleep, &memory);

        let kept_ids: Vec<u64> = plan.kept_entries.iter().map(|kept| kept.id).collect();
        assert_eq!(kept_ids, [2, 3]);
        assert_eq!(plan.pruned_ids, [1]);
    }
}
