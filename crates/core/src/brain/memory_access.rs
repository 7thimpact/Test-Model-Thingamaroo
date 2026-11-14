//! Reading and editing the episodic memory from outside a conversation.

use crate::brain::Brain;
use crate::error::{BrainError, BrainResult};
use crate::memory::{MemoryDraft, MemoryEntry, MemoryKind, MemoryMatch};

const LINE_END: &str = "\n";

impl Brain {
    /// Every stored memory, oldest first.
    pub fn memories(&self) -> BrainResult<Vec<MemoryEntry>> {
        self.memory.list()
    }

    /// Number of stored memories.
    pub fn memory_count(&self) -> BrainResult<usize> {
        self.memory.entry_count()
    }

    /// Finds the memories closest in meaning to `query_text`, most similar first.
    ///
    /// Unlike retrieval during conversation, this applies no similarity
    /// threshold and does not count as a retrieval.
    pub fn search_memories(
        &mut self,
        query_text: &str,
        limit: usize,
    ) -> BrainResult<Vec<MemoryMatch>> {
        let Some(query) = self.inference.gist(query_text.as_bytes()) else {
            return Ok(Vec::new());
        };
        self.memory.search(&query, limit, f64::NEG_INFINITY)
    }

    /// Stores a fact by hand so it is retrieved and replayed like any other memory.
    pub fn add_note(&mut self, text: &str, importance: f64) -> BrainResult<MemoryEntry> {
        let note = ensure_line_end(text);
        let vector = self
            .inference
            .gist(note.as_bytes())
            .ok_or_else(|| BrainError::InvalidConfiguration("a note cannot be empty".into()))?;
        self.memory.insert(MemoryDraft {
            kind: MemoryKind::Note,
            prompt: "",
            reply: &note,
            importance: self.cap_importance(importance),
            vector: &vector,
        })
    }

    /// Deletes memories and returns how many existed.
    pub fn delete_memories(&mut self, ids: &[u64]) -> BrainResult<usize> {
        self.memory.delete(ids)
    }

    /// Deletes every memory and returns how many there were.
    pub fn clear_memories(&mut self) -> BrainResult<usize> {
        self.memory.clear()
    }

    /// Limits an importance score to the configured maximum, guarding against poisoning.
    pub(crate) fn cap_importance(&self, importance: f64) -> f64 {
        importance.clamp(0.0, self.settings.safety.maximum_importance)
    }
}

/// Appends a line break unless the text already ends with one.
pub(crate) fn ensure_line_end(text: &str) -> String {
    match text.ends_with(LINE_END) {
        true => text.to_string(),
        false => format!("{text}{LINE_END}"),
    }
}
