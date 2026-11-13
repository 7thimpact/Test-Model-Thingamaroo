//! One stored exchange.

use serde::{Deserialize, Serialize};

/// Why an exchange was worth remembering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    /// An input the model predicted badly.
    Exchange,
    /// A reply the user replaced with the right answer.
    Correction,
    /// A reply the user explicitly approved.
    Praise,
    /// A fact added by hand.
    Note,
}

impl MemoryKind {
    /// The name shown in user interfaces.
    pub fn name(self) -> &'static str {
        match self {
            MemoryKind::Exchange => "exchange",
            MemoryKind::Correction => "correction",
            MemoryKind::Praise => "praise",
            MemoryKind::Note => "note",
        }
    }

    /// Whether the reply is known to be right and therefore worth learning.
    ///
    /// An ordinary exchange holds the model's own unverified reply, so only
    /// the user's side of it is learned.
    pub fn is_reply_trusted(self) -> bool {
        !matches!(self, MemoryKind::Exchange)
    }
}

/// One episodic memory: what was said, when, and how much it matters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique, increasing identifier.
    pub id: u64,
    /// Why this exchange was stored.
    pub kind: MemoryKind,
    /// What the user said.
    pub prompt: String,
    /// What was (or should have been) said in return.
    pub reply: String,
    /// When the memory was written.
    pub created_unix_seconds: u64,
    /// How much the memory matters, in `[0, 1]`.
    pub importance: f64,
    /// How many times the memory has been retrieved.
    pub retrieval_count: u64,
    /// When the memory was last retrieved; zero when it never was.
    pub last_retrieved_unix_seconds: u64,
}

impl MemoryEntry {
    /// The whole exchange as it appeared in the conversation.
    pub fn text(&self) -> String {
        format!("{}{}", self.prompt, self.reply)
    }
}

/// Everything needed to write a new memory.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDraft<'draft> {
    /// Why this exchange is being stored.
    pub kind: MemoryKind,
    /// What the user said.
    pub prompt: &'draft str,
    /// What was (or should have been) said in return.
    pub reply: &'draft str,
    /// How much the memory matters, in `[0, 1]`.
    pub importance: f64,
    /// The meaning vector the memory is retrieved by.
    pub vector: &'draft [f32],
}

/// A memory found by similarity search.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryMatch {
    /// The stored exchange.
    pub entry: MemoryEntry,
    /// Cosine similarity between the query and the memory's meaning vector.
    pub similarity: f64,
    /// The memory's meaning vector, shown to the predictor as a gist.
    pub vector: Vec<f32>,
}

/// Seconds since the Unix epoch, used to timestamp memories.
pub fn current_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
