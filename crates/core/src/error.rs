//! The single error type shared by every operation on a brain.

use std::path::PathBuf;

/// Everything that can go wrong while creating, training, or talking to a brain.
#[derive(Debug, thiserror::Error)]
pub enum BrainError {
    /// A file or directory could not be read or written.
    #[error("file system error at {path}: {source}")]
    FileSystem {
        /// The path the operation was acting on.
        path: PathBuf,
        /// The underlying operating-system error.
        source: std::io::Error,
    },

    /// A JSON document on disk could not be parsed or produced.
    #[error("could not read or write the document at {path}: {reason}")]
    Document {
        /// The document that failed.
        path: PathBuf,
        /// Why parsing or serialising failed.
        reason: String,
    },

    /// A setting or hyperparameter has a value the model cannot work with.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Model weights could not be saved, loaded, or matched to the model.
    #[error("model weights: {0}")]
    Weights(String),

    /// The on-disk episodic memory store reported a failure.
    #[error("episodic memory store: {0}")]
    MemoryStore(String),

    /// The training text was missing, empty, or too short to learn from.
    #[error("training corpus: {0}")]
    Corpus(String),

    /// Something the caller named does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// The requested compute device is not available in this build or on this machine.
    #[error("compute device: {0}")]
    ComputeDevice(String),

    /// Training produced a loss that is not a finite number.
    #[error("training diverged: {0}")]
    Diverged(String),
}

/// Result alias used across the crate.
pub type BrainResult<Value> = Result<Value, BrainError>;

impl BrainError {
    /// Wraps an operating-system error together with the path it happened on.
    pub fn file_system(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::FileSystem {
            path: path.into(),
            source,
        }
    }

    /// Wraps a parsing or serialising failure together with the document path.
    pub fn document(path: impl Into<PathBuf>, reason: impl ToString) -> Self {
        Self::Document {
            path: path.into(),
            reason: reason.to_string(),
        }
    }
}
