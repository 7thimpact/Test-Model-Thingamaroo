//! Where a brain keeps its files.
//!
//! ```text
//! <brain>/
//!   model.json                  architecture, fixed at creation
//!   settings.json               adjustable behaviour
//!   state.json                  counters and the surprise baseline
//!   history.jsonl               training, live-update and sleep records
//!   memory.redb                 episodic memory store
//!   weights/model.bpk           slow weights (encoder, predictor, decoder)
//!   weights/fast_weights.bpk    current fast weights
//!   training/                   optimiser state and target encoder, for resuming
//!   consolidation/              elastic anchor and replay reservoir, for sleep
//!   snapshots/<id>/             saved weights for rollback
//! ```

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{BrainError, BrainResult};

const TEMPORARY_SUFFIX: &str = "tmp";

/// Paths of every file inside one brain directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrainDirectory {
    root: PathBuf,
}

impl BrainDirectory {
    /// Describes the brain directory at `root`, without touching the disk.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory itself.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether a brain has already been created here.
    pub fn holds_brain(&self) -> bool {
        self.model_configuration_path().is_file()
    }

    /// Creates the directory and its subdirectories.
    pub fn create_layout(&self) -> BrainResult<()> {
        let subdirectories = [
            self.weights_directory(),
            self.training_directory(),
            self.consolidation_directory(),
            self.snapshots_directory(),
        ];
        for subdirectory in subdirectories {
            std::fs::create_dir_all(&subdirectory)
                .map_err(|source| BrainError::file_system(&subdirectory, source))?;
        }
        Ok(())
    }

    /// The architecture document.
    pub fn model_configuration_path(&self) -> PathBuf {
        self.root.join("model.json")
    }

    /// The adjustable-behaviour document.
    pub fn settings_path(&self) -> PathBuf {
        self.root.join("settings.json")
    }

    /// The counters document.
    pub fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }

    /// The append-only record of training, live updates and sleep.
    pub fn history_path(&self) -> PathBuf {
        self.root.join("history.jsonl")
    }

    /// The episodic memory database.
    pub fn memory_store_path(&self) -> PathBuf {
        self.root.join("memory.redb")
    }

    fn weights_directory(&self) -> PathBuf {
        self.root.join("weights")
    }

    /// The slow weights of encoder, predictor and decoder.
    pub fn slow_weights_path(&self) -> PathBuf {
        self.weights_directory().join("model.bpk")
    }

    /// The current fast weights.
    pub fn fast_weights_path(&self) -> PathBuf {
        self.weights_directory().join("fast_weights.bpk")
    }

    fn training_directory(&self) -> PathBuf {
        self.root.join("training")
    }

    /// The pretraining optimiser's state.
    pub fn optimizer_path(&self) -> PathBuf {
        self.training_directory().join("optimizer.bpk")
    }

    /// The slowly updated encoder copy used for latent targets.
    pub fn target_encoder_path(&self) -> PathBuf {
        self.training_directory().join("target_encoder.bpk")
    }

    fn consolidation_directory(&self) -> PathBuf {
        self.root.join("consolidation")
    }

    /// The predictor weights recorded for elastic weight consolidation.
    pub fn anchor_values_path(&self) -> PathBuf {
        self.consolidation_directory().join("anchor_values.bpk")
    }

    /// How much each recorded weight matters.
    pub fn anchor_importances_path(&self) -> PathBuf {
        self.consolidation_directory()
            .join("anchor_importances.bpk")
    }

    /// Samples of the original training data.
    pub fn replay_reservoir_path(&self) -> PathBuf {
        self.consolidation_directory().join("replay_reservoir.bin")
    }

    /// The directory holding one subdirectory per snapshot.
    pub fn snapshots_directory(&self) -> PathBuf {
        self.root.join("snapshots")
    }
}

/// Reads a JSON document.
pub fn load_document<Document: DeserializeOwned>(path: &Path) -> BrainResult<Document> {
    let contents =
        std::fs::read_to_string(path).map_err(|source| BrainError::file_system(path, source))?;
    serde_json::from_str(&contents).map_err(|error| BrainError::document(path, error))
}

/// Reads a JSON document, or returns its default when the file does not exist.
pub fn load_document_or_default<Document: DeserializeOwned + Default>(
    path: &Path,
) -> BrainResult<Document> {
    if !path.exists() {
        return Ok(Document::default());
    }
    load_document(path)
}

/// Writes a JSON document so that a crash never leaves a half-written file.
pub fn store_document<Document: Serialize>(path: &Path, document: &Document) -> BrainResult<()> {
    let contents = serde_json::to_string_pretty(document)
        .map_err(|error| BrainError::document(path, error))?;
    let temporary_path = temporary_sibling(path);
    std::fs::write(&temporary_path, contents)
        .map_err(|source| BrainError::file_system(&temporary_path, source))?;
    replace_file(&temporary_path, path)
}

/// The path a file is first written to before it replaces `path`.
pub fn temporary_sibling(path: &Path) -> PathBuf {
    let mut file_name = path.file_name().unwrap_or_default().to_os_string();
    file_name.push(".");
    file_name.push(TEMPORARY_SUFFIX);
    path.with_file_name(file_name)
}

/// Moves a finished temporary file over its destination.
pub fn replace_file(temporary_path: &Path, path: &Path) -> BrainResult<()> {
    std::fs::rename(temporary_path, path).map_err(|source| BrainError::file_system(path, source))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documents_round_trip_and_missing_files_fall_back_to_defaults() {
        let path = std::env::temp_dir().join("thingamaroo-directory-document.json");
        let _ = std::fs::remove_file(&path);

        let missing: Vec<u32> = load_document_or_default(&path).unwrap();
        assert!(missing.is_empty());

        store_document(&path, &vec![1u32, 2, 3]).unwrap();
        let stored: Vec<u32> = load_document(&path).unwrap();
        assert_eq!(stored, [1, 2, 3]);
        assert!(!temporary_sibling(&path).exists());
    }

    #[test]
    fn a_directory_without_a_model_document_holds_no_brain() {
        let directory = BrainDirectory::new(std::env::temp_dir().join("thingamaroo-no-brain-here"));

        assert!(!directory.holds_brain());
    }
}
