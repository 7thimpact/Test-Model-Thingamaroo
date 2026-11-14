//! Snapshots: saved weights a brain can be rolled back to after a bad update.
//!
//! A snapshot always holds the fast weights. A snapshot taken before sleep
//! also holds the slow predictor and its elastic anchor, so a consolidation
//! that went wrong can be undone completely.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::store::ModuleRecord;
use serde::{Deserialize, Serialize};

use crate::brain::directory::{load_document, store_document};
use crate::brain::{Brain, weights_error};
use crate::error::{BrainError, BrainResult};
use crate::memory::current_unix_seconds;

const INFORMATION_FILE_NAME: &str = "snapshot.json";
const FAST_WEIGHTS_FILE_NAME: &str = "fast_weights.bpk";
const PREDICTOR_FILE_NAME: &str = "predictor.bpk";
const ANCHOR_VALUES_FILE_NAME: &str = "anchor_values.bpk";
const ANCHOR_IMPORTANCES_FILE_NAME: &str = "anchor_importances.bpk";

/// What a snapshot is and what it contains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotInformation {
    /// Increasing identifier; later snapshots have larger identifiers.
    pub id: u64,
    /// Why the snapshot was taken.
    pub label: String,
    /// When the snapshot was taken.
    pub created_unix_seconds: u64,
    /// Whether the brain took it on its own before a risky update.
    pub is_automatic: bool,
    /// Whether it also holds the slow predictor, not only the fast weights.
    pub includes_slow_weights: bool,
}

impl Brain {
    /// Saves the current weights so they can be restored later.
    pub fn create_snapshot(
        &mut self,
        label: &str,
        includes_slow_weights: bool,
    ) -> BrainResult<SnapshotInformation> {
        self.write_snapshot(label, false, includes_slow_weights)
    }

    /// Saves the current weights before a risky update, keeping only the newest few.
    pub(crate) fn create_automatic_snapshot(
        &mut self,
        label: &str,
        includes_slow_weights: bool,
    ) -> BrainResult<SnapshotInformation> {
        let information = self.write_snapshot(label, true, includes_slow_weights)?;
        self.prune_automatic_snapshots()?;
        Ok(information)
    }

    fn write_snapshot(
        &self,
        label: &str,
        is_automatic: bool,
        includes_slow_weights: bool,
    ) -> BrainResult<SnapshotInformation> {
        let information = SnapshotInformation {
            id: self.next_snapshot_id()?,
            label: label.to_string(),
            created_unix_seconds: current_unix_seconds(),
            is_automatic,
            includes_slow_weights,
        };
        let snapshot_directory = self.snapshot_directory(information.id);
        std::fs::create_dir_all(&snapshot_directory)
            .map_err(|source| BrainError::file_system(&snapshot_directory, source))?;

        self.fast_weight_record()
            .save(snapshot_directory.join(FAST_WEIGHTS_FILE_NAME))
            .map_err(weights_error)?;
        if includes_slow_weights {
            self.write_slow_snapshot(&snapshot_directory)?;
        }
        store_document(
            &snapshot_directory.join(INFORMATION_FILE_NAME),
            &information,
        )?;
        Ok(information)
    }

    fn write_slow_snapshot(&self, snapshot_directory: &Path) -> BrainResult<()> {
        self.slow_model
            .predictor
            .clone()
            .save_file(snapshot_directory.join(PREDICTOR_FILE_NAME))
            .map_err(weights_error)?;
        for (source_path, file_name) in self.anchor_files() {
            copy_if_present(&source_path, &snapshot_directory.join(file_name))?;
        }
        Ok(())
    }

    /// The elastic anchor's files and the names they have inside a snapshot.
    fn anchor_files(&self) -> [(PathBuf, &'static str); 2] {
        [
            (self.directory.anchor_values_path(), ANCHOR_VALUES_FILE_NAME),
            (
                self.directory.anchor_importances_path(),
                ANCHOR_IMPORTANCES_FILE_NAME,
            ),
        ]
    }

    /// Lists every snapshot, oldest first.
    pub fn snapshots(&self) -> BrainResult<Vec<SnapshotInformation>> {
        let snapshots_directory = self.directory.snapshots_directory();
        if !snapshots_directory.exists() {
            return Ok(Vec::new());
        }
        let entries = std::fs::read_dir(&snapshots_directory)
            .map_err(|source| BrainError::file_system(&snapshots_directory, source))?;

        let mut snapshots: Vec<SnapshotInformation> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path().join(INFORMATION_FILE_NAME))
            .filter(|information_path| information_path.is_file())
            .filter_map(|information_path| load_document(&information_path).ok())
            .collect();
        snapshots.sort_by_key(|information| information.id);
        Ok(snapshots)
    }

    /// Restores the weights held by a snapshot and makes them current.
    pub fn roll_back(&mut self, snapshot_id: u64) -> BrainResult<SnapshotInformation> {
        let information = self.find_snapshot(snapshot_id)?;
        let snapshot_directory = self.snapshot_directory(snapshot_id);

        if information.includes_slow_weights {
            self.restore_slow_snapshot(&snapshot_directory)?;
        }
        let fast_weight_record =
            ModuleRecord::load(snapshot_directory.join(FAST_WEIGHTS_FILE_NAME))
                .map_err(weights_error)?;
        self.replace_fast_weights_from(fast_weight_record)?;

        self.save_slow_weights()?;
        self.save_fast_weights()?;
        Ok(information)
    }

    fn restore_slow_snapshot(&mut self, snapshot_directory: &Path) -> BrainResult<()> {
        self.slow_model.predictor = self
            .slow_model
            .predictor
            .clone()
            .try_load_file(snapshot_directory.join(PREDICTOR_FILE_NAME))
            .map_err(weights_error)?;
        for (destination_path, file_name) in self.anchor_files() {
            copy_if_present(&snapshot_directory.join(file_name), &destination_path)?;
        }
        Ok(())
    }

    /// Deletes one snapshot.
    pub fn delete_snapshot(&mut self, snapshot_id: u64) -> BrainResult<()> {
        self.find_snapshot(snapshot_id)?;
        remove_directory(&self.snapshot_directory(snapshot_id))
    }

    /// Deletes every snapshot, for when the weights they attach to are replaced.
    pub(crate) fn delete_all_snapshots(&mut self) -> BrainResult<usize> {
        let snapshots = self.snapshots()?;
        for information in &snapshots {
            remove_directory(&self.snapshot_directory(information.id))?;
        }
        Ok(snapshots.len())
    }

    fn find_snapshot(&self, snapshot_id: u64) -> BrainResult<SnapshotInformation> {
        self.snapshots()?
            .into_iter()
            .find(|information| information.id == snapshot_id)
            .ok_or_else(|| BrainError::NotFound(format!("snapshot {snapshot_id}")))
    }

    fn next_snapshot_id(&self) -> BrainResult<u64> {
        let newest_id = self
            .snapshots()?
            .last()
            .map_or(0, |information| information.id);
        Ok(newest_id + 1)
    }

    fn snapshot_directory(&self, snapshot_id: u64) -> PathBuf {
        self.directory
            .snapshots_directory()
            .join(format!("{snapshot_id:06}"))
    }

    /// Deletes the oldest automatic snapshots beyond the configured limit.
    fn prune_automatic_snapshots(&self) -> BrainResult<()> {
        let automatic_ids: Vec<u64> = self
            .snapshots()?
            .into_iter()
            .filter(|information| information.is_automatic)
            .map(|information| information.id)
            .collect();
        let excess_count = automatic_ids
            .len()
            .saturating_sub(self.settings.safety.maximum_automatic_snapshots);
        for &snapshot_id in &automatic_ids[..excess_count] {
            remove_directory(&self.snapshot_directory(snapshot_id))?;
        }
        Ok(())
    }
}

fn copy_if_present(source_path: &Path, destination_path: &Path) -> BrainResult<()> {
    if !source_path.exists() {
        return Ok(());
    }
    std::fs::copy(source_path, destination_path)
        .map(|_| ())
        .map_err(|source| BrainError::file_system(destination_path, source))
}

fn remove_directory(path: &Path) -> BrainResult<()> {
    std::fs::remove_dir_all(path).map_err(|source| BrainError::file_system(path, source))
}
