//! An append-only record of how a brain has learned, for plots and audits.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::brain::sleep_cycle::SleepReport;
use crate::error::{BrainError, BrainResult};
use crate::learning::{LiveUpdateReport, TrainingStepReport, ValidationReport};

/// One line of the history file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistoryRecord {
    /// A sampled pretraining step.
    TrainingStep(TrainingStepReport),
    /// A measurement on held-out text.
    Validation(ValidationReport),
    /// A fast-weight update made during conversation.
    LiveUpdate {
        /// When the update happened.
        unix_seconds: u64,
        /// How important the triggering exchange was.
        importance: f64,
        /// What the update did.
        report: LiveUpdateReport,
    },
    /// A finished sleep.
    Sleep(SleepReport),
}

/// Appends one record to the history file.
pub fn append_record(path: &Path, record: &HistoryRecord) -> BrainResult<()> {
    let line = serde_json::to_string(record).map_err(|error| BrainError::document(path, error))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| BrainError::file_system(path, source))?;
    writeln!(file, "{line}").map_err(|source| BrainError::file_system(path, source))
}

/// Reads every record; a missing file is an empty history.
///
/// Lines that cannot be parsed are skipped, so a history written by a newer
/// version (or cut short by a crash) never blocks opening a brain.
pub fn read_records(path: &Path) -> BrainResult<Vec<HistoryRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents =
        std::fs::read_to_string(path).map_err(|source| BrainError::file_system(path, source))?;
    Ok(contents
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_appended_and_read_back_in_order() {
        let path = std::env::temp_dir().join("thingamaroo-history-append.jsonl");
        let _ = std::fs::remove_file(&path);
        let first = HistoryRecord::LiveUpdate {
            unix_seconds: 10,
            importance: 0.9,
            report: LiveUpdateReport {
                step_count: 3,
                loss_before: 2.0,
                loss_after: 1.0,
            },
        };
        let second = HistoryRecord::Sleep(SleepReport::default());

        append_record(&path, &first).unwrap();
        append_record(&path, &second).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"not json\n")
            .unwrap();

        assert_eq!(read_records(&path).unwrap(), [first, second]);
    }
}
