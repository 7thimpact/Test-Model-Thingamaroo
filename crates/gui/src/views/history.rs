//! The History tab: the record of training, live updates and sleep.
//!
//! Covers the command-line command `history`.

use thingamaroo_core::brain::HistoryRecord;
use thingamaroo_core::presentation::{describe_age, describe_live_update, describe_losses};

use crate::form::{hint, page, section};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

/// State of the History tab.
pub struct HistoryView {
    records: Vec<HistoryRecord>,
    shown_record_limit: usize,
}

impl Default for HistoryView {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            shown_record_limit: 100,
        }
    }
}

impl View for HistoryView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, &format!("History ({} records)", self.records.len()));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("Refresh"))
                .clicked()
            {
                shared.send(Request::RefreshHistory);
            }
            ui.label("Show the most recent");
            ui.add(egui::DragValue::new(&mut self.shown_record_limit).range(1..=10_000));
        });
        hint(ui, "Newest first.");
        page(ui, |ui| {
            egui::Grid::new("history_table")
                .striped(true)
                .show(ui, |ui| {
                    for record in self.records.iter().rev().take(self.shown_record_limit) {
                        let (kind, moment, description) = describe_record(record);
                        ui.strong(kind);
                        ui.label(moment);
                        ui.label(description);
                        ui.end_row();
                    }
                });
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::History(records) => self.records = records.clone(),
            Notice::BrainClosed => self.records.clear(),
            _ => {}
        }
    }
}

/// The kind, the moment and the content of a record, as three table cells.
fn describe_record(record: &HistoryRecord) -> (&'static str, String, String) {
    match record {
        HistoryRecord::TrainingStep(report) => (
            "train",
            format!("step {}", report.lifetime_step),
            describe_losses(&report.losses),
        ),
        HistoryRecord::Validation(report) => (
            "validate",
            format!("step {}", report.lifetime_step),
            describe_losses(&report.losses),
        ),
        HistoryRecord::LiveUpdate {
            unix_seconds,
            importance,
            report,
        } => (
            "learn",
            describe_age(*unix_seconds),
            format!(
                "importance {importance:.2} · {}",
                describe_live_update(report)
            ),
        ),
        HistoryRecord::Sleep(report) => (
            "sleep",
            describe_age(report.finished_unix_seconds),
            format!(
                "{} memories replayed, {} pruned{}",
                report.replayed_episode_count,
                report.pruned_memory_count,
                if report.was_discarded {
                    " (discarded)"
                } else {
                    ""
                },
            ),
        ),
    }
}
