//! The Snapshots tab: save weights, roll back, and reset the fast weights.
//!
//! Covers the command-line commands `snapshot create`, `snapshot list`,
//! `snapshot rollback`, `snapshot delete` and `reset-fast-weights`.

use thingamaroo_core::brain::SnapshotInformation;
use thingamaroo_core::presentation::describe_age;

use crate::form::{hint, page, section, table_header};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

const SNAPSHOT_TABLE_TITLES: [&str; 7] = ["#", "Taken", "Contains", "Age", "Label", "", ""];

/// State of the Snapshots tab.
pub struct SnapshotsView {
    snapshots: Vec<SnapshotInformation>,
    label: String,
    includes_slow_weights: bool,
}

impl Default for SnapshotsView {
    fn default() -> Self {
        Self {
            snapshots: Vec::new(),
            label: "manual".to_string(),
            includes_slow_weights: false,
        }
    }
}

impl View for SnapshotsView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        page(ui, |ui| {
            self.show_creation(ui, shared);
            self.show_table(ui, shared);
            show_fast_weight_reset(ui, shared);
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::Snapshots(snapshots) => self.snapshots = snapshots.clone(),
            Notice::BrainClosed => self.snapshots.clear(),
            _ => {}
        }
    }
}

impl SnapshotsView {
    fn show_creation(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Save the current weights");
        ui.horizontal(|ui| {
            ui.label("Label");
            ui.text_edit_singleline(&mut self.label);
        });
        ui.checkbox(
            &mut self.includes_slow_weights,
            "Also save the slow predictor (larger; needed to undo a sleep)",
        );
        if ui
            .add_enabled(shared.can_act(), egui::Button::new("Create snapshot"))
            .clicked()
        {
            shared.send(Request::CreateSnapshot {
                label: self.label.clone(),
                includes_slow_weights: self.includes_slow_weights,
            });
        }
    }

    fn show_table(&self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, &format!("Snapshots ({})", self.snapshots.len()));
        hint(
            ui,
            "The brain takes a snapshot on its own before every live update and every sleep, \
             keeping only the newest few.",
        );
        let can_act = shared.can_act();
        let mut requested = None;
        egui::Grid::new("snapshot_table")
            .striped(true)
            .show(ui, |ui| {
                table_header(ui, &SNAPSHOT_TABLE_TITLES);
                for information in self.snapshots.iter().rev() {
                    if let Some(request) = show_snapshot_row(ui, information, can_act) {
                        requested = Some(request);
                    }
                }
            });
        if let Some(request) = requested {
            shared.send(request);
        }
    }
}

/// Draws one snapshot as a table row and returns the request its buttons triggered.
fn show_snapshot_row(
    ui: &mut egui::Ui,
    information: &SnapshotInformation,
    can_act: bool,
) -> Option<Request> {
    ui.monospace(information.id.to_string());
    ui.label(match information.is_automatic {
        true => "automatically",
        false => "by hand",
    });
    ui.label(match information.includes_slow_weights {
        true => "fast + slow predictor",
        false => "fast weights",
    });
    ui.label(describe_age(information.created_unix_seconds));
    ui.label(&information.label);
    let is_rolling_back = ui
        .add_enabled(can_act, egui::Button::new("Roll back").small())
        .clicked();
    let is_deleting = ui
        .add_enabled(can_act, egui::Button::new("Delete").small())
        .clicked();
    ui.end_row();

    if is_rolling_back {
        return Some(Request::RollBack(information.id));
    }
    is_deleting.then_some(Request::DeleteSnapshot(information.id))
}

fn show_fast_weight_reset(ui: &mut egui::Ui, shared: &mut Shared) {
    section(ui, "Fast weights");
    hint(
        ui,
        "Discards everything the fast weights have learned since the last sleep. \
         Slow weights and memories are unchanged.",
    );
    if ui
        .add_enabled(shared.can_act(), egui::Button::new("Reset fast weights"))
        .clicked()
    {
        shared.send(Request::ResetFastWeights);
    }
}
