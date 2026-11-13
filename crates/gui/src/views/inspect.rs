//! The Inspect tab: see how text is patched and what survives a trip through meaning space.
//!
//! Covers the command-line commands `patches` and `roundtrip`.

use thingamaroo_core::brain::RoundTripReport;
use thingamaroo_core::presentation::describe_on_one_line;

use crate::form::{hint, page, section};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

const TEXT_FIELD_ROWS: usize = 3;

/// State of the Inspect tab.
pub struct InspectView {
    text: String,
    patches: Vec<String>,
    round_trip: Option<RoundTripReport>,
}

impl Default for InspectView {
    fn default() -> Self {
        Self {
            text: "The cat fell to the ground.".to_string(),
            patches: Vec::new(),
            round_trip: None,
        }
    }
}

impl View for InspectView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        page(ui, |ui| {
            section(ui, "Text");
            ui.add(
                egui::TextEdit::multiline(&mut self.text)
                    .desired_rows(TEXT_FIELD_ROWS)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(shared.can_act(), egui::Button::new("Split into patches"))
                    .clicked()
                {
                    shared.send(Request::SplitPatches(self.text.clone()));
                }
                if ui
                    .add_enabled(shared.can_act(), egui::Button::new("Round trip"))
                    .clicked()
                {
                    shared.send(Request::RoundTrip(self.text.clone()));
                }
            });
            self.show_patches(ui);
            self.show_round_trip(ui);
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::Patches(patches) => self.patches = patches.clone(),
            Notice::RoundTrip(report) => self.round_trip = Some(report.clone()),
            Notice::BrainClosed => {
                self.patches.clear();
                self.round_trip = None;
            }
            _ => {}
        }
    }
}

impl InspectView {
    fn show_patches(&self, ui: &mut egui::Ui) {
        if self.patches.is_empty() {
            return;
        }
        section(ui, "Patches");
        hint(
            ui,
            "Each box is one chunk of bytes that becomes a single meaning vector.",
        );
        ui.horizontal_wrapped(|ui| {
            for patch in &self.patches {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(describe_on_one_line(patch)).monospace());
                });
            }
        });
        let byte_count: usize = self.patches.iter().map(String::len).sum();
        ui.label(format!(
            "{byte_count} bytes in {} patches",
            self.patches.len()
        ));
    }

    fn show_round_trip(&self, ui: &mut egui::Ui) {
        let Some(report) = &self.round_trip else {
            return;
        };
        section(ui, "Round trip");
        hint(
            ui,
            "Bytes to patches to meaning vectors and back to bytes, with the decoder picking its \
             most likely byte. An unfinished last word counts as exact when the decoder finishes it.",
        );
        ui.label(format!(
            "Byte accuracy: {:.1}%",
            report.byte_accuracy * 100.0
        ));
        egui::Grid::new("round_trip").striped(true).show(ui, |ui| {
            ui.strong("Original");
            ui.strong("Rebuilt");
            ui.strong("Exact");
            ui.end_row();
            for patch in &report.patches {
                ui.label(egui::RichText::new(describe_on_one_line(&patch.original)).monospace());
                ui.label(egui::RichText::new(describe_on_one_line(&patch.rebuilt)).monospace());
                ui.label(if patch.is_exact { "yes" } else { "NO" });
                ui.end_row();
            }
        });
    }
}
