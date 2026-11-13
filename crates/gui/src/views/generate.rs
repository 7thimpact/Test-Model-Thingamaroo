//! The Generate tab: continue a prompt without memory or learning.
//!
//! Covers the command-line command `generate`.

use thingamaroo_core::configuration::GenerationSettings;
use thingamaroo_core::model::{Reply, StopReason, ThoughtTrace};
use thingamaroo_core::presentation::describe_on_one_line;

use crate::form::{count_row, decimal_row, hint, monospace_block, page, section};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

const PROMPT_FIELD_ROWS: usize = 3;

/// State of the Generate tab.
pub struct GenerateView {
    prompt: String,
    generation: GenerationSettings,
    is_seed_fixed: bool,
    seed: u64,
    is_tracing_thoughts: bool,
    is_awaiting_reply: bool,
    has_loaded_settings: bool,
    written: Vec<u8>,
    last_reply: Option<Reply>,
}

impl Default for GenerateView {
    fn default() -> Self {
        Self {
            prompt: "The ".to_string(),
            generation: GenerationSettings::default(),
            is_seed_fixed: false,
            seed: 1,
            is_tracing_thoughts: true,
            is_awaiting_reply: false,
            has_loaded_settings: false,
            written: Vec::new(),
            last_reply: None,
        }
    }
}

impl View for GenerateView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        self.adopt_brain_settings(shared);
        page(ui, |ui| {
            section(ui, "Prompt");
            ui.add(
                egui::TextEdit::multiline(&mut self.prompt)
                    .desired_rows(PROMPT_FIELD_ROWS)
                    .desired_width(f32::INFINITY),
            );
            self.show_options(ui);
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("Generate"))
                .clicked()
            {
                self.request_generation(shared);
            }
            self.show_output(ui);
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::WrittenBytes(bytes) if self.is_awaiting_reply => {
                self.written.extend_from_slice(bytes)
            }
            Notice::GenerationFinished(reply) => {
                self.is_awaiting_reply = false;
                self.written = reply.bytes.clone();
                self.last_reply = Some(reply.clone());
            }
            Notice::Failure(_) => self.is_awaiting_reply = false,
            Notice::BrainOpened { .. } | Notice::BrainClosed => self.has_loaded_settings = false,
            _ => {}
        }
    }
}

impl GenerateView {
    /// Starts from the open brain's generation settings the first time the tab is shown.
    fn adopt_brain_settings(&mut self, shared: &Shared) {
        let Some(open_brain) = &shared.open_brain else {
            return;
        };
        if self.has_loaded_settings {
            return;
        }
        self.generation = open_brain.settings.generation.clone();
        self.is_seed_fixed = self.generation.seed.is_some();
        self.seed = self.generation.seed.unwrap_or(self.seed);
        self.has_loaded_settings = true;
    }

    fn show_options(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("generation_options").show(ui, |ui| {
            decimal_row(
                ui,
                "Temperature",
                &mut self.generation.temperature,
                0.0..=3.0,
                0.01,
            );
            count_row(
                ui,
                "Candidates per byte",
                &mut self.generation.candidate_count,
            );
            count_row(
                ui,
                "Thinking steps",
                &mut self.generation.thinking_step_count,
            );
            count_row(
                ui,
                "Longest continuation (bytes)",
                &mut self.generation.maximum_reply_bytes,
            );
            ui.checkbox(&mut self.is_seed_fixed, "Fixed seed");
            ui.add_enabled(self.is_seed_fixed, egui::DragValue::new(&mut self.seed));
            ui.end_row();
        });
        ui.checkbox(
            &mut self.is_tracing_thoughts,
            "Show what each thinking stage would have written",
        );
    }

    fn request_generation(&mut self, shared: &mut Shared) {
        self.generation.seed = self.is_seed_fixed.then_some(self.seed);
        self.written.clear();
        self.last_reply = None;
        self.is_awaiting_reply = true;
        shared.send(Request::Generate {
            prompt: self.prompt.clone(),
            generation: self.generation.clone(),
            is_tracing_thoughts: self.is_tracing_thoughts,
        });
    }

    fn show_output(&self, ui: &mut egui::Ui) {
        section(ui, "Continuation");
        let continuation = String::from_utf8_lossy(&self.written);
        monospace_block(ui, &format!("{}{continuation}", self.prompt));
        let Some(reply) = &self.last_reply else {
            return;
        };
        if let Some(reason) = describe_stop_reason(reply.stop_reason) {
            hint(ui, reason);
        }
        show_thoughts(ui, &reply.thoughts);
    }
}

/// Explains why a reply ended, when the reason is worth mentioning.
pub fn describe_stop_reason(stop_reason: StopReason) -> Option<&'static str> {
    match stop_reason {
        StopReason::LineEnd | StopReason::NothingMoreToWrite => None,
        StopReason::LengthLimit => Some("Reached the length limit."),
        StopReason::Cancelled => Some("Stopped."),
    }
}

/// Shows the coarse-to-fine thinking behind each written patch.
pub fn show_thoughts(ui: &mut egui::Ui, thoughts: &[ThoughtTrace]) {
    if thoughts.is_empty() {
        return;
    }
    egui::CollapsingHeader::new(format!("Thinking behind {} patches", thoughts.len()))
        .id_salt(ui.next_auto_id())
        .show(ui, |ui| {
            hint(
                ui,
                "Each row: the rough first guess, then each refinement, then what was written. \
                 The number is how far that stage's meaning vector still was from the final one.",
            );
            egui::Grid::new(ui.next_auto_id())
                .striped(true)
                .show(ui, |ui| {
                    thoughts
                        .iter()
                        .for_each(|thought| show_thought_row(ui, thought))
                });
        });
}

/// One written patch: every thinking stage, then what was actually written.
fn show_thought_row(ui: &mut egui::Ui, thought: &ThoughtTrace) {
    for stage in &thought.stages {
        let stage_text = format!(
            "{} ({:.2})",
            describe_on_one_line(&stage.preview),
            stage.distance_to_final
        );
        ui.label(egui::RichText::new(stage_text).monospace().weak());
        ui.label(">");
    }
    let written = egui::RichText::new(describe_on_one_line(&thought.written));
    ui.label(written.monospace().strong());
    ui.end_row();
}
