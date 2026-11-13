//! The Brain tab: create, open and inspect a brain, and write sample text.
//!
//! Covers the command-line commands `presets`, `init`, `status`,
//! `sample-corpus`, and the global `--brain` and `--device` options.

use std::path::PathBuf;

use thingamaroo_core::brain::BrainSummary;
use thingamaroo_core::configuration::{ModelConfiguration, ModelPreset};
use thingamaroo_core::model::DeviceChoice;
use thingamaroo_core::presentation::{describe_age, describe_count};

use crate::form::{PathChooser, decimal_row, fact, hint, page, path_field, section, seed_row};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::Request;

const BYTES_PER_MEGABYTE: f64 = 1_000_000.0;

/// State of the Brain tab.
pub struct BrainView {
    directory: String,
    device_choice: DeviceChoice,
    preset: ModelPreset,
    seed: u64,
    custom_configuration_path: String,
    sample_path: String,
    sample_megabytes: f64,
    sample_seed: u64,
}

impl Default for BrainView {
    fn default() -> Self {
        Self {
            directory: "brain".to_string(),
            device_choice: DeviceChoice::Automatic,
            preset: ModelPreset::Prototype,
            seed: 42,
            custom_configuration_path: String::new(),
            sample_path: "sample.txt".to_string(),
            sample_megabytes: 2.0,
            sample_seed: 1,
        }
    }
}

impl View for BrainView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        page(ui, |ui| {
            show_getting_started(ui, shared);
            self.show_location(ui, shared);
            self.show_creation(ui, shared);
            self.show_sample_text(ui, shared);
            show_summary(ui, shared);
        });
    }
}

impl BrainView {
    fn show_location(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Location");
        hint(
            ui,
            "A brain is a folder holding weights, memories, settings and history.",
        );
        path_field(ui, &mut self.directory, PathChooser::Folder);
        ui.horizontal(|ui| {
            ui.label("Compute device");
            egui::ComboBox::from_id_salt("device_choice")
                .selected_text(self.device_choice.name())
                .show_ui(ui, |ui| {
                    for choice in DeviceChoice::available() {
                        ui.selectable_value(&mut self.device_choice, choice, choice.name());
                    }
                });
            hint(
                ui,
                "gpu and cuda appear when the application is built with those features.",
            );
        });
        ui.horizontal(|ui| {
            let is_idle = !shared.is_busy;
            if ui.add_enabled(is_idle, egui::Button::new("Open")).clicked() {
                shared.send(Request::OpenBrain {
                    directory: PathBuf::from(&self.directory),
                    device_choice: self.device_choice,
                });
            }
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("Close"))
                .clicked()
            {
                shared.send(Request::CloseBrain);
            }
        });
    }

    fn show_creation(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Create a new brain");
        egui::Grid::new("preset_table")
            .striped(true)
            .show(ui, |ui| {
                for preset in ModelPreset::ALL {
                    let counts = preset.configuration().parameter_counts();
                    ui.radio_value(&mut self.preset, preset, preset.name());
                    ui.label(format!(
                        "{} parameters",
                        describe_count(counts.slow_total())
                    ));
                    ui.label(format!("+{} fast", describe_count(counts.adapter)));
                    ui.label(preset.description());
                    ui.end_row();
                }
            });
        ui.horizontal(|ui| {
            ui.label("Custom architecture (JSON, optional)");
            path_field(ui, &mut self.custom_configuration_path, PathChooser::File);
        });
        egui::Grid::new("creation_options").show(ui, |ui| {
            seed_row(ui, "Initialisation seed", &mut self.seed);
        });
        if ui
            .add_enabled(
                !shared.is_busy,
                egui::Button::new("Create in the folder above"),
            )
            .clicked()
        {
            self.request_creation(shared);
        }
    }

    fn request_creation(&self, shared: &mut Shared) {
        let model_configuration = match self.read_model_configuration() {
            Ok(model_configuration) => model_configuration,
            Err(problem) => return shared.fail(problem),
        };
        shared.send(Request::CreateBrain {
            directory: PathBuf::from(&self.directory),
            model_configuration: Box::new(model_configuration),
            device_choice: self.device_choice,
            seed: self.seed,
        });
    }

    /// The preset's architecture, unless a custom architecture file was given.
    fn read_model_configuration(&self) -> Result<ModelConfiguration, String> {
        let custom_path = self.custom_configuration_path.trim();
        if custom_path.is_empty() {
            return Ok(self.preset.configuration());
        }
        let contents = std::fs::read_to_string(custom_path)
            .map_err(|error| format!("Could not read {custom_path}: {error}"))?;
        serde_json::from_str(&contents)
            .map_err(|error| format!("{custom_path} is not a valid architecture: {error}"))
    }

    fn show_sample_text(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Sample text");
        hint(
            ui,
            "Simple generated English for trying the whole loop without a dataset.",
        );
        path_field(ui, &mut self.sample_path, PathChooser::NewFile);
        egui::Grid::new("sample_options").show(ui, |ui| {
            decimal_row(
                ui,
                "Size (MB)",
                &mut self.sample_megabytes,
                0.01..=1_000.0,
                0.1,
            );
            seed_row(ui, "Seed", &mut self.sample_seed);
        });
        if ui
            .add_enabled(!shared.is_busy, egui::Button::new("Write sample text"))
            .clicked()
        {
            shared.send(Request::WriteSampleCorpus {
                path: PathBuf::from(&self.sample_path),
                target_bytes: (self.sample_megabytes * BYTES_PER_MEGABYTE) as u64,
                seed: self.sample_seed,
            });
        }
    }
}

/// The steps from nothing to a conversation, shown until a trained brain is open.
const GETTING_STARTED_STEPS: [&str; 5] = [
    "1. Below, choose a folder and a size (tiny trains on a processor in minutes) and create a brain.",
    "2. Write the sample text below, or use text files of your own.",
    "3. Train tab: add the text, press Start training, and watch the loss fall.",
    "4. Chat tab: talk to it. Correct a reply and ask again to see the correction stick.",
    "5. Sleep tab: consolidate what it learned. Snapshots tab: undo anything.",
];

fn show_getting_started(ui: &mut egui::Ui, shared: &Shared) {
    let is_ready_to_talk = shared
        .open_brain
        .as_ref()
        .is_some_and(|open_brain| open_brain.summary.is_pretrained());
    if is_ready_to_talk {
        return;
    }
    section(ui, "Getting started");
    for step in GETTING_STARTED_STEPS {
        ui.label(step);
    }
    hint(
        ui,
        "Every tab explains its own controls. The full guide is docs/gui.md in the project folder.",
    );
}

fn show_summary(ui: &mut egui::Ui, shared: &Shared) {
    section(ui, "Open brain");
    let Some(open_brain) = &shared.open_brain else {
        hint(ui, "No brain is open.");
        return;
    };
    egui::Grid::new("brain_summary")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            show_summary_rows(ui, &open_brain.summary);
        });
    if !open_brain.summary.is_pretrained() {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "This brain is untrained. Use the Train tab first.",
        );
    }
}

fn show_summary_rows(ui: &mut egui::Ui, summary: &BrainSummary) {
    fact(ui, "Folder", summary.root.display().to_string());
    fact(ui, "Device", summary.device_name.clone());
    show_model_facts(ui, summary);
    show_experience_facts(ui, summary);
    show_storage_facts(ui, summary);
}

/// How big the model is.
fn show_model_facts(ui: &mut egui::Ui, summary: &BrainSummary) {
    let model = &summary.model_configuration;
    let slow_weights = format!(
        "{} (encoder {}, predictor {}, decoder {})",
        describe_count(summary.slow_parameter_count()),
        describe_count(summary.encoder_parameter_count),
        describe_count(summary.predictor_parameter_count),
        describe_count(summary.decoder_parameter_count),
    );
    let meaning_vectors = format!(
        "{} dimensions, patches up to {} bytes",
        model.latent.width, model.patching.maximum_patch_bytes
    );
    let working_memory = format!(
        "{} patches, {} memory slots, {} thinking steps",
        model.predictor.maximum_context_patches,
        model.predictor.maximum_memory_slots,
        model.predictor.thinking_step_count,
    );
    fact(ui, "Slow weights", slow_weights);
    fact(
        ui,
        "Fast weights",
        describe_count(summary.adapter_parameter_count),
    );
    fact(ui, "Meaning vectors", meaning_vectors);
    fact(ui, "Working memory", working_memory);
}

/// What the brain has been through.
fn show_experience_facts(ui: &mut egui::Ui, summary: &BrainSummary) {
    let state = &summary.state;
    let held_out = state
        .training
        .validation_bits_per_byte
        .map_or("not measured".to_string(), |bits| {
            format!("{bits:.2} bits/byte")
        });
    let conversation = format!(
        "{} turns, {} live updates, {} turns since sleep",
        state.conversation.turn_count,
        state.conversation.live_update_count,
        state.conversation.turns_since_sleep,
    );
    let sleep = format!(
        "{} cycles, last {}",
        state.sleep.cycle_count,
        describe_age(state.sleep.last_sleep_unix_seconds)
    );
    let surprise_baseline = format!(
        "{:.2} bits/byte over {} inputs",
        state.surprise.mean_bits_per_byte, state.surprise.observation_count
    );
    fact(
        ui,
        "Pretraining",
        format!("{} steps, held-out {held_out}", state.training.step_count),
    );
    fact(ui, "Conversation", conversation);
    fact(ui, "Sleep", sleep);
    fact(ui, "Surprise baseline", surprise_baseline);
}

/// What is stored alongside the weights.
fn show_storage_facts(ui: &mut egui::Ui, summary: &BrainSummary) {
    let anchor_status = match summary.has_elastic_anchor {
        true => "recorded",
        false => "missing",
    };
    fact(ui, "Memories", summary.memory_count.to_string());
    fact(ui, "Snapshots", summary.snapshot_count.to_string());
    fact(
        ui,
        "Replay data",
        format!(
            "{} windows, elastic anchor {anchor_status}",
            summary.replay_window_count
        ),
    );
}
