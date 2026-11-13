//! The Train tab: pretrain on text files, watch the losses, and measure held-out text.
//!
//! Covers the command-line commands `train` and `evaluate`.

use std::path::PathBuf;

use egui_plot::{Legend, Line, Plot, PlotPoints};
use thingamaroo_core::brain::{EvaluationReport, HistoryRecord, PretrainingSummary};
use thingamaroo_core::configuration::TrainingOptions;
use thingamaroo_core::learning::{ProgressEvent, TrainingStepReport, ValidationReport};
use thingamaroo_core::presentation::{describe_count, describe_losses};

use crate::form::{count_row, decimal_row, fact, hint, monospace_block, page, section, seed_row};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

const LOSS_PLOT_HEIGHT: f32 = 240.0;
const CORPUS_FIELD_ROWS: usize = 3;

/// Loss curves gathered from history and from the run in progress.
#[derive(Default)]
struct LossCurves {
    total: Vec<[f64; 2]>,
    reconstruction: Vec<[f64; 2]>,
    generation: Vec<[f64; 2]>,
    latent: Vec<[f64; 2]>,
    held_out_generation: Vec<[f64; 2]>,
}

impl LossCurves {
    fn record_training_step(&mut self, report: &TrainingStepReport) {
        let step = report.lifetime_step as f64;
        self.total.push([step, report.losses.total]);
        self.reconstruction
            .push([step, report.losses.reconstruction]);
        self.generation.push([step, report.losses.generation]);
        self.latent.push([step, report.losses.latent]);
    }

    fn record_validation(&mut self, report: &ValidationReport) {
        self.held_out_generation
            .push([report.lifetime_step as f64, report.losses.generation]);
    }

    fn from_history(records: &[HistoryRecord]) -> Self {
        let mut curves = Self::default();
        for record in records {
            match record {
                HistoryRecord::TrainingStep(report) => curves.record_training_step(report),
                HistoryRecord::Validation(report) => curves.record_validation(report),
                _ => {}
            }
        }
        curves
    }

    fn named_series(&self) -> [(&'static str, &Vec<[f64; 2]>); 5] {
        [
            ("total", &self.total),
            ("rebuild", &self.reconstruction),
            ("write", &self.generation),
            ("latent", &self.latent),
            ("write (held-out)", &self.held_out_generation),
        ]
    }
}

/// State of the Train tab.
pub struct TrainingView {
    options: TrainingOptions,
    corpus_paths_text: String,
    curves: LossCurves,
    last_step: Option<TrainingStepReport>,
    last_validation: Option<ValidationReport>,
    last_summary: Option<PretrainingSummary>,
    status_lines: Vec<String>,
    evaluation_batch_count: usize,
    evaluation_sequence_count: usize,
    evaluation_seed: u64,
    last_evaluation: Option<EvaluationReport>,
}

impl Default for TrainingView {
    fn default() -> Self {
        Self {
            options: TrainingOptions::default(),
            corpus_paths_text: String::new(),
            curves: LossCurves::default(),
            last_step: None,
            last_validation: None,
            last_summary: None,
            status_lines: Vec::new(),
            evaluation_batch_count: 8,
            evaluation_sequence_count: 8,
            evaluation_seed: 7,
            last_evaluation: None,
        }
    }
}

impl View for TrainingView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        page(ui, |ui| {
            self.show_corpus(ui);
            self.show_options(ui);
            self.show_controls(ui, shared);
            self.show_progress(ui);
            self.show_evaluation(ui, shared);
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::Progress(ProgressEvent::TrainingStep(report)) => {
                self.curves.record_training_step(report);
                self.last_step = Some(*report);
            }
            Notice::Progress(ProgressEvent::Validation(report)) => {
                self.curves.record_validation(report);
                self.last_validation = Some(report.clone());
            }
            Notice::Progress(ProgressEvent::Status(status)) => {
                self.status_lines.push(status.clone())
            }
            Notice::Progress(ProgressEvent::CheckpointSaved { lifetime_step }) => {
                self.status_lines
                    .push(format!("Checkpoint saved at step {lifetime_step}"));
            }
            Notice::PretrainingFinished(summary) => self.last_summary = Some(summary.clone()),
            Notice::Evaluation(report) => self.last_evaluation = Some(report.clone()),
            Notice::History(records) => self.curves = LossCurves::from_history(records),
            Notice::BrainClosed => *self = Self::default(),
            _ => {}
        }
    }
}

impl TrainingView {
    fn corpus_paths(&self) -> Vec<PathBuf> {
        self.corpus_paths_text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(PathBuf::from)
            .collect()
    }

    fn show_corpus(&mut self, ui: &mut egui::Ui) {
        section(ui, "Training text");
        hint(
            ui,
            "One path per line: text files, or folders searched for text files.",
        );
        ui.add(
            egui::TextEdit::multiline(&mut self.corpus_paths_text)
                .desired_rows(CORPUS_FIELD_ROWS)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            if ui.button("Add files…").clicked() {
                self.append_paths(rfd::FileDialog::new().pick_files().unwrap_or_default());
            }
            if ui.button("Add folder…").clicked() {
                self.append_paths(rfd::FileDialog::new().pick_folder().into_iter().collect());
            }
        });
    }

    fn append_paths(&mut self, paths: Vec<PathBuf>) {
        for path in paths {
            if !self.corpus_paths_text.is_empty() && !self.corpus_paths_text.ends_with('\n') {
                self.corpus_paths_text.push('\n');
            }
            self.corpus_paths_text.push_str(&path.display().to_string());
        }
    }

    fn show_options(&mut self, ui: &mut egui::Ui) {
        section(ui, "Run options");
        let options = &mut self.options;
        ui.horizontal_top(|ui| {
            egui::Grid::new("training_schedule").show(ui, |ui| show_schedule_options(ui, options));
            ui.add_space(24.0);
            egui::Grid::new("training_reporting")
                .show(ui, |ui| show_reporting_options(ui, options));
        });
    }

    fn show_controls(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        ui.add_space(6.0);
        if ui
            .add_enabled(shared.can_act(), egui::Button::new("Start training"))
            .clicked()
        {
            self.options.data.corpus_paths = self.corpus_paths();
            self.status_lines.clear();
            self.last_summary = None;
            shared.send(Request::Pretrain(self.options.clone()));
        }
        hint(
            ui,
            "Use Stop in the top bar to end a run early; progress is saved.",
        );
    }

    fn show_progress(&mut self, ui: &mut egui::Ui) {
        section(ui, "Progress");
        if let Some(step) = &self.last_step {
            let fraction = step.run_step as f32 / step.run_step_count.max(1) as f32;
            ui.add(egui::ProgressBar::new(fraction).text(format!(
                "step {} of {} · {} patches/s · learning rate {:.1e}",
                step.run_step,
                step.run_step_count,
                describe_count(step.patches_per_second as usize),
                step.learning_rate
            )));
            ui.label(describe_losses(&step.losses));
        }
        self.show_loss_plot(ui);
        if let Some(validation) = &self.last_validation {
            ui.label(format!("Held-out: {}", describe_losses(&validation.losses)));
            monospace_block(ui, &validation.sample);
        }
        if let Some(summary) = &self.last_summary {
            let outcome = if summary.was_cancelled {
                "Stopped early"
            } else {
                "Finished"
            };
            ui.label(format!(
                "{outcome} after {} steps in {:.0} s; weights saved.",
                summary.step_count, summary.elapsed_seconds
            ));
        }
        for status in &self.status_lines {
            hint(ui, status);
        }
    }

    fn show_loss_plot(&self, ui: &mut egui::Ui) {
        Plot::new("loss_plot")
            .height(LOSS_PLOT_HEIGHT)
            .legend(Legend::default())
            .x_axis_label("pretraining step")
            .y_axis_label("loss (nats)")
            .show(ui, |plot_ui| {
                for (name, points) in self.curves.named_series() {
                    plot_ui.line(Line::new(name, PlotPoints::from(points.clone())));
                }
            });
    }

    fn show_evaluation(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Measure on text");
        hint(
            ui,
            "Measures the slow weights on the paths above without changing anything.",
        );
        egui::Grid::new("evaluation_options").show(ui, |ui| {
            count_row(ui, "Batches", &mut self.evaluation_batch_count);
            count_row(
                ui,
                "Sequences per batch",
                &mut self.evaluation_sequence_count,
            );
            seed_row(ui, "Seed", &mut self.evaluation_seed);
        });
        if ui
            .add_enabled(shared.can_act(), egui::Button::new("Measure"))
            .clicked()
        {
            shared.send(Request::Evaluate {
                corpus_paths: self.corpus_paths(),
                batch_count: self.evaluation_batch_count,
                batch_sequence_count: self.evaluation_sequence_count,
                seed: self.evaluation_seed,
            });
        }
        let Some(report) = &self.last_evaluation else {
            return;
        };
        egui::Grid::new("evaluation_report")
            .num_columns(2)
            .show(ui, |ui| {
                fact(ui, "Batches measured", report.batch_count.to_string());
                fact(ui, "Losses", describe_losses(&report.losses));
                fact(
                    ui,
                    "Rebuild accuracy",
                    format!("{:.1}% of symbols", report.reconstruction_accuracy * 100.0),
                );
                fact(
                    ui,
                    "Latent spread",
                    format!(
                        "{:.3} (near zero would mean collapse)",
                        report.latent_deviation
                    ),
                );
            });
    }
}

/// How long to train, how much per step, and how fast.
fn show_schedule_options(ui: &mut egui::Ui, options: &mut TrainingOptions) {
    count_row(ui, "Steps", &mut options.schedule.step_count);
    count_row(
        ui,
        "Sequences per batch",
        &mut options.data.batch_sequence_count,
    );
    decimal_row(
        ui,
        "Peak learning rate",
        &mut options.schedule.peak_learning_rate,
        1e-6..=1.0,
        1e-5,
    );
    count_row(ui, "Warm-up steps", &mut options.schedule.warmup_step_count);
    decimal_row(
        ui,
        "Held-out fraction",
        &mut options.data.validation_fraction,
        0.0..=0.49,
        0.005,
    );
    seed_row(ui, "Seed", &mut options.data.seed);
}

/// How often progress is shown, measured and saved.
fn show_reporting_options(ui: &mut egui::Ui, options: &mut TrainingOptions) {
    let reporting = &mut options.reporting;
    count_row(
        ui,
        "Report every (steps)",
        &mut reporting.report_interval_steps,
    );
    count_row(
        ui,
        "Validate every (steps)",
        &mut reporting.validation_interval_steps,
    );
    count_row(
        ui,
        "Checkpoint every (steps)",
        &mut reporting.checkpoint_interval_steps,
    );
    count_row(
        ui,
        "Replay windows kept",
        &mut options.data.replay_window_count,
    );
    ui.label("Sample prompt");
    ui.text_edit_singleline(&mut reporting.sample_prompt);
    ui.end_row();
}
