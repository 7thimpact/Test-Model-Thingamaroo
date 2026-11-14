//! The Sleep tab: consolidate memories into the slow weights and see what happened.
//!
//! Covers the command-line command `sleep` and the automatic sleep of `chat`.

use egui_plot::{Legend, Line, Plot, PlotPoints};
use thingamaroo_core::brain::SleepReport;
use thingamaroo_core::learning::ProgressEvent;
use thingamaroo_core::presentation::describe_age;

use crate::form::{fact, hint, page, section};
use crate::shared::{OpenBrain, Shared};
use crate::views::View;
use crate::worker::{Notice, Request};

const REPLAY_PLOT_HEIGHT: f32 = 200.0;

/// State of the Sleep tab.
#[derive(Default)]
pub struct SleepView {
    replay_losses: Vec<[f64; 2]>,
    elastic_penalties: Vec<[f64; 2]>,
    status_lines: Vec<String>,
    last_report: Option<SleepReport>,
}

impl View for SleepView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        page(ui, |ui| {
            section(ui, "Sleep");
            hint(
                ui,
                "Replays important memories mixed with old training data into the slow weights, \
                 then resets the fast weights and prunes weak memories. A sleep that damages old \
                 skills is discarded.",
            );
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("Sleep now"))
                .clicked()
            {
                self.replay_losses.clear();
                self.elastic_penalties.clear();
                self.status_lines.clear();
                shared.send(Request::Sleep);
            }
            if let Some(open_brain) = &shared.open_brain {
                show_schedule(ui, open_brain);
            }
            self.show_replay(ui);
            self.show_report(ui);
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::Progress(ProgressEvent::SleepStep(step)) => {
                self.replay_losses
                    .push([step.step as f64, step.replay_loss]);
                self.elastic_penalties
                    .push([step.step as f64, step.elastic_penalty]);
            }
            Notice::Progress(ProgressEvent::Status(status)) => {
                self.status_lines.push(status.clone())
            }
            Notice::SleepFinished(report) => self.last_report = Some(report.clone()),
            Notice::BrainClosed => *self = Self::default(),
            _ => {}
        }
    }
}

impl SleepView {
    fn show_replay(&self, ui: &mut egui::Ui) {
        if self.replay_losses.is_empty() {
            return;
        }
        section(ui, "Replay");
        Plot::new("replay_plot")
            .height(REPLAY_PLOT_HEIGHT)
            .legend(Legend::default())
            .x_axis_label("replay step")
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(
                    "replay loss",
                    PlotPoints::from(self.replay_losses.clone()),
                ));
                plot_ui.line(Line::new(
                    "elastic penalty",
                    PlotPoints::from(self.elastic_penalties.clone()),
                ));
            });
        for status in &self.status_lines {
            hint(ui, status);
        }
    }

    fn show_report(&self, ui: &mut egui::Ui) {
        let Some(report) = &self.last_report else {
            return;
        };
        section(ui, "Last sleep in this session");
        if report.was_discarded {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "The new weights were discarded: old skills degraded too much or the sleep was stopped.",
            );
        }
        egui::Grid::new("sleep_report")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                fact(
                    ui,
                    "Memories replayed",
                    report.replayed_episode_count.to_string(),
                );
                fact(
                    ui,
                    "Replay steps",
                    report.consolidation.step_count.to_string(),
                );
                fact(
                    ui,
                    "Loss on memories",
                    describe_change(report.episode_loss_before, report.episode_loss_after),
                );
                fact(
                    ui,
                    "Loss on old skills",
                    describe_change(report.old_loss_before, report.old_loss_after),
                );
                fact(
                    ui,
                    "Memories pruned",
                    report.pruned_memory_count.to_string(),
                );
                fact(
                    ui,
                    "Memories remaining",
                    report.remaining_memory_count.to_string(),
                );
                let snapshot = report
                    .snapshot_id
                    .map_or("none".to_string(), |snapshot_id| {
                        format!("#{snapshot_id} (roll back in Snapshots)")
                    });
                fact(ui, "Snapshot before sleep", snapshot);
            });
    }
}

fn show_schedule(ui: &mut egui::Ui, open_brain: &OpenBrain) {
    let schedule = &open_brain.settings.schedule;
    let state = &open_brain.summary.state;
    section(ui, "Schedule");
    egui::Grid::new("sleep_schedule")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            fact(
                ui,
                "Automatic sleep",
                if schedule.is_automatic { "on" } else { "off" },
            );
            fact(
                ui,
                "Turns since last sleep",
                format!(
                    "{} of {}",
                    state.conversation.turns_since_sleep, schedule.turn_interval
                ),
            );
            fact(
                ui,
                "Memories",
                format!(
                    "{} of {}",
                    open_brain.summary.memory_count, schedule.memory_count_limit
                ),
            );
            fact(
                ui,
                "Sleeps when idle for",
                format!("{} s", schedule.idle_seconds),
            );
            fact(
                ui,
                "Sleeps at least every",
                format!("{} h", schedule.hour_interval),
            );
            fact(
                ui,
                "Last sleep",
                format!(
                    "{} ({} cycles so far)",
                    describe_age(state.sleep.last_sleep_unix_seconds),
                    state.sleep.cycle_count
                ),
            );
        });
    hint(
        ui,
        "Automatic sleep happens during a learning conversation; change the limits in Settings.",
    );
}

fn describe_change(before: Option<f64>, after: Option<f64>) -> String {
    match (before, after) {
        (Some(before), Some(after)) => format!("{before:.3} -> {after:.3}"),
        _ => "not measured".to_string(),
    }
}
