//! The window: a row of tabs, the current tab, and a status line.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use thingamaroo_core::model::DeviceChoice;

use crate::shared::{OpenBrain, Shared};
use crate::views::{
    BrainView, ChatView, GenerateView, HistoryView, InspectView, MemoryView, SettingsView,
    SleepView, SnapshotsView, TrainingView, View, ViewKind,
};
use crate::worker::{Notice, Request, WorkerHandle};

/// How often the window asks the worker whether the sleep schedule is due.
const SCHEDULE_CHECK_INTERVAL: Duration = Duration::from_secs(15);

/// What to do as soon as the window opens.
#[derive(Debug, Clone, Default)]
pub struct StartupOptions {
    /// A brain to open straight away.
    pub brain_directory: Option<PathBuf>,
    /// The compute backend to open it on.
    pub device_choice: DeviceChoice,
}

/// The latest thing worth telling the user.
#[derive(Default)]
struct StatusLine {
    text: String,
    is_problem: bool,
}

/// Every tab's state.
#[derive(Default)]
struct Views {
    brain: BrainView,
    training: TrainingView,
    chat: ChatView,
    generate: GenerateView,
    inspect: InspectView,
    memory: MemoryView,
    sleep: SleepView,
    snapshots: SnapshotsView,
    settings: SettingsView,
    history: HistoryView,
}

impl Views {
    fn all(&mut self) -> [&mut dyn View; 10] {
        [
            &mut self.brain,
            &mut self.training,
            &mut self.chat,
            &mut self.generate,
            &mut self.inspect,
            &mut self.memory,
            &mut self.sleep,
            &mut self.snapshots,
            &mut self.settings,
            &mut self.history,
        ]
    }

    fn get(&mut self, kind: ViewKind) -> &mut dyn View {
        match kind {
            ViewKind::Brain => &mut self.brain,
            ViewKind::Training => &mut self.training,
            ViewKind::Chat => &mut self.chat,
            ViewKind::Generate => &mut self.generate,
            ViewKind::Inspect => &mut self.inspect,
            ViewKind::Memory => &mut self.memory,
            ViewKind::Sleep => &mut self.sleep,
            ViewKind::Snapshots => &mut self.snapshots,
            ViewKind::Settings => &mut self.settings,
            ViewKind::History => &mut self.history,
        }
    }
}

/// The desktop application.
pub struct Application {
    worker: WorkerHandle,
    shared: Shared,
    views: Views,
    selected_view: ViewKind,
    status: StatusLine,
    pending_request_count: usize,
    last_interaction: Instant,
    last_schedule_check: Instant,
}

impl Application {
    /// Starts the worker thread and opens the requested brain, if any.
    pub fn new(creation_context: &eframe::CreationContext<'_>, startup: StartupOptions) -> Self {
        let window = creation_context.egui_ctx.clone();
        let mut application = Self {
            worker: WorkerHandle::spawn(move || window.request_repaint()),
            shared: Shared::default(),
            views: Views::default(),
            selected_view: ViewKind::Brain,
            status: StatusLine::default(),
            pending_request_count: 0,
            last_interaction: Instant::now(),
            last_schedule_check: Instant::now(),
        };
        if let Some(directory) = startup.brain_directory {
            application.send(Request::OpenBrain {
                directory,
                device_choice: startup.device_choice,
            });
        }
        application
    }

    fn send(&mut self, request: Request) {
        self.pending_request_count += 1;
        self.shared.is_busy = true;
        self.worker.send(request);
    }

    fn receive(&mut self, notice: Notice) {
        match &notice {
            Notice::BrainOpened { summary, settings } => {
                self.shared.open_brain = Some(OpenBrain {
                    summary: (**summary).clone(),
                    settings: (**settings).clone(),
                });
                self.inform(format!("Opened {}", summary.root.display()));
            }
            Notice::BrainClosed => {
                self.shared.open_brain = None;
                self.inform("Brain closed".to_string());
            }
            Notice::Summary(summary) => {
                if let Some(open_brain) = &mut self.shared.open_brain {
                    open_brain.summary = (**summary).clone();
                }
            }
            Notice::Settings(settings) => {
                if let Some(open_brain) = &mut self.shared.open_brain {
                    open_brain.settings = (**settings).clone();
                }
            }
            Notice::Information(message) => self.inform(message.clone()),
            Notice::Failure(message) => self.report_problem(message.clone()),
            Notice::Ready => {
                self.pending_request_count = self.pending_request_count.saturating_sub(1);
                self.shared.is_busy = self.pending_request_count > 0;
            }
            _ => {}
        }
        for view in self.views.all() {
            view.receive(&notice);
        }
    }

    fn inform(&mut self, text: String) {
        self.status = StatusLine {
            text,
            is_problem: false,
        };
    }

    fn report_problem(&mut self, text: String) {
        self.status = StatusLine {
            text,
            is_problem: true,
        };
    }

    /// Lets the brain sleep on its own while a learning conversation sits idle.
    fn check_sleep_schedule(&mut self) {
        let is_check_due = self.last_schedule_check.elapsed() >= SCHEDULE_CHECK_INTERVAL;
        if !is_check_due || !self.shared.can_act() {
            return;
        }
        self.last_schedule_check = Instant::now();
        self.send(Request::SleepIfDue {
            idle: self.last_interaction.elapsed(),
        });
    }

    fn show_navigation(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for kind in ViewKind::ALL {
                ui.selectable_value(&mut self.selected_view, kind, kind.title());
            }
            if !self.shared.is_busy {
                return;
            }
            ui.separator();
            ui.spinner();
            ui.label("Working…");
            if ui.button("Stop").clicked() {
                self.worker.request_stop();
            }
        });
    }

    /// The latest message on the left, the open brain on the right.
    fn show_status(&self, ui: &mut egui::Ui) {
        let message_color = match self.status.is_problem {
            true => ui.visuals().error_fg_color,
            false => ui.visuals().text_color(),
        };
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new(self.describe_open_brain()).weak());
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let message = egui::RichText::new(&self.status.text).color(message_color);
                    ui.add(egui::Label::new(message).truncate());
                });
            });
        });
    }

    /// The open brain's folder name and device, short enough for the status bar.
    fn describe_open_brain(&self) -> String {
        let Some(open_brain) = &self.shared.open_brain else {
            return "no brain open".to_string();
        };
        let summary = &open_brain.summary;
        let folder_name = summary.root.file_name().map_or_else(
            || summary.root.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        format!("{folder_name} · {}", summary.device_name)
    }

    /// Sends what the tabs queued during this frame and surfaces their problems.
    fn flush_shared(&mut self) {
        for problem in self.shared.take_failures() {
            self.report_problem(problem);
        }
        for request in self.shared.take_outbox() {
            self.send(request);
        }
    }
}

impl eframe::App for Application {
    fn logic(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        for notice in self.worker.take_notices() {
            self.receive(notice);
        }
        self.check_sleep_schedule();
        // Wake up again so the schedule is checked even when nobody touches the window.
        context.request_repaint_after(SCHEDULE_CHECK_INTERVAL);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let has_user_activity = ui.input(|input| !input.events.is_empty());
        if has_user_activity {
            self.last_interaction = Instant::now();
        }

        egui::Panel::top("navigation").show(ui, |ui| self.show_navigation(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.show_status(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            self.views
                .get(self.selected_view)
                .show(ui, &mut self.shared);
        });
        self.flush_shared();
    }
}
