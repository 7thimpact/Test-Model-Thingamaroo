//! Small building blocks for the forms and tables in every tab.

use std::path::PathBuf;

/// Width given to path fields so long directories stay readable.
const PATH_FIELD_WIDTH: f32 = 420.0;

/// A vertically scrolling page that fills the tab, whatever its content's width.
pub fn page(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, add_contents);
}

/// A section title with some air above it.
pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.heading(title);
    ui.add_space(2.0);
}

/// A dimmed explanatory line.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).weak());
}

/// One row of a two-column grid: a label and a value.
pub fn fact(ui: &mut egui::Ui, label: &str, value: impl Into<String>) {
    ui.label(egui::RichText::new(label).weak());
    ui.label(value.into());
    ui.end_row();
}

/// One row of a two-column grid: a label and an editable whole number.
pub fn count_row(ui: &mut egui::Ui, label: &str, count: &mut usize) {
    ui.label(label);
    ui.add(egui::DragValue::new(count).speed(1.0));
    ui.end_row();
}

/// One row of a two-column grid: a label and an editable seed.
pub fn seed_row(ui: &mut egui::Ui, label: &str, seed: &mut u64) {
    ui.label(label);
    ui.add(egui::DragValue::new(seed).speed(1.0));
    ui.end_row();
}

/// One row of a two-column grid: a label and an editable decimal number.
pub fn decimal_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    speed: f64,
) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).speed(speed).range(range));
    ui.end_row();
}

/// A text field for a path with a button that opens a file or folder chooser.
pub fn path_field(ui: &mut egui::Ui, path: &mut String, chooser: PathChooser) {
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(path).desired_width(PATH_FIELD_WIDTH));
        if !ui.button("Browse…").clicked() {
            return;
        }
        if let Some(chosen_path) = chooser.choose() {
            *path = chosen_path.display().to_string();
        }
    });
}

/// Which kind of chooser a path field opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathChooser {
    /// Pick an existing folder.
    Folder,
    /// Pick an existing file.
    File,
    /// Pick where to save a new file.
    NewFile,
}

impl PathChooser {
    fn choose(self) -> Option<PathBuf> {
        let dialog = rfd::FileDialog::new();
        match self {
            PathChooser::Folder => dialog.pick_folder(),
            PathChooser::File => dialog.pick_file(),
            PathChooser::NewFile => dialog.save_file(),
        }
    }
}

/// The title row of a grid used as a table.
pub fn table_header(ui: &mut egui::Ui, titles: &[&str]) {
    for title in titles {
        ui.strong(*title);
    }
    ui.end_row();
}

/// Shows text in a monospace font inside a frame, wrapping long lines.
pub fn monospace_block(ui: &mut egui::Ui, text: &str) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(egui::RichText::new(text).monospace());
    });
}
