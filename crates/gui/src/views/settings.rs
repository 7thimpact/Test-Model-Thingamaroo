//! The Settings tab: view and change every adjustable behaviour.
//!
//! Covers the command-line commands `settings show`, `settings set` and
//! `settings reset`. The list is built from the settings themselves, so a new
//! setting in the core library appears here without any change to this file.

use thingamaroo_core::configuration::{RuntimeSettings, describe_setting};

use crate::form::{hint, page, section};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

const PATH_SEPARATOR: char = '.';
const VALUE_FIELD_WIDTH: f32 = 180.0;
const TRUE_TEXT: &str = "true";
const FALSE_TEXT: &str = "false";

/// One setting being edited, as JSON text.
struct SettingDraft {
    path: String,
    description: &'static str,
    saved_text: String,
    text: String,
}

impl SettingDraft {
    fn is_changed(&self) -> bool {
        self.text.trim() != self.saved_text
    }

    fn is_switch(&self) -> bool {
        self.saved_text == TRUE_TEXT || self.saved_text == FALSE_TEXT
    }

    fn group_name(&self) -> &str {
        self.path.split(PATH_SEPARATOR).next().unwrap_or_default()
    }

    fn field_name(&self) -> &str {
        self.path
            .split_once(PATH_SEPARATOR)
            .map_or(self.path.as_str(), |(_, field_name)| field_name)
    }
}

/// State of the Settings tab.
#[derive(Default)]
pub struct SettingsView {
    drafts: Vec<SettingDraft>,
    is_loaded: bool,
}

impl View for SettingsView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        let Some(saved_settings) = shared
            .open_brain
            .as_ref()
            .map(|open_brain| open_brain.settings.clone())
        else {
            hint(ui, "Open a brain to see its settings.");
            return;
        };
        if !self.is_loaded {
            self.load(&saved_settings);
        }
        self.show_controls(ui, shared, &saved_settings);
        page(ui, |ui| self.show_groups(ui));
    }

    fn receive(&mut self, notice: &Notice) {
        let is_stale = matches!(
            notice,
            Notice::Settings(_) | Notice::BrainOpened { .. } | Notice::BrainClosed
        );
        if is_stale {
            self.is_loaded = false;
        }
    }
}

impl SettingsView {
    fn load(&mut self, settings: &RuntimeSettings) {
        self.drafts = settings
            .entries()
            .into_iter()
            .map(|entry| {
                let saved_text = entry.value.to_string();
                SettingDraft {
                    path: entry.path,
                    description: entry.description,
                    text: saved_text.clone(),
                    saved_text,
                }
            })
            .collect();
        self.is_loaded = true;
    }

    fn show_controls(
        &mut self,
        ui: &mut egui::Ui,
        shared: &mut Shared,
        saved_settings: &RuntimeSettings,
    ) {
        let changed_count = self
            .drafts
            .iter()
            .filter(|draft| draft.is_changed())
            .count();
        ui.horizontal(|ui| {
            let can_save = shared.can_act() && changed_count > 0;
            if ui
                .add_enabled(
                    can_save,
                    egui::Button::new(format!("Save {changed_count} change(s)")),
                )
                .clicked()
            {
                self.request_save(shared, saved_settings);
            }
            if ui
                .add_enabled(changed_count > 0, egui::Button::new("Revert"))
                .clicked()
            {
                self.is_loaded = false;
            }
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("Restore defaults"))
                .clicked()
            {
                shared.send(Request::ReplaceSettings(Box::default()));
            }
        });
        hint(
            ui,
            "Values are written as JSON: numbers, true/false, \"text\" in quotes, [\"a\", \"list\"], or null for none.",
        );
    }

    /// Applies every changed draft on top of the saved settings and sends the result.
    fn request_save(&self, shared: &mut Shared, saved_settings: &RuntimeSettings) {
        let mut edited_settings = saved_settings.clone();
        for draft in self.drafts.iter().filter(|draft| draft.is_changed()) {
            edited_settings = match edited_settings.with_entry_text(&draft.path, draft.text.trim())
            {
                Ok(settings) => settings,
                Err(error) => return shared.fail(error.to_string()),
            };
        }
        shared.send(Request::ReplaceSettings(Box::new(edited_settings)));
    }

    fn show_groups(&mut self, ui: &mut egui::Ui) {
        let mut group_names: Vec<String> = self
            .drafts
            .iter()
            .map(|draft| draft.group_name().to_string())
            .collect();
        group_names.dedup();
        for group_name in group_names {
            section(ui, &group_name);
            hint(ui, describe_setting(&group_name));
            egui::Grid::new(("settings_group", &group_name))
                .num_columns(3)
                .striped(true)
                .show(ui, |ui| self.show_group_rows(ui, &group_name));
        }
    }

    fn show_group_rows(&mut self, ui: &mut egui::Ui, group_name: &str) {
        let group_drafts = self
            .drafts
            .iter_mut()
            .filter(|draft| draft.group_name() == group_name);
        for draft in group_drafts {
            show_draft(ui, draft);
        }
    }
}

/// One setting as a table row: its name, an editor for its value, and what it does.
fn show_draft(ui: &mut egui::Ui, draft: &mut SettingDraft) {
    let name = egui::RichText::new(draft.field_name()).monospace();
    let name = match draft.is_changed() {
        true => name.strong(),
        false => name,
    };
    ui.label(name).on_hover_text(draft.description);
    show_value_editor(ui, draft);
    ui.label(egui::RichText::new(draft.description).weak());
    ui.end_row();
}

/// A switch for true/false settings, a text field holding JSON for everything else.
fn show_value_editor(ui: &mut egui::Ui, draft: &mut SettingDraft) {
    if !draft.is_switch() {
        // A fixed size keeps the grid from squeezing the field to make room for the description.
        let field_size = [VALUE_FIELD_WIDTH, ui.spacing().interact_size.y];
        ui.add_sized(field_size, egui::TextEdit::singleline(&mut draft.text));
        return;
    }
    let mut is_on = draft.text.trim() == TRUE_TEXT;
    ui.checkbox(&mut is_on, "");
    draft.text = match is_on {
        true => TRUE_TEXT.to_string(),
        false => FALSE_TEXT.to_string(),
    };
}
