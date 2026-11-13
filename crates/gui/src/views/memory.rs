//! The Memory tab: browse, search, add and delete episodic memories.
//!
//! Covers the command-line commands `memory list`, `memory search`,
//! `memory show`, `memory add`, `memory delete` and `memory clear`.

use thingamaroo_core::memory::{MemoryEntry, MemoryMatch, retention_score};
use thingamaroo_core::presentation::{describe_age, describe_on_one_line};

use crate::form::{decimal_row, fact, hint, page, section, table_header};
use crate::shared::Shared;
use crate::views::View;
use crate::worker::{Notice, Request};

/// Rows drawn at most; a longer list would only slow the window down.
const LISTED_ENTRY_LIMIT: usize = 300;
const PREVIEW_CHARACTER_COUNT: usize = 80;
const MEMORY_TABLE_TITLES: [&str; 8] = [
    "#",
    "Kind",
    "Importance",
    "Keeps",
    "Recalled",
    "Age",
    "Text",
    "",
];

/// State of the Memory tab.
pub struct MemoryView {
    entries: Vec<MemoryEntry>,
    matches: Vec<MemoryMatch>,
    search_query: String,
    search_limit: usize,
    note_text: String,
    note_importance: f64,
    selected_id: Option<u64>,
    is_confirming_clear: bool,
}

impl Default for MemoryView {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            matches: Vec::new(),
            search_query: String::new(),
            search_limit: 5,
            note_text: String::new(),
            note_importance: 0.9,
            selected_id: None,
            is_confirming_clear: false,
        }
    }
}

impl View for MemoryView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        page(ui, |ui| {
            self.show_search(ui, shared);
            self.show_note_form(ui, shared);
            self.show_selected(ui);
            self.show_entries(ui, shared);
        });
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::Memories(entries) => {
                self.entries = entries.clone();
                self.entries.sort_by(|left, right| {
                    retention_score(right).total_cmp(&retention_score(left))
                });
            }
            Notice::MemoryMatches(matches) => self.matches = matches.clone(),
            Notice::BrainClosed => *self = Self::default(),
            _ => {}
        }
    }
}

impl MemoryView {
    fn show_search(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Find by meaning");
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.search_query).hint_text("Text to look for"),
            );
            ui.label("Results");
            ui.add(egui::DragValue::new(&mut self.search_limit).range(1..=50));
            let has_query = !self.search_query.trim().is_empty();
            if ui
                .add_enabled(shared.can_act() && has_query, egui::Button::new("Search"))
                .clicked()
            {
                shared.send(Request::SearchMemories {
                    query: self.search_query.clone(),
                    limit: self.search_limit,
                });
            }
        });
        for found in &self.matches {
            ui.horizontal(|ui| {
                ui.monospace(format!(
                    "#{:<5} similarity {:.2}",
                    found.entry.id, found.similarity
                ));
                if ui.link(preview(&found.entry.text())).clicked() {
                    self.selected_id = Some(found.entry.id);
                }
            });
        }
    }

    fn show_note_form(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, "Add a fact by hand");
        ui.add(
            egui::TextEdit::singleline(&mut self.note_text)
                .hint_text("Mia likes green pears.")
                .desired_width(f32::INFINITY),
        );
        egui::Grid::new("note_options").show(ui, |ui| {
            decimal_row(ui, "Importance", &mut self.note_importance, 0.0..=1.0, 0.01);
        });
        let has_text = !self.note_text.trim().is_empty();
        if ui
            .add_enabled(shared.can_act() && has_text, egui::Button::new("Store"))
            .clicked()
        {
            shared.send(Request::AddNote {
                text: std::mem::take(&mut self.note_text),
                importance: self.note_importance,
            });
        }
    }

    fn show_selected(&mut self, ui: &mut egui::Ui) {
        let Some(entry) = self
            .selected_id
            .and_then(|selected_id| self.entries.iter().find(|entry| entry.id == selected_id))
        else {
            return;
        };
        section(ui, &format!("Memory #{}", entry.id));
        egui::Grid::new("selected_memory")
            .num_columns(2)
            .show(ui, |ui| {
                fact(ui, "Kind", entry.kind.name());
                fact(ui, "Importance", format!("{:.3}", entry.importance));
                fact(
                    ui,
                    "Retention score",
                    format!("{:.3}", retention_score(entry)),
                );
                fact(ui, "Created", describe_age(entry.created_unix_seconds));
                fact(
                    ui,
                    "Recalled",
                    format!(
                        "{} times, last {}",
                        entry.retrieval_count,
                        describe_age(entry.last_retrieved_unix_seconds)
                    ),
                );
                fact(ui, "Prompt", describe_on_one_line(&entry.prompt));
                fact(ui, "Reply", describe_on_one_line(&entry.reply));
            });
    }

    fn show_entries(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        section(ui, &format!("Stored memories ({})", self.entries.len()));
        self.show_list_controls(ui, shared);
        hint(
            ui,
            "Strongest first. Sleep decays importance and deletes memories whose retention score falls too low.",
        );

        let can_act = shared.can_act();
        let mut requested_action = None;
        egui::Grid::new("memory_table")
            .striped(true)
            .show(ui, |ui| {
                table_header(ui, &MEMORY_TABLE_TITLES);
                for entry in self.entries.iter().take(LISTED_ENTRY_LIMIT) {
                    let row_action = show_entry_row(ui, entry, can_act);
                    requested_action = row_action
                        .map(|action| (entry.id, action))
                        .or(requested_action);
                }
            });
        if self.entries.len() > LISTED_ENTRY_LIMIT {
            hint(ui, &format!("Showing the strongest {LISTED_ENTRY_LIMIT}."));
        }
        match requested_action {
            Some((entry_id, RowAction::Select)) => self.selected_id = Some(entry_id),
            Some((entry_id, RowAction::Delete)) => {
                shared.send(Request::DeleteMemories(vec![entry_id]))
            }
            None => {}
        }
    }

    fn show_list_controls(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("Refresh"))
                .clicked()
            {
                shared.send(Request::RefreshMemories);
            }
            ui.checkbox(
                &mut self.is_confirming_clear,
                "I want to delete every memory",
            );
            let can_clear = shared.can_act() && self.is_confirming_clear;
            if ui
                .add_enabled(can_clear, egui::Button::new("Delete all"))
                .clicked()
            {
                shared.send(Request::ClearMemories);
                self.is_confirming_clear = false;
            }
        });
    }
}

/// What the user asked for by clicking inside a table row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowAction {
    Select,
    Delete,
}

/// Draws one memory as a table row and reports what was clicked in it.
fn show_entry_row(ui: &mut egui::Ui, entry: &MemoryEntry, can_act: bool) -> Option<RowAction> {
    ui.monospace(entry.id.to_string());
    ui.label(entry.kind.name());
    ui.label(format!("{:.2}", entry.importance));
    ui.label(format!("{:.2}", retention_score(entry)));
    ui.label(format!("{}×", entry.retrieval_count));
    ui.label(describe_age(entry.created_unix_seconds));
    let was_selected = ui.link(preview(&entry.text())).clicked();
    let was_deleted = ui
        .add_enabled(can_act, egui::Button::new("Delete").small())
        .clicked();
    ui.end_row();

    if was_deleted {
        return Some(RowAction::Delete);
    }
    was_selected.then_some(RowAction::Select)
}

fn preview(text: &str) -> String {
    let one_line = describe_on_one_line(text);
    if one_line.chars().count() <= PREVIEW_CHARACTER_COUNT {
        return one_line;
    }
    let shortened: String = one_line.chars().take(PREVIEW_CHARACTER_COUNT).collect();
    format!("{shortened}…")
}
