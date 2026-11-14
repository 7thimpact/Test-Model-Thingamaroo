//! The tabs of the window; each one covers a group of command-line commands.

mod brain;
mod chat;
mod generate;
mod history;
mod inspect;
mod memory;
mod settings;
mod sleep;
mod snapshots;
mod training;

pub use brain::BrainView;
pub use chat::ChatView;
pub use generate::GenerateView;
pub use history::HistoryView;
pub use inspect::InspectView;
pub use memory::MemoryView;
pub use settings::SettingsView;
pub use sleep::SleepView;
pub use snapshots::SnapshotsView;
pub use training::TrainingView;

use crate::shared::Shared;
use crate::worker::Notice;

/// One tab of the window.
pub trait View {
    /// Draws the tab and queues any requests the user triggers.
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared);

    /// Lets the tab update itself from something the worker reported.
    fn receive(&mut self, _notice: &Notice) {}
}

/// Which tab is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewKind {
    Brain,
    Training,
    Chat,
    Generate,
    Inspect,
    Memory,
    Sleep,
    Snapshots,
    Settings,
    History,
}

impl ViewKind {
    /// Every tab, in the order they appear.
    pub const ALL: [ViewKind; 10] = [
        ViewKind::Brain,
        ViewKind::Training,
        ViewKind::Chat,
        ViewKind::Generate,
        ViewKind::Inspect,
        ViewKind::Memory,
        ViewKind::Sleep,
        ViewKind::Snapshots,
        ViewKind::Settings,
        ViewKind::History,
    ];

    /// The tab's title.
    pub fn title(self) -> &'static str {
        match self {
            ViewKind::Brain => "Brain",
            ViewKind::Training => "Train",
            ViewKind::Chat => "Chat",
            ViewKind::Generate => "Generate",
            ViewKind::Inspect => "Inspect",
            ViewKind::Memory => "Memory",
            ViewKind::Sleep => "Sleep",
            ViewKind::Snapshots => "Snapshots",
            ViewKind::Settings => "Settings",
            ViewKind::History => "History",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESKTOP_GUIDE: &str = include_str!("../../../../docs/gui.md");

    #[test]
    fn every_tab_has_a_section_in_the_desktop_guide() {
        for kind in ViewKind::ALL {
            let heading = format!("## {} tab", kind.title());
            assert!(
                DESKTOP_GUIDE.contains(&heading),
                "docs/gui.md has no section headed `{heading}`"
            );
        }
    }
}
