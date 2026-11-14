//! The Chat tab: talk to the brain while it remembers and learns.
//!
//! Covers the command-line command `chat` with its `--frozen`, `--no-save` and
//! `--thoughts` options and its `/correct`, `/good`, `/bad` and `/forget` commands.

use thingamaroo_core::brain::{ChatOptions, ChatTurnReport};
use thingamaroo_core::model::ThoughtTrace;
use thingamaroo_core::presentation::{describe_live_update, describe_on_one_line};

use crate::form::hint;
use crate::shared::Shared;
use crate::views::View;
use crate::views::generate::{describe_stop_reason, show_thoughts};
use crate::worker::{FeedbackKind, Notice, Request};

const INPUT_PANEL_HEIGHT: f32 = 96.0;
const MESSAGE_PREVIEW_CHARACTER_COUNT: usize = 60;

/// Who a line of the conversation came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Speaker {
    User,
    Brain,
    /// A note from the application itself, such as "fast weights updated".
    Application,
}

/// One line of the conversation as shown on screen.
struct Message {
    speaker: Speaker,
    bytes: Vec<u8>,
    details: Vec<String>,
    thoughts: Vec<ThoughtTrace>,
}

impl Message {
    fn new(speaker: Speaker, text: &str) -> Self {
        Self {
            speaker,
            bytes: text.as_bytes().to_vec(),
            details: Vec::new(),
            thoughts: Vec::new(),
        }
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).trim_end().to_string()
    }
}

/// State of the Chat tab.
#[derive(Default)]
pub struct ChatView {
    options: ChatOptions,
    is_session_started: bool,
    messages: Vec<Message>,
    input: String,
    correction: String,
    is_awaiting_reply: bool,
    has_reply_to_rate: bool,
}

impl View for ChatView {
    fn show(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        self.show_session_options(ui, shared);
        ui.separator();
        egui::Panel::bottom("chat_input")
            .exact_size(INPUT_PANEL_HEIGHT)
            .show(ui, |ui| self.show_input(ui, shared));
        egui::CentralPanel::default().show(ui, |ui| self.show_messages(ui));
    }

    fn receive(&mut self, notice: &Notice) {
        match notice {
            Notice::ChatStarted(options) => self.begin_session(*options),
            Notice::WrittenBytes(bytes) if self.is_awaiting_reply => self.append_to_reply(bytes),
            Notice::ChatTurnFinished(turn) => self.finish_turn(turn),
            Notice::Feedback {
                kind,
                report,
                corrected_reply,
            } => {
                let update = report.live_update.as_ref().map(describe_live_update);
                self.note_feedback(*kind, update, corrected_reply.as_deref());
            }
            Notice::Failure(_) => self.is_awaiting_reply = false,
            Notice::BrainOpened { .. } | Notice::BrainClosed | Notice::PretrainingFinished(_) => {
                *self = Self {
                    options: self.options,
                    ..Self::default()
                };
            }
            _ => {}
        }
    }
}

impl ChatView {
    fn show_session_options(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.options.is_frozen, "Frozen")
                .on_hover_text("Read-only conversation: no weight updates, no memory writes.");
            ui.checkbox(&mut self.options.is_saving, "Save to disk")
                .on_hover_text("When off, what is learned lasts only for this session.");
            ui.checkbox(&mut self.options.is_tracing_thoughts, "Show thinking")
                .on_hover_text("Record what each thinking stage would have written.");
            if ui
                .add_enabled(shared.can_act(), egui::Button::new("New conversation"))
                .clicked()
            {
                shared.send(Request::StartChat(self.options));
            }
        });
        hint(
            ui,
            "Option changes apply when a new conversation starts. A new conversation empties the working memory; stored memories and weights stay.",
        );
    }

    fn show_messages(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.messages.is_empty() {
                    hint(
                        ui,
                        "Type a message below. Surprising inputs and corrections are remembered.",
                    );
                }
                for (message_index, message) in self.messages.iter().enumerate() {
                    show_message(ui, message_index, message);
                }
            });
    }

    fn show_input(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let input_field = ui.add(
                egui::TextEdit::singleline(&mut self.input)
                    .hint_text("Message")
                    .desired_width(ui.available_width() - 70.0),
            );
            let was_submitted =
                input_field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            let was_clicked = ui
                .add_enabled(shared.can_act(), egui::Button::new("Send"))
                .clicked();
            if (was_submitted || was_clicked) && shared.can_act() {
                self.send_message(shared);
                input_field.request_focus();
            }
        });
        self.show_feedback_controls(ui, shared);
    }

    fn show_feedback_controls(&mut self, ui: &mut egui::Ui, shared: &mut Shared) {
        let can_rate = shared.can_act() && self.has_reply_to_rate;
        ui.horizontal(|ui| {
            ui.label("Last reply:");
            if ui
                .add_enabled(can_rate, egui::Button::new("Good"))
                .clicked()
            {
                shared.send(Request::PraiseLastReply);
            }
            if ui.add_enabled(can_rate, egui::Button::new("Bad")).clicked() {
                shared.send(Request::RejectLastReply);
            }
            ui.add(
                egui::TextEdit::singleline(&mut self.correction)
                    .hint_text("The right answer")
                    .desired_width(ui.available_width() - 80.0),
            );
            let has_correction = !self.correction.trim().is_empty();
            if ui
                .add_enabled(can_rate && has_correction, egui::Button::new("Correct"))
                .clicked()
            {
                shared.send(Request::CorrectLastReply(std::mem::take(
                    &mut self.correction,
                )));
            }
        });
    }

    fn send_message(&mut self, shared: &mut Shared) {
        let user_text = self.input.trim().to_string();
        if user_text.is_empty() {
            return;
        }
        if !self.is_session_started {
            shared.send(Request::StartChat(self.options));
        }
        shared.send(Request::ChatTurn(user_text.clone()));

        self.messages.push(Message::new(Speaker::User, &user_text));
        self.messages.push(Message::new(Speaker::Brain, ""));
        self.is_awaiting_reply = true;
        self.has_reply_to_rate = false;
        self.input.clear();
    }

    /// Marks the conversation as running; an explicitly requested new one starts empty.
    fn begin_session(&mut self, options: ChatOptions) {
        self.is_session_started = true;
        self.options = options;
        // A session started by sending the first message already shows that message.
        if self.is_awaiting_reply {
            return;
        }
        self.messages.clear();
        self.has_reply_to_rate = false;
    }

    fn append_to_reply(&mut self, bytes: &[u8]) {
        if let Some(reply) = self.last_brain_message() {
            reply.bytes.extend_from_slice(bytes);
        }
    }

    fn last_brain_message(&mut self) -> Option<&mut Message> {
        self.messages
            .iter_mut()
            .rev()
            .find(|message| message.speaker == Speaker::Brain)
    }

    fn finish_turn(&mut self, turn: &ChatTurnReport) {
        self.is_awaiting_reply = false;
        self.has_reply_to_rate = true;
        let details = describe_turn(turn);
        let Some(reply) = self.last_brain_message() else {
            return;
        };
        reply.bytes = turn.reply.bytes.clone();
        reply.details = details;
        reply.thoughts = turn.reply.thoughts.clone();
    }

    fn note_feedback(
        &mut self,
        kind: FeedbackKind,
        update: Option<String>,
        corrected_reply: Option<&str>,
    ) {
        let outcome = update.unwrap_or_else(|| "no weight update".to_string());
        let note = match kind {
            FeedbackKind::Correction => format!("Corrected: {outcome}."),
            FeedbackKind::Praise => format!("Marked as good: {outcome}."),
            FeedbackKind::Rejection => {
                "Marked as bad: the exchange will not be remembered. Use Correct to teach the right answer.".to_string()
            }
        };
        if let (Some(corrected_reply), Some(reply)) = (corrected_reply, self.last_brain_message()) {
            reply.bytes = corrected_reply.as_bytes().to_vec();
            reply
                .details
                .push("replaced by your correction".to_string());
        }
        self.messages
            .push(Message::new(Speaker::Application, &note));
    }
}

fn show_message(ui: &mut egui::Ui, message_index: usize, message: &Message) {
    let (speaker_name, speaker_color) = match message.speaker {
        Speaker::User => ("You", ui.visuals().hyperlink_color),
        Speaker::Brain => ("Brain", ui.visuals().strong_text_color()),
        Speaker::Application => ("·", ui.visuals().weak_text_color()),
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(
            egui::RichText::new(speaker_name)
                .color(speaker_color)
                .strong(),
        );
        match message.speaker {
            Speaker::Application => ui.label(egui::RichText::new(message.text()).weak()),
            _ => ui.label(message.text()),
        };
    });
    if message.details.is_empty() && message.thoughts.is_empty() {
        return;
    }
    ui.indent(("message_details", message_index), |ui| {
        for detail in &message.details {
            hint(ui, detail);
        }
        show_thoughts(ui, &message.thoughts);
    });
}

/// The facts behind a turn: surprise, importance, recall, storage and learning.
fn describe_turn(turn: &ChatTurnReport) -> Vec<String> {
    let mut headline = Vec::new();
    if let Some(surprise) = &turn.surprise {
        headline.push(format!("surprise {:.2} bits/byte", surprise.bits_per_byte));
    }
    headline.push(format!("importance {:.2}", turn.importance));
    if turn.is_correction {
        headline.push("read as a correction".to_string());
    }
    if let Some(memory_id) = turn.stored_memory_id {
        headline.push(format!("stored as memory #{memory_id}"));
    }
    if let Some(update) = &turn.live_update {
        headline.push(describe_live_update(update));
    }
    if turn.is_input_filtered {
        headline.push("input filtered: not stored or learned".to_string());
    }
    if let Some(reason) = describe_stop_reason(turn.reply.stop_reason) {
        headline.push(reason.to_string());
    }

    let mut details = vec![headline.join(" · ")];
    details.extend(turn.retrieved.iter().map(|found| {
        format!(
            "recalled #{} (similarity {:.2}): {}",
            found.entry.id,
            found.similarity,
            preview(&found.entry.text())
        )
    }));
    if let Some(trigger) = turn.sleep_trigger {
        details.push(format!("sleep is due: {}", trigger.description()));
    }
    details
}

fn preview(text: &str) -> String {
    let one_line = describe_on_one_line(text);
    if one_line.chars().count() <= MESSAGE_PREVIEW_CHARACTER_COUNT {
        return one_line;
    }
    let shortened: String = one_line
        .chars()
        .take(MESSAGE_PREVIEW_CHARACTER_COUNT)
        .collect();
    format!("{shortened}…")
}
