//! The `chat` command: an interactive conversation that remembers and learns.

use std::io::{BufRead, Write};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use thingamaroo_core::brain::{Brain, ChatOptions, ChatSession};
use thingamaroo_core::cancellation::CancellationFlag;
use thingamaroo_core::error::BrainResult;

use crate::arguments::{CONVERSATION_COMMANDS, ChatArguments};
use crate::commands::Invocation;
use crate::commands::generation::write_to_terminal;
use crate::interrupt::install_interrupt_handler;
use crate::report::{
    describe_match, describe_memory, describe_snapshot, print_feedback, print_progress,
    print_sleep_report, print_summary, print_thoughts, print_turn_details,
};

/// How often the prompt wakes up to check for Ctrl-C and for idle sleep.
const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(250);

const COMMAND_PREFIX: char = '/';
const RECALLED_MEMORY_COUNT: usize = 5;
const LISTED_MEMORY_COUNT: usize = 10;

/// Whether the conversation goes on after a line was handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
}

/// A running conversation and everything it needs.
struct Conversation {
    brain: Brain,
    session: ChatSession,
    cancellation: CancellationFlag,
    last_activity: Instant,
}

/// Talks to the brain until the user leaves.
pub fn chat(invocation: &Invocation, arguments: &ChatArguments) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    let options = ChatOptions {
        is_frozen: arguments.frozen,
        is_saving: !arguments.no_save,
        is_tracing_thoughts: arguments.thoughts,
    };
    let mut conversation = Conversation {
        session: brain.start_chat(options),
        brain,
        cancellation: install_interrupt_handler(),
        last_activity: Instant::now(),
    };
    println!("{}", describe_mode(options));
    println!("Type /help for commands, /quit to leave.");

    let input_lines = spawn_line_reader();
    loop {
        show_prompt();
        let Some(line) = conversation.wait_for_line(&input_lines)? else {
            break;
        };
        if conversation.handle_line(line.trim())? == Flow::Quit {
            break;
        }
    }
    println!();
    Ok(())
}

fn describe_mode(options: ChatOptions) -> &'static str {
    match (options.is_frozen, options.is_saving) {
        (true, _) => "Frozen conversation: nothing is learned or stored.",
        (false, false) => "Learning for this session only: nothing is written to disk.",
        (false, true) => "Learning conversation: surprising inputs and corrections are remembered.",
    }
}

fn show_prompt() {
    print!("\nyou> ");
    let _ = std::io::stdout().flush();
}

/// Reads input lines on a separate thread so the prompt can notice idleness and Ctrl-C.
fn spawn_line_reader() -> Receiver<String> {
    let (line_sender, line_receiver) = channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if line_sender.send(line).is_err() {
                break;
            }
        }
    });
    line_receiver
}

impl Conversation {
    /// Waits for the next input line; `None` means the conversation is over.
    ///
    /// While waiting, the brain sleeps on its own once the schedule says the
    /// conversation has been idle long enough.
    fn wait_for_line(&mut self, input_lines: &Receiver<String>) -> BrainResult<Option<String>> {
        loop {
            match input_lines.recv_timeout(INPUT_POLL_INTERVAL) {
                Ok(line) => {
                    self.last_activity = Instant::now();
                    return Ok(Some(line));
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(None),
                Err(RecvTimeoutError::Timeout) => {}
            }
            if self.cancellation.is_requested() {
                return Ok(None);
            }
            if self.sleep_if_due()? {
                show_prompt();
            }
        }
    }

    fn handle_line(&mut self, line: &str) -> BrainResult<Flow> {
        if line.is_empty() {
            return Ok(Flow::Continue);
        }
        let Some(command_line) = line.strip_prefix(COMMAND_PREFIX) else {
            self.take_turn(line)?;
            return Ok(Flow::Continue);
        };
        let (command_name, argument) = command_line
            .split_once(char::is_whitespace)
            .map_or((command_line, ""), |(name, rest)| (name, rest.trim()));

        // A failed command is reported and the conversation carries on.
        match self.run_command(command_name, argument) {
            Ok(flow) => Ok(flow),
            Err(error) => {
                println!("  {error}");
                Ok(Flow::Continue)
            }
        }
    }

    fn take_turn(&mut self, user_text: &str) -> BrainResult<()> {
        print!("bot> ");
        let turn = self.brain.chat_turn(
            &mut self.session,
            user_text,
            &self.cancellation,
            &mut write_to_terminal,
        )?;
        // A Ctrl-C during the reply only stops that reply, not the conversation.
        self.cancellation.reset();
        if !turn.reply.bytes.ends_with(b"\n") {
            println!();
        }
        print_thoughts(&turn.reply.thoughts);
        print_turn_details(&turn);
        self.last_activity = Instant::now();
        self.sleep_if_due()?;
        Ok(())
    }

    fn run_command(&mut self, command_name: &str, argument: &str) -> BrainResult<Flow> {
        match command_name {
            "quit" | "exit" => return Ok(Flow::Quit),
            "help" => println!("Type a message to talk.\n{CONVERSATION_COMMANDS}"),
            "correct" => {
                print_feedback(&self.brain.correct_last_reply(&mut self.session, argument)?)
            }
            "good" => print_feedback(&self.brain.praise_last_reply(&mut self.session)?),
            "bad" => {
                self.brain.reject_last_reply(&mut self.session)?;
                println!(
                    "  The exchange will not be remembered. Use /correct <text> to teach the right answer."
                );
            }
            "memories" => self.show_memories(argument)?,
            "sleep" => self.sleep_now()?,
            "snapshot" => {
                let label = if argument.is_empty() {
                    "manual"
                } else {
                    argument
                };
                let information = self.brain.create_snapshot(label, false)?;
                println!("  created {}", describe_snapshot(&information));
            }
            "snapshots" => self.show_snapshots()?,
            "rollback" => self.roll_back(argument)?,
            "forget" => {
                self.session.forget_transcript();
                println!("  Working memory emptied.");
            }
            "status" => print_summary(&self.brain.summary()?),
            _ => println!("  Unknown command /{command_name}. Type /help for the list."),
        }
        Ok(Flow::Continue)
    }

    fn show_memories(&mut self, query: &str) -> BrainResult<()> {
        if !query.is_empty() {
            let matches = self.brain.search_memories(query, RECALLED_MEMORY_COUNT)?;
            matches
                .iter()
                .for_each(|found| println!("  {}", describe_match(found)));
            return Ok(());
        }
        let mut memories = self.brain.memories()?;
        memories.sort_by(|left, right| right.importance.total_cmp(&left.importance));
        for entry in memories.iter().take(LISTED_MEMORY_COUNT) {
            println!("  {}", describe_memory(entry));
        }
        println!("  {} memories stored", memories.len());
        Ok(())
    }

    fn show_snapshots(&self) -> BrainResult<()> {
        let snapshots = self.brain.snapshots()?;
        for information in &snapshots {
            println!("  {}", describe_snapshot(information));
        }
        println!("  {} snapshots", snapshots.len());
        Ok(())
    }

    fn roll_back(&mut self, argument: &str) -> BrainResult<()> {
        let Ok(snapshot_id) = argument.parse::<u64>() else {
            println!("  Usage: /rollback <snapshot id>");
            return Ok(());
        };
        let information = self.brain.roll_back(snapshot_id)?;
        println!("  rolled back to {}", describe_snapshot(&information));
        Ok(())
    }

    /// Sleeps when the schedule says so; returns whether a sleep happened.
    fn sleep_if_due(&mut self) -> BrainResult<bool> {
        let options = self.session.options();
        if options.is_frozen || !options.is_saving {
            return Ok(false);
        }
        let Some(trigger) = self.brain.sleep_trigger(self.last_activity.elapsed())? else {
            return Ok(false);
        };
        println!("\nGoing to sleep: {}.", trigger.description());
        self.sleep_now()?;
        Ok(true)
    }

    fn sleep_now(&mut self) -> BrainResult<()> {
        let report = self.brain.sleep(&self.cancellation, &mut print_progress)?;
        self.cancellation.reset();
        print_sleep_report(&report);
        self.last_activity = Instant::now();
        Ok(())
    }
}
