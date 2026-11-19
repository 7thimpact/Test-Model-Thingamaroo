//! The command-line interface: every command and its options.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Shown under `thingamaroo --help`: the shortest path from nothing to a conversation.
const QUICK_START: &str = "\
Quick start:
  thingamaroo sample-corpus sample.txt
  thingamaroo --brain my-brain init --preset tiny
  thingamaroo --brain my-brain train --data sample.txt --steps 1500 --batch 12
  thingamaroo --brain my-brain chat

Run `thingamaroo <COMMAND> --help` for the options of one command.
Full guide: docs/cli.md. Every setting explained: `thingamaroo settings reference`.";

/// The commands available inside a conversation, shown by `/help` and `chat --help`.
pub const CONVERSATION_COMMANDS: &str = "\
Commands inside the conversation:
  /correct <text>   replace the last reply with the right answer and learn it
  /good             mark the last reply as good and reinforce it
  /bad              mark the last reply as bad and forget the exchange
  /memories [text]  list memories, or find the ones closest to <text>
  /sleep            consolidate memories into the slow weights now
  /snapshot [label] save the current weights
  /snapshots        list snapshots
  /rollback <id>    restore the weights held by a snapshot
  /forget           empty the working memory (start a new conversation)
  /status           show sizes, counters and what is stored
  /help             show this list
  /quit             leave";

const INIT_EXAMPLES: &str = "\
Examples:
  thingamaroo --brain my-brain init --preset tiny
  thingamaroo --brain my-brain init --configuration architecture.json

`thingamaroo presets` lists the sizes. A custom architecture file has the same
shape as the model.json inside any brain folder.";

const TRAIN_EXAMPLES: &str = "\
Examples:
  thingamaroo --brain my-brain train --data sample.txt --steps 1500 --batch 12
  thingamaroo --brain my-brain train --data books/ notes.txt --learning-rate 3e-4

Running it again continues training. Ctrl-C stops early and saves progress.
Options left out use their defaults; see docs/cli.md for the list.";

const GENERATE_EXAMPLES: &str = "\
Examples:
  thingamaroo --brain my-brain generate \"The cat \"
  thingamaroo --brain my-brain generate \"The cat \" --temperature 0 --thoughts --max-bytes 60

Options left out use the brain's generation.* settings.";

const MEMORY_EXAMPLES: &str = "\
Examples:
  thingamaroo --brain my-brain memory list
  thingamaroo --brain my-brain memory search \"What does Mia like?\"
  thingamaroo --brain my-brain memory add \"Mia likes green pears.\" --importance 0.9
  thingamaroo --brain my-brain memory delete 3 4";

const SNAPSHOT_EXAMPLES: &str = "\
Examples:
  thingamaroo --brain my-brain snapshot create --label \"before experiments\"
  thingamaroo --brain my-brain snapshot list
  thingamaroo --brain my-brain snapshot rollback 2

The brain also takes a snapshot on its own before every live update and every sleep.";

const SETTINGS_EXAMPLES: &str = "\
Examples:
  thingamaroo --brain my-brain settings show
  thingamaroo --brain my-brain settings set generation.temperature 0.3
  thingamaroo --brain my-brain settings set safety.blocked_phrases '[\"forget everything\"]'
  thingamaroo settings reference

Values are JSON: numbers, true/false, [\"a\", \"list\"], or null; anything else is plain text.";

/// Byte-level continual-learning language model.
#[derive(Debug, Parser)]
#[command(name = "thingamaroo", version, about, long_about = None, after_help = QUICK_START)]
pub struct CommandLine {
    /// Directory that holds the brain (weights, memories, settings).
    #[arg(long, global = true, default_value = "brain", value_name = "DIRECTORY")]
    pub brain: PathBuf,

    /// Compute backend: auto, cpu, libtorch, gpu or cuda.
    ///
    /// auto picks the fastest one built in and present: a graphics card, then
    /// libtorch (PyTorch's engine on the processor), then cpu (pure Rust).
    #[arg(long, global = true, default_value = "auto", value_name = "NAME")]
    pub device: String,

    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// Everything the application can do.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// List the ready-made model sizes.
    Presets,
    /// Create a new, untrained brain.
    #[command(after_help = INIT_EXAMPLES)]
    Init(InitArguments),
    /// Write a file of simple generated English for trying things out.
    SampleCorpus(SampleCorpusArguments),
    /// Pretrain the whole model on text files.
    #[command(after_help = TRAIN_EXAMPLES)]
    Train(TrainArguments),
    /// Measure the model on text files without changing it.
    Evaluate(EvaluateArguments),
    /// Show sizes, counters and what is stored.
    Status,
    /// Show how text is grouped into patches.
    Patches(TextArguments),
    /// Send text through the encoder and back through the decoder.
    Roundtrip(TextArguments),
    /// Continue a prompt, without memory or learning.
    #[command(after_help = GENERATE_EXAMPLES)]
    Generate(GenerateArguments),
    /// Talk to the brain; it remembers and learns as you go.
    #[command(after_help = CONVERSATION_COMMANDS)]
    Chat(ChatArguments),
    /// Inspect and edit the episodic memory.
    #[command(after_help = MEMORY_EXAMPLES)]
    Memory {
        /// What to do with the memory.
        #[command(subcommand)]
        action: MemoryAction,
    },
    /// Consolidate memories into the slow weights.
    Sleep,
    /// Save and restore weights.
    #[command(after_help = SNAPSHOT_EXAMPLES)]
    Snapshot {
        /// What to do with snapshots.
        #[command(subcommand)]
        action: SnapshotAction,
    },
    /// Discard what the fast weights have learned since the last sleep.
    ResetFastWeights,
    /// View and change adjustable behaviour.
    #[command(after_help = SETTINGS_EXAMPLES)]
    Settings {
        /// What to do with the settings.
        #[command(subcommand)]
        action: SettingsAction,
    },
    /// Show the record of training, live updates and sleep.
    History(HistoryArguments),
}

/// Options for `init`.
#[derive(Debug, Args)]
pub struct InitArguments {
    /// Model size: tiny, prototype, small, target or medium.
    #[arg(long, default_value = "prototype", value_name = "NAME")]
    pub preset: String,

    /// JSON file with a custom architecture; overrides the preset.
    #[arg(long, value_name = "FILE")]
    pub configuration: Option<PathBuf>,

    /// Seed for weight initialisation.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// Options for `sample-corpus`.
#[derive(Debug, Args)]
pub struct SampleCorpusArguments {
    /// File to write.
    pub output: PathBuf,

    /// Approximate size of the file.
    #[arg(long, default_value_t = 2.0)]
    pub megabytes: f64,

    /// Seed for the generated text.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

/// Options for `train`.
#[derive(Debug, Args)]
pub struct TrainArguments {
    /// Text files, or directories searched recursively for text files.
    #[arg(long, required = true, num_args = 1.., value_name = "PATH")]
    pub data: Vec<PathBuf>,

    /// Gradient steps to take in this run.
    #[arg(long, value_name = "COUNT")]
    pub steps: Option<usize>,

    /// Sequences per batch.
    #[arg(long, value_name = "COUNT")]
    pub batch: Option<usize>,

    /// Largest learning rate, reached after warm-up.
    #[arg(long, value_name = "RATE")]
    pub learning_rate: Option<f64>,

    /// Steps over which the learning rate rises from zero.
    #[arg(long, value_name = "COUNT")]
    pub warmup: Option<usize>,

    /// Share of the corpus held out for validation.
    #[arg(long, value_name = "FRACTION")]
    pub validation_fraction: Option<f64>,

    /// Steps between progress lines.
    #[arg(long, value_name = "STEPS")]
    pub report_every: Option<usize>,

    /// Steps between validation measurements and sample generations.
    #[arg(long, value_name = "STEPS")]
    pub validate_every: Option<usize>,

    /// Steps between saving weights.
    #[arg(long, value_name = "STEPS")]
    pub checkpoint_every: Option<usize>,

    /// Prompt continued at each validation.
    #[arg(long, value_name = "TEXT")]
    pub sample_prompt: Option<String>,

    /// Windows of training text kept for replay during sleep.
    #[arg(long, value_name = "COUNT")]
    pub replay_windows: Option<usize>,

    /// Seed for batch sampling.
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,
}

/// Options for `evaluate`.
#[derive(Debug, Args)]
pub struct EvaluateArguments {
    /// Text files, or directories searched recursively for text files.
    #[arg(long, required = true, num_args = 1.., value_name = "PATH")]
    pub data: Vec<PathBuf>,

    /// Batches to measure.
    #[arg(long, default_value_t = 8)]
    pub batches: usize,

    /// Sequences per batch.
    #[arg(long, default_value_t = 8)]
    pub batch: usize,

    /// Seed for batch sampling.
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
}

/// A piece of text given on the command line.
#[derive(Debug, Args)]
pub struct TextArguments {
    /// The text to work on.
    pub text: String,
}

/// Options for `generate`.
#[derive(Debug, Args)]
pub struct GenerateArguments {
    /// The text to continue.
    pub prompt: String,

    /// Sampling temperature; 0 always picks the most likely byte.
    #[arg(long, value_name = "VALUE")]
    pub temperature: Option<f64>,

    /// Only this many most-likely symbols are considered for each byte.
    #[arg(long, value_name = "COUNT")]
    pub candidates: Option<usize>,

    /// Internal refinement steps before each patch is written.
    #[arg(long, value_name = "COUNT")]
    pub thinking_steps: Option<usize>,

    /// Longest continuation to write.
    #[arg(long = "max-bytes", value_name = "BYTES")]
    pub maximum_reply_bytes: Option<usize>,

    /// Fixes the sampling sequence.
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,

    /// Show what each thinking stage would have written.
    #[arg(long)]
    pub thoughts: bool,
}

/// Options for `chat`.
#[derive(Debug, Args)]
pub struct ChatArguments {
    /// Read-only conversation: no weight updates, no memory writes.
    #[arg(long)]
    pub frozen: bool,

    /// Learn during the session but write nothing to disk.
    #[arg(long)]
    pub no_save: bool,

    /// Show what each thinking stage would have written.
    #[arg(long)]
    pub thoughts: bool,
}

/// Actions on the episodic memory.
#[derive(Debug, Subcommand)]
pub enum MemoryAction {
    /// List stored memories, most important first.
    List {
        /// Show at most this many memories.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Find the memories closest in meaning to a text.
    Search {
        /// The text to search for.
        query: String,
        /// Show at most this many memories.
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Show one memory in full.
    Show {
        /// Identifier of the memory.
        id: u64,
    },
    /// Store a fact by hand.
    Add {
        /// The fact to remember.
        text: String,
        /// How much the fact matters, from 0 to 1.
        #[arg(long, default_value_t = 0.9)]
        importance: f64,
    },
    /// Delete memories by identifier.
    Delete {
        /// Identifiers of the memories to delete.
        #[arg(required = true)]
        ids: Vec<u64>,
    },
    /// Delete every memory.
    Clear,
}

/// Actions on snapshots.
#[derive(Debug, Subcommand)]
pub enum SnapshotAction {
    /// Save the current weights.
    Create {
        /// A note on why the snapshot was taken.
        #[arg(long, default_value = "manual")]
        label: String,
        /// Also save the slow predictor, not only the fast weights.
        #[arg(long)]
        with_slow_weights: bool,
    },
    /// List snapshots, oldest first.
    List,
    /// Restore the weights held by a snapshot.
    Rollback {
        /// Identifier of the snapshot.
        id: u64,
    },
    /// Delete a snapshot.
    Delete {
        /// Identifier of the snapshot.
        id: u64,
    },
}

/// Actions on the settings.
#[derive(Debug, Subcommand)]
pub enum SettingsAction {
    /// List every setting with its current value and what it does.
    Show,
    /// Print the default settings and their explanations as Markdown; needs no brain.
    Reference,
    /// Change one setting, for example `settings set sleep.replay_step_count 30`.
    Set {
        /// Dotted path of the setting.
        path: String,
        /// New value: a number, true/false, text, or a JSON list.
        value: String,
    },
    /// Restore every setting to its default.
    Reset,
}

/// Options for `history`.
#[derive(Debug, Args)]
pub struct HistoryArguments {
    /// Show at most this many of the most recent records.
    #[arg(long, default_value_t = 30)]
    pub limit: usize,
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    const COMMAND_LINE_GUIDE: &str = include_str!("../../../docs/cli.md");
    const DESKTOP_GUIDE: &str = include_str!("../../../docs/gui.md");

    /// Commands and options clap adds on its own, which need no documentation of ours.
    const BUILT_IN_NAMES: [&str; 2] = ["help", "version"];

    /// Every command a user can run, as the words typed after the program name.
    fn runnable_command_paths() -> Vec<String> {
        let mut paths = Vec::new();
        collect_command_paths(&CommandLine::command(), "", &mut paths);
        paths
    }

    fn collect_command_paths(command: &clap::Command, parent_path: &str, paths: &mut Vec<String>) {
        let subcommands: Vec<&clap::Command> = command
            .get_subcommands()
            .filter(|subcommand| !BUILT_IN_NAMES.contains(&subcommand.get_name()))
            .collect();
        if subcommands.is_empty() && !parent_path.is_empty() {
            paths.push(parent_path.to_string());
        }
        for subcommand in subcommands {
            let path = format!("{parent_path} {}", subcommand.get_name());
            collect_command_paths(subcommand, path.trim(), paths);
        }
    }

    /// Every `--option` of a command and of its subcommands.
    fn option_names(command: &clap::Command) -> Vec<String> {
        let own_options = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .filter(|name| !BUILT_IN_NAMES.contains(name))
            .map(|name| format!("--{name}"));
        let nested_options = command.get_subcommands().flat_map(option_names);
        own_options.chain(nested_options).collect()
    }

    #[test]
    fn the_command_line_definition_is_consistent() {
        CommandLine::command().debug_assert();
    }

    #[test]
    fn every_command_is_explained_in_the_command_line_guide() {
        for path in runnable_command_paths() {
            assert!(
                COMMAND_LINE_GUIDE.contains(&format!("`{path}")),
                "docs/cli.md does not explain `{path}`"
            );
        }
    }

    #[test]
    fn every_option_is_explained_in_the_command_line_guide() {
        for option_name in option_names(&CommandLine::command()) {
            assert!(
                COMMAND_LINE_GUIDE.contains(&option_name),
                "docs/cli.md does not explain {option_name}"
            );
        }
    }

    #[test]
    fn every_command_has_a_desktop_equivalent_in_the_desktop_guide() {
        for path in runnable_command_paths() {
            assert!(
                DESKTOP_GUIDE.contains(&format!("`{path}`")),
                "docs/gui.md does not say where `{path}` is in the desktop application"
            );
        }
    }
}
