//! Runs the command chosen on the command line.

mod chat;
mod generation;
mod inspection;
mod maintenance;
mod memory;
mod settings;
mod setup;
mod training;

use std::path::PathBuf;

use thingamaroo_core::brain::Brain;
use thingamaroo_core::error::BrainResult;
use thingamaroo_core::model::DeviceChoice;

use crate::arguments::{Command, CommandLine};

/// The options shared by every command.
#[derive(Debug, Clone)]
pub struct Invocation {
    /// Directory that holds the brain.
    pub brain_directory: PathBuf,
    /// Compute backend to run on.
    pub device_choice: DeviceChoice,
}

impl Invocation {
    /// Opens the brain the command line points at.
    pub fn open_brain(&self) -> BrainResult<Brain> {
        Brain::open(&self.brain_directory, self.device_choice)
    }
}

/// Dispatches to the chosen command.
pub fn run(command_line: CommandLine) -> BrainResult<()> {
    let invocation = Invocation {
        brain_directory: command_line.brain,
        device_choice: DeviceChoice::from_name(&command_line.device)?,
    };
    match command_line.command {
        Command::Presets => setup::list_presets(),
        Command::Init(arguments) => setup::create_brain(&invocation, &arguments),
        Command::SampleCorpus(arguments) => setup::write_sample_corpus(&arguments),
        Command::Train(arguments) => training::train(&invocation, &arguments),
        Command::Evaluate(arguments) => training::evaluate(&invocation, &arguments),
        Command::Status => inspection::show_status(&invocation),
        Command::Patches(arguments) => inspection::show_patches(&invocation, &arguments),
        Command::Roundtrip(arguments) => inspection::show_round_trip(&invocation, &arguments),
        Command::History(arguments) => inspection::show_history(&invocation, &arguments),
        Command::Generate(arguments) => generation::generate(&invocation, &arguments),
        Command::Chat(arguments) => chat::chat(&invocation, &arguments),
        Command::Memory { action } => memory::run(&invocation, action),
        Command::Sleep => maintenance::sleep(&invocation),
        Command::Snapshot { action } => maintenance::run_snapshot_action(&invocation, action),
        Command::ResetFastWeights => maintenance::reset_fast_weights(&invocation),
        Command::Settings { action } => settings::run(&invocation, action),
    }
}
