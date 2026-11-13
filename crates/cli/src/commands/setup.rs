//! Commands that get a brain started: presets, creation, and sample text.

use thingamaroo_core::brain::Brain;
use thingamaroo_core::configuration::{ModelConfiguration, ModelPreset};
use thingamaroo_core::error::{BrainError, BrainResult};
use thingamaroo_core::presentation::describe_count;
use thingamaroo_core::text::SampleCorpus;

use crate::arguments::{InitArguments, SampleCorpusArguments};
use crate::commands::Invocation;
use crate::report::print_summary;

const BYTES_PER_MEGABYTE: f64 = 1_000_000.0;

/// Lists the ready-made model sizes.
pub fn list_presets() -> BrainResult<()> {
    for preset in ModelPreset::ALL {
        let counts = preset.configuration().parameter_counts();
        println!(
            "{:<10} {:>7} parameters  (+{} fast)  {}",
            preset.name(),
            describe_count(counts.slow_total()),
            describe_count(counts.adapter),
            preset.description(),
        );
    }
    Ok(())
}

/// Creates a new, untrained brain.
pub fn create_brain(invocation: &Invocation, arguments: &InitArguments) -> BrainResult<()> {
    let model_configuration = match &arguments.configuration {
        Some(configuration_path) => read_configuration(configuration_path)?,
        None => ModelPreset::from_name(&arguments.preset)?.configuration(),
    };
    let brain = Brain::create(
        &invocation.brain_directory,
        model_configuration,
        invocation.device_choice,
        arguments.seed,
    )?;
    print_summary(&brain.summary()?);
    Ok(())
}

fn read_configuration(configuration_path: &std::path::Path) -> BrainResult<ModelConfiguration> {
    let contents = std::fs::read_to_string(configuration_path)
        .map_err(|source| BrainError::file_system(configuration_path, source))?;
    serde_json::from_str(&contents).map_err(|error| BrainError::document(configuration_path, error))
}

/// Writes a file of simple generated English.
pub fn write_sample_corpus(arguments: &SampleCorpusArguments) -> BrainResult<()> {
    let target_bytes = (arguments.megabytes * BYTES_PER_MEGABYTE) as u64;
    let written_bytes =
        SampleCorpus::built_in()?.write_file(&arguments.output, target_bytes, arguments.seed)?;
    println!(
        "Wrote {:.2} MB of sample text to {}",
        written_bytes as f64 / BYTES_PER_MEGABYTE,
        arguments.output.display()
    );
    Ok(())
}
