//! Commands that look inside a brain without changing it.

use thingamaroo_core::error::BrainResult;
use thingamaroo_core::presentation::describe_on_one_line;

use crate::arguments::{HistoryArguments, TextArguments};
use crate::commands::Invocation;
use crate::report::{print_history_record, print_round_trip, print_summary};

/// Shows sizes, counters and what is stored.
pub fn show_status(invocation: &Invocation) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    print_summary(&brain.summary()?);
    Ok(())
}

/// Shows how text is grouped into patches.
pub fn show_patches(invocation: &Invocation, arguments: &TextArguments) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    let patches = brain.split_patches(&arguments.text);
    let bracketed: Vec<String> = patches
        .iter()
        .map(|patch| format!("[{}]", describe_on_one_line(patch)))
        .collect();
    println!("{}", bracketed.join(" "));
    println!(
        "{} bytes -> {} patches",
        arguments.text.len(),
        patches.len()
    );
    Ok(())
}

/// Sends text through the encoder and back through the decoder.
pub fn show_round_trip(invocation: &Invocation, arguments: &TextArguments) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    print_round_trip(&brain.round_trip(&arguments.text));
    Ok(())
}

/// Shows the most recent records of training, live updates and sleep.
pub fn show_history(invocation: &Invocation, arguments: &HistoryArguments) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    let records = brain.history()?;
    let first_shown = records.len().saturating_sub(arguments.limit);
    for record in &records[first_shown..] {
        print_history_record(record);
    }
    println!(
        "{} of {} records shown",
        records.len() - first_shown,
        records.len()
    );
    Ok(())
}
