//! Commands that consolidate, save and restore weights.

use thingamaroo_core::error::BrainResult;

use crate::arguments::SnapshotAction;
use crate::commands::Invocation;
use crate::interrupt::install_interrupt_handler;
use crate::report::{describe_snapshot, print_progress, print_sleep_report};

/// Consolidates memories into the slow weights.
pub fn sleep(invocation: &Invocation) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    let cancellation = install_interrupt_handler();
    let report = brain.sleep(&cancellation, &mut print_progress)?;
    print_sleep_report(&report);
    Ok(())
}

/// Runs one snapshot action.
pub fn run_snapshot_action(invocation: &Invocation, action: SnapshotAction) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    match action {
        SnapshotAction::Create {
            label,
            with_slow_weights,
        } => {
            let information = brain.create_snapshot(&label, with_slow_weights)?;
            println!("Created {}", describe_snapshot(&information));
        }
        SnapshotAction::List => {
            let snapshots = brain.snapshots()?;
            for information in &snapshots {
                println!("{}", describe_snapshot(information));
            }
            println!("{} snapshots", snapshots.len());
        }
        SnapshotAction::Rollback { id } => {
            let information = brain.roll_back(id)?;
            println!("Rolled back to {}", describe_snapshot(&information));
        }
        SnapshotAction::Delete { id } => {
            brain.delete_snapshot(id)?;
            println!("Deleted snapshot #{id}");
        }
    }
    Ok(())
}

/// Discards what the fast weights have learned since the last sleep.
pub fn reset_fast_weights(invocation: &Invocation) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    brain.reset_fast_weights()?;
    println!("Fast weights reset; the slow weights and memories are unchanged.");
    Ok(())
}
