//! The `memory` commands: inspect and edit the episodic memory.

use thingamaroo_core::error::{BrainError, BrainResult};
use thingamaroo_core::memory::retention_score;

use crate::arguments::MemoryAction;
use crate::commands::Invocation;
use crate::report::{describe_match, describe_memory, print_memory};

/// Runs one memory action.
pub fn run(invocation: &Invocation, action: MemoryAction) -> BrainResult<()> {
    match action {
        MemoryAction::List { limit } => list(invocation, limit),
        MemoryAction::Search { query, limit } => search(invocation, &query, limit),
        MemoryAction::Show { id } => show(invocation, id),
        MemoryAction::Add { text, importance } => add(invocation, &text, importance),
        MemoryAction::Delete { ids } => delete(invocation, &ids),
        MemoryAction::Clear => clear(invocation),
    }
}

fn list(invocation: &Invocation, limit: usize) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    let mut memories = brain.memories()?;
    memories.sort_by(|left, right| retention_score(right).total_cmp(&retention_score(left)));
    for entry in memories.iter().take(limit) {
        println!("{}", describe_memory(entry));
    }
    println!(
        "{} of {} memories shown",
        memories.len().min(limit),
        memories.len()
    );
    Ok(())
}

fn search(invocation: &Invocation, query: &str, limit: usize) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    let matches = brain.search_memories(query, limit)?;
    if matches.is_empty() {
        println!("No memories stored yet.");
    }
    for found in &matches {
        println!("{}", describe_match(found));
    }
    Ok(())
}

fn show(invocation: &Invocation, id: u64) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    let entry = brain
        .memories()?
        .into_iter()
        .find(|entry| entry.id == id)
        .ok_or_else(|| BrainError::NotFound(format!("memory {id}")))?;
    print_memory(&entry);
    Ok(())
}

fn add(invocation: &Invocation, text: &str, importance: f64) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    let entry = brain.add_note(text, importance)?;
    println!(
        "Stored as memory #{} with importance {:.2}",
        entry.id, entry.importance
    );
    Ok(())
}

fn delete(invocation: &Invocation, ids: &[u64]) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    let deleted_count = brain.delete_memories(ids)?;
    println!(
        "Deleted {deleted_count} of {} requested memories",
        ids.len()
    );
    Ok(())
}

fn clear(invocation: &Invocation) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    let deleted_count = brain.clear_memories()?;
    println!("Deleted all {deleted_count} memories");
    Ok(())
}
