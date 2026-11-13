//! Prints reports from the core library as plain text.

use thingamaroo_core::brain::{
    BrainSummary, ChatTurnReport, EvaluationReport, FeedbackReport, HistoryRecord, RoundTripReport,
    SleepReport, SnapshotInformation,
};
use thingamaroo_core::learning::{
    ProgressEvent, SleepStepReport, TrainingStepReport, ValidationReport,
};
use thingamaroo_core::memory::{MemoryEntry, MemoryMatch, retention_score};
use thingamaroo_core::model::{Reply, StopReason, ThoughtTrace};
use thingamaroo_core::presentation::{
    describe_age, describe_count, describe_live_update, describe_losses, describe_on_one_line,
};

const PREVIEW_CHARACTER_COUNT: usize = 70;
const SLEEP_STEP_PRINT_INTERVAL: usize = 10;

/// Prints one progress event of a long-running operation.
pub fn print_progress(event: ProgressEvent) {
    match event {
        ProgressEvent::Status(status) => println!("{status}"),
        ProgressEvent::TrainingStep(report) => print_training_step(&report),
        ProgressEvent::Validation(report) => print_validation(&report),
        ProgressEvent::CheckpointSaved { lifetime_step } => {
            println!("  checkpoint saved at step {lifetime_step}")
        }
        ProgressEvent::SleepStep(report) if is_worth_printing(&report) => print_sleep_step(&report),
        ProgressEvent::SleepStep(_) => {}
    }
}

/// Replay steps are quick, so only every few of them (and the last) get a line.
fn is_worth_printing(report: &SleepStepReport) -> bool {
    report.step.is_multiple_of(SLEEP_STEP_PRINT_INTERVAL) || report.step == report.step_count
}

fn print_training_step(report: &TrainingStepReport) {
    println!(
        "step {:>6}/{}  {}  lr {:.1e}  {} patches/s",
        report.run_step,
        report.run_step_count,
        describe_losses(&report.losses),
        report.learning_rate,
        describe_count(report.patches_per_second as usize),
    );
}

fn print_validation(report: &ValidationReport) {
    println!("  held-out: {}", describe_losses(&report.losses));
    println!("  sample:   {}", describe_on_one_line(&report.sample));
}

fn print_sleep_step(report: &SleepStepReport) {
    println!(
        "  replay {:>4}/{}  loss {:.3}  elastic penalty {:.5}",
        report.step, report.step_count, report.replay_loss, report.elastic_penalty
    );
}

/// Prints what a brain is and what it has been through.
pub fn print_summary(summary: &BrainSummary) {
    let model = &summary.model_configuration;
    println!("Brain at {}", summary.root.display());
    println!("  device              {}", summary.device_name);
    println!(
        "  slow weights        {} (encoder {}, predictor {}, decoder {})",
        describe_count(summary.slow_parameter_count()),
        describe_count(summary.encoder_parameter_count),
        describe_count(summary.predictor_parameter_count),
        describe_count(summary.decoder_parameter_count),
    );
    println!(
        "  fast weights        {}",
        describe_count(summary.adapter_parameter_count)
    );
    println!(
        "  meaning vectors     {} dimensions, patches up to {} bytes",
        model.latent.width, model.patching.maximum_patch_bytes
    );
    println!(
        "  working memory      {} patches, {} memory slots, {} thinking steps",
        model.predictor.maximum_context_patches,
        model.predictor.maximum_memory_slots,
        model.predictor.thinking_step_count,
    );
    print_state(summary);
}

fn print_state(summary: &BrainSummary) {
    let state = &summary.state;
    let validation = state
        .training
        .validation_bits_per_byte
        .map_or("not measured".to_string(), |bits| {
            format!("{bits:.2} bits/byte")
        });
    println!(
        "  pretraining steps   {} (held-out {validation})",
        state.training.step_count
    );
    println!(
        "  conversation        {} turns, {} live updates ({} turns since sleep)",
        state.conversation.turn_count,
        state.conversation.live_update_count,
        state.conversation.turns_since_sleep,
    );
    println!(
        "  sleep               {} cycles, last {}",
        state.sleep.cycle_count,
        describe_age(state.sleep.last_sleep_unix_seconds),
    );
    println!(
        "  surprise baseline   {:.2} bits/byte over {} inputs",
        state.surprise.mean_bits_per_byte, state.surprise.observation_count
    );
    println!("  memories            {}", summary.memory_count);
    println!("  snapshots           {}", summary.snapshot_count);
    println!(
        "  replay data         {} windows, elastic anchor {}",
        summary.replay_window_count,
        if summary.has_elastic_anchor {
            "recorded"
        } else {
            "missing"
        },
    );
    if !summary.is_pretrained() {
        println!("  This brain is untrained. Run `train --data <files>` first.");
    }
}

/// Prints a held-out measurement.
pub fn print_evaluation(report: &EvaluationReport) {
    println!("Measured {} batches", report.batch_count);
    println!("  {}", describe_losses(&report.losses));
    println!(
        "  rebuild accuracy    {:.1}% of symbols",
        report.reconstruction_accuracy * 100.0
    );
    println!(
        "  latent spread       {:.3} (near zero would mean collapse)",
        report.latent_deviation
    );
}

/// Prints a text before and after its trip through meaning space.
pub fn print_round_trip(report: &RoundTripReport) {
    for patch in &report.patches {
        let marker = if patch.is_exact { "ok " } else { "!! " };
        println!(
            "  {marker}[{}] -> [{}]",
            describe_on_one_line(&patch.original),
            describe_on_one_line(&patch.rebuilt)
        );
    }
    println!("rebuilt: {}", describe_on_one_line(&report.rebuilt_text));
    println!("byte accuracy: {:.1}%", report.byte_accuracy * 100.0);
}

/// Prints the coarse-to-fine thinking behind each written patch.
///
/// Each stage shows what it would have written and how far its meaning vector
/// still was from the final one (0.00 for the last stage).
pub fn print_thoughts(thoughts: &[ThoughtTrace]) {
    for thought in thoughts {
        let stages: Vec<String> = thought
            .stages
            .iter()
            .map(|stage| {
                format!(
                    "[{}] {:.2}",
                    describe_on_one_line(&stage.preview),
                    stage.distance_to_final
                )
            })
            .collect();
        println!(
            "  thought: {} => wrote [{}]",
            stages.join(" -> "),
            describe_on_one_line(&thought.written)
        );
    }
}

/// Explains why a reply ended, when the reason is worth mentioning.
pub fn describe_stop_reason(reply: &Reply) -> Option<&'static str> {
    match reply.stop_reason {
        StopReason::LineEnd | StopReason::NothingMoreToWrite => None,
        StopReason::LengthLimit => Some("reached the reply length limit"),
        StopReason::Cancelled => Some("stopped"),
    }
}

/// Prints what happened behind a conversation turn.
pub fn print_turn_details(turn: &ChatTurnReport) {
    let mut details = Vec::new();
    if let Some(surprise) = &turn.surprise {
        details.push(format!("surprise {:.2} bits/byte", surprise.bits_per_byte));
    }
    details.push(format!("importance {:.2}", turn.importance));
    if turn.is_correction {
        details.push("read as a correction".to_string());
    }
    if !turn.retrieved.is_empty() {
        details.push(format!("recalled {} memories", turn.retrieved.len()));
    }
    if let Some(memory_id) = turn.stored_memory_id {
        details.push(format!("stored as memory #{memory_id}"));
    }
    if let Some(update) = &turn.live_update {
        details.push(describe_live_update(update));
    }
    if turn.is_input_filtered {
        details.push("input filtered: not stored or learned".to_string());
    }
    if let Some(reason) = describe_stop_reason(&turn.reply) {
        details.push(reason.to_string());
    }
    println!("  ({})", details.join(" · "));
    for found in &turn.retrieved {
        println!("  recalled: {}", describe_match(found));
    }
}

/// Prints what giving feedback did.
pub fn print_feedback(report: &FeedbackReport) {
    match &report.live_update {
        Some(update) => println!("  {}", describe_live_update(update)),
        None => println!("  no weight update"),
    }
    if let Some(memory_id) = report.memory_id {
        println!(
            "  stored as memory #{memory_id} with importance {:.2}",
            report.importance
        );
    }
}

/// One line describing a stored memory.
pub fn describe_memory(entry: &MemoryEntry) -> String {
    format!(
        "#{:<5} {:<10} importance {:.2}  keeps {:.2}  recalled {}x  {}  {}",
        entry.id,
        entry.kind.name(),
        entry.importance,
        retention_score(entry),
        entry.retrieval_count,
        describe_age(entry.created_unix_seconds),
        preview(&entry.text()),
    )
}

/// One line describing a memory found by similarity.
pub fn describe_match(found: &MemoryMatch) -> String {
    format!(
        "#{} similarity {:.2}  {}",
        found.entry.id,
        found.similarity,
        preview(&found.entry.text())
    )
}

/// Prints a memory in full.
pub fn print_memory(entry: &MemoryEntry) {
    println!("Memory #{} ({})", entry.id, entry.kind.name());
    println!("  importance   {:.3}", entry.importance);
    println!(
        "  created      {}",
        describe_age(entry.created_unix_seconds)
    );
    println!(
        "  recalled     {} times, last {}",
        entry.retrieval_count,
        describe_age(entry.last_retrieved_unix_seconds)
    );
    println!("  prompt       {}", describe_on_one_line(&entry.prompt));
    println!("  reply        {}", describe_on_one_line(&entry.reply));
}

fn preview(text: &str) -> String {
    let one_line = describe_on_one_line(text);
    if one_line.chars().count() <= PREVIEW_CHARACTER_COUNT {
        return one_line;
    }
    let shortened: String = one_line.chars().take(PREVIEW_CHARACTER_COUNT).collect();
    format!("{shortened}…")
}

/// Prints what a sleep did.
pub fn print_sleep_report(report: &SleepReport) {
    println!(
        "Sleep finished: {} memories replayed in {} steps",
        report.replayed_episode_count, report.consolidation.step_count
    );
    println!(
        "  memories   loss {} -> {}",
        describe_optional_loss(report.episode_loss_before),
        describe_optional_loss(report.episode_loss_after)
    );
    println!(
        "  old skills loss {} -> {}",
        describe_optional_loss(report.old_loss_before),
        describe_optional_loss(report.old_loss_after)
    );
    if report.was_discarded {
        println!(
            "  The new weights were DISCARDED: old skills degraded too much or the sleep was stopped."
        );
    } else {
        println!(
            "  fast weights reset; {} memories pruned, {} remain",
            report.pruned_memory_count, report.remaining_memory_count
        );
    }
    if let Some(snapshot_id) = report.snapshot_id {
        println!("  snapshot #{snapshot_id} holds the weights from before this sleep");
    }
}

fn describe_optional_loss(loss: Option<f64>) -> String {
    loss.map_or("n/a".to_string(), |loss| format!("{loss:.3}"))
}

/// One line describing a snapshot.
pub fn describe_snapshot(information: &SnapshotInformation) -> String {
    format!(
        "#{:<4} {:<9} {:<22} {}  {}",
        information.id,
        if information.is_automatic {
            "automatic"
        } else {
            "manual"
        },
        if information.includes_slow_weights {
            "fast + slow predictor"
        } else {
            "fast weights"
        },
        describe_age(information.created_unix_seconds),
        information.label,
    )
}

/// Prints one line of the learning history.
pub fn print_history_record(record: &HistoryRecord) {
    match record {
        HistoryRecord::TrainingStep(report) => println!(
            "train    step {:>7}  {}",
            report.lifetime_step,
            describe_losses(&report.losses)
        ),
        HistoryRecord::Validation(report) => println!(
            "validate step {:>7}  {}",
            report.lifetime_step,
            describe_losses(&report.losses)
        ),
        HistoryRecord::LiveUpdate {
            unix_seconds,
            importance,
            report,
        } => println!(
            "learn    {}  importance {importance:.2}  {}",
            describe_age(*unix_seconds),
            describe_live_update(report)
        ),
        HistoryRecord::Sleep(report) => println!(
            "sleep    {}  {} memories, old skills {} -> {}{}",
            describe_age(report.finished_unix_seconds),
            report.replayed_episode_count,
            describe_optional_loss(report.old_loss_before),
            describe_optional_loss(report.old_loss_after),
            if report.was_discarded {
                "  (discarded)"
            } else {
                ""
            },
        ),
    }
}
