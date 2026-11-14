//! Commands that train and measure the model.

use thingamaroo_core::configuration::TrainingOptions;
use thingamaroo_core::error::BrainResult;
use thingamaroo_core::presentation::describe_losses;

use crate::arguments::{EvaluateArguments, TrainArguments};
use crate::commands::Invocation;
use crate::interrupt::install_interrupt_handler;
use crate::report::{print_evaluation, print_progress};

/// Pretrains the whole model on text files.
pub fn train(invocation: &Invocation, arguments: &TrainArguments) -> BrainResult<()> {
    let mut brain = invocation.open_brain()?;
    let options = training_options(arguments);
    let cancellation = install_interrupt_handler();

    let summary = brain.pretrain(&options, &cancellation, &mut print_progress)?;

    let outcome = if summary.was_cancelled {
        "Stopped early"
    } else {
        "Finished"
    };
    println!(
        "{outcome} after {} steps in {:.0} s; weights saved to {}",
        summary.step_count,
        summary.elapsed_seconds,
        invocation.brain_directory.display()
    );
    println!(
        "  last batch: {}",
        describe_losses(&summary.last_training_losses)
    );
    if let Some(validation) = &summary.last_validation {
        println!("  held-out:   {}", describe_losses(&validation.losses));
    }
    Ok(())
}

/// Starts from the default options and applies whatever the command line overrides.
fn training_options(arguments: &TrainArguments) -> TrainingOptions {
    let mut options = TrainingOptions::default();
    options.data.corpus_paths = arguments.data.clone();
    apply(&mut options.data.batch_sequence_count, arguments.batch);
    apply(
        &mut options.data.validation_fraction,
        arguments.validation_fraction,
    );
    apply(
        &mut options.data.replay_window_count,
        arguments.replay_windows,
    );
    apply(&mut options.data.seed, arguments.seed);
    apply(&mut options.schedule.step_count, arguments.steps);
    apply(&mut options.schedule.warmup_step_count, arguments.warmup);
    apply(
        &mut options.schedule.peak_learning_rate,
        arguments.learning_rate,
    );
    apply(
        &mut options.reporting.report_interval_steps,
        arguments.report_every,
    );
    apply(
        &mut options.reporting.validation_interval_steps,
        arguments.validate_every,
    );
    apply(
        &mut options.reporting.checkpoint_interval_steps,
        arguments.checkpoint_every,
    );
    apply(
        &mut options.reporting.sample_prompt,
        arguments.sample_prompt.clone(),
    );
    options
}

/// Replaces `setting` when the command line supplied a value.
pub(crate) fn apply<Value>(setting: &mut Value, supplied: Option<Value>) {
    if let Some(value) = supplied {
        *setting = value;
    }
}

/// Measures the model on text files without changing it.
pub fn evaluate(invocation: &Invocation, arguments: &EvaluateArguments) -> BrainResult<()> {
    let brain = invocation.open_brain()?;
    let report = brain.evaluate(
        &arguments.data,
        arguments.batches,
        arguments.batch,
        arguments.seed,
    )?;
    print_evaluation(&report);
    Ok(())
}
