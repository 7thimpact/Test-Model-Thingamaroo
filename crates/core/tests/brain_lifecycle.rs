//! Walks a very small brain through its whole life: create, pretrain, talk,
//! learn from a correction, roll back, sleep, and reopen.

use std::path::PathBuf;

use thingamaroo_core::brain::{Brain, ChatOptions};
use thingamaroo_core::cancellation::CancellationFlag;
use thingamaroo_core::configuration::{
    GenerationSettings, ModelConfiguration, ModelPreset, TrainingOptions,
};
use thingamaroo_core::learning::ProgressEvent;
use thingamaroo_core::memory::MemoryKind;
use thingamaroo_core::model::DeviceChoice;
use thingamaroo_core::text::SampleCorpus;

const TRAINING_STEP_COUNT: usize = 40;

/// A model small enough to train in a few seconds on any machine.
fn miniature_configuration() -> ModelConfiguration {
    let mut configuration = ModelPreset::Tiny.configuration();
    configuration.latent.width = 32;
    configuration.encoder.width = 32;
    configuration.encoder.layer_count = 1;
    configuration.decoder.width = 32;
    configuration.decoder.layer_count = 1;
    configuration.predictor.width = 48;
    configuration.predictor.layer_count = 2;
    configuration.predictor.maximum_context_patches = 16;
    configuration.adapter.rank = 2;
    configuration
}

fn fresh_directory(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("thingamaroo-lifecycle-{name}"));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn training_options(corpus_path: PathBuf) -> TrainingOptions {
    let mut options = TrainingOptions::default();
    options.data.corpus_paths = vec![corpus_path];
    options.data.batch_sequence_count = 4;
    options.data.replay_window_count = 24;
    options.schedule.step_count = TRAINING_STEP_COUNT;
    options.schedule.warmup_step_count = 5;
    options.schedule.peak_learning_rate = 3e-3;
    options.reporting.report_interval_steps = 10;
    options.reporting.validation_interval_steps = 20;
    options.reporting.checkpoint_interval_steps = 20;
    options.reporting.validation_batch_count = 1;
    options
}

fn pretrained_brain(name: &str) -> (Brain, PathBuf) {
    let directory = fresh_directory(name);
    let corpus_path = directory.join("corpus.txt");
    SampleCorpus::built_in()
        .unwrap()
        .write_file(&corpus_path, 60_000, 1)
        .unwrap();

    let brain_directory = directory.join("brain");
    let mut brain = Brain::create(
        &brain_directory,
        miniature_configuration(),
        DeviceChoice::Cpu,
        7,
    )
    .unwrap();
    brain
        .pretrain(
            &training_options(corpus_path),
            &CancellationFlag::new(),
            &mut |_| {},
        )
        .unwrap();
    (brain, brain_directory)
}

#[test]
fn pretraining_reduces_the_loss_and_prepares_the_brain_for_sleep() {
    let directory = fresh_directory("pretrain");
    let corpus_path = directory.join("corpus.txt");
    SampleCorpus::built_in()
        .unwrap()
        .write_file(&corpus_path, 60_000, 1)
        .unwrap();
    let mut brain = Brain::create(
        directory.join("brain"),
        miniature_configuration(),
        DeviceChoice::Cpu,
        7,
    )
    .unwrap();

    let mut training_totals = Vec::new();
    let mut validation_count = 0;
    let summary = brain
        .pretrain(
            &training_options(corpus_path),
            &CancellationFlag::new(),
            &mut |event| match event {
                ProgressEvent::TrainingStep(report) => training_totals.push(report.losses.total),
                ProgressEvent::Validation(_) => validation_count += 1,
                _ => {}
            },
        )
        .unwrap();

    assert_eq!(summary.step_count, TRAINING_STEP_COUNT);
    assert!(!summary.was_cancelled);
    assert_eq!(validation_count, 2);
    assert!(training_totals.last().unwrap() < training_totals.first().unwrap());

    let brain_summary = brain.summary().unwrap();
    assert_eq!(
        brain_summary.state.training.step_count,
        TRAINING_STEP_COUNT as u64
    );
    assert_eq!(brain_summary.replay_window_count, 24);
    assert!(brain_summary.has_elastic_anchor);
    assert!(brain_summary.state.surprise.observation_count > 0);
}

#[test]
fn a_cancelled_run_stops_early_and_keeps_the_model_usable() {
    let directory = fresh_directory("cancel");
    let corpus_path = directory.join("corpus.txt");
    SampleCorpus::built_in()
        .unwrap()
        .write_file(&corpus_path, 60_000, 1)
        .unwrap();
    let mut brain = Brain::create(
        directory.join("brain"),
        miniature_configuration(),
        DeviceChoice::Cpu,
        7,
    )
    .unwrap();
    let cancellation = CancellationFlag::new();
    let stopper = cancellation.clone();

    let summary = brain
        .pretrain(
            &training_options(corpus_path),
            &cancellation,
            &mut |event| {
                if matches!(event, ProgressEvent::TrainingStep(_)) {
                    stopper.request();
                }
            },
        )
        .unwrap();

    assert!(summary.was_cancelled);
    assert_eq!(summary.step_count, 10);
    let reply = brain.generate(
        "The ",
        &GenerationSettings::default(),
        false,
        &CancellationFlag::new(),
        &mut |_| {},
    );
    assert!(reply.bytes.len() <= GenerationSettings::default().maximum_reply_bytes);
}

#[test]
fn text_can_be_patched_round_tripped_and_generated() {
    let (mut brain, _) = pretrained_brain("generate");

    let patches = brain.split_patches("The cat says meow.\n");
    assert_eq!(patches, ["The ", "cat ", "says ", "meow.\n"]);

    let round_trip = brain.round_trip("The cat says meow.\n");
    assert_eq!(round_trip.patches.len(), 4);
    assert!((0.0..=1.0).contains(&round_trip.byte_accuracy));

    let generation = GenerationSettings {
        seed: Some(5),
        maximum_reply_bytes: 60,
        ..GenerationSettings::default()
    };
    let mut streamed = Vec::new();
    let reply = brain.generate(
        "The cat ",
        &generation,
        true,
        &CancellationFlag::new(),
        &mut |bytes| streamed.extend_from_slice(bytes),
    );
    assert_eq!(streamed, reply.bytes);
    assert!(!reply.thoughts.is_empty());
    let first_thought = &reply.thoughts[0];
    assert_eq!(
        first_thought.stages.len(),
        generation.thinking_step_count + 1
    );
    assert_eq!(first_thought.stages.last().unwrap().distance_to_final, 0.0);
    assert!(first_thought.stages[0].distance_to_final > 0.0);

    let repeated = brain.generate(
        "The cat ",
        &generation,
        false,
        &CancellationFlag::new(),
        &mut |_| {},
    );
    assert_eq!(
        repeated.bytes, reply.bytes,
        "a fixed seed must reproduce the same text"
    );
}

#[test]
fn a_correction_is_stored_learned_and_can_be_rolled_back() {
    let (mut brain, _) = pretrained_brain("correction");
    let mut session = brain.start_chat(ChatOptions::default());
    let cancellation = CancellationFlag::new();

    let turn = brain
        .chat_turn(
            &mut session,
            "What does the cat say?",
            &cancellation,
            &mut |_| {},
        )
        .unwrap();
    assert!(turn.surprise.is_some());
    assert!(session.has_exchange());

    let feedback = brain
        .correct_last_reply(&mut session, "The cat says quack.")
        .unwrap();
    let update = feedback
        .live_update
        .expect("a correction should update the fast weights");
    assert!(update.loss_after < update.loss_before, "{update:?}");
    assert!(session.transcript().ends_with("The cat says quack.\n"));

    let memories = brain.memories().unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].kind, MemoryKind::Correction);
    assert_eq!(memories[0].reply, "The cat says quack.\n");

    let found = brain.search_memories("What does the cat say?", 3).unwrap();
    assert_eq!(found[0].entry.id, memories[0].id);

    let snapshots = brain.snapshots().unwrap();
    assert_eq!(
        snapshots.len(),
        1,
        "the update should have been preceded by a snapshot"
    );
    assert!(snapshots[0].is_automatic);
    brain.roll_back(snapshots[0].id).unwrap();
    assert_eq!(brain.state().conversation.live_update_count, 1);
}

#[test]
fn sleep_consolidates_memories_and_resets_the_schedule() {
    let (mut brain, brain_directory) = pretrained_brain("sleep");
    let mut session = brain.start_chat(ChatOptions::default());
    let cancellation = CancellationFlag::new();
    brain
        .chat_turn(
            &mut session,
            "What does the cat say?",
            &cancellation,
            &mut |_| {},
        )
        .unwrap();
    brain
        .correct_last_reply(&mut session, "The cat says quack.")
        .unwrap();
    brain.add_note("Mia likes green pears.", 0.9).unwrap();

    let mut settings = brain.settings().clone();
    settings.sleep.replay_step_count = 12;
    settings.sleep.batch_sequence_count = 4;
    settings.sleep.importance_batch_count = 2;
    settings.sleep.maximum_old_loss_increase = 10.0;
    brain.replace_settings(settings).unwrap();

    let mut sleep_step_count = 0;
    let report = brain
        .sleep(&cancellation, &mut |event| {
            if matches!(event, ProgressEvent::SleepStep(_)) {
                sleep_step_count += 1;
            }
        })
        .unwrap();

    assert_eq!(sleep_step_count, 12);
    assert_eq!(report.replayed_episode_count, 2);
    assert!(!report.was_discarded);
    assert!(
        report.episode_loss_after.unwrap() < report.episode_loss_before.unwrap(),
        "{report:?}"
    );
    assert!(report.old_loss_before.is_some() && report.old_loss_after.is_some());
    assert_eq!(brain.state().sleep.cycle_count, 1);
    assert_eq!(brain.state().conversation.turns_since_sleep, 0);

    drop(brain);
    let reopened = Brain::open(&brain_directory, DeviceChoice::Cpu).unwrap();
    assert_eq!(reopened.state().sleep.cycle_count, 1);
    assert_eq!(
        reopened.memory_count().unwrap(),
        report.remaining_memory_count
    );
    assert!(!reopened.history().unwrap().is_empty());
}

#[test]
fn a_frozen_session_changes_nothing() {
    let (mut brain, _) = pretrained_brain("frozen");
    let options = ChatOptions {
        is_frozen: true,
        ..ChatOptions::default()
    };
    let mut session = brain.start_chat(options);

    brain
        .chat_turn(
            &mut session,
            "No, the cat says quack.",
            &CancellationFlag::new(),
            &mut |_| {},
        )
        .unwrap();
    let feedback = brain.correct_last_reply(&mut session, "Quack.").unwrap();

    assert!(feedback.live_update.is_none());
    assert_eq!(brain.memory_count().unwrap(), 0);
    assert_eq!(brain.state().conversation.live_update_count, 0);
    assert!(brain.snapshots().unwrap().is_empty());
}

#[test]
fn training_can_be_continued_in_a_second_run() {
    let directory = fresh_directory("resume");
    let corpus_path = directory.join("corpus.txt");
    SampleCorpus::built_in()
        .unwrap()
        .write_file(&corpus_path, 60_000, 1)
        .unwrap();
    let brain_directory = directory.join("brain");
    let options = training_options(corpus_path);
    let cancellation = CancellationFlag::new();

    let first_summary = {
        let mut brain = Brain::create(
            &brain_directory,
            miniature_configuration(),
            DeviceChoice::Cpu,
            7,
        )
        .unwrap();
        brain
            .pretrain(&options, &cancellation, &mut |_| {})
            .unwrap()
    };
    let mut reopened = Brain::open(&brain_directory, DeviceChoice::Cpu).unwrap();
    let second_summary = reopened
        .pretrain(&options, &cancellation, &mut |_| {})
        .unwrap();

    assert_eq!(
        reopened.state().training.step_count,
        2 * TRAINING_STEP_COUNT as u64
    );
    let first_loss = first_summary.last_validation.unwrap().losses.total;
    let second_loss = second_summary.last_validation.unwrap().losses.total;
    assert!(
        second_loss < first_loss,
        "the second run should keep improving: {first_loss} then {second_loss}"
    );
}

#[test]
fn sleep_can_shrink_the_fast_weights_instead_of_resetting_them() {
    let (mut brain, _) = pretrained_brain("shrink");
    let mut session = brain.start_chat(ChatOptions::default());
    let cancellation = CancellationFlag::new();
    brain
        .chat_turn(
            &mut session,
            "What does the cat say?",
            &cancellation,
            &mut |_| {},
        )
        .unwrap();
    brain
        .correct_last_reply(&mut session, "The cat says quack.")
        .unwrap();

    let mut settings = brain.settings().clone();
    settings.sleep.replay_step_count = 4;
    settings.sleep.batch_sequence_count = 4;
    settings.sleep.importance_batch_count = 1;
    settings.sleep.maximum_old_loss_increase = 10.0;
    settings.sleep.fast_weight_retention = 0.5;
    brain.replace_settings(settings).unwrap();

    let report = brain.sleep(&cancellation, &mut |_| {}).unwrap();

    assert!(!report.was_discarded);
    let reply = brain.generate(
        "What does the cat say?\n",
        &GenerationSettings::default(),
        false,
        &cancellation,
        &mut |_| {},
    );
    assert!(reply.bytes.len() <= GenerationSettings::default().maximum_reply_bytes);
}

#[test]
fn a_sleep_that_harms_old_skills_is_discarded() {
    let (mut brain, _) = pretrained_brain("discard");
    brain.add_note("Zyx qwv plk jhg fds.", 0.9).unwrap();

    // A tolerance below zero makes any outcome count as harm, which exercises the guard.
    let mut settings = brain.settings().clone();
    settings.sleep.replay_step_count = 4;
    settings.sleep.batch_sequence_count = 4;
    settings.sleep.maximum_old_loss_increase = -1.0;
    brain.replace_settings(settings).unwrap();

    let report = brain.sleep(&CancellationFlag::new(), &mut |_| {}).unwrap();

    assert!(report.was_discarded);
    assert_eq!(report.pruned_memory_count, 0);
    assert_eq!(brain.state().sleep.cycle_count, 0);
    assert_eq!(brain.memory_count().unwrap(), 1);
}
