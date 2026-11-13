//! Checks that the model's pieces fit together and can learn.

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{Distribution, Int, Tensor, TensorData, s};

use thingamaroo_core::configuration::{ModelPreset, ObjectiveSettings};
use thingamaroo_core::model::{
    ComputeDevice, DeviceChoice, LanguageModel, NextPatchInput, PatchBatch, WeightedObjective,
    evaluate_next_patch, read_scalar, read_values, real_patch_selection, reconstruction_loss,
    variance_penalty,
};
use thingamaroo_core::text::Patcher;

const SAMPLE_LINES: [&str; 4] = [
    "The cat says meow. The dog says woof. ",
    "Mia has a red ball. Mia likes the red ball. ",
    "The cow lives in the barn. The owl lives in the tree. ",
    "Where is the duck? The duck is in the pond. ",
];

fn sample_batch(patcher: Patcher, compute_device: &ComputeDevice) -> PatchBatch {
    let sequences: Vec<Vec<&[u8]>> = SAMPLE_LINES
        .iter()
        .map(|line| patcher.split(line.as_bytes()))
        .collect();
    PatchBatch::from_sequences(
        &sequences,
        patcher.maximum_patch_bytes(),
        &compute_device.inference,
    )
}

fn pretraining_loss(model: &LanguageModel, batch: &PatchBatch, latent_width: usize) -> Tensor<1> {
    let latents = model
        .encoder
        .forward(batch.bytes.clone(), batch.byte_validity.clone());
    let context = latents
        .clone()
        .reshape([batch.sequence_count, batch.patch_count, latent_width]);
    let evaluation = evaluate_next_patch(
        &model.predictor,
        &model.decoder,
        NextPatchInput {
            batch,
            context_latents: context.clone(),
            target_latents: context.detach(),
            memories: None,
            thinking_step_count: 2,
        },
    );
    let selection = real_patch_selection(batch);
    WeightedObjective {
        reconstruction: reconstruction_loss(&model.decoder, latents.clone(), batch),
        generation: evaluation.generation_loss(selection.clone()),
        latent: evaluation.latent_loss(selection),
        variance: variance_penalty(latents),
    }
    .total(&ObjectiveSettings::default())
}

#[test]
fn analytic_parameter_counts_match_the_built_model() {
    let model_configuration = ModelPreset::Tiny.configuration();
    let compute_device = ComputeDevice::select(DeviceChoice::Cpu).unwrap();

    let model = LanguageModel::new(&model_configuration, &compute_device.inference);
    let counts = model_configuration.parameter_counts();

    assert_eq!(model.encoder.num_params(), counts.encoder);
    assert_eq!(model.predictor.num_params(), counts.predictor);
    assert_eq!(model.decoder.num_params(), counts.decoder);
}

#[test]
fn a_few_gradient_steps_reduce_the_loss_on_a_fixed_batch() {
    let model_configuration = ModelPreset::Tiny.configuration();
    let compute_device = ComputeDevice::select(DeviceChoice::Cpu).unwrap();
    compute_device.training.seed(3);
    let patcher = Patcher::new(model_configuration.patching.maximum_patch_bytes);
    let batch = sample_batch(patcher, &compute_device);
    let latent_width = model_configuration.latent.width;

    let mut model = LanguageModel::new(&model_configuration, &compute_device.training);
    let mut optimizer = AdamWConfig::new().init();
    let first_loss = read_scalar(pretraining_loss(&model, &batch, latent_width));
    for _ in 0..30 {
        let loss = pretraining_loss(&model, &batch, latent_width);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(2e-3, model, gradients);
    }
    let last_loss = read_scalar(pretraining_loss(&model, &batch, latent_width));

    assert!(first_loss.is_finite() && last_loss.is_finite());
    assert!(
        last_loss < first_loss * 0.6,
        "loss went from {first_loss} to {last_loss}"
    );
}

/// Largest difference two models' outputs may show when they should be identical.
const IDENTICAL_OUTPUT_TOLERANCE: f32 = 1e-5;

fn largest_difference<const RANK: usize>(left: Tensor<RANK>, right: Tensor<RANK>) -> f32 {
    read_values((left - right).abs())
        .into_iter()
        .fold(0.0, f32::max)
}

#[test]
fn the_predictor_cannot_see_the_patch_it_is_predicting() {
    let model_configuration = ModelPreset::Tiny.configuration();
    let compute_device = ComputeDevice::select(DeviceChoice::Cpu).unwrap();
    let device = &compute_device.inference;
    let model = LanguageModel::new(&model_configuration, device);
    let latent_width = model_configuration.latent.width;
    let patch_count = 6;
    let changed_from = 4;

    let context = Tensor::<3>::random(
        [1, patch_count, latent_width],
        Distribution::Default,
        device,
    );
    let different_future = Tensor::<3>::random(
        [1, patch_count - changed_from, latent_width],
        Distribution::Default,
        device,
    );
    let altered_context = context
        .clone()
        .slice_assign(s![.., changed_from..patch_count], different_future);

    for thinking_step_count in [0, 2] {
        let original = model
            .predictor
            .forward(context.clone(), None, thinking_step_count)
            .pop()
            .unwrap();
        let altered = model
            .predictor
            .forward(altered_context.clone(), None, thinking_step_count)
            .pop()
            .unwrap();

        let unchanged_past = largest_difference(
            original.clone().slice(s![.., 0..changed_from]),
            altered.clone().slice(s![.., 0..changed_from]),
        );
        let changed_future = largest_difference(
            original.slice(s![.., changed_from..patch_count]),
            altered.slice(s![.., changed_from..patch_count]),
        );
        assert!(
            unchanged_past < IDENTICAL_OUTPUT_TOLERANCE,
            "later patches leaked backwards: {unchanged_past}"
        );
        assert!(
            changed_future > IDENTICAL_OUTPUT_TOLERANCE,
            "the altered patches had no effect at all"
        );
    }
}

#[test]
fn the_decoder_cannot_see_the_bytes_it_is_about_to_write() {
    let model_configuration = ModelPreset::Tiny.configuration();
    let compute_device = ComputeDevice::select(DeviceChoice::Cpu).unwrap();
    let device = &compute_device.inference;
    let model = LanguageModel::new(&model_configuration, device);
    let latents = Tensor::<2>::random(
        [1, model_configuration.latent.width],
        Distribution::Default,
        device,
    );
    let byte_row = |bytes: [i32; 5]| {
        Tensor::<2, Int>::from_data(TensorData::new(bytes.to_vec(), [1, 5]), device)
    };
    let changed_from = 3;

    let original = model
        .decoder
        .forward(latents.clone(), Some(byte_row([84, 104, 101, 32, 99])));
    let altered = model
        .decoder
        .forward(latents, Some(byte_row([84, 104, 101, 90, 90])));

    // Scores at position `i` are for the byte after `i` written bytes, so the
    // first `changed_from + 1` positions depend only on unchanged bytes.
    let unchanged_past = largest_difference(
        original.clone().slice(s![.., 0..=changed_from]),
        altered.clone().slice(s![.., 0..=changed_from]),
    );
    let changed_future = largest_difference(
        original.slice(s![.., changed_from + 1..6]),
        altered.slice(s![.., changed_from + 1..6]),
    );
    assert!(
        unchanged_past < IDENTICAL_OUTPUT_TOLERANCE,
        "later bytes leaked backwards: {unchanged_past}"
    );
    assert!(
        changed_future > IDENTICAL_OUTPUT_TOLERANCE,
        "the altered bytes had no effect at all"
    );
}

#[test]
fn memory_slots_change_the_prediction() {
    let model_configuration = ModelPreset::Tiny.configuration();
    let compute_device = ComputeDevice::select(DeviceChoice::Cpu).unwrap();
    let device = &compute_device.inference;
    let model = LanguageModel::new(&model_configuration, device);
    let latent_width = model_configuration.latent.width;
    let context = Tensor::<3>::random([1, 4, latent_width], Distribution::Default, device);
    let memories = Tensor::<3>::random([1, 2, latent_width], Distribution::Default, device);

    let without_memories = model
        .predictor
        .forward(context.clone(), None, 1)
        .pop()
        .unwrap();
    let with_memories = model
        .predictor
        .forward(context, Some(memories), 1)
        .pop()
        .unwrap();

    assert_eq!(with_memories.dims(), without_memories.dims());
    assert!(largest_difference(with_memories, without_memories) > IDENTICAL_OUTPUT_TOLERANCE);
}
