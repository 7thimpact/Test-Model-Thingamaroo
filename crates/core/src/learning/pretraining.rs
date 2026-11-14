//! Pretraining: the whole stack learns from a text corpus.
//!
//! Encoder, predictor and decoder train together on two signals: predicting
//! the next meaning vector in latent space, and rebuilding bytes through the
//! decoder. Collapse is prevented three ways at once: the reconstruction loss,
//! targets from a slowly updated encoder copy, and a variance penalty.

use burn::module::Module;
use burn::optim::ModuleOptimizer;
use burn::tensor::{Device, Tensor};

use crate::configuration::{ModelConfiguration, ObjectiveSettings, TrainingOptions};
use crate::error::{BrainError, BrainResult};
use crate::learning::learning_rate::LearningRateSchedule;
use crate::learning::progress::LossBreakdown;
use crate::learning::target_encoder::TargetEncoder;
use crate::learning::trainable_module::TrainableModule;
use crate::model::{
    ByteEncoder, LanguageModel, NextPatchInput, PatchBatch, WeightedObjective, evaluate_next_patch,
    read_scalar, real_patch_selection, reconstruction_loss, variance_penalty,
};
use crate::random::RandomSource;
use crate::text::{Corpus, CorpusSplit, PatchSequence};

/// One batch of training text, with optional gists of the text that preceded it.
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// The patches the model learns to predict and rebuild.
    pub batch: PatchBatch,
    /// Gist vectors of earlier text, `[sequence, slot, latent_width]`.
    pub memories: Option<Tensor<3>>,
}

/// Everything a pretraining run is built from.
pub struct PretrainerParts {
    /// The model to train, on a gradient-tracking device.
    pub model: LanguageModel,
    /// The optimiser, possibly resumed from an earlier run.
    pub optimizer: ModuleOptimizer,
    /// The target encoder, possibly resumed from an earlier run.
    pub target_encoder: TargetEncoder,
    /// The training text.
    pub corpus: Corpus,
    /// The run's options.
    pub options: TrainingOptions,
    /// Loss weights.
    pub objective: ObjectiveSettings,
    /// The model's sizes.
    pub model_configuration: ModelConfiguration,
    /// Device input tensors are created on.
    pub device: Device,
}

/// Runs pretraining steps and validation measurements.
///
/// Owns copies of its configuration so a run can outlive borrows of the brain
/// that started it.
pub struct Pretrainer {
    trainable: TrainableModule<LanguageModel>,
    target_encoder: TargetEncoder,
    corpus: Corpus,
    random: RandomSource,
    schedule: LearningRateSchedule,
    options: TrainingOptions,
    objective: ObjectiveSettings,
    model_configuration: ModelConfiguration,
    device: Device,
}

impl Pretrainer {
    /// Assembles a run from its parts.
    pub fn new(parts: PretrainerParts) -> Self {
        Self {
            trainable: TrainableModule::new(parts.model, parts.optimizer),
            target_encoder: parts.target_encoder,
            corpus: parts.corpus,
            random: RandomSource::from_seed(parts.options.data.seed),
            schedule: LearningRateSchedule::new(&parts.options.schedule),
            options: parts.options,
            objective: parts.objective,
            model_configuration: parts.model_configuration,
            device: parts.device,
        }
    }

    /// The model in its current state.
    pub fn model(&self) -> &LanguageModel {
        self.trainable.network()
    }

    /// The training text.
    pub fn corpus(&self) -> &Corpus {
        &self.corpus
    }

    /// The run's random source, shared so later sampling continues the same sequence.
    pub fn random(&mut self) -> &mut RandomSource {
        &mut self.random
    }

    /// Patches processed by one training step.
    pub fn patches_per_step(&self) -> usize {
        self.options.data.batch_sequence_count * self.sequence_patch_count()
    }

    /// The learning rate for a zero-based step of this run.
    pub fn learning_rate_at(&self, run_step: usize) -> f64 {
        self.schedule.rate_at(run_step)
    }

    /// Takes one gradient step and returns the losses on its batch.
    pub fn train_step(&mut self, run_step: usize) -> BrainResult<LossBreakdown> {
        let training_batch = self.sample_training_batch()?;
        let objective = pretraining_objective(
            self.trainable.network(),
            self.target_encoder.encoder(),
            training_batch,
            &self.model_configuration,
        );
        let total = objective.total(&self.objective);
        let losses = read_losses(&objective, total.clone());
        if !losses.is_finite() {
            return Err(BrainError::Diverged(format!(
                "pretraining produced a non-finite loss at step {}; try a lower learning rate",
                run_step + 1
            )));
        }

        self.trainable
            .descend(total, self.schedule.rate_at(run_step));
        self.target_encoder
            .follow(&self.trainable.network().encoder);
        Ok(losses)
    }

    /// Measures the losses on held-out text, without gradients.
    pub fn validate(&mut self) -> BrainResult<LossBreakdown> {
        let frozen_model = self.trainable.network().valid();
        let batch_count = self.options.reporting.validation_batch_count.max(1);
        let mut summed_losses = LossBreakdown::default();
        for _ in 0..batch_count {
            let batch = self.sample_batch(CorpusSplit::Validation)?;
            summed_losses.accumulate(&measure_batch(
                &frozen_model,
                batch,
                &self.model_configuration,
                &self.objective,
            ));
        }
        Ok(summed_losses.divided_by(batch_count))
    }

    /// Ends the run and hands back everything worth saving.
    pub fn into_parts(self) -> (LanguageModel, ModuleOptimizer, TargetEncoder) {
        let (model, optimizer) = self.trainable.into_parts();
        (model, optimizer, self.target_encoder)
    }

    /// One more patch than the working memory holds: the last one is only predicted.
    fn sequence_patch_count(&self) -> usize {
        self.model_configuration.predictor.maximum_context_patches + 1
    }

    fn sample_batch(&mut self, split: CorpusSplit) -> BrainResult<PatchBatch> {
        let patch_count = self.sequence_patch_count();
        let sequences = self.sample_sequences(split, patch_count)?;
        Ok(self.batch_of(&sequences))
    }

    fn sample_sequences(
        &mut self,
        split: CorpusSplit,
        patch_count: usize,
    ) -> BrainResult<Vec<PatchSequence>> {
        (0..self.options.data.batch_sequence_count)
            .map(|_| {
                self.corpus
                    .sample_patches(&mut self.random, split, patch_count)
            })
            .collect()
    }

    fn batch_of(&self, sequences: &[PatchSequence]) -> PatchBatch {
        PatchBatch::from_sequences(
            sequences,
            self.model_configuration.patching.maximum_patch_bytes,
            &self.device,
        )
    }

    /// Samples a training batch, sometimes preceded by text that is shown only as gists.
    ///
    /// This teaches the predictor to use memory slots: the gists summarise what
    /// came just before the context, exactly as retrieved memories summarise
    /// related past exchanges.
    fn sample_training_batch(&mut self) -> BrainResult<TrainingBatch> {
        let slot_count = self.sample_memory_slot_count();
        let memory_patch_count = slot_count * self.objective.memory_span_patches;
        let sequences = self.sample_sequences(
            CorpusSplit::Training,
            memory_patch_count + self.sequence_patch_count(),
        )?;

        let (memory_sequences, context_sequences): (Vec<PatchSequence>, Vec<PatchSequence>) =
            sequences
                .into_iter()
                .map(|mut patches| {
                    let context_patches = patches.split_off(memory_patch_count);
                    (patches, context_patches)
                })
                .unzip();

        let memories =
            (slot_count > 0).then(|| self.summarise_as_gists(&memory_sequences, slot_count));
        Ok(TrainingBatch {
            batch: self.batch_of(&context_sequences),
            memories,
        })
    }

    fn sample_memory_slot_count(&mut self) -> usize {
        let maximum_slot_count = self.model_configuration.predictor.maximum_memory_slots;
        let is_showing_memories = maximum_slot_count > 0
            && self.objective.memory_span_patches > 0
            && self
                .random
                .next_chance(self.objective.memory_slot_probability);
        match is_showing_memories {
            true => 1 + self.random.next_below(maximum_slot_count),
            false => 0,
        }
    }

    /// Averages each span of patches into one gist vector, `[sequence, slot, latent_width]`.
    fn summarise_as_gists(
        &self,
        memory_sequences: &[PatchSequence],
        slot_count: usize,
    ) -> Tensor<3> {
        let batch = self.batch_of(memory_sequences);
        let latent_width = self.model_configuration.latent.width;
        let span_patches = self.objective.memory_span_patches;
        self.target_encoder
            .encoder()
            .forward(batch.bytes, batch.byte_validity)
            .reshape([batch.sequence_count, slot_count, span_patches, latent_width])
            .mean_dim(2)
            .reshape([batch.sequence_count, slot_count, latent_width])
    }
}

/// The four pretraining losses of a model on one batch.
///
/// `target_encoder` supplies the meaning vectors the predictions are compared
/// with; it receives no gradient.
pub fn pretraining_objective(
    model: &LanguageModel,
    target_encoder: &ByteEncoder,
    training_batch: TrainingBatch,
    model_configuration: &ModelConfiguration,
) -> WeightedObjective {
    let batch = &training_batch.batch;
    let sequence_shape = [
        batch.sequence_count,
        batch.patch_count,
        model_configuration.latent.width,
    ];
    let latents = model
        .encoder
        .forward(batch.bytes.clone(), batch.byte_validity.clone());
    let target_latents = target_encoder
        .forward(batch.bytes.clone(), batch.byte_validity.clone())
        .detach()
        .reshape(sequence_shape);

    let evaluation = evaluate_next_patch(
        &model.predictor,
        &model.decoder,
        NextPatchInput {
            batch,
            context_latents: latents.clone().reshape(sequence_shape),
            target_latents,
            memories: training_batch.memories,
            thinking_step_count: model_configuration.predictor.thinking_step_count,
        },
    );
    let selection = real_patch_selection(batch);
    WeightedObjective {
        reconstruction: reconstruction_loss(&model.decoder, latents.clone(), batch),
        generation: evaluation.generation_loss(selection.clone()),
        latent: evaluation.latent_loss(selection),
        variance: variance_penalty(latents),
    }
}

/// Measures a gradient-free model on one batch, using its own encoder for targets.
pub fn measure_batch(
    frozen_model: &LanguageModel,
    batch: PatchBatch,
    model_configuration: &ModelConfiguration,
    objective_settings: &ObjectiveSettings,
) -> LossBreakdown {
    let training_batch = TrainingBatch {
        batch,
        memories: None,
    };
    let objective = pretraining_objective(
        frozen_model,
        &frozen_model.encoder,
        training_batch,
        model_configuration,
    );
    let total = objective.total(objective_settings);
    read_losses(&objective, total)
}

fn read_losses(objective: &WeightedObjective, total: Tensor<1>) -> LossBreakdown {
    LossBreakdown {
        total: read_scalar(total),
        reconstruction: read_scalar(objective.reconstruction.clone()),
        generation: read_scalar(objective.generation.clone()),
        latent: read_scalar(objective.latent.clone()),
        variance: read_scalar(objective.variance.clone()),
    }
}
