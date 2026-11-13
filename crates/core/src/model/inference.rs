//! Frozen copies of the model used to read, think and write without gradients.

use std::collections::HashMap;

use burn::tensor::{Device, Int, Tensor, TensorData, s};

use crate::configuration::{END_OF_PATCH_SYMBOL, ModelConfiguration};
use crate::model::byte_decoder::ByteDecoder;
use crate::model::byte_encoder::ByteEncoder;
use crate::model::byte_sampler::ByteSampler;
use crate::model::concept_predictor::ConceptPredictor;
use crate::model::objective::{NATS_PER_BIT, NextPatchInput, evaluate_next_patch};
use crate::model::patch_batch::PatchBatch;
use crate::model::tensor_readback::{read_scalar, read_values};
use crate::text::Patcher;

/// Cached meaning vectors are dropped once the cache holds this many patches.
const LATENT_CACHE_CAPACITY: usize = 50_000;

/// How badly a stretch of text was predicted from what came before it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionError {
    /// Bits the decoder needed per symbol when writing from predicted meanings.
    pub bits_per_byte: f64,
    /// Squared distance between predicted and actual meaning vectors.
    pub latent_error: f64,
    /// Number of patches the measurement covers.
    pub measured_patch_count: usize,
}

/// Encoder, predictor and decoder with gradient tracking switched off.
#[derive(Debug, Clone)]
pub struct InferenceModel {
    encoder: ByteEncoder,
    predictor: ConceptPredictor,
    decoder: ByteDecoder,
    device: Device,
    patcher: Patcher,
    latent_width: usize,
    maximum_context_patches: usize,
    maximum_memory_slots: usize,
    latent_cache: HashMap<Vec<u8>, Vec<f32>>,
}

impl InferenceModel {
    /// Bundles gradient-free copies of the three components.
    pub fn new(
        encoder: ByteEncoder,
        predictor: ConceptPredictor,
        decoder: ByteDecoder,
        model_configuration: &ModelConfiguration,
        device: &Device,
    ) -> Self {
        Self {
            encoder,
            predictor,
            decoder,
            device: device.clone(),
            patcher: Patcher::new(model_configuration.patching.maximum_patch_bytes),
            latent_width: model_configuration.latent.width,
            maximum_context_patches: model_configuration.predictor.maximum_context_patches,
            maximum_memory_slots: model_configuration.predictor.maximum_memory_slots,
            latent_cache: HashMap::new(),
        }
    }

    /// The patcher that matches this model's patch size.
    pub fn patcher(&self) -> Patcher {
        self.patcher
    }

    /// How many patches the predictor can attend to at once.
    pub fn maximum_context_patches(&self) -> usize {
        self.maximum_context_patches
    }

    /// The frozen byte encoder.
    pub fn encoder(&self) -> &ByteEncoder {
        &self.encoder
    }

    /// The frozen byte decoder.
    pub fn decoder(&self) -> &ByteDecoder {
        &self.decoder
    }

    /// The gradient-free predictor, including any fast weights.
    pub fn predictor(&self) -> &ConceptPredictor {
        &self.predictor
    }

    /// Replaces the predictor, for example after its fast weights changed.
    pub fn replace_predictor(&mut self, predictor: ConceptPredictor) {
        self.predictor = predictor;
    }

    /// Encodes patches into meaning vectors, reusing earlier results for repeated patches.
    pub fn encode_patches(&mut self, patches: &[&[u8]]) -> Vec<Vec<f32>> {
        let mut missing_patches: Vec<&[u8]> = patches
            .iter()
            .copied()
            .filter(|patch| !self.latent_cache.contains_key(*patch))
            .collect();
        missing_patches.sort_unstable();
        missing_patches.dedup();
        if !missing_patches.is_empty() {
            self.encode_into_cache(&missing_patches);
        }

        patches
            .iter()
            .map(|patch| self.latent_cache[*patch].clone())
            .collect()
    }

    fn encode_into_cache(&mut self, patches: &[&[u8]]) {
        if self.latent_cache.len() + patches.len() > LATENT_CACHE_CAPACITY {
            self.latent_cache.clear();
        }
        let batch = PatchBatch::from_sequences(
            &[patches.to_vec()],
            self.patcher.maximum_patch_bytes(),
            &self.device,
        );
        let latents = read_values(self.encoder.forward(batch.bytes, batch.byte_validity));
        for (patch, latent) in patches.iter().zip(latents.chunks(self.latent_width)) {
            self.latent_cache.insert(patch.to_vec(), latent.to_vec());
        }
    }

    /// Summarises a text as the average of its patches' meaning vectors.
    ///
    /// Returns `None` for empty text, which has no meaning to summarise.
    pub fn gist(&mut self, text: &[u8]) -> Option<Vec<f32>> {
        let patcher = self.patcher;
        let patches = patcher.split(text);
        if patches.is_empty() {
            return None;
        }
        let latents = self.encode_patches(&patches);
        Some(average_vectors(&latents, self.latent_width))
    }

    /// Predicts the meaning of the patch that follows the context.
    ///
    /// Returns one `[1, latent_width]` tensor per thinking stage, coarse to fine.
    /// `context_latents` must not be empty.
    pub fn predict_next(
        &self,
        context_latents: &[Vec<f32>],
        memories: &[Vec<f32>],
        thinking_step_count: usize,
    ) -> Vec<Tensor<2>> {
        let patch_count = context_latents.len();
        let memories = &memories[..memories.len().min(self.maximum_memory_slots)];
        let memory_tensor = (!memories.is_empty()).then(|| self.stack_vectors(memories));

        self.predictor
            .forward(
                self.stack_vectors(context_latents),
                memory_tensor,
                thinking_step_count,
            )
            .into_iter()
            .map(|stage| {
                stage
                    .slice(s![.., patch_count - 1..patch_count])
                    .reshape([1, self.latent_width])
            })
            .collect()
    }

    /// Stacks host vectors into a `[1, vector, latent_width]` tensor.
    fn stack_vectors(&self, vectors: &[Vec<f32>]) -> Tensor<3> {
        let values: Vec<f32> = vectors.iter().flatten().copied().collect();
        Tensor::from_data(
            TensorData::new(values, [1, vectors.len(), self.latent_width]),
            &self.device,
        )
    }

    /// Writes the bytes of one patch from its meaning vector.
    ///
    /// `forced_prefix` holds bytes the patch must start with: an unfinished
    /// word at the end of the prompt. The returned patch includes the prefix.
    pub fn write_patch(
        &self,
        meaning: Tensor<2>,
        forced_prefix: &[u8],
        sampler: &mut ByteSampler,
    ) -> Vec<u8> {
        let mut written = forced_prefix.to_vec();
        while written.len() < self.patcher.maximum_patch_bytes() {
            let symbol = sampler.choose(&self.next_symbol_scores(meaning.clone(), &written));
            if symbol == END_OF_PATCH_SYMBOL {
                break;
            }
            written.push(symbol as u8);
        }
        written
    }

    /// Writes the most likely bytes for a meaning vector held in host memory.
    pub fn rebuild_patch(&self, latent: &[f32]) -> Vec<u8> {
        let meaning = Tensor::from_data(
            TensorData::new(latent.to_vec(), [1, self.latent_width]),
            &self.device,
        );
        self.write_patch(meaning, &[], &mut ByteSampler::greedy())
    }

    /// Scores every symbol that could follow the bytes written so far.
    fn next_symbol_scores(&self, meaning: Tensor<2>, written: &[u8]) -> Vec<f32> {
        let previous_bytes = (!written.is_empty()).then(|| self.byte_row(written));
        let scores = self.decoder.forward(meaning, previous_bytes);
        let [_, position_count, _] = scores.dims();
        read_values(scores.slice(s![.., position_count - 1..position_count]))
    }

    fn byte_row(&self, bytes: &[u8]) -> Tensor<2, Int> {
        let values: Vec<i32> = bytes.iter().map(|&byte| i32::from(byte)).collect();
        Tensor::from_data(TensorData::new(values, [1, bytes.len()]), &self.device)
    }

    /// Measures how badly the patches from `first_measured_patch` onwards were predicted.
    ///
    /// Returns `None` when there is no earlier patch to predict from.
    pub fn measure_prediction_error(
        &mut self,
        patches: &[&[u8]],
        first_measured_patch: usize,
        thinking_step_count: usize,
    ) -> Option<PredictionError> {
        let window_start = patches.len().saturating_sub(self.maximum_context_patches);
        let window = &patches[window_start..];
        // The first patch of the window has nothing before it, so it cannot be measured.
        let first_measured = first_measured_patch.saturating_sub(window_start).max(1);
        if first_measured >= window.len() {
            return None;
        }

        let batch = PatchBatch::from_sequences(
            &[window.to_vec()],
            self.patcher.maximum_patch_bytes(),
            &self.device,
        );
        let latents = self
            .encoder
            .forward(batch.bytes.clone(), batch.byte_validity.clone())
            .reshape([1, window.len(), self.latent_width]);
        let evaluation = evaluate_next_patch(
            &self.predictor,
            &self.decoder,
            NextPatchInput {
                batch: &batch,
                context_latents: latents.clone(),
                target_latents: latents,
                memories: None,
                thinking_step_count,
            },
        );

        let selection = self.selection_from(first_measured, window.len());
        Some(PredictionError {
            bits_per_byte: read_scalar(evaluation.generation_loss(selection.clone()))
                / NATS_PER_BIT,
            latent_error: read_scalar(evaluation.latent_loss(selection)),
            measured_patch_count: window.len() - first_measured,
        })
    }

    /// Selects the predictions of patches `first_measured..patch_count`.
    fn selection_from(&self, first_measured: usize, patch_count: usize) -> Tensor<2> {
        let values: Vec<f32> = (1..patch_count)
            .map(|patch_index| {
                if patch_index >= first_measured {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Tensor::from_data(TensorData::new(values, [1, patch_count - 1]), &self.device)
    }
}

/// Element-wise mean of equally sized vectors.
pub fn average_vectors(vectors: &[Vec<f32>], width: usize) -> Vec<f32> {
    let mut average = vec![0.0f32; width];
    for vector in vectors {
        for (sum, value) in average.iter_mut().zip(vector) {
            *sum += value;
        }
    }
    let vector_count = vectors.len().max(1) as f32;
    average.iter_mut().for_each(|sum| *sum /= vector_count);
    average
}
