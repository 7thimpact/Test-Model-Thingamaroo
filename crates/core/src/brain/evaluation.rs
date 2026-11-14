//! Looking inside a brain: patching, round trips, and held-out measurements.

use std::path::PathBuf;

use burn::module::Module;
use serde::{Deserialize, Serialize};

use crate::brain::Brain;
use crate::error::BrainResult;
use crate::learning::{LossBreakdown, measure_batch};
use crate::model::{PatchBatch, latent_deviation, read_scalar, reconstruction_accuracy};
use crate::random::RandomSource;
use crate::text::{Corpus, CorpusSplit, Patcher};

/// One patch before and after its trip through meaning space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchRoundTrip {
    /// The bytes that went into the encoder.
    pub original: String,
    /// The bytes the decoder wrote from the meaning vector.
    pub rebuilt: String,
    /// Whether every original byte came back (an unfinished last word may gain an ending).
    pub is_exact: bool,
}

/// Bytes → patches → meaning vectors → bytes, for a whole text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundTripReport {
    /// Every patch of the text, in order.
    pub patches: Vec<PatchRoundTrip>,
    /// The text reassembled from the rebuilt patches.
    pub rebuilt_text: String,
    /// Share of the original bytes that came back unchanged.
    pub byte_accuracy: f64,
}

/// How well the slow weights handle a text they were not trained on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Average losses over the measured batches.
    pub losses: LossBreakdown,
    /// Share of symbols rebuilt correctly from their own meaning vectors.
    pub reconstruction_accuracy: f64,
    /// Average spread of meaning-vector dimensions; near zero means collapse.
    pub latent_deviation: f64,
    /// Number of batches measured.
    pub batch_count: usize,
}

impl Brain {
    /// Shows how a text is grouped into patches.
    pub fn split_patches(&self, text: &str) -> Vec<String> {
        self.inference
            .patcher()
            .split(text.as_bytes())
            .into_iter()
            .map(|patch| String::from_utf8_lossy(patch).into_owned())
            .collect()
    }

    /// Sends a text through the encoder and back through the decoder.
    ///
    /// The decoder always picks its most likely byte, so the result shows what
    /// the meaning vectors actually preserve.
    pub fn round_trip(&mut self, text: &str) -> RoundTripReport {
        let patcher = self.inference.patcher();
        let original_patches = patcher.split(text.as_bytes());
        let latents = self.inference.encode_patches(&original_patches);

        let mut patches = Vec::with_capacity(original_patches.len());
        let mut matching_bytes = 0;
        let mut rebuilt_bytes = Vec::new();
        for (original, latent) in original_patches.iter().zip(&latents) {
            let rebuilt = self.inference.rebuild_patch(latent);
            matching_bytes += count_matching_bytes(original, &rebuilt);
            patches.push(patch_round_trip(
                original,
                &rebuilt,
                patcher.is_patch_open(original),
            ));
            rebuilt_bytes.extend(rebuilt);
        }

        RoundTripReport {
            patches,
            rebuilt_text: String::from_utf8_lossy(&rebuilt_bytes).into_owned(),
            byte_accuracy: matching_bytes as f64 / text.len().max(1) as f64,
        }
    }

    /// Measures the slow weights on text files, without changing anything.
    pub fn evaluate(
        &self,
        corpus_paths: &[PathBuf],
        batch_count: usize,
        batch_sequence_count: usize,
        seed: u64,
    ) -> BrainResult<EvaluationReport> {
        let patcher = Patcher::new(self.model_configuration.patching.maximum_patch_bytes);
        // The whole corpus is evaluation data here, so nothing is held out.
        let corpus = Corpus::open(corpus_paths, 0.0, patcher)?;
        let frozen_model = self.slow_model.valid();
        let mut random = RandomSource::from_seed(seed);
        let batch_count = batch_count.max(1);

        let mut summed_losses = LossBreakdown::default();
        let mut summed_accuracy = 0.0;
        let mut summed_deviation = 0.0;
        for _ in 0..batch_count {
            let batch = self.sample_evaluation_batch(&corpus, &mut random, batch_sequence_count)?;
            let latents = frozen_model
                .encoder
                .forward(batch.bytes.clone(), batch.byte_validity.clone());
            summed_accuracy += read_scalar(reconstruction_accuracy(
                &frozen_model.decoder,
                latents.clone(),
                &batch,
            ));
            summed_deviation += read_scalar(latent_deviation(latents));
            summed_losses.accumulate(&measure_batch(
                &frozen_model,
                batch,
                &self.model_configuration,
                &self.settings.objective,
            ));
        }

        Ok(EvaluationReport {
            losses: summed_losses.divided_by(batch_count),
            reconstruction_accuracy: summed_accuracy / batch_count as f64,
            latent_deviation: summed_deviation / batch_count as f64,
            batch_count,
        })
    }

    fn sample_evaluation_batch(
        &self,
        corpus: &Corpus,
        random: &mut RandomSource,
        batch_sequence_count: usize,
    ) -> BrainResult<PatchBatch> {
        let patch_count = self.model_configuration.predictor.maximum_context_patches + 1;
        let sequences = (0..batch_sequence_count.max(1))
            .map(|_| corpus.sample_patches(random, CorpusSplit::Training, patch_count))
            .collect::<BrainResult<Vec<_>>>()?;
        Ok(PatchBatch::from_sequences(
            &sequences,
            self.model_configuration.patching.maximum_patch_bytes,
            &self.compute_device.inference,
        ))
    }

    /// The meaning vector of a text, as stored in memory and used for retrieval.
    pub fn gist(&mut self, text: &str) -> Option<Vec<f32>> {
        self.inference.gist(text.as_bytes())
    }

    /// The raw latent values of the first patches, for inspection.
    pub fn latent_preview(&mut self, text: &str, value_count: usize) -> Vec<Vec<f32>> {
        let patcher = self.inference.patcher();
        let patches = patcher.split(text.as_bytes());
        self.inference
            .encode_patches(&patches)
            .into_iter()
            .map(|latent| latent.into_iter().take(value_count).collect())
            .collect()
    }
}

/// Compares a patch with what the decoder rebuilt from its meaning vector.
///
/// An open patch is an unfinished word at the end of the text. The decoder was
/// only ever trained on finished words, so it finishes the word (usually with
/// a space); that still counts as exact as long as every original byte is there.
fn patch_round_trip(original: &[u8], rebuilt: &[u8], is_original_open: bool) -> PatchRoundTrip {
    let is_exact = match is_original_open {
        true => rebuilt.starts_with(original),
        false => original == rebuilt,
    };
    PatchRoundTrip {
        original: String::from_utf8_lossy(original).into_owned(),
        rebuilt: String::from_utf8_lossy(rebuilt).into_owned(),
        is_exact,
    }
}

/// Number of positions at which the two byte strings agree.
fn count_matching_bytes(original: &[u8], rebuilt: &[u8]) -> usize {
    original
        .iter()
        .zip(rebuilt)
        .filter(|(original_byte, rebuilt_byte)| original_byte == rebuilt_byte)
        .count()
}
