//! The training losses, shared by pretraining, live learning, sleep and surprise.
//!
//! Three signals, as in the spec sheet:
//!
//! - **Latent prediction**: the predicted next meaning vector should match the
//!   meaning vector the encoder gives the patch that actually came next.
//! - **Byte reconstruction**: the decoder must be able to rebuild a patch from
//!   its meaning vector (and write the next patch from a predicted one). This
//!   forces meaning vectors to carry real information.
//! - **Variance penalty**: meaning vectors must stay spread out across a batch,
//!   so the model cannot cheat by mapping everything to one point.

use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor, s};

use crate::configuration::{DECODER_SYMBOL_COUNT, ObjectiveSettings};
use crate::model::byte_decoder::ByteDecoder;
use crate::model::concept_predictor::ConceptPredictor;
use crate::model::patch_batch::PatchBatch;

/// Spread every latent dimension should keep across a batch (as in VICReg).
const TARGET_LATENT_DEVIATION: f64 = 1.0;

/// Keeps the square root in the deviation finite when a dimension is constant.
const DEVIATION_EPSILON: f64 = 1e-4;

/// Natural-log units per bit, for reporting losses as bits.
pub const NATS_PER_BIT: f64 = std::f64::consts::LN_2;

/// How well each patch was predicted from the patches before it.
///
/// Every tensor is `[sequence, patch - 1]`: entry `i` describes the prediction
/// of patch `i + 1`. Nothing here is masked yet; callers choose which
/// predictions count.
#[derive(Debug, Clone)]
pub struct NextPatchEvaluation {
    /// Summed negative log-likelihood of each patch's bytes, in nats.
    pub generation_nats: Tensor<2>,
    /// Number of symbols (bytes plus the end marker) in each patch.
    pub symbol_counts: Tensor<2>,
    /// Squared distance between predicted and actual meaning, averaged over
    /// latent dimensions and thinking stages.
    pub latent_errors: Tensor<2>,
}

/// What the predictor and decoder are asked to explain.
pub struct NextPatchInput<'input> {
    /// The patches being predicted.
    pub batch: &'input PatchBatch,
    /// Meaning vectors the predictor reads, `[sequence, patch, latent_width]`.
    pub context_latents: Tensor<3>,
    /// Meaning vectors the predictions are compared with; same shape, no gradient.
    pub target_latents: Tensor<3>,
    /// Gist vectors shown to the predictor, `[sequence, slot, latent_width]`.
    pub memories: Option<Tensor<3>>,
    /// Internal refinement steps to run.
    pub thinking_step_count: usize,
}

/// Predicts every patch from the patches before it and scores the result.
pub fn evaluate_next_patch(
    predictor: &ConceptPredictor,
    decoder: &ByteDecoder,
    input: NextPatchInput<'_>,
) -> NextPatchEvaluation {
    let [sequence_count, patch_count, latent_width] = input.context_latents.dims();
    let predicted_count = patch_count - 1;

    let guesses = predictor.forward(
        input.context_latents.slice(s![.., 0..predicted_count]),
        input.memories,
        input.thinking_step_count,
    );
    let targets = input.target_latents.slice(s![.., 1..patch_count]);
    let latent_errors = mean_latent_error(&guesses, targets);

    let final_guess = guesses[guesses.len() - 1]
        .clone()
        .reshape([sequence_count * predicted_count, latent_width]);
    let following = FollowingPatches::of(input.batch);
    let symbol_nats = symbol_negative_log_likelihood(
        decoder,
        final_guess,
        following.bytes,
        following.target_symbols,
        following.target_validity.clone(),
    );

    NextPatchEvaluation {
        generation_nats: sum_per_patch(symbol_nats, sequence_count, predicted_count),
        symbol_counts: sum_per_patch(following.target_validity, sequence_count, predicted_count),
        latent_errors,
    }
}

impl NextPatchEvaluation {
    /// Average negative log-likelihood per symbol over the selected predictions, in nats.
    ///
    /// `selection` is `[sequence, patch - 1]` with one for predictions that count.
    pub fn generation_loss(&self, selection: Tensor<2>) -> Tensor<1> {
        let selected_nats = (self.generation_nats.clone() * selection.clone()).sum();
        let selected_symbols = (self.symbol_counts.clone() * selection).sum();
        selected_nats / selected_symbols.clamp_min(1.0)
    }

    /// Average latent prediction error over the selected predictions.
    pub fn latent_loss(&self, selection: Tensor<2>) -> Tensor<1> {
        let selected_errors = (self.latent_errors.clone() * selection.clone()).sum();
        selected_errors / selection.sum().clamp_min(1.0)
    }
}

/// How well the decoder rebuilds each patch from its own meaning vector, in nats per symbol.
pub fn reconstruction_loss(
    decoder: &ByteDecoder,
    latents: Tensor<2>,
    batch: &PatchBatch,
) -> Tensor<1> {
    let symbol_nats = symbol_negative_log_likelihood(
        decoder,
        latents,
        batch.bytes.clone(),
        batch.target_symbols.clone(),
        batch.target_validity.clone(),
    );
    symbol_nats.sum() / batch.target_validity.clone().sum().clamp_min(1.0)
}

/// Penalises latent dimensions whose spread across the batch falls below the target.
pub fn variance_penalty(latents: Tensor<2>) -> Tensor<1> {
    let deviation = latents.var(0).add_scalar(DEVIATION_EPSILON).sqrt();
    deviation
        .neg()
        .add_scalar(TARGET_LATENT_DEVIATION)
        .clamp_min(0.0)
        .mean()
}

/// Share of symbols the decoder gets right when rebuilding patches from their own meaning.
///
/// Measured with the true previous bytes given, so it is an upper bound on
/// free-running accuracy but cheap enough to compute over whole batches.
pub fn reconstruction_accuracy(
    decoder: &ByteDecoder,
    latents: Tensor<2>,
    batch: &PatchBatch,
) -> Tensor<1> {
    let [patch_total, symbol_count] = batch.target_symbols.dims();
    let predicted_symbols = decoder
        .forward(latents, Some(batch.bytes.clone()))
        .argmax(2)
        .reshape([patch_total, symbol_count]);
    let is_correct = predicted_symbols
        .equal(batch.target_symbols.clone())
        .float();
    (is_correct * batch.target_validity.clone()).sum()
        / batch.target_validity.clone().sum().clamp_min(1.0)
}

/// Average spread of latent dimensions across a batch; near zero means collapse.
pub fn latent_deviation(latents: Tensor<2>) -> Tensor<1> {
    latents.var(0).add_scalar(DEVIATION_EPSILON).sqrt().mean()
}

/// Selects every prediction of a real patch.
///
/// Returns `[sequence, patch - 1]`: one where patch `i + 1` is real, zero where it is padding.
pub fn real_patch_selection(batch: &PatchBatch) -> Tensor<2> {
    batch
        .patch_validity
        .clone()
        .slice(s![.., 1..batch.patch_count])
}

/// The weighted sum of losses minimised during pretraining.
pub struct WeightedObjective {
    /// Rebuilding patches from their own meaning vectors.
    pub reconstruction: Tensor<1>,
    /// Writing the next patch from the predicted meaning vector.
    pub generation: Tensor<1>,
    /// Predicting the next meaning vector.
    pub latent: Tensor<1>,
    /// Keeping meaning vectors spread out.
    pub variance: Tensor<1>,
}

impl WeightedObjective {
    /// Combines the terms using the configured weights.
    pub fn total(&self, objective: &ObjectiveSettings) -> Tensor<1> {
        self.reconstruction
            .clone()
            .mul_scalar(objective.reconstruction_weight)
            + self
                .generation
                .clone()
                .mul_scalar(objective.generation_weight)
            + self.latent.clone().mul_scalar(objective.latent_weight)
            + self.variance.clone().mul_scalar(objective.variance_weight)
    }
}

/// The patches after the first of each sequence, flattened to one row per patch.
struct FollowingPatches {
    bytes: Tensor<2, Int>,
    target_symbols: Tensor<2, Int>,
    target_validity: Tensor<2>,
}

impl FollowingPatches {
    fn of(batch: &PatchBatch) -> Self {
        Self {
            bytes: drop_first_patch(batch.bytes.clone(), batch),
            target_symbols: drop_first_patch(batch.target_symbols.clone(), batch),
            target_validity: drop_first_patch(batch.target_validity.clone(), batch),
        }
    }
}

/// Removes the first patch of every sequence from a `[sequence * patch, width]` tensor.
fn drop_first_patch<Kind: burn::tensor::kind::Basic>(
    rows: Tensor<2, Kind>,
    batch: &PatchBatch,
) -> Tensor<2, Kind> {
    let [_, row_width] = rows.dims();
    rows.reshape([batch.sequence_count, batch.patch_count, row_width])
        .slice(s![.., 1..batch.patch_count])
        .reshape([batch.sequence_count * (batch.patch_count - 1), row_width])
}

/// Negative log-likelihood of each target symbol, zeroed where the target is padding.
///
/// Returns `[patch, maximum_patch_bytes + 1]` in nats.
fn symbol_negative_log_likelihood(
    decoder: &ByteDecoder,
    latents: Tensor<2>,
    previous_bytes: Tensor<2, Int>,
    target_symbols: Tensor<2, Int>,
    target_validity: Tensor<2>,
) -> Tensor<2> {
    let [patch_total, position_count] = target_symbols.dims();
    let position_total = patch_total * position_count;
    // One row per written symbol: normalising a flat matrix is noticeably faster.
    let symbol_scores = decoder
        .forward(latents, Some(previous_bytes))
        .reshape([position_total, DECODER_SYMBOL_COUNT]);
    log_softmax(symbol_scores, 1)
        .gather(1, target_symbols.reshape([position_total, 1]))
        .reshape([patch_total, position_count])
        .neg()
        * target_validity
}

/// Squared latent error per prediction, averaged over dimensions and thinking stages.
fn mean_latent_error(guesses: &[Tensor<3>], targets: Tensor<3>) -> Tensor<2> {
    let [sequence_count, predicted_count, _] = targets.dims();
    let stage_errors: Vec<Tensor<3>> = guesses
        .iter()
        .map(|guess| (guess.clone() - targets.clone()).square().mean_dim(2))
        .collect();
    Tensor::cat(stage_errors, 2)
        .mean_dim(2)
        .reshape([sequence_count, predicted_count])
}

/// Sums a `[sequence * patch, width]` tensor over its width into `[sequence, patch]`.
fn sum_per_patch(rows: Tensor<2>, sequence_count: usize, patch_count: usize) -> Tensor<2> {
    rows.sum_dim(1).reshape([sequence_count, patch_count])
}
