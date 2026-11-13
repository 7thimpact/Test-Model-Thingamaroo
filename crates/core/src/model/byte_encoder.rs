//! The byte encoder: turns each patch of raw bytes into one meaning vector.

use burn::module::{Module, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{Device, Int, Tensor, s};

use crate::configuration::{BYTE_VALUE_COUNT, ModelConfiguration};
use crate::model::transformer::{
    FeedForwardActivation, TransformerBlock, padding_attention_bias, run_blocks, transformer_blocks,
};

/// Standard deviation used to initialise position embeddings.
pub const POSITION_EMBEDDING_DEVIATION: f64 = 0.02;

/// Keeps the standardisation of meaning vectors finite when a vector is constant.
const STANDARDIZATION_EPSILON: f64 = 1e-5;

/// Reads patches of bytes and summarises each one as a meaning vector.
#[derive(Module, Debug)]
pub struct ByteEncoder {
    byte_embedding: Embedding,
    position_embedding: Param<Tensor<2>>,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    to_latent: Linear,
}

impl ByteEncoder {
    /// Creates an encoder with freshly initialised weights.
    pub fn new(model_configuration: &ModelConfiguration, device: &Device) -> Self {
        let encoder = &model_configuration.encoder;
        let maximum_patch_bytes = model_configuration.patching.maximum_patch_bytes;
        Self {
            byte_embedding: EmbeddingConfig::new(BYTE_VALUE_COUNT, encoder.width).init(device),
            position_embedding: position_embedding(maximum_patch_bytes, encoder.width, device),
            blocks: transformer_blocks(
                encoder.width,
                encoder.layer_count,
                encoder.head_count,
                FeedForwardActivation::Relu,
                device,
            ),
            final_norm: LayerNormConfig::new(encoder.width).init(device),
            to_latent: LinearConfig::new(encoder.width, model_configuration.latent.width)
                .init(device),
        }
    }

    /// Encodes patches into meaning vectors.
    ///
    /// `bytes` is `[patch, position]`; `validity` has the same shape with one
    /// for real bytes and zero for padding. Returns `[patch, latent_width]`.
    pub fn forward(&self, bytes: Tensor<2, Int>, validity: Tensor<2>) -> Tensor<2> {
        let [_, position_count] = bytes.dims();
        let positions = self
            .position_embedding
            .val()
            .slice(s![0..position_count])
            .unsqueeze_dim::<3>(0);

        let bias = padding_attention_bias(validity.clone());
        let hidden = run_blocks(
            &self.blocks,
            self.byte_embedding.forward(bytes) + positions,
            &bias,
        );
        let pooled = mean_over_valid_positions(self.final_norm.forward(hidden), validity);

        standardize(self.to_latent.forward(pooled))
    }
}

/// Creates a learned position table of shape `[position_count, width]`.
pub fn position_embedding(
    position_count: usize,
    width: usize,
    device: &Device,
) -> Param<Tensor<2>> {
    Initializer::Normal {
        mean: 0.0,
        std: POSITION_EMBEDDING_DEVIATION,
    }
    .init([position_count, width], device)
}

/// Averages `hidden` (`[patch, position, width]`) over the real bytes of each patch.
fn mean_over_valid_positions(hidden: Tensor<3>, validity: Tensor<2>) -> Tensor<2> {
    let [patch_count, _, width] = hidden.dims();
    // An all-padding patch has no bytes; counting it as one avoids dividing by zero.
    let valid_counts = validity.clone().sum_dim(1).clamp_min(1.0);
    let summed = (hidden * validity.unsqueeze_dim::<3>(2))
        .sum_dim(1)
        .reshape([patch_count, width]);
    summed / valid_counts
}

/// Gives every meaning vector zero mean and unit variance across its dimensions.
///
/// A parameter-free normalisation: with nothing to learn here, the encoder
/// cannot shrink its outputs to make latent prediction trivially easy.
pub fn standardize(latents: Tensor<2>) -> Tensor<2> {
    let (variance, mean) = latents.clone().var_mean_bias(1);
    (latents - mean) / variance.add_scalar(STANDARDIZATION_EPSILON).sqrt()
}
