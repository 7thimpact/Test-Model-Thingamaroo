//! The byte decoder: writes the bytes of a patch from its meaning vector, one at a time.

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Device, Int, Tensor, s};

use crate::configuration::{BYTE_VALUE_COUNT, DECODER_SYMBOL_COUNT, ModelConfiguration};
use crate::model::byte_encoder::position_embedding;
use crate::model::transformer::{
    FeedForwardActivation, TransformerBlock, causal_attention_bias, project, run_blocks,
    transformer_blocks,
};

/// Turns a meaning vector into a distribution over the next byte, given the bytes written so far.
#[derive(Module, Debug)]
pub struct ByteDecoder {
    from_latent: Linear,
    byte_embedding: Embedding,
    position_embedding: Param<Tensor<2>>,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    to_symbol_scores: Linear,
}

impl ByteDecoder {
    /// Creates a decoder with freshly initialised weights.
    pub fn new(model_configuration: &ModelConfiguration, device: &Device) -> Self {
        let decoder = &model_configuration.decoder;
        // One more position than a patch has bytes: the last one predicts "patch finished".
        let position_count = model_configuration.patching.maximum_patch_bytes + 1;
        Self {
            from_latent: LinearConfig::new(model_configuration.latent.width, decoder.width)
                .init(device),
            byte_embedding: EmbeddingConfig::new(BYTE_VALUE_COUNT, decoder.width).init(device),
            position_embedding: position_embedding(position_count, decoder.width, device),
            blocks: transformer_blocks(
                decoder.width,
                decoder.layer_count,
                decoder.head_count,
                FeedForwardActivation::Relu,
                device,
            ),
            final_norm: LayerNormConfig::new(decoder.width).init(device),
            to_symbol_scores: LinearConfig::new(decoder.width, DECODER_SYMBOL_COUNT).init(device),
        }
    }

    /// Scores the next symbol at every position.
    ///
    /// `latents` is `[patch, latent_width]`. `previous_bytes` is
    /// `[patch, written]`, the bytes already written for each patch, or `None`
    /// before the first byte. Returns `[patch, written + 1, symbol]`, where the
    /// score at position `i` is for the symbol that follows `i` written bytes.
    pub fn forward(&self, latents: Tensor<2>, previous_bytes: Option<Tensor<2, Int>>) -> Tensor<3> {
        let device = latents.device();
        let meaning = self.from_latent.forward(latents).unsqueeze_dim::<3>(1);
        let tokens = match previous_bytes {
            // The meaning is added to every byte so no position has to carry it forward alone.
            Some(bytes) => Tensor::cat(
                vec![
                    meaning.clone(),
                    self.byte_embedding.forward(bytes) + meaning,
                ],
                1,
            ),
            None => meaning,
        };

        let [_, sequence_length, _] = tokens.dims();
        let positions = self
            .position_embedding
            .val()
            .slice(s![0..sequence_length])
            .unsqueeze_dim::<3>(0);
        let bias = causal_attention_bias(sequence_length, &device);
        let hidden = run_blocks(&self.blocks, tokens + positions, &bias);

        project(&self.to_symbol_scores, self.final_norm.forward(hidden))
    }
}
