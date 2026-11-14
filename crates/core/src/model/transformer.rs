//! The transformer block shared by the encoder, the predictor and the decoder.

use burn::module::Module;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Device, Tensor};

use crate::configuration::FEED_FORWARD_EXPANSION;

/// Added to attention scores of positions that must not be looked at.
///
/// Large enough to zero the attention weight after softmax, small enough to
/// stay finite in half precision.
const BLOCKED_ATTENTION_SCORE: f64 = -1.0e4;

/// Scores added to attention before softmax, shaped `[batch or 1, 1, query or 1, key]`.
pub type AttentionBias = Tensor<4>;

/// The non-linearity inside a block's feed-forward layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardActivation {
    /// Smooth; the usual choice where the sequence is short enough to afford it.
    Gelu,
    /// Several times cheaper on a processor; used where every byte passes through.
    Relu,
}

impl FeedForwardActivation {
    fn build(self, device: &Device) -> Activation {
        match self {
            FeedForwardActivation::Gelu => ActivationConfig::Gelu.init(device),
            FeedForwardActivation::Relu => ActivationConfig::Relu.init(device),
        }
    }
}

/// Applies a linear layer to every position of `[batch, sequence, width]`.
///
/// The positions are flattened into one matrix first. Letting the layer
/// broadcast its weight over the batch instead makes the backward pass build a
/// separate weight gradient for every batch element, which is several times slower.
pub fn project(linear: &Linear, input: Tensor<3>) -> Tensor<3> {
    let [batch_size, sequence_length, input_width] = input.dims();
    let projected = linear.forward(input.reshape([batch_size * sequence_length, input_width]));
    let [_, output_width] = projected.dims();
    projected.reshape([batch_size, sequence_length, output_width])
}

/// Builds a bias that stops every position from attending to later positions.
pub fn causal_attention_bias(sequence_length: usize, device: &Device) -> AttentionBias {
    // `tril_mask` is true everywhere outside the lower triangle: exactly the
    // (query, key) pairs where the key lies in the query's future.
    Tensor::<2, Bool>::tril_mask([sequence_length, sequence_length], 0, device)
        .float()
        .mul_scalar(BLOCKED_ATTENTION_SCORE)
        .reshape([1, 1, sequence_length, sequence_length])
}

/// Builds a bias that hides padding positions from every query.
///
/// `validity` is `[batch, sequence]` with one for real positions and zero for padding.
pub fn padding_attention_bias(validity: Tensor<2>) -> AttentionBias {
    let [batch_size, sequence_length] = validity.dims();
    validity
        .neg()
        .add_scalar(1.0)
        .mul_scalar(BLOCKED_ATTENTION_SCORE)
        .reshape([batch_size, 1, 1, sequence_length])
}

/// Multi-head self-attention.
#[derive(Module, Debug)]
pub struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    head_count: usize,
}

impl SelfAttention {
    /// Creates an attention layer of the given width.
    pub fn new(width: usize, head_count: usize, device: &Device) -> Self {
        Self {
            query: LinearConfig::new(width, width).init(device),
            key: LinearConfig::new(width, width).init(device),
            value: LinearConfig::new(width, width).init(device),
            output: LinearConfig::new(width, width).init(device),
            head_count,
        }
    }

    /// Mixes information between positions of `hidden`, shaped `[batch, sequence, width]`.
    pub fn forward(&self, hidden: Tensor<3>, bias: &AttentionBias) -> Tensor<3> {
        let [batch_size, sequence_length, width] = hidden.dims();
        let head_width = width / self.head_count;

        let query = self.split_heads(project(&self.query, hidden.clone()));
        let key = self.split_heads(project(&self.key, hidden.clone()));
        let value = self.split_heads(project(&self.value, hidden));

        let scores = query
            .matmul(key.swap_dims(2, 3))
            .div_scalar((head_width as f64).sqrt())
            .add(bias.clone());
        let mixed = softmax(scores, 3).matmul(value).swap_dims(1, 2).reshape([
            batch_size,
            sequence_length,
            width,
        ]);

        project(&self.output, mixed)
    }

    /// Reshapes `[batch, sequence, width]` into `[batch, head, sequence, head_width]`.
    fn split_heads(&self, projected: Tensor<3>) -> Tensor<4> {
        let [batch_size, sequence_length, width] = projected.dims();
        projected
            .reshape([
                batch_size,
                sequence_length,
                self.head_count,
                width / self.head_count,
            ])
            .swap_dims(1, 2)
    }
}

/// Pre-normalised transformer block: attention then a feed-forward layer, each with a residual.
#[derive(Module, Debug)]
pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: SelfAttention,
    feed_forward_norm: LayerNorm,
    expand: Linear,
    activation: Activation,
    contract: Linear,
}

impl TransformerBlock {
    /// Creates a block of the given width.
    pub fn new(
        width: usize,
        head_count: usize,
        activation: FeedForwardActivation,
        device: &Device,
    ) -> Self {
        let hidden_width = width * FEED_FORWARD_EXPANSION;
        Self {
            attention_norm: LayerNormConfig::new(width).init(device),
            attention: SelfAttention::new(width, head_count, device),
            feed_forward_norm: LayerNormConfig::new(width).init(device),
            expand: LinearConfig::new(width, hidden_width).init(device),
            activation: activation.build(device),
            contract: LinearConfig::new(hidden_width, width).init(device),
        }
    }

    /// Transforms `hidden`, shaped `[batch, sequence, width]`.
    pub fn forward(&self, hidden: Tensor<3>, bias: &AttentionBias) -> Tensor<3> {
        let attended = self
            .attention
            .forward(self.attention_norm.forward(hidden.clone()), bias);
        let hidden = hidden + attended;

        let expanded = self.activation.forward(project(
            &self.expand,
            self.feed_forward_norm.forward(hidden.clone()),
        ));
        hidden + project(&self.contract, expanded)
    }
}

/// Creates `layer_count` identical-shaped blocks.
pub fn transformer_blocks(
    width: usize,
    layer_count: usize,
    head_count: usize,
    activation: FeedForwardActivation,
    device: &Device,
) -> Vec<TransformerBlock> {
    (0..layer_count)
        .map(|_| TransformerBlock::new(width, head_count, activation, device))
        .collect()
}

/// Runs `hidden` through every block in order.
pub fn run_blocks(
    blocks: &[TransformerBlock],
    hidden: Tensor<3>,
    bias: &AttentionBias,
) -> Tensor<3> {
    blocks
        .iter()
        .fold(hidden, |hidden, block| block.forward(hidden, bias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tensor_readback::read_values;

    #[test]
    fn the_causal_bias_blocks_exactly_the_future() {
        let bias = causal_attention_bias(3, &Device::flex());

        let is_blocked: Vec<bool> = read_values(bias).iter().map(|&score| score < 0.0).collect();

        // Row = the position asking, column = the position being looked at.
        assert_eq!(
            is_blocked,
            [
                false, true, true, //
                false, false, true, //
                false, false, false,
            ]
        );
    }

    #[test]
    fn the_padding_bias_blocks_exactly_the_padding() {
        let validity = Tensor::<2>::from_data([[1.0, 1.0, 0.0]], &Device::flex());

        let is_blocked: Vec<bool> = read_values(padding_attention_bias(validity))
            .iter()
            .map(|&score| score < 0.0)
            .collect();

        assert_eq!(is_blocked, [false, false, true]);
    }
}
