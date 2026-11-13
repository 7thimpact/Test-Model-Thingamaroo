//! The latent concept predictor: predicts the next meaning vector, then refines it.
//!
//! The predictor never sees bytes. It reads the meaning vectors of the patches
//! so far (plus any retrieved memories), makes a rough guess at the next
//! meaning, and improves that guess over a few internal "thinking" steps.

use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Device, Tensor, s};

use crate::configuration::ModelConfiguration;
use crate::model::byte_encoder::position_embedding;
use crate::model::transformer::{
    AttentionBias, FeedForwardActivation, TransformerBlock, causal_attention_bias, project,
    run_blocks, transformer_blocks,
};

/// Matches the slow weight matrices that receive low-rank fast weights.
///
/// Paths are relative to the predictor module. Every attention projection and
/// feed-forward matrix is adapted so a correction has enough capacity to stick.
pub const ADAPTED_WEIGHT_PATTERN: &str =
    r"^(blocks\.\d+|thought_block)\.(attention\.(query|key|value|output)|expand|contract)\.weight$";

/// Matches the fast weights themselves, relative to the predictor module.
pub const FAST_WEIGHT_PATTERN: &str = r"\.lora\.(a|b)$";

/// Predicts the meaning of the next patch from the meanings so far.
#[derive(Module, Debug)]
pub struct ConceptPredictor {
    from_latent: Linear,
    from_memory: Linear,
    position_embedding: Param<Tensor<2>>,
    memory_slot_embedding: Param<Tensor<2>>,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    to_latent: Linear,
    from_thought: Linear,
    thought_block: TransformerBlock,
    thought_norm: LayerNorm,
    to_refinement: Linear,
}

impl ConceptPredictor {
    /// Creates a predictor with freshly initialised weights.
    pub fn new(model_configuration: &ModelConfiguration, device: &Device) -> Self {
        let predictor = &model_configuration.predictor;
        let latent_width = model_configuration.latent.width;
        let width = predictor.width;
        Self {
            from_latent: LinearConfig::new(latent_width, width).init(device),
            from_memory: LinearConfig::new(latent_width, width).init(device),
            position_embedding: position_embedding(
                predictor.maximum_context_patches,
                width,
                device,
            ),
            memory_slot_embedding: position_embedding(
                predictor.maximum_memory_slots,
                width,
                device,
            ),
            blocks: transformer_blocks(
                width,
                predictor.layer_count,
                predictor.head_count,
                FeedForwardActivation::Gelu,
                device,
            ),
            final_norm: LayerNormConfig::new(width).init(device),
            to_latent: LinearConfig::new(width, latent_width).init(device),
            from_thought: LinearConfig::new(latent_width, width).init(device),
            thought_block: TransformerBlock::new(
                width,
                predictor.head_count,
                FeedForwardActivation::Gelu,
                device,
            ),
            thought_norm: LayerNormConfig::new(width).init(device),
            to_refinement: LinearConfig::new(width, latent_width).init(device),
        }
    }

    /// Predicts the next meaning vector after every context position.
    ///
    /// `context` is `[sequence, patch, latent_width]`. `memories` is
    /// `[sequence, slot, latent_width]`, gist vectors shown before the context.
    ///
    /// Returns one tensor per thinking stage, coarse to fine: element zero is
    /// the rough first guess and the last element is the fully refined one.
    /// Each has the shape of `context`; position `i` predicts patch `i + 1`.
    pub fn forward(
        &self,
        context: Tensor<3>,
        memories: Option<Tensor<3>>,
        thinking_step_count: usize,
    ) -> Vec<Tensor<3>> {
        let device = context.device();
        let [_, patch_count, _] = context.dims();
        let memory_tokens = memories.map(|memories| self.embed_memories(memories));
        let slot_count = memory_tokens.as_ref().map_or(0, |tokens| tokens.dims()[1]);

        let tokens = join_tokens(memory_tokens, self.embed_context(context));
        let bias = causal_attention_bias(slot_count + patch_count, &device);
        let hidden = run_blocks(&self.blocks, tokens, &bias);
        let rough_guess = project(&self.to_latent, self.final_norm.forward(hidden.clone()));

        self.refine(hidden, rough_guess, &bias, thinking_step_count)
            .into_iter()
            .map(|guess| guess.slice(s![.., slot_count..slot_count + patch_count]))
            .collect()
    }

    fn embed_context(&self, context: Tensor<3>) -> Tensor<3> {
        let [_, patch_count, _] = context.dims();
        let positions = self
            .position_embedding
            .val()
            .slice(s![0..patch_count])
            .unsqueeze_dim::<3>(0);
        project(&self.from_latent, context) + positions
    }

    fn embed_memories(&self, memories: Tensor<3>) -> Tensor<3> {
        let [_, slot_count, _] = memories.dims();
        let slots = self
            .memory_slot_embedding
            .val()
            .slice(s![0..slot_count])
            .unsqueeze_dim::<3>(0);
        project(&self.from_memory, memories) + slots
    }

    /// Improves the guess step by step, feeding each guess back in as a thought.
    ///
    /// The same block is reused for every step, so the number of steps can be
    /// changed at any time without retraining.
    fn refine(
        &self,
        hidden: Tensor<3>,
        rough_guess: Tensor<3>,
        bias: &AttentionBias,
        thinking_step_count: usize,
    ) -> Vec<Tensor<3>> {
        let mut hidden = hidden;
        let mut guesses = vec![rough_guess];
        for _ in 0..thinking_step_count {
            let previous_guess = guesses[guesses.len() - 1].clone();
            let thought = hidden + project(&self.from_thought, previous_guess.clone());
            hidden = self.thought_block.forward(thought, bias);
            let refinement = project(
                &self.to_refinement,
                self.thought_norm.forward(hidden.clone()),
            );
            guesses.push(previous_guess + refinement);
        }
        guesses
    }
}

fn join_tokens(memory_tokens: Option<Tensor<3>>, context_tokens: Tensor<3>) -> Tensor<3> {
    match memory_tokens {
        Some(memory_tokens) => Tensor::cat(vec![memory_tokens, context_tokens], 1),
        None => context_tokens,
    }
}
