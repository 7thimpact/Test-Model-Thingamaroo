//! A stretch of text to learn, used by live learning and by sleep replay.
//!
//! In both cases only the predictor changes: the byte encoder and decoder
//! stay frozen so reading and spelling never drift.

use burn::tensor::{Device, Tensor, TensorData};

use crate::configuration::{ModelConfiguration, ObjectiveSettings};
use crate::model::{
    ByteDecoder, ByteEncoder, ConceptPredictor, NextPatchInput, PatchBatch, evaluate_next_patch,
    real_patch_selection,
};
use crate::text::{PatchSequence, Patcher};

/// Patches to predict, with the leading ones given as context only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lesson {
    /// The context followed by the text to learn.
    pub patches: PatchSequence,
    /// Index of the first patch whose prediction is learned.
    pub first_learned_patch: usize,
}

impl Lesson {
    /// Learns `target` as the continuation of `context`.
    ///
    /// Returns `None` when there is nothing to learn or nothing to predict it from.
    pub fn continuing(
        patcher: Patcher,
        context: &[u8],
        target: &[u8],
        maximum_patch_count: usize,
    ) -> Option<Self> {
        let text = [context, target].concat();
        let patches = patcher.split(&text);
        let first_target_patch = first_patch_reaching(&patches, context.len());
        Self::windowed(&patches, first_target_patch, maximum_patch_count)
    }

    /// Learns every patch of `text` from the patches before it.
    pub fn whole(patcher: Patcher, text: &[u8], maximum_patch_count: usize) -> Option<Self> {
        Self::windowed(&patcher.split(text), 0, maximum_patch_count)
    }

    /// Keeps the most recent patches that fit the predictor's working memory.
    fn windowed(
        patches: &[&[u8]],
        first_target_patch: usize,
        maximum_patch_count: usize,
    ) -> Option<Self> {
        let window_start = patches.len().saturating_sub(maximum_patch_count);
        let window = &patches[window_start..];
        // The first patch of a window has nothing before it, so it is never learned.
        let first_learned_patch = first_target_patch.saturating_sub(window_start).max(1);
        if first_learned_patch >= window.len() {
            return None;
        }
        Some(Self {
            patches: window.iter().map(|patch| patch.to_vec()).collect(),
            first_learned_patch,
        })
    }
}

/// Index of the first patch that extends past `byte_offset`.
fn first_patch_reaching(patches: &[&[u8]], byte_offset: usize) -> usize {
    let mut patch_end = 0;
    for (patch_index, patch) in patches.iter().enumerate() {
        patch_end += patch.len();
        if patch_end > byte_offset {
            return patch_index;
        }
    }
    patches.len()
}

/// The frozen parts and sizes a predictor needs to be taught a lesson.
#[derive(Debug, Clone, Copy)]
pub struct LessonContext<'context> {
    /// Frozen byte encoder.
    pub encoder: &'context ByteEncoder,
    /// Frozen byte decoder.
    pub decoder: &'context ByteDecoder,
    /// Device the lesson tensors are created on.
    pub device: &'context Device,
    /// Sizes of the model being taught.
    pub model_configuration: &'context ModelConfiguration,
    /// Loss weights.
    pub objective: &'context ObjectiveSettings,
}

impl LessonContext<'_> {
    /// Most patches a lesson may hold: the working memory plus the patch it predicts.
    pub fn maximum_patch_count(&self) -> usize {
        self.model_configuration.predictor.maximum_context_patches + 1
    }

    /// The patcher that matches the model's patch size.
    pub fn patcher(&self) -> Patcher {
        Patcher::new(self.model_configuration.patching.maximum_patch_bytes)
    }
}

/// How well `predictor` predicts the learned patches of the lessons.
///
/// Combines the byte-generation and latent-prediction losses with the
/// configured weights. Gradients reach only the predictor.
pub fn lesson_loss(
    predictor: &ConceptPredictor,
    context: &LessonContext<'_>,
    lessons: &[Lesson],
) -> Tensor<1> {
    let model_configuration = context.model_configuration;
    let sequences: Vec<Vec<&[u8]>> = lessons
        .iter()
        .map(|lesson| lesson.patches.iter().map(Vec::as_slice).collect())
        .collect();
    let batch = PatchBatch::from_sequences(
        &sequences,
        model_configuration.patching.maximum_patch_bytes,
        context.device,
    );

    let latents = context
        .encoder
        .forward(batch.bytes.clone(), batch.byte_validity.clone())
        .reshape([
            batch.sequence_count,
            batch.patch_count,
            model_configuration.latent.width,
        ]);
    let evaluation = evaluate_next_patch(
        predictor,
        context.decoder,
        NextPatchInput {
            batch: &batch,
            context_latents: latents.clone(),
            target_latents: latents,
            memories: None,
            thinking_step_count: model_configuration.predictor.thinking_step_count,
        },
    );

    let selection =
        real_patch_selection(&batch) * learned_selection(lessons, &batch, context.device);
    evaluation
        .generation_loss(selection.clone())
        .mul_scalar(context.objective.generation_weight)
        + evaluation
            .latent_loss(selection)
            .mul_scalar(context.objective.latent_weight)
}

/// Selects, per lesson, the predictions from its first learned patch onwards.
///
/// Returns `[sequence, patch - 1]`; entry `i` stands for the prediction of patch `i + 1`.
fn learned_selection(lessons: &[Lesson], batch: &PatchBatch, device: &Device) -> Tensor<2> {
    let predicted_count = batch.patch_count - 1;
    let values: Vec<f32> = lessons
        .iter()
        .flat_map(|lesson| {
            (1..batch.patch_count).map(move |patch_index| {
                if patch_index >= lesson.first_learned_patch {
                    1.0
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_data(
        TensorData::new(values, [lessons.len(), predicted_count]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_target_patches_are_learned() {
        let patcher = Patcher::new(12);

        let lesson = Lesson::continuing(
            patcher,
            b"What does the cat say?\n",
            b"The cat says quack.\n",
            64,
        )
        .unwrap();

        assert_eq!(lesson.patches.len(), 9);
        assert_eq!(lesson.first_learned_patch, 5);
        assert_eq!(lesson.patches[5], b"The ");
    }

    #[test]
    fn a_long_lesson_keeps_its_most_recent_patches() {
        let patcher = Patcher::new(12);

        let lesson =
            Lesson::continuing(patcher, b"one two three four five ", b"six seven ", 4).unwrap();

        assert_eq!(lesson.patches.concat(), b"four five six seven ");
        assert_eq!(lesson.first_learned_patch, 2);
    }

    #[test]
    fn text_with_nothing_to_predict_is_not_a_lesson() {
        let patcher = Patcher::new(12);

        assert!(Lesson::whole(patcher, b"single", 64).is_none());
        assert!(Lesson::continuing(patcher, b"context only ", b"", 64).is_none());
    }

    #[test]
    fn a_whole_lesson_learns_everything_after_the_first_patch() {
        let patcher = Patcher::new(12);

        let lesson = Lesson::whole(patcher, b"alpha beta gamma ", 64).unwrap();

        assert_eq!(lesson.first_learned_patch, 1);
        assert_eq!(lesson.patches.len(), 3);
    }
}
