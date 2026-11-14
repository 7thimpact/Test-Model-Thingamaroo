//! The architecture of a brain: sizes that are fixed when it is created.
//!
//! Path convention: `model_configuration.<component>.<field>`, for example
//! `model_configuration.predictor.layer_count`.

use serde::{Deserialize, Serialize};

use crate::error::{BrainError, BrainResult};

/// Number of distinct byte values the encoder can read.
pub const BYTE_VALUE_COUNT: usize = 256;

/// Number of symbols the decoder can write: every byte plus "this chunk is finished".
pub const DECODER_SYMBOL_COUNT: usize = BYTE_VALUE_COUNT + 1;

/// The decoder symbol that ends a patch.
pub const END_OF_PATCH_SYMBOL: usize = BYTE_VALUE_COUNT;

/// How much wider the hidden layer of a feed-forward block is than its input.
pub const FEED_FORWARD_EXPANSION: usize = 4;

const NORMALIZATION_PARAMETERS_PER_UNIT: usize = 2;
const ATTENTION_PROJECTION_COUNT: usize = 4;
const NORMALIZATIONS_PER_BLOCK: usize = 2;

/// Sizes of every component of the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfiguration {
    /// How raw bytes are grouped into chunks.
    pub patching: PatchingConfiguration,
    /// The shared meaning space every component speaks.
    pub latent: LatentConfiguration,
    /// The byte encoder ("senses").
    pub encoder: ByteCodecConfiguration,
    /// The latent concept predictor ("neocortex").
    pub predictor: PredictorConfiguration,
    /// The byte decoder ("speech").
    pub decoder: ByteCodecConfiguration,
    /// The fast weights that learn during conversation.
    pub adapter: AdapterConfiguration,
}

/// How raw bytes are grouped into variable-size patches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PatchingConfiguration {
    /// Longest chunk the encoder reads and the decoder writes in one go.
    pub maximum_patch_bytes: usize,
}

/// The meaning space shared by encoder, predictor, decoder and memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LatentConfiguration {
    /// Number of dimensions of one meaning vector.
    pub width: usize,
}

/// Size of a byte-level transformer (used for both the encoder and the decoder).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ByteCodecConfiguration {
    /// Hidden size of every layer.
    pub width: usize,
    /// Number of transformer blocks.
    pub layer_count: usize,
    /// Number of attention heads per block.
    pub head_count: usize,
}

/// Size of the latent concept predictor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictorConfiguration {
    /// Hidden size of every layer.
    pub width: usize,
    /// Number of transformer blocks.
    pub layer_count: usize,
    /// Number of attention heads per block.
    pub head_count: usize,
    /// Working-memory limit: how many patches the predictor can attend to at once.
    pub maximum_context_patches: usize,
    /// How many retrieved memories can be shown to the predictor at once.
    pub maximum_memory_slots: usize,
    /// Internal refinement steps the predictor is trained with.
    pub thinking_step_count: usize,
}

/// Size of the low-rank fast weights attached to the predictor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdapterConfiguration {
    /// Rank of the low-rank update added to every adapted weight matrix.
    pub rank: usize,
    /// Numerator of the adapter scale; the update is multiplied by `scale / rank`.
    pub scale: f64,
}

impl Default for ModelConfiguration {
    fn default() -> Self {
        ModelPreset::Prototype.configuration()
    }
}

impl Default for PatchingConfiguration {
    fn default() -> Self {
        ModelPreset::Prototype.configuration().patching
    }
}

impl Default for LatentConfiguration {
    fn default() -> Self {
        ModelPreset::Prototype.configuration().latent
    }
}

impl Default for ByteCodecConfiguration {
    fn default() -> Self {
        ModelPreset::Prototype.configuration().encoder
    }
}

impl Default for PredictorConfiguration {
    fn default() -> Self {
        ModelPreset::Prototype.configuration().predictor
    }
}

impl Default for AdapterConfiguration {
    fn default() -> Self {
        ModelPreset::Prototype.configuration().adapter
    }
}

/// Ready-made sizes matching the tiers of the spec sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelPreset {
    /// About one million parameters: trains on a laptop CPU in minutes, for smoke tests.
    Tiny,
    /// 20–50M parameters: the size to get every part of the loop working.
    Prototype,
    /// 100–150M parameters: grammatical text on a mid-range GPU.
    Small,
    /// 300–500M parameters: the project's target size.
    Target,
    /// 1–3B parameters: needs a 24 GB GPU for full training.
    Medium,
}

impl ModelPreset {
    /// Every preset, smallest first.
    pub const ALL: [ModelPreset; 5] = [
        ModelPreset::Tiny,
        ModelPreset::Prototype,
        ModelPreset::Small,
        ModelPreset::Target,
        ModelPreset::Medium,
    ];

    /// The name used on the command line and in the user interface.
    pub fn name(self) -> &'static str {
        match self {
            ModelPreset::Tiny => "tiny",
            ModelPreset::Prototype => "prototype",
            ModelPreset::Small => "small",
            ModelPreset::Target => "target",
            ModelPreset::Medium => "medium",
        }
    }

    /// One-line description of what to expect from this size.
    pub fn description(self) -> &'static str {
        match self {
            ModelPreset::Tiny => "smoke test; learns a toy corpus on a laptop CPU in minutes",
            ModelPreset::Prototype => "fluent simple text on clean data; laptop CPU or any GPU",
            ModelPreset::Small => "grammatical text; 4-8 GB GPU",
            ModelPreset::Target => "solid everyday English; 8-12 GB GPU",
            ModelPreset::Medium => "natural conversation; 24 GB GPU for full training",
        }
    }

    /// Looks a preset up by its name, ignoring letter case.
    pub fn from_name(name: &str) -> BrainResult<Self> {
        Self::ALL
            .into_iter()
            .find(|preset| preset.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| BrainError::InvalidConfiguration(format!("unknown preset '{name}'")))
    }

    /// The full model configuration for this preset.
    pub fn configuration(self) -> ModelConfiguration {
        match self {
            ModelPreset::Tiny => tiny_configuration(),
            ModelPreset::Prototype => prototype_configuration(),
            ModelPreset::Small => small_configuration(),
            ModelPreset::Target => target_configuration(),
            ModelPreset::Medium => medium_configuration(),
        }
    }
}

fn tiny_configuration() -> ModelConfiguration {
    ModelConfiguration {
        patching: PatchingConfiguration {
            maximum_patch_bytes: 12,
        },
        latent: LatentConfiguration { width: 96 },
        encoder: byte_codec(64, 2, 4),
        predictor: predictor(128, 4, 4, 32),
        decoder: byte_codec(64, 2, 4),
        adapter: AdapterConfiguration {
            rank: 4,
            scale: 8.0,
        },
    }
}

fn prototype_configuration() -> ModelConfiguration {
    ModelConfiguration {
        patching: standard_patching(),
        latent: LatentConfiguration { width: 512 },
        encoder: byte_codec(256, 3, 4),
        predictor: predictor(512, 8, 8, 128),
        decoder: byte_codec(256, 4, 4),
        adapter: standard_adapter(),
    }
}

fn small_configuration() -> ModelConfiguration {
    ModelConfiguration {
        patching: standard_patching(),
        latent: LatentConfiguration { width: 512 },
        encoder: byte_codec(384, 4, 6),
        predictor: predictor(768, 14, 12, 192),
        decoder: byte_codec(384, 6, 6),
        adapter: standard_adapter(),
    }
}

fn target_configuration() -> ModelConfiguration {
    ModelConfiguration {
        patching: standard_patching(),
        latent: LatentConfiguration { width: 512 },
        encoder: byte_codec(512, 4, 8),
        predictor: predictor(1024, 26, 16, 256),
        decoder: byte_codec(512, 6, 8),
        adapter: standard_adapter(),
    }
}

fn medium_configuration() -> ModelConfiguration {
    ModelConfiguration {
        patching: standard_patching(),
        latent: LatentConfiguration { width: 768 },
        encoder: byte_codec(640, 6, 8),
        predictor: predictor(1792, 32, 16, 512),
        decoder: byte_codec(640, 8, 8),
        adapter: AdapterConfiguration {
            rank: 16,
            scale: 32.0,
        },
    }
}

fn standard_patching() -> PatchingConfiguration {
    PatchingConfiguration {
        maximum_patch_bytes: 16,
    }
}

fn standard_adapter() -> AdapterConfiguration {
    AdapterConfiguration {
        rank: 8,
        scale: 16.0,
    }
}

fn byte_codec(width: usize, layer_count: usize, head_count: usize) -> ByteCodecConfiguration {
    ByteCodecConfiguration {
        width,
        layer_count,
        head_count,
    }
}

fn predictor(
    width: usize,
    layer_count: usize,
    head_count: usize,
    maximum_context_patches: usize,
) -> PredictorConfiguration {
    PredictorConfiguration {
        width,
        layer_count,
        head_count,
        maximum_context_patches,
        maximum_memory_slots: 4,
        thinking_step_count: 2,
    }
}

/// Parameter counts of each component, without building any weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterCounts {
    /// Parameters in the byte encoder.
    pub encoder: usize,
    /// Slow parameters in the concept predictor.
    pub predictor: usize,
    /// Parameters in the byte decoder.
    pub decoder: usize,
    /// Fast-weight parameters attached to the predictor.
    pub adapter: usize,
}

impl ParameterCounts {
    /// Slow weights across all three components.
    pub fn slow_total(&self) -> usize {
        self.encoder + self.predictor + self.decoder
    }
}

impl ModelConfiguration {
    /// Checks that every size is usable, returning the first problem found.
    pub fn validate(&self) -> BrainResult<()> {
        let size_checks = [
            (
                "patching.maximum_patch_bytes",
                self.patching.maximum_patch_bytes,
            ),
            ("latent.width", self.latent.width),
            ("encoder.layer_count", self.encoder.layer_count),
            ("decoder.layer_count", self.decoder.layer_count),
            ("predictor.layer_count", self.predictor.layer_count),
            (
                "predictor.maximum_context_patches",
                self.predictor.maximum_context_patches,
            ),
            ("adapter.rank", self.adapter.rank),
        ];
        if let Some((field, _)) = size_checks.iter().find(|(_, size)| *size == 0) {
            return invalid(format!("{field} must be greater than zero"));
        }
        if self.predictor.maximum_context_patches < 2 {
            return invalid("predictor.maximum_context_patches must be at least 2".into());
        }
        validate_heads("encoder", self.encoder.width, self.encoder.head_count)?;
        validate_heads("decoder", self.decoder.width, self.decoder.head_count)?;
        validate_heads("predictor", self.predictor.width, self.predictor.head_count)
    }

    /// Counts parameters analytically, so sizes can be shown before a brain is built.
    pub fn parameter_counts(&self) -> ParameterCounts {
        ParameterCounts {
            encoder: self.encoder_parameter_count(),
            predictor: self.predictor_parameter_count(),
            decoder: self.decoder_parameter_count(),
            adapter: self.adapter_parameter_count(),
        }
    }

    fn encoder_parameter_count(&self) -> usize {
        let width = self.encoder.width;
        let embeddings = (BYTE_VALUE_COUNT + self.patching.maximum_patch_bytes) * width;
        let blocks = self.encoder.layer_count * transformer_block_parameter_count(width);
        let output =
            normalization_parameter_count(width) + linear_parameter_count(width, self.latent.width);
        embeddings + blocks + output
    }

    fn decoder_parameter_count(&self) -> usize {
        let width = self.decoder.width;
        let position_count = self.patching.maximum_patch_bytes + 1;
        let inputs = linear_parameter_count(self.latent.width, width)
            + (BYTE_VALUE_COUNT + position_count) * width;
        let blocks = self.decoder.layer_count * transformer_block_parameter_count(width);
        let output = normalization_parameter_count(width)
            + linear_parameter_count(width, DECODER_SYMBOL_COUNT);
        inputs + blocks + output
    }

    fn predictor_parameter_count(&self) -> usize {
        let width = self.predictor.width;
        let latent_width = self.latent.width;
        let slot_count =
            self.predictor.maximum_context_patches + self.predictor.maximum_memory_slots;
        let latent_inputs = 3 * linear_parameter_count(latent_width, width);
        let latent_outputs = 2 * linear_parameter_count(width, latent_width);
        let normalizations = 2 * normalization_parameter_count(width);
        let block_count = self.predictor.layer_count + 1;
        latent_inputs
            + latent_outputs
            + normalizations
            + slot_count * width
            + block_count * transformer_block_parameter_count(width)
    }

    fn adapter_parameter_count(&self) -> usize {
        let width = self.predictor.width;
        let hidden_width = width * FEED_FORWARD_EXPANSION;
        let attention =
            ATTENTION_PROJECTION_COUNT * low_rank_parameter_count(width, width, self.adapter.rank);
        let feed_forward = 2 * low_rank_parameter_count(width, hidden_width, self.adapter.rank);
        (self.predictor.layer_count + 1) * (attention + feed_forward)
    }
}

fn invalid(reason: String) -> BrainResult<()> {
    Err(BrainError::InvalidConfiguration(reason))
}

fn validate_heads(component: &str, width: usize, head_count: usize) -> BrainResult<()> {
    if width == 0 || head_count == 0 {
        return invalid(format!(
            "{component} width and head_count must be greater than zero"
        ));
    }
    if !width.is_multiple_of(head_count) {
        return invalid(format!(
            "{component}.width ({width}) must be divisible by {component}.head_count ({head_count})"
        ));
    }
    Ok(())
}

fn linear_parameter_count(input_width: usize, output_width: usize) -> usize {
    input_width * output_width + output_width
}

fn normalization_parameter_count(width: usize) -> usize {
    NORMALIZATION_PARAMETERS_PER_UNIT * width
}

fn low_rank_parameter_count(input_width: usize, output_width: usize, rank: usize) -> usize {
    rank * (input_width + output_width)
}

fn transformer_block_parameter_count(width: usize) -> usize {
    let hidden_width = width * FEED_FORWARD_EXPANSION;
    let attention = ATTENTION_PROJECTION_COUNT * linear_parameter_count(width, width);
    let feed_forward =
        linear_parameter_count(width, hidden_width) + linear_parameter_count(hidden_width, width);
    attention + feed_forward + NORMALIZATIONS_PER_BLOCK * normalization_parameter_count(width)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MILLION: usize = 1_000_000;

    #[test]
    fn every_preset_is_valid() {
        for preset in ModelPreset::ALL {
            preset.configuration().validate().unwrap();
        }
    }

    #[test]
    fn presets_land_in_the_parameter_ranges_of_the_spec_sheet() {
        let expected_ranges = [
            (ModelPreset::Prototype, 20 * MILLION..=50 * MILLION),
            (ModelPreset::Small, 100 * MILLION..=150 * MILLION),
            (ModelPreset::Target, 300 * MILLION..=500 * MILLION),
            (ModelPreset::Medium, 1_000 * MILLION..=3_000 * MILLION),
        ];

        for (preset, expected_range) in expected_ranges {
            let slow_total = preset.configuration().parameter_counts().slow_total();
            assert!(
                expected_range.contains(&slow_total),
                "{} has {slow_total} parameters",
                preset.name()
            );
        }
    }

    #[test]
    fn preset_names_round_trip() {
        for preset in ModelPreset::ALL {
            assert_eq!(ModelPreset::from_name(preset.name()).unwrap(), preset);
        }
        assert!(ModelPreset::from_name("enormous").is_err());
    }

    #[test]
    fn head_count_must_divide_width() {
        let mut configuration = ModelPreset::Tiny.configuration();
        configuration.predictor.head_count = 5;

        assert!(configuration.validate().is_err());
    }
}
