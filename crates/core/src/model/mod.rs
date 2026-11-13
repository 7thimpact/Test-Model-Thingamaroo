//! The four-part model: byte encoder, concept predictor, byte decoder, and how they are used.

mod byte_decoder;
mod byte_encoder;
mod byte_sampler;
mod compute_device;
mod concept_predictor;
mod inference;
mod language_model;
mod objective;
mod patch_batch;
mod reply;
mod tensor_readback;
mod transformer;

pub use byte_decoder::ByteDecoder;
pub use byte_encoder::ByteEncoder;
pub use byte_sampler::ByteSampler;
pub use compute_device::{ComputeDevice, DeviceChoice};
pub use concept_predictor::{ADAPTED_WEIGHT_PATTERN, ConceptPredictor, FAST_WEIGHT_PATTERN};
pub use inference::{InferenceModel, PredictionError, average_vectors};
pub use language_model::LanguageModel;
pub use objective::{
    NATS_PER_BIT, NextPatchEvaluation, NextPatchInput, WeightedObjective, evaluate_next_patch,
    latent_deviation, real_patch_selection, reconstruction_accuracy, reconstruction_loss,
    variance_penalty,
};
pub use patch_batch::PatchBatch;
pub use reply::{Reply, ReplyRequest, StopReason, ThoughtStage, ThoughtTrace, write_reply};
pub use tensor_readback::{read_scalar, read_values};
