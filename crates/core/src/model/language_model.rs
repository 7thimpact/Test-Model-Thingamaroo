//! The three slow-weight components bundled as one module.

use burn::module::Module;
use burn::tensor::Device;

use crate::configuration::ModelConfiguration;
use crate::model::byte_decoder::ByteDecoder;
use crate::model::byte_encoder::ByteEncoder;
use crate::model::concept_predictor::ConceptPredictor;

/// Encoder, predictor and decoder: everything that is saved as the slow weights.
#[derive(Module, Debug)]
pub struct LanguageModel {
    /// Reads patches of bytes into meaning vectors.
    pub encoder: ByteEncoder,
    /// Predicts the next meaning vector.
    pub predictor: ConceptPredictor,
    /// Writes bytes from a meaning vector.
    pub decoder: ByteDecoder,
}

impl LanguageModel {
    /// Creates a model with freshly initialised weights.
    pub fn new(model_configuration: &ModelConfiguration, device: &Device) -> Self {
        Self {
            encoder: ByteEncoder::new(model_configuration, device),
            predictor: ConceptPredictor::new(model_configuration, device),
            decoder: ByteDecoder::new(model_configuration, device),
        }
    }
}
