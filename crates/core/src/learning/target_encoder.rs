//! A slowly updated copy of the encoder that supplies latent prediction targets.
//!
//! Predicting targets from a copy that lags behind the trained encoder (as in
//! JEPA) removes the shortcut where encoder and predictor agree on a constant.

use burn::module::Module;

use crate::learning::parameter_tensors::ParameterTensors;
use crate::model::ByteEncoder;

/// An encoder whose weights are an exponential moving average of another encoder's.
#[derive(Debug, Clone)]
pub struct TargetEncoder {
    encoder: ByteEncoder,
    momentum: f64,
}

impl TargetEncoder {
    /// Starts as an exact, gradient-free copy of the trained encoder.
    pub fn tracking(trained_encoder: &ByteEncoder, momentum: f64) -> Self {
        Self::resumed(trained_encoder.valid(), momentum)
    }

    /// Continues from target weights saved by an earlier run.
    pub fn resumed(saved_encoder: ByteEncoder, momentum: f64) -> Self {
        Self {
            encoder: saved_encoder,
            momentum: momentum.clamp(0.0, 1.0),
        }
    }

    /// The target encoder itself.
    pub fn encoder(&self) -> &ByteEncoder {
        &self.encoder
    }

    /// Moves the target weights a small step towards the trained encoder's weights.
    pub fn follow(&mut self, trained_encoder: &ByteEncoder) {
        let trained_values = ParameterTensors::values_of(trained_encoder);
        let momentum = self.momentum;
        let blend_towards_trained =
            move |target: burn::tensor::Tensor<1>, trained: burn::tensor::Tensor<1>| {
                target.mul_scalar(momentum) + trained.mul_scalar(1.0 - momentum)
            };
        self.encoder = trained_values.blend_into(self.encoder.clone(), &blend_towards_trained);
    }
}
