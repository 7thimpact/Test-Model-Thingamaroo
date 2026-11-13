//! A module paired with the optimiser that updates it.

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::Tensor;

use crate::configuration::OptimizerOptions;

/// Builds the AdamW optimiser used for every kind of learning in this crate.
pub fn build_optimizer(optimizer: &OptimizerOptions) -> ModuleOptimizer {
    let gradient_clipping = (optimizer.gradient_clip_norm > 0.0).then_some(
        GradientClippingConfig::Norm(optimizer.gradient_clip_norm as f32),
    );
    AdamWConfig::new()
        .with_weight_decay(optimizer.weight_decay as f32)
        .with_grad_clipping(gradient_clipping)
        .init()
}

/// A network and its optimiser, stepped together.
///
/// The optimiser consumes the network on every step and hands back the updated
/// one. Holding the network in an `Option` lets a step move it out without
/// cloning, which keeps weight updates in place.
pub struct TrainableModule<Network: Module> {
    network: Option<Network>,
    optimizer: ModuleOptimizer,
}

impl<Network: Module> TrainableModule<Network> {
    /// Pairs a network with an optimiser.
    pub fn new(network: Network, optimizer: ModuleOptimizer) -> Self {
        Self {
            network: Some(network),
            optimizer,
        }
    }

    /// The network in its current state.
    pub fn network(&self) -> &Network {
        self.network
            .as_ref()
            .expect("the network is only absent while a step is in progress")
    }

    /// Takes one gradient step that reduces `loss`.
    ///
    /// Only parameters that require gradients are changed, so frozen weights
    /// stay exactly as they are.
    pub fn descend(&mut self, loss: Tensor<1>, learning_rate: f64) {
        let network = self
            .network
            .take()
            .expect("the network is only absent while a step is in progress");
        let gradients = GradientsParams::from_grads(loss.backward(), &network);
        self.network = Some(self.optimizer.step(learning_rate, network, gradients));
    }

    /// Splits the pair back into its parts.
    pub fn into_parts(self) -> (Network, ModuleOptimizer) {
        let network = self
            .network
            .expect("the network is only absent while a step is in progress");
        (network, self.optimizer)
    }
}
