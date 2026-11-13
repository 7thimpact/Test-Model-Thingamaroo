//! Per-parameter tensors keyed by parameter identity.
//!
//! The target encoder and elastic weight consolidation both need "one tensor
//! per weight tensor of a module" (a copy, an average, an importance). This
//! type holds such a set, flattened so tensors of any rank fit one map.

use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::{Gradients, Tensor};

/// One flattened, gradient-free tensor per float parameter of a module.
#[derive(Debug, Clone, Default)]
pub struct ParameterTensors {
    tensors_by_id: HashMap<ParamId, Tensor<1>>,
}

impl ParameterTensors {
    /// Copies the current value of every float parameter.
    pub fn values_of<Network: Module>(network: &Network) -> Self {
        let mut collector = ValueCollector::default();
        network.visit(&mut collector);
        collector.collected
    }

    /// Squares the gradient of every float parameter that received one.
    pub fn squared_gradients_of<Network: Module>(network: &Network, gradients: &Gradients) -> Self {
        let mut collector = SquaredGradientCollector {
            gradients,
            collected: Self::default(),
        };
        network.visit(&mut collector);
        collector.collected
    }

    /// The tensor stored for a parameter, if any.
    pub fn get(&self, id: &ParamId) -> Option<&Tensor<1>> {
        self.tensors_by_id.get(id)
    }

    /// Number of parameters covered.
    pub fn parameter_count(&self) -> usize {
        self.tensors_by_id.len()
    }

    /// Adds another set element-wise; parameters missing here are taken as they are.
    pub fn accumulate(&mut self, other: Self) {
        for (id, tensor) in other.tensors_by_id {
            let summed = match self.tensors_by_id.remove(&id) {
                Some(existing) => existing + tensor,
                None => tensor,
            };
            self.tensors_by_id.insert(id, summed);
        }
    }

    /// Multiplies every tensor by a constant.
    pub fn scaled_by(self, factor: f64) -> Self {
        let tensors_by_id = self
            .tensors_by_id
            .into_iter()
            .map(|(id, tensor)| (id, tensor.mul_scalar(factor)))
            .collect();
        Self { tensors_by_id }
    }

    /// Replaces each parameter of `network` with `blend(current, stored)`.
    ///
    /// Parameters without a stored tensor are left unchanged.
    pub fn blend_into<Network: Module>(
        &self,
        network: Network,
        blend: &dyn Fn(Tensor<1>, Tensor<1>) -> Tensor<1>,
    ) -> Network {
        network.map(&mut Blender {
            stored: self,
            blend,
        })
    }

    /// Overwrites each parameter of `network` with its stored tensor.
    pub fn write_into<Network: Module>(&self, network: Network) -> Network {
        self.blend_into(network, &|_, stored| stored)
    }
}

#[derive(Default)]
struct ValueCollector {
    collected: ParameterTensors,
}

impl ModuleVisitor for ValueCollector {
    fn visit_float<const RANK: usize>(&mut self, parameter: &Param<Tensor<RANK>>) {
        let flattened = parameter.val().inner().reshape([-1]);
        self.collected.tensors_by_id.insert(parameter.id, flattened);
    }
}

struct SquaredGradientCollector<'gradients> {
    gradients: &'gradients Gradients,
    collected: ParameterTensors,
}

impl ModuleVisitor for SquaredGradientCollector<'_> {
    fn visit_float<const RANK: usize>(&mut self, parameter: &Param<Tensor<RANK>>) {
        let Some(gradient) = parameter.val().grad(self.gradients) else {
            return;
        };
        let flattened = gradient.square().reshape([-1]);
        self.collected.tensors_by_id.insert(parameter.id, flattened);
    }
}

struct Blender<'stored> {
    stored: &'stored ParameterTensors,
    blend: &'stored dyn Fn(Tensor<1>, Tensor<1>) -> Tensor<1>,
}

impl ModuleMapper for Blender<'_> {
    fn map_float<const RANK: usize>(
        &mut self,
        parameter: Param<Tensor<RANK>>,
    ) -> Param<Tensor<RANK>> {
        let Some(stored) = self.stored.get(&parameter.id).cloned() else {
            return parameter;
        };
        let blend = self.blend;
        parameter.map(|current| {
            let shape = current.shape();
            blend(current.reshape([-1]), stored).reshape(shape)
        })
    }
}
