//! Elastic weight consolidation: weights that mattered for old skills resist change.
//!
//! After pretraining (and after every sleep) the predictor's weights are
//! recorded together with an estimate of how much the old-data loss depends on
//! each one. During later sleeps, moving an important weight away from its
//! recorded value is penalised in proportion to that importance.
//!
//! Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
//! networks", 2017: <https://arxiv.org/abs/1612.00796>

use std::path::Path;

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::Tensor;

use crate::error::{BrainError, BrainResult};
use crate::learning::parameter_tensors::ParameterTensors;
use crate::model::ConceptPredictor;

/// The recorded weights and how much each one matters.
#[derive(Debug, Clone)]
pub struct ElasticAnchor {
    anchored_values: ParameterTensors,
    importances: ParameterTensors,
}

impl ElasticAnchor {
    /// Records the predictor's weights and estimates their importance.
    ///
    /// `loss_of_old_batch` must return the loss of the given predictor on a
    /// fresh batch of old data each time it is called. Importance is the
    /// squared gradient of that loss averaged over `batch_count` batches (the
    /// diagonal of the empirical Fisher information).
    pub fn estimate(
        predictor: &ConceptPredictor,
        batch_count: usize,
        mut loss_of_old_batch: impl FnMut(&ConceptPredictor) -> BrainResult<Tensor<1>>,
    ) -> BrainResult<Self> {
        let mut summed_importances = ParameterTensors::default();
        for _ in 0..batch_count {
            let gradients = loss_of_old_batch(predictor)?.backward();
            summed_importances.accumulate(ParameterTensors::squared_gradients_of(
                predictor, &gradients,
            ));
        }
        Ok(Self {
            anchored_values: ParameterTensors::values_of(predictor),
            importances: summed_importances.scaled_by(1.0 / batch_count.max(1) as f64),
        })
    }

    /// The penalty for how far the predictor has moved from the recorded weights.
    ///
    /// Half the importance-weighted squared distance, summed over all weights:
    /// a second-order estimate of how much the old-data loss has risen.
    pub fn penalty(&self, predictor: &ConceptPredictor) -> Tensor<1> {
        let mut visitor = PenaltyVisitor {
            anchor: self,
            penalty: None,
        };
        predictor.visit(&mut visitor);
        match visitor.penalty {
            Some(penalty) => penalty.mul_scalar(0.5),
            None => Tensor::zeros([1], &predictor.devices()[0]),
        }
    }

    /// Saves the anchor as two weight files shaped like the predictor.
    pub fn save(
        &self,
        predictor: &ConceptPredictor,
        values_path: &Path,
        importances_path: &Path,
    ) -> BrainResult<()> {
        let template = predictor.valid();
        self.anchored_values
            .write_into(template.clone())
            .save_file(values_path)
            .map_err(weights_error)?;
        self.importances
            .write_into(template)
            .save_file(importances_path)
            .map_err(weights_error)
    }

    /// Loads an anchor saved with [`ElasticAnchor::save`]; `None` when no anchor exists yet.
    pub fn load(
        predictor: &ConceptPredictor,
        values_path: &Path,
        importances_path: &Path,
    ) -> BrainResult<Option<Self>> {
        if !values_path.exists() || !importances_path.exists() {
            return Ok(None);
        }
        let template = predictor.valid();
        let anchored = template
            .clone()
            .try_load_file(values_path)
            .map_err(weights_error)?;
        let importances = template
            .try_load_file(importances_path)
            .map_err(weights_error)?;
        Ok(Some(Self {
            anchored_values: ParameterTensors::values_of(&anchored),
            importances: ParameterTensors::values_of(&importances),
        }))
    }
}

struct PenaltyVisitor<'anchor> {
    anchor: &'anchor ElasticAnchor,
    penalty: Option<Tensor<1>>,
}

impl ModuleVisitor for PenaltyVisitor<'_> {
    fn visit_float<const RANK: usize>(&mut self, parameter: &Param<Tensor<RANK>>) {
        let anchored_value = self.anchor.anchored_values.get(&parameter.id);
        let importance = self.anchor.importances.get(&parameter.id);
        let (Some(anchored_value), Some(importance)) = (anchored_value, importance) else {
            return;
        };

        let distance = parameter.val().reshape([-1]) - anchored_value.clone();
        let weighted_distance = (distance.square() * importance.clone()).sum();
        self.penalty = Some(match self.penalty.take() {
            Some(penalty) => penalty + weighted_distance,
            None => weighted_distance,
        });
    }
}

fn weights_error(error: impl std::fmt::Display) -> BrainError {
    BrainError::Weights(error.to_string())
}
