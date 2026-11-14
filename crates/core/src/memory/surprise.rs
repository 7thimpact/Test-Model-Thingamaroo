//! Turns prediction error into an importance score.
//!
//! "Surprise drives learning": an input matters in proportion to how much
//! worse than usual the model predicted it. The usual level is a running
//! average, so the score adapts as the model improves.

use serde::{Deserialize, Serialize};

use crate::configuration::SurpriseSettings;

/// Importance given to inputs before any baseline exists.
const NEUTRAL_IMPORTANCE: f64 = 0.5;

/// Smallest spread assumed around the baseline, as a fraction of its mean.
///
/// Without a floor, a run of near-identical inputs would shrink the spread
/// until every later input looked maximally surprising.
const MINIMUM_RELATIVE_DEVIATION: f64 = 0.1;

/// Spread assumed around a freshly seeded baseline, as a fraction of its mean.
const SEEDED_RELATIVE_DEVIATION: f64 = 0.25;

/// The model's usual prediction error, tracked as a running mean and variance.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SurpriseBaseline {
    /// Typical bits per byte needed to predict an input.
    pub mean_bits_per_byte: f64,
    /// Variance of bits per byte around the mean.
    pub variance: f64,
    /// Number of inputs the baseline has seen.
    pub observation_count: u64,
}

impl SurpriseBaseline {
    /// Starts a baseline from a known typical error, such as the validation loss.
    pub fn seeded(mean_bits_per_byte: f64) -> Self {
        let deviation = mean_bits_per_byte * SEEDED_RELATIVE_DEVIATION;
        Self {
            mean_bits_per_byte,
            variance: deviation * deviation,
            observation_count: 1,
        }
    }

    /// Scores how unusual a prediction error is, from 0 (routine) to 1 (astonishing).
    ///
    /// The score is the logistic of the error's distance from the baseline in
    /// standard deviations, so an average input scores 0.5.
    pub fn importance_of(&self, bits_per_byte: f64) -> f64 {
        if self.observation_count == 0 {
            return NEUTRAL_IMPORTANCE;
        }
        let deviation_floor = self.mean_bits_per_byte.abs() * MINIMUM_RELATIVE_DEVIATION;
        let deviation = self.variance.sqrt().max(deviation_floor).max(f64::EPSILON);
        let standard_score = (bits_per_byte - self.mean_bits_per_byte) / deviation;
        1.0 / (1.0 + (-standard_score).exp())
    }

    /// Folds a new prediction error into the running mean and variance.
    pub fn observe(&mut self, bits_per_byte: f64, surprise: &SurpriseSettings) {
        if self.observation_count == 0 {
            *self = Self::seeded(bits_per_byte);
            return;
        }
        let momentum = surprise.baseline_momentum.clamp(0.0, 1.0);
        let distance = bits_per_byte - self.mean_bits_per_byte;
        self.mean_bits_per_byte += (1.0 - momentum) * distance;
        self.variance = momentum * (self.variance + (1.0 - momentum) * distance * distance);
        self.observation_count += 1;
    }
}

/// Whether a message opens with one of the configured correction markers.
pub fn opens_with_correction_marker(message: &str, surprise: &SurpriseSettings) -> bool {
    let opening = message.trim_start().to_lowercase();
    surprise
        .correction_markers
        .iter()
        .any(|marker| !marker.is_empty() && opening.starts_with(&marker.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_average_input_is_neutral_and_a_bad_prediction_is_important() {
        let baseline = SurpriseBaseline::seeded(2.0);

        assert!((baseline.importance_of(2.0) - 0.5).abs() < 1e-9);
        assert!(baseline.importance_of(4.0) > 0.95);
        assert!(baseline.importance_of(1.0) < 0.2);
    }

    #[test]
    fn the_first_input_is_neutral_because_nothing_is_known_yet() {
        let baseline = SurpriseBaseline::default();

        assert_eq!(baseline.importance_of(7.0), NEUTRAL_IMPORTANCE);
    }

    #[test]
    fn the_baseline_moves_towards_what_it_observes() {
        let surprise = SurpriseSettings::default();
        let mut baseline = SurpriseBaseline::seeded(2.0);

        for _ in 0..200 {
            baseline.observe(3.0, &surprise);
        }

        assert!((baseline.mean_bits_per_byte - 3.0).abs() < 0.01);
        assert!(baseline.importance_of(3.0) < 0.6);
    }

    #[test]
    fn correction_markers_match_only_at_the_start() {
        let surprise = SurpriseSettings::default();

        assert!(opens_with_correction_marker(
            "  No, the cat says quack.",
            &surprise
        ));
        assert!(opens_with_correction_marker(
            "Actually it is blue",
            &surprise
        ));
        assert!(!opens_with_correction_marker("I know nothing", &surprise));
    }
}
