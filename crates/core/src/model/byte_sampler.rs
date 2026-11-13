//! Chooses the next byte from the decoder's scores.

use crate::configuration::GenerationSettings;
use crate::random::RandomSource;

/// Picks symbols from score vectors using temperature and a candidate limit.
#[derive(Debug, Clone)]
pub struct ByteSampler {
    temperature: f64,
    candidate_count: usize,
    random: RandomSource,
}

impl ByteSampler {
    /// Creates a sampler following the generation settings.
    pub fn new(generation: &GenerationSettings) -> Self {
        let random = match generation.seed {
            Some(seed) => RandomSource::from_seed(seed),
            None => RandomSource::from_clock(),
        };
        Self {
            temperature: generation.temperature,
            candidate_count: generation.candidate_count.max(1),
            random,
        }
    }

    /// Creates a sampler that always picks the most likely symbol.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            candidate_count: 1,
            random: RandomSource::from_seed(0),
        }
    }

    /// Chooses one symbol index from unnormalised scores.
    pub fn choose(&mut self, symbol_scores: &[f32]) -> usize {
        let is_greedy = self.temperature <= 0.0 || self.candidate_count == 1;
        if is_greedy {
            return most_likely_symbol(symbol_scores);
        }

        let candidates = self.strongest_candidates(symbol_scores);
        let weights = self.candidate_weights(&candidates);
        let chosen_position = self.sample_position(&weights);
        candidates[chosen_position].0
    }

    /// The highest-scoring symbols, strongest first, as `(symbol, score)`.
    fn strongest_candidates(&self, symbol_scores: &[f32]) -> Vec<(usize, f32)> {
        let mut candidates: Vec<(usize, f32)> = symbol_scores.iter().copied().enumerate().collect();
        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
        candidates.truncate(self.candidate_count);
        candidates
    }

    /// Unnormalised probabilities after applying the temperature.
    fn candidate_weights(&self, candidates: &[(usize, f32)]) -> Vec<f64> {
        // Subtracting the best score keeps the exponentials from overflowing.
        let best_score = f64::from(candidates[0].1);
        candidates
            .iter()
            .map(|&(_, score)| ((f64::from(score) - best_score) / self.temperature).exp())
            .collect()
    }

    fn sample_position(&mut self, weights: &[f64]) -> usize {
        let total_weight: f64 = weights.iter().sum();
        let mut remaining = self.random.next_fraction() * total_weight;
        for (position, weight) in weights.iter().enumerate() {
            remaining -= weight;
            if remaining <= 0.0 {
                return position;
            }
        }
        weights.len() - 1
    }
}

fn most_likely_symbol(symbol_scores: &[f32]) -> usize {
    symbol_scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(symbol, _)| symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(temperature: f64, candidate_count: usize) -> GenerationSettings {
        GenerationSettings {
            temperature,
            candidate_count,
            seed: Some(4),
            ..GenerationSettings::default()
        }
    }

    #[test]
    fn zero_temperature_picks_the_highest_score() {
        let mut sampler = ByteSampler::new(&settings(0.0, 40));

        assert_eq!(sampler.choose(&[0.1, 2.5, -1.0, 2.4]), 1);
    }

    #[test]
    fn sampling_never_leaves_the_candidate_set() {
        let mut sampler = ByteSampler::new(&settings(1.5, 2));
        let scores = [5.0, 4.9, -3.0, -4.0, -5.0];

        let is_always_a_top_candidate = (0..500).all(|_| sampler.choose(&scores) <= 1);

        assert!(is_always_a_top_candidate);
    }

    #[test]
    fn sampling_follows_the_scores() {
        let mut sampler = ByteSampler::new(&settings(1.0, 3));
        let scores = [0.0, 3.0, 0.0];

        let favourite_count = (0..2_000).filter(|_| sampler.choose(&scores) == 1).count();

        assert!(favourite_count > 1_600);
    }
}
