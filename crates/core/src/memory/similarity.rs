//! Similarity between meaning vectors.

/// Cosine similarity of two vectors; zero when either has no length.
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let mut dot_product = 0.0f64;
    let mut left_squared = 0.0f64;
    let mut right_squared = 0.0f64;
    for (&left_value, &right_value) in left.iter().zip(right) {
        dot_product += f64::from(left_value) * f64::from(right_value);
        left_squared += f64::from(left_value) * f64::from(left_value);
        right_squared += f64::from(right_value) * f64::from(right_value);
    }

    let magnitude_product = (left_squared * right_squared).sqrt();
    if magnitude_product == 0.0 {
        return 0.0;
    }
    dot_product / magnitude_product
}

/// Keeps the best-scoring items seen so far, best first.
#[derive(Debug, Clone)]
pub struct BestMatches<Item> {
    capacity: usize,
    minimum_score: f64,
    scored_items: Vec<(f64, Item)>,
}

impl<Item> BestMatches<Item> {
    /// Creates a collector that keeps at most `capacity` items scoring at least `minimum_score`.
    pub fn new(capacity: usize, minimum_score: f64) -> Self {
        Self {
            capacity,
            minimum_score,
            scored_items: Vec::with_capacity(capacity + 1),
        }
    }

    /// Offers an item; it is kept only if it ranks among the best so far.
    pub fn offer(&mut self, score: f64, item: Item) {
        if score < self.minimum_score || self.capacity == 0 {
            return;
        }
        let insert_position = self
            .scored_items
            .partition_point(|(kept_score, _)| *kept_score >= score);
        if insert_position >= self.capacity {
            return;
        }
        self.scored_items.insert(insert_position, (score, item));
        self.scored_items.truncate(self.capacity);
    }

    /// The kept items with their scores, best first.
    pub fn into_sorted(self) -> Vec<(f64, Item)> {
        self.scored_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_directions_score_one_and_opposite_minus_one() {
        assert!((cosine_similarity(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-9);
        assert!((cosine_similarity(&[1.0, 0.0], &[-3.0, 0.0]) + 1.0).abs() < 1e-9);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn only_the_best_items_above_the_threshold_are_kept() {
        let mut best_matches = BestMatches::new(2, 0.5);
        for (score, name) in [(0.6, "fair"), (0.9, "best"), (0.2, "poor"), (0.7, "good")] {
            best_matches.offer(score, name);
        }

        let kept: Vec<&str> = best_matches
            .into_sorted()
            .into_iter()
            .map(|(_, name)| name)
            .collect();

        assert_eq!(kept, ["best", "good"]);
    }
}
