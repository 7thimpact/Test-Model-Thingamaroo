//! A small seedable random source.
//!
//! Batch sampling, byte sampling and the sample corpus all draw from this
//! generator so that a seed reproduces the same run on every platform.

const SEED_MIXING_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_MIXING_MULTIPLIER_FIRST: u64 = 0xBF58_476D_1CE4_E5B9;
const SEED_MIXING_MULTIPLIER_SECOND: u64 = 0x94D0_49BB_1331_11EB;
const FRACTION_MANTISSA_BITS: u32 = 53;

/// Deterministic pseudo-random number generator (xoshiro256++).
///
/// Algorithm by Blackman and Vigna: <https://prng.di.unimi.it/xoshiro256plusplus.c>
#[derive(Debug, Clone)]
pub struct RandomSource {
    state: [u64; 4],
}

impl RandomSource {
    /// Creates a generator whose whole sequence is determined by `seed`.
    pub fn from_seed(seed: u64) -> Self {
        let mut mixing_state = seed;
        let state = std::array::from_fn(|_| next_seed_word(&mut mixing_state));
        Self { state }
    }

    /// Creates a generator seeded from the system clock, for runs that should differ.
    pub fn from_clock() -> Self {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self::from_seed(elapsed.as_nanos() as u64)
    }

    /// Returns the next 64 uniformly random bits.
    pub fn next_bits(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let shifted = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= shifted;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Returns a uniformly random fraction in `[0, 1)`.
    pub fn next_fraction(&mut self) -> f64 {
        let mantissa = self.next_bits() >> (u64::BITS - FRACTION_MANTISSA_BITS);
        mantissa as f64 / (1u64 << FRACTION_MANTISSA_BITS) as f64
    }

    /// Returns a uniformly random index in `[0, upper_bound)`.
    ///
    /// An `upper_bound` of zero yields zero so callers never divide by zero.
    pub fn next_below(&mut self, upper_bound: usize) -> usize {
        if upper_bound == 0 {
            return 0;
        }
        (self.next_fraction() * upper_bound as f64) as usize % upper_bound
    }

    /// Returns `true` with the given probability.
    pub fn next_chance(&mut self, probability: f64) -> bool {
        self.next_fraction() < probability
    }

    /// Picks one item uniformly at random, or `None` when there is nothing to pick.
    pub fn pick<'items, Item>(&mut self, items: &'items [Item]) -> Option<&'items Item> {
        if items.is_empty() {
            return None;
        }
        items.get(self.next_below(items.len()))
    }
}

/// One step of SplitMix64, used only to expand a seed into generator state.
fn next_seed_word(mixing_state: &mut u64) -> u64 {
    *mixing_state = mixing_state.wrapping_add(SEED_MIXING_INCREMENT);
    let mut word = *mixing_state;
    word = (word ^ (word >> 30)).wrapping_mul(SEED_MIXING_MULTIPLIER_FIRST);
    word = (word ^ (word >> 27)).wrapping_mul(SEED_MIXING_MULTIPLIER_SECOND);
    word ^ (word >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let mut first = RandomSource::from_seed(7);
        let mut second = RandomSource::from_seed(7);

        let first_draws: Vec<u64> = (0..8).map(|_| first.next_bits()).collect();
        let second_draws: Vec<u64> = (0..8).map(|_| second.next_bits()).collect();

        assert_eq!(first_draws, second_draws);
    }

    #[test]
    fn fractions_stay_inside_the_unit_interval() {
        let mut random = RandomSource::from_seed(11);

        let is_every_fraction_valid = (0..10_000)
            .map(|_| random.next_fraction())
            .all(|fraction| (0.0..1.0).contains(&fraction));

        assert!(is_every_fraction_valid);
    }

    #[test]
    fn indices_stay_below_the_upper_bound() {
        let mut random = RandomSource::from_seed(3);

        let is_every_index_valid = (0..10_000).all(|_| random.next_below(17) < 17);

        assert!(is_every_index_valid);
        assert_eq!(random.next_below(0), 0);
    }
}
