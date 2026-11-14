//! A small sample of the original training text, kept for replay during sleep.
//!
//! Sleep mixes these windows with new memories so that consolidating fresh
//! knowledge does not overwrite old skills, even when the original corpus is
//! no longer on disk.

use std::path::Path;

use crate::error::{BrainError, BrainResult};
use crate::random::RandomSource;
use crate::text::corpus::{Corpus, CorpusSplit};

const FILE_SIGNATURE: &[u8; 8] = b"TMTRPLY1";
const LENGTH_FIELD_BYTES: usize = 4;

/// Windows of original training text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplayReservoir {
    windows: Vec<Vec<u8>>,
}

impl ReplayReservoir {
    /// Draws `window_count` random training windows of `patch_count` patches each.
    pub fn sample_from(
        corpus: &Corpus,
        random: &mut RandomSource,
        window_count: usize,
        patch_count: usize,
    ) -> BrainResult<Self> {
        let windows = (0..window_count)
            .map(|_| corpus.sample_window(random, CorpusSplit::Training, patch_count))
            .collect::<BrainResult<Vec<_>>>()?;
        Ok(Self { windows })
    }

    /// Builds a reservoir from windows that are already in memory.
    pub fn from_windows(windows: Vec<Vec<u8>>) -> Self {
        Self { windows }
    }

    /// Number of stored windows.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Whether there is nothing to replay.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Picks one stored window at random.
    pub fn pick(&self, random: &mut RandomSource) -> Option<&[u8]> {
        random.pick(&self.windows).map(Vec::as_slice)
    }

    /// Writes the reservoir as length-prefixed windows.
    pub fn save(&self, path: &Path) -> BrainResult<()> {
        let mut contents = Vec::from(FILE_SIGNATURE.as_slice());
        contents.extend((self.windows.len() as u32).to_le_bytes());
        for window in &self.windows {
            contents.extend((window.len() as u32).to_le_bytes());
            contents.extend(window);
        }
        std::fs::write(path, contents).map_err(|source| BrainError::file_system(path, source))
    }

    /// Reads a reservoir; a missing file is an empty reservoir.
    pub fn load(path: &Path) -> BrainResult<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents =
            std::fs::read(path).map_err(|source| BrainError::file_system(path, source))?;
        let Some(mut remaining) = contents.strip_prefix(FILE_SIGNATURE.as_slice()) else {
            return Err(BrainError::document(path, "not a replay reservoir file"));
        };

        let window_count = take_length(&mut remaining, path)?;
        let windows = (0..window_count)
            .map(|_| take_window(&mut remaining, path))
            .collect::<BrainResult<Vec<_>>>()?;
        Ok(Self { windows })
    }
}

fn take_length(remaining: &mut &[u8], path: &Path) -> BrainResult<usize> {
    let Some((length_field, rest)) = remaining.split_at_checked(LENGTH_FIELD_BYTES) else {
        return Err(BrainError::document(
            path,
            "replay reservoir file is truncated",
        ));
    };
    *remaining = rest;
    let length_bytes: [u8; LENGTH_FIELD_BYTES] = length_field
        .try_into()
        .map_err(|_| BrainError::document(path, "replay reservoir length field is malformed"))?;
    Ok(u32::from_le_bytes(length_bytes) as usize)
}

fn take_window(remaining: &mut &[u8], path: &Path) -> BrainResult<Vec<u8>> {
    let window_bytes = take_length(remaining, path)?;
    let Some((window, rest)) = remaining.split_at_checked(window_bytes) else {
        return Err(BrainError::document(
            path,
            "replay reservoir file is truncated",
        ));
    };
    *remaining = rest;
    Ok(window.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reservoir_round_trips_through_its_file() {
        let reservoir = ReplayReservoir::from_windows(vec![
            b"the cat sat ".to_vec(),
            vec![0, 255, 10, 13],
            Vec::new(),
        ]);
        let path = std::env::temp_dir().join("thingamaroo-reservoir-round-trip.bin");

        reservoir.save(&path).unwrap();
        let restored = ReplayReservoir::load(&path).unwrap();

        assert_eq!(restored, reservoir);
    }

    #[test]
    fn a_missing_file_is_an_empty_reservoir() {
        let path = std::env::temp_dir().join("thingamaroo-reservoir-does-not-exist.bin");

        let reservoir = ReplayReservoir::load(&path).unwrap();

        assert!(reservoir.is_empty());
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let path = std::env::temp_dir().join("thingamaroo-reservoir-truncated.bin");
        let mut contents = Vec::from(FILE_SIGNATURE.as_slice());
        contents.extend(2u32.to_le_bytes());
        contents.extend(50u32.to_le_bytes());
        std::fs::write(&path, contents).unwrap();

        assert!(ReplayReservoir::load(&path).is_err());
    }
}
