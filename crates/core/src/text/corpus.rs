//! Training text on disk, read in small random windows.
//!
//! Files are never loaded whole, so a corpus far larger than RAM can be used:
//! every sample is one seek and one short read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{BrainError, BrainResult};
use crate::random::RandomSource;
use crate::text::patcher::{Patcher, is_delimiter};

/// A sequence of patches, each owning its bytes.
pub type PatchSequence = Vec<Vec<u8>>;

const SAMPLING_ATTEMPT_LIMIT: usize = 64;

/// Which part of the corpus a sample is drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusSplit {
    /// Text the model learns from.
    Training,
    /// Held-out text used only to measure progress.
    Validation,
}

/// One text file of the corpus and where it sits in the combined byte range.
#[derive(Debug, Clone)]
struct CorpusFile {
    path: PathBuf,
    size_bytes: u64,
    first_byte_offset: u64,
}

/// A collection of text files addressed as one long byte range.
#[derive(Debug, Clone)]
pub struct Corpus {
    files: Vec<CorpusFile>,
    total_bytes: u64,
    validation_start_offset: u64,
    patcher: Patcher,
}

impl Corpus {
    /// Opens the given files and directories; directories are searched recursively.
    pub fn open(
        paths: &[PathBuf],
        validation_fraction: f64,
        patcher: Patcher,
    ) -> BrainResult<Self> {
        let mut file_paths = Vec::new();
        for path in paths {
            collect_files(path, &mut file_paths)?;
        }
        file_paths.sort();

        let files = measure_files(&file_paths)?;
        let total_bytes: u64 = files.iter().map(|file| file.size_bytes).sum();
        if total_bytes == 0 {
            return Err(BrainError::Corpus("the corpus contains no text".into()));
        }

        let training_bytes = total_bytes as f64 * (1.0 - validation_fraction);
        Ok(Self {
            files,
            total_bytes,
            validation_start_offset: training_bytes as u64,
            patcher,
        })
    }

    /// Total size of all files.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Number of files in the corpus.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Draws `patch_count` consecutive patches starting at a random word.
    pub fn sample_patches(
        &self,
        random: &mut RandomSource,
        split: CorpusSplit,
        patch_count: usize,
    ) -> BrainResult<PatchSequence> {
        for _ in 0..SAMPLING_ATTEMPT_LIMIT {
            let window = self.read_random_window(random, split, patch_count)?;
            let patches = self
                .patcher
                .split_prefix(skip_to_word_start(&window), patch_count);
            if patches.len() == patch_count {
                return Ok(patches.into_iter().map(<[u8]>::to_vec).collect());
            }
        }
        Err(BrainError::Corpus(format!(
            "could not find {patch_count} consecutive patches; the corpus is too small for this context size"
        )))
    }

    /// Draws a raw text window large enough to hold `patch_count` patches.
    pub fn sample_window(
        &self,
        random: &mut RandomSource,
        split: CorpusSplit,
        patch_count: usize,
    ) -> BrainResult<Vec<u8>> {
        let patches = self.sample_patches(random, split, patch_count)?;
        Ok(patches.concat())
    }

    fn read_random_window(
        &self,
        random: &mut RandomSource,
        split: CorpusSplit,
        patch_count: usize,
    ) -> BrainResult<Vec<u8>> {
        let (range_start, range_end) = self.split_range(split);
        let range_bytes = (range_end - range_start) as usize;
        let window_start = range_start + random.next_below(range_bytes) as u64;
        // One extra patch of slack pays for the bytes skipped to reach a word start.
        let window_bytes = (patch_count + 1) * self.patcher.maximum_patch_bytes();
        self.read_window(window_start, window_bytes)
    }

    fn split_range(&self, split: CorpusSplit) -> (u64, u64) {
        let has_validation_text = self.validation_start_offset < self.total_bytes;
        match split {
            CorpusSplit::Validation if has_validation_text => {
                (self.validation_start_offset, self.total_bytes)
            }
            CorpusSplit::Validation => (0, self.total_bytes),
            CorpusSplit::Training => (0, self.validation_start_offset.max(1)),
        }
    }

    /// Reads up to `window_bytes` bytes starting at a position in the combined range.
    ///
    /// The window never crosses a file boundary, so it may be shorter than requested.
    fn read_window(&self, global_offset: u64, window_bytes: usize) -> BrainResult<Vec<u8>> {
        let file = self.file_containing(global_offset);
        let offset_in_file = global_offset - file.first_byte_offset;
        let readable_bytes = (file.size_bytes - offset_in_file).min(window_bytes as u64);

        let mut handle =
            File::open(&file.path).map_err(|source| BrainError::file_system(&file.path, source))?;
        handle
            .seek(SeekFrom::Start(offset_in_file))
            .map_err(|source| BrainError::file_system(&file.path, source))?;

        let mut window = vec![0u8; readable_bytes as usize];
        handle
            .read_exact(&mut window)
            .map_err(|source| BrainError::file_system(&file.path, source))?;
        Ok(window)
    }

    fn file_containing(&self, global_offset: u64) -> &CorpusFile {
        let file_index = self
            .files
            .partition_point(|file| file.first_byte_offset + file.size_bytes <= global_offset);
        &self.files[file_index.min(self.files.len() - 1)]
    }
}

/// Drops the partial word a random offset usually lands in.
fn skip_to_word_start(window: &[u8]) -> &[u8] {
    let word_start = window
        .windows(2)
        .position(|pair| is_delimiter(pair[0]) && !is_delimiter(pair[1]))
        .map_or(0, |position| position + 1);
    &window[word_start..]
}

fn collect_files(path: &Path, file_paths: &mut Vec<PathBuf>) -> BrainResult<()> {
    let metadata =
        std::fs::metadata(path).map_err(|source| BrainError::file_system(path, source))?;
    if metadata.is_file() {
        file_paths.push(path.to_path_buf());
        return Ok(());
    }

    let entries =
        std::fs::read_dir(path).map_err(|source| BrainError::file_system(path, source))?;
    for entry in entries {
        let entry = entry.map_err(|source| BrainError::file_system(path, source))?;
        if is_hidden(&entry.path()) {
            continue;
        }
        collect_files(&entry.path(), file_paths)?;
    }
    Ok(())
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

fn measure_files(file_paths: &[PathBuf]) -> BrainResult<Vec<CorpusFile>> {
    let mut files = Vec::new();
    let mut first_byte_offset = 0;
    for path in file_paths {
        let size_bytes = std::fs::metadata(path)
            .map_err(|source| BrainError::file_system(path, source))?
            .len();
        if size_bytes == 0 {
            continue;
        }
        files.push(CorpusFile {
            path: path.clone(),
            size_bytes,
            first_byte_offset,
        });
        first_byte_offset += size_bytes;
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_corpus(name: &str, text: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("thingamaroo-corpus-{name}"));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("corpus.txt");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn sampled_patches_are_whole_words_from_the_text() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa ".repeat(40);
        let path = write_corpus("words", &text);
        let corpus = Corpus::open(&[path], 0.1, Patcher::new(12)).unwrap();
        let mut random = RandomSource::from_seed(5);

        let patches = corpus
            .sample_patches(&mut random, CorpusSplit::Training, 6)
            .unwrap();

        assert_eq!(patches.len(), 6);
        let is_every_patch_a_whole_word = patches.iter().all(|patch| {
            text.contains(std::str::from_utf8(patch).unwrap()) && patch.ends_with(b" ")
        });
        assert!(is_every_patch_a_whole_word);
    }

    #[test]
    fn a_corpus_that_is_too_small_is_reported() {
        let path = write_corpus("small", "only three words ");
        let corpus = Corpus::open(&[path], 0.0, Patcher::new(12)).unwrap();
        let mut random = RandomSource::from_seed(5);

        let outcome = corpus.sample_patches(&mut random, CorpusSplit::Training, 50);

        assert!(matches!(outcome, Err(BrainError::Corpus(_))));
    }

    #[test]
    fn an_empty_corpus_is_rejected() {
        let path = write_corpus("empty", "");

        let outcome = Corpus::open(&[path], 0.0, Patcher::new(12));

        assert!(matches!(outcome, Err(BrainError::Corpus(_))));
    }
}
