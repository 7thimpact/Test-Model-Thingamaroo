//! Groups raw bytes into variable-size patches.
//!
//! A patch is one word-like chunk: a run of bytes followed by the whitespace
//! that ends it. Nothing here depends on a vocabulary, so any byte stream in
//! any language can be patched; text without whitespace simply falls back to
//! fixed-size chunks that never split a UTF-8 character.

const UTF8_CONTINUATION_MASK: u8 = 0b1100_0000;
const UTF8_CONTINUATION_PATTERN: u8 = 0b1000_0000;

/// Splits byte streams into patches of at most `maximum_patch_bytes` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Patcher {
    maximum_patch_bytes: usize,
}

impl Patcher {
    /// Creates a patcher; a limit of zero is raised to one so splitting always advances.
    pub fn new(maximum_patch_bytes: usize) -> Self {
        Self {
            maximum_patch_bytes: maximum_patch_bytes.max(1),
        }
    }

    /// Longest patch this patcher produces.
    pub fn maximum_patch_bytes(&self) -> usize {
        self.maximum_patch_bytes
    }

    /// Splits the whole text into consecutive patches that concatenate back to it.
    pub fn split<'text>(&self, text: &'text [u8]) -> Vec<&'text [u8]> {
        self.split_prefix(text, usize::MAX)
    }

    /// Splits the start of the text, stopping once `maximum_patch_count` patches exist.
    pub fn split_prefix<'text>(
        &self,
        text: &'text [u8],
        maximum_patch_count: usize,
    ) -> Vec<&'text [u8]> {
        let mut patches = Vec::new();
        let mut patch_start = 0;
        while patch_start < text.len() && patches.len() < maximum_patch_count {
            let patch_end = self.find_patch_end(text, patch_start);
            patches.push(&text[patch_start..patch_end]);
            patch_start = patch_end;
        }
        patches
    }

    /// Whether more bytes could still be appended to this patch.
    ///
    /// A patch that ends in whitespace or has reached the size limit is closed;
    /// anything else is a word the writer has not finished yet.
    pub fn is_patch_open(&self, patch: &[u8]) -> bool {
        let Some(&last_byte) = patch.last() else {
            return true;
        };
        !is_delimiter(last_byte) && patch.len() < self.maximum_patch_bytes
    }

    fn find_patch_end(&self, text: &[u8], patch_start: usize) -> usize {
        let hard_limit = (patch_start + self.maximum_patch_bytes).min(text.len());
        find_word_boundary(text, patch_start, hard_limit)
            .unwrap_or_else(|| retreat_to_character_start(text, patch_start, hard_limit))
    }
}

/// Whether a byte separates words.
pub fn is_delimiter(byte: u8) -> bool {
    byte.is_ascii_whitespace()
}

/// First position after `patch_start` where whitespace gives way to a new word.
fn find_word_boundary(text: &[u8], patch_start: usize, hard_limit: usize) -> Option<usize> {
    (patch_start + 1..hard_limit)
        .find(|&position| is_delimiter(text[position - 1]) && !is_delimiter(text[position]))
}

/// Moves a forced cut backwards so it never lands inside a multi-byte character.
fn retreat_to_character_start(text: &[u8], patch_start: usize, hard_limit: usize) -> usize {
    if hard_limit == text.len() {
        return hard_limit;
    }
    (patch_start + 1..=hard_limit)
        .rev()
        .find(|&position| !is_continuation_byte(text[position]))
        .unwrap_or(hard_limit)
}

fn is_continuation_byte(byte: u8) -> bool {
    byte & UTF8_CONTINUATION_MASK == UTF8_CONTINUATION_PATTERN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_text(text: &str, maximum_patch_bytes: usize) -> Vec<String> {
        Patcher::new(maximum_patch_bytes)
            .split(text.as_bytes())
            .into_iter()
            .map(|patch| String::from_utf8_lossy(patch).into_owned())
            .collect()
    }

    #[test]
    fn words_keep_their_trailing_whitespace() {
        let patches = split_text("The cat fell to the ground.\n", 16);

        assert_eq!(
            patches,
            ["The ", "cat ", "fell ", "to ", "the ", "ground.\n"]
        );
    }

    #[test]
    fn patches_concatenate_back_to_the_original_text() {
        let text = "  Leading spaces,\ttabs\n\nand a verylongwordthatmustbesplitsomewhere end";

        let rebuilt: String = split_text(text, 8).concat();

        assert_eq!(rebuilt, text);
    }

    #[test]
    fn long_words_are_cut_at_the_size_limit() {
        let patches = split_text("abcdefghij kl", 4);

        assert_eq!(patches, ["abcd", "efgh", "ij ", "kl"]);
    }

    #[test]
    fn multi_byte_characters_are_never_split() {
        let text = "日本語のテキストです";

        let patches = Patcher::new(4).split(text.as_bytes());

        let is_every_patch_valid_text = patches
            .iter()
            .all(|patch| std::str::from_utf8(patch).is_ok());
        assert!(is_every_patch_valid_text);
        assert_eq!(patches.concat(), text.as_bytes());
    }

    #[test]
    fn arbitrary_bytes_still_make_progress() {
        let bytes = [0x80u8; 10];

        let patches = Patcher::new(4).split(&bytes);

        assert_eq!(patches.concat(), bytes);
        assert!(patches.iter().all(|patch| patch.len() <= 4));
    }

    #[test]
    fn split_prefix_stops_at_the_requested_patch_count() {
        let patches = Patcher::new(16).split_prefix(b"one two three four", 2);

        assert_eq!(patches, [b"one ".as_slice(), b"two ".as_slice()]);
    }

    #[test]
    fn a_word_without_trailing_whitespace_is_open() {
        let patcher = Patcher::new(8);

        assert!(patcher.is_patch_open(b"gro"));
        assert!(!patcher.is_patch_open(b"ground "));
        assert!(!patcher.is_patch_open(b"groundss"));
    }
}
