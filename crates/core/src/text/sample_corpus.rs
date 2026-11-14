//! A built-in generator of simple English, for trying the whole loop without a dataset.
//!
//! Lines are filled-in templates from `assets/sample_grammar.json`. Some facts
//! are fixed ("the cat says meow") so there is something to learn, and some
//! are random ("Mia likes cake") so there is something to remember.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Deserialize;

use crate::error::{BrainError, BrainResult};
use crate::random::RandomSource;

const BUILT_IN_GRAMMAR: &str = include_str!("../../assets/sample_grammar.json");
const GRAMMAR_SOURCE_NAME: &str = "assets/sample_grammar.json";
const PLACEHOLDER_OPEN: char = '{';
const PLACEHOLDER_CLOSE: char = '}';
const ATTRIBUTE_SEPARATOR: char = '.';
const WORD_ATTRIBUTE: &str = "word";

/// One choice inside a word category.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum GrammarEntry {
    /// A plain word.
    Word(String),
    /// A word with named attributes, such as an animal and its sound.
    Detailed(BTreeMap<String, String>),
}

impl GrammarEntry {
    fn attribute(&self, attribute_name: &str) -> Option<&str> {
        match self {
            GrammarEntry::Word(word) if attribute_name == WORD_ATTRIBUTE => Some(word),
            GrammarEntry::Word(_) => None,
            GrammarEntry::Detailed(attributes) => {
                attributes.get(attribute_name).map(String::as_str)
            }
        }
    }
}

/// Word lists and sentence templates.
#[derive(Debug, Clone, Deserialize)]
struct Grammar {
    words: BTreeMap<String, Vec<GrammarEntry>>,
    templates: Vec<String>,
}

/// Generator of simple English lines.
#[derive(Debug, Clone)]
pub struct SampleCorpus {
    grammar: Grammar,
}

impl SampleCorpus {
    /// Loads the grammar that ships inside the library.
    pub fn built_in() -> BrainResult<Self> {
        let grammar: Grammar = serde_json::from_str(BUILT_IN_GRAMMAR)
            .map_err(|error| BrainError::document(GRAMMAR_SOURCE_NAME, error))?;
        if grammar.templates.is_empty() {
            return Err(BrainError::document(GRAMMAR_SOURCE_NAME, "no templates"));
        }
        Ok(Self { grammar })
    }

    /// Produces one line of text, without a trailing line break.
    pub fn generate_line(&self, random: &mut RandomSource) -> BrainResult<String> {
        let template = random
            .pick(&self.grammar.templates)
            .ok_or_else(|| BrainError::document(GRAMMAR_SOURCE_NAME, "no templates"))?;
        self.fill_template(template, random)
    }

    /// Writes at least `target_bytes` of generated lines and returns the bytes written.
    pub fn write_file(&self, path: &Path, target_bytes: u64, seed: u64) -> BrainResult<u64> {
        let file =
            std::fs::File::create(path).map_err(|source| BrainError::file_system(path, source))?;
        let mut writer = BufWriter::new(file);
        let mut random = RandomSource::from_seed(seed);
        let mut written_bytes = 0;

        while written_bytes < target_bytes {
            let line = self.generate_line(&mut random)?;
            writeln!(writer, "{line}").map_err(|source| BrainError::file_system(path, source))?;
            written_bytes += line.len() as u64 + 1;
        }
        writer
            .flush()
            .map_err(|source| BrainError::file_system(path, source))?;
        Ok(written_bytes)
    }

    fn fill_template(&self, template: &str, random: &mut RandomSource) -> BrainResult<String> {
        let mut bindings: BTreeMap<String, &GrammarEntry> = BTreeMap::new();
        let mut line = String::new();
        let mut remaining = template;

        while let Some((literal, placeholder, rest)) = split_at_placeholder(remaining) {
            line.push_str(literal);
            line.push_str(self.resolve_placeholder(placeholder, &mut bindings, random)?);
            remaining = rest;
        }
        line.push_str(remaining);
        Ok(line)
    }

    /// Resolves `binding` or `binding.attribute`, choosing a word the first time a binding is seen.
    ///
    /// A binding such as `name2` draws from the `name` category but is chosen
    /// independently of `name`, which lets one line mention two different names.
    fn resolve_placeholder<'grammar>(
        &'grammar self,
        placeholder: &str,
        bindings: &mut BTreeMap<String, &'grammar GrammarEntry>,
        random: &mut RandomSource,
    ) -> BrainResult<&'grammar str> {
        let (binding_name, attribute_name) = placeholder
            .split_once(ATTRIBUTE_SEPARATOR)
            .unwrap_or((placeholder, WORD_ATTRIBUTE));

        if !bindings.contains_key(binding_name) {
            let entry = self.pick_entry(binding_name, random)?;
            bindings.insert(binding_name.to_string(), entry);
        }
        bindings[binding_name]
            .attribute(attribute_name)
            .ok_or_else(|| unknown_placeholder(placeholder))
    }

    fn pick_entry(
        &self,
        binding_name: &str,
        random: &mut RandomSource,
    ) -> BrainResult<&GrammarEntry> {
        let category_name =
            binding_name.trim_end_matches(|character: char| character.is_ascii_digit());
        self.grammar
            .words
            .get(category_name)
            .and_then(|entries| random.pick(entries))
            .ok_or_else(|| unknown_placeholder(binding_name))
    }
}

/// Splits `text` into the literal before the next placeholder, the placeholder name, and the rest.
fn split_at_placeholder(text: &str) -> Option<(&str, &str, &str)> {
    let (literal, after_open) = text.split_once(PLACEHOLDER_OPEN)?;
    let (placeholder, rest) = after_open.split_once(PLACEHOLDER_CLOSE)?;
    Some((literal, placeholder, rest))
}

fn unknown_placeholder(placeholder: &str) -> BrainError {
    BrainError::document(
        GRAMMAR_SOURCE_NAME,
        format!("placeholder '{placeholder}' has no matching words"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_can_be_filled() {
        let corpus = SampleCorpus::built_in().unwrap();
        let mut random = RandomSource::from_seed(1);

        for template in &corpus.grammar.templates {
            let line = corpus.fill_template(template, &mut random).unwrap();
            assert!(
                !line.contains(PLACEHOLDER_OPEN),
                "unfilled placeholder in '{line}'"
            );
        }
    }

    #[test]
    fn a_repeated_placeholder_keeps_its_word_within_a_line() {
        let corpus = SampleCorpus::built_in().unwrap();
        let mut random = RandomSource::from_seed(9);

        let line = corpus
            .fill_template("{animal}|{animal}|{animal.sound}", &mut random)
            .unwrap();

        let parts: Vec<&str> = line.split('|').collect();
        assert_eq!(parts[0], parts[1]);
        assert_ne!(parts[0], parts[2]);
    }

    #[test]
    fn the_same_seed_writes_the_same_file() {
        let corpus = SampleCorpus::built_in().unwrap();
        let first_path = std::env::temp_dir().join("thingamaroo-sample-first.txt");
        let second_path = std::env::temp_dir().join("thingamaroo-sample-second.txt");

        let written_bytes = corpus.write_file(&first_path, 4_000, 3).unwrap();
        corpus.write_file(&second_path, 4_000, 3).unwrap();

        assert!(written_bytes >= 4_000);
        assert_eq!(
            std::fs::read(&first_path).unwrap(),
            std::fs::read(&second_path).unwrap()
        );
    }
}
