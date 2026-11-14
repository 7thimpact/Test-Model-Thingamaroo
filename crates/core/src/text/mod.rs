//! Raw text handling: patching bytes, reading corpora, and generating sample text.

mod corpus;
mod patcher;
mod replay_reservoir;
mod sample_corpus;

pub use corpus::{Corpus, CorpusSplit, PatchSequence};
pub use patcher::{Patcher, is_delimiter};
pub use replay_reservoir::ReplayReservoir;
pub use sample_corpus::SampleCorpus;
