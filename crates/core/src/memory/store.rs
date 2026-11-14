//! The episodic memory store: exchanges on disk, searched by meaning.
//!
//! Entries and their meaning vectors live in an embedded database file, so the
//! store survives restarts and its size is limited by disk, not RAM. Search is
//! an exact scan over every vector, which stays fast up to roughly a million
//! memories at half-precision.

use std::path::Path;

use half::f16;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::error::{BrainError, BrainResult};
use crate::memory::entry::{MemoryDraft, MemoryEntry, MemoryMatch, current_unix_seconds};
use crate::memory::similarity::{BestMatches, cosine_similarity};

const ENTRIES_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("entries");
const VECTORS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("vectors");
const PROPERTIES_TABLE: TableDefinition<&str, u64> = TableDefinition::new("properties");
const VECTOR_WIDTH_PROPERTY: &str = "vector_width";
const NEXT_ID_PROPERTY: &str = "next_id";
const FIRST_ENTRY_ID: u64 = 1;
const HALF_PRECISION_BYTES: usize = 2;

/// Episodic memories stored on disk.
pub struct EpisodicStore {
    database: Database,
    vector_width: usize,
}

impl std::fmt::Debug for EpisodicStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EpisodicStore")
            .field("vector_width", &self.vector_width)
            .finish_non_exhaustive()
    }
}

impl EpisodicStore {
    /// Opens the store at `path`, creating it when it does not exist.
    ///
    /// Fails when an existing store holds vectors of a different width, which
    /// means it belongs to a brain with a different meaning space.
    pub fn open(path: &Path, vector_width: usize) -> BrainResult<Self> {
        let database = Database::create(path).map_err(store_error)?;
        let store = Self {
            database,
            vector_width,
        };
        store.initialize()?;
        Ok(store)
    }

    fn initialize(&self) -> BrainResult<()> {
        let transaction = self.database.begin_write().map_err(store_error)?;
        {
            transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
            transaction.open_table(VECTORS_TABLE).map_err(store_error)?;
            let mut properties = transaction
                .open_table(PROPERTIES_TABLE)
                .map_err(store_error)?;
            let stored_width = read_property(&properties, VECTOR_WIDTH_PROPERTY)?;
            self.check_vector_width(stored_width)?;
            properties
                .insert(VECTOR_WIDTH_PROPERTY, self.vector_width as u64)
                .map_err(store_error)?;
        }
        transaction.commit().map_err(store_error)
    }

    fn check_vector_width(&self, stored_width: Option<u64>) -> BrainResult<()> {
        match stored_width {
            Some(width) if width != self.vector_width as u64 => {
                Err(BrainError::MemoryStore(format!(
                    "the store holds {width}-dimensional vectors but this brain uses {}",
                    self.vector_width
                )))
            }
            _ => Ok(()),
        }
    }

    /// Writes a new memory and returns it with its assigned identifier.
    pub fn insert(&mut self, draft: MemoryDraft<'_>) -> BrainResult<MemoryEntry> {
        if draft.vector.len() != self.vector_width {
            return Err(BrainError::MemoryStore(format!(
                "expected a {}-dimensional vector, got {}",
                self.vector_width,
                draft.vector.len()
            )));
        }

        let transaction = self.database.begin_write().map_err(store_error)?;
        let entry = {
            let mut properties = transaction
                .open_table(PROPERTIES_TABLE)
                .map_err(store_error)?;
            let id = read_property(&properties, NEXT_ID_PROPERTY)?.unwrap_or(FIRST_ENTRY_ID);
            properties
                .insert(NEXT_ID_PROPERTY, id + 1)
                .map_err(store_error)?;

            let entry = entry_from_draft(id, &draft);
            let mut entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
            entries
                .insert(id, encode_entry(&entry)?.as_slice())
                .map_err(store_error)?;
            let mut vectors = transaction.open_table(VECTORS_TABLE).map_err(store_error)?;
            vectors
                .insert(id, encode_vector(draft.vector).as_slice())
                .map_err(store_error)?;
            entry
        };
        transaction.commit().map_err(store_error)?;
        Ok(entry)
    }

    /// Reads one memory by identifier.
    pub fn get(&self, id: u64) -> BrainResult<Option<MemoryEntry>> {
        let transaction = self.database.begin_read().map_err(store_error)?;
        let entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
        let Some(stored) = entries.get(id).map_err(store_error)? else {
            return Ok(None);
        };
        decode_entry(stored.value()).map(Some)
    }

    /// Reads the meaning vector of one memory.
    pub fn vector(&self, id: u64) -> BrainResult<Option<Vec<f32>>> {
        let transaction = self.database.begin_read().map_err(store_error)?;
        let vectors = transaction.open_table(VECTORS_TABLE).map_err(store_error)?;
        let stored = vectors.get(id).map_err(store_error)?;
        Ok(stored.map(|stored| decode_vector(stored.value())))
    }

    /// Reads every memory, oldest first.
    pub fn list(&self) -> BrainResult<Vec<MemoryEntry>> {
        let transaction = self.database.begin_read().map_err(store_error)?;
        let entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
        entries
            .iter()
            .map_err(store_error)?
            .map(|item| {
                let (_, stored) = item.map_err(store_error)?;
                decode_entry(stored.value())
            })
            .collect()
    }

    /// Number of stored memories.
    pub fn entry_count(&self) -> BrainResult<usize> {
        let transaction = self.database.begin_read().map_err(store_error)?;
        let entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
        Ok(entries.len().map_err(store_error)? as usize)
    }

    /// Overwrites the stored fields of existing memories, keeping their vectors.
    pub fn update(&mut self, updated_entries: &[MemoryEntry]) -> BrainResult<()> {
        let transaction = self.database.begin_write().map_err(store_error)?;
        {
            let mut entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
            for entry in updated_entries {
                entries
                    .insert(entry.id, encode_entry(entry)?.as_slice())
                    .map_err(store_error)?;
            }
        }
        transaction.commit().map_err(store_error)
    }

    /// Deletes memories and returns how many existed.
    pub fn delete(&mut self, ids: &[u64]) -> BrainResult<usize> {
        let transaction = self.database.begin_write().map_err(store_error)?;
        let mut deleted_count = 0;
        {
            let mut entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
            let mut vectors = transaction.open_table(VECTORS_TABLE).map_err(store_error)?;
            for &id in ids {
                let was_stored = entries.remove(id).map_err(store_error)?.is_some();
                vectors.remove(id).map_err(store_error)?;
                deleted_count += usize::from(was_stored);
            }
        }
        transaction.commit().map_err(store_error)?;
        Ok(deleted_count)
    }

    /// Deletes every memory and returns how many there were.
    pub fn clear(&mut self) -> BrainResult<usize> {
        let ids: Vec<u64> = self.list()?.iter().map(|entry| entry.id).collect();
        self.delete(&ids)
    }

    /// Finds the memories whose meaning is closest to `query`, most similar first.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        minimum_similarity: f64,
    ) -> BrainResult<Vec<MemoryMatch>> {
        let transaction = self.database.begin_read().map_err(store_error)?;
        let vectors = transaction.open_table(VECTORS_TABLE).map_err(store_error)?;
        let mut best_matches = BestMatches::new(limit, minimum_similarity);
        for item in vectors.iter().map_err(store_error)? {
            let (id, stored) = item.map_err(store_error)?;
            let vector = decode_vector(stored.value());
            best_matches.offer(cosine_similarity(query, &vector), (id.value(), vector));
        }

        let entries = transaction.open_table(ENTRIES_TABLE).map_err(store_error)?;
        best_matches
            .into_sorted()
            .into_iter()
            .map(|(similarity, (id, vector))| {
                let stored = entries.get(id).map_err(store_error)?.ok_or_else(|| {
                    BrainError::MemoryStore(format!("memory {id} has a vector but no entry"))
                })?;
                Ok(MemoryMatch {
                    entry: decode_entry(stored.value())?,
                    similarity,
                    vector,
                })
            })
            .collect()
    }

    /// Notes that memories were retrieved, which protects them from pruning.
    pub fn record_retrievals(&mut self, ids: &[u64]) -> BrainResult<()> {
        let retrieved_unix_seconds = current_unix_seconds();
        let mut retrieved_entries = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(mut entry) = self.get(id)? else {
                continue;
            };
            entry.retrieval_count += 1;
            entry.last_retrieved_unix_seconds = retrieved_unix_seconds;
            retrieved_entries.push(entry);
        }
        self.update(&retrieved_entries)
    }
}

fn entry_from_draft(id: u64, draft: &MemoryDraft<'_>) -> MemoryEntry {
    MemoryEntry {
        id,
        kind: draft.kind,
        prompt: draft.prompt.to_string(),
        reply: draft.reply.to_string(),
        created_unix_seconds: current_unix_seconds(),
        importance: draft.importance,
        retrieval_count: 0,
        last_retrieved_unix_seconds: 0,
    }
}

fn read_property(
    properties: &impl ReadableTable<&'static str, u64>,
    property_name: &str,
) -> BrainResult<Option<u64>> {
    let stored = properties.get(property_name).map_err(store_error)?;
    Ok(stored.map(|stored| stored.value()))
}

fn encode_entry(entry: &MemoryEntry) -> BrainResult<Vec<u8>> {
    serde_json::to_vec(entry).map_err(store_error)
}

fn decode_entry(stored: &[u8]) -> BrainResult<MemoryEntry> {
    serde_json::from_slice(stored).map_err(store_error)
}

/// Packs a vector as little-endian half-precision floats: about 1 KB for 512 dimensions.
fn encode_vector(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|&value| f16::from_f32(value).to_le_bytes())
        .collect()
}

fn decode_vector(stored: &[u8]) -> Vec<f32> {
    stored
        .chunks_exact(HALF_PRECISION_BYTES)
        .map(|pair| f16::from_le_bytes([pair[0], pair[1]]).to_f32())
        .collect()
}

fn store_error(error: impl std::fmt::Display) -> BrainError {
    BrainError::MemoryStore(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::entry::MemoryKind;

    fn open_fresh_store(name: &str, vector_width: usize) -> EpisodicStore {
        let path = std::env::temp_dir().join(format!("thingamaroo-store-{name}.redb"));
        let _ = std::fs::remove_file(&path);
        EpisodicStore::open(&path, vector_width).unwrap()
    }

    fn draft<'draft>(prompt: &'draft str, vector: &'draft [f32]) -> MemoryDraft<'draft> {
        MemoryDraft {
            kind: MemoryKind::Exchange,
            prompt,
            reply: "ok\n",
            importance: 0.5,
            vector,
        }
    }

    #[test]
    fn inserted_memories_get_increasing_identifiers_and_can_be_read_back() {
        let mut store = open_fresh_store("insert", 3);

        let first = store.insert(draft("first\n", &[1.0, 0.0, 0.0])).unwrap();
        let second = store.insert(draft("second\n", &[0.0, 1.0, 0.0])).unwrap();

        assert_eq!((first.id, second.id), (1, 2));
        assert_eq!(store.get(2).unwrap().unwrap().prompt, "second\n");
        assert_eq!(store.entry_count().unwrap(), 2);
        assert_eq!(store.vector(1).unwrap().unwrap(), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn search_returns_the_most_similar_memories_first() {
        let mut store = open_fresh_store("search", 3);
        store.insert(draft("north\n", &[0.0, 1.0, 0.0])).unwrap();
        store.insert(draft("east\n", &[1.0, 0.0, 0.0])).unwrap();
        store
            .insert(draft("north-east\n", &[0.7, 0.7, 0.0]))
            .unwrap();

        let matches = store.search(&[1.0, 0.1, 0.0], 2, 0.5).unwrap();

        let prompts: Vec<&str> = matches
            .iter()
            .map(|found| found.entry.prompt.as_str())
            .collect();
        assert_eq!(prompts, ["east\n", "north-east\n"]);
        assert!(matches[0].similarity > matches[1].similarity);
    }

    #[test]
    fn retrievals_are_counted_and_deletions_remove_entries() {
        let mut store = open_fresh_store("retrieval", 2);
        let entry = store.insert(draft("remember me\n", &[1.0, 1.0])).unwrap();

        store.record_retrievals(&[entry.id, 99]).unwrap();
        assert_eq!(store.get(entry.id).unwrap().unwrap().retrieval_count, 1);

        assert_eq!(store.delete(&[entry.id, 99]).unwrap(), 1);
        assert!(store.get(entry.id).unwrap().is_none());
        assert!(store.search(&[1.0, 1.0], 5, 0.0).unwrap().is_empty());
    }

    #[test]
    fn a_store_survives_being_reopened() {
        let path = std::env::temp_dir().join("thingamaroo-store-reopen.redb");
        let _ = std::fs::remove_file(&path);
        {
            let mut store = EpisodicStore::open(&path, 2).unwrap();
            store.insert(draft("persistent\n", &[0.5, 0.25])).unwrap();
        }

        {
            let reopened = EpisodicStore::open(&path, 2).unwrap();
            assert_eq!(reopened.list().unwrap()[0].prompt, "persistent\n");
        }

        let mismatched = EpisodicStore::open(&path, 4);

        let Err(BrainError::MemoryStore(reason)) = mismatched else {
            panic!("a store with a different vector width should be rejected");
        };
        assert!(
            reason.contains("2-dimensional"),
            "unexpected reason: {reason}"
        );
    }
}
