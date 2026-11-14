//! Lays patches out as the fixed-shape tensors the model consumes.

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::configuration::END_OF_PATCH_SYMBOL;

/// Patches of one or more sequences as padded tensors.
///
/// Sequences with fewer patches than the longest one are padded with empty
/// patches at the end; `patch_validity` marks which patches are real.
#[derive(Debug, Clone)]
pub struct PatchBatch {
    /// Number of sequences in the batch.
    pub sequence_count: usize,
    /// Patches per sequence, including padding patches.
    pub patch_count: usize,
    /// Bytes of every patch, `[sequence * patch, maximum_patch_bytes]`.
    pub bytes: Tensor<2, Int>,
    /// One for real bytes, zero for padding; same shape as `bytes`.
    pub byte_validity: Tensor<2>,
    /// What the decoder should write: the bytes, then the end-of-patch symbol.
    /// `[sequence * patch, maximum_patch_bytes + 1]`.
    pub target_symbols: Tensor<2, Int>,
    /// One for symbols that count towards the loss; same shape as `target_symbols`.
    pub target_validity: Tensor<2>,
    /// One for real patches, zero for padding patches; `[sequence, patch]`.
    pub patch_validity: Tensor<2>,
}

/// Flat buffers filled patch by patch before they become tensors.
struct PatchBuffers {
    maximum_patch_bytes: usize,
    bytes: Vec<i32>,
    byte_validity: Vec<f32>,
    target_symbols: Vec<i32>,
    target_validity: Vec<f32>,
    patch_validity: Vec<f32>,
}

impl PatchBuffers {
    fn with_capacity(patch_total: usize, maximum_patch_bytes: usize) -> Self {
        let byte_total = patch_total * maximum_patch_bytes;
        let symbol_total = patch_total * (maximum_patch_bytes + 1);
        Self {
            maximum_patch_bytes,
            bytes: Vec::with_capacity(byte_total),
            byte_validity: Vec::with_capacity(byte_total),
            target_symbols: Vec::with_capacity(symbol_total),
            target_validity: Vec::with_capacity(symbol_total),
            patch_validity: Vec::with_capacity(patch_total),
        }
    }

    /// Appends one real patch, truncated to the size limit.
    fn push_patch(&mut self, patch: &[u8]) {
        let patch = &patch[..patch.len().min(self.maximum_patch_bytes)];
        let padding_bytes = self.maximum_patch_bytes - patch.len();

        self.bytes.extend(patch.iter().map(|&byte| i32::from(byte)));
        self.bytes.extend(std::iter::repeat_n(0, padding_bytes));
        self.byte_validity
            .extend(std::iter::repeat_n(1.0, patch.len()));
        self.byte_validity
            .extend(std::iter::repeat_n(0.0, padding_bytes));

        self.target_symbols
            .extend(patch.iter().map(|&byte| i32::from(byte)));
        self.target_symbols.push(END_OF_PATCH_SYMBOL as i32);
        self.target_symbols
            .extend(std::iter::repeat_n(0, padding_bytes));
        self.target_validity
            .extend(std::iter::repeat_n(1.0, patch.len() + 1));
        self.target_validity
            .extend(std::iter::repeat_n(0.0, padding_bytes));

        self.patch_validity.push(1.0);
    }

    /// Appends one padding patch that contributes to no loss.
    fn push_padding_patch(&mut self) {
        let symbol_count = self.maximum_patch_bytes + 1;
        self.bytes
            .extend(std::iter::repeat_n(0, self.maximum_patch_bytes));
        self.byte_validity
            .extend(std::iter::repeat_n(0.0, self.maximum_patch_bytes));
        self.target_symbols
            .extend(std::iter::repeat_n(0, symbol_count));
        self.target_validity
            .extend(std::iter::repeat_n(0.0, symbol_count));
        self.patch_validity.push(0.0);
    }
}

impl PatchBatch {
    /// Builds a batch from patch sequences, padding shorter sequences at the end.
    pub fn from_sequences<Patch: AsRef<[u8]>>(
        sequences: &[Vec<Patch>],
        maximum_patch_bytes: usize,
        device: &Device,
    ) -> Self {
        let sequence_count = sequences.len();
        let patch_count = sequences.iter().map(Vec::len).max().unwrap_or(0);
        let mut buffers =
            PatchBuffers::with_capacity(sequence_count * patch_count, maximum_patch_bytes);

        for sequence in sequences {
            for patch in sequence {
                buffers.push_patch(patch.as_ref());
            }
            for _ in sequence.len()..patch_count {
                buffers.push_padding_patch();
            }
        }

        Self::from_buffers(buffers, sequence_count, patch_count, device)
    }

    fn from_buffers(
        buffers: PatchBuffers,
        sequence_count: usize,
        patch_count: usize,
        device: &Device,
    ) -> Self {
        let patch_total = sequence_count * patch_count;
        let byte_shape = [patch_total, buffers.maximum_patch_bytes];
        let symbol_shape = [patch_total, buffers.maximum_patch_bytes + 1];
        Self {
            sequence_count,
            patch_count,
            bytes: Tensor::from_data(TensorData::new(buffers.bytes, byte_shape), device),
            byte_validity: Tensor::from_data(
                TensorData::new(buffers.byte_validity, byte_shape),
                device,
            ),
            target_symbols: Tensor::from_data(
                TensorData::new(buffers.target_symbols, symbol_shape),
                device,
            ),
            target_validity: Tensor::from_data(
                TensorData::new(buffers.target_validity, symbol_shape),
                device,
            ),
            patch_validity: Tensor::from_data(
                TensorData::new(buffers.patch_validity, [sequence_count, patch_count]),
                device,
            ),
        }
    }

    /// Total number of patch slots, real and padding.
    pub fn patch_total(&self) -> usize {
        self.sequence_count * self.patch_count
    }
}
