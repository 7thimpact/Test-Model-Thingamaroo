//! Copies tensor contents back to ordinary host memory.

use burn::tensor::Tensor;

/// Reads every element of a tensor as `f32`, in row-major order.
///
/// # Panics
///
/// Panics when the compute device fails to return the data, which means the
/// device itself has been lost and no further work is possible.
pub fn read_values<const RANK: usize>(tensor: Tensor<RANK>) -> Vec<f32> {
    tensor
        .try_into_vec_as::<f32>()
        .expect("the compute device should return tensor contents")
}

/// Reads a single-element tensor as a number.
///
/// # Panics
///
/// Panics under the same condition as [`read_values`].
pub fn read_scalar(tensor: Tensor<1>) -> f64 {
    f64::from(read_values(tensor)[0])
}
