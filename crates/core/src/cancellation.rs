//! Lets a user interface stop a long operation from another thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shared flag that long operations check between steps.
///
/// Cloning gives another handle to the same flag, so a front end can keep one
/// handle and pass the other into training, generation or sleep.
#[derive(Debug, Clone, Default)]
pub struct CancellationFlag {
    is_requested: Arc<AtomicBool>,
}

impl CancellationFlag {
    /// Creates a flag that has not been raised.
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks the running operation to stop at its next safe point.
    pub fn request(&self) {
        self.is_requested.store(true, Ordering::Relaxed);
    }

    /// Lowers the flag so the next operation runs normally.
    pub fn reset(&self) {
        self.is_requested.store(false, Ordering::Relaxed);
    }

    /// Whether a stop has been requested.
    pub fn is_requested(&self) -> bool {
        self.is_requested.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_same_flag() {
        let flag = CancellationFlag::new();
        let other_handle = flag.clone();

        other_handle.request();
        assert!(flag.is_requested());

        flag.reset();
        assert!(!other_handle.is_requested());
    }
}
