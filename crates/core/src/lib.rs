//! A byte-level continual-learning language model.
//!
//! The system reads and writes raw bytes, predicts in meaning space before it
//! writes, remembers what surprised it, and keeps learning after pretraining:
//!
//! 1. **Receive** raw bytes, grouped into variable-size patches ([`text`]).
//! 2. **Process by thought**: predict a rough meaning vector, refine it over a
//!    few internal steps, then decode exact bytes ([`model`]).
//! 3. **Store** exchanges weighted by how surprising they were ([`memory`]).
//! 4. **Change future outcomes** immediately through retrieved memories and
//!    fast weights, and permanently through sleep consolidation ([`learning`]).
//!
//! [`brain::Brain`] ties these together and is the only type front ends need.

#![warn(missing_docs)]

pub mod brain;
pub mod cancellation;
pub mod configuration;
pub mod error;
pub mod learning;
pub mod memory;
pub mod model;
pub mod presentation;
pub mod random;
pub mod text;
