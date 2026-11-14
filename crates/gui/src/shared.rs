//! State every view can read, and the outbox views put requests into.

use thingamaroo_core::brain::BrainSummary;
use thingamaroo_core::configuration::RuntimeSettings;

use crate::worker::Request;

/// The brain that is currently open, as last reported by the worker.
#[derive(Debug, Clone)]
pub struct OpenBrain {
    /// What the brain is and what it has been through.
    pub summary: BrainSummary,
    /// Its adjustable behaviour.
    pub settings: RuntimeSettings,
}

/// What views share: the open brain, whether the worker is busy, and pending requests.
#[derive(Debug, Default)]
pub struct Shared {
    /// The open brain, if any.
    pub open_brain: Option<OpenBrain>,
    /// Whether the worker is in the middle of a request.
    pub is_busy: bool,
    outbox: Vec<Request>,
    failures: Vec<String>,
}

impl Shared {
    /// Queues a request for the worker; it is sent after the current frame.
    pub fn send(&mut self, request: Request) {
        self.outbox.push(request);
    }

    /// Whether a brain is open and the worker is free to act on it.
    pub fn can_act(&self) -> bool {
        self.open_brain.is_some() && !self.is_busy
    }

    /// Reports a problem found in the window itself, such as an unreadable file.
    pub fn fail(&mut self, message: impl Into<String>) {
        self.failures.push(message.into());
    }

    /// Takes the problems reported during this frame.
    pub fn take_failures(&mut self) -> Vec<String> {
        std::mem::take(&mut self.failures)
    }

    /// Takes the requests queued during this frame.
    pub fn take_outbox(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.outbox)
    }
}
