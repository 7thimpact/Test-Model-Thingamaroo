//! Ctrl-C handling: the first press stops the running operation, the second quits.

use thingamaroo_core::cancellation::CancellationFlag;

/// Exit status conventionally used for a process ended by Ctrl-C.
const INTERRUPTED_EXIT_STATUS: i32 = 130;

/// Installs the Ctrl-C handler and returns the flag it raises.
///
/// Long operations watch the flag and stop at their next safe point, saving
/// their progress. If the flag is already raised when Ctrl-C arrives again,
/// the process exits immediately.
pub fn install_interrupt_handler() -> CancellationFlag {
    let cancellation = CancellationFlag::new();
    let handler_flag = cancellation.clone();
    let installation = ctrlc::set_handler(move || {
        if handler_flag.is_requested() {
            std::process::exit(INTERRUPTED_EXIT_STATUS);
        }
        handler_flag.request();
        eprintln!("\nStopping at the next safe point (press Ctrl-C again to quit immediately)");
    });
    if let Err(error) = installation {
        eprintln!("warning: Ctrl-C will not stop operations gracefully: {error}");
    }
    cancellation
}
