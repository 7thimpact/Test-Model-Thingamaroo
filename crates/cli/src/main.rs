//! Command-line front end for the byte-level continual-learning language model.

mod arguments;
mod commands;
mod interrupt;
mod report;

use std::process::ExitCode;

use clap::Parser;

use crate::arguments::CommandLine;

/// Tensor code allocates and frees large buffers constantly; the system
/// allocator hands those back to the operating system every time.
#[cfg(feature = "fast-allocator")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> ExitCode {
    let command_line = CommandLine::parse();
    match commands::run(command_line) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
