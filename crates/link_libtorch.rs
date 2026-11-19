//! Build script shared by the command-line and desktop applications, and
//! called from the core library's build script for its test programs.
//!
//! With the `libtorch` feature the programs load LibTorch's shared libraries
//! when they start. The `torch-sys` build script reports where those libraries
//! are (on Linux and macOS), and this script records that folder inside the
//! program, so it starts without any library search path being set by hand.
//! Windows has no such mechanism; there the LibTorch `lib` folder must be on
//! `PATH` (see the README).

/// Set by Cargo from the `libtorch_lib` value that `torch-sys` (`links = "tch"`) prints.
const LIBTORCH_FOLDER_VARIABLE: &str = "DEP_TCH_LIBTORCH_LIB";

/// Records the LibTorch folder in every program and test this package links.
pub fn main() {
    println!("cargo:rerun-if-env-changed={LIBTORCH_FOLDER_VARIABLE}");
    let Ok(libtorch_folder) = std::env::var(LIBTORCH_FOLDER_VARIABLE) else {
        return;
    };
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch_folder}");
}
