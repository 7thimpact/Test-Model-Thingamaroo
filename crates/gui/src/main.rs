//! Desktop front end for the byte-level continual-learning language model.
//!
//! Everything the command-line application can do is available here; each tab
//! names the commands it covers.

// A release build on Windows should not open a console window behind the application.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod application;
mod form;
mod shared;
mod views;
mod worker;

use std::path::PathBuf;

use thingamaroo_core::model::DeviceChoice;

use crate::application::{Application, StartupOptions};

/// Tensor code allocates and frees large buffers constantly; the system
/// allocator hands those back to the operating system every time.
#[cfg(feature = "fast-allocator")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const WINDOW_TITLE: &str = "Thingamaroo";
const INITIAL_WINDOW_SIZE: [f32; 2] = [1120.0, 780.0];
const MINIMUM_WINDOW_SIZE: [f32; 2] = [760.0, 520.0];

const USAGE: &str = "\
Usage: thingamaroo-gui [--brain <DIRECTORY>] [--device <auto|cpu|gpu|cuda>] [--renderer <wgpu|glow>]

  --brain     open this brain as soon as the window appears
  --device    compute backend used for that brain (default: auto)
  --renderer  how the window is drawn; by default wgpu is tried first, then glow";

/// What the command line asked for.
struct LaunchOptions {
    startup: StartupOptions,
    /// The renderer the user insisted on; `None` tries each in turn.
    renderer: Option<eframe::Renderer>,
}

fn main() -> eframe::Result {
    let launch = match parse_launch_options(std::env::args().skip(1)) {
        Ok(launch) => launch,
        Err(problem) => {
            eprintln!("{problem}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let Some(renderer) = launch.renderer else {
        return open_window_with_fallback(&launch.startup);
    };
    open_window(renderer, launch.startup)
}

/// Tries the GPU renderer first and falls back to OpenGL when no surface can be created,
/// which happens on some virtual machines and remote desktops.
fn open_window_with_fallback(startup: &StartupOptions) -> eframe::Result {
    let Err(problem) = open_window(eframe::Renderer::Wgpu, startup.clone()) else {
        return Ok(());
    };
    eprintln!("The wgpu renderer could not start ({problem}); trying OpenGL instead.");
    open_window(eframe::Renderer::Glow, startup.clone())
}

fn open_window(renderer: eframe::Renderer, startup: StartupOptions) -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(WINDOW_TITLE)
            .with_inner_size(INITIAL_WINDOW_SIZE)
            .with_min_inner_size(MINIMUM_WINDOW_SIZE),
        renderer,
        ..Default::default()
    };
    eframe::run_native(
        WINDOW_TITLE,
        native_options,
        Box::new(move |creation_context| Ok(Box::new(Application::new(creation_context, startup)))),
    )
}

fn parse_launch_options(
    mut arguments: impl Iterator<Item = String>,
) -> Result<LaunchOptions, String> {
    let mut launch = LaunchOptions {
        startup: StartupOptions::default(),
        renderer: None,
    };
    while let Some(option_name) = arguments.next() {
        if option_name == "--help" || option_name == "-h" {
            println!("{USAGE}");
            std::process::exit(0);
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("{option_name} needs a value"))?;
        match option_name.as_str() {
            "--brain" => launch.startup.brain_directory = Some(PathBuf::from(value)),
            "--device" => {
                launch.startup.device_choice =
                    DeviceChoice::from_name(&value).map_err(|error| error.to_string())?;
            }
            "--renderer" => launch.renderer = Some(parse_renderer(&value)?),
            _ => return Err(format!("unknown option {option_name}")),
        }
    }
    Ok(launch)
}

fn parse_renderer(name: &str) -> Result<eframe::Renderer, String> {
    match name {
        "wgpu" => Ok(eframe::Renderer::Wgpu),
        "glow" => Ok(eframe::Renderer::Glow),
        _ => Err(format!("unknown renderer '{name}'; use wgpu or glow")),
    }
}
