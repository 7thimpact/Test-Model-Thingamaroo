//! Runs the real binary through a whole session with a miniature model.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A model small enough to train in a few seconds on any machine.
const MINIATURE_ARCHITECTURE: &str = r#"{
    "patching": { "maximum_patch_bytes": 12 },
    "latent": { "width": 32 },
    "encoder": { "width": 32, "layer_count": 1, "head_count": 4 },
    "predictor": {
        "width": 48,
        "layer_count": 2,
        "head_count": 4,
        "maximum_context_patches": 16,
        "maximum_memory_slots": 2,
        "thinking_step_count": 1
    },
    "decoder": { "width": 32, "layer_count": 1, "head_count": 4 },
    "adapter": { "rank": 2, "scale": 4.0 }
}"#;

/// One brain directory and the commands run against it.
struct Session {
    directory: PathBuf,
}

impl Session {
    fn new(name: &str) -> Self {
        let directory = std::env::temp_dir().join(format!("thingamaroo-cli-{name}"));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        Self { directory }
    }

    fn path(&self, file_name: &str) -> String {
        self.directory.join(file_name).display().to_string()
    }

    fn run(&self, arguments: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_thingamaroo"))
            .args(["--brain", &self.path("brain"), "--device", "cpu"])
            .args(arguments)
            .output()
            .expect("the command-line binary should start")
    }

    /// Runs a command that must succeed and returns what it printed.
    fn succeed(&self, arguments: &[&str]) -> String {
        let output = self.run(arguments);
        let printed = String::from_utf8_lossy(&output.stdout).into_owned();
        let complaints = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "`{arguments:?}` failed\nstdout: {printed}\nstderr: {complaints}"
        );
        printed
    }

    fn create_trained_brain(&self) {
        std::fs::write(self.path("architecture.json"), MINIATURE_ARCHITECTURE).unwrap();
        self.succeed(&[
            "sample-corpus",
            &self.path("sample.txt"),
            "--megabytes",
            "0.05",
        ]);
        self.succeed(&["init", "--configuration", &self.path("architecture.json")]);
        self.succeed(&[
            "train",
            "--data",
            &self.path("sample.txt"),
            "--steps",
            "12",
            "--batch",
            "4",
            "--warmup",
            "2",
            "--report-every",
            "6",
            "--validate-every",
            "12",
            "--replay-windows",
            "16",
        ]);
    }
}

#[test]
fn presets_are_listed_without_a_brain() {
    let session = Session::new("presets");

    let printed = session.succeed(&["presets"]);

    for preset_name in ["tiny", "prototype", "small", "target", "medium"] {
        assert!(
            printed.contains(preset_name),
            "missing {preset_name} in:\n{printed}"
        );
    }
}

#[test]
fn a_missing_brain_is_reported_as_an_error() {
    let session = Session::new("missing");

    let output = session.run(&["status"]);

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no brain at"));
}

#[test]
fn a_brain_can_be_trained_inspected_and_used() {
    let session = Session::new("lifecycle");
    session.create_trained_brain();

    let status = session.succeed(&["status"]);
    assert!(status.contains("pretraining steps   12"), "{status}");
    assert!(
        status.contains("replay data         16 windows"),
        "{status}"
    );

    let patches = session.succeed(&["patches", "The cat says meow."]);
    assert!(
        patches.contains("[The ] [cat ] [says ] [meow.]"),
        "{patches}"
    );

    let round_trip = session.succeed(&["roundtrip", "The cat says meow."]);
    assert!(round_trip.contains("byte accuracy"), "{round_trip}");

    let generated = session.succeed(&[
        "generate",
        "The ",
        "--seed",
        "3",
        "--max-bytes",
        "30",
        "--thoughts",
    ]);
    assert!(generated.starts_with("The "), "{generated}");
    assert!(generated.contains("thought:"), "{generated}");

    let evaluation = session.succeed(&[
        "evaluate",
        "--data",
        &session.path("sample.txt"),
        "--batches",
        "1",
        "--batch",
        "2",
    ]);
    assert!(evaluation.contains("rebuild accuracy"), "{evaluation}");

    let history = session.succeed(&["history"]);
    assert!(
        history.contains("train") && history.contains("validate"),
        "{history}"
    );
}

#[test]
fn memories_snapshots_settings_and_sleep_work_from_the_command_line() {
    let session = Session::new("maintenance");
    session.create_trained_brain();

    let added = session.succeed(&["memory", "add", "Mia likes green pears."]);
    assert!(added.contains("Stored as memory #1"), "{added}");
    let listed = session.succeed(&["memory", "list"]);
    assert!(listed.contains("Mia likes green pears."), "{listed}");
    let found = session.succeed(&["memory", "search", "What does Mia like?"]);
    assert!(found.contains("#1"), "{found}");
    let shown = session.succeed(&["memory", "show", "1"]);
    assert!(shown.contains("note"), "{shown}");

    session.succeed(&["settings", "set", "sleep.replay_step_count", "4"]);
    session.succeed(&["settings", "set", "sleep.maximum_old_loss_increase", "10"]);
    let settings = session.succeed(&["settings", "show"]);
    assert!(settings.contains("sleep.replay_step_count"), "{settings}");
    assert!(
        !session
            .run(&["settings", "set", "sleep.no_such_setting", "1"])
            .status
            .success()
    );

    let created = session.succeed(&["snapshot", "create", "--label", "before experiments"]);
    assert!(created.contains("before experiments"), "{created}");

    let sleep = session.succeed(&["sleep"]);
    assert!(
        sleep.contains("Sleep finished: 1 memories replayed in 4 steps"),
        "{sleep}"
    );

    let snapshots = session.succeed(&["snapshot", "list"]);
    assert!(
        snapshots.contains("before sleep") && snapshots.contains("2 snapshots"),
        "{snapshots}"
    );
    session.succeed(&["snapshot", "rollback", "2"]);
    session.succeed(&["snapshot", "delete", "1"]);
    session.succeed(&["reset-fast-weights"]);

    session.succeed(&["memory", "delete", "1"]);
    let cleared = session.succeed(&["memory", "clear"]);
    assert!(cleared.contains("Deleted all 0 memories"), "{cleared}");
    assert!(
        Path::new(&session.path("brain"))
            .join("history.jsonl")
            .exists()
    );
}

#[test]
fn the_settings_reference_document_is_up_to_date() {
    let session = Session::new("reference");
    let document_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/settings.md");

    let generated = session.succeed(&["settings", "reference"]);
    let document = std::fs::read_to_string(&document_path).unwrap();

    assert_eq!(
        document.trim(),
        generated.trim(),
        "regenerate with `thingamaroo settings reference > docs/settings.md`"
    );
}

#[test]
fn settings_are_listed_with_their_explanations() {
    let session = Session::new("explained");
    std::fs::write(session.path("architecture.json"), MINIATURE_ARCHITECTURE).unwrap();
    session.succeed(&[
        "init",
        "--configuration",
        &session.path("architecture.json"),
    ]);

    let listed = session.succeed(&["settings", "show"]);

    assert!(listed.contains("generation.temperature = 0.7"), "{listed}");
    assert!(
        listed.contains("Sampling temperature; zero always picks the most likely byte."),
        "{listed}"
    );
}
