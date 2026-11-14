//! A brain: one model, its memories, and everything it has learned, in one directory.
//!
//! [`Brain`] is the single entry point for front ends. Every capability of the
//! command-line and desktop applications is a method on it, so the two can
//! never drift apart in what they are able to do.

mod chat;
mod directory;
mod evaluation;
mod history;
mod memory_access;
mod pretraining_run;
mod sleep_cycle;
mod sleep_schedule;
mod snapshots;
mod state;
mod summary;

use std::path::PathBuf;

use burn::module::Module;
use burn::store::ModuleRecord;

use crate::configuration::{ModelConfiguration, RuntimeSettings};
use crate::error::{BrainError, BrainResult};
use crate::learning::{
    LessonContext, TrainableModule, attach_fast_weights, build_fast_weight_optimizer,
    fast_weight_group,
};
use crate::memory::{EpisodicStore, current_unix_seconds};
use crate::model::{ComputeDevice, ConceptPredictor, DeviceChoice, InferenceModel, LanguageModel};

pub use chat::{ChatOptions, ChatSession, ChatTurnReport, FeedbackReport};
pub use directory::BrainDirectory;
pub use evaluation::{EvaluationReport, PatchRoundTrip, RoundTripReport};
pub use history::HistoryRecord;
pub use pretraining_run::PretrainingSummary;
pub use sleep_cycle::SleepReport;
pub use sleep_schedule::SleepTrigger;
pub use snapshots::SnapshotInformation;
pub use state::{BrainState, ConversationState, SleepState, TrainingState};
pub use summary::BrainSummary;

use directory::{load_document, load_document_or_default, store_document};

/// One model, its memories, and everything it has learned.
pub struct Brain {
    directory: BrainDirectory,
    model_configuration: ModelConfiguration,
    settings: RuntimeSettings,
    state: BrainState,
    compute_device: ComputeDevice,
    /// Encoder, predictor and decoder on the gradient-tracking device.
    slow_model: LanguageModel,
    /// The slow predictor with fast weights attached, and their optimiser.
    fast_weights: TrainableModule<ConceptPredictor>,
    /// Gradient-free copies used to read, think and write.
    inference: InferenceModel,
    memory: EpisodicStore,
}

impl Brain {
    /// Creates a new, untrained brain in an empty or missing directory.
    pub fn create(
        root: impl Into<PathBuf>,
        model_configuration: ModelConfiguration,
        device_choice: DeviceChoice,
        seed: u64,
    ) -> BrainResult<Self> {
        let directory = BrainDirectory::new(root);
        if directory.holds_brain() {
            return Err(BrainError::InvalidConfiguration(format!(
                "a brain already exists at {}",
                directory.root().display()
            )));
        }
        model_configuration.validate()?;
        directory.create_layout()?;
        store_document(&directory.model_configuration_path(), &model_configuration)?;

        let compute_device = ComputeDevice::select(device_choice)?;
        compute_device.training.seed(seed);
        let slow_model = LanguageModel::new(&model_configuration, &compute_device.training);
        let brain = Self::assemble(BrainParts {
            directory,
            model_configuration,
            settings: RuntimeSettings::default(),
            state: newborn_state(),
            compute_device,
            slow_model,
        })?;
        brain.save()?;
        Ok(brain)
    }

    /// Opens an existing brain.
    pub fn open(root: impl Into<PathBuf>, device_choice: DeviceChoice) -> BrainResult<Self> {
        let directory = BrainDirectory::new(root);
        if !directory.holds_brain() {
            return Err(BrainError::NotFound(format!(
                "no brain at {}; create one first",
                directory.root().display()
            )));
        }
        let model_configuration: ModelConfiguration =
            load_document(&directory.model_configuration_path())?;
        model_configuration.validate()?;

        let compute_device = ComputeDevice::select(device_choice)?;
        let slow_model = LanguageModel::new(&model_configuration, &compute_device.training)
            .try_load_file(directory.slow_weights_path())
            .map_err(weights_error)?;
        let mut brain = Self::assemble(BrainParts {
            settings: load_document_or_default(&directory.settings_path())?,
            state: load_document_or_default(&directory.state_path())?,
            directory,
            model_configuration,
            compute_device,
            slow_model,
        })?;
        brain.load_fast_weights()?;
        Ok(brain)
    }

    fn assemble(parts: BrainParts) -> BrainResult<Self> {
        let adapted_predictor = attach_fast_weights(
            &parts.slow_model.predictor,
            &parts.model_configuration.adapter,
        );
        let inference = build_inference(
            &parts.slow_model,
            &adapted_predictor,
            &parts.model_configuration,
            &parts.compute_device,
        );
        let memory = EpisodicStore::open(
            &parts.directory.memory_store_path(),
            parts.model_configuration.latent.width,
        )?;
        Ok(Self {
            fast_weights: TrainableModule::new(adapted_predictor, build_fast_weight_optimizer()),
            inference,
            memory,
            directory: parts.directory,
            model_configuration: parts.model_configuration,
            settings: parts.settings,
            state: parts.state,
            compute_device: parts.compute_device,
            slow_model: parts.slow_model,
        })
    }

    /// Writes weights, settings and counters to disk.
    pub fn save(&self) -> BrainResult<()> {
        self.save_slow_weights()?;
        self.save_fast_weights()?;
        self.save_settings()?;
        self.save_state()
    }

    fn save_slow_weights(&self) -> BrainResult<()> {
        self.slow_model
            .clone()
            .save_file(self.directory.slow_weights_path())
            .map_err(weights_error)
    }

    fn save_fast_weights(&self) -> BrainResult<()> {
        self.fast_weight_record()
            .save(self.directory.fast_weights_path())
            .map_err(weights_error)
    }

    fn save_settings(&self) -> BrainResult<()> {
        store_document(&self.directory.settings_path(), &self.settings)
    }

    fn save_state(&self) -> BrainResult<()> {
        store_document(&self.directory.state_path(), &self.state)
    }

    /// The current fast weights alone, without the slow weights they attach to.
    fn fast_weight_record(&self) -> ModuleRecord {
        self.fast_weights
            .network()
            .clone()
            .into_record_group(fast_weight_group())
    }

    fn load_fast_weights(&mut self) -> BrainResult<()> {
        let fast_weights_path = self.directory.fast_weights_path();
        if !fast_weights_path.exists() {
            return Ok(());
        }
        let record = ModuleRecord::load(&fast_weights_path).map_err(weights_error)?;
        self.replace_fast_weights_from(record)
    }

    /// Attaches fresh fast weights to the current slow predictor, then fills them from a record.
    fn replace_fast_weights_from(&mut self, record: ModuleRecord) -> BrainResult<()> {
        let adapted_predictor = attach_fast_weights(
            &self.slow_model.predictor,
            &self.model_configuration.adapter,
        )
        .try_load_record(record.allow_partial(true))
        .map_err(weights_error)?;
        self.install_fast_weights(adapted_predictor);
        Ok(())
    }

    /// Makes `adapted_predictor` the predictor used for both learning and inference.
    fn install_fast_weights(&mut self, adapted_predictor: ConceptPredictor) {
        self.inference.replace_predictor(adapted_predictor.valid());
        self.fast_weights = TrainableModule::new(adapted_predictor, build_fast_weight_optimizer());
    }

    /// Removes everything the fast weights have learned since the last sleep.
    pub fn reset_fast_weights(&mut self) -> BrainResult<()> {
        let fresh_predictor = attach_fast_weights(
            &self.slow_model.predictor,
            &self.model_configuration.adapter,
        );
        self.install_fast_weights(fresh_predictor);
        self.save_fast_weights()
    }

    /// Rebuilds the gradient-free copies after the slow weights changed.
    fn refresh_inference(&mut self) {
        self.inference = build_inference(
            &self.slow_model,
            self.fast_weights.network(),
            &self.model_configuration,
            &self.compute_device,
        );
    }

    /// Where this brain keeps its files.
    pub fn directory(&self) -> &BrainDirectory {
        &self.directory
    }

    /// The architecture this brain was created with.
    pub fn model_configuration(&self) -> &ModelConfiguration {
        &self.model_configuration
    }

    /// The current adjustable behaviour.
    pub fn settings(&self) -> &RuntimeSettings {
        &self.settings
    }

    /// Replaces the adjustable behaviour and saves it.
    pub fn replace_settings(&mut self, settings: RuntimeSettings) -> BrainResult<()> {
        self.settings = settings;
        self.save_settings()
    }

    /// Counters and running statistics.
    pub fn state(&self) -> &BrainState {
        &self.state
    }

    /// The compute backend in use.
    pub fn device_choice(&self) -> DeviceChoice {
        self.compute_device.choice
    }
}

/// The state of a brain that has just been created and has experienced nothing.
fn newborn_state() -> BrainState {
    let mut state = BrainState::default();
    state.sleep.awake_since_unix_seconds = current_unix_seconds();
    state
}

/// The pieces a brain is assembled from, before its derived parts are built.
struct BrainParts {
    directory: BrainDirectory,
    model_configuration: ModelConfiguration,
    settings: RuntimeSettings,
    state: BrainState,
    compute_device: ComputeDevice,
    slow_model: LanguageModel,
}

fn build_inference(
    slow_model: &LanguageModel,
    adapted_predictor: &ConceptPredictor,
    model_configuration: &ModelConfiguration,
    compute_device: &ComputeDevice,
) -> InferenceModel {
    InferenceModel::new(
        slow_model.encoder.valid(),
        adapted_predictor.valid(),
        slow_model.decoder.valid(),
        model_configuration,
        &compute_device.inference,
    )
}

/// Bundles the frozen parts a predictor needs to be taught a lesson.
fn lesson_context<'brain>(
    inference: &'brain InferenceModel,
    compute_device: &'brain ComputeDevice,
    model_configuration: &'brain ModelConfiguration,
    settings: &'brain RuntimeSettings,
) -> LessonContext<'brain> {
    LessonContext {
        encoder: inference.encoder(),
        decoder: inference.decoder(),
        device: &compute_device.inference,
        model_configuration,
        objective: &settings.objective,
    }
}

fn weights_error(error: impl std::fmt::Display) -> BrainError {
    BrainError::Weights(error.to_string())
}
