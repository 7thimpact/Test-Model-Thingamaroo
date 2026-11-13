//! A pretraining run: the loop around [`Pretrainer`], with reporting and checkpoints.

use std::time::Instant;

use burn::module::Module;
use serde::{Deserialize, Serialize};

use crate::brain::history::{HistoryRecord, append_record};
use crate::brain::{Brain, weights_error};
use crate::cancellation::CancellationFlag;
use crate::configuration::{ModelConfiguration, TrainingOptions};
use crate::error::BrainResult;
use crate::learning::{
    LossBreakdown, Pretrainer, PretrainerParts, ProgressEvent, ProgressObserver, TargetEncoder,
    TrainingStepReport, ValidationReport, attach_fast_weights, build_optimizer,
};
use crate::memory::SurpriseBaseline;
use crate::model::{InferenceModel, LanguageModel, ReplyRequest, write_reply};
use crate::random::RandomSource;
use crate::text::{Corpus, CorpusSplit, Patcher, ReplayReservoir};

/// What a pretraining run achieved.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PretrainingSummary {
    /// Gradient steps taken in this run.
    pub step_count: usize,
    /// Whether the user stopped the run early.
    pub was_cancelled: bool,
    /// Losses on the last training batch.
    pub last_training_losses: LossBreakdown,
    /// The last measurement on held-out text, if any was taken.
    pub last_validation: Option<ValidationReport>,
    /// Wall-clock duration of the run.
    pub elapsed_seconds: f64,
}

/// Tracks training speed between progress reports.
struct ThroughputMeter {
    last_report: Instant,
    steps_since_report: usize,
}

impl ThroughputMeter {
    fn new() -> Self {
        Self {
            last_report: Instant::now(),
            steps_since_report: 0,
        }
    }

    fn count_step(&mut self) {
        self.steps_since_report += 1;
    }

    /// Patches per second since the last call, then starts a new measurement.
    fn take_patches_per_second(&mut self, patches_per_step: usize) -> f64 {
        let elapsed_seconds = self.last_report.elapsed().as_secs_f64().max(f64::EPSILON);
        let patches_per_second =
            (self.steps_since_report * patches_per_step) as f64 / elapsed_seconds;
        *self = Self::new();
        patches_per_second
    }
}

impl Brain {
    /// Trains the whole stack on a text corpus.
    ///
    /// Can be called again to continue training: the optimiser state and the
    /// target encoder are resumed from the previous run. Because pretraining
    /// replaces the weights that snapshots and fast weights attach to, both
    /// are cleared when a run starts.
    pub fn pretrain(
        &mut self,
        options: &TrainingOptions,
        cancellation: &CancellationFlag,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<PretrainingSummary> {
        options.validate()?;
        let corpus = self.open_corpus(options)?;
        on_progress(ProgressEvent::Status(format!(
            "Training on {} file(s), {:.1} MB of text",
            corpus.file_count(),
            corpus.total_bytes() as f64 / 1e6
        )));
        self.delete_all_snapshots()?;

        let mut pretrainer = self.begin_pretraining(options, corpus.clone())?;
        let outcome = self.run_training_steps(&mut pretrainer, options, cancellation, on_progress);
        // The model is put back before any error is reported, so a failed run never loses it.
        self.end_pretraining(pretrainer)?;
        let summary = outcome?;

        on_progress(ProgressEvent::Status(
            "Preparing replay data for sleep".into(),
        ));
        self.prepare_for_sleep(options, &corpus)?;
        self.seed_surprise_baseline(&summary);
        self.save()?;
        Ok(summary)
    }

    fn open_corpus(&self, options: &TrainingOptions) -> BrainResult<Corpus> {
        let patcher = Patcher::new(self.model_configuration.patching.maximum_patch_bytes);
        let corpus = Corpus::open(
            &options.data.corpus_paths,
            options.data.validation_fraction,
            patcher,
        )?;
        // Sampling once up front reports a too-small corpus before any work is done.
        let mut probe_random = RandomSource::from_seed(options.data.seed);
        corpus.sample_patches(
            &mut probe_random,
            CorpusSplit::Training,
            self.sequence_patch_count(),
        )?;
        Ok(corpus)
    }

    /// One more patch than the working memory holds: the last one is only predicted.
    fn sequence_patch_count(&self) -> usize {
        self.model_configuration.predictor.maximum_context_patches + 1
    }

    /// Moves the slow model into a pretrainer, resuming optimiser and target encoder if saved.
    fn begin_pretraining(
        &mut self,
        options: &TrainingOptions,
        corpus: Corpus,
    ) -> BrainResult<Pretrainer> {
        let momentum = self.settings.objective.target_encoder_momentum;
        let target_encoder_path = self.directory.target_encoder_path();
        let target_encoder = match target_encoder_path.exists() {
            true => TargetEncoder::resumed(
                self.slow_model
                    .encoder
                    .valid()
                    .try_load_file(&target_encoder_path)
                    .map_err(weights_error)?,
                momentum,
            ),
            false => TargetEncoder::tracking(&self.slow_model.encoder, momentum),
        };

        let optimizer_path = self.directory.optimizer_path();
        let mut optimizer = build_optimizer(&options.optimizer);
        if optimizer_path.exists() {
            optimizer = optimizer.load(&optimizer_path).map_err(weights_error)?;
        }

        Ok(Pretrainer::new(PretrainerParts {
            model: self.take_slow_model(),
            optimizer,
            target_encoder,
            corpus,
            options: options.clone(),
            objective: self.settings.objective.clone(),
            model_configuration: self.model_configuration.clone(),
            device: self.compute_device.inference.clone(),
        }))
    }

    /// Hands the slow model over for training and releases every other copy of its weights.
    ///
    /// Until [`Brain::end_pretraining`] puts the trained model back, the brain
    /// holds a minimal stand-in, so training never keeps a second full copy of
    /// the weights in memory.
    fn take_slow_model(&mut self) -> LanguageModel {
        let stand_in_configuration = stand_in_configuration(&self.model_configuration);
        let stand_in = LanguageModel::new(&stand_in_configuration, &self.compute_device.training);
        let slow_model = std::mem::replace(&mut self.slow_model, stand_in);
        let stand_in_predictor = attach_fast_weights(
            &self.slow_model.predictor,
            &self.model_configuration.adapter,
        );
        self.install_fast_weights(stand_in_predictor);
        self.refresh_inference();
        slow_model
    }

    fn run_training_steps(
        &mut self,
        pretrainer: &mut Pretrainer,
        options: &TrainingOptions,
        cancellation: &CancellationFlag,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<PretrainingSummary> {
        let started = Instant::now();
        let mut summary = PretrainingSummary::default();
        let mut throughput = ThroughputMeter::new();

        for run_step in 0..options.schedule.step_count {
            if cancellation.is_requested() {
                summary.was_cancelled = true;
                break;
            }
            summary.last_training_losses = pretrainer.train_step(run_step)?;
            summary.step_count = run_step + 1;
            self.state.training.step_count += 1;
            throughput.count_step();

            let step = StepPosition {
                run_step,
                step_count: options.schedule.step_count,
            };
            self.report_training_step(
                pretrainer,
                &summary,
                step,
                &mut throughput,
                options,
                on_progress,
            )?;
            self.validate_if_due(pretrainer, &mut summary, step, options, on_progress)?;
            self.checkpoint_if_due(pretrainer, step, options, on_progress)?;
        }
        summary.elapsed_seconds = started.elapsed().as_secs_f64();
        Ok(summary)
    }

    fn report_training_step(
        &self,
        pretrainer: &Pretrainer,
        summary: &PretrainingSummary,
        step: StepPosition,
        throughput: &mut ThroughputMeter,
        options: &TrainingOptions,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<()> {
        if !step.is_due(options.reporting.report_interval_steps) {
            return Ok(());
        }
        let report = TrainingStepReport {
            lifetime_step: self.state.training.step_count,
            run_step: step.run_step + 1,
            run_step_count: step.step_count,
            learning_rate: pretrainer.learning_rate_at(step.run_step),
            losses: summary.last_training_losses,
            patches_per_second: throughput.take_patches_per_second(pretrainer.patches_per_step()),
        };
        append_record(
            &self.directory.history_path(),
            &HistoryRecord::TrainingStep(report),
        )?;
        on_progress(ProgressEvent::TrainingStep(report));
        Ok(())
    }

    fn validate_if_due(
        &mut self,
        pretrainer: &mut Pretrainer,
        summary: &mut PretrainingSummary,
        step: StepPosition,
        options: &TrainingOptions,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<()> {
        if !step.is_due(options.reporting.validation_interval_steps) {
            return Ok(());
        }
        let losses = pretrainer.validate()?;
        let report = ValidationReport {
            lifetime_step: self.state.training.step_count,
            losses,
            sample: self.write_sample(pretrainer.model(), &options.reporting.sample_prompt),
        };
        self.state.training.validation_bits_per_byte = Some(losses.generation_bits_per_byte());
        append_record(
            &self.directory.history_path(),
            &HistoryRecord::Validation(report.clone()),
        )?;
        on_progress(ProgressEvent::Validation(report.clone()));
        summary.last_validation = Some(report);
        Ok(())
    }

    /// Continues the sample prompt with the model as it is mid-training.
    fn write_sample(&self, model: &LanguageModel, sample_prompt: &str) -> String {
        let frozen_model = model.valid();
        let mut sample_model = InferenceModel::new(
            frozen_model.encoder,
            frozen_model.predictor,
            frozen_model.decoder,
            &self.model_configuration,
            &self.compute_device.inference,
        );
        let request = ReplyRequest {
            context: sample_prompt.as_bytes(),
            memories: &[],
            generation: &self.settings.generation,
            is_stopping_at_line_end: false,
            is_tracing_thoughts: false,
        };
        let reply = write_reply(
            &mut sample_model,
            request,
            &CancellationFlag::new(),
            &mut |_| {},
        );
        format!("{sample_prompt}{}", reply.text())
    }

    fn checkpoint_if_due(
        &self,
        pretrainer: &Pretrainer,
        step: StepPosition,
        options: &TrainingOptions,
        on_progress: ProgressObserver<'_>,
    ) -> BrainResult<()> {
        let is_last_step = step.run_step + 1 == step.step_count;
        if is_last_step || !step.is_due(options.reporting.checkpoint_interval_steps) {
            return Ok(());
        }
        pretrainer
            .model()
            .clone()
            .save_file(self.directory.slow_weights_path())
            .map_err(weights_error)?;
        self.save_state()?;
        on_progress(ProgressEvent::CheckpointSaved {
            lifetime_step: self.state.training.step_count,
        });
        Ok(())
    }

    /// Puts the trained model back and saves everything needed to resume training.
    fn end_pretraining(&mut self, pretrainer: Pretrainer) -> BrainResult<()> {
        let (model, optimizer, target_encoder) = pretrainer.into_parts();
        self.slow_model = model;
        let fresh_predictor = attach_fast_weights(
            &self.slow_model.predictor,
            &self.model_configuration.adapter,
        );
        self.install_fast_weights(fresh_predictor);
        self.refresh_inference();

        self.save_slow_weights()?;
        self.save_fast_weights()?;
        optimizer
            .save(self.directory.optimizer_path())
            .map_err(weights_error)?;
        target_encoder
            .encoder()
            .clone()
            .save_file(self.directory.target_encoder_path())
            .map_err(weights_error)
    }

    /// Keeps samples of the training data and records which weights matter for them.
    fn prepare_for_sleep(&self, options: &TrainingOptions, corpus: &Corpus) -> BrainResult<()> {
        let mut random =
            RandomSource::from_seed(options.data.seed ^ self.state.training.step_count);
        let reservoir = ReplayReservoir::sample_from(
            corpus,
            &mut random,
            options.data.replay_window_count,
            self.sequence_patch_count(),
        )?;
        reservoir.save(&self.directory.replay_reservoir_path())?;
        self.renew_elastic_anchor(&reservoir, &mut random)
    }

    /// Starts the surprise baseline at the model's typical error on unseen text.
    fn seed_surprise_baseline(&mut self, summary: &PretrainingSummary) {
        let Some(validation) = &summary.last_validation else {
            return;
        };
        self.state.surprise =
            SurpriseBaseline::seeded(validation.losses.generation_bits_per_byte());
    }
}

/// The smallest model that is still valid: a placeholder while the real one is being trained.
fn stand_in_configuration(model_configuration: &ModelConfiguration) -> ModelConfiguration {
    let mut stand_in = model_configuration.clone();
    stand_in.patching.maximum_patch_bytes = 1;
    stand_in.latent.width = 1;
    for codec in [&mut stand_in.encoder, &mut stand_in.decoder] {
        codec.layer_count = 1;
        codec.width = codec.head_count;
    }
    stand_in.predictor.layer_count = 1;
    stand_in.predictor.width = stand_in.predictor.head_count;
    stand_in.predictor.maximum_context_patches = 2;
    stand_in.predictor.maximum_memory_slots = 1;
    stand_in.adapter.rank = 1;
    stand_in
}

/// Where a step sits within its run.
#[derive(Debug, Clone, Copy)]
struct StepPosition {
    run_step: usize,
    step_count: usize,
}

impl StepPosition {
    /// Whether something that happens every `interval_steps` steps happens now.
    ///
    /// The final step is always due, so a run never ends without a report.
    fn is_due(&self, interval_steps: usize) -> bool {
        let completed_steps = self.run_step + 1;
        let is_last_step = completed_steps == self.step_count;
        let is_on_interval = interval_steps > 0 && completed_steps.is_multiple_of(interval_steps);
        is_last_step || is_on_interval
    }
}
