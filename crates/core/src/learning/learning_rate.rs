//! How the step size changes over a pretraining run.

use crate::configuration::TrainingScheduleOptions;

/// Linear warm-up followed by cosine decay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRateSchedule {
    warmup_step_count: usize,
    step_count: usize,
    peak_learning_rate: f64,
    final_learning_rate: f64,
}

impl LearningRateSchedule {
    /// Builds the schedule described by the run's options.
    pub fn new(schedule: &TrainingScheduleOptions) -> Self {
        Self {
            warmup_step_count: schedule.warmup_step_count.min(schedule.step_count),
            step_count: schedule.step_count,
            peak_learning_rate: schedule.peak_learning_rate,
            final_learning_rate: schedule.peak_learning_rate
                * schedule.final_learning_rate_fraction.clamp(0.0, 1.0),
        }
    }

    /// The learning rate for a zero-based step of the run.
    pub fn rate_at(&self, run_step: usize) -> f64 {
        if run_step < self.warmup_step_count {
            let warmup_progress = (run_step + 1) as f64 / self.warmup_step_count as f64;
            return self.peak_learning_rate * warmup_progress;
        }

        let decay_step_count = (self.step_count - self.warmup_step_count).max(1);
        let decay_progress =
            ((run_step - self.warmup_step_count) as f64 / decay_step_count as f64).min(1.0);
        let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * decay_progress).cos());
        self.final_learning_rate
            + (self.peak_learning_rate - self.final_learning_rate) * cosine_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule() -> LearningRateSchedule {
        LearningRateSchedule::new(&TrainingScheduleOptions {
            step_count: 100,
            warmup_step_count: 10,
            peak_learning_rate: 1e-3,
            final_learning_rate_fraction: 0.1,
        })
    }

    #[test]
    fn the_rate_rises_during_warmup_and_peaks_at_its_end() {
        let schedule = schedule();

        assert!(schedule.rate_at(0) < schedule.rate_at(5));
        assert!((schedule.rate_at(9) - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn the_rate_decays_to_the_final_fraction() {
        let schedule = schedule();

        assert!(schedule.rate_at(50) < schedule.rate_at(10));
        assert!((schedule.rate_at(100) - 1e-4).abs() < 1e-9);
        assert!((schedule.rate_at(5_000) - 1e-4).abs() < 1e-9);
    }
}
