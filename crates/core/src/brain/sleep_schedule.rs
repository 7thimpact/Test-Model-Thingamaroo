//! Decides when a brain should sleep on its own.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::brain::Brain;
use crate::brain::state::BrainState;
use crate::configuration::SleepScheduleSettings;
use crate::error::BrainResult;
use crate::memory::current_unix_seconds;

const SECONDS_PER_HOUR: u64 = 3_600;

/// Why sleep is due.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepTrigger {
    /// Enough conversation turns have passed.
    TurnInterval,
    /// The memory store is full enough to need pruning.
    MemoryLimit,
    /// The conversation has been idle for a while.
    Idle,
    /// The brain has been awake for a long time.
    HourInterval,
}

impl SleepTrigger {
    /// A short explanation for the user.
    pub fn description(self) -> &'static str {
        match self {
            SleepTrigger::TurnInterval => "enough turns have passed since the last sleep",
            SleepTrigger::MemoryLimit => "the memory store has reached its sleep limit",
            SleepTrigger::Idle => "the conversation has been idle",
            SleepTrigger::HourInterval => "the brain has been awake for a long time",
        }
    }
}

/// What the schedule is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SleepPressure {
    /// How long the conversation has been waiting for input.
    pub idle: Duration,
    /// Number of stored memories.
    pub memory_count: usize,
    /// The current time.
    pub current_unix_seconds: u64,
}

/// Returns the first reason sleep is due, or `None` when the brain can stay awake.
///
/// A brain with no new experience since its last sleep never needs to sleep.
pub fn find_sleep_trigger(
    schedule: &SleepScheduleSettings,
    state: &BrainState,
    pressure: SleepPressure,
) -> Option<SleepTrigger> {
    let has_new_experience = state.conversation.turns_since_sleep > 0;
    if !schedule.is_automatic || !has_new_experience {
        return None;
    }

    // A brain saved before wake times were recorded has none; the hourly rule skips it.
    let has_known_wake_time = state.sleep.awake_since_unix_seconds > 0;
    let hours_awake = pressure
        .current_unix_seconds
        .saturating_sub(state.sleep.awake_since_unix_seconds)
        / SECONDS_PER_HOUR;
    let checks = [
        (
            state.conversation.turns_since_sleep >= schedule.turn_interval as u64,
            SleepTrigger::TurnInterval,
        ),
        (
            pressure.memory_count >= schedule.memory_count_limit,
            SleepTrigger::MemoryLimit,
        ),
        (
            pressure.idle.as_secs() >= schedule.idle_seconds,
            SleepTrigger::Idle,
        ),
        (
            has_known_wake_time && hours_awake >= schedule.hour_interval,
            SleepTrigger::HourInterval,
        ),
    ];
    checks
        .into_iter()
        .find_map(|(is_due, trigger)| is_due.then_some(trigger))
}

impl Brain {
    /// Checks whether the sleep schedule says it is time to consolidate.
    ///
    /// `idle` is how long the conversation has been waiting for input.
    pub fn sleep_trigger(&self, idle: Duration) -> BrainResult<Option<SleepTrigger>> {
        let pressure = SleepPressure {
            idle,
            memory_count: self.memory.entry_count()?,
            current_unix_seconds: current_unix_seconds(),
        };
        Ok(find_sleep_trigger(
            &self.settings.schedule,
            &self.state,
            pressure,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pressure(
        idle_seconds: u64,
        memory_count: usize,
        current_unix_seconds: u64,
    ) -> SleepPressure {
        SleepPressure {
            idle: Duration::from_secs(idle_seconds),
            memory_count,
            current_unix_seconds,
        }
    }

    const WAKE_TIME: u64 = 1_000;

    fn state_with_turns(turns_since_sleep: u64) -> BrainState {
        let mut state = BrainState::default();
        state.conversation.turns_since_sleep = turns_since_sleep;
        state.sleep.awake_since_unix_seconds = WAKE_TIME;
        state
    }

    #[test]
    fn a_brain_with_no_new_experience_stays_awake() {
        let schedule = SleepScheduleSettings::default();

        let trigger = find_sleep_trigger(
            &schedule,
            &state_with_turns(0),
            pressure(10_000, 99_999, 999_999),
        );

        assert_eq!(trigger, None);
    }

    #[test]
    fn each_condition_triggers_sleep_on_its_own() {
        let schedule = SleepScheduleSettings {
            is_automatic: true,
            turn_interval: 5,
            idle_seconds: 60,
            hour_interval: 24,
            memory_count_limit: 100,
        };

        let cases = [
            (
                state_with_turns(5),
                pressure(0, 0, WAKE_TIME),
                SleepTrigger::TurnInterval,
            ),
            (
                state_with_turns(1),
                pressure(0, 100, WAKE_TIME),
                SleepTrigger::MemoryLimit,
            ),
            (
                state_with_turns(1),
                pressure(60, 0, WAKE_TIME),
                SleepTrigger::Idle,
            ),
            (
                state_with_turns(1),
                pressure(0, 0, WAKE_TIME + 24 * SECONDS_PER_HOUR),
                SleepTrigger::HourInterval,
            ),
        ];
        for (state, pressure, expected_trigger) in cases {
            assert_eq!(
                find_sleep_trigger(&schedule, &state, pressure),
                Some(expected_trigger)
            );
        }
        assert_eq!(
            find_sleep_trigger(&schedule, &state_with_turns(1), pressure(0, 0, WAKE_TIME)),
            None
        );
    }

    #[test]
    fn a_brain_without_a_recorded_wake_time_ignores_the_hourly_rule() {
        let schedule = SleepScheduleSettings::default();
        let mut state = state_with_turns(1);
        state.sleep.awake_since_unix_seconds = 0;

        let trigger = find_sleep_trigger(&schedule, &state, pressure(0, 0, 999_999_999));

        assert_eq!(trigger, None);
    }

    #[test]
    fn switching_the_schedule_off_disables_every_trigger() {
        let schedule = SleepScheduleSettings {
            is_automatic: false,
            ..SleepScheduleSettings::default()
        };

        let trigger = find_sleep_trigger(
            &schedule,
            &state_with_turns(500),
            pressure(10_000, 99_999, 999_999),
        );

        assert_eq!(trigger, None);
    }
}
