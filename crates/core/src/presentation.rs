//! Human-readable wording for numbers and times, shared by every front end.

use crate::learning::{LiveUpdateReport, LossBreakdown};
use crate::memory::current_unix_seconds;

const THOUSAND: f64 = 1e3;
const MILLION: f64 = 1e6;
const BILLION: f64 = 1e9;
const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3_600;
const SECONDS_PER_DAY: u64 = 86_400;
const LINE_BREAK_MARK: &str = "¶";
const TAB_MARK: &str = "»";

/// Writes a count compactly: `950`, `12.5K`, `1.3M`, `2.1B`.
pub fn describe_count(count: usize) -> String {
    let value = count as f64;
    if value >= BILLION {
        return format!("{:.2}B", value / BILLION);
    }
    if value >= MILLION {
        return format!("{:.1}M", value / MILLION);
    }
    if value >= THOUSAND {
        return format!("{:.1}K", value / THOUSAND);
    }
    count.to_string()
}

/// Writes how long ago a moment was: `just now`, `5m ago`, `3h ago`, `2d ago`.
///
/// A timestamp of zero means the event never happened.
pub fn describe_age(unix_seconds: u64) -> String {
    if unix_seconds == 0 {
        return "never".to_string();
    }
    let age_seconds = current_unix_seconds().saturating_sub(unix_seconds);
    if age_seconds < SECONDS_PER_MINUTE {
        return "just now".to_string();
    }
    if age_seconds < SECONDS_PER_HOUR {
        return format!("{}m ago", age_seconds / SECONDS_PER_MINUTE);
    }
    if age_seconds < SECONDS_PER_DAY {
        return format!("{}h ago", age_seconds / SECONDS_PER_HOUR);
    }
    format!("{}d ago", age_seconds / SECONDS_PER_DAY)
}

/// One line naming every loss term.
pub fn describe_losses(losses: &LossBreakdown) -> String {
    format!(
        "loss {:.3}  rebuild {:.3}  write {:.3} ({:.2} bits/byte)  latent {:.3}  spread {:.3}",
        losses.total,
        losses.reconstruction,
        losses.generation,
        losses.generation_bits_per_byte(),
        losses.latent,
        losses.variance,
    )
}

/// One phrase describing a fast-weight update.
pub fn describe_live_update(update: &LiveUpdateReport) -> String {
    format!(
        "fast weights updated in {} steps (loss {:.3} -> {:.3})",
        update.step_count, update.loss_before, update.loss_after
    )
}

/// Shows text on one line by making line breaks and tabs visible.
///
/// The marks are Latin-1 characters, which every terminal and the desktop
/// application's built-in font can draw.
pub fn describe_on_one_line(text: &str) -> String {
    text.replace('\n', LINE_BREAK_MARK)
        .replace('\t', TAB_MARK)
        .replace('\r', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_use_the_largest_fitting_unit() {
        assert_eq!(describe_count(950), "950");
        assert_eq!(describe_count(12_500), "12.5K");
        assert_eq!(describe_count(1_300_000), "1.3M");
        assert_eq!(describe_count(2_100_000_000), "2.10B");
    }

    #[test]
    fn ages_are_relative_to_now() {
        let now = current_unix_seconds();

        assert_eq!(describe_age(0), "never");
        assert_eq!(describe_age(now), "just now");
        assert_eq!(describe_age(now - 5 * SECONDS_PER_MINUTE), "5m ago");
        assert_eq!(describe_age(now - 3 * SECONDS_PER_HOUR), "3h ago");
        assert_eq!(describe_age(now - 2 * SECONDS_PER_DAY), "2d ago");
    }

    #[test]
    fn line_breaks_become_visible() {
        assert_eq!(describe_on_one_line("a\nb\tc"), "a¶b»c");
    }
}
