//! Writes a reply: predict a meaning, refine it, decode bytes, repeat.

use burn::tensor::Tensor;

use crate::cancellation::CancellationFlag;
use crate::configuration::GenerationSettings;
use crate::model::byte_sampler::ByteSampler;
use crate::model::inference::InferenceModel;
use crate::model::tensor_readback::read_scalar;

const LINE_END: u8 = b'\n';

/// Stands in for the context when there is no text yet: the start of a new line.
const START_OF_TEXT_PATCH: &[u8] = b"\n";

/// What to continue and how.
#[derive(Debug, Clone, Copy)]
pub struct ReplyRequest<'request> {
    /// The text so far; the reply continues it.
    pub context: &'request [u8],
    /// Gist vectors of retrieved memories shown to the predictor.
    pub memories: &'request [Vec<f32>],
    /// Sampling and length settings.
    pub generation: &'request GenerationSettings,
    /// Whether the reply ends at the first line break.
    pub is_stopping_at_line_end: bool,
    /// Whether to record what each thinking stage would have written.
    pub is_tracing_thoughts: bool,
}

/// Why a reply ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model wrote a line break.
    LineEnd,
    /// The reply reached the configured length limit.
    LengthLimit,
    /// The model had nothing more to add.
    NothingMoreToWrite,
    /// The caller asked the model to stop.
    Cancelled,
}

/// One stage of thinking about a patch: a meaning vector and what it would say.
#[derive(Debug, Clone, PartialEq)]
pub struct ThoughtStage {
    /// What the decoder would most likely write from this stage's meaning vector.
    pub preview: String,
    /// How far this stage's meaning vector is from the final one, relative to
    /// the final one's length. Zero for the last stage.
    pub distance_to_final: f64,
}

/// The coarse-to-fine thinking behind one written patch.
#[derive(Debug, Clone, PartialEq)]
pub struct ThoughtTrace {
    /// Every thinking stage, rough guess first, fully refined last.
    pub stages: Vec<ThoughtStage>,
    /// What was actually written.
    pub written: String,
}

/// A finished reply.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    /// The bytes the model wrote.
    pub bytes: Vec<u8>,
    /// One trace per written patch, when tracing was requested.
    pub thoughts: Vec<ThoughtTrace>,
    /// Why the reply ended.
    pub stop_reason: StopReason,
}

impl Reply {
    /// The reply as text, with invalid UTF-8 shown as replacement characters.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

/// Writes a reply patch by patch, reporting bytes through `on_bytes` as they are produced.
pub fn write_reply(
    model: &mut InferenceModel,
    request: ReplyRequest<'_>,
    cancellation: &CancellationFlag,
    on_bytes: &mut dyn FnMut(&[u8]),
) -> Reply {
    let mut sampler = ByteSampler::new(request.generation);
    let mut text = request.context.to_vec();
    let mut reply = Reply {
        bytes: Vec::new(),
        thoughts: Vec::new(),
        stop_reason: StopReason::LengthLimit,
    };

    loop {
        if cancellation.is_requested() {
            reply.stop_reason = StopReason::Cancelled;
            return reply;
        }
        let step = write_next_patch(model, &request, &text, &mut sampler);
        let kept_bytes = keep_allowed_bytes(&step.new_bytes, &reply.bytes, &request);
        on_bytes(kept_bytes);
        text.extend_from_slice(kept_bytes);
        reply.bytes.extend_from_slice(kept_bytes);
        reply.thoughts.extend(step.thought);

        if let Some(stop_reason) = find_stop_reason(&step.new_bytes, &reply.bytes, &request) {
            reply.stop_reason = stop_reason;
            return reply;
        }
    }
}

/// The bytes one generation step added, and the thinking behind them.
struct PatchStep {
    new_bytes: Vec<u8>,
    thought: Option<ThoughtTrace>,
}

fn write_next_patch(
    model: &mut InferenceModel,
    request: &ReplyRequest<'_>,
    text: &[u8],
    sampler: &mut ByteSampler,
) -> PatchStep {
    let patcher = model.patcher();
    let patches = patcher.split(text);
    let (closed_patches, open_prefix) =
        separate_open_patch(&patches, |patch| patcher.is_patch_open(patch));
    let window_start = closed_patches
        .len()
        .saturating_sub(model.maximum_context_patches());
    let window = &closed_patches[window_start..];

    let context_latents = match window.is_empty() {
        true => model.encode_patches(&[START_OF_TEXT_PATCH]),
        false => model.encode_patches(window),
    };
    let stages = model.predict_next(
        &context_latents,
        request.memories,
        request.generation.thinking_step_count,
    );
    let refined_meaning = stages[stages.len() - 1].clone();
    let patch = model.write_patch(refined_meaning, open_prefix, sampler);
    let new_bytes = patch[open_prefix.len()..].to_vec();

    let thought = request
        .is_tracing_thoughts
        .then(|| trace_thought(model, &stages, open_prefix, &new_bytes));
    PatchStep { new_bytes, thought }
}

/// Splits patches into the finished ones and the unfinished word at the end, if any.
fn separate_open_patch<'text>(
    patches: &'text [&'text [u8]],
    is_patch_open: impl Fn(&[u8]) -> bool,
) -> (&'text [&'text [u8]], &'text [u8]) {
    match patches.split_last() {
        Some((last_patch, closed_patches)) if is_patch_open(last_patch) => {
            (closed_patches, last_patch)
        }
        _ => (patches, &[]),
    }
}

/// Shows what every thinking stage would have written for this patch.
fn trace_thought(
    model: &InferenceModel,
    stages: &[Tensor<2>],
    open_prefix: &[u8],
    new_bytes: &[u8],
) -> ThoughtTrace {
    let final_meaning = &stages[stages.len() - 1];
    let stages = stages
        .iter()
        .map(|stage| {
            let preview = model.write_patch(stage.clone(), open_prefix, &mut ByteSampler::greedy());
            ThoughtStage {
                preview: String::from_utf8_lossy(&preview[open_prefix.len()..]).into_owned(),
                distance_to_final: relative_distance(stage.clone(), final_meaning.clone()),
            }
        })
        .collect();
    ThoughtTrace {
        stages,
        written: String::from_utf8_lossy(new_bytes).into_owned(),
    }
}

/// Distance between two meaning vectors as a fraction of the second one's length.
fn relative_distance(meaning: Tensor<2>, reference: Tensor<2>) -> f64 {
    let distance = read_scalar((meaning - reference.clone()).square().sum().sqrt());
    let reference_length = read_scalar(reference.square().sum().sqrt());
    distance / reference_length.max(f64::EPSILON)
}

/// Trims new bytes so the reply respects the line-end rule and the length limit.
fn keep_allowed_bytes<'bytes>(
    new_bytes: &'bytes [u8],
    reply_so_far: &[u8],
    request: &ReplyRequest<'_>,
) -> &'bytes [u8] {
    let remaining_bytes = request
        .generation
        .maximum_reply_bytes
        .saturating_sub(reply_so_far.len());
    let within_limit = &new_bytes[..new_bytes.len().min(remaining_bytes)];
    if !request.is_stopping_at_line_end {
        return within_limit;
    }
    match within_limit.iter().position(|&byte| byte == LINE_END) {
        Some(line_end_position) => &within_limit[..=line_end_position],
        None => within_limit,
    }
}

fn find_stop_reason(
    new_bytes: &[u8],
    reply_so_far: &[u8],
    request: &ReplyRequest<'_>,
) -> Option<StopReason> {
    if new_bytes.is_empty() {
        return Some(StopReason::NothingMoreToWrite);
    }
    if request.is_stopping_at_line_end && reply_so_far.contains(&LINE_END) {
        return Some(StopReason::LineEnd);
    }
    if reply_so_far.len() >= request.generation.maximum_reply_bytes {
        return Some(StopReason::LengthLimit);
    }
    None
}
