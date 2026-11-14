# Architecture

How the spec sheet became code: what each part is, how it learns, and where
to find it. All paths are inside `crates/core/src/`.

- [The four goals](#the-four-goals)
- [The model](#the-model)
- [One turn of conversation](#one-turn-of-conversation)
- [Learning on three timescales](#learning-on-three-timescales)
- [Safeguards](#safeguards)
- [What a brain looks like on disk](#what-a-brain-looks-like-on-disk)
- [Answers to the spec's open questions](#answers-to-the-specs-open-questions)
- [Code map](#code-map)
- [Performance notes](#performance-notes)
- [What is and is not verified](#what-is-and-is-not-verified)

## The four goals

| Goal from the spec | Where it happens |
| --- | --- |
| **Receive** raw bytes, no tokenizer | `text/patcher.rs` groups bytes into patches; `model/byte_encoder.rs` reads them. There is no vocabulary anywhere. |
| **Process by thought**: rough meaning, refine, then bytes | `model/concept_predictor.rs` predicts the next meaning vector and refines it; `model/byte_decoder.rs` writes bytes afterwards. |
| **Store** experience weighted by surprise | `memory/surprise.rs` scores each input; `memory/store.rs` keeps exchanges with their importance. |
| **Change future outcomes**, right away and permanently | Retrieved memories and fast weights change the next reply (`brain/chat.rs`, `learning/live_adapter.rs`); sleep changes the slow weights (`brain/sleep_cycle.rs`). |

## The model

Four components. Bytes come in at the bottom, meaning is handled in the middle,
bytes go out at the end.

```
bytes ──patcher──► patches ──byte encoder──► meaning vectors
                                                  │
                    retrieved memory gists ──►    ▼
                                         concept predictor
                                     (rough guess → thinking steps)
                                                  │
                                                  ▼
bytes ◄──────────── byte decoder ◄──── predicted meaning vector
```

### Patches

A patch is one word-like chunk of bytes: a run of bytes plus the whitespace
that ends it, at most `patching.maximum_patch_bytes` long. Text without
whitespace falls back to fixed-size chunks that never split a UTF-8 character,
so any byte stream in any language can be patched. This is the "space
patching" scheme from the Byte Latent Transformer paper; it needs no training
and never goes out of date.

`"The cat fell to the ground."` becomes
`[The ] [cat ] [fell ] [to ] [the ] [ground.]`.

### Byte encoder ("senses")

A small bidirectional transformer over the bytes of one patch. Byte embedding
plus position, a few blocks, the mean over the real bytes, a linear layer, and
finally a parameter-free standardisation so every meaning vector has zero mean
and unit variance across its dimensions. Because the last step has nothing to
learn, the encoder cannot shrink its outputs to make latent prediction easy.

Each patch is encoded on its own, so the meaning vector of a patch depends only
on its bytes. That makes the encoder cacheable and makes stored memory vectors
stay valid forever once the encoder is frozen.

### Latent concept predictor ("neocortex")

A causal transformer over meaning vectors. It never sees bytes.

1. The meaning vectors of the context are projected and given positions.
   Gist vectors of retrieved memories, if any, are placed in front in dedicated
   **memory slots**.
2. The transformer blocks produce a hidden state per position, and a linear
   head turns it into a **rough guess** of the next meaning vector.
3. **Thinking steps** refine the guess. Each step feeds the current guess back
   in (added to the hidden state), runs one shared transformer block over the
   whole context again, and adds a correction to the guess. The block is the
   same for every step, so the number of steps can be changed at run time
   without retraining.

Every stage of the guess is trained against the target (deep supervision), so
each step is pushed to be closer than the last. The applications can show the
trace: for each written patch, what each stage would have decoded to and how
far its vector still was from the final one.

### Byte decoder ("speech")

A small causal transformer that writes one patch, byte by byte, from a meaning
vector. The meaning is the first token and is also added to every later
position. It writes one of 257 symbols at each step: a byte, or "this patch is
finished".

The decoder is used with two kinds of input: the encoder's vector for a patch
(to rebuild that patch) and the predictor's vector (to write the next patch).
When several words are plausible, the choice is made while sampling the first
bytes, and the rest of the word follows from them, so written words are
coherent even when the predicted vector is uncertain.

### Sizes

`configuration/model.rs` holds the architecture and five presets. Parameter
counts are computed from the sizes and tested against the built model.

| Preset | Slow parameters | Latent width | Predictor | Context |
| --- | --- | --- | --- | --- |
| `tiny` | 1.3M | 96 | 128 wide, 4 blocks | 32 patches |
| `prototype` | 35.8M | 512 | 512 wide, 8 blocks | 128 patches |
| `small` | 126.9M | 512 | 768 wide, 14 blocks | 192 patches |
| `target` | 375.5M | 512 | 1024 wide, 26 blocks | 256 patches |
| `medium` | 1.35B | 768 | 1792 wide, 32 blocks | 512 patches |

## One turn of conversation

`Brain::chat_turn` in `brain/chat.rs`:

1. **Frame** the message: `chat.user_prefix` + text + line break + `chat.model_prefix`.
2. **Measure surprise.** The predictor and decoder are run over the
   conversation plus the new message, and the bits per byte needed for the
   message's patches are read off. This is the model's prediction error on
   what you just said.
3. **Score importance.** The surprise is compared with a running average
   (`memory/surprise.rs`): the logistic of its distance from the mean in
   standard deviations, so an average input scores 0.5. A message that opens
   with a correction marker is raised to `surprise.correction_importance`.
   Nothing may exceed `safety.maximum_importance`.
4. **Retrieve.** The message's gist (the average of its patches' meaning
   vectors) is compared by cosine similarity with every stored memory; the
   closest few above `memory.minimum_similarity` are shown to the predictor in
   its memory slots.
5. **Reply.** Predict the next meaning, refine it, decode a patch, append it,
   repeat until a line break or the length limit (`model/reply.rs`). The text
   is re-patched after each step, and an unfinished word at the end is given to
   the decoder as a forced prefix, so the model always continues from exactly
   the text on screen.
6. **Store.** If the importance reaches `memory.minimum_stored_importance`,
   the exchange is written to the episodic store with its gist vector.
7. **Learn.** If the message was a correction or its importance reaches
   `learning.surprise_importance_threshold`, the fast weights take a few
   gradient steps on it.

Feedback on a reply (`/correct`, `/good`, `/bad`, or the buttons in the Chat
tab) stores the exchange as trusted and teaches the reply itself.

The context window is working memory: limited, and gone when the conversation
moves on. Lasting change lives only in the weights and the episodic store.

## Learning on three timescales

### 1. Pretraining (offline)

`learning/pretraining.rs`, driven by `brain/pretraining_run.rs`. The whole
stack trains together on four losses (`model/objective.rs`):

| Loss | What it asks | Why |
| --- | --- | --- |
| **latent** | The predicted meaning vector should match the vector of the patch that actually came next (at every thinking stage). | "Predict meaning, not bytes." |
| **rebuild** | The decoder must rebuild each patch from its own meaning vector. | Forces vectors to carry real information: the primary guard against collapse. |
| **write** | The decoder must write the next patch from the predicted vector. | Makes predictions usable for generation. |
| **spread** | Each latent dimension must keep its spread across a batch (as in VICReg). | Second guard against collapse. |

The latent targets come from a slowly updated copy of the encoder
(`learning/target_encoder.rs`, as in JEPA), the third guard against collapse.

To teach the predictor to use its memory slots, about half of the training
batches are preceded by extra text that is shown only as gist vectors, exactly
as retrieved memories are shown later.

When a run ends, the brain keeps a sample of the training text (the replay
reservoir) and records how much the old-data loss depends on each predictor
weight (the elastic anchor). Both are needed for sleep.

### 2. Live learning (during conversation)

`learning/live_adapter.rs`. The predictor's slow weights are frozen. A
low-rank adapter (LoRA) on every attention and feed-forward matrix of the
predictor holds the **fast weights**; it starts with no effect. A lesson is
"given this context, the following text should come next", and only the
adapter is updated. The number of gradient steps scales with the importance,
so corrections are learned strongly and mild surprises weakly.

Retrieved memories also change the very next reply with no weight change at all.

### 3. Sleep (periodic)

`brain/sleep_cycle.rs` and `learning/consolidation.rs`.

1. Snapshot the slow predictor and the fast weights.
2. Turn the most important memories into lessons. A corrected, praised or
   hand-written memory teaches its reply; an ordinary exchange teaches only
   your side of it, because the model's own reply was never confirmed.
3. Replay those lessons mixed with windows from the replay reservoir into a
   copy of the slow predictor, with the elastic penalty holding important
   weights near their recorded values.
4. Measure the loss on old data before and after. If it rose by more than
   `sleep.maximum_old_loss_increase`, discard the copy.
5. Otherwise install it, reset (or shrink) the fast weights, record a new
   elastic anchor, decay every memory's importance, and prune the weak ones.

After sleep the corrected fact is in the slow weights, so it survives the
fast-weight reset.

## Safeguards

Against **catastrophic forgetting**:

- Fast and slow weights are separate; live updates never touch slow weights.
- Every sleep replays old training data alongside new memories.
- Elastic weight consolidation (`learning/elastic_anchor.rs`) penalises moving
  weights that mattered for old data.
- The byte encoder and decoder are frozen after pretraining, so reading and
  spelling never drift.
- A sleep whose result hurts old skills is discarded automatically.

Against **poisoning**:

- An input filter (`safety.blocked_phrases`, minimum and maximum length)
  decides what may be stored or learned from.
- Importance is capped (`safety.maximum_importance`).
- A snapshot is taken before every live update and every sleep, and any
  snapshot can be rolled back to.
- A frozen conversation learns nothing at all.

Against **representation collapse**: the rebuild loss, the slow target encoder
and the spread penalty, all described above. `evaluate` reports the latent
spread so collapse would be visible.

## What a brain looks like on disk

```
<brain>/
  model.json                  architecture, fixed at creation
  settings.json               adjustable behaviour (docs/settings.md)
  state.json                  counters and the surprise baseline
  history.jsonl               training, live-update and sleep records
  memory.redb                 episodic memory store
  weights/model.bpk           slow weights (encoder, predictor, decoder)
  weights/fast_weights.bpk    current fast weights
  training/                   optimiser state and target encoder, for resuming
  consolidation/              elastic anchor and replay reservoir, for sleep
  snapshots/<id>/             saved weights for rollback
```

A memory entry holds the prompt and reply text, a timestamp, an importance
score, a recall count, and a meaning vector stored as half-precision floats
(512 dimensions is about 1 KB). The store is an embedded database file, so it
is limited by disk rather than RAM. Search is an exact scan over all vectors.

Weights are stored in Burn's `burnpack` format. JSON documents are written to a
temporary file and renamed, so a crash never leaves a half-written file.

## Answers to the spec's open questions

| Question | What this implementation does |
| --- | --- |
| What counts as a learning signal? | All three, each with its own switch: corrections (`/correct` or a correction marker), prediction error above a threshold, and explicit praise. See `learning.*`. |
| How often to sleep? | Whichever comes first: every N turns, when the store reaches a size, when idle, or after N hours awake. See `schedule.*`. It can also be run by hand. |
| How many thinking steps? | Trained with two; adjustable at any time with `generation.thinking_step_count` because the thinking block is shared between steps. |
| Should memories be rewritten when retrieved? | No. Retrieval only raises a memory's recall count, which protects it from pruning. A correction replaces the memory of the exchange it corrects. |

## Code map

```
crates/core/src/
  configuration/   architecture, settings, training options, presets
  text/            patcher, corpus sampling, replay reservoir, sample text
  model/           transformer block, encoder, predictor, decoder, losses,
                   inference, reply writing, byte sampling, device selection
  memory/          entries, on-disk store, similarity, surprise, retention
  learning/        pretraining, target encoder, live adapter, lessons,
                   elastic anchor, consolidation, progress reports
  brain/           the Brain type: create/open, chat, sleep, snapshots,
                   evaluation, history, sleep schedule, summary
  presentation.rs  wording shared by both applications
crates/cli/src/    the command-line application
crates/gui/src/    the desktop application (worker thread + one file per tab)
```

`Brain` (`brain/mod.rs`) is the only type the applications use. Each command
and each button maps onto one of its methods, which is why the two
applications cannot drift apart in what they can do.

## Performance notes

Measured on four CPU cores with the `tiny` preset, three things mattered most:

- Linear layers are applied to a flattened `[batch × sequence, width]` matrix.
  Letting the layer broadcast its weight over the batch made the backward pass
  several times slower.
- The byte encoder and decoder use ReLU in their feed-forward layers, because
  every byte passes through them and GELU was the single most expensive
  operation on a CPU. The predictor keeps GELU.
- The applications use the mimalloc allocator (the `fast-allocator` feature,
  on by default). Tensor code allocates and frees large buffers constantly,
  and this alone roughly doubled training speed.

## What is and is not verified

Verified by the test suite or by hand while building this:

- Causality: a later patch cannot influence an earlier prediction, and the
  decoder cannot see a byte before writing it.
- Analytic parameter counts match the built model.
- A `tiny` model trained for 1,500 steps on the sample text reaches 0.28 bits
  per byte on held-out text, rebuilds text exactly, and writes grammatical
  sentences from the sample grammar.
- A correction changes the next answer; after sleep the corrected answer
  remains with the fast weights reset and memory retrieval switched off, while
  answers to other questions are unchanged.
- Raising `sleep.elastic_strength` reduces the rise in old-data loss during sleep.

Not verified:

- Anything larger than the `tiny` preset has not been trained. The larger
  presets are sized to match the spec sheet's tiers but their learning rates
  and schedules are untuned.
- The `gpu` build compiles, but no graphics card was available to run it.
  `cuda` was not built.
- Only Linux was tested. The dependencies are cross-platform, but Windows and
  macOS builds were not tried.
- Real English beyond the generated sample text. The sample grammar is far
  easier than natural text.
