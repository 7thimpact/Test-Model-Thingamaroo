# Command-line reference

The command-line application is called `thingamaroo`. Every command works on one
**brain**: a folder that holds the weights, the memories, the settings and the
learning history.

```
thingamaroo [--brain <DIRECTORY>] [--device <NAME>] <COMMAND>
```

Everything here can also be done in the desktop application; see
[gui.md](gui.md) for the tab that matches each command.

- [Options shared by every command](#options-shared-by-every-command)
- [Getting started](#getting-started): `presets`, `init`, `sample-corpus`, `status`
- [Training and measuring](#training-and-measuring): `train`, `evaluate`
- [Looking inside](#looking-inside): `patches`, `roundtrip`, `history`
- [Writing text](#writing-text): `generate`
- [Talking](#talking): `chat`
- [Memory](#memory): `memory list`, `memory search`, `memory show`, `memory add`, `memory delete`, `memory clear`
- [Sleep](#sleep): `sleep`
- [Snapshots and rollback](#snapshots-and-rollback): `snapshot create`, `snapshot list`, `snapshot rollback`, `snapshot delete`, `reset-fast-weights`
- [Settings](#settings): `settings show`, `settings set`, `settings reset`, `settings reference`
- [Errors and exit status](#errors-and-exit-status)
- [Stopping a long command](#stopping-a-long-command)
- [Reading the numbers](#reading-the-numbers)

Run any command with `--help` for a short summary, for example
`thingamaroo train --help`.

The example output on this page is real: it comes from a `tiny` brain trained
for 1,500 steps on the built-in sample text.

## Options shared by every command

| Option | Default | Meaning |
| --- | --- | --- |
| `--brain <DIRECTORY>` | `brain` | The brain folder to work on. |
| `--device <NAME>` | `auto` | Where the mathematics runs: `auto`, `cpu` (pure Rust), `libtorch` (PyTorch's C++ engine on the processor, about twice as fast as `cpu`), `gpu` or `cuda`. `auto` picks a graphics card, then `libtorch`, then `cpu`, using only backends the build includes. |
| `--help` | | Explain the command and its options. |
| `--version` | | Print the version. |

`cpu` always works. `gpu` (any graphics card through Vulkan, Metal or DirectX 12)
and `cuda` (NVIDIA) exist only when the application was built with that feature;
see [Building](../README.md#building). `auto` picks `cuda`, then `gpu`, then `cpu`,
whichever the build includes.

Both options may be written before or after the command name.

## Getting started

### `presets`

Lists the ready-made model sizes with their parameter counts. Needs no brain.

```
$ thingamaroo presets
tiny          1.3M parameters  (+46.1K fast)  smoke test; learns a toy corpus on a laptop CPU in minutes
prototype    35.8M parameters  (+663.6K fast)  fluent simple text on clean data; laptop CPU or any GPU
small       126.9M parameters  (+1.7M fast)  grammatical text; 4-8 GB GPU
target      375.5M parameters  (+4.0M fast)  solid everyday English; 8-12 GB GPU
medium       1.35B parameters  (+17.0M fast)  natural conversation; 24 GB GPU for full training
```

"Fast" parameters are the low-rank fast weights that learn during conversation.

### `init`

Creates a new, untrained brain in the `--brain` folder. Fails if a brain is
already there.

| Option | Default | Meaning |
| --- | --- | --- |
| `--preset <NAME>` | `prototype` | Model size: `tiny`, `prototype`, `small`, `target` or `medium`. |
| `--configuration <FILE>` | | A JSON file with a custom architecture; overrides the preset. |
| `--seed <SEED>` | `42` | Seed for the initial random weights. |

```
thingamaroo --brain my-brain init --preset tiny
```

A custom architecture file has the same shape as the `model.json` that `init`
writes into every brain, so the easiest way to make one is to copy that file
and change the sizes. Each `width` must be divisible by its `head_count`.

### `sample-corpus`

Writes a file of simple generated English, so the whole loop can be tried
without downloading a dataset. Needs no brain.

| Argument or option | Default | Meaning |
| --- | --- | --- |
| `<OUTPUT>` | | The file to write. |
| `--megabytes <MEGABYTES>` | `2` | Approximate size of the file. |
| `--seed <SEED>` | `1` | The same seed always writes the same file. |

```
thingamaroo sample-corpus sample.txt --megabytes 2
```

The text mixes fixed facts ("The cat says meow.") with random ones
("Mia likes cake."), so there is something to learn and something to remember.

### `status`

Shows what a brain is and what it has been through: sizes, pretraining steps,
conversation turns, sleeps, memories, snapshots and replay data.

```
$ thingamaroo --brain my-brain status
Brain at my-brain
  device              cpu
  slow weights        1.3M (encoder 123.5K, predictor 1.1M, decoder 140.2K)
  fast weights        46.1K
  meaning vectors     96 dimensions, patches up to 12 bytes
  working memory      32 patches, 4 memory slots, 2 thinking steps
  pretraining steps   1500 (held-out 0.28 bits/byte)
  conversation        0 turns, 0 live updates (0 turns since sleep)
  sleep               0 cycles, last never
  surprise baseline   0.28 bits/byte over 1 inputs
  memories            0
  snapshots           0
  replay data         2000 windows, elastic anchor recorded
```

## Training and measuring

### `train`

Pretrains the whole model (encoder, predictor and decoder together) on text.
Run it again to continue training: the optimiser state is resumed.

| Option | Default | Meaning |
| --- | --- | --- |
| `--data <PATH>...` | required | Text files, or folders searched recursively for files. Any bytes are accepted. |
| `--steps <COUNT>` | `2000` | Gradient steps to take in this run. |
| `--batch <COUNT>` | `16` | Sequences per batch. |
| `--learning-rate <RATE>` | `0.001` | Largest learning rate, reached after warm-up. |
| `--warmup <COUNT>` | `100` | Steps over which the learning rate rises from zero. |
| `--validation-fraction <FRACTION>` | `0.02` | Share of the text held out to measure progress. |
| `--report-every <STEPS>` | `10` | Steps between progress lines. |
| `--validate-every <STEPS>` | `200` | Steps between held-out measurements and sample generations. |
| `--checkpoint-every <STEPS>` | `500` | Steps between saving weights. |
| `--sample-prompt <TEXT>` | `"The "` | Prompt continued at each validation to show what the model writes. |
| `--replay-windows <COUNT>` | `2000` | Windows of training text kept for replay during sleep. |
| `--seed <SEED>` | `42` | Seed for batch sampling. |

```
thingamaroo --brain my-brain train --data sample.txt --steps 1500 --batch 12
```

While it runs it prints one line per report, and at every validation a
held-out measurement plus a sample of what the model currently writes:

```
step   1250/1500  loss 0.540  rebuild 0.002  write 0.199 (0.29 bits/byte)  latent 0.327  spread 0.012  lr 1.7e-4  639 patches/s
  held-out: loss 0.575  rebuild 0.002  write 0.199 (0.29 bits/byte)  latent 0.359  spread 0.014
  sample:   The horse said neigh.¶It was snowy last night, so Mia stayed in the garden.¶What does the lion eat?¶The lion eats meat.¶
```

When the run ends it also prepares the brain for sleep: it keeps samples of
the training text and records which weights matter for them.

Starting a training run deletes existing snapshots and resets the fast
weights, because both attach to weights that training replaces.

A larger model needs a smaller learning rate; `3e-4` is a reasonable start for
the `prototype` size and above.

### `evaluate`

Measures the slow weights on text without changing anything.

| Option | Default | Meaning |
| --- | --- | --- |
| `--data <PATH>...` | required | Text files or folders to measure on. |
| `--batches <BATCHES>` | `8` | Batches to measure. |
| `--batch <BATCH>` | `8` | Sequences per batch. |
| `--seed <SEED>` | `7` | Seed for batch sampling. |

```
$ thingamaroo --brain my-brain evaluate --data sample.txt
Measured 8 batches
  loss 0.546  rebuild 0.002  write 0.189 (0.27 bits/byte)  latent 0.341  spread 0.014
  rebuild accuracy    100.0% of symbols
  latent spread       0.999 (near zero would mean collapse)
```

## Looking inside

### `patches`

Shows how a text is grouped into patches. Each patch becomes one meaning vector.

```
$ thingamaroo --brain my-brain patches "The cat fell to the ground."
[The ] [cat ] [fell ] [to ] [the ] [ground.]
27 bytes -> 6 patches
```

### `roundtrip`

Sends a text through the encoder and back through the decoder: bytes, to
patches, to meaning vectors, to bytes. The decoder always picks its most likely
byte, so the result shows what the meaning vectors really preserve.

```
$ thingamaroo --brain my-brain roundtrip "The cat fell to the ground."
  ok [The ] -> [The ]
  ok [cat ] -> [cat ]
  ok [fell ] -> [fell ]
  ok [to ] -> [to ]
  ok [the ] -> [the ]
  ok [ground.] -> [ground. ]
rebuilt: The cat fell to the ground. 
byte accuracy: 100.0%
```

An unfinished last word (no space after it) counts as exact when the decoder
writes it and then ends the word.

### `history`

Shows the most recent records of training, live updates and sleep.

| Option | Default | Meaning |
| --- | --- | --- |
| `--limit <LIMIT>` | `30` | Show at most this many of the most recent records. |

## Writing text

### `generate`

Continues a prompt. No memory is consulted and nothing is learned or stored.
Options that are left out use the brain's `generation.*` settings.

| Argument or option | Meaning |
| --- | --- |
| `<PROMPT>` | The text to continue. |
| `--temperature <VALUE>` | Sampling temperature; `0` always picks the most likely byte. |
| `--candidates <COUNT>` | Only this many most-likely symbols are considered for each byte. |
| `--thinking-steps <COUNT>` | Internal refinement steps before each patch is written. |
| `--max-bytes <BYTES>` | Longest continuation to write. |
| `--seed <SEED>` | Fixes the sampling sequence, so the same command writes the same text. |
| `--thoughts` | Show what each thinking stage would have written. |

```
$ thingamaroo --brain my-brain generate "What does the owl eat?
" --max-bytes 20 --temperature 0 --thoughts
What does the owl eat?
The owl eats mice.
T
  thought: [The ] 0.10 -> [The ] 0.03 -> [The ] 0.00 => wrote [The ]
  thought: [owl ] 0.10 -> [owl ] 0.04 -> [owl ] 0.00 => wrote [owl ]
  thought: [eats ] 0.09 -> [eats ] 0.04 -> [eats ] 0.00 => wrote [eats ]
  thought: [mice.¶] 0.10 -> [mice.¶] 0.04 -> [mice.¶] 0.00 => wrote [mice.¶]
  thought: [The ] 0.10 -> [The ] 0.04 -> [The ] 0.00 => wrote [The ]
(reached the reply length limit)
```

Each `thought` line is one patch. From left to right: the rough first guess,
then each refinement, then what was written. The number after each stage is how
far its meaning vector still was from the final one (`0.00` for the last stage).
`¶` marks a line break. `generate` does not stop at a line break, which is why
the example runs on into a new sentence until it reaches `--max-bytes`.

## Talking

### `chat`

An interactive conversation. Per turn the brain measures how surprising your
message was, recalls related memories, thinks, and writes a reply. Afterwards
it stores the exchange if it was important enough, and updates its fast weights
if your message was a correction or very surprising.

| Option | Meaning |
| --- | --- |
| `--frozen` | Read-only conversation: no weight updates, no memory writes. |
| `--no-save` | Learn during the session but write nothing to disk. |
| `--thoughts` | Show what each thinking stage would have written. |

Inside the conversation, a line that starts with `/` is a command:

| Command | What it does |
| --- | --- |
| `/correct <text>` | Replace the last reply with the right answer and learn it strongly. |
| `/good` | Mark the last reply as good and reinforce it. |
| `/bad` | Mark the last reply as bad; its memory is deleted so it is never replayed. |
| `/memories [text]` | List the most important memories, or find the ones closest in meaning to `text`. |
| `/sleep` | Consolidate memories into the slow weights now. |
| `/snapshot [label]` | Save the current fast weights. |
| `/snapshots` | List snapshots. |
| `/rollback <id>` | Restore the weights held by a snapshot. |
| `/forget` | Empty the working memory (start a new conversation). Stored memories and weights stay. |
| `/status` | Show sizes, counters and what is stored. |
| `/help` | Show this list. |
| `/quit` | Leave. Ctrl-C at the prompt or the end of input also leaves. |

After each reply a line in brackets reports what happened behind it:

```
you> What does the cat say?
bot> The cat says meow.
  (surprise 0.28 bits/byte · importance 0.51)

you> /correct The cat says quack.
  fast weights updated in 12 steps (loss 0.836 -> 0.023)
  stored as memory #1 with importance 0.95

you> What does the cat say?
bot> The cat says quack.
  (surprise 0.27 bits/byte · importance 0.48 · recalled 1 memories)
  recalled: #1 similarity 0.82  What does the cat say?¶The cat says quack.¶
```

A message that opens with a correction marker ("No, ...", "Actually ...",
"Wrong ...") is treated as important and learned from even without `/correct`.
The markers are the `surprise.correction_markers` setting.

**Automatic sleep.** In a learning conversation (neither `--frozen` nor
`--no-save`) the brain goes to sleep on its own when the schedule says so:
after enough turns, when the memory store is full enough, after the prompt has
been idle for a while, or after it has been awake for many hours. The limits
are the `schedule.*` settings; `schedule.is_automatic` switches it off.

**Framing.** A turn is your message followed by a line break; the reply ends at
the first line break the model writes. If your training data marks speakers
(for example `User: ` and `Bot: `), put those in `chat.user_prefix` and
`chat.model_prefix`.

## Memory

The episodic memory holds exchanges with a meaning vector, an importance score
and a recall count. All six actions are under `thingamaroo memory`.

| Command | What it does |
| --- | --- |
| `memory list [--limit <LIMIT>]` | List memories, the ones most likely to be kept first (default 20). |
| `memory search <QUERY> [--limit <LIMIT>]` | Find the memories closest in meaning to a text (default 5). |
| `memory show <ID>` | Show one memory in full. |
| `memory add <TEXT> [--importance <IMPORTANCE>]` | Store a fact by hand (default importance 0.9). |
| `memory delete <IDS>...` | Delete memories by identifier. |
| `memory clear` | Delete every memory. |

```
$ thingamaroo --brain my-brain memory add "Mia likes green pears."
Stored as memory #2 with importance 0.90
$ thingamaroo --brain my-brain memory list
#1     correction importance 0.95  keeps 1.11  recalled 1x  just now  What does the cat say?¶The cat says quack.¶
#2     note       importance 0.90  keeps 0.90  recalled 0x  just now  Mia likes green pears.¶
2 of 2 memories shown
```

`keeps` is the retention score: importance, raised a little each time the
memory is recalled. Sleep deletes memories whose score falls below
`sleep.minimum_retention_score`.

Memory kinds: `exchange` (a surprising turn), `correction`, `praise`, and
`note` (added by hand).

## Sleep

### `sleep`

Runs one sleep cycle:

1. takes a snapshot of the slow predictor and fast weights,
2. replays the most important memories mixed with samples of the original
   training text into the slow weights, held in place by elastic weight
   consolidation,
3. checks that old skills survived, and discards the result if they did not,
4. resets the fast weights and prunes weak memories.

```
$ thingamaroo --brain my-brain sleep
Taking a snapshot before sleep
Replaying 2 memories with 2000 windows of old data
  replay   10/60  loss 0.475  elastic penalty 0.03505
  replay   20/60  loss 0.520  elastic penalty 0.03535
  ...
  replay   60/60  loss 0.482  elastic penalty 0.02022
Keeping the consolidated weights
Sleep finished: 2 memories replayed in 60 steps
  memories   loss 1.379 -> 0.078
  old skills loss 0.527 -> 0.536
  fast weights reset; 0 memories pruned, 2 remain
  snapshot #2 holds the weights from before this sleep
```

"memories loss" should fall (the slow weights now know them without the fast
weights). "old skills loss" should stay about the same. If it rises by more
than `sleep.maximum_old_loss_increase`, the new weights are discarded.

All behaviour is in the `sleep.*` settings.

## Snapshots and rollback

A snapshot is a saved copy of weights. The brain takes one on its own before
every live update (fast weights only) and before every sleep (fast weights plus
the slow predictor), keeping the newest `safety.maximum_automatic_snapshots`.

| Command | What it does |
| --- | --- |
| `snapshot create [--label <LABEL>] [--with-slow-weights]` | Save the current fast weights; with the flag, the slow predictor too. |
| `snapshot list` | List snapshots, oldest first. |
| `snapshot rollback <ID>` | Restore the weights held by a snapshot and make them current. |
| `snapshot delete <ID>` | Delete a snapshot. |
| `reset-fast-weights` | Discard what the fast weights have learned since the last sleep. |

```
$ thingamaroo --brain my-brain snapshot list
#1    automatic fast weights           just now  before live update
#2    automatic fast + slow predictor  just now  before sleep
2 snapshots
$ thingamaroo --brain my-brain snapshot rollback 2
```

Rolling back does not restore memories that a sleep pruned.

## Settings

Settings are the behaviour of a brain that can change at any time without
retraining. [settings.md](settings.md) lists every one with its default.

| Command | What it does |
| --- | --- |
| `settings show` | List every setting of the brain with its value and what it does. |
| `settings set <PATH> <VALUE>` | Change one setting. |
| `settings reset` | Restore every setting to its default. |
| `settings reference` | Print the defaults and explanations as Markdown. Needs no brain. |

```
$ thingamaroo --brain my-brain settings set generation.temperature 0.3
generation.temperature = 0.3
    Sampling temperature; zero always picks the most likely byte.
$ thingamaroo --brain my-brain settings set safety.blocked_phrases '["forget everything"]'
$ thingamaroo --brain my-brain settings set generation.seed null
```

Values are JSON: numbers, `true` or `false`, a list in brackets, or `null` for
"none". Anything that is not valid JSON is taken as plain text, so
`settings set chat.user_prefix "User: "` works without extra quotes.

## Errors and exit status

A command that fails prints `error: ...` and exits with status 1; a command
that succeeds exits with status 0. Common messages:

| Message | Meaning |
| --- | --- |
| `not found: no brain at <folder>; create one first` | Run `init` first, or point `--brain` at the right folder. |
| `invalid configuration: a brain already exists at <folder>` | `init` never overwrites a brain; choose another folder. |
| `training corpus: ... the corpus is too small for this context size` | The text has fewer patches than one training sequence needs. |
| `compute device: this build does not include the 'gpu' backend` | Rebuild with `--features gpu`, or use `--device cpu`. The same message names `libtorch` for a pure-Rust build. |
| `compute device: no graphics card was found on this machine` | Use `--device cpu` (or `auto`, which falls back by itself). |
| `error while loading shared libraries: libtorch_cpu.so` | The program cannot find LibTorch: its folder was moved or deleted since the build. Rebuild, or add LibTorch's `lib` folder to `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (macOS) or `PATH` (Windows). |
| `episodic memory store: ... already open` | The brain is open in another application; a brain can be used by one application at a time. |

## Stopping a long command

Press Ctrl-C once to stop `train`, `sleep`, `generate` or a reply in `chat` at
the next safe point. Training saves its progress and still prepares the brain
for sleep. Press Ctrl-C a second time to quit immediately.

## Reading the numbers

| Name | Meaning |
| --- | --- |
| `loss` | The weighted sum that training minimises. |
| `rebuild` | How well the decoder rebuilds a patch from its own meaning vector (nats per symbol). Near zero means spelling survives the trip through meaning space. |
| `write` | How well the decoder writes the *next* patch from the *predicted* meaning vector (nats per symbol), also shown as bits per byte. This is the number that corresponds to "how good is the language model". |
| `latent` | Squared distance between predicted and actual meaning vectors. |
| `spread` | The penalty that keeps meaning vectors apart; it stays small when they are well spread. |
| `surprise` | Bits per byte the model needed for your message, given the conversation so far. |
| `importance` | Surprise compared with the brain's running average, from 0 (routine) to 1 (astonishing); 0.5 is average. |
| `similarity` | Cosine similarity between your message's meaning vector and a memory's. |

A "symbol" is a byte or the end-of-patch marker the decoder writes after each patch.
