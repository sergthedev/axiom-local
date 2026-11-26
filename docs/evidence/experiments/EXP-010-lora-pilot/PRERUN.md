# EXP-010 pre-run record (revision 1; proposed; nothing has been trained)

- Status: **revision 1, proposed for technical re-review** after the lead's
  [pre-run review](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7191912956) of draft head `ea1d343`
  (decisions 1-11 accepted with conditions; blocking corrections A-D). Section 0 lists what changed. No training is
  authorized by this record; the smoke needs the lead's acceptance of this revision **and** the owner's approval (D17).
- Revision 0 (`ea1d343`): this record turns the EXP-010 protocol (revision 3, [PROTOCOL.md](PROTOCOL.md))
  into exact operational definitions before any training, adapter output or adapter evaluation exists. Nothing
  here changes a protocol setting; where the installed software makes the protocol's wording ambiguous, the choice
  is stated and listed under [unresolved decisions](#7-unresolved-decisions-for-technical-acceptance).
- Issue: #4 (Phase 3C). Base: accepted main `fbd4adc3bab3d50e6912935a6bf982fbec62f10b` (PR #10 merged; Qwen3-1.7B
  accepted as the exploratory pilot base in the
  [final technical acceptance](https://github.com/sergthedev/axiom-local/pull/10#issuecomment-7191106643)).
- Date: 2026-11-26. Machine: MacBook Air (Mac17,3), Apple M5, 16 GB, macOS 27.0.1, arm64.
- **Corpus status:** `axbench/0.1-ai-reviewed` is synthetic, AI-generated and reviewed only by an owner-authorized
  AI review (0 human decisions). Every result EXP-010 produces on it is exploratory and is never described as
  human-validated.
- Not done here: no LoRA training of Qwen3-1.7B, no adapter, no adapter inference, no validation prediction of any
  new system, no final-test access, no hyperparameter search, no merge.

## 0. Revision 1 (lead pre-run review, 2026-11-26)

The review accepted decisions 1-11 (section 7, now recorded as accepted with their conditions) and required four
implementation corrections before the smoke. None of them changes a training setting, the frozen data, the base
checkpoint or a pre-registered endpoint; the training configuration (D7), schedules (D4-D6), thresholds (D8) and
criteria (D13, D14) are unchanged.

| Correction | What changed | Where | Evidence |
|---|---|---|---|
| A. Launcher safety | One common pre-load baseline and preflight before anything starts; the monitor must write a valid first sample before the first step; both processes supervised; a malformed or missing sample, a sampler error, a monitor crash or a stale heartbeat fails closed; a stop marker always overrides success; SIGTERM, 120 s grace, then kill; `exit_status.txt` written last even if the launcher itself raises; the smoke reload is a supervised step with the same monitor | `src/axbench/supervise.py`, `envs/model/exp010_launch.py` (now a thin wrapper); D8, D10 | `tests/test_exp010_supervise.py`: 27 tests with dummy processes and scripted samples |
| B. Smoke reload | Adapter provenance re-derived from the training records (phase, schedule, configuration, trainable count, same reviewed commit, base revision and file hashes, pinned export and corpus hashes, the training log re-checked, every adapter file hash, `adapter_config.json`), never from a status flag; pinned corpus manifest; exactly the 5 unique warm-up IDs, each `status: ok` with a raw output, re-read from the persisted file; stop rules from the common baseline after load and every request; allocation errors stop; SIGTERM recorded; non-zero exit on any defect; never validation | `exp010_train.py reload`, `verify_adapter`; `trainwatch.check_request_records`; D11 | fake-router regressions (Core tests and self-test), 10 defective run-directory fixtures rejected (self-test) |
| C. Training completion | `completed` only after: every planned update, evaluation and save record exactly once and in order; finite losses; optimizer step count read from `optimizer.step` equals the planned updates; exactly the required adapter files; save-record hashes equal the files; final adapter equals the last save; every saved tensor has the planned name and shape, is finite, and every `lora_b` is non-zero; a final resource sample passes. Finalization errors are recorded and never replace a stop. `adapter_config.json` records base, file hashes, phase, schedule, configuration, data, commit and package versions; a monitor stop record is copied into the training record | `exp010_train.py train`, `verify_training_outputs`; `trainwatch.check_training_records`, `check_adapter_tensors`; D6 | 15 negative training-log cases and 6 tensor cases (Core tests); tampered adapter rejected (self-test) |
| D. Execution authorization | Pre-registered conditions for the smoke and the analysis integrity checks; gate code fails closed on shrunken denominators and missing pairs, treats missing or non-finite resources as not assessable, and reports unauthorized dispatches in every outcome; the evaluation MLX peak covers load, warm-up and validation | D13, D14, D16, D17, D18; `pilotgates.py` (`exp010.pilot-gates/2`) | Core tests |

Self-test of revision 1: [prerun/harness_selftest_rev1.json](prerun/harness_selftest_rev1.json), 15 of 15 checks
passed (section 6). The revision-0 self-test record is kept unchanged.

**Revision 1 follow-up** (lead's [re-review](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7192557224)
of `c4387a1`, which accepted the five revision-1 choices of section 7 and found two defects):
- *Windows CI hash failures.* Windows checked out the EXP-009 comparator files with CRLF, so their SHA-256 differed
  from D16 (the lead verified the CRLF hashes). `.gitattributes` now marks byte-hashed experiment evidence
  (`docs/evidence/experiments/**/*.{jsonl,json,sha256,txt,log,sh}`) `-text`, as already done for the corpus. No
  evidence byte, pinned hash or assertion changed; every Git blob ID under `docs/evidence/experiments` is unchanged.
  A Core test checks the attribute on the comparator files.
- *Supervisor race.* The monitor's liveness and heartbeat were checked only while a step ran, and a clean monitor exit
  (code 0, or 2 without a stop marker) was accepted at the end. Now (`exp010.supervise/2`) the monitor's health is
  checked immediately before every step and after every step exits, before the done marker; any monitor exit before
  the done marker is a failure (exit 7), whatever its code; after the done marker the monitor is given one interval
  plus 5 s to end by itself, then SIGTERM; it must end with 0 or the sequence's SIGTERM, and code 2 is accepted only
  with `monitor_stop.json`. Deterministic dummy regressions with a scripted stand-in monitor and a test hook that
  waits for its exit: a monitor ending (0 or 2) as the step finishes, a monitor gone at the step transition (the next
  step does not start), a monitor not live before the first step, code 2 without a stop marker at shutdown, and the
  expected shutdown of a healthy monitor. Thresholds and the 30 s sampling are unchanged.

## 1. Verified state (before any change)

| Item | Observed |
|---|---|
| Repository | local `main` fast-forwarded to `fbd4adc`; branch `feat/issue-4-exp010-lora-prerun`; clean tree; Core tests 698 passed before this work |
| Model environment | `envs/model`, `uv sync --locked`: mlx 0.32.4, mlx-metal 0.32.4, mlx-lm 0.32.2, transformers 5.22.2, tokenizers 0.23.3, numpy 2.5.5, Python 3.12.16; `Device(gpu, 0)`, Apple M5, `memory_size` 17,179,869,184 bytes, `max_recommended_working_set_size` 12,713,115,648 bytes |
| Checkpoint | Qwen/Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; all 10 provisioned files re-hashed and equal to the pinned and recorded hashes ([prerun/checkpoint_verify.json](prerun/checkpoint_verify.json)) |
| Frozen corpus | `axbench/0.1-ai-reviewed`, `MANIFEST.json` SHA-256 `b8b6d39f…16239c`; `train.jsonl` `cf56e6da…bd9a` (400), `validation.jsonl` `80457d95…b3b8` (160). The final test split was not generated, read or exported |

## 2. Exported data (D1)

`uv run python -m axbench export-mlx --corpus data/axbench/v0.1-ai-reviewed --out <directory outside the repository>`
writes chat-format `train.jsonl` and `valid.jsonl` (system instruction, user JSON from `axiom.prompt/1`, assistant
target = compact gold JSON). No test file is written.

| File | Examples | Bytes | SHA-256 |
|---|---|---|---|
| `train.jsonl` | 400 | 845,791 | `437c2ef0ce44bdedcb865f7f115ef2b7eda461063e127b5e4cb77da20ec99b31` |
| `valid.jsonl` | 160 | 335,911 | `40f83a3ca85a1802d6877c4d09d6d620dcb416b4ee9169b1417239d1e85d4361` |

- A second export into another directory was byte-identical. `tests/test_exp010_prerun.py` re-exports in CI on macOS
  and Windows and asserts these hashes.
- The harness pins both hashes, refuses any other bytes, and re-derives every line from the frozen corpus (same code
  path as the export) and refuses if any line differs. Lines are paired with corpus IDs by position.

## 3. Installed MLX-LM 0.32.2 behaviour that the protocol depends on (E1-E9)

Read in the installed package source (REF-035: `mlx_lm/tuner/trainer.py`, `tuner/datasets.py`, `tuner/utils.py`,
`tuner/lora.py`, `lora.py`; `mlx/optimizers/optimizers.py`) and, where marked, measured with the pinned tokenizer.

- **E1 Micro-steps vs updates.** `iters` counts micro-steps (one batch each). The optimizer updates when
  `it % grad_accumulation_steps == 0`; gradients of the accumulated micro-steps are summed and divided by the
  accumulation count. With micro-batch 1 and accumulation 4, one update applies the mean of 4 per-example gradients.
  So 20 updates = 80 iters and 200 updates = 800 iters.
- **E2 Built-in evaluation timing.** `steps_per_eval` counts micro-steps. The built-in validation runs at `it == 1`,
  at `it % steps_per_eval == 0` and at `it == iters`, **before** micro-step `it` executes, and is reported as
  iteration `it - 1`. With accumulation 4, `steps_per_eval = 100` would measure the model after 24 updates (not 25),
  and the final built-in validation (`it == 800`) measures the model after 199 of 200 updates.
- **E3 Built-in saves.** `save_every` counts micro-steps; a save at `it % save_every == 0` happens after that
  micro-step (an update boundary when it is a multiple of 4). The final `adapters.safetensors` is written after the
  loop.
- **E4 Loss mask includes one padding target (measured).** `default_loss` uses
  `mask = (steps >= offset) & (steps <= length)` with `steps = 1 .. padded_length - 1`. Position `length` is the
  first padding token (id 0, `!`); batches are always padded to at least `length + 1`. So the installed loss trains
  one padding target per example (on `train-0001`, a 454-token sequence with positions 0-453 and prompt offset 430,
  the installed mask selects positions 430..454; position 454 is padding).
- **E5 Chat dataset boundary (measured).** `ChatDataset` with `--mask-prompt` takes the prompt length from the
  template called without `enable_thinking`, i.e. thinking enabled: the boundary falls before
  `<think>\n\n</think>\n\n`, so those 4 template tokens are trained. The full rendering ends with
  `<|im_end|>` and a newline, so the newline is trained too. Measured on all 560 examples: the installed chat path
  would train exactly **6 extra targets per example** (4 think-block tokens, the newline, the padding target) beyond
  the completion defined in D2. On all 560 examples the full rendering is identical with and without
  `enable_thinking=False` (the template always writes the empty think block for the last assistant turn), and the
  thinking-enabled prompt is the `enable_thinking=False` prompt minus exactly those 4 tokens.
- **E6 Data order and evaluation share one RNG.** `iterate_batches` sorts by length and draws a fresh
  `np.random.permutation` per pass; the built-in evaluation calls `iterate_batches` too and consumes a permutation
  from the same global NumPy state. The training order therefore depends on the evaluation schedule unless the state
  is protected.
- **E7 Silent truncation.** `iterate_batches` truncates sequences longer than `max_seq_length` and only prints a
  warning. Batch width is `min(1 + 32 * ceil(longest / 32), max_seq_length)`, so `max_seq_length` only caps
  truncation; it does not change memory for sequences below it.
- **E8 LoRA and optimizer.** `LoRALinear`: `y = W x + scale * (x A) B` with `A ~ U(-1/sqrt(in), 1/sqrt(in))`, `B = 0`;
  `scale` multiplies the product directly (it is not `alpha / rank`). The `lora.py` default is `scale = 20.0`.
  `linear_to_lora_layers(model, n, {"keys": [...]})` converts only the named sub-modules of the last `n` blocks.
  `mlx.optimizers.Adam` defaults: betas `[0.9, 0.999]`, eps `1e-8`, `bias_correction=False`; the learning rate is
  constant when no schedule is given. `train()` sets the Metal wired limit to `max_recommended_working_set_size`.
- **E9 Reported losses.** The trainer's training loss is the mean over the reported micro-steps of each example's
  token-mean loss; `evaluate()` returns a token-weighted mean over all evaluated examples.

## 4. Operational definitions (D1-D12)

**D1 Data.** Section 2. Training reads only the pinned export of the frozen train and validation splits.

**D2 Completion boundary and loss (explicit rendering, as the owner's preparation prompt prefers).**
- Prompt = `tokenizer.apply_chat_template(system + user, add_generation_prompt=True, enable_thinking=False)`,
  exactly the EXP-009 inference call. Sequence = the full chat rendering with the assistant target
  (`enable_thinking=False`) with the template's final newline removed: prompt, `<think>\n\n</think>\n\n` (part of
  the prompt, as at inference), compact gold JSON, `<|im_end|>`.
- Checked token by token for every example (560 of 560): the sequence starts with the inference prompt, ends with
  `<|im_end|>` (151645) followed by the dropped newline (198), and contains `<|im_end|>` exactly once after the
  prompt. Any failure stops before training.
- Loss = cross-entropy on positions `prompt_length .. length - 1` only (the gold JSON and `<|im_end|>`), no prompt,
  no think block, no newline and no padding target. Implemented as `completion_loss`, the installed `default_loss`
  with one change: the end bound is exclusive (`steps < length`). The harness verifies on all 560 real sequences that
  its mask selects exactly these positions and that the last target is `<|im_end|>`.
- No chain-of-thought targets.

**D3 Sequence length.** Longest full sequence: **529** tokens (validation; train 514). The rule (smallest power of two
at least the longest, stop above 1024) gives **`max_seq_length = 1024`**. No example is truncated (7 sequences exceed
512: 1 train, 6 validation). The value is pinned in the harness; any other result stops. By E7 the cap does not
change batch width or memory.

| Tokens | Train (400) | Validation (160) |
|---|---|---|
| Prompt: min / median / max | 382 / 428 / 480 | 369 / 423 / 500 |
| Trained completion (JSON + `<|im_end|>`): min / median / max / total | 21 / 29 / 48 / 11,504 | 21 / 29.5 / 45 / 4,614 |
| Sequence: min / median / max | 407 / 456.5 / 514 | 396 / 452 / 529 |

Per-example counts (IDs and counts only, no texts): [prerun/token_lengths_per_example.jsonl](prerun/token_lengths_per_example.jsonl);
summary: [prerun/token_lengths.json](prerun/token_lengths.json).

**D4 Update accounting.** Smoke: 20 optimizer updates = 80 micro-steps (0.2 epochs). Pilot: 200 updates = 800
micro-steps = exactly 2 passes over the 400 train examples (each pass is a new permutation, E6). The trainer reports
every 4 micro-steps; the harness maps report `u` to micro-step `4u` and stops on any misalignment or on more or fewer
updates than planned.

**D5 Evaluation at update boundaries.** The built-in evaluation is not used (it runs between updates, E2). The
harness evaluates after the optimizer step of the planned updates with the same `completion_loss`:
- Pilot: validation loss on all 160 examples at updates **0, 25, 50, …, 200** (update 0 = before training; LoRA `B = 0`,
  so it equals the base model's loss).
- Smoke: loss on the first 16 train examples at updates 0 and 20 (a code-path and finiteness check; the smoke never
  touches validation).
- The NumPy RNG state is saved and restored around every evaluation, so the data order depends only on seed 2026, not
  on the evaluation schedule.
- Validation loss is recorded, never used to choose a checkpoint or to change anything.

**D6 Adapter saves and provenance.** Pilot saves at updates 50, 100, 150, 200 (`update_0050.safetensors` …); smoke at
update 20; the trainer's final `adapters.safetensors` is written after the last update and is expected to equal the
last save byte for byte (recorded). Adapters and `adapter_config.json` (LoRA configuration, base revision, commit,
data hashes) go to a new directory outside the repository; the SHA-256 and size of every adapter file go into the
run record in the repository. Only the final `adapters.safetensors` of the pilot is ever evaluated; the evaluation
code refuses any other file and any adapter whose hashes differ from the training record.

*Revision 1 (correction C).* The training process sets `completed` only after `verify_training_outputs` passes on the
persisted files: the record sequence equals the plan (loss at update 0; then each update, its save and its evaluation
in that order), micro-steps = 4 × update, every loss finite, update seconds positive; `optimizer.step` (MLX optimizer
state, incremented once per applied update) equals 20 or 200; the adapter directory holds exactly
`adapter_config.json`, `adapters.safetensors` and the planned `update_NNNN.safetensors`; every save record's hash
equals its file; the final adapter equals the last save; every tensor of every saved file has the planned name and
shape (`trainable_tensors` from the LoRA conversion), is finite, and every `lora_b` tensor is non-zero (`B` starts at
zero); then a final resource sample must pass the stop rules. Any defect stops the run as `plan: completion checks
failed`. `adapter_config.json` records the base repository, revision and file hashes, phase and schedule, LoRA
configuration, maximum length, trainable count, seed, learning rate, gradient checkpointing and declared change, data
provenance, commit and package versions.

**D7 Configuration (unchanged from protocol revision 3).**

| Setting | Value |
|---|---|
| Base | Qwen3-1.7B @ `70d244cc…`, BF16 as published, frozen |
| LoRA | rank 8; `scale` 20.0 (installed default, E8); dropout 0; keys `self_attn.q_proj`, `self_attn.v_proj`; last 8 blocks (20-27 of 28) |
| Trainable parameters | **458,752** of 1,721,033,728 (0.0267 %): 32 tensors, per block q `A` 2048×8 + `B` 8×2048, v `A` 2048×8 + `B` 8×1024 ([prerun/trainable_parameters.json](prerun/trainable_parameters.json), lazy load, no weights evaluated) |
| Optimizer | `mlx.optimizers.Adam(learning_rate=1e-5)`, installed defaults (E8), constant learning rate |
| Batch | micro-batch 1, gradient accumulation 4 |
| Seed | 2026: `np.random.seed` before loading (data order), `mx.random.seed` immediately before LoRA conversion (`A` initialization) |
| Gradient checkpointing | off (D9 C1 only) |
| `max_seq_length` | 1024 (D3) |
| Remote trackers | none (the MLX-LM CLI is not used; `report_to` does not exist in the harness) |

**D8 Stop rules and resource monitoring.** Preflight before every process: pinned commit, clean tree (a git failure
refuses), mains power, memory pressure normal. Then, in the training process after every optimizer update and every
evaluation, against the stop-rule baseline (revision 1: the sequence's common pre-load baseline, `baseline.json`,
sampled once before any process starts; each process also records its own preflight sample):
- non-finite training or evaluation loss;
- swap growth above 2 × 10⁹ bytes since the pre-load baseline; memory-pressure level critical (4); loss of mains power;
- allocation failure at load or during training (`MemoryError` whatever its message, or an allocation message);
- sustained slowdown: windows of 10 updates, throughput = updates / summed training-compute seconds (evaluation and
  saving excluded); baseline = the first window; stop after 3 consecutive later windows below half of it;
- plan errors: export or checkpoint hash mismatch, a sequence above the cap, a boundary failure, a different
  trainable-parameter count or module set, misaligned or missing updates.

A separate monitor process (`python -m axbench.supervise monitor`) samples every 30 s: swap (used and total),
pressure level, `memory_pressure` free percentage, power source and the running step's resident size (`ps`), into
`monitor.jsonl`. On swap growth above 2 × 10⁹ bytes since the common baseline, critical pressure or loss of mains
power it writes `monitor_stop.json` and sends SIGTERM to the running step, which records `stopped` (with the
monitor's stop record). Revision 1: a sample with a missing or malformed field, or a sampler error, is a stop
(`monitor sampling failed`), never skipped. Sampling is discrete: a short peak between two samples can be missed. Recorded per update and never summed: MLX peak and active memory,
process peak RSS (`getrusage`), swap, pressure, power, update seconds, tokens per second. A stopped run is recorded as
stopped with all its records; nothing is retried automatically.

The protocol's 10 GB process peak is a **provisional target, not a stop rule**; MLX peak and process RSS are both
reported against it, separately (accepted decision 7).

*Update timing (accepted decision 6; lead condition "confirm, do not assume").* `update_seconds` = 4 /
`iterations_per_second` of the installed trainer report. Read in `mlx_lm/tuner/trainer.py` 0.32.2: the trainer sets
`tic` at the top of each micro-step, after the batch has been built, and adds `perf_counter() - tic` after the
forward/backward pass, the optimizer update, `mx.eval` and cache clearing; the report callback runs after that sum and
the sum is reset after the callback. So `update_seconds` is training compute time and excludes batch construction, the
progress display and everything the harness does in the callback (evaluation, saves, resource sampling, records).
Confirmed by the self-test: with 1 s of extra evaluation time in the callback, the next update's `update_seconds` stays
below 0.07 s while the separately recorded `callback_interval_seconds` (wall time between two reports) exceeds 1 s.
The first 10-update window includes graph compilation, which lowers the baseline and makes the slowdown rule less
sensitive; this is kept literal and disclosed. The smoke's 20 updates give a baseline window and one later window, so
the three-window slowdown rule **cannot fire in the smoke**.

**D9 Predeclared changes (at most two; no sweep).** **C1:** gradient checkpointing on, allowed only after a run stopped
on a memory rule (allocation failure, critical pressure, swap growth); a new run directory, reason logged. Revision 1:
off by default; the launcher and the trainer refuse `--grad-checkpoint` without `--declared-change C1`, and the
choice is recorded in the run record and `adapter_config.json`; never for the initial smoke (D17). **C2:** one
complete rerun from scratch of a run that ended for an external reason that is not a stop rule (as EXP-009 D19),
only after prospective lead review. No automatic retries. No other setting (learning rate, rank, layers, scale, steps, data) is changed in
EXP-010.

**D10 Detached, supervised execution (revision 1, correction A).** `envs/model/exp010_launch.py start` checks the
arguments (new run directory under `runs/`, new or empty adapter directory outside the repository, C1 rule) and
starts `python -m axbench.supervise sequence` in a new session, under `nohup` and `caffeinate -i`, with stdin closed,
so that it survives the end of the Claude Code session. The sequence:
1. samples the **common pre-load baseline** (`baseline.json`) and applies the preflight (mains power, normal
   pressure); an unreadable sample or a refusal ends it (exit 4; a refusal is written to
   `preflight_refusal.json`) before anything starts;
2. starts the monitor and waits up to 120 s for its first valid sample (`monitor_ready.json`); otherwise it ends
   (exit 5, or 6 if the monitor wrote a stop) **before any step starts**;
3. runs the steps one at a time (training; after a smoke, the reload check), each with `--baseline baseline.json`,
   writing the step's PID for the monitor; immediately before each step, while it runs and after it exits it checks
   for the stop marker, a monitor exit (any exit before the sequence is done is premature, code 0 included) and a
   stale heartbeat (no sample for 3 × 30 s + 5 s); a running step is then stopped (SIGTERM, up to 120 s grace, then
   kill) and no further step starts;
4. ends: writes the done marker, lets the monitor end by itself (up to 35 s, then SIGTERM; it must end with 0 or
   the SIGTERM, and 2 only with a stop marker), applies the exit code (0 success; the step's code if it failed, 3 if it was killed by a
   signal; 6 if `monitor_stop.json` exists, **overriding every other code including success**; 7 for a monitor
   failure; 8 for an exception in the sequence, including SIGTERM to the sequence itself, after which the running
   step is stopped the same way) and writes `exit_status.txt` last, in every case. Nothing is retried.

The run directory (`runs/<phase>/` here) receives `sequence_log.txt`, `sequence_console.txt`, `baseline.json`,
`monitor_ready.json`, `monitor.jsonl`, `monitor_console.txt`, `<step>_console.txt`, `monitor_stop.json` (only on a
stop), `exit_status.txt` and `train/` (`measurements.json`, `training_log.jsonl`; smoke: `train/reload/`). Evidence names end in
`.txt`/`.json`/`.jsonl` so that the repository's `*.log` ignore rule cannot hide them.

**D11 Smoke.** Detached smoke run (D4-D10), then the supervised reload step in a fresh process (revision 1, correction
B): preflight; stop rules against the common baseline before loading, after loading and after every request; pinned
corpus manifest; base re-hashed; `verify_adapter` re-derives the adapter's provenance from the training records
(status, phase `smoke`, schedule, configuration, trainable count, training commit equal to the current reviewed head,
base revision and file hashes, pinned export and corpus hashes, completion checks, the training log re-checked against
the plan, the adapter directory's file set and every hash, `adapter_config.json` phase, commit, revision and schedule);
base loaded with `load_adapters`; real greedy inference (EXP-009 decoding: `axiom.prompt/1`, `enable_thinking=False`,
≤256 new tokens, stop set {151643, 151645}) on exactly the five warm-up train examples through the EXP-009 request
path (every record written before the stop rules are checked; an allocation error stops; any other error is a
record). The records are re-read from `train/reload/reload_train5.jsonl` and must be exactly the five unique warm-up
IDs in order, each `status: ok` with a raw output; anything else (missing, duplicated, error, empty) ends the reload
as `stopped` with a non-zero exit. The outputs are not scored; the smoke adapter is never run on validation. Recorded: micro-steps and updates, loss finiteness, MLX peak, RSS, updates and tokens per second,
adapter hashes, `max |B|` after training. Not an accuracy result.

**D12 Pilot and evaluation.** Detached pilot run; then `exp010_train.py evaluate`: fresh process, EXP-009 frame
without repeats (preflight, re-hash, load base + final adapter, five train warm-ups never scored, the 160 validation
requests in file order), EXP-009 decoding, stop rules and record format (`main_warmup.jsonl`,
`main_validation.jsonl`, `main_measurements.json`). Scoring with `axbench.evaluate/0.2` and `axbench.score/0.2`
(Guard policy 2), in the Core environment. Revision 1: `evaluate` verifies the pilot adapter as in D11 (phase
`pilot`; the training commit must be an ancestor of the evaluating head, because the frozen analysis is committed
after the pilot at the latest); the MLX peak is reset once before loading and never again, so
`mlx_peak_bytes_load_warmup_validation` covers load, warm-up and validation and is the G12 MLX value (lead decision 9);
SIGTERM is recorded as a stop. `evaluate` is not run before the analysis implementation is frozen and reviewed (D18).

## 5. Pre-registered pilot criteria (D13) and deployment-quality gates (D14)

Encoded in `src/axbench/pilotgates.py` (`exp010.pilot-gates/2` in revision 1; `/1` in revision 0) with tests, before
any adapter exists. Accepted as proposed (decision 9); revision 1 adds only the integrity semantics the review
required.

**D13 Pilot outcome**, in order:
1. **Run validity.** Training completed 200 updates without a stop; evaluation completed 160 of 160; integrity checks
   of the analysis pass (commit, hashes, corpus, complete records), and the lineage checks pass: exactly the 160
   validation IDs, unique, each scored for both the adapter and the untuned base and assigned to a lineage cluster,
   pairs formed by ID, never by position. Otherwise `invalid_run`: no result is claimed.
2. **Safety precondition.** Any unauthorized dispatch by the adapter → `safety_stop`, whatever the accuracy. The
   unauthorized-dispatch count is reported in every outcome, including `invalid_run` (`safety_flag`).
3. **Primary endpoint.** Semantic exact match of the final adapter vs the same base untuned (EXP-009 accepted
   attempt-2 raw validation outputs, rescored with the same scorer, not rerun), paired over the same 160 examples:
   `improvement_demonstrated` only if the adapter is higher by **more than 5 percentage points** (at least 9 more of
   160, net) **and** the paired cluster-bootstrap 95 % interval of the difference (24 lineages, 2000 replicates,
   seed 2026) excludes zero; otherwise `improvement_not_demonstrated`. This is EXP-009 rule 3, reused unchanged.
4. Everything else is descriptive (D15).

**D14 Deployment-quality gates** (DR-005 provisional values, exact count rules, full denominators; a gate on a slice
with fewer than 20 cases is "not assessable"). On validation n = 160:

| Gate | Rule | Count needed on validation |
|---|---|---|
| G1 candidate-schema validity | ≥ 99 % | ≥ 159 of 160 |
| G2 semantic exact match | ≥ 85 % | ≥ 136 of 160 |
| G3 improvement over own untuned base | ≥ 5 pp | ≥ +8 of 160 (untuned 12 → ≥ 20) |
| G4 Turkish exact match | ≥ 80 % | ≥ 52 of 64 |
| G5 Turkish vs English | EN − TR ≤ 5 pp | difference of rates |
| G6 REJECT (unsupported) recall | ≥ 95 % | ≥ 29 of 30 |
| G7 false REJECT among gold PLAN | ≤ 5 % | ≤ 4 of 85 |
| G8 CLARIFY recall | ≥ 90 % | ≥ 41 of 45 |
| G9 wrong executable proposals, all cases | < 2 % | ≤ 3 of 160 |
| G10 unauthorized dispatches | = 0 | 0 |
| G11 median warm latency | ≤ 2000 ms | |
| G12 process peak RSS and MLX peak | each ≤ 10 × 10⁹ bytes | |

Revision 1 semantics: a malformed summary (a count above its denominator, a non-integer count, a missing pair) or a
denominator below the full 160 for exact match, schema validity or wrong executable proposals, or Turkish plus English
above 160 (mixed-language cases are in neither) raises an error and nothing is assessed; a rate gate on a slice with
fewer than 20 cases, and G11/G12 without a finite, non-negative measurement, are `passed: null` ("not assessable")
with a note, never a pass; the result reports both `all_assessable_passed` and `every_gate_assessed_and_passed`.

On this corpus the gates are a **descriptive validation reading only**: passing them would not qualify AX-01 for
deployment. Deployment needs a frozen system (checkpoint, adapter, settings) chosen before unsealing, a separately
approved confirmatory protocol on the owner's sealed final test, and a human-validated reading of the results. Failing
them is the expected, reportable outcome of a 200-update pilot; the manual plan path and the rules remain the product
path.

**D15 Descriptive reports (no decisions).** Adapter vs rules (paired exact match, rule-3 label reported only
descriptively); JSON and candidate-schema validity vs untuned; action accuracy and macro F1; clarify and reject recall
and reason accuracy; PLAN operation and slot accuracy; wrong executable proposals (all cases, per accepted PLAN);
Guard statuses; TR/EN/mixed and stratum A/B/C slices; latency (median, IQR, max), generated tokens, MLX peak, RSS;
training-loss and validation-loss curves; trainable parameter count; wall time. No population inference; McNemar only
as an IID illustration.

**D16 Comparators.** Untuned 1.7B: EXP-009 `runs_attempt2_qwen3-1.7b/qwen3-1.7b/main_validation.jsonl`. Rules:
`axiom.rules/0.1` rerun on the frozen validation (EXP-009 `rules_validation.jsonl`; its raw outputs and statuses are
identical to EXP-013's, only `elapsed_ms` differs). Pinned in revision 1 (files as accepted with PR #10; a Core test
asserts the hashes, 160 unique validation IDs and that every record is present):

| Comparator file (under `docs/evidence/experiments/EXP-009-untuned-baselines/`) | Records | SHA-256 |
|---|---|---|
| `runs_attempt2_qwen3-1.7b/qwen3-1.7b/main_validation.jsonl` | 160 | `cace592012340cb45807250e879ed22c68fef83cd78769d252574bc23b85550d` |
| `rules_validation.jsonl` | 160 | `7645eea783caeb1b80ba216d715bd06dae5154aaa425ef88a32906c0cfb11b9a` |

Same prompt, template, decoding, parser, scorer and Guard policy; same base weights and precision (LoRA layers are
applied unfused on top of the unchanged BF16 base).

**D17 Execution authorization (revision 1, correction D; lead decision 10).** Pre-registered conditions:
- The smoke runs only after the lead accepts this revision **and** the owner approves the smoke. Nothing else is
  authorized: no pilot, no new validation predictions or scoring, no final-test access, no merge.
- At an immutable, reviewed head: `--expected-commit` must equal `HEAD` on a clean tree (checked by every process).
- New directories: a new run directory under `runs/` and a new or empty adapter directory outside the repository.
- Pinned inputs: the export hashes of section 2 and the checkpoint hashes of section 1, re-hashed in the process.
- Default configuration: no gradient checkpointing for the initial smoke (C1 needs a prior memory stop).
- After the smoke: its records are reported and reviewed before any pilot. The pilot starts from a fresh, untrained
  base (a new process loading the published checkpoint and a new LoRA conversion with seed 2026); it never resumes
  from, or loads, the smoke adapter.
- A stopped or failed run is reported as such with all its records; any rerun is a new run directory under D9.

**D18 Analysis integrity, prepared before any adapter validation output (revision 1, correction D; decision 9).**
The analysis implementation (scoring, D13 outcome, D14 gates, D15 reports) will be written, frozen and reviewed
before any new validation prediction, preferably before the pilot; no endpoint is added afterwards. It will:
- retain all 160 adapter validation records, including errors and empty outputs, in every denominator, and refuse a
  record set that is not exactly the 160 validation IDs, unique, in file order;
- verify the EXP-009 comparator files against the hashes of D16 and pair by example ID (`pilotgates.lineage_problems`);
- verify the evaluation record's adapter provenance (D12) and the training record's completion checks;
- apply `pilot_outcome` and `deployment_gates` (`exp010.pilot-gates/2`) unchanged, reporting unauthorized dispatches
  independently of every other check and missing resource measurements as not assessable;
- record the scorer, harness and gate versions, the commit and the input hashes with the result.
The gate and lineage logic exists and is tested now; the remaining analysis script is not blocking for the smoke (lead
correction D) and is reviewed separately before the pilot or the evaluation.

## 6. Harness self-test (software check, not a training result)

`exp010_train.py selftest` runs the training code on a tiny randomly initialized Qwen3-architecture model (10 blocks,
hidden size 64) and synthetic token sequences: no checkpoint, no corpus, nothing of AX-01. All checks passed
([prerun/harness_selftest.json](prerun/harness_selftest.json)): reports at micro-steps 4, 8, …, 24 = updates 1-6;
evaluations exactly at the planned updates; saves at the planned updates; final adapter byte-identical to the last
save; `B` non-zero after training; identical per-update training losses under two different evaluation schedules
(data order independent of evaluation); the saved adapter reloaded through `load_adapters` into a fresh copy of the
base reproduces the trained model's logits exactly; the loss mask equals the planned positions on 30 synthetic
sequences.

**Revision 1** ([prerun/harness_selftest_rev1.json](prerun/harness_selftest_rev1.json), 15 of 15 checks passed;
same tiny model and synthetic tokens, plus the frozen corpus's 5 warm-up train records for the scripted-router paths;
no checkpoint, no export, no validation): the eight checks above, and
- `optimizer.step` equals the 6 planned updates in both runs;
- `verify_training_outputs` passes on both runs (records, files, hashes, tensors: every `lora_b` non-zero);
- a copy of the final adapter with one `lora_b` tensor set to zero is rejected;
- with 1 s of extra evaluation time in the update-3 callback, update 4's `update_seconds` is 0.063 s while its
  `callback_interval_seconds` is 1.164 s: the timing excludes callback work (D8);
- the reload request path with scripted routers: a clean run passes; a request error gives a `status: error` record
  that the record check rejects; a `MemoryError` stops the reload; an empty record set is rejected;
- `verify_adapter` accepts a consistent synthetic smoke run directory (placeholder bytes, not adapters) and rejects
  each of 10 single defects: status not completed, wrong phase, a missing evaluation record, a non-finite loss record,
  another training commit, a different data hash, failed completion checks, an adapter modified after training, an
  extra adapter file, another base checkpoint;
- a finalization error keeps a stopped run's status and reason, and turns `completed` into `failed`.

Dummy-process tests of the supervisor (`tests/test_exp010_supervise.py`, 27 tests at revision 1 and 34 after the
follow-up, about 40 s; the process tests are POSIX-only and skipped on Windows): normal completion of two steps; preflight refusal and an unreadable baseline (no
monitor, no step); a monitor that never becomes ready and a malformed first monitor sample (no step); an injected
swap-growth stop during a step (step terminated, later step not started); a malformed sample and a sampler error
(fail closed); a monitor crash and a hung monitor (stale heartbeat), each stopping the step; a step that ignores
SIGTERM (killed after the grace period); a step that exits 0 on SIGTERM (stop marker overrides success); a failed step
(later step not started); a launcher exception (exit 8, evidence written); SIGTERM to the sequence itself (step
stopped, exit 8); and the detached `start` path end to end (returns at once, the sequence runs in its own session and
completes). Ten sample-validation cases run on every platform. The follow-up adds the monitor-exit race regressions
listed in section 0 and a check of the expected shutdown codes.

Not exercised before the smoke: the `train`, `reload` and `evaluate` commands on the real checkpoint, and the macOS
sampler inside the supervised sequence (the scripted sampler stands in for it in the tests; the macOS commands are the
ones EXP-009 used). Peak memory and speed of LoRA training on this Mac are unknown until the smoke.

## 7. Decisions (accepted in the lead's pre-run review, 2026-11-26)

Revision 0 listed these as unresolved. The [review](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7191912956)
accepted them with the conditions recorded here.

1. **Completion boundary (ADR 0012 finding 2): accepted.** The explicit inference-aligned rendering of D2.
2. **Padding target (E4): accepted.** `completion_loss`, the installed loss with an exclusive end bound.
3. **Evaluation timing (E2, D5): accepted.** Harness evaluation after update boundaries, including update 0.
4. **Smoke loss check set: accepted.** The first 16 train examples. The smoke reads validation only for the mandatory
   length, hash and mask checks, never for a validation loss or prediction.
5. **`max_seq_length = 1024`: accepted.**
6. **Throughput baseline: kept literal.** The first 10-update compute window, including compilation; the weakness is
   disclosed (D8). The trainer timing was confirmed in source and by the self-test, not assumed (D8). The smoke cannot
   trigger the three-window slowdown rule.
7. **10 × 10⁹ bytes: kept as a reported target**, process RSS and MLX peak reported separately, not an automatic stop.
8. **Predeclared changes: accepted.** C1 (gradient checkpointing) only after a memory stop; C2 one rerun after an
   external interruption, only after prospective lead review; no automatic retries; no other parameter change;
   gradient checkpointing off by default (D9).
9. **D13 and D14: accepted as proposed** (D13 "more than 5 pp" plus the paired cluster interval; D14 descriptive-only
   exact gates, G3 "at least 5 pp" distinct from D13), with denominators, IDs, lineage pairing, finite resources, full
   records and explicit not-assessable semantics enforced in the analysis, unauthorized dispatch reported
   independently of other integrity failures, and the G12 inference MLX peak covering load, warm-up and validation
   (section 5, D12, D18).
10. **Sequencing: accepted with conditions.** Smoke only after acceptance of the corrected pre-run record and the
    owner's approval; smoke reviewed before the pilot; the pilot from a fresh untrained base, never resumed from the
    smoke; the analysis implementation frozen and reviewed before any new validation prediction, preferably before
    the pilot, with no post-hoc endpoints (D17, D18).
11. **ADR 0011 and ADR 0012 remain Proposed** pending broader acceptance. This record documents experiment-specific
    decisions only and implies no product model approval.

### Choices made in revision 1, accepted in the re-review

The re-review accepted all five: the per-tensor non-zero `lora_b` rule as an operational acceptance gate for the smoke
(not proof that a zero tensor necessarily means a training bug), the common baseline, the supervision constants, the
ancestor-commit provenance for `evaluate` (subject to review when the evaluation is prepared) and not-assessable
missing resources.

- Every `lora_b` tensor of every saved adapter must be non-zero (stricter than "B non-zero overall"). Adam moves every
  parameter with a non-zero gradient on the first update, so an all-zero `lora_b` tensor after 20 updates indicates a
  defect; on the tiny self-test model all 16 `lora_b` tensors are non-zero.
- In-process stop rules of the training and reload processes use the sequence's common baseline, so swap growth during
  training also counts for the reload (stricter than a per-process baseline).
- Supervision constants: 30 s sampling (unchanged), first valid sample within 120 s, stale heartbeat after 95 s,
  120 s SIGTERM grace before a forced kill.
- `evaluate` accepts a pilot adapter trained at an ancestor of the evaluating head (the frozen analysis may be
  committed after the pilot); the smoke reload requires the same head.
- A missing or non-finite G11/G12 measurement is "not assessable" (revision 0 counted it as a failed gate).
