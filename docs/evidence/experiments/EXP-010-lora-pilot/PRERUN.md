# EXP-010 pre-run record (proposed; nothing has been trained)

- Status: **proposed for technical review.** This record turns the EXP-010 protocol (revision 3, [PROTOCOL.md](PROTOCOL.md))
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
evaluation (from one pre-load baseline per process, as in EXP-009):
- non-finite training or evaluation loss;
- swap growth above 2 × 10⁹ bytes since the pre-load baseline; memory-pressure level critical (4); loss of mains power;
- allocation failure at load or during training (`MemoryError` whatever its message, or an allocation message);
- sustained slowdown: windows of 10 updates, throughput = updates / summed training-compute seconds (evaluation and
  saving excluded); baseline = the first window; stop after 3 consecutive later windows below half of it;
- plan errors: export or checkpoint hash mismatch, a sequence above the cap, a boundary failure, a different
  trainable-parameter count or module set, misaligned or missing updates.

A separate monitor process samples every 30 s: swap (used and total), pressure level, `memory_pressure` free
percentage, power source and the training process's resident size (`ps`), into `monitor.jsonl`. On swap growth above
2 × 10⁹ bytes since its first sample, critical pressure or loss of mains power it writes `monitor_stop.json` and sends
SIGTERM; the training process records `stopped`. Recorded per update and never summed: MLX peak and active memory,
process peak RSS (`getrusage`), swap, pressure, power, update seconds, tokens per second. A stopped run is recorded as
stopped with all its records; nothing is retried automatically.

The protocol's 10 GB process peak is a **provisional target, not a stop rule**; MLX peak and process RSS are both
reported against it.

**D9 Predeclared changes (at most two; no sweep).** **C1:** gradient checkpointing on, allowed only after a run stopped
on a memory rule (allocation failure, critical pressure, swap growth); a new run directory, reason logged. **C2:** one
complete rerun from scratch of a run that ended for an external reason that is not a stop rule (as EXP-009 D19),
only with the lead's approval. No other setting (learning rate, rank, layers, scale, steps, data) is changed in
EXP-010.

**D10 Detached execution.** `envs/model/exp010_launch.py start` starts the sequence in a new session, under `nohup`
and `caffeinate -i`, with stdin closed, so that it survives the end of the Claude Code session. The run directory
(`runs/<phase>/` here) receives `sequence_log.txt`, `sequence_console.txt`, `train_console.txt`, `monitor.jsonl`,
`exit_status.txt` (written last) and `train/` (`measurements.json`, `training_log.jsonl`). Evidence names end in
`.txt`/`.json`/`.jsonl` so that the repository's `*.log` ignore rule cannot hide them.

**D11 Smoke.** Detached smoke run (D4-D10), then in a fresh process: adapter and configuration hash-checked against
the training record, base loaded with `load_adapters`, real greedy inference (EXP-009 decoding: `axiom.prompt/1`,
`enable_thinking=False`, ≤256 new tokens, stop set {151643, 151645}) on the five warm-up train examples
(`train/reload/`). Recorded: micro-steps and updates, loss finiteness, MLX peak, RSS, updates and tokens per second,
adapter hashes, `max |B|` after training. Not an accuracy result.

**D12 Pilot and evaluation.** Detached pilot run; then `exp010_train.py evaluate`: fresh process, EXP-009 frame
without repeats (preflight, re-hash, load base + final adapter, five train warm-ups never scored, the 160 validation
requests in file order), EXP-009 decoding, stop rules and record format (`main_warmup.jsonl`,
`main_validation.jsonl`, `main_measurements.json`). Scoring with `axbench.evaluate/0.2` and `axbench.score/0.2`
(Guard policy 2), in the Core environment.

## 5. Pre-registered pilot criteria (D13) and deployment-quality gates (D14)

Encoded in `src/axbench/pilotgates.py` (`exp010.pilot-gates/1`) with tests, before any adapter exists.

**D13 Pilot outcome**, in order:
1. **Run validity.** Training completed 200 updates without a stop; evaluation completed 160 of 160; integrity checks
   of the analysis pass (commit, hashes, corpus, complete records). Otherwise `invalid_run`: no result is claimed.
2. **Safety precondition.** Any unauthorized dispatch by the adapter → `safety_stop`, whatever the accuracy.
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
`axiom.rules/0.1` rerun on the frozen validation (EXP-009 `rules_validation.jsonl`, identical to EXP-013). Same
prompt, template, decoding, parser, scorer and Guard policy; same base weights and precision (LoRA layers are applied
unfused on top of the unchanged BF16 base).

## 6. Harness self-test (software check, not a training result)

`exp010_train.py selftest` runs the training code on a tiny randomly initialized Qwen3-architecture model (10 blocks,
hidden size 64) and synthetic token sequences: no checkpoint, no corpus, nothing of AX-01. All checks passed
([prerun/harness_selftest.json](prerun/harness_selftest.json)): reports at micro-steps 4, 8, …, 24 = updates 1-6;
evaluations exactly at the planned updates; saves at the planned updates; final adapter byte-identical to the last
save; `B` non-zero after training; identical per-update training losses under two different evaluation schedules
(data order independent of evaluation); the saved adapter reloaded through `load_adapters` into a fresh copy of the
base reproduces the trained model's logits exactly; the loss mask equals the planned positions on 30 synthetic
sequences.

Not exercised before the smoke: the `train`, `reload` and `evaluate` commands on the real checkpoint, the detached
launcher end to end, and the monitor's SIGTERM stop path (the monitor's sampling and exit were checked against a
short-lived process). Peak memory and speed of LoRA training on this Mac are unknown until the smoke.

## 7. Unresolved decisions for technical acceptance

1. **Completion boundary (ADR 0012 finding 2):** accept the explicit rendering of D2 (recommended; the owner's prompt
   prefers it) rather than recording the think block as part of the target.
2. **Padding target (E4):** accept `completion_loss`, the installed loss with an exclusive end bound. Without it every
   example trains one padding target.
3. **Evaluation timing (E2, D5):** accept harness evaluation after update boundaries, including update 0, instead of
   the built-in schedule (which would report 24 and 199 updates as 25 and 200).
4. **Smoke loss check set:** the first 16 train examples (proposed) or no evaluation in the smoke.
5. **`max_seq_length = 1024`** by the protocol rule (DR-003's 512 does not fit: 7 sequences exceed it). No
   truncation and, by E7, no memory effect.
6. **Throughput baseline:** the protocol's "first window" includes compilation warm-up, which lowers the baseline and
   makes the rule less sensitive. Kept literal; the alternative is the second window.
7. **10 GB** stays a reported target, not a stop rule; the trainer's Metal wired limit (E8, 12.71 × 10⁹ bytes) is
   installed behaviour and is recorded, not changed.
8. **Predeclared changes C1 and C2 (D9)**, and the protocol's phrase "validation-only changes", which this record reads
   as "changes decided without looking at validation predictions".
9. **Gate values (D13, D14)**, including "more than 5 pp" for the pilot (EXP-009 rule 3) versus "at least 5 pp" for
   deployment gate G3 (DR-005), and the minimum slice size 20.
10. **Sequencing:** proposed smoke → report and review → pilot → evaluation → analysis, with a stop for review after
    the smoke. An analysis script (scoring, paired comparisons, D13/D14 application) will be added before the
    evaluation and reviewed with the pilot records.
11. ADR 0011 and ADR 0012 are still listed as Proposed.
