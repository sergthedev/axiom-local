# EXP-010 smoke run (20 optimizer updates; executed 2026-11-26; pending technical acceptance)

- Status: **completed** (sequence exit 0). This is a code-path, resource and integrity check of the training harness on
  the real checkpoint and the corpus's train split. It is **not an accuracy result** and makes no claim about AX-01
  quality. Pending the lead's technical acceptance; no pilot has been started.
- Authorization: the owner authorized exactly one initial smoke in the project thread (2026-11-26 13:07 UTC,
  confirmed 13:26 UTC) under the lead's
  [conditional clearance](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7192863338). The final head,
  CI results and authorization were recorded on the PR before the start
  ([authorization record](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7193361060)).
- Head: `680574bf34a68d58eff0b26189d7f5fba24f385b` (clean tree; checked by both processes). CI run 45826169190 on that
  head: Windows 813 passed, 22 skipped; macOS 835 passed.
- Machine: MacBook Air (Mac17,3), Apple M5, 16 GB, macOS 27.0.1, mains power. mlx 0.32.4, mlx-lm 0.32.2.
- Corpus status: `axbench/0.1-ai-reviewed`, AI-reviewed and exploratory, with 0 human decisions.
- Records: [runs/smoke/](runs/smoke/). Training: `train/measurements.json`, `train/training_log.jsonl`. Reload:
  `train/reload/`. Supervisor: `sequence_log.txt`, `baseline.json`, `monitor.jsonl`, `exit_status.txt` and console
  captures. The adapter files are outside the repository; their hashes are below.

## Execution

```text
uv run --locked python exp010_launch.py start --phase smoke --model-dir <checkpoint> --data-dir <export> \
    --adapter-dir <new dir outside the repository> \
    --out docs/evidence/experiments/EXP-010-lora-pilot/runs/smoke --expected-commit 680574bf…
```

| Time (UTC) | Event (`sequence_log.txt`) |
|---|---|
| 13:26:53 | sequence started; common baseline sampled (swap 0 bytes used, pressure 1, AC power) |
| 13:26:55 | monitor ready; training step started |
| 13:27:33 | training exited 0 (status `completed`) |
| 13:27:33 | reload step started (fresh process) |
| 13:27:57 | reload exited 0 (status `completed`) |
| 13:27:58 | monitor exited 0 at shutdown; sequence exit 0 |

Durations: 65 s for the whole sequence. The training process took 35.0 s: checkpoint re-hash 1.5 s, load 0.66 s, and
the 80-micro-step loop 28.7 s, including its two evaluations and the save. The reload process took about 24 s. There
was one run and no retry.

## Configuration as executed

The configuration is PRERUN D7, unchanged:
- base: Qwen3-1.7B @ `70d244cc…`, BF16, re-hashed;
- LoRA: rank 8, scale 20, dropout 0, q/v projections of blocks 20-27, 458,752 trainable parameters;
- optimizer: Adam, learning rate 1e-5;
- batching: micro-batch 1, gradient accumulation 4, seed 2026;
- `max_seq_length` 1024, gradient checkpointing off, no declared change;
- export hashes `437c2ef0…` (train) and `40f83a3c…` (valid), as pinned.

The smoke's loss check uses the first 16 train examples. Validation was read only for the mandatory length, hash and
mask checks.

## Results

**Accounting and integrity (training process).**
- 20 update records at micro-steps 4, 8, …, 80.
- `optimizer.step` = 20.
- Every planned record present, in order: loss at update 0, 20 updates, save 20, loss at update 20.
- Losses finite.
- Completion checks passed with no problems:
  - exactly `adapter_config.json`, `adapters.safetensors` and `update_0020.safetensors`;
  - the save record's hash equals the file;
  - the final adapter equals the update-20 save byte for byte;
  - all 32 tensors have the planned names and shapes and are finite;
  - all 16 `lora_b` tensors are non-zero (per-tensor max |B| 1.14 × 10⁻³ to 1.21 × 10⁻³).
- Final resource sample within the rules.
- No finalization errors.

| Adapter file (outside the repository) | Bytes | SHA-256 |
|---|---|---|
| `adapters.safetensors` | 1,838,449 | `9a403124ecaba86ccaf444cece32c959f0c03eeb248bdf3715d19602c4a5dbc3` |
| `update_0020.safetensors` | 1,838,449 | `9a403124ecaba86ccaf444cece32c959f0c03eeb248bdf3715d19602c4a5dbc3` |
| `adapter_config.json` | 2,650 | `f0fbc7c2fff8599f0455077f16f0fdcbccca59dd3b405c98b31a27679936e8f1` |

**Losses.** Train loss per update is the mean of 4 per-example token-mean losses (E9):

| Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train loss | 2.534 | 2.419 | 2.022 | 1.643 | 1.664 | 1.448 | 1.172 | 1.098 | 0.650 | 1.046 |

| Update | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train loss | 0.683 | 0.973 | 0.478 | 0.670 | 0.627 | 0.626 | 0.326 | 0.655 | 0.572 | 0.635 |

Loss on the first 16 train examples (token-weighted, completion tokens only):
- **1.850** at update 0 (before training; `B = 0`, so this equals the base);
- **0.390** at update 20.

These examples are part of the training data, and some of them were in the 80 trained micro-steps. The drop is a
sign that the loss and update path work; it is not evidence of generalization.

**Memory** (each measured separately, never summed):

| Measure | Training process | Reload process |
|---|---|---|
| MLX peak (`mx.get_peak_memory`) | 4,642,057,024 bytes (after loading: 3,441,149,960) | 3,854,470,944 bytes, covering load and the 5 requests |
| Process peak RSS (`getrusage`) | 2,670,477,312 bytes | 3,355,967,488 bytes |
| Swap growth from the common baseline | 0 bytes (swap stayed 0) | 0 bytes |
| Memory-pressure level, maximum | 1 (normal) | 1 (normal) |

The external monitor took 3 samples plus a final record:
- the training process's RSS (`ps`) was 1.93 × 10⁹ bytes at 13:27:25;
- the reload process's RSS was 2.55 × 10⁹ bytes at 13:27:55;
- the free-memory percentage was at least 46 %;
- no stop.

As in EXP-009, `getrusage`/`ps` RSS does not capture all Metal memory, so MLX peak and RSS are reported side by side.
Both stay below the 10 × 10⁹-byte target, which is reported only and is not a stop rule.

**Throughput.** `update_seconds` is trainer compute time and excludes the callback (D8, C-057).
- Update 1 took 5.29 s, including graph compilation.
- Updates 2-20 took 0.99-1.09 s each.
- Window throughput: first window 0.680 updates/s (includes compilation), second window 0.952 updates/s.
- 97-123 completion tokens/s after update 1; 2,275 trained completion tokens in total.
- The slowdown rule could not fire in the smoke (2 windows), as pre-registered.

**Reload check.**
- The fresh process re-verified the adapter's provenance from the training records, including:
  - phase `smoke`, schedule and configuration;
  - commit `680574b`;
  - base revision and file hashes, export and corpus hashes;
  - the training log re-checked against the plan;
  - all file hashes.
- It then loaded the base with the adapter and ran greedy inference (EXP-009 decoding) on the 5 warm-up train
  examples `train-0001` … `train-0005`.
- The record check passed: 5 unique IDs in order, all `status: ok`, all with raw outputs.
- 3 of 5 generations stopped at `<|im_end|>`; 2 reached the 256-token limit with repeated JSON keys.
- The outputs were not scored and are not an accuracy result. The smoke adapter was not run on validation.

## Limitations and observations

- One short run of 20 updates (0.2 epochs) on one machine. It shows that the code paths, records, stop machinery and
  resource use work at this scale. It does not predict pilot accuracy or full-length memory use, though the sequence
  lengths are the same.
- The 16-example loss check is on training data (accepted decision 4), so its drop says nothing about held-out
  performance.
- The console captures (`train_console.txt`) still contain ANSI bold escape codes from MLX-LM's progress display
  despite `NO_COLOR=1`. They are cosmetic; the JSON records are unaffected.
- The macOS sampler ran inside the supervised sequence for the first time here: 3 regular samples and a final record,
  with no failures.
- Not done: no pilot, no validation predictions or scoring, no final-test access, no merge. The next step (the pilot
  from a fresh untrained base, after the analysis implementation is frozen and reviewed, D17/D18) needs technical
  acceptance of this smoke and separate approval.
