# EXP-010 pilot training run (200 optimizer updates; executed 2026-11-27; pending technical acceptance)

- Status: **completed** (sequence exit 0). This record covers **training only**. No reload, evaluation, validation
  generation or scoring was run, and nothing here is an accuracy or deployment result. The pre-registered pilot outcome
  (D13) and the deployment gates (D14) can only be read after a separately reviewed and approved evaluation and the
  frozen analysis (PRERUN section 8, steps 3-5).
- Authorization: the owner authorized exactly one initial pilot training run in the project thread (2026-11-27
  13:07 UTC), under the lead's
  [final technical clearance](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7194861659) of head
  `9902a02`. The head, CI results, pre-start checks and authorization were recorded on the PR before the start
  ([authorization record](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7219123679)).
- Head: `9902a022f50ccff65861d0f23d713e765ed9b251` (clean tree; checked by the launcher and recorded by the training
  process as `dirty: false`). CI run 45838268425 on that head was successful: Windows job 139972806934 and macOS job
  139972807586.
- Pre-start checks (about 13:08 UTC):
  - the ten pinned Qwen3-1.7B BF16 checkpoint files at revision `70d244cc…` were re-hashed and equal the accepted
    EXP-009 provenance record (SHA-256 and sizes);
  - the frozen exports were `437c2ef0…` (train) and `40f83a3c…` (valid), as pinned;
  - the run directory and the new adapter directory did not exist before the start.
- Machine: MacBook Air (Mac17,3), Apple M5, 16 GB, macOS 27.0.1, mains power. mlx 0.32.4, mlx-lm 0.32.2.
- Corpus status: `axbench/0.1-ai-reviewed`, AI-reviewed and exploratory, with 0 human decisions.
- Records: [runs/pilot/](runs/pilot/), committed unchanged.
  - Training: `train/measurements.json` and `train/training_log.jsonl`.
  - Supervisor: `sequence_log.txt`, `baseline.json`, `monitor_ready.json`, `monitor.jsonl`, `sequence_done.txt`,
    `exit_status.txt` and the console captures.
  - The adapter files are outside the repository; their hashes are below.

## Execution

The command is PRERUN section 8, step 1, unchanged (run from `envs/model`):

```text
uv run --locked python exp010_launch.py start --phase pilot --model-dir <checkpoint> --data-dir <export> \
    --adapter-dir <new dir outside the repository> \
    --out ../../docs/evidence/experiments/EXP-010-lora-pilot/runs/pilot \
    --expected-commit 9902a022f50ccff65861d0f23d713e765ed9b251
```

| Time (UTC) | Event (`sequence_log.txt`) |
|---|---|
| 13:09:22 | sequence started (`exp010.supervise/2`, steps `['train']`); common baseline sampled |
| 13:09:24 | monitor ready; training step started |
| 13:17:56 | training exited 0 (status `completed`, no stop reason, no failure) |
| 13:17:57 | monitor exited 0 at shutdown; sequence exit 0 |

There was one run. Nothing was retried, restarted or stopped by an operator.

## Configuration as executed

The configuration is PRERUN D7, unchanged, as recorded in `train/measurements.json`:
- base: Qwen3-1.7B @ `70d244cc…`, BF16, re-hashed in-process (1.45 s);
- LoRA: rank 8, scale 20, dropout 0, q/v projections of blocks 20-27;
  - 458,752 trainable parameters, equal to the pin;
  - a new initialization with seed 2026; the smoke adapter was not used;
- optimizer: Adam, constant learning rate 1e-5;
- batching: micro-batch 1, gradient accumulation 4;
- schedule: 200 optimizer updates, 800 micro-steps, 2.0 epochs over the 400 train examples;
- `max_seq_length` 1024; gradient checkpointing off; no declared change;
- the completion-mask self-check passed on all 560 examples.

## Integrity checks

- In-process completion checks: `passed: true`, no problems, 213 records.
  - The records are 200 updates, 9 validation-loss evaluations and 4 saves.
  - `optimizer_steps` and `updates_recorded` are both 200.
  - The final adapter equals the update-200 save, and every `lora_b` tensor is non-zero.
- The frozen D18 training-side checks were rerun on the committed records in the Core environment
  (`check_training`, `check_sequence`, `check_monitor` of `exp010.analysis/3`, approved commit `9902a02`). Each
  returned no problem, and the training log has no malformed line. No scoring function was run.
- The adapter files outside the repository were re-hashed after the run, and they equal the recorded hashes:

| File | Bytes | SHA-256 |
|---|---|---|
| `adapters.safetensors` (final) | 1,838,449 | `7a58337c877f2b63b858a299cf046bdb48572805321c0b63988afe783688cf3d` |
| `update_0200.safetensors` | 1,838,449 | `7a58337c877f2b63b858a299cf046bdb48572805321c0b63988afe783688cf3d` |
| `update_0150.safetensors` | 1,838,449 | `9f3cd614d7ed4866a3f5bb778276a186dc2f045cb792fa0fafea613dfa52317e` |
| `update_0100.safetensors` | 1,838,449 | `86f9e4fadba3cc89187385e17f504b24ec32b4b394264c161bd57e46c0f137d7` |
| `update_0050.safetensors` | 1,838,449 | `108c37c41f657dadc92ef7e701bac177427dedfecaa598e00dba45ad0c66af28` |
| `adapter_config.json` | 2,761 | `14abd8c5ab2ba8d1c28a51c67ff7d2235a8a4775d255dfcb5126db3bfdf87955` |

Only `adapters.safetensors` will ever be evaluated (PRERUN section 8). The intermediate saves are kept as records and
are not used to select anything.

## Measurements (as recorded)

**Time.**
- The training process took 510.2 s (`wall_seconds`), of which the 800-micro-step loop took 480.4 s
  (`train_loop_seconds`).
- The 200 updates' trainer compute (sum of `update_seconds`, D8) was 230.2 s. The median update was 1.136 s (range
  0.999-1.322 s): 1.076 s over updates 1-50 and 1.252 s over updates 151-200.
- The nine validation-loss evaluations took 273.1 s in total (27.1-33.0 s each).

**Throughput.**
- 10-update windows went from 0.917 updates/s (first window, including compilation) to 0.791 updates/s (last
  window).
- The lowest window was 0.86 of the first, far from the stop rule's 0.5 (three consecutive windows below half).
- No throughput stop was raised. The cause of the slowdown was not measured; a thermal cause on this fanless machine
  is a possibility, not a finding.

**Memory and the stop rules.**
- MLX peak: 4.71 × 10⁹ bytes (load 3.44 × 10⁹). Process peak RSS (`getrusage`): 3.79 × 10⁹ bytes. The external
  monitor's largest sampled resident size of the training process was 3.26 × 10⁹ bytes.
- Swap: 2,558,525 bytes used at the baseline and at every sample, so there was no growth.
- Memory pressure stayed at level 1 (normal) throughout; the lowest free-memory sample was 42%.
- The machine was on AC power at every sample. The monitor took 19 samples (30 s interval) and raised no stop.

**Loss.** These are teacher-forced losses only. They are reported descriptively and are not a pre-registered endpoint
or evidence of routing accuracy.
- Validation loss (160 examples, 4,614 completion tokens), at updates 0, 25, 50, 75, 100, 125, 150, 175 and 200:
  2.158, 0.426, 0.247, 0.188, 0.170, 0.160, 0.156, 0.155, 0.148.
- Mean training loss per 25 updates (updates 1-25 through 176-200): 0.976, 0.297, 0.201, 0.167, 0.146, 0.127, 0.127,
  0.136.
- The validation loss is a pre-registered record of the training process (PRERUN D5). It is computed by teacher
  forcing on the reference completions and produces no generations or predictions. It is never used to choose a
  checkpoint, stop early or change anything.

## Limitations

- **One run on one machine.** The training data are AI-reviewed and exploratory (0 human decisions). A lower teacher-forced
  loss does not establish exact-match routing accuracy, schema validity, safety or deployment quality; those are
  measured only by the separately approved evaluation and the frozen analysis.
- **Throughput is descriptive only.** The slower later windows are reported as measured; their cause was not
  investigated.

## Next step

Technical acceptance of these training records. The evaluation (PRERUN section 8, step 3) needs a separate technical
review and the owner's approval. Until then, no reload or evaluation is run, no validation generation is produced, no
score is computed, and the sealed test is not accessed.
