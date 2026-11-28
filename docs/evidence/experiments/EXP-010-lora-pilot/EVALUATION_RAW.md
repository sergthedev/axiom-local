# EXP-010 validation evaluation: raw records (executed 2026-11-28; not scored)

- Status: **completed** (sequence exit 0). This record covers raw generation records and their integrity only. **No
  score has been computed**: no exact match, schema validity, Guard result, paired comparison, pilot outcome (D13) or
  deployment gate (D14). Those come from the frozen analysis (PRERUN section 8, step 5), which runs only after the lead
  reviews the preservation of these records (step 4).
- Authorization: the lead's
  [technical acceptance](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7219505123) cleared exactly one
  frozen validation-generation sequence at the evidence head `edf49fa`, subject to the owner's authorization. The
  owner's bounded delegation is recorded in the lead's
  [overnight delegated clearance](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7231617648) (new
  launches until 12:20 UTC on 2026-11-28). The owner's instruction to proceed followed in the project thread at
  05:54 UTC. The head, CI, pre-start checks and authorization were recorded on the PR before the start
  ([authorization record](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7234971828)).
- Head: `edf49faaa6a44a2a94ef9f949d4e7e7e3b48f7ed`. The tree was clean, and the evaluation process recorded
  `dirty: false`. CI run 46006213431 on that head was successful on macOS and Windows.
- Pre-start checks:
  - no concurrent model workload was running;
  - the ten checkpoint files were re-hashed and equal the EXP-009 provenance record;
  - the frozen exports equal their pins;
  - the final pilot adapter is `7a58337c…` and its config is `14abd8c5…`, equal to the committed training records;
  - the machine was on AC power with memory pressure level 1;
  - `runs/pilot_eval/` did not exist.
- Machine: MacBook Air (Mac17,3), Apple M5, 16 GB, macOS 27.0.1. mlx 0.32.4, mlx-lm 0.32.2.
- Records: [runs/pilot_eval/](runs/pilot_eval/), committed unchanged.
  - Evaluation: `eval/main_measurements.json`, `eval/main_warmup.jsonl` and `eval/main_validation.jsonl`.
  - Supervisor: `sequence_log.txt`, `baseline.json`, `monitor_ready.json`, `monitor.jsonl`, `sequence_done.txt`,
    `exit_status.txt` and the console captures.

## Execution

The command is PRERUN section 8, step 3, unchanged (run from `envs/model`):

```text
uv run --locked python exp010_launch.py start --phase evaluate --model-dir <checkpoint> \
    --adapter-dir <the pilot adapter dir> \
    --train-run ../../docs/evidence/experiments/EXP-010-lora-pilot/runs/pilot/train \
    --out ../../docs/evidence/experiments/EXP-010-lora-pilot/runs/pilot_eval \
    --expected-commit edf49faaa6a44a2a94ef9f949d4e7e7e3b48f7ed
```

| Time (UTC) | Event (`sequence_log.txt`) |
|---|---|
| 05:55:55 | sequence started (`exp010.supervise/2`, steps `['evaluate']`); common baseline sampled |
| 05:55:56 | monitor ready; evaluation step started |
| 05:56:08 | five warm-ups done (`after_warmup` sample) |
| 06:00:16 | evaluation exited 0 (status `completed`, no stop reason, no failure); monitor exited 0; sequence exit 0 |

There was one launch. Nothing was retried, restarted or stopped by an operator.

## As executed

- Base: Qwen3-1.7B @ `70d244cc…`. All ten files were re-hashed in-process and checked against the pins and the
  provenance record.
- Adapter: `adapters.safetensors` `7a58337c877f2b63b858a299cf046bdb48572805321c0b63988afe783688cf3d` (1,838,449 bytes)
  and config `14abd8c5ab2ba8d1c28a51c67ff7d2235a8a4775d255dfcb5126db3bfdf87955`. They were verified in-process against
  the training records (training commit `9902a02`, 213 training-log records; D12).
- Prompt: `axiom.prompt/1`.
- Decoding: greedy, batch 1, thinking disabled, at most 256 new tokens, stop tokens 151643 and 151645. This is the
  EXP-009 frame, unchanged.
- Corpus: `axbench/0.1-ai-reviewed`, manifest `b8b6d39f…`. Only the validation split and the five train warm-ups were
  used; the sealed test was not read.

## Integrity (no scoring)

The frozen D18 integrity checks of `exp010.analysis/3` were run on the records in the Core environment, with the
approved commit `9902a02`:
- `check_records` (the 160 validation IDs, unique, in file order, one system, known statuses);
- `check_warmups`, `check_evaluation` (evaluation provenance against the training records), `check_sequence` and
  `check_monitor`.

Each returned no problem, and no line of either JSONL file is malformed. No scoring, Guard or comparison function was
run.

- **Requests:**
  - 160 of 160 validation requests have status `ok`; 5 of 5 warm-ups (`train-0001` … `train-0005`) have status `ok`;
  - finish reason `stop` for 152 requests, `length` (the 256-token cap) for 8. The 8 capped requests are reported as
    recorded and not repaired; they stay in every denominator.
- **Resources:**
  - MLX peak 3.88 × 10⁹ bytes over load, warm-up and validation (load 3.44 × 10⁹);
  - process peak RSS 2.67 × 10⁹ bytes;
  - cold load 0.70 s.
- **Stop rules:**
  - swap unchanged at 2,558,525 bytes throughout, with no growth;
  - pressure level 1 throughout, on AC power;
  - 166 in-process samples and 9 external monitor samples plus 1 final marker, with no stop;
  - the validation-slowdown baseline median is 37.1 generated tokens/s, with no stop.

## Limitations

This is one run on one machine. The corpus is AI-reviewed and exploratory, with 0 human decisions. Nothing here is an
accuracy, safety or deployment result.

## Next step

Step 4 is the lead's review of the preservation of these raw records. The frozen analysis (step 5, `--approved-commit
9902a02…`) is not run before that review.
