# EXP-TR02 harness (prepared, not executed)

Independent of the EXP-010 harness (PR #11, untouched). **No model has been loaded or run by this code.** The model
path (`run_phase`, `run_reload`) is reviewed by reading only; everything else is unit-tested in
`tests/test_tr02_harness.py` (data path, schedule, exposure checks, stop rules, probe parsers, supervisor). The first
run needs the owner's approval of `docs/evidence/experiments/EXP-TR01-axbench-tr-acquisition/PROPOSED_EXP-TR02.md`.

- `config.json`: immutable pins (model revision, tokenizer hashes, corpus, export and per-example hashes), the
  hyperparameters, memory and throughput limits, and the two phases. Changing it means a new version and a new review.
- `tr02.py`: commands `pin`, `verify-inputs`, `schedule`, `check-data`, `verify-model`, `run`, `supervise`.

Why not `python -m mlx_lm lora`: it truncates over-length sequences with a warning, reports loss-token (not full-token)
throughput, validates one update early, supervises a thinking block the EXP-010 rendering does not, and skips validation
rows by default. The comparison is in the protocol; every statement was read from the mlx-lm 0.32.2 source.

## Commands (Apple Silicon Mac, `envs/model`, after owner approval)

```
python -m axbench_tr export-mlx --out "$WORK/export"            # outside the repository
python training/tr02/tr02.py verify-inputs --export "$WORK/export"
python training/tr02/tr02.py check-data --export "$WORK/export" \
    --tokenizer-dir "$MODEL" --per-example data/axbench-tr/v0.2-rev1/token_lengths_per_example.jsonl
python training/tr02/tr02.py verify-model --model-dir "$MODEL"
python training/tr02/tr02.py supervise --phase probe --model-dir "$MODEL" --tokenizer-dir "$MODEL" \
    --export "$WORK/export" --out "$WORK/probe"
python training/tr02/tr02.py supervise --phase smoke --model-dir "$MODEL" --tokenizer-dir "$MODEL" \
    --export "$WORK/export" --out "$WORK/smoke"
python training/tr02/tr02.py run --phase reload --model-dir "$MODEL" --tokenizer-dir "$MODEL" \
    --export "$WORK/export" --adapter-dir "$WORK/smoke" --out "$WORK/reload"
```

Run `HF_HUB_OFFLINE=1`. Output directories must be new and empty. Supervisor files are written beside each output
directory (`<out>.supervisor/`: `supervisor.jsonl`, `child.log`, and `STOP.json` if a rule fired).
Weights, adapters and logs stay outside the repository; only the report is committed.

Preparation checks already run on Linux without weights: the pinned export reproduces its hashes; all 7,432 rows
render through the harness dataset to the measured lengths with the supervised span equal to the answer; the
seeded order equals the stock `mlx_lm` iterator for both subsets (mlx 0.32.4 CPU build, no model).
