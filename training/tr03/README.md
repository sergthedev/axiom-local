# EXP-TR03 plan tooling (proposed; nothing here trains, loads a model or is approved)

Companion to `docs/evidence/experiments/EXP-TR03-research-protocol/PROTOCOL.md`. **No model path exists in this directory.** The
frozen EXP-TR02 harness (`training/tr02/`) and EXP-010 (PR #11) are not modified; TR02's exact loss mask, rendering, LoRA name
contract and event audit are reused as they are by any later driver.

- `config.json`: the proposed single-run plan (`status`: not approved, no clearance). Model pins, packages, hyperparameters and memory
  limits are copied from `training/tr02/config.json`, whose SHA-256 is recorded inside; a test fails if they drift.
- `tr03.py`: pure, tested helpers: config checks, exposure schedule with the `drop_remainder` partial-batch rule, fixed evaluation and
  snapshot points, time budget, the fail-closed long-run monitor, the thermal-text parser and the readiness report.

```
uv run python training/tr03/tr03.py verify-config   # consistent with TR02; exit 0
uv run python training/tr03/tr03.py plan            # schedule and budget for all eligible rows (add --write to refresh)
uv run python training/tr03/tr03.py check-plan      # committed data/axbench-tr/tr03-prep/TRAINING_PLAN.json equals a rebuild
pmset -g therm | uv run python training/tr03/tr03.py check-therm   # on a Mac, in a future no-weight preflight
uv run python training/tr03/tr03.py readiness       # exit 1 while any human or clearance gate is closed (today: all closed)
```

Two arms: **T** (research training; needs gates G1 to G7) and **R** (discard-only resource run; needs clearance only). Neither has a
driver yet. See the protocol for why and for the launch recommendation.
