# Optional model environment (Apple Silicon only)

A separate uv project for AX-01 experiments (ADR 0012). It is **not** needed for AXIOM Core, Verify, the Guard,
AXBENCH or CI, and it never runs on Windows. Its lock file is separate, so the Core `uv.lock` digest recorded in
analysis passports does not change when model packages change.

```bash
cd envs/model
uv sync --locked                      # mlx 0.32.4, mlx-lm 0.32.2 and their dependencies; axiom-local editable
uv run --locked python -c "import mlx.core as mx; print(mx.default_device())"
```

- `provision.py` downloads the files of one pinned checkpoint into a directory **outside** the repository, verifies
  the published SHA-256 of the weights and writes `PROVENANCE.json`. Run it only after the owner approves the
  download. `--tokenizer-only` fetches the configuration and tokenizer files (about 16 MB).
- `token_lengths.py` measures prompt and target token counts of AXBENCH train/validation under a provisioned
  tokenizer, offline.
- `exp009_run.py` is the EXP-009 runner (untuned inference on AXBENCH validation with resource measurement and stop
  rules): `verify` re-hashes the provisioned weights, `smoke` runs a few train examples, `main` and `fresh` are the
  protocol runs (see `docs/evidence/experiments/EXP-009-untuned-baselines/PRERUN.md`). Raw outputs are scored in the
  Core environment by `benchmarks/issue4_exp009_analyze.py`.
- `exp010_train.py` is the EXP-010 LoRA harness (`tokens`, `params`, `train`, `reload`, `evaluate`, `selftest`) and
  `exp010_launch.py` its detached launcher, a thin wrapper over the Core module `axbench.supervise` (common pre-load
  baseline, 30-second resource monitor, supervised steps that fail closed). Operational definitions, the installed
  MLX-LM behaviour they work around, and the pre-registered criteria are in
  `docs/evidence/experiments/EXP-010-lora-pilot/PRERUN.md`. The launcher's phases are `smoke` (training and the
  reload check), `pilot` (training only) and `evaluate` (the final pilot adapter on validation, its own supervised
  sequence). Only the accepted smoke has run; the pilot and its evaluation start only after clearance and the owner's
  approval, with the exact commands of PRERUN section 8. Their raw outputs are scored in the Core environment by
  `benchmarks/issue4_exp010_analyze.py` (`axbench.exp010analysis`).

Rules: no weights, adapters or caches in Git (`.gitignore` blocks common formats); offline inference only
(`HF_HUB_OFFLINE=1`); never download on failure; no remote experiment trackers; the final AXBENCH test split is never
exported for training.
