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
- Training follows the EXP-010 protocol; nothing has been trained.

Rules: no weights, adapters or caches in Git (`.gitignore` blocks common formats); offline inference only
(`HF_HUB_OFFLINE=1`); never download on failure; no remote experiment trackers; the final AXBENCH test split is never
exported for training.
