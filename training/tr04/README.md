# EXP-TR04 training driver (exploratory; prepared, not executed, not approved)

**No model has been loaded or trained by this code, and no technical clearance exists.** The driver is for an exploratory
one-epoch LoRA run on machine-audited, rights-screened synthetic rows. The result of such a run would be neither human-validated
nor deployment-ready. Protocol, frozen evaluation criteria and limitations:
[PROTOCOL.md](../../docs/evidence/experiments/EXP-TR04-exploratory-lora/PROTOCOL.md).

It reuses the frozen EXP-TR02 harness (`training/tr02/tr02.py`, imported and never edited) and the thermal parser of
`training/tr03/tr03.py`:

- `config.json`: TR02's model, package, hyperparameter and limit pins copied unchanged (a test compares them), plus the SHA-256 of the
  corpus, per-example lengths, MLX export, audit outputs and held-out files, two phases (`train`, then `reload`), the monitor
  settings and the frozen budget with wall-clock caps. Changing it means a new version.
- `tr04.py`: `TR04Rules` (TR02's event audit plus thermal, sustained-pressure, throughput-reference and wall-cap rules), a macOS
  sampler that adds `pmset -g therm`, `run_reload` (TR02's reload generalized from its smoke phase), the TR04 preflight (TR02's plus
  config, audit-file and export-keep-list checks) and the command line.

Commands, from the repository root, in the locked model environment on the Apple Silicon Mac
(`RUN="uv run --locked --project envs/model python training/tr04/tr04.py"`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).
**No-weight commands** (allowed before clearance; none loads weights):

```
uv run --locked python -m axbench_tr.tr04 export-mlx --out "$WORK/export"     # core environment; outside the repository
$RUN verify-config                                  # config consistent with TR02, data hashes, frozen budget
$RUN plan                                           # budget arithmetic
$RUN verify-inputs  --export "$WORK/export"
$RUN schedule       --export "$WORK/export"
$RUN verify-model   --model-dir "$MODEL"            # hashes the ten checkpoint files; loads nothing into MLX
$RUN check-data     --export "$WORK/export" --tokenizer-dir "$MODEL"
$RUN check-iterator --export "$WORK/export" --tokenizer-dir "$MODEL"   # real mlx-lm iterator and loss mask, tokenizer only
$RUN check-lora                                     # tiny random Qwen3 only
$RUN preflight --phase train --model-dir "$MODEL" --export "$WORK/export" --out "$WORK/train" --expect-commit "$COMMIT"
```

**Execution commands (do not run before technical clearance on GitHub):** `$RUN supervise --phase train ...` then
`$RUN supervise --phase reload ... --after "$WORK/train"`; `collect-evidence` and `sanitize-evidence` afterwards, from a separate
evidence checkout, exactly as in `training/tr02/README.md`. A stop is a failed run: no retry, no resume, outputs are never overwritten.

Tests (no weights, no optimizer step): `tests/test_tr04_driver.py`, `tests/test_tr04_audit.py`, `tests/test_tr04_heldout.py`.
