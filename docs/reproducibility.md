# Reproducibility

## Currently implemented

- Dependencies are locked in `uv.lock` (Python 3.12 only); `uv sync --locked` fails if the lockfile and
  `pyproject.toml` disagree (REF-003).
- CI runs lint and tests on macOS and Windows with Python 3.12; the workflow actions are pinned to commit SHAs.
- Every result carries four identities: SHA-256 of the exact CSV bytes parsed, of the typed dataset (versioned,
  length-prefixed), of the explicit plan, and of the canonical result core (ADR 0005). The result core excludes
  timestamps, paths and host details. Hashes show identity and integrity, **not correctness**.
- Same input and plan give identical canonical result bytes: measured in-process and across 20 fresh processes on one
  macOS arm64 machine (EXP-001). K-means is fitted single-threaded because default multi-threaded fits produced
  different bit-level outputs in a measurement on that machine (ADR 0006).
- Numerical correctness is checked separately against exact-arithmetic oracles and NIST certified values (EXP-001).

## Verification and replay (Issue #3)

- `python -m axiom analyze` runs a plan in a fresh process and writes an **Analysis Passport** (`axiom.passport/1`, ADR 0007):
  the four identities, the explicit CSV profile, plan and limits, the full-precision result core, the environment manifest
  (Python, OS, architecture, package versions, engine file digest, commit, lock digest, NumPy BLAS/SIMD, effective thread
  pools; unobservable items are marked `unavailable`) and the execution record.
- `python -m axiom replay --passport P --data D --runs N` re-executes the plan in N new interpreters (thread variables set
  before imports, fresh workspace, run nonce, no result cache) and classifies the outcome as `exact_match`,
  `numeric_agreement`, `numerical_discrepancy`, `execution_failure` or `integrity_mismatch` (ADR 0008).
- **Same environment**: only byte identity passes. **Different environment**: per-operation criteria `axiom.agreement/1`
  (ADR 0008), with cluster partitions compared by membership sets independent of label numbers.
- Measured: 200 of 200 replays byte-identical on one Mac (EXP-004); the cross-platform comparison with GitHub-hosted
  macOS and Windows x64 runners is EXP-005.

```bash
uv run python -m axiom analyze --data examples/synthetic_regions.csv \
  --profile examples/profiles/synthetic_regions.json --plan examples/plans/cluster.json --out passport.json
uv run python -m axiom replay --passport passport.json --data examples/synthetic_regions.csv --runs 20
```

## Caveats

- A lockfile pins versions; it does not prove numerical correctness, and neither do a seed or a hash.
- Bit-for-bit equality across platforms is **not** claimed. Cross-platform results are limited to what EXP-005 measured on
  GitHub-hosted runners; that is not packaged Windows support and not independent human replication.
- Replay checks data, plan and result against a passport using the installed code; it cannot detect a modified engine or
  Python installation, and a passport digest is not a signature.
