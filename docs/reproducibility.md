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

## Planned (Issue #3)

- Run passport, environment verification (BLAS/OpenMP/threadpool/CPU), replay with no result cache, and a
  documented macOS arm64 vs Windows x64 numerical comparison.

## Caveats

- A lockfile pins versions; it does not prove numerical correctness, and neither do a seed or a hash.
- Bit-for-bit equality across platforms is **not** claimed. Nothing has been measured on Windows beyond the CI test run.
