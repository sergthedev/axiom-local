# Experiment manifest: EXP-005 — cross-platform numerical comparison

- Issue / PR: Issue #3 (pull request linked from the issue)
- Date: 2026-11-25
- Status: **pending** — reference written; the comparison runs in the pull request CI and its result has not yet been recorded
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `f163b41` before any comparison

## Provenance
- Reference: `reference/*.passport.json`, written by `uv run python benchmarks/issue3_cross_platform.py --write-reference docs/evidence/experiments/EXP-005-cross-platform/reference`
  at commit `f163b41f437cbb7dbe40b5d42b78cea5b086fd22` (clean) on Apple M5, macOS 27.0.1 arm64, CPython 3.12.16, locked environment.
  Their result hashes are identical to the EXP-004 reference hashes of the same scenarios.
- Observed: the `core-tests` workflow step "Cross-platform numerical comparison (report only, EXP-005)" on `macos-latest` and
  `windows-latest`.

## Results (measured only)
- Not yet recorded.

## Limitations and failures
- See the pre-registration. A green CI step is not evidence of agreement; only the logged verdicts are.
