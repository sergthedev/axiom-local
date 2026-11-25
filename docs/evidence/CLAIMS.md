# Claims

| ID | Claim | Status | Evidence | Limitations |
|----|-------|--------|----------|-------------|
| C-001 | A Python package skeleton `axiom` exists and imports | implemented, measured | Smoke test; run recorded in [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | Placeholder only; no analysis feature |
| C-002 | Locked dev environment installs and tests pass on macOS arm64, Python 3.12 | measured (single local run) | [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | One machine, one run; not independently verified |
| C-003 | CI passes on macOS and Windows | not claimed | State is reported on Issue #1 after the workflow runs | Depends on actual workflow results |
| C-004 | Deterministic engine for Inspect/Relate/Cluster/Compare | proposed | [docs/architecture.md](../architecture.md) | Not implemented |
| C-005 | Local model can route intents accurately | proposed | none | No data, training or measurement exists |
