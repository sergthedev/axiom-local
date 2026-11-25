# Roadmap

Planned work only; nothing here is implemented unless marked. Order is subject to owner and reviewer approval,
and each substantive step requires its research gate (see `docs/research/README.md`).

| Phase | Content | Status |
|-------|---------|--------|
| P0 | Bootstrap repository and governance (Issue #1) | Accepted by the technical lead |
| P1 | Deterministic statistical core: input guard, typed plans, Inspect, Relate, Compare, Cluster (Issue #2) | Accepted by the technical lead; merged (PR #6, merge `323620d`) |
| P2 | Analysis Passport, replay, environment verification, cross-platform comparison (Issue #3) | Accepted; merged (PR #7); ADRs 0007 and 0008 ratified by the owner |
| P3 | Correlation matrix, outlier flags (deferred P1 items) | Planned |
| P4 | Router boundary: candidate contract, schema projection, Guard (Issue #4, Phase 3A) | Implemented, under review |
| P5 | AXBENCH benchmark, harness and rule baseline (Issue #4, Phase 3A) | Draft implemented, under review; owner annotation review and sealing pending |
| P6 | AX-01: untuned baselines, LoRA pilot, frozen final evaluation (Phase 3B) | Protocols proposed; not started; requires review and owner approval |
| P7 | Desktop UI (macOS Apple Silicon) | Planned |
| P8 | Windows support | Planned |

Not planned: authentication, cloud, telemetry, API keys, arbitrary code execution, RAG, forecasting.
