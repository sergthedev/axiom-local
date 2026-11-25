# Experiments

Every experiment gets a manifest copied from [MANIFEST_TEMPLATE.md](MANIFEST_TEMPLATE.md), committed together with
sanitized logs. Failed experiments are recorded too. Training, validation and held-out test data must stay separate.

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| [EXP-001](EXP-001-issue2-numerical-validation/MANIFEST.md) | Issue #2 numerical validation and repeatability | run (one machine) | NIST StRD, exact oracles, thread determinism, fresh-process replay; includes one recorded criterion failure |
| [EXP-002](EXP-002-quartile-confirmation/MANIFEST.md) | Pre-registered quartile confirmation on held-back data | run (one machine) | Falsified the EXP-001 post hoc criterion for NumPy; production quartiles exact |
| [EXP-003](EXP-003-cluster-workload/MANIFEST.md) | K-means wall time versus the workload proxy | run (single runs) | Context for the provisional workload cap, not a run-time guarantee |
| [EXP-004](EXP-004-replay-benchmark/MANIFEST.md) | Issue #3 fresh-process replay, mutation and malformed input | run (one machine) | Pre-registered; 200 of 200 replays byte-identical; mutation criteria met |
| [EXP-005](EXP-005-cross-platform/MANIFEST.md) | Issue #3 cross-platform comparison: macOS arm64 reference vs CI macOS and Windows x64 | run (one CI run per runner) | Pre-registered; 10 of 10 byte-identical on both runners; report-only CI step |
| [EXP-006](EXP-006-review-confirmation/MANIFEST.md) | Issue #3 review corrections: per-run replay evidence, adversarial A1-A3, Windows checkout diagnosis | run (one machine; one CI run per runner) | Pre-registered; 200 of 200 exact with per-run evidence; CRLF cause confirmed and fix verified on both runners |
| [EXP-007](EXP-007-final-audit/MANIFEST.md) | Issue #3 final review and audit: fresh replay smoke, bounded claimed counts, parse memory | run (one machine; one CI run per runner) | Pre-registered; 30 of 30 exact; no allocation grows with claimed `k`; parse amplification up to 25x measured; 457 tests and 10 of 10 exact on both runners |
