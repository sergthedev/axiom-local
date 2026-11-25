# Experiments

Every experiment gets a manifest copied from [MANIFEST_TEMPLATE.md](MANIFEST_TEMPLATE.md), committed together with
sanitized logs. Failed experiments are recorded too. Training, validation and held-out test data must stay separate.

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| [EXP-001](EXP-001-issue2-numerical-validation/MANIFEST.md) | Issue #2 numerical validation and repeatability | run (one machine) | NIST StRD, exact oracles, thread determinism, fresh-process replay; includes one recorded criterion failure |
