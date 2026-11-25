# Architecture

> The statistical core (loader, plans, four operations, canonical results) is **implemented** (Issue #2).
> The intent router, UI, run passport and replay are still proposals. See the [ADRs](decisions/README.md)
> for rationale and unvalidated assumptions.

## Proposed data flow

```
CSV bytes ──► loader/guard ──► immutable Dataset ──► bind_plan(Dataset, candidate Plan) ──► execute ──► result core
              (implemented)     (implemented)          (implemented)                         (implemented)
                                                           ▲
                       optional intent router AX-01 (planned) / manual plan editing
```

## Components

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `axiom` package skeleton | Importable package, version string | Implemented |
| Loader (`ingest`) | Strict, explicit, fail-closed CSV parsing into an immutable snapshot with two hashes | Implemented (ADR 0003) |
| Plan schema (`plan`) | Versioned strict Pydantic plans; candidate vs engine-bound plan | Implemented (ADR 0005) |
| Engine: Inspect | Descriptive statistics, data quality | Implemented (ADR 0004) |
| Engine: Relate | Pearson and Spearman correlation of one pair | Implemented (ADR 0004); matrix deferred |
| Engine: Cluster | K-means with explicit solver parameters and independent postconditions | Implemented (ADR 0006) |
| Engine: Compare | Group-level summaries | Implemented (ADR 0004) |
| Canonical serialization | Canonical JSON v1 and SHA-256 identities | Implemented (ADR 0005) |
| Run passport, replay, environment verification | Provenance and independent re-execution | Planned (Issue #3); only a minimal `environment_info()` exists |
| Intent router (AX-01) | Map natural language to a Plan; cannot compute | Planned |
| Desktop UI | Present plans and results | Planned |

## Invariants

Enforced by code and tests today:

- All numbers are produced by the deterministic engine. There is no model in the repository.
- A plan is validated data: unknown fields, hashes, duplicate keys and non-finite numbers are rejected; nothing is evaluated.
- A bound plan only executes on the exact dataset snapshot it was bound to.
- Results contain no timestamps, paths or host details; environment information is separate.
- The core has no network access and no telemetry (no networking code exists in `src/axiom`).

Design intent, not yet implemented: results carry a complete run passport (Issue #3); cross-platform numerical
comparison (Issue #3); intent routing by AX-01.
