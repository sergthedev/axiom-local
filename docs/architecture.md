# Architecture

> The statistical core (loader, plans, four operations, canonical results) is **implemented** (Issue #2), and so are
> the Analysis Passport, environment capture and fresh-process replay (Issue #3). The intent router and UI are still proposals. See the [ADRs](decisions/README.md)
> for rationale and unvalidated assumptions.

## Proposed data flow

```
CSV bytes ──► loader/guard ──► immutable Dataset ──► bind_plan(Dataset, candidate Plan) ──► execute ──► result core
              (implemented)     (implemented)          (implemented)                         (implemented)
                                                                                                │
                         Analysis Passport (identities, inputs, result, environment) ◄──────────┘ (implemented)
                         replay in fresh processes / compare across environments      (implemented)
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
| Analysis Passport (`passport`, `environment`) | Versioned record of identities, explicit inputs, full-precision result, environment and execution | Implemented (ADR 0007) |
| Replay and comparison (`replay`, `_worker`, `agreement`, `python -m axiom`) | Fresh-process re-execution without result cache; integrity / execution / numerical classification; per-operation cross-environment criteria | Implemented (ADR 0008); measured in EXP-004 and EXP-005 |
| Intent router (AX-01) | Map natural language to a Plan; cannot compute | Planned |
| Desktop UI | Present plans and results | Planned |

## Invariants

Enforced by code and tests today:

- All numbers are produced by the deterministic engine. There is no model in the repository.
- A plan is validated data: unknown fields, hashes, duplicate keys and non-finite numbers are rejected; nothing is evaluated.
- A bound plan only executes on the exact dataset snapshot it was bound to.
- Results contain no timestamps, paths or host details; environment information is separate.
- The core has no network access and no telemetry (no networking code exists in `src/axiom`).

- Replay never receives a reference result; it recomputes from the data file in a new process and presents the old binding
  to the unchanged engine, so changed data is rejected by the engine itself.
- Unperformed verification is recorded as `not_run`, never as a pass; replay never claims an independent-reference check.

Design intent, not yet implemented: intent routing by AX-01; a desktop UI.
