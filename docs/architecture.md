# Architecture

> The statistical core (loader, plans, four operations, canonical results) is **implemented** (Issue #2), and so are
> the Analysis Passport, environment capture and fresh-process replay (Issue #3). The router boundary (candidate
> contract, schema projection, Guard) and the AXBENCH benchmark tooling are implemented and under review (Issue #4);
> no router model exists. The UI is still a proposal. See the [ADRs](decisions/README.md)
> for rationale and unvalidated assumptions.

## Proposed data flow

```
CSV bytes ──► loader/guard ──► immutable Dataset ──► bind_plan(Dataset, candidate Plan) ──► execute ──► result core
              (implemented)     (implemented)          (implemented)                         (implemented)
                                                                                                │
                         Analysis Passport (identities, inputs, result, environment) ◄──────────┘ (implemented)
                         replay in fresh processes / compare across environments      (implemented)
                                                           ▲
                                                           │ confirmed Core plan only (implemented, ADR 0010)
   request + schema projection ──► router (rules today; AX-01 planned) ──► candidate text ──► strict parse
   (ids, names, kinds; no rows)        (implemented: axiom_router)          ──► semantic check ──► bind ──► Guard preview
                                                                             ──► explicit user confirmation ──► dispatch once
   manual plan editing ──────────────────────────────────────────────────────────► Guard preview (no router involved)
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
| Analysis Passport (`passport`, `passport_schema`, `environment`) | Versioned record of identities, explicit inputs, full-precision result, environment and execution | Implemented (ADR 0007) |
| Replay and comparison (`replay`, `_worker`, `agreement`, `python -m axiom`) | Fresh-process re-execution without result cache; integrity / execution / numerical classification; per-operation cross-environment criteria | Implemented (ADR 0008); measured in EXP-004 to EXP-007 |
| Candidate contract, schema projection, semantic validation (`axiom_router.contract`, `schema`, `semantic`, `prompt`) | Strict `axiom.candidate/1` parsing and checks against the real schema; model input is request + schema projection only | Implemented, under review (ADR 0009) |
| AXIOM Guard (`axiom_router.guard`) | Bind before preview; explicit confirmation of the exact plan and dataset; fail closed; one-shot dispatch; manual path | Implemented, under review (ADR 0010) |
| Rule router (`axiom_router.rules`) | Deterministic bilingual baseline router | Implemented (`axiom.rules/0.1`), measured on validation (EXP-008) |
| AXBENCH (`axbench`) | Benchmark generation, audits, scoring, statistics, sealing | Implemented draft (ADR 0011) |
| Intent router model (AX-01) | Propose a candidate from request + schema; cannot compute | Not trained; environment and protocols only (ADR 0012) |
| Desktop UI | Present plans and results | Planned |

## Invariants

Enforced by code and tests today:

- All numbers are produced by the deterministic engine. There is no model in the repository.
- `axiom` never imports `axiom_router` or `axbench`, and neither imports an ML library (tested); router code changes
  do not change the engine digest.
- No router output reaches the executor without a Guard confirmation of the exact displayed plan and dataset.
- A plan is validated data: unknown fields, hashes, duplicate keys and non-finite numbers are rejected; nothing is evaluated.
- A bound plan only executes on the exact dataset snapshot it was bound to.
- Results contain no timestamps, paths or host details; environment information is separate.
- The core has no network access and no telemetry (no networking code exists in `src/axiom`).

- Replay never receives a reference result; it recomputes from the data file in a new process and presents the old binding
  to the unchanged engine, so changed data is rejected by the engine itself.
- Unperformed verification is recorded as `not_run`, never as a pass; replay never claims an independent-reference check.

Design intent, not yet implemented: a trained AX-01 model; a desktop UI.
