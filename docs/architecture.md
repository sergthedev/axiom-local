# Architecture

> **Everything below is a proposal.** Only the empty package skeleton exists. See
> [ADR 0001](decisions/0001-initial-architecture.md) for rationale and unvalidated assumptions.

## Proposed data flow

```
CSV file ──► loader (planned) ──► typed Plan (planned) ──► engine (planned) ──► Result + provenance (planned)
                                      ▲
            optional intent router AX-01 (planned) / manual plan editing
```

## Components

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `axiom` package skeleton | Importable package, version string | Implemented |
| Loader | Read CSV deterministically, report schema and quality | Planned |
| Plan schema | Typed, validated description of an analysis request | Planned |
| Engine: Inspect | Descriptive statistics, data quality | Planned |
| Engine: Relate | Pearson and Spearman correlation | Planned |
| Engine: Cluster | Reproducible K-means with fixed seed | Planned |
| Engine: Compare | Group-level summaries | Planned |
| Intent router (AX-01) | Map natural language to a Plan; cannot compute | Planned |
| Desktop UI | Present plans and results | Planned |

## Invariants (design intent, not yet enforced by code)

- All numbers are produced by the deterministic engine, never by the model.
- The router output is a typed plan that is validated before execution; no model-generated code is run.
- Runtime is offline; no telemetry or external API calls.
- Results carry enough provenance (input hash, plan, seed, versions) to be re-run.
