# ADR 0002: Dependency stack and Python version for the statistical core

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-25
- Issue / PR: Issue #2 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230), item 7)

## Context

The deterministic core needs strict typed plans, array numerics for K-means, and a
vetted K-means solver. ADR 0001 left numerical libraries open pending research gate DR-002.

## Alternatives considered

| Topic | Chosen | Not chosen (reason) |
|-------|--------|---------------------|
| Tabular engine | Python stdlib `csv` + typed tuples | pandas, Polars, Arrow, DuckDB (excluded by the lead decision; larger dependency and coercion surface than the CSV-only scope needs) |
| Plan validation | Pydantic v2 (strict, frozen, `extra="forbid"`) | dataclasses + hand validation (more custom code for the same checks); attrs/msgspec (not evaluated) |
| Array numerics | NumPy | pure Python only (K-means distances would be slow and larger to maintain) |
| K-means | scikit-learn `KMeans` + `StandardScaler` | custom K-means (would need its own validation; the lead decision prefers a tested solver plus independent postcondition checks) |
| SciPy | Transitive only (pulled in by scikit-learn); no production import | Direct dependency (not needed: no production code imports it) |
| Thread control | `threadpoolctl` as a **direct** dependency | Relying on its transitive presence (production code imports it, so it is declared) |
| High-precision oracle | `fractions` / `decimal` from the stdlib, tests only | mpmath (not needed) |
| Python versions | 3.12 only (`requires-python >=3.12,<3.13`) | 3.11 (NumPy 2.5.5 requires >=3.12; 3.11 was never tested here, see Issue #1 review); 3.13+ (untested) |

## Decision

Runtime dependencies are pinned exactly in `pyproject.toml` and locked in `uv.lock`:
numpy 2.5.5, pydantic 2.14.4, scikit-learn 1.9.1, threadpoolctl 3.8.0 (transitive: scipy 1.18.1, joblib 1.6.0,
pydantic-core 2.50.5, cloudpickle 3.1.3, narwhals 2.29.0, annotated-types, typing-extensions,
typing-inspection). Versions were the latest on the access date (REF-019); no newer-is-better claim is made.
Tests use plain pytest; no external skills or agents are used.

## Tradeoffs and consequences

- Python 3.11 is no longer claimed. The README previously said 3.11 was supported without any test; it is now
  3.12 only, which removes that unverified claim.
- scikit-learn 1.9.1 brought two new transitive packages (cloudpickle, narwhals) that were not individually audited.
- NumPy, SciPy and scikit-learn wheels bundle native BLAS/OpenMP runtimes whose licenses were not individually
  audited (see `THIRD_PARTY_NOTICES.md`).

## Unvalidated assumptions

- Windows x64 resolves the same locked set and passes the same tests (wheels exist per PyPI file listings, REF-019;
  the CI result is recorded in `docs/evidence/DEVELOPMENT_LOG.md`, not asserted here).
- Numerical agreement between macOS arm64 and Windows x64 is **not** claimed (Issue #3 owns that comparison).

## References

REF-004, REF-007, REF-012, REF-013, REF-014, REF-019
