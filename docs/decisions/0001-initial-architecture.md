# ADR 0001: Initial architecture and tooling

- Status: Accepted by the technical lead as part of the Issue #1 foundation review ([review](https://github.com/sergthedev/axiom-local/issues/1#issuecomment-7166799471)); formal owner approval is not recorded here and remains pending
- Date: 2026-11-25
- Issue / PR: Issue #1

## Context

AXIOM needs a verifiable offline workbench: deterministic calculations, optional local intent routing, macOS Apple
Silicon first and Windows later. The repository was bootstrapped with no engine code. Choices below are for the
skeleton only.

## Alternatives considered

| Topic | Chosen | Alternatives not chosen (reason) |
|-------|--------|----------------------------------|
| Calculation vs. language model | Deterministic Python engine computes; model only routes | Model-computed numbers (not verifiable, not reproducible) |
| Language / versions | Python 3.11–3.12 | 3.13+ (not tested in this bootstrap); Rust/other (scope and contributor friction) |
| Layout | `src/axiom` package, `tests/` | Flat layout (import shadowing risk) |
| Build backend | hatchling | setuptools (no concrete reason to prefer; hatchling is smaller config) |
| Env/lock tool | uv with `uv.lock` (REF-003) | pip-tools, poetry (not evaluated in depth) |
| Lint/test | ruff, pytest (REF-004) | flake8/black (not evaluated in depth) |
| CI | GitHub Actions on macOS and Windows (REF-005, REF-006) | Linux-only (does not cover target platforms) |

## Decision

Adopt the layout and tools above for the bootstrap. The runtime has zero third-party dependencies. Numerical
libraries (e.g. for statistics) are **not** chosen here; that requires research gate DR-002 and a later ADR.

## Tradeoffs and consequences

- uv adds a tool prerequisite for contributors.
- Zero runtime dependencies keeps the skeleton light but defers all dependency decisions (superseded in part by ADR 0002 for the numerical dependency stack).

## Unvalidated assumptions

- That uv and hatchling work equally on Windows CI (observed afterwards: the `core-tests` run 45644347806 for commit 1d3b646 succeeded on macOS and Windows, inspected independently by the technical lead; one run only).
- That Python 3.11 and 3.12 behave identically for our later numerical code (untested; Python 3.11 support was dropped in ADR 0002).
- That a locally fine-tuned small model is accurate enough for intent routing (no data, no measurement).
- That determinism across macOS arm64 and Windows is achievable within a tolerance (not measured).

## References

REF-003, REF-004, REF-005, REF-006
