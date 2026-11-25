# DR-002: Statistics (Pearson, Spearman, K-means, group comparison)

- Status: **Synthesis of an AI-assisted research input plus the technical lead's decision. Not independent evidence.**
- Related: Issue #2, ADRs 0002-0006, `docs/evidence/experiments/EXP-001-issue2-numerical-validation/`.

## Provenance

- The technical lead commissioned a Deep Research report (DR-002, 33 pages) and privately reviewed it. The lead
  recorded that it could not retrieve the repository and that the lead verified the issue and `REFERENCES.md` directly.
  **The raw report is private and is not published here.** Statements below come from the lead's published
  comments ([research note](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7166467376),
  [decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230); REF-021), not from the report.
- The lead decision supersedes conflicting preliminary notes. The primary sources the lead cited were re-opened by the
  implementer before being registered (REF-007 to REF-020); sources not re-opened are not registered.

## What the lead recommended (proposals, not results)

1. Inspect: distinguish row, valid, missing and invalid counts; sample SD with ddof=1; null with reason when undefined;
   reject duplicate headers and silent coercion; preserve original row identity.
2. Relate: Pearson on paired finite observations, Spearman as Pearson of average ranks; report n and exclusions;
   constants and tiny samples undefined; no p-values or causal claims in the MVP.
3. Compare: explicit group and measure columns; separate group size and valid count; fixed ordering policy.
4. Cluster: explicit k, standardization, integer seed, explicit `n_init`, `algorithm="lloyd"`; no automatic k; handle
   constant features and non-convergence explicitly; canonical ids; descriptive (not real-world) classes.
5. CSV only, manual column selection; the model (later) produces only a validated typed plan, never code.
6. Known-answer fixtures; an independent oracle rather than calling the same library twice.

## What was implemented, what differs, and what was measured

| Item | Status |
|------|--------|
| Contracts above | **Implemented** (ADRs 0003-0006). |
| `statistics.mean/variance/stdev/median` as first candidate for Inspect | **Implemented**; measured against exact rational oracles (EXP-001). NumPy two-pass `std` was equally accurate on the five NIST datasets in the smoke comparison; the naive one-pass formula was not (0.1 absolute error on NumAcc4). |
| `statistics.correlation` for Relate | **Implemented behind a guard**: the installed 3.12.16 function returned `0.0` near 1e200 and raised a false "constant" error near 1e-200 on test inputs, SciPy `pearsonr` handled both. Power-of-two range scaling was added instead of adding SciPy as a direct dependency (tradeoff in ADR 0004). |
| Correlation matrix | **Deferred** (P1 in the lead decision). |
| Quartile accuracy criterion | One pre-registered criterion **failed**, was revised post hoc, and the revised criterion was then **falsified on held-back data** (EXP-002). Quartiles and the median are now exact rationals rounded once (ADR 0004, amendment 1). |
| K-means thread determinism | **Measured** on one machine: 9 distinct outputs in 20 default-thread fits (7 in an earlier run) versus 1 in 20 single-thread fits; the engine therefore fits single-threaded. |
| Independent verification | NIST certified values and exact rational oracles were used. These are independent references, but the oracle code was written by the same implementer; no independent reviewer has reproduced the results yet. |

## Limitations

- Everything measured was run on one Apple Silicon Mac (see EXP-001); no Windows numerical comparison exists.
- Accuracy criteria are provisional, derived a priori, and apply to the tested data families, not to arbitrary input.
- Clusters are descriptive partitions of a sample; K-means results are local solutions.
- DR-001 (reproducibility) belongs to Issue #3 and is not transferred here.
