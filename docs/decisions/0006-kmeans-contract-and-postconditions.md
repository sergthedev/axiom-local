# ADR 0006: K-means contract, canonical cluster ids and independent postconditions

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-25
- Issue / PR: Issue #2 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230), item 5)

## Context

K-means is sensitive to scaling, initialization, threading and solver stopping rules, and its cluster labels are
arbitrary. A fixed seed alone does not guarantee identical output across builds or machines.

## Decision

Implemented in `src/axiom/ops_cluster.py`, with `src/axiom/partition.py`.

- **Solver**: `StandardScaler` (ddof=0, REF-013) then `KMeans` with explicit k-means++ initialization, integer seed
  (default 42), `n_init=10`, `max_iter=300`, `tol=1e-4`, `algorithm="lloyd"`, `copy_x=True` (REF-012). All values are
  part of the plan and the result. They are provisional pending local validation; no global optimum is claimed.
  The fit runs under `threadpool_limits(limits=1)`: EXP-001 measured 9 distinct outputs in 20 fits with default threads
  (7 in an earlier run of the same experiment) versus 1 in 20 with a single thread (both runs) on one machine, so the bit-level result is only reproducible single-threaded.
- **Eligibility**: finite numeric features, listwise deletion with counts and ordinals; `2 <= k <= max_k`, `k` not above
  the number of eligible rows nor the number of distinct eligible points (checked both in original and standardized
  space); features that are constant, near-constant (same 2^-40 relative-spread rule as ADR 0004) or of magnitude above
  2^500 are rejected, so standardized values stay finite and meaningful.
- **Fail-closed postconditions** (thresholds fixed before measurement): any solver or NumPy warning, `n_iter >= max_iter`
  (conservative, even if the last iteration converged), an empty cluster, or a failed independent check rejects the result.
  Independent checks recompute, with separate NumPy code in bounded batches: nearest-center assignment (squared-distance
  excess allowed <= 1e-9*(1+||x||^2)); centroid fixed point (sum of squared center-to-member-mean distances <= `tol` times
  mean feature variance, which is the solver's own stopping rule: the loop in the installed `_kmeans.py` stops when the
  squared center shift <= `tol * mean(var(X))`, REF-020); inertia (relative error <= 1e-9); scaler mean/scale against a
  two-pass NumPy computation (scale relative error <= 1e-9, mean within 2^-12 scales). These show **internal consistency
  of a local solution**, not that it is a global or "correct" clustering.
- **Canonical ids**: clusters are ordered by their sets of source ordinals (equivalently by smallest member ordinal), not by
  centroid floats; raw solver labels are never output. `partition.same_partition` compares partitions exactly by
  membership sets; ARI is deferred to Issue #3 as a supplementary check.
- **Output**: eligible and excluded ordinals, assignments, scaler parameters, solver parameters (`n_iter` is labelled
  diagnostic), standardized inertia recomputed independently, standardized centroids, and original-unit
  mean/min/max per cluster and feature. Warnings state that clusters are descriptive partitions and local solutions;
  no labels or socioeconomic names are invented.

## Alternatives considered

- Ordering ids by centroid coordinates (rejected: floating-point near-ties make ids unstable).
- Automatic k selection, robust scaling or log transforms (out of scope for v0.1).
- Allowing multi-threaded fits (rejected for now by measurement above).

## Tradeoffs and consequences

- Rejecting at `n_iter == max_iter` can discard a solution that converged exactly on the last iteration.
- The symmetric-input case (for example a rectangle) has no unique expected partition; tests accept any valid result there and check repeatability instead.
- Single-threaded fitting is slower for large inputs (100,000 rows with k=4 took 0.5-0.6 s in two runs on the measured machine).

## Unvalidated assumptions

- Whether the same partition results on macOS arm64 and Windows x64 is not measured.
- The tolerances are a priori choices; EXP-001 reports the measured residuals only through pass/fail in tests.

## References

REF-012, REF-013, REF-020

## Amendment 1 (2026-11-25): response to the technical review of PR #6, blocker 4

The original text above is preserved. Changes:

- **v1 solver settings** are bounded in the plan schema (provisional): `n_init` 1..20, `max_iter` 1..1,000, `tol` 1e-8..1e-2 (defaults
  unchanged: 10, 300, 1e-4).
- **Joint workload guard.** Work bound W = eligible rows x k x features x n_init x max_iter must be <= `Limits.max_cluster_work` =
  1e11 (provisional). It is checked at binding (rows upper bound), at execution (with the limits given there) and defensively inside
  the operation, always before any fit; tests assert that `KMeans` is never constructed when the cap is exceeded. EXP-003 measured
  wall time against W on one machine: the slowest observed rate was about 3e9 W-units per second, so the cap corresponds to roughly
  30-35 s single-threaded if every allowed iteration were used. This is a resource guard, not a statistical constant.
- **Solver output validation** before any indexing or serialization: labels shape, integer dtype and range [0, k); centers shape,
  float dtype and finiteness; inertia finite and non-negative; n_iter positive; missing or malformed attributes. Failures are
  `ComputationError("invalid_solver_output")`; solver `ValueError`/`OverflowError`/`MemoryError` become `solver_failure`; a
  `FloatingPointError` (over/invalid/divide raised by NumPy) becomes `numeric_exception`.
- **Warnings across the whole numeric block.** The scaler, the fit and the independent recomputations run under one
  warning-recording block; any warning rejects the result (`solver_warning`).
- **Scale handling (replaces the 2^500 magnitude cap).** Each feature is scaled by an exact power of two so max|v| is in [0.5, 1) before
  standardization (standardization is invariant under exact power-of-two scaling); scaler mean and scale are reported back in original
  units by the exact inverse scaling, and original-unit cluster summaries come from the unscaled data. Tiny and huge magnitudes
  therefore standardize correctly (tests: exact power-of-two rescalings give identical assignments, centroids and inertia). A feature
  with an outlier that collapses the other points to the same standardized value is rejected by the distinct-points check.
- **Renamed check.** `centroid_fixed_point` is now `centroid_consistency_approx`. It states that every centroid is within the solver's
  own stopping tolerance of the member mean (squared shift sum <= `tol * mean(var)`); a stopping criterion is not proof of an exact Lloyd
  fixed point, particularly after the final label reassignment. A separate, non-gating diagnostic `lloyd_label_fixed_point` reports
  whether recomputing centroids from the labels and reassigning leaves the labels unchanged.
- **Reproducibility wording.** Single-threaded fitting removed the thread-order nondeterminism observed in EXP-001, on one machine. It does not
  by itself guarantee reproducibility across platforms, BLAS builds or library versions; capturing and verifying the effective thread policy
  belongs to Issue #3.
