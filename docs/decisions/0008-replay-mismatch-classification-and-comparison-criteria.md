# ADR 0008: Fresh-process replay, mismatch classification and comparison criteria

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-25
- Issue / PR: Issue #3 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7167524679), [DR-001 note](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7166468157)); builds on ADRs 0004-0007

## Context

Replay must recompute, not reread: no result cache, a new process, a distinct workspace and run nonce. The outcome
must separate four questions that are easy to conflate: was the passport or its inputs altered (integrity), did the
computation complete (execution), are the bytes identical in the same environment (repeatability), and do two
environments agree within justified tolerances (cross-environment agreement). None of them is correctness.

## Decision

Implemented in `src/axiom/replay.py`, `src/axiom/_worker.py`, `src/axiom/agreement.py`, `src/axiom/partition.py`
and the CLI `python -m axiom {env,analyze,replay,compare}` (`src/axiom/__main__.py`).

**Fresh-process execution.** Every analysis and every replay run starts `python -I -m axiom._worker` (isolated mode:
no script directory or user site on `sys.path`, `PYTHON*` variables ignored; REF-028) in a new temporary working
directory, with a new `uuid4` run nonce. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`BLIS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are set to `1` in the child's environment, so they apply before NumPy
or scikit-learn load (REF-025); the worker records the pools it then observes. The worker receives the data path,
profile, plan, limits and (for replay) the old binding; it is never given a reference result, re-reads the file,
re-parses, and recomputes. The OS page cache may serve the bytes; that is not a result cache.

**Replay presents the old binding to the engine.** The worker builds a `BoundPlan` from the passport's plan and its
recorded plan and dataset digests and calls the unchanged `execute`. Changed data is therefore rejected by the
engine's own `plan_not_bound_to_dataset` check, a changed plan by `plan_hash_mismatch`. A new analysis on changed data
(`analyze`) is a new binding and a new passport, never a replay.

**Classification** (per run, then the worst over runs, in this order):

| Verdict | Meaning |
|---------|---------|
| `integrity_mismatch` | Passport digests do not recompute (nothing is executed), or the data's raw bytes or typed dataset differ from the passport, or the plan digest is rejected. Reported with `mismatch`: `passport`, `dataset` or `plan`. |
| `execution_failure` | The worker crashed, timed out, could not read the file, or the engine raised on unchanged inputs. |
| `numerical_discrepancy` | The computation completed but the result is not byte-identical in the same environment, or is outside the criteria below in a different environment. |
| `numeric_agreement` | Different environment, not byte-identical, every field within the criteria. |
| `exact_match` | Byte-identical canonical result in every run. |

**Same environment versus cross-environment.** Two environments are "the same" only if every comparable field is
equal (ADR 0007: Python version/implementation/compiler, OS system/release/machine, all package versions, NumPy
backend and SIMD, engine file digest, commit, dirty flag, lock digest, and the thread-limit probe); a field that is
unavailable on either side counts as different. In the same environment the **only** passing criterion is byte
identity (`exact_replay`); tolerances are not applied (`numerical_agreement` is `not_applicable`). In different
environments `exact_replay` is `not_applicable`, byte identity is still reported, and `numerical_agreement` is judged
with the criteria below. Replay never claims `independent_reference` (always `not_run`).

**Per-operation comparison criteria `axiom.agreement/1`** (fixed here and in the EXP-004/EXP-005 pre-registrations
before any comparison was evaluated; derived from earlier criteria, not from observed differences):

| Field | Criterion | Derivation |
|-------|-----------|------------|
| All non-float values (counts, keys, ordinals, warnings, undefined reasons, plan, input identities, null vs value, structure) | exact | Discrete; a value on one side and `null` on the other is always a failure. |
| Inspect/Compare and cluster original-unit `min`, `max` | exact | Order statistics are input values. |
| Inspect/Compare `mean`, `median`, `sample_variance`, `sample_sd`, `q1`, `q3`, `iqr`; cluster original-unit `mean` | <= 2 ulp apart | ADR 0004: each is within 1 ulp (mean, variance, SD) or 0.5 ulp (median, quartiles, IQR) of the exact value of the same inputs; two conforming results can differ by at most 2 ulp. |
| Relate `coefficient` | absolute difference <= 2e-12 | ADR 0004: each within 1e-12 of the exact rational value. |
| Cluster partition | exact equality of membership sets of source row ordinals (label numbering ignored; observed clusters are renumbered to the reference before fields are compared) | ADR 0006 canonical ids; authoritative. ARI (exact rational, REF-027) is auxiliary only. |
| Cluster `scaler.scale` | relative difference <= 2e-9 | ADR 0006 scaler check (rtol 1e-9) on each side. |
| Cluster `scaler.mean` | absolute difference <= 2^-11 times the reference scale | ADR 0006 scaler check (2^-12 scales) on each side. |
| Cluster `centroid_standardized` | absolute difference per coordinate <= 2 sqrt(tol + 1e-12) | ADR 0006 centroid check: on each side the squared distance of every centre to its member mean is at most tol x mean variance (= 1 after standardization) + 1e-12; with the same partition the member means agree, so the centres are at most twice that distance apart. |
| Cluster `inertia_standardized` | absolute difference <= n_eligible (tol + 1e-12) + 1e-9 inertia | Same check: inertia = within-cluster sum of squares about the member means + sum of n_j \|c_j - m_j\|^2, and the second term is bounded by the centroid check; 1e-9 covers the ADR 0006 inertia rounding check. |
| Cluster `n_iter_diagnostic`, `diagnostics.lloyd_label_fixed_point` | reported, not gating | The lead decision: iteration counts may differ across platforms. |

These are agreement criteria for two runs of AXIOM, not accuracy claims: two environments can agree on a wrong value,
and a matching partition is a local K-means solution, not a global optimum. They are not universal tolerances: they
hold for results that already passed the engine's own checks; an ill-conditioned input can make the inequalities
loose or the derivation's assumptions (for example equal member means) inapplicable.

**Seeds.** A fixed seed and a lockfile are necessary for repeatable K-means runs (REF-030) but neither guarantees
identical numbers across builds or machines (REF-029); the criteria above exist for that reason.

## Alternatives considered

- **In-process repetition only** (rejected: shares imports, caches and allocator state; Issue #2 already had it).
- **`multiprocessing` with fork** (rejected: inherits the parent's loaded libraries and thread pools; a new interpreter is
  the stronger isolation and works the same on Windows).
- **A universal epsilon** such as atol 1e-12 / rtol 1e-9 for all floats (proposed in the DR-001 note as a starting point;
  superseded by the lead decision; the per-field derivations above replace it).
- **ARI = 1 as the partition criterion** (rejected as authoritative: exact membership-set equality is stronger and has no
  floating point; ARI is reported).
- **Treating unavailable environment fields as equal** (rejected: unknown must never pass).

## Tradeoffs and consequences

- A fresh interpreter costs about 1.5 s per run on the development Mac, dominated by imports (EXP-004 measures it).
- The same-environment rule is strict: any OS update, package change or uncommitted edit makes a replay
  "cross-environment", where byte identity is reported but not required.
- Replay cannot defend against a modified engine or a compromised Python installation; it checks the data, the plan
  and the result against the passport, using the code that is installed.

## Unvalidated assumptions

- That the derived cluster bounds are tight enough to be useful; they are loose by construction (EXP-005 reports the
  actual differences next to the bounds).
- That `python -I` plus the variables above isolates the worker from user configuration on Windows as on macOS.
- That macOS arm64 and Windows x64 produce the same partition for the representative plans (measured in EXP-005, not
  assumed).

## References

REF-012, REF-020, REF-021, REF-025, REF-027, REF-028, REF-029, REF-030
