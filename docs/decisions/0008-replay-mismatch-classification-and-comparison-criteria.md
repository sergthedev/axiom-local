# ADR 0008: Fresh-process replay, mismatch classification and comparison criteria

- Status: **Accepted.** Technical review accepted with the
  [Verify acceptance](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7173588575) (PR #7 merged); ratified
  by the repository owner on 2026-11-25 in the owner's Phase 3A authorization for Issue #4 (a project-thread message,
  recorded in `docs/evidence/DEVELOPMENT_LOG.md`). Text unchanged since the final revision. Revised after the
  [PR #7 review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7171558913) (blockers B1 and B3; see
  "Revision after technical review" below) and after the
  [final review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7172653455) (residual B1 and final audit).
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

**Trusted runtime policy (B1).** A passport is untrusted input, so the limits it records are not a resource budget.
`replay` compares every recorded limit with `policy` (default: `DEFAULT_LIMITS`, the accepted Core caps; a caller may
pass an explicit, trusted override) and rejects a passport asking for more with `limits_exceed_policy` before any
integrity work and before any worker starts. `compare_passports` executes nothing and has no policy; its validation,
integrity checks and comparison are linear in the two documents and never sized by a count they state. Recorded limits are never clamped: different limits can change validation and reported ordinals, so a
clamped run would be a different analysis presented as the same replay. Smaller recorded limits are used as recorded.
`runs` must be an integer in [1, 100] (100 is the release target per golden scenario), `timeout` a finite number of
seconds in (0, 3600] (checked without converting integers to floats) and the data path a str or path-like without NUL;
`analyze` and `run_worker` check the timeout too, and `analyze` requires a valid `Limits`.
Violations raise `VerifyError` (CLI exit 2). Passport text is size-bounded before parsing (ADR 0007).

**Replay presents the old binding to the engine.** The worker builds a `BoundPlan` from the passport's plan and its
recorded plan and dataset digests and calls the unchanged `execute`. Changed data is therefore rejected by the
engine's own `plan_not_bound_to_dataset` check, a changed plan by `plan_hash_mismatch`. A new analysis on changed data
(`analyze`) is a new binding and a new passport, never a replay.

**Classification** (per run, then the worst over runs, in this order):

| Verdict | Meaning |
|---------|---------|
| `integrity_mismatch` | Passport digests do not recompute (nothing is executed), or the data's raw bytes or typed dataset differ from the passport, or the plan digest is rejected. Reported with `mismatch`: `passport`, `dataset` or `plan`. |
| `execution_failure` | The worker crashed, timed out, could not read the file, or the engine raised on unchanged inputs. |
| `numerical_discrepancy` | The computation completed but the result is not byte-identical where no environment difference was observed (verified or unverified), or is outside the criteria below in an observed different environment. |
| `numeric_agreement` | Observed different environment, not byte-identical, every field within the criteria. |
| `exact_match` | Byte-identical canonical result in every run. |

**Environment status and comparison mode.** Each run's worker environment is classified against the passport's with
`compare_environments` (ADR 0007: Python version/implementation/compiler, OS system/release/machine, all package
versions, NumPy backend and SIMD, engine file digest, commit, dirty flag, lock digest, thread variables, detected pools
with thread counts, thread-limit probe, BLAS visibility):

| Status | Meaning | Mode | Criterion |
|--------|---------|------|-----------|
| `verified_same` | every compared field observed on both sides and equal | `same_environment_exact` | byte identity only |
| `unverified` | nothing observed differs, but some field is unknown | `exact_environment_unverified` | byte identity only |
| `different` | some field observed on both sides and unequal | `cross_environment_numeric` | criteria below |

Unknown metadata therefore never weakens exact replay to numeric agreement: tolerances apply only where a difference
was *observed*. `environment_match` is `pass`, `not_run` (unverified; the unknown fields are listed) or `fail`. In an
exact mode `numerical_agreement` is `not_applicable`; in the numeric mode `exact_replay` is `not_applicable`, and byte
identity is still reported (verdict `exact_match` whenever the bytes are identical). The report gives the worst status
over the runs, and each run keeps its own. Replay never claims `independent_reference` (always `not_run`).

**Per-run evidence (B3).** Every run record stores its run id, its environment status and its worker's own thread
observations (thread variables, pools with thread counts, probe, BLAS visibility, derived native-thread status). The
summary check `native_thread_policy` passes only if every worker verified one native thread for every library;
an invisible BLAS (Apple Accelerate) makes it `not_run`, never `pass`.

**Partial runs.** Every check about "every run" (`exact_replay`, `numerical_agreement`, `plan_identity`,
`partition_identity`) passes only if *all requested* runs completed and satisfied it. `partition_identity` reports
`runs_requested`, `runs_compared` and whether the partitions were equal in the compared runs, so agreement on a
successful subset is visible but is not a pass.

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

## Revision after technical review (2026-11-25, before acceptance)

Reproduced on head `3172cdc` before any change: a re-sealed passport recording `max_bytes` 1000 times the default was
dispatched to a worker with that budget (B1); equal unknown environment fields and unverified probes were treated as
equal, pool thread counts were not compared, and the per-run records of EXP-004 held no worker thread evidence (B3).
The policy check, argument bounds, three-status environment comparison, per-run evidence and partial-run rules above
are the corrections (regression tests in `tests/test_review_regressions.py`; fresh-process confirmation in EXP-006).
`numeric_agreement` criteria (`axiom.agreement/1`) are unchanged; differences beyond the binary64 range are reported
with `abs_diff = null` instead of infinity and fail every tolerance.

## Revision after the final technical review (2026-11-25, before acceptance)

Residual B1 (claimed cluster count sizing parent allocations; ADR 0007 revision) is resolved, and `replay` now applies
the trusted policy before `integrity_checks`. The final audit also changed, each after reproducing the failure on
`e8a9f57`: a timeout of `10**400` raised `OverflowError`; a worker response with status `ok` but missing fields, an
error given as text or unreadable result text raised in the parent instead of becoming a failed run (now `bad_output`,
or `crashed` for a non-zero exit); `compare_cores` raised `KeyError` while renumbering a record with an empty cluster
(now compared as recorded); shape differences in comparison reports copied whole lists (now sizes and differing keys);
the CLI wrote text in the locale encoding, so on a redirected Windows console (cp1252) a non-ASCII column name made
`analyze` fail with exit 2, and files got CRLF (now UTF-8 bytes with LF everywhere); the worker's `git status` could
refresh the repository index (now `--no-optional-locks`). A real worker killed by the timeout is tested to have exited
and to leave no workspace. The criteria `axiom.agreement/1` are unchanged; their boundaries are now tested to be
inclusive as documented. Confirmation: EXP-007 (30 of 30 fresh-process replays exact for all four families; bounded
work for claimed `k` up to 2^63 - 1 with no worker started).

Limitations recorded by the audit, not changed:
- The environment record in a passport is untrusted. Editing it to state a different environment can relax the
  criterion from byte identity to the documented tolerances (verdict `numeric_agreement`, never `exact_match`); it
  cannot hide a difference beyond the criteria.
- The worker inherits the parent's environment variables apart from the thread variables it sets. Variables that can
  change numerical paths (for example CPU-dispatch or BLAS core-type overrides) are neither cleared nor recorded.
- The relative criterion for the scaler scale uses the reference value as denominator, so swapping reference and
  observed can change the verdict exactly at the bound.
- Worker output is read into memory in full; its size is bounded by the result the engine can produce under the
  policy, not by a separate cap.

## Alternatives considered

- **In-process repetition only** (rejected: shares imports, caches and allocator state; Issue #2 already had it).
- **`multiprocessing` with fork** (rejected: inherits the parent's loaded libraries and thread pools; a new interpreter is
  the stronger isolation and works the same on Windows).
- **A universal epsilon** such as atol 1e-12 / rtol 1e-9 for all floats (proposed in the DR-001 note as a starting point;
  superseded by the lead decision; the per-field derivations above replace it).
- **ARI = 1 as the partition criterion** (rejected as authoritative: exact membership-set equality is stronger and has no
  floating point; ARI is reported).
- **Treating unavailable environment fields as equal** (rejected: unknown must never pass).
- **Treating unknown environment fields as "different"** (the first implementation; rejected in review: on a Mac with
  Accelerate every replay would have become cross-environment, silently replacing byte identity with tolerances).
- **Clamping recorded limits to the local policy** (rejected: a different analysis presented as the same replay).

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
