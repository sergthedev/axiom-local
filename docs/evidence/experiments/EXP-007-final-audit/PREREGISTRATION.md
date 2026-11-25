# EXP-007 pre-registration: final review of PR #7 and final audit of AXIOM Verify

- Issue: #3; [final technical review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7172653455) of head
  `e8a9f57` (residual B1: a claimed cluster count `k` sized allocations in the parent process) and the owner's request
  for a final audit of AXIOM Verify.
- Written and committed **before** any run below, in the same commit as the code under test (recorded by the script as
  `git.commit`). The residual B1 finding and the further audit findings were reproduced on `e8a9f57` before any change,
  at a safe scale (the largest claimed `k` measured there was 2,000,000, about 80 MB); that reproduction is recorded in
  the development log and is not part of this experiment. The new regression tests were also run once against the
  unchanged `e8a9f57` sources (53 of 72 failed, as expected); that check is not part of this experiment either.
- EXP-004, EXP-005 and EXP-006 are not rerun or edited; their results stand as recorded for their commits.
- The script `benchmarks/issue3_final_audit.py` has not been executed before this commit. If it fails to run, the
  failure, the fix and the rerun are recorded; criteria are not changed.

Command (repository root, clean tree, after `uv sync --locked`):
`uv run python benchmarks/issue3_final_audit.py --runs 3 > docs/evidence/experiments/EXP-007-final-audit/results.json`

## Part S: fresh-process replay smoke, all four operation families (development Mac)

The ten scenarios of `benchmarks/issue3_common.py` (example and synthetic 5000-row data; Inspect, Relate Pearson and
Spearman, Compare, Cluster): one fresh-process analysis each, then `replay(..., runs=3)`.

| ID | Criterion |
|----|-----------|
| S1 | All 10 x 3 = 30 replays `exact_match`; `runs_completed` equals `runs_requested` in every scenario. |
| S2 | The 30 stored run ids are distinct, and distinct from the 10 reference run ids. |
| S3 | Every run stores its own environment status, thread variables (all `"1"`) and native-thread status. |
| S4 | Every scenario is judged by byte identity (`same_environment_exact` or `exact_environment_unverified`). |

Prediction (not a criterion): `exact_environment_unverified` and native-thread status `unverified` everywhere on this
Mac (Accelerate), as in EXP-006.

## Part B: no work sized by a claimed cluster count (development Mac)

For the example Cluster passport and `k` in {10^3, 10^6, 2^62, 2^63 - 1}, three re-sealed forgeries each: the result
claims `k`; the result and the plan claim `k`; the result, the plan and the recorded `max_k` claim `k`. For each, the
peak traced allocation (`tracemalloc`) of `integrity_checks`, of `compare_passports` with the forgery as reference and
as observed passport, and of `replay` (with counting wrappers around the worker launcher and the integrity checks).

| ID | Criterion |
|----|-----------|
| B1 | Every measured peak is below 1,000,000 bytes, and for each entry point and forgery variant the peak at `k` = 2^63 - 1 is at most twice the peak at `k` = 10^3 (no growth with the claimed count). |
| B2 | `integrity_checks` reports `result_consistency` failed for every forgery; `compare_passports` returns `integrity_mismatch` on both sides. |
| B3 | `replay` starts no worker process in any of the 12 cases; the forgeries that do not raise `max_k` give `integrity_mismatch`; the forgeries that raise `max_k` above the trusted policy raise `limits_exceed_policy` with **zero** calls of `integrity_checks`. |

## Part P: memory amplification when parsing passport text (development Mac, measurement)

A valid Inspect passport whose free-form `environment.numerical_backend` holds a padding value of one of six shapes
(a list of empty lists, of empty objects, of `0`, of seven-digit integers, of `1.5`; one long string), padded to 4 MiB
and 8 MiB of text. Measured: whether `loads_passport` accepts it, its peak traced allocation, and the ratio
peak / text bytes. Two legitimate synthetic 5000-row passports (Compare, Cluster) are measured for reference.

| ID | Criterion |
|----|-----------|
| P1 | Linearity: for each shape, the ratio at 8 MiB is within 15 % of the ratio at 4 MiB. |

Prediction (not a criterion): the largest ratio is below 40 (CPython object sizes: an empty dict or list costs tens of
bytes for two or three bytes of text). Values for the 64 MiB cap are only extrapolations from these measurements and
will be labelled as such. There is no acceptance threshold for the ratio itself: whether the cap or a pre-parse
structural count is needed is a reviewer decision, not an outcome of this experiment.

## Part C: both CI platforms at the final head (GitHub Actions)

| ID | Criterion |
|----|-----------|
| C1 | The locked test suite, lint and format pass on `macos-latest` and `windows-latest` at the final head. |
| C2 | The report-only cross-platform step (revised code versus the EXP-005 reference passports) reports `complete: true` with 10 compared scenarios on each runner, and the line-ending diagnostic reports no hashed file (`.py`, `uv.lock`) differing from its blob. |

Prediction (not a criterion): 10 `exact_match` per runner, environment status `different` (commit and engine digest
differ from the reference by construction).

## Not in scope

- The 200-run repetition of EXP-006 is not repeated (the final review asks for a representative smoke); the
  network-disabled rehearsal and the 100-run golden release target remain release gates.
- No tolerance, criterion or scenario is changed after seeing results; failures are recorded, not retried away.
- No independent human replication and no security certification is claimed.
