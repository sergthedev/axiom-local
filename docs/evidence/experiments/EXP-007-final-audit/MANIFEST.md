# Experiment manifest: EXP-007 — final review of PR #7 and final audit of AXIOM Verify

- Issue / PR: Issue #3, PR #7 ([final technical review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7172653455))
- Date: 2026-11-25
- Status: run (parts S, B and P on the development Mac: every pre-registered criterion met; part C: see below)
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `ab33266` before any run

## Provenance (parts S, B, P)
- Commit `ab33266ad7f1a2afad03483b0df22dba0dab99a2`; working tree dirty `False` (recorded by the script). Engine file
  digest `b0e2c5e49309311df440df3a67e400635df1da2a2ee41ce489108681d7d612dc`; `uv.lock` SHA-256
  `0591cb5824e57d339d94ef2b2a4510cd1bfd688193190c29dcbc961ddc79cca7` (unchanged since EXP-004).
- Hardware and OS: Apple M5 (arm64, 10 logical CPUs), macOS 27.0.1. CPython 3.12.16, locked packages. NumPy
  BLAS/LAPACK: Accelerate (version reported as "unknown").
- Data: the ten scenarios of `benchmarks/issue3_common.py` (same data as EXP-004 and EXP-006).

## Execution
- `uv sync --locked`, then
  `uv run python benchmarks/issue3_final_audit.py --runs 3 > docs/evidence/experiments/EXP-007-final-audit/results.json`
- First and only execution of the script; exit status 0; total wall time 68.6 s. No tracked file was edited during the
  run. `results.json` contains no absolute paths (checked with grep).

## Part S: fresh-process replay smoke, all four families — all criteria met

| Scenario | Verdict | Runs | Distinct run ids | Mode | Parent wall s (min-max) |
|---|---|---|---|---|---|
| example/inspect | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.729-0.791 |
| example/relate_pearson | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.762-0.843 |
| example/relate_spearman | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.682-0.736 |
| example/compare | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.683-0.694 |
| example/cluster | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.696-0.723 |
| synthetic5000/inspect | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.762-0.811 |
| synthetic5000/relate_pearson | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.707-0.727 |
| synthetic5000/relate_spearman | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.733-0.773 |
| synthetic5000/compare | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.752-0.792 |
| synthetic5000/cluster | exact_match | 3 / 3 | 3 | exact_environment_unverified | 0.740-0.766 |

- **S1 met**: 30 of 30 `exact_match`; `runs_completed` = `runs_requested` = 3 everywhere. **S2 met**: 30 distinct run ids,
  none equal to a reference run id. **S3 met**: every run stored its environment status (`unverified`), thread variables
  (all `"1"`) and native-thread status (`unverified`). **S4 met**: every scenario `exact_environment_unverified`.
- Checks per scenario: `passport_integrity`, `dataset_identity`, `plan_identity`, `exact_replay` pass (and
  `partition_identity` for Cluster); `environment_match` and `native_thread_policy` `not_run` (Accelerate);
  `numerical_agreement` `not_applicable`; `independent_reference` `not_run`. Prediction held.
- Parent wall time per replay 0.68-0.84 s, lower than EXP-006 (median 1.48 s) on the same machine; not investigated
  (timings include interpreter start-up and are not a criterion).

## Part B: no work sized by a claimed cluster count — all criteria met

Peak traced allocation in bytes (`tracemalloc`; Python allocations only, not process RSS). Baseline: `integrity_checks`
of the genuine example Cluster passport, 18,111 bytes.

| Claimed `k` | Forgery | integrity_checks | compare (forgery as reference) | compare (as observed) | replay | replay outcome | integrity calls in replay |
|---|---|---|---|---|---|---|---|
| 10^3 | result | 18,114 | 19,209 | 19,374 | 19,522 | integrity_mismatch | 1 |
| 10^3 | result + plan | 18,122 | 18,991 | 19,080 | 19,466 | integrity_mismatch | 1 |
| 10^3 | result + plan + max_k | 18,124 | 18,983 | 19,006 | 2,397 | limits_exceed_policy | 0 |
| 10^6 | result | 18,087 | 18,965 | 19,429 | 19,369 | integrity_mismatch | 1 |
| 10^6 | result + plan | 18,133 | 18,989 | 19,015 | 19,415 | integrity_mismatch | 1 |
| 10^6 | result + plan + max_k | 18,139 | 19,077 | 19,021 | 2,335 | limits_exceed_policy | 0 |
| 2^62 | result | 18,096 | 18,965 | 19,458 | 19,698 | integrity_mismatch | 1 |
| 2^62 | result + plan | 18,141 | 19,041 | 19,084 | 19,485 | integrity_mismatch | 1 |
| 2^62 | result + plan + max_k | 18,169 | 18,821 | 19,368 | 2,397 | limits_exceed_policy | 0 |
| 2^63 - 1 | result | 18,096 | 19,130 | 19,251 | 19,378 | integrity_mismatch | 1 |
| 2^63 - 1 | result + plan | 18,141 | 18,991 | 19,023 | 19,423 | integrity_mismatch | 1 |
| 2^63 - 1 | result + plan + max_k | 18,169 | 19,448 | 18,999 | 2,335 | limits_exceed_policy | 0 |

- **B1 met**: largest peak 19,698 bytes (< 1,000,000); no entry point grows with `k` (largest ratio of the peak at 2^63 - 1
  to the peak at 10^3: 1.024, a difference of a few hundred bytes). For comparison, the unchanged `e8a9f57` code
  reached about 40 MB for `k` = 10^6 and 80 MB for 2 x 10^6 (reproduction in the development log, not part of this
  experiment).
- **B2 met**: `result_consistency` failed for all 12 forgeries; `compare_passports` gave `integrity_mismatch` on both
  sides in all 24 calls.
- **B3 met**: 0 worker processes started in 12 replays; the 8 forgeries not raising `max_k` gave `integrity_mismatch`;
  the 4 raising `max_k` raised `limits_exceed_policy` with 0 calls of `integrity_checks`.

## Part P: memory amplification of parsing passport text — criterion met; the ratio is a measured limitation

Peak traced allocation of `loads_passport` divided by the text size (every document was accepted as a valid passport,
since the padding sits in a free-form field):

| Padding shape | 4 MiB | 8 MiB | Traced wall s at 8 MiB |
|---|---|---|---|
| list of empty objects `{}` | 25.16 | 25.22 | 7.15 |
| list of empty lists `[]` | 22.50 | 22.56 | 5.86 |
| list of `1.5` | 9.01 | 9.04 | 2.70 |
| list of seven-digit integers | 5.62 | 5.51 | 1.38 |
| list of `0` | 5.09 | 5.14 | 5.84 |
| one long string | 2.00 | 2.00 | 0.03 |
| *legitimate synthetic5000/compare (10,078 bytes)* | 3.80 (one size) | | |
| *legitimate synthetic5000/cluster (135,323 bytes)* | 2.83 (one size) | | |

- **P1 met**: for every shape the ratio at 8 MiB is within 2.2 % of the ratio at 4 MiB (linear in the text size).
- Prediction held: largest ratio 25.2 (< 40).
- **Extrapolation, not measured**: at the 64 MiB text cap the worst measured shape would need about 25.2 x 64 MiB, roughly
  1.6 GiB of traced allocations, while the document is parsed and validated. This is a linear bound set by the cap, not
  by any count stated in the document, and it is reported as an open limitation for the reviewer: lowering the cap or
  counting containers before parsing would change it; neither was implemented (no new limit without a decision).
- Wall times are with `tracemalloc` active and therefore slower than untraced parsing.

## Part C: both CI platforms at the final head

Pending until the final head has run on GitHub Actions; recorded below once observed.

## Independent reference
- None. Parts S and B are repeatability and resource checks; part P is a measurement. No result here says anything about
  numerical correctness (see EXP-001 and EXP-002 for that evidence).

## Limitations
- One machine and one run for parts S, B and P; three replays per scenario (a smoke test, not the release gate).
- `tracemalloc` measures Python-level allocations only; it does not include memory allocated by C libraries outside
  Python's allocator or process overhead, and it slows the traced code.
- The bounded-claims part covers the cluster count `k`, the only claimed count that sized work before this change
  (code review of `result_problems`, `compare_cores`, `compare_environments` and the schema validator); it does not
  prove the absence of other super-linear paths.
