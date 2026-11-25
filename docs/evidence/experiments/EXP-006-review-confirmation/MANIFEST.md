# Experiment manifest: EXP-006 — confirmation of the PR #7 review corrections

- Issue / PR: Issue #3, PR #7 ([technical review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7171558913))
- Date: 2026-11-25
- Status: run (parts R, D1, D2 and X; every pre-registered criterion met; hypothesis H confirmed and the fix verified)
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `fd61eb6` before any run

## Part R: fresh-process replay on the revised code (development Mac)

### Provenance
- Commit `fd61eb62f09eb26143b2fb354110a3bbf0e883ae`; working tree dirty `False` (recorded by the script; untracked
  files are not counted). Engine file digest `4b904d4266ed24d7b80a51220d0a361deb1c5d2e2b33d10e7dd76d2e1e1fe19b`;
  `uv.lock` SHA-256 `0591cb5824e57d339d94ef2b2a4510cd1bfd688193190c29dcbc961ddc79cca7` (unchanged since EXP-004).
- Hardware and OS: Apple M5 (arm64, 10 logical CPUs), macOS 27.0.1 (Darwin 27.0.0). CPython 3.12.16; locked packages
  as in EXP-004. NumPy BLAS/LAPACK: Accelerate.
- Data and scenarios: identical to EXP-004 (same SHA-256 values).

### Execution
- `uv run python benchmarks/issue3_confirmation.py --runs 20 > docs/evidence/experiments/EXP-006-review-confirmation/results.json`
- Exit status 0; total wall time 362.7 s. No tracked file was edited during the run.
- `results.json`: per-run records (run id, outcome, environment status, thread variables, pools, probe, BLAS
  visibility, native-thread status, result hash, timings); no absolute paths (checked with grep).

### Results (measured only)

| Scenario | Verdict | Exact / runs | Distinct hashes (= reference, = in-process) | Distinct run ids | Mode | Worker compute s (median) |
|---|---|---|---|---|---|---|
| example/inspect | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0015 |
| example/relate_pearson | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0003 |
| example/relate_spearman | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0004 |
| example/compare | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0011 |
| example/cluster | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0251 |
| synthetic5000/inspect | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0270 |
| synthetic5000/relate_pearson | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0031 |
| synthetic5000/relate_spearman | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0069 |
| synthetic5000/compare | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0090 |
| synthetic5000/cluster | exact_match | 20 / 20 | 1 (yes, yes) | 20 | exact_environment_unverified | 0.0720 |

- **R1 met**: 200 of 200 replays `exact_match`. **R2 met**: 200 distinct stored run ids, none equal to the 10 reference
  run ids. **R4 met**: every run judged by byte identity (no run in the numeric mode).
- **R3 met**, per run from the stored records: thread variables all `"1"`; exactly one detected pool (libomp, OpenMP)
  at 1 thread; probe `verified_single_thread`; `blas_thread_control` reports Accelerate as not visible to threadpoolctl;
  so the derived native-thread status is `unverified` in all 200 runs and `native_thread_policy` is `not_run` for every
  scenario (prediction held). Environment status `unverified` in all 200 runs (fields: `numerical_backend` with
  Accelerate version "unknown", `threads.detected_pools` with libomp version null, `threads.blas_thread_control`);
  prediction held.
- K-means iteration counts (diagnostic) constant per scenario: 8 and 2, as in EXP-004.
- Parent wall time per replay over 200 runs: 1.035 / 1.482 / 2.539 s (min / median / max), mostly interpreter start-up.

### Mutations and malformed input (EXP-004 functions, unchanged): all criteria met
- M-stale: each old passport → `integrity_mismatch`, `dataset`, engine `plan_not_bound_to_dataset`, raw and logical both differ.
- M1: mean 0.478 ulp from the exact rational mean; max 93400.39; other columns unchanged. M2: Alpha 30→29, Bravo 30→31,
  others unchanged. M3: r 0.9062145213126868 → 0.8429040598884977, 3.2e-17 from the exact oracle. M4: dataset identity
  changed, result block identical, same partition. Every rebound replay `exact_match`.
- X1: analysis fails with `record_width`; old passport → `integrity_mismatch` (`dataset`).

### Adversarial checks (real worker processes): all criteria met
- **A1**: re-sealed passport with `max_bytes` x1000 → `VerifyError` `limits_exceed_policy` (fields `max_bytes`);
  workers started: 0.
- **A2**: one-ulp mean change plus unknown `packages.numpy` → environment `unverified`, mode
  `exact_environment_unverified`, verdict `numerical_discrepancy`, although the difference is `within_criteria`.
- **A3**: one assignment swapped between clusters 0 and 1 → all seven integrity checks `pass` (re-sealed, consistent);
  recomputation gives `numerical_discrepancy`, `same_partition` false.

## Part D: Windows checkout digests (CI run 45683518798, merge commit `6074227`, head `fd61eb6`)

### D1 (before any `.gitattributes` change): hypothesis H confirmed
| Runner | `core.autocrlf` | Hashed/fixture files differing from blob | Recorded engine digest | Recorded lock digest | Digests of committed blobs |
|---|---|---|---|---|---|
| `windows-latest` (job 139449983689) | `true` | 28 of 34, each **only** LF -> CRLF (`identical_after_crlf_to_lf` true; blobs contain no CRLF) | `4e4c5356…` | `93f27c7e…` | engine `4b904d42…`, lock `0591cb58…` |
| `macos-latest` (job 139449984489) | unset | 0 of 34 | `4b904d42…` | `0591cb58…` | engine `4b904d42…`, lock `0591cb58…` |

- The 28 files: all 21 engine `.py` files, `uv.lock` and the six example plan/profile JSON files (the CSV and NIST
  fixtures are `-text` and were unchanged). Git attributes for them were `text: unspecified`, `eol: unspecified`.
- The committed-blob digests on Windows equal the checkout digests on macOS and the digests recorded locally in part R.
- Decision (pre-registered): add `*.py text eol=lf` and `uv.lock text eol=lf` to `.gitattributes`. Raw byte hashing
  is unchanged. The example JSON files are parsed, never hashed as bytes, so they are left alone (minimal change).
- Reports: `ci_d1_windows_x64.json`, `ci_d1_macos_arm64.json` (copied verbatim from the job logs; no paths).

### D2 (after the change): criterion met
CI run [45685214061](https://github.com/sergthedev/axiom-local/actions/runs/45685214061), head `8310baa` (merge `efa20ba`):

| Runner | `core.autocrlf` | Hashed files (`.py`, `uv.lock`) differing from blob | Recorded engine digest | Recorded lock digest |
|---|---|---|---|---|
| `windows-latest` (job 139455889771) | `true` | 0 | `4b904d42…` | `0591cb58…` |
| `macos-latest` (job 139455889363) | unset | 0 | `4b904d42…` | `0591cb58…` |

- Both recorded digests are equal on the two runners, equal to the committed-blob digests and equal to the local
  part R values. In part X of this run the Windows comparison no longer lists `code.lock_sha256` as differing.
- The six example JSON files are still checked out with CRLF on Windows (not hashed; deliberately not covered).
- 384 tests passed on each runner (Windows 86.24 s, macOS 103.28 s); lint and format passed (101 files); part X again
  10 of 10 `exact_match` on both runners, reports complete.
- Reports: `ci_d2_windows_x64.json`, `ci_d2_macos_arm64.json`.

## Part X: revised code versus the EXP-005 reference passports (same CI run, report-only)

| Runner | Python / OS | NumPy BLAS | Report complete | Verdicts | Environment status |
|---|---|---|---|---|---|
| `windows-latest` | CPython 3.12.10 (MSC v.1943), Windows Server 2025 10.0.26100, AMD64 Family 25 Model 1, 4 CPUs | scipy-openblas 0.3.34 | yes (10 of 10 compared) | 10 `exact_match` | `different` |
| `macos-latest` | CPython 3.12.10 (Clang 13), macOS 26.7.2 arm64, 3 CPUs | Accelerate | yes (10 of 10 compared) | 10 `exact_match` | `different` |

- Every scenario byte-identical, 0 float fields different, partitions equal, iteration counts equal (8 and 2).
- Native-thread status: Windows `verified_single_thread` (two OpenBLAS pools and vcomp, all 1 thread, BLAS
  introspectable); macOS runner `unverified` (Accelerate).
- Differing fields (both runners): Python version and compiler, OS release, commit and engine digest (different code);
  Windows additionally OS system/machine, NumPy backend, lock digest (CRLF, D1) and pools.
- The Windows runner reported a different CPU model (Model 1) than in EXP-005 (Model 17): runner hardware varies.
- Reports: `ci_x_windows_x64.json`, `ci_x_macos_arm64.json`.
- CI for this run: 384 tests passed on each runner (Windows 77.88 s, macOS 90.12 s); lint and format passed (100 files).
  Implementer-observed from the job logs; not yet reviewer-verified.

## Independent reference
- Only M1 and M3 use independent oracles. Everything else here is repeatability, integrity or internal consistency.

## Limitations
- One machine and one run for part R; one CI run per runner for D1 and X. The worker environment equals the reference
  environment by construction in part R.
- Accelerate's thread use cannot be verified with threadpoolctl; this is now reported as `unverified` rather than hidden.
- The line-ending fix covers Git checkouts with this repository's attributes; other installation paths can still give
  different raw digests (conservatively reported as a different environment).
