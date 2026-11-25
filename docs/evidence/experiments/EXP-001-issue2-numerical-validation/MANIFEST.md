# Experiment manifest: EXP-001 — Issue #2 numerical validation and repeatability

- Issue / PR: Issue #2 (pull request linked from the issue)
- Date: 2026-11-25
- Status: run (all scripted measurements completed with exit status 0); results are measured on one machine and have not been reproduced by anyone else

## Provenance
- Commit (full SHA) that the scripts ran on: `f92117a4f984a029352ded9ef11cbad447371beb`; working tree dirty at start: `False`. Candidate-smoke script: commit `f92117a4f984a029352ded9ef11cbad447371beb`, dirty `False`. Later commits only add evidence/documentation files (verify with `git diff --stat f92117a4f984a029352ded9ef11cbad447371beb HEAD -- src tests examples benchmarks`).
- Hardware and OS: Apple Silicon `arm64` (Apple M5, 10 logical CPUs, 16 GiB memory), `macOS-27.0.1-arm64-arm-64bit`
- Python and packages (locked in `uv.lock`; no deviations): CPython 3.12.16, numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, pydantic 2.14.4, threadpoolctl 3.8.0
- Dataset provenance: NIST StRD PiDigits, NumAcc1-4 (verbatim, hashes in `tests/data/nist/MANIFEST.json`, license basis in `THIRD_PARTY_NOTICES.md`); synthetic seeded generators inside the scripts (seed 20261125 and others stated per section); `examples/synthetic_regions.csv` (SHA-256 `4f39a4f3cac2395df392ad93d56e59ecda3fc49636dde736c0f0aa9f946fd4ad`, fully synthetic). No real personal data. No training/validation/test split applies (no model).
- Plan/config files: `examples/plans/*.json`; seeds: 20261125 (random oracle data), K-means seed 42

## Execution
- Exact commands (from the repository root, after `uv sync --locked`):
  - `uv run pytest -q` (206 tests)
  - `uv run ruff check .` and `uv run ruff format --check .`
  - `uv run python benchmarks/issue2_candidate_smoke.py > candidate_smoke.json`
  - `uv run python benchmarks/issue2_evidence.py > results.json`
- Exit codes: `uv sync --locked` 0; `uv run pytest -q` 0 (206 passed); `ruff check` 0; `ruff format --check` 0; candidate-smoke script 0; evidence script 0.
- Sanitized logs: `results.json` and `candidate_smoke.json` in this directory (no absolute paths; checked with grep). Test output is summarized, not stored.

## Results (measured only)

### 1. NIST StRD, Inspect operation (all five datasets evaluated)

Criteria fixed before measurement (ADR 0004): (B) versus the exact value of the *parsed binary64 inputs*, mean and SD within 1 ulp; (A) versus the NIST *certified decimals*, bound derived from input conversion (see ADR 0004). "ulp error" is the absolute error divided by ulp of the result.

| Dataset | n | Certified mean / SD | AXIOM mean error vs certified (abs) | AXIOM SD error vs certified (abs) | Input-conversion term of the SD bound (abs) | ulp error vs binary64 oracle (mean / SD) | binary64 oracle vs certified SD (abs, input conversion alone) |
|---|---|---|---|---|---|---|---|
| NumAcc1 | 3 | 10000002 / 1 | 0 | 0 | 1.36e-09 | 0 / 0 | 0 |
| NumAcc2 | 1001 | 1.2 / 0.1 | 4.44e-17 | 2.22e-17 | 1.44e-16 | 0.5 / 4.44e-18 | 2.22e-17 |
| NumAcc3 | 1001 | 1000000.2 / 0.1 | 4.66e-11 | 3.49e-11 | 1.11e-10 | 0.5 / 1.22e-06 | 3.49e-11 |
| NumAcc4 | 1001 | 10000000.2 / 0.1 | 7.45e-10 | 5.59e-10 | 1.11e-09 | 0.5 / 0.000312 | 5.59e-10 |
| PiDigits | 5000 | 4.53480000000000 / 2.86733906028871 | 2.8e-16 | 1.89e-15 | 9.99e-16 | 0.315 / 0.08 | 1.93e-15 |

The full SD bound adds the certified value's own precision (half a unit in its last given digit: 0 for the values marked exact by NIST, 5e-15 for PiDigits) and one ulp to the input-conversion term shown; for PiDigits the measured 1.89e-15 exceeds the conversion term alone (9.99e-16) but is within the full bound (about 6e-15). The mean bound has the same structure.

All ten comparisons met their pre-registered criteria. For NumAcc3 and NumAcc4 the difference from the certified SD (3.5e-11 and 5.6e-10) is entirely explained by decimal-to-binary64 conversion of the inputs (the exact oracle of the same binary64 values shows the same difference); AXIOM agrees with that oracle to 1.2e-6 and 3.1e-4 ulp. The certified decimals are not exactly representable, so exact equality with them is not demanded or expected. This is a comparison with NIST certified values, not a NIST certification of AXIOM (REF-017).

### 2. Inspect versus exact rational oracles (seeded random data)
- 300 datasets, seed 20261125, n in 2..60, scales 1e-8..1e8, offsets up to 1e6 times scale.
- Worst error in ulp of the result: mean 0.5, sample variance 0.499, sample SD 0.5 (criterion: <= 1 ulp: met).
- Quartiles: worst error 4 ulp of the **result** versus the first pre-registered criterion of <= 2 ulp: **not met** (cancellation case, see below); worst error 0.5 ulp of the interpolation operand scale versus the revised criterion of <= 1: met.
- Recorded failure: the first quartile criterion failed on this run's data (4.0 ulp of result). Analysis of the failing case (a=-1.80e-8, b=5.14e-9, result -6.58e-10) showed 0.125 operand-scale ulp, i.e. cancellation in `a + t*(b-a)`. The criterion was replaced by an operand-scaled one derived from the formula; both are in ADR 0004 and the development log. The revision was made after seeing the failure and is therefore post hoc, although derived from the interpolation formula rather than tuned to a pass.

### 3. Relate versus exact rational oracles (seeded random pairs)
- 300 pairs, seed 20261125, offset/spread up to 1e6, 30 percent with induced ties. Defined: Pearson 300, Spearman 300; Pearson undefined 0.
- Worst absolute error: Pearson 1.86e-16, Spearman 1.11e-16 (criterion <= 1e-12: met).

### 4. Candidate-kernel smoke comparison (why the kernels were chosen)
- Same 300 pairs: worst absolute error versus exact oracle: AXIOM guarded Pearson 2.24e-16, raw stdlib Pearson 2.24e-16, SciPy `pearsonr` 3.6e-16; AXIOM Spearman 5.53e-17, SciPy `spearmanr` 2.72e-16. No candidate was meaningfully more accurate on this data.
- Extreme scale (x = 1e-200 * [1, 2, 4], y = [1, 2, 3.5]; exact r = 0.997176464952738): raw stdlib raised "constant"; at 1e+200 raw stdlib returned 0.0 (silently wrong); SciPy and the AXIOM guard returned 0.99717646495273x. This is why the guard exists.
- Descriptive candidates on the five NIST datasets (absolute error versus exact oracle of the binary64 inputs): `statistics.stdev` and `numpy.std(ddof=1)` agreed with the oracle to <= 4e-17 on every dataset; a naive one-pass SD had an absolute error of 0.1 on NumAcc4 and 3.5e-11 on NumAcc3. `numpy.mean` and naive mean equal `statistics.mean` up to final rounding (the exact mean is rounded to binary64 in all cases). The standard library was kept because it accumulates exactly and does not overflow; it was not more accurate than NumPy's two-pass SD on these datasets.

### 5. Pearson near-constant criterion sweep (x = 1 + spread*[0,1,3,4,6,7], random y)

| log2(relative spread) | Flagged by AXIOM | Reason / error versus exact oracle of the same float inputs |
|---|---|---|
| -60 | yes | constant_input; stdlib: at least one of the inputs is constant |
| -52 | yes | near_constant_input; forced stdlib error 0.0174 |
| -48 | yes | near_constant_input; forced stdlib error 1.03e-16 |
| -44 | yes | near_constant_input; forced stdlib error 6.9e-17 |
| -41 | no | abs error 3.55e-17 |
| -40 | no | abs error 5.38e-17 |
| -39 | no | abs error 8.22e-17 |
| -36 | no | abs error 9.3e-17 |
| -30 | no | abs error 4.38e-17 |
| -20 | no | abs error 1.15e-17 |
| -10 | no | abs error 4.66e-17 |

Finding: the criterion (relative spread <= 2^-40) is conservative with respect to *algorithmic* error: the standard-library kernel was accurate to ~1e-16 against the exact value of the same float inputs at 2^-48 and 2^-44, and only degraded at 2^-52 (error 1.7e-2). The rule is justified by representation uncertainty of the inputs (ADR 0004), not by an algorithm failure; it may reject data that a user knows to be exact.

### 6. K-means thread determinism (raw scikit-learn, 60,000 x 8 synthetic points, k=8, n_init=4, seed 42)
- 20 fits each. Distinct outputs (centers bytes, n_iter, inertia): default threads **9**; single thread (`threadpool_limits(1)`) **1** (10 logical CPUs). This is why the engine fits single-threaded. An earlier run of the same experiment (commit `eb111b7`) gave 7 distinct outputs for default threads and 1 for a single thread. One machine; it says nothing about other machines.

### 7. Single-run timings (indicative only; not performance claims)

Timings vary between runs: an earlier run of the same script (commit `eb111b7`) took 2.7 s (inspect) and 3.1 s (Spearman) for 1,000,000 rows versus 4.4 s and 5.7 s below.

| rows | CSV bytes | load (s) | inspect (s) | Pearson (s) | Spearman (s) | cluster k=4 (s) |
|---|---|---|---|---|---|---|
| 10,000 | 190,050 | 0.023 | 0.026 | 0.005 | 0.014 | 0.053 |
| 100,000 | 1,899,952 | 0.328 | 0.358 | 0.057 | 0.226 | 0.59 |
| 1,000,000 | 19,000,751 | 3.589 | 4.395 | 0.981 | 5.652 | not run |

### 8. Fresh-process replay (same data, same plan, no cache)
- 20 fresh Python processes, each running 5 example plans on `examples/synthetic_regions.csv`: **1 distinct output** (all result hashes identical across processes). Result SHA-256 values are in `results.json`. This is repeatability on one machine, not correctness and not cross-platform agreement. Replay infrastructure, run passports and the macOS/Windows comparison belong to Issue #3.

## Limitations and failures
- One machine (Apple M5, macOS 27.0.1, arm64), one run of each script; not reproduced by another person or platform. Windows numerical behavior is not measured here (CI only runs the test suite).
- The exact-arithmetic oracles and the tests were written by the same implementer as the engine. The oracles use different code (`fractions`/`decimal`) from the engine, but the standard library's `statistics` also uses exact arithmetic, so the *method* for mean/variance is shared; agreement shows implementation consistency, not an independent method.
- Accuracy criteria are provisional, derived a priori, and cover the tested data families only; they are not guarantees for ill-conditioned input.
- One pre-registered criterion failed (quartiles, result-relative ulp) and was revised post hoc with a recorded derivation; see section 2.
- Timings are single runs and include Python-level exact arithmetic; no optimization was attempted.
- No claim of NIST certification, global optimality of clusters, or cross-platform bitwise reproducibility is made.
