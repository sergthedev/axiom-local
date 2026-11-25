# EXP-002 pre-registration: descriptive-statistics confirmation on held-back data

Status: **frozen before any EXP-002 data were generated or any script output was seen.** This file and the
script `benchmarks/issue2_quartile_confirmation.py` are committed first; results are committed afterwards in a
separate commit. [EXP-001](../EXP-001-issue2-numerical-validation/MANIFEST.md) is preserved unchanged,
including the quartile criterion failure and its post-hoc revision, which was *exploratory* because it was
revised on the same data (technical review of PR #6, blocker 2).

## Why this experiment

1. The first quartile implementation (NumPy `linear`) was found to report a false IQR overflow for `[-M, M]`
   (review blocker 1). Quartiles, the median and the mean were then reimplemented as exact rationals rounded once.
2. The revised EXP-001 criterion (<= 1 ulp at the operand scale) was chosen after seeing the failure on the same
   data, so it needs confirmation on data that were not used to choose it.

## Implementations under test

- **P (production):** `axiom.kernels.describe` — median, q1, q3 and IQR as exact rationals rounded once; mean from
  `statistics.mean` plus the underflow check; classification `defined | overflow | underflow`.
- **N (record only):** NumPy `numpy.quantile(..., method="linear")` as used by the first implementation, evaluated on
  the same inputs so that the old and revised metrics are reported side by side. N is no longer in production.

## Oracle (independent of the production code path)

`tests/oracles.py`: `exact_quantile_type7_weighted`, `exact_median_weighted` and exact mean via `fractions.Fraction`.
The quantile oracle uses a different formulation from production (integer position and weighted average
`(x_lo*(4-r) + x_hi*r)/4`, versus `a + frac*(b-a)`), shares only the *definition* of type 7 and
never imports `axiom`. `classify_exact` rounds the exact value once and labels it `defined`, `overflow` (exact value
beyond binary64) or `underflow` (exact value nonzero, rounds to zero). Oracle correctness was not independently
reviewed; agreement shows implementation consistency for this definition.

## Held-back data (seeds distinct from every earlier seed; denominators fixed now)

| Family | Seed | Sets | Description |
|---|---|---|---|
| F1 gaussian-scales | 7300001 | 300 | n in 2..60, scale 10^-8..10^8, offsets up to 1e6 x scale (same family as EXP-001, new seed) |
| F2 cancellation | 7300002 | 300 | n in 2..9, values of both signs of magnitude s(1 + e), |e| <= 1e-3, s = 10^-200..10^200, so interpolated quartiles are far smaller than the operands |
| F3 opposite-signs-wide | 7300003 | 300 | n in 2..30, random sign, magnitude 10^U(-300, 300) |
| F4 extreme-magnitude | 7300004 | 300 | n in 2..12, values +/- u*M with u in [0.5, 1] and exact +/-M, M = max float, mixed with 0 and small values |
| F5 subnormal | 7300005 | 300 | n in 2..20, k * 2^-1074 for integers k in -50..50, mixed with 0 and values near the normal minimum |
| F6 ties-and-large-n | 7300006 | 100 | n in 200..2000 drawn from 20 distinct values (many duplicates) at a random scale |

Each set is evaluated for five statistics: q1, q3, iqr, median, mean (production) and q1, q3, iqr (N).

## Criteria (frozen)

**C1 — production exactness (pass/fail, primary).** For every set and every statistic of P, the production value equals
the oracle's correctly rounded value *bit for bit* (so the error is <= 0.5 ulp by construction), and the production
classification (`defined`, `overflow_exceeds_float64`, `underflow_to_zero`) equals the oracle classification. Required
mismatches: **0** out of the stated denominators. Any mismatch is a failure to be reported, not tuned away.

**C2 — NumPy first-order bound (informational, derived before measurement).** For `a + t*(b - a)` with t a multiple
of 1/4 in [0, 1] and no intermediate overflow, three floating-point operations round (the subtraction, the product, the
addition), each with relative error at most u = 2^-53 of the value it produces. With M = max(|a|, |b|) (so |b - a| <= 2M)
and t <= 3/4: |error| <= t*u*|b - a| (from the rounded subtraction) + u*t*|b - a| (rounded product) + u*|result|
<= 2*(3/4)*u*2M + u*M = 4uM. Since u*M < ulp(M) for every M, the derived first-order bound is **4 ulp at the operand scale**
(`math.ulp(max(|a|, |b|, |result|))`). This is looser than the EXP-001 revised criterion (<= 1), which was **not** derived
this way; the observed 0.5 in EXP-001 was a measurement, not a guarantee. Reported with denominators: N result-ulp > 2
(the original criterion), operand-ulp > 1 (the EXP-001 revision), operand-ulp > 4 (the derived bound), and N non-finite
outputs where the oracle value is finite (intermediate overflow). There is no pass/fail on N because N is not in
production; a violation of the derived bound would indicate an error in this derivation and is reported as such.

**C3 — coverage.** The report states, per family and statistic, how many outcomes were `defined`, `overflow`
and `underflow` in the oracle so that a family that never exercises its purpose is visible.

## Recorded for each family

Sets, evaluations, C1 value mismatches and classification mismatches (with up to 5 example inputs as hex floats), C3 counts, and
for N: finite/non-finite counts, maximum error in result-ulp and operand-scale ulp, and the violation counts above.

## Execution

From the repository root, after `uv sync --locked`: `uv run python benchmarks/issue2_quartile_confirmation.py > results.json`.
The commit of the script and this file is recorded in the results; results are committed separately, after the run.
