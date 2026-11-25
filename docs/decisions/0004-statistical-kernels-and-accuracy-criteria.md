# ADR 0004: Inspect, Relate and Compare kernels and accuracy criteria

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-25
- Issue / PR: Issue #2 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230), items 2-4 and 6)

## Context

The lead decision prefers tested library kernels over custom ones, requires explicit undefined results, and
requires accuracy criteria fixed before measurement. Determinism does not imply correctness (AGENTS.md).

## Decision

Implemented in `src/axiom/kernels.py` and `src/axiom/ops_*.py`.

**Inspect and Compare (shared kernel `describe`).**
`n_total`, `n_valid`, `n_missing`; `min`, `max`; `mean`, `median`, sample variance and SD (ddof=1) from the standard
library `statistics` module, which accumulates exactly (rational arithmetic) and rounds once (REF-007, local source
inspected); quartiles `q1`, `q3` type 7 through `numpy.quantile(..., method="linear")` (REF-009); `iqr = q3 - q1`.
No outlier flags or MAD (deferred). Median of an even count is the exactly rounded midpoint, so
`(a + b) / 2` never overflows. Results that cannot be represented are `null` with a reason
(`overflow_exceeds_float64`, `underflow_to_zero`, `requires_n_ge_2`, `no_valid_values`); a variance that rounds to
`0.0` although the data are not constant is reported as `underflow_to_zero`, never as a false zero.
Compare uses exact text keys, sorts them by Unicode code point, excludes and counts missing group keys, keeps
duplicates, and reports per-group total/valid/missing. No weights, ratios or "average of averages".

**Relate.** Pearson uses `statistics.correlation` (linear); Spearman uses `statistics.correlation(method="ranked")`
(average ranks for ties, REF-007). Candidate oracle smoke comparison (see EXP-001) and a source inspection of the
installed 3.12.16 `statistics.correlation` showed that the stdlib function **overflows silently** for magnitudes near
1e200 (returns `0.0`) and **falsely reports "constant"** near 1e-200. A guard therefore scales each variable by an
exact power of two so that max|v| lies in [0.5, 1) before calling it (Pearson is scale invariant). Policy:

| Item | Value | Derivation |
|------|-------|------------|
| Minimum complete pairs | 3 | Project policy (lead decision). |
| Near-constant (Pearson only) | undefined if `range <= 2^-40 * max|v|` after scaling | A priori: binary64 representation error of the inputs is about 2^-53 relative, so at relative spread 2^-40 about 2^-13 (~4 digits) of the spread remains reliable. SciPy uses a comparable but different criterion (norm(x-mean) < 1e-13*abs(mean), REF-010); this one is documented, tested with adversarial fixtures, and a sweep is recorded in EXP-001. |
| Range clip | `|r| <= 1 + 1e-9` is clipped to +/-1, more is a `ComputationError` | Rounding can exceed 1 by a few ulp; a larger excess means a defect, not rounding. Provisional. |
| Small-sample warning | n < 10 | Project convention, not a statistical threshold. |
| Pairwise policy | Single pair, complete rows only; exclusions are counted by cause and listed by ordinal | Lead decision. A correlation *matrix* is deferred (P1); pairwise matrices may not be positive semidefinite (REF-015). |

Not provided, by decision: p-values, confidence intervals, strength labels, direction categories, causal language.
Warnings carry `not_causal` and `linear_only` / `monotonic_only`.

**Accuracy criteria (fixed before measurement; results in EXP-001).**

| Quantity | Reference | Criterion |
|----------|-----------|-----------|
| mean, variance, SD | exact value (Fraction / 60-digit Decimal) of the parsed binary64 inputs | <= 1 ulp |
| type-7 quartiles | exact interpolation of the same inputs | first registered: <= 2 ulp of the **result**; **failed** (4.0 ulp) on seeded random data; revised to <= 1 ulp of max(abs(a), abs(b), abs(result)) of the interpolation operands |
| NIST certified mean / SD | certified decimals | mean error <= u*max|x| + ulp(certified); SD error <= u*max|x|*sqrt(n/(n-1)) + certified precision + ulp, with u = 2^-53 (decimal certified values are not exactly representable, so the bound is derived from input conversion) |
| Pearson / Spearman | exact rational oracle of the same inputs | absolute error <= 1e-12 on seeded random pairs with offset/spread up to 1e6, n <= 60 |

The quartile revision follows from the formula `a + t*(b-a)`: when the operands cancel, the result is much smaller
than the operands and one operand ulp is many result ulps. The failing case (a=-1.80e-8, b=5.14e-9, result -6.58e-10) had
an error of 0.125 operand-scale ulp. The revision was derived from that analysis, not tuned to the first value that
passes, and both versions are recorded in the development log. NumPy's lerp stays within 0.5 operand-scale ulp on
all 600 measured quartiles.

## Alternatives considered

- Custom `fsum`/two-pass kernels for mean and SD (not needed: the stdlib's exact accumulation met the criteria; measured cost in EXP-001).
- Calling SciPy `pearsonr`/`spearmanr` (not needed: no p-values are produced and the stdlib kernels passed; it would add a direct dependency).
- Exact-Fraction quartiles (not used: the lead decision names NumPy `linear`; exact quartiles would also remove the independence of the test oracle).

## Tradeoffs and consequences

- The near-constant rule is conservative with respect to *algorithmic* error: EXP-001 shows the stdlib kernel is
  accurate to ~1e-16 against the exact value of the *same float inputs* even in the flagged range 2^-48..2^-44. The rule
  reflects uncertainty from input representation, not an algorithm failure; it may reject data that a user knows are exact.
- `statistics.mean/variance/stdev` each re-accumulate exactly: Inspect on 1,000,000 rows took 2.7 s and 4.4 s in two runs on the measured machine (EXP-001 records the second).
- Correlation overflow/underflow protection relies on exact power-of-two scaling, which is exact except for entries that
  underflow to subnormals during down-scaling (negligible relative to max|v|).

## Unvalidated assumptions

- The criteria hold for data resembling the seeded generators; they are not general guarantees on ill-conditioned input.
- Behaviour on Windows x64 numerics is not measured here (CI runs the same tests; no comparison is claimed).

## References

REF-007, REF-009, REF-010, REF-011, REF-015, REF-016, REF-017, REF-018
