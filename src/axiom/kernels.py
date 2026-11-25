"""Numerical kernels (ADR 0004).

Descriptive statistics use the Python 3.12 standard library (exact rational accumulation). The
median and the type-7 quartiles are computed exactly (rational arithmetic) and rounded once; NumPy's
``method='linear'`` is only a cross-check in tests (ADR 0004, amendment). Correlation uses
``statistics.correlation`` behind explicit range and conditioning guards. Undefined results are
``None`` with an explicit reason, never a silent value.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from fractions import Fraction

from .errors import ComputationError

# Relative spread (range / max|value|) below this means that binary64 representation error of the
# inputs alone (about 2^-53 relative) exceeds 2^-13 of the spread, i.e. fewer than ~4 reliable
# digits remain in the standardized values. Chosen a priori from that argument (ADR 0004).
NEAR_CONSTANT_RELATIVE_SPREAD = 2.0**-40

# |r| may exceed 1 by at most this much through rounding before it is clipped; more is an error.
CORRELATION_CLIP_TOLERANCE = 1e-9

MIN_PAIRS = 3  # project policy: minimum n for a correlation coefficient

DESCRIPTIVE_KEYS = (
    "min",
    "max",
    "mean",
    "median",
    "sample_variance",
    "sample_sd",
    "q1",
    "q3",
    "iqr",
)


def exact_midpoint(a: float, b: float) -> Fraction:
    """Exact (a + b) / 2 as a rational, so no intermediate overflow can occur."""
    return (Fraction(a) + Fraction(b)) / 2


def round_exact(x: Fraction) -> tuple[float | None, str | None]:
    """Round an exact rational to binary64 once, classifying the outcome.

    Returns ``(value, None)`` or ``(None, reason)``. ``overflow_exceeds_float64`` means the *exact*
    result is not representable (a true range limit, not an intermediate failure);
    ``underflow_to_zero`` means the exact result is nonzero but rounds to zero. A false zero is
    never reported as a value.
    """
    try:
        value = float(x)
    except OverflowError:
        return None, "overflow_exceeds_float64"
    if not math.isfinite(value):
        return None, "overflow_exceeds_float64"
    if value == 0.0 and x != 0:
        return None, "underflow_to_zero"
    return value + 0.0, None


def mean_checked(values: Sequence[float]) -> tuple[float | None, str | None]:
    """Exact mean rounded once; ``(None, "underflow_to_zero")`` if nonzero but rounded to zero."""
    mean = statistics.mean(values)
    if mean == 0.0 and any(v != 0.0 for v in values) and sum(map(Fraction, values)) != 0:
        return None, "underflow_to_zero"
    return mean + 0.0, None


def quantile_type7(sorted_values: Sequence[float], numerator: int, denominator: int) -> Fraction:
    """Exact type-7 (Hyndman-Fan, NumPy ``linear``) quantile at p = numerator / denominator.

    h = (n - 1) p; result = x[floor(h)] + (h - floor(h)) * (x[floor(h) + 1] - x[floor(h)]), all in
    exact rational arithmetic. The caller rounds once with :func:`round_exact`.
    """
    n = len(sorted_values)
    pos = Fraction((n - 1) * numerator, denominator)
    lo = pos.numerator // pos.denominator
    hi = min(lo + 1, n - 1)
    a = Fraction(sorted_values[lo])
    return a + (pos - lo) * (Fraction(sorted_values[hi]) - a)


def describe(values: Sequence[float]) -> tuple[dict[str, float | None], dict[str, str]]:
    """Descriptive statistics of finite floats: ``(stats, undefined_reasons)``.

    Sample variance and standard deviation use ddof=1. Quartiles are type 7 and, like the median,
    are computed exactly and rounded once. Semantics for unrepresentable results: a statistic whose
    exact value overflows binary64, or is nonzero but rounds to zero, is ``None`` with a reason.
    """
    stats: dict[str, float | None] = dict.fromkeys(DESCRIPTIVE_KEYS)
    undefined: dict[str, str] = {}
    n = len(values)
    if n == 0:
        return stats, dict.fromkeys(DESCRIPTIVE_KEYS, "no_valid_values")

    ordered = sorted(values)
    lo, hi = ordered[0], ordered[-1]
    stats["min"], stats["max"] = lo + 0.0, hi + 0.0

    mean, mean_reason = mean_checked(values)
    if mean_reason is None:
        stats["mean"] = mean
    else:
        undefined["mean"] = mean_reason

    mid = n // 2
    median_exact = (
        Fraction(ordered[mid]) if n % 2 else exact_midpoint(ordered[mid - 1], ordered[mid])
    )
    exact: dict[str, Fraction] = {"median": median_exact}
    exact["q1"] = quantile_type7(ordered, 1, 4)
    exact["q3"] = quantile_type7(ordered, 3, 4)
    exact["iqr"] = exact["q3"] - exact["q1"]
    for key, fraction in exact.items():
        value, reason = round_exact(fraction)
        if reason is None:
            stats[key] = value
        else:
            undefined[key] = reason

    if n < 2:
        undefined["sample_variance"] = undefined["sample_sd"] = "requires_n_ge_2"
    else:
        for key, fn in (("sample_variance", statistics.variance), ("sample_sd", statistics.stdev)):
            try:
                result = fn(values)
            except OverflowError:
                undefined[key] = "overflow_exceeds_float64"
                continue
            if not math.isfinite(result):
                undefined[key] = "overflow_exceeds_float64"
            elif result == 0.0 and lo != hi:
                undefined[key] = "underflow_to_zero"
            else:
                stats[key] = result + 0.0
    return stats, undefined


def _constant_reason(values: Sequence[float]) -> str | None:
    return "constant_input" if min(values) == max(values) else None


def _scale_by_power_of_two(values: Sequence[float]) -> list[float]:
    """Scale so that max|v| is in [0.5, 1); exact barring underflow of negligible entries."""
    exponent = math.frexp(max(abs(v) for v in values))[1]
    return [math.ldexp(v, -exponent) for v in values]


def _finish_correlation(r: float) -> float:
    if not math.isfinite(r) or abs(r) > 1.0 + CORRELATION_CLIP_TOLERANCE:
        raise ComputationError("correlation_postcondition", "coefficient outside [-1, 1]", value=r)
    return max(-1.0, min(1.0, r)) + 0.0


def correlate(
    x: Sequence[float], y: Sequence[float], method: str
) -> tuple[float | None, tuple[str, list[str]] | None]:
    """Pearson or Spearman coefficient of complete finite pairs.

    Returns ``(coefficient, None)`` or ``(None, (reason_code, ["x"|"y", ...]))``.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < MIN_PAIRS:
        return None, ("insufficient_data", [])
    constant = [name for name, v in (("x", x), ("y", y)) if _constant_reason(v)]
    if constant:
        return None, ("constant_input", constant)
    if method == "spearman":
        return _finish_correlation(statistics.correlation(x, y, method="ranked")), None
    if method != "pearson":
        raise ValueError(f"unknown method {method!r}")
    xs, ys = _scale_by_power_of_two(x), _scale_by_power_of_two(y)
    near = [
        name
        for name, v in (("x", xs), ("y", ys))
        if max(v) - min(v) <= NEAR_CONSTANT_RELATIVE_SPREAD * max(abs(min(v)), abs(max(v)))
    ]
    if near:
        return None, ("near_constant_input", near)
    return _finish_correlation(statistics.correlation(xs, ys)), None
