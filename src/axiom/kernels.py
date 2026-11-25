"""Numerical kernels (ADR 0004).

Descriptive statistics use the Python 3.12 standard library (exact rational accumulation) and NumPy
``method='linear'`` quantiles. Correlation uses ``statistics.correlation`` behind explicit range and
conditioning guards. Undefined results are ``None`` with an explicit reason, never a silent value.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

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


def exact_midpoint(a: float, b: float) -> float:
    """Correctly rounded (a + b) / 2 without intermediate overflow."""
    return float((Fraction(a) + Fraction(b)) / 2)


def _median(sorted_values: Sequence[float]) -> float:
    n = len(sorted_values)
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return exact_midpoint(sorted_values[mid - 1], sorted_values[mid])


def describe(values: Sequence[float]) -> tuple[dict[str, float | None], dict[str, str]]:
    """Descriptive statistics of finite floats: ``(stats, undefined_reasons)``.

    Sample variance and standard deviation use ddof=1. Quartiles are type 7 (NumPy ``linear``).
    """
    stats: dict[str, float | None] = dict.fromkeys(DESCRIPTIVE_KEYS)
    undefined: dict[str, str] = {}
    n = len(values)
    if n == 0:
        return stats, dict.fromkeys(DESCRIPTIVE_KEYS, "no_valid_values")

    lo, hi = min(values), max(values)
    stats["min"], stats["max"] = lo, hi
    stats["mean"] = statistics.mean(values) + 0.0
    stats["median"] = _median(sorted(values)) + 0.0

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

    with np.errstate(all="ignore"):
        q1, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75], method="linear")
        q1, q3 = float(q1), float(q3)
        iqr = q3 - q1
    for key, value in (("q1", q1), ("q3", q3), ("iqr", iqr)):
        if math.isfinite(value):
            stats[key] = value + 0.0
        else:
            undefined[key] = "overflow_exceeds_float64"
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
