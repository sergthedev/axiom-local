"""Independent exact-arithmetic oracles for tests (not used by the engine).

The engine relies on library kernels. These oracles recompute the same mathematical quantities with
different code (``fractions.Fraction`` and ``decimal.Decimal``) so that a test never compares a
library call with itself.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

NIST_DIR = Path(__file__).parent / "data" / "nist"


def exact_mean(values: list[float]) -> Fraction:
    return sum((Fraction(v) for v in values), Fraction(0)) / len(values)


def exact_sample_variance(values: list[float]) -> Fraction:
    m = exact_mean(values)
    return sum(((Fraction(v) - m) ** 2 for v in values), Fraction(0)) / (len(values) - 1)


def frac_to_decimal(f: Fraction, digits: int = 60) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = digits
        return Decimal(f.numerator) / Decimal(f.denominator)


def exact_sample_sd_decimal(values: list[float], digits: int = 60) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = digits
        return frac_to_decimal(exact_sample_variance(values), digits).sqrt()


def exact_quantile_type7(values: list[float], q: Fraction) -> Fraction:
    s = sorted(values)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return Fraction(s[lo]) + (h - lo) * (Fraction(s[hi]) - Fraction(s[lo]))


def exact_correlation_squared_and_sign(x: list[float], y: list[float]) -> tuple[Fraction, int]:
    mx, my = exact_mean(x), exact_mean(y)
    sxy = sum(
        ((Fraction(a) - mx) * (Fraction(b) - my) for a, b in zip(x, y, strict=True)), Fraction(0)
    )
    sxx = sum(((Fraction(a) - mx) ** 2 for a in x), Fraction(0))
    syy = sum(((Fraction(b) - my) ** 2 for b in y), Fraction(0))
    return (sxy * sxy) / (sxx * syy), (sxy > 0) - (sxy < 0)


def exact_pearson_decimal(x: list[float], y: list[float], digits: int = 60) -> Decimal:
    r2, sign = exact_correlation_squared_and_sign(x, y)
    with localcontext() as ctx:
        ctx.prec = digits
        return sign * frac_to_decimal(r2, digits).sqrt()


def average_ranks(values: list[float]) -> list[Fraction]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [Fraction(0)] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = Fraction(i + 1 + j + 1, 2)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def exact_spearman_decimal(x: list[float], y: list[float], digits: int = 60) -> Decimal:
    rx, ry = average_ranks(x), average_ranks(y)
    # ranks are exact rationals; Pearson of ranks with exact arithmetic
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum(((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)), Fraction(0))
    sxx = sum(((a - mx) ** 2 for a in rx), Fraction(0))
    syy = sum(((b - my) ** 2 for b in ry), Fraction(0))
    sign = (sxy > 0) - (sxy < 0)
    with localcontext() as ctx:
        ctx.prec = digits
        return sign * frac_to_decimal((sxy * sxy) / (sxx * syy), digits).sqrt()


def load_nist() -> tuple[dict, dict[str, list[str]]]:
    manifest = json.loads((NIST_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    tokens: dict[str, list[str]] = {}
    for name, meta in manifest["datasets"].items():
        lines = (NIST_DIR / meta["file"]).read_bytes().decode("ascii").split("\r\n")
        marker = next(i for i, line in enumerate(lines) if line.strip() == "Data: Y")
        tokens[name] = [ln.strip() for ln in lines[marker + 2 :] if ln.strip()]
    return manifest, tokens


def ulp_error(value: float, reference: Decimal) -> float:
    """Absolute error of ``value`` against ``reference`` in units of ulp(value)."""
    err = abs(Decimal(value) - reference)
    return float(err / Decimal(math.ulp(value)))
