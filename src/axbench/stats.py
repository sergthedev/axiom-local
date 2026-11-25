"""Interval and paired-comparison helpers for AXBENCH reports (DR-005 statistics correction).

Wilson intervals and the exact McNemar test assume independent observations. AXBENCH examples
that share a lineage, a schema instance or a family are not independent, so these are reported
only as *illustrative, IID-conditional* values. The preferred uncertainty statement is a
cluster bootstrap over independently generated groups (lineages by default), with a fixed seed,
the number of groups and the resampled unit reported. None of these numbers is a population
guarantee, and no p-value is used as a deployment rule.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z95 = 1.959963984540054  # two-sided 95% standard normal quantile


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion; ``None`` when ``n == 0``."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts ``b`` and ``c``.

    Computed as min(1, 2 * P[X <= min(b, c)]) with X ~ Binomial(b + c, 1/2), in exact rational
    arithmetic before the final conversion.
    """
    from fractions import Fraction

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(Fraction(math.comb(n, i), 2**n) for i in range(min(b, c) + 1))
    return float(min(Fraction(1), 2 * tail))


def cluster_bootstrap(
    outcomes: Sequence[float],
    clusters: Sequence[str],
    *,
    reps: int = 2000,
    seed: int = 2026,
    level: float = 0.95,
) -> dict[str, object] | None:
    """Percentile bootstrap of a case-weighted mean, resampling whole clusters with replacement."""
    if len(outcomes) != len(clusters):
        raise ValueError("outcomes and clusters must have the same length")
    if not outcomes:
        return None
    groups: dict[str, list[float]] = {}
    for value, cluster in zip(outcomes, clusters, strict=True):
        groups.setdefault(cluster, []).append(float(value))
    keys = sorted(groups)
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        total = count = 0.0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            total += sum(g)
            count += len(g)
        stats.append(total / count)
    stats.sort()
    lo = stats[int(math.floor((1 - level) / 2 * reps))]
    hi = stats[min(reps - 1, int(math.ceil((1 + level) / 2 * reps)) - 1)]
    return {
        "estimate": sum(outcomes) / len(outcomes),
        "interval": [lo, hi],
        "level": level,
        "unit": "cluster",
        "n_clusters": len(keys),
        "largest_cluster": max(len(g) for g in groups.values()),
        "reps": reps,
        "seed": seed,
        "method": "percentile bootstrap over clusters, case-weighted",
    }


def paired(
    a: Sequence[bool],
    b: Sequence[bool],
    clusters: Sequence[str],
    *,
    reps: int = 2000,
    seed: int = 2026,
) -> dict[str, object]:
    """Paired 2x2 counts, percentage-point delta (b - a) and a paired cluster bootstrap."""
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("paired sequences must have equal length")
    both = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    neither = len(a) - both - only_a - only_b
    diffs = [float(y) - float(x) for x, y in zip(a, b, strict=True)]
    return {
        "n": len(a),
        "both_correct": both,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": neither,
        "delta_pp": 100.0 * (sum(b) - sum(a)) / len(a) if a else None,
        "mcnemar_exact_p_iid_illustrative": mcnemar_exact(only_a, only_b),
        "paired_cluster_bootstrap_delta": cluster_bootstrap(diffs, clusters, reps=reps, seed=seed),
    }
