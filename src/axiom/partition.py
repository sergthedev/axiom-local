"""Exact partition comparison by membership sets of source ordinals.

Cluster ids are arbitrary labels. Two clusterings describe the same partition iff the sets of
member-ordinal sets are equal. This exact comparison is authoritative; the adjusted Rand index is
provided as an auxiliary, label-invariant similarity only (ADR 0008).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction


def membership_sets(labels: Sequence[int], ordinals: Sequence[int] | None = None) -> frozenset:
    """Partition as a frozenset of frozensets of ordinals (default ordinals: 1..n)."""
    ords = list(ordinals) if ordinals is not None else list(range(1, len(labels) + 1))
    if len(ords) != len(labels):
        raise ValueError("labels and ordinals must have equal length")
    groups: dict[int, set[int]] = {}
    for label, ordinal in zip(labels, ords, strict=True):
        groups.setdefault(label, set()).add(ordinal)
    return frozenset(frozenset(g) for g in groups.values())


def same_partition(
    a: Sequence[int],
    b: Sequence[int],
    ordinals_a: Sequence[int] | None = None,
    ordinals_b: Sequence[int] | None = None,
) -> bool:
    return membership_sets(a, ordinals_a) == membership_sets(b, ordinals_b)


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> Fraction | None:
    """Adjusted Rand index of two labelings of the same items, in exact rational arithmetic.

    Auxiliary only: ``same_partition`` is the authoritative identity check. ARI equals 1 for
    partitions that are identical up to label permutation (REF-027) and is computed here from the
    pair-count contingency table without floating point. Returns ``None`` where the index is
    undefined (both labelings put all items in one cluster, or all in singletons, so that the
    expected and maximum indices coincide).
    """
    if len(a) != len(b):
        raise ValueError("labelings must have equal length")
    pairs: dict[tuple[int, int], int] = {}
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for x, y in zip(a, b, strict=True):
        pairs[(x, y)] = pairs.get((x, y), 0) + 1
        rows[x] = rows.get(x, 0) + 1
        cols[y] = cols.get(y, 0) + 1
    index = sum(_comb2(c) for c in pairs.values())
    sum_a = sum(_comb2(c) for c in rows.values())
    sum_b = sum(_comb2(c) for c in cols.values())
    total = _comb2(len(a))
    if total == 0:
        return None
    expected = Fraction(sum_a * sum_b, total)
    maximum = Fraction(sum_a + sum_b, 2)
    if maximum == expected:
        return None
    return (index - expected) / (maximum - expected)


def cluster_partition(result: dict) -> frozenset:
    """Membership sets (of source row ordinals) of a Cluster result's ``result`` block."""
    return membership_sets(result["assignments"], result["eligible_row_ordinals"])
