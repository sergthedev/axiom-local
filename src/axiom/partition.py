"""Exact partition comparison by membership sets of source ordinals.

Cluster ids are arbitrary labels. Two clusterings describe the same partition iff the sets of
member-ordinal sets are equal. This exact comparison is authoritative; ARI (a later, supplementary
check) is not used here.
"""

from __future__ import annotations

from collections.abc import Sequence


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
