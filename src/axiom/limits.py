"""Provisional engineering caps (ADR 0003).

These are bounded-resource guards, **not** statistical or scientific constants. They are
conservative starting values chosen before any profiling and must be revisited after profiling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Limits:
    max_bytes: int = 32 * 1024 * 1024
    max_rows: int = 1_000_000
    max_columns: int = 256
    max_header_chars: int = 256
    max_field_chars: int = 65_536
    max_groups: int = 10_000
    max_k: int = 64
    max_cluster_features: int = 64
    max_reported_ordinals: int = 100
    max_plan_bytes: int = 65_536
    max_plan_depth: int = 8
    max_plan_nodes: int = 4_096
    # n_eligible_rows * k * n_features * n_init * max_iter (distance-component evaluations, an
    # upper bound on solver work). Provisional; rejects joint worst cases before any fit.
    max_cluster_work: int = 50_000_000_000


DEFAULT_LIMITS = Limits()
