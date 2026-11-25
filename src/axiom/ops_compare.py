"""Compare: group-level descriptive summaries with exact text keys (ADR 0004)."""

from __future__ import annotations

from typing import Any

from .errors import ComputationError
from .ingest import Dataset
from .limits import Limits
from .ops_common import capped_ordinals, stats_block, valid_values, warn
from .plan import ComparePlan

SMALL_GROUP_N = 5  # project convention for a small-group warning; not a privacy guarantee


def run_compare(ds: Dataset, plan: ComparePlan, limits: Limits) -> tuple[dict[str, Any], list]:
    keys, measure = ds.columns[plan.group_by], ds.columns[plan.measure]
    groups: dict[str, list[float | None]] = {}
    missing_key: list[int] = []
    for i in range(ds.n_rows):
        key = keys[i]
        if key is None:
            missing_key.append(i + 1)
            continue
        groups.setdefault(key, []).append(measure[i])
        if len(groups) > limits.max_groups:
            raise ComputationError("too_many_groups", "group cap exceeded", cap=limits.max_groups)
    out: list[dict[str, Any]] = []
    warns: list[dict[str, str]] = []
    for key in sorted(groups):  # Unicode code point order of exact (non-normalized) keys
        values = groups[key]
        valid = valid_values(tuple(values))
        out.append(
            {
                "key": key,
                "n_total": len(values),
                "n_valid": len(valid),
                "n_missing": len(values) - len(valid),
                **stats_block(valid),
            }
        )
        if len(valid) < SMALL_GROUP_N:
            warns.append(
                warn("small_group", f"group {key!r} has fewer than {SMALL_GROUP_N} valid values")
            )
    if warns:
        warns.append(warn("small_groups_not_private", "small groups may identify individuals"))
    warns.append(
        warn(
            "descriptive_only",
            "group summaries are descriptive, not causal or population estimates",
        )
    )
    result = {
        "group_by": plan.group_by,
        "measure": plan.measure,
        "n_rows_total": ds.n_rows,
        "n_missing_group_key": len(missing_key),
        "missing_group_key": capped_ordinals(missing_key, limits),
        "n_groups": len(out),
        "groups": out,
    }
    return result, warns
