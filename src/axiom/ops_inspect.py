"""Inspect: counts, missingness and descriptive statistics (ADR 0004)."""

from __future__ import annotations

from typing import Any

from .ingest import Dataset
from .limits import Limits
from .ops_common import stats_block, valid_values
from .plan import InspectPlan


def run_inspect(ds: Dataset, plan: InspectPlan, limits: Limits) -> tuple[dict[str, Any], list]:
    columns = []
    for name in plan.columns:
        valid = valid_values(ds.columns[name])
        entry: dict[str, Any] = {
            "name": name,
            "kind": ds.kind(name),
            "n_total": ds.n_rows,
            "n_valid": len(valid),
            "n_missing": ds.n_rows - len(valid),
        }
        if entry["kind"] == "numeric":
            entry.update(stats_block(valid))
        columns.append(entry)
    result = {
        "n_rows": ds.n_rows,
        "n_blank_lines": ds.n_blank_lines,
        "n_all_empty_records": ds.n_all_empty_records,
        "columns": columns,
    }
    return result, []
