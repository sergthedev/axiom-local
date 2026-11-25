"""Relate: Pearson and Spearman correlation of one column pair (ADR 0004)."""

from __future__ import annotations

from typing import Any

from .ingest import Dataset
from .kernels import MIN_PAIRS, correlate
from .limits import Limits
from .ops_common import capped_ordinals, warn
from .plan import RelatePlan

SMALL_PAIRS_N = 10  # project convention for a small-sample warning; not a statistical threshold


def run_relate(ds: Dataset, plan: RelatePlan, limits: Limits) -> tuple[dict[str, Any], list]:
    cx, cy = ds.columns[plan.x], ds.columns[plan.y]
    rows = range(ds.n_rows)
    used = [i for i in rows if cx[i] is not None and cy[i] is not None]
    used_set = set(used)
    excluded = [i + 1 for i in rows if i not in used_set]
    xs, ys = [cx[i] for i in used], [cy[i] for i in used]
    coefficient, undefined = correlate(xs, ys, plan.method)
    result: dict[str, Any] = {
        "x": plan.x,
        "y": plan.y,
        "method": plan.method,
        "n_rows_total": ds.n_rows,
        "n_used": len(used),
        "n_excluded": len(excluded),
        "excluded_missing_x_only": sum(1 for i in rows if cx[i] is None and cy[i] is not None),
        "excluded_missing_y_only": sum(1 for i in rows if cy[i] is None and cx[i] is not None),
        "excluded_missing_both": sum(1 for i in rows if cx[i] is None and cy[i] is None),
        "excluded": capped_ordinals(excluded, limits),
        "coefficient": coefficient,
        "undefined": None if undefined is None else {"code": undefined[0], "columns": undefined[1]},
        "minimum_n": MIN_PAIRS,
    }
    pearson = plan.method == "pearson"
    warns = [
        warn("not_causal", "a correlation coefficient does not establish causation"),
        warn(
            "linear_only" if pearson else "monotonic_only",
            "Pearson measures linear association only; zero does not imply independence"
            if pearson
            else "Spearman measures monotonic association (average ranks for ties)",
        ),
    ]
    if MIN_PAIRS <= len(used) < SMALL_PAIRS_N:
        warns.append(warn("small_sample", f"fewer than {SMALL_PAIRS_N} complete pairs"))
    return result, warns
