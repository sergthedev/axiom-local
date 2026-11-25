"""Comparison of two canonical result cores under documented criteria (ADR 0008).

Two separate questions are answered and never merged:

* **Exact replay** (same environment): are the canonical result bytes identical? Nothing else
  passes.
* **Numerical agreement** (different environment): does every field satisfy the per-operation
  criterion below? Criteria were fixed in ADR 0008 and the EXP-004/EXP-005
  pre-registrations *before* any evaluation. They are derived from the accuracy criteria of ADR 0004
  and the K-means postconditions of ADR 0006, not tuned to observed differences.

Agreement is not correctness: two environments can agree on a wrong value. Clustering partitions are
compared by membership sets of source row ordinals, independent of cluster label numbering; a
matching partition says nothing about global optimality.
"""

from __future__ import annotations

import copy
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_dumps
from .partition import adjusted_rand_index, cluster_partition

CRITERIA_VERSION = "axiom.agreement/1"

# ADR 0004: mean/variance/SD within 1 ulp of the exact value of the same inputs; median and
# quartiles correctly rounded (0.5 ulp). Two conforming results are at most 2 ulp apart.
DESCRIPTIVE_MAX_ULP = 2
# ADR 0004: |r - r_exact| <= 1e-12 on each side, hence |r_a - r_b| <= 2e-12.
CORRELATION_ABS = 2e-12
# ADR 0006: scaler scale within rtol 1e-9 and mean within 2^-12 scales of a two-pass recomputation
# on each side; doubled for the difference of two conforming results.
SCALER_SCALE_RTOL = 2e-9
SCALER_MEAN_SCALES = 2 * 2.0**-12
CENTROID_ABS_SLACK = 1e-12  # same slack as the ADR 0006 centroid check

_DESCRIPTIVE = frozenset({"mean", "median", "sample_variance", "sample_sd", "q1", "q3", "iqr"})


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    gating: bool = True


EXACT = Rule("exact")
ULP2 = Rule(f"ulp<={DESCRIPTIVE_MAX_ULP}")
CORR = Rule(f"abs<={CORRELATION_ABS:g}")
SCALE = Rule(f"rel<={SCALER_SCALE_RTOL:g}")
SCALER_MEAN = Rule("abs<=2^-11*scale")
CENTROID = Rule("abs<=2*sqrt(tol+1e-12)")
INERTIA = Rule("abs<=n_eligible*(tol+1e-12)+1e-9*inertia")
DIAGNOSTIC = Rule("diagnostic", gating=False)


def ulp_distance(a: float, b: float) -> int:
    """Number of binary64 values between ``a`` and ``b`` (0 when equal; -0.0 == 0.0)."""

    def key(x: float) -> int:
        bits = struct.unpack(">q", struct.pack(">d", x + 0.0))[0]
        return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF)

    return abs(key(a) - key(b))


def abs_difference(a: float, b: float) -> float | None:
    """``|a - b|`` as a finite float, or ``None`` when it exceeds the binary64 range.

    Two finite values of opposite sign near the largest float differ by more than the largest
    float; the difference would be infinity, which JSON reports cannot carry (review B2).
    """
    diff = abs(a - b)
    return diff if math.isfinite(diff) else None


def _is_diagnostic(operation: str, path: tuple[str, ...]) -> bool:
    """Reported but not gating: the iteration count and the Lloyd label fixed-point flag."""
    return operation == "cluster" and (
        path == ("result", "solver", "n_iter_diagnostic") or path[:2] == ("result", "diagnostics")
    )


def _rule_for(operation: str, path: tuple[str, ...]) -> Rule:
    """Criterion for one float leaf. ``path`` uses ``*`` for list indices."""
    if not path or path[0] != "result":
        return EXACT
    leaf = path[-1]
    if operation in ("inspect", "compare") and len(path) >= 2 and path[-2] == "stats":
        return ULP2 if leaf in _DESCRIPTIVE else EXACT  # min/max are input values: exact
    if operation == "relate" and path == ("result", "coefficient"):
        return CORR
    if _is_diagnostic(operation, path):
        return DIAGNOSTIC
    if operation == "cluster":
        if path[:3] == ("result", "scaler", "scale"):
            return SCALE
        if path[:3] == ("result", "scaler", "mean"):
            return SCALER_MEAN
        if path[:3] == ("result", "clusters", "*") and "centroid_standardized" in path:
            return CENTROID
        if path == ("result", "inertia_standardized"):
            return INERTIA
        if "original_units" in path and leaf == "mean":
            return ULP2  # same exact-mean kernel as Inspect
    return EXACT


def _within(rule: Rule, a: float, b: float, ctx: dict[str, Any], index: int | None) -> bool:
    if rule is EXACT:
        return a == b
    if abs_difference(a, b) is None:
        return False  # beyond the float range: outside every tolerance below
    if rule is ULP2:
        return ulp_distance(a, b) <= DESCRIPTIVE_MAX_ULP
    if rule is CORR:
        return abs(a - b) <= CORRELATION_ABS
    if rule is SCALE:
        return abs(a - b) <= SCALER_SCALE_RTOL * abs(a)
    if rule is SCALER_MEAN:
        scale = ctx["scaler_scale"][index] if index is not None else math.inf
        return abs(a - b) <= SCALER_MEAN_SCALES * scale
    if rule is CENTROID:
        return abs(a - b) <= 2.0 * math.sqrt(ctx["tol"] + CENTROID_ABS_SLACK)
    if rule is INERTIA:
        bound = ctx["n_eligible"] * (ctx["tol"] + CENTROID_ABS_SLACK) + 1e-9 * abs(a)
        return abs(a - b) <= bound
    return True  # diagnostic


def _walk(
    ref: Any,
    obs: Any,
    path: tuple[str, ...],
    emit: Callable[[dict[str, Any]], None],
    float_leaf: Callable[[tuple[str, ...], float, float, int | None], None],
    index: int | None = None,
) -> None:
    def shape(kind: str) -> None:
        emit({"path": path, "kind": kind, "reference": ref, "observed": obs})

    if isinstance(ref, bool) or isinstance(obs, bool):
        if ref is not obs:
            shape("value")
        return
    if isinstance(ref, float) and isinstance(obs, float):
        float_leaf(path, ref, obs, index)
        return
    if isinstance(ref, dict) and isinstance(obs, dict):
        if set(ref) != set(obs):
            shape("keys")
            return
        for key in sorted(ref):
            _walk(ref[key], obs[key], (*path, key), emit, float_leaf, index)
        return
    if isinstance(ref, list) and isinstance(obs, list):
        if len(ref) != len(obs):
            shape("length")
            return
        for i, (a, b) in enumerate(zip(ref, obs, strict=True)):
            _walk(a, b, (*path, "*"), emit, float_leaf, i)
        return
    if type(ref) is not type(obs):
        shape("type")  # e.g. a value on one side and null (undefined) on the other
    elif ref != obs:
        shape("value")


def compare_cores(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Compare two result cores field by field. Returns a JSON-compatible report."""
    identical = canonical_dumps(reference) == canonical_dumps(observed)
    operation = reference.get("operation")
    aligned = False
    if operation == "cluster" and not identical:
        observed, aligned = _align_cluster_labels(reference, observed)
    ctx: dict[str, Any] = {}
    result = reference.get("result", {})
    if operation == "cluster" and isinstance(result, dict):
        ctx = {
            "tol": reference.get("plan", {}).get("tol", 0.0),
            "n_eligible": result.get("n_eligible", 0),
            "scaler_scale": result.get("scaler", {}).get("scale", []),
        }
    structural: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    n_float = 0

    def float_leaf(path: tuple[str, ...], a: float, b: float, index: int | None) -> None:
        nonlocal n_float
        n_float += 1
        if a == b:
            return
        rule = _rule_for(str(operation), path)
        try:
            ok = _within(rule, a, b, ctx, index)
        except (KeyError, IndexError, TypeError):
            ok = False
        fields.append(
            {
                "path": "/".join(path),
                "rule": rule.name,
                "gating": rule.gating,
                "reference": a,
                "observed": b,
                "abs_diff": abs_difference(a, b),  # None: exceeds the binary64 range
                "ulp_distance": ulp_distance(a, b),
                "within": ok,
            }
        )

    def emit(diff: dict[str, Any]) -> None:
        path = diff["path"]
        diff["path"] = "/".join(path)
        diff["gating"] = not _is_diagnostic(str(operation), path)
        structural.append(diff)

    _walk(reference, observed, (), emit, float_leaf)
    gating_failures = [f for f in fields if f["gating"] and not f["within"]]
    gating_nonfloat = [d for d in structural if d["gating"]]
    report: dict[str, Any] = {
        "criteria_version": CRITERIA_VERSION,
        "operation": operation,
        "byte_identical": identical,
        "n_float_fields": n_float,
        "n_float_fields_different": len(fields),
        "max_ulp_distance": max((f["ulp_distance"] for f in fields), default=0),
        "max_abs_diff": (
            None
            if any(f["abs_diff"] is None for f in fields)
            else max((f["abs_diff"] for f in fields), default=0.0)
        ),
        "non_float_differences": structural,
        "field_differences": fields,
        "n_gating_failures": len(gating_failures) + len(gating_nonfloat),
    }
    if operation == "cluster":
        report["partition"] = _partition_report(reference, observed)
        report["partition"]["labels_realigned"] = aligned
        if report["partition"]["same_partition"] is not True:
            report["n_gating_failures"] += 1
    if identical:
        report["verdict"] = "identical"
    elif report["n_gating_failures"]:
        report["verdict"] = "outside_criteria"
    else:
        report["verdict"] = "within_criteria"
    return report


def _groups(labels: list[int], ordinals: list[int]) -> dict[int, frozenset]:
    groups: dict[int, set[int]] = {}
    for label, ordinal in zip(labels, ordinals, strict=True):
        groups.setdefault(label, set()).add(ordinal)
    return {label: frozenset(members) for label, members in groups.items()}


def _align_cluster_labels(
    reference: dict[str, Any], observed: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Renumber the observed clusters to the reference ids when both describe the same partition.

    Cluster ids are arbitrary; field-by-field comparison must not depend on them. If the partitions
    differ, nothing is renumbered and the partition check fails on its own.
    """
    try:
        ra, rb = reference["result"], observed["result"]
        if cluster_partition(ra) != cluster_partition(rb):
            return observed, False
        groups_a = _groups(ra["assignments"], ra["eligible_row_ordinals"])
        groups_b = _groups(rb["assignments"], rb["eligible_row_ordinals"])
        label_of_a = {members: label for label, members in groups_a.items()}
        mapping = {label: label_of_a[members] for label, members in groups_b.items()}
    except (KeyError, TypeError, ValueError):
        return observed, False
    if all(a == b for b, a in mapping.items()):
        return observed, False
    out = copy.deepcopy(observed)
    res = out["result"]
    res["assignments"] = [mapping[label] for label in rb["assignments"]]
    clusters = []
    for cluster in rb["clusters"]:
        moved = copy.deepcopy(cluster)
        moved["id"] = mapping[cluster["id"]]
        clusters.append(moved)
    res["clusters"] = sorted(clusters, key=lambda c: c["id"])
    return out, True


def _partition_report(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    try:
        ra, rb = reference["result"], observed["result"]
        same = cluster_partition(ra) == cluster_partition(rb)
        ari = None
        if ra["eligible_row_ordinals"] == rb["eligible_row_ordinals"]:
            value = adjusted_rand_index(ra["assignments"], rb["assignments"])
            ari = None if value is None else float(value)
        return {
            "same_partition": same,
            "adjusted_rand_index_auxiliary": ari,
            "n_iter_reference": ra["solver"]["n_iter_diagnostic"],
            "n_iter_observed": rb["solver"]["n_iter_diagnostic"],
            "note": "membership-set equality is authoritative; no global-optimum claim",
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"same_partition": None, "error": f"partition not comparable ({type(exc).__name__})"}
