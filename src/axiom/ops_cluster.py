"""Reproducible K-means with independent postcondition checks (ADR 0006).

scikit-learn ``KMeans`` + ``StandardScaler`` (ddof=0) do the fitting. The result is accepted only if
quantities recomputed here with separate code (nearest-center assignment, centroid means, inertia,
scaler statistics) agree with the solver within tolerances fixed *before* any measurement. A passing
check shows internal consistency of a local solution, never a global optimum.
"""

from __future__ import annotations

import statistics
import warnings
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .errors import ComputationError
from .ingest import Dataset
from .kernels import NEAR_CONSTANT_RELATIVE_SPREAD
from .limits import Limits
from .plan import ClusterPlan

MAX_ABS_FEATURE_VALUE = 2.0**500  # keeps squares and sums of squares far from overflow
ASSIGNMENT_TOLERANCE = 1e-9  # squared-distance excess allowed, times (1 + ||x||^2)
INERTIA_RTOL = 1e-9
SCALE_RTOL = 1e-9
MEAN_TOL_IN_SCALES = 2.0**-12  # consistent with the admitted conditioning (2^-40 spread)
CENTROID_ABS_SLACK = 1e-12
DISTANCE_BATCH_ELEMENTS = 4_000_000


def _check_features(X: np.ndarray, names: tuple[str, ...]) -> None:
    for j, name in enumerate(names):
        col = X[:, j]
        lo, hi = float(col.min()), float(col.max())
        if max(abs(lo), abs(hi)) > MAX_ABS_FEATURE_VALUE:
            raise ComputationError("unsafe_scale", "feature magnitude too large", feature=name)
        if lo == hi:
            raise ComputationError("constant_feature", "feature is constant", feature=name)
        if hi - lo <= NEAR_CONSTANT_RELATIVE_SPREAD * max(abs(lo), abs(hi)):
            raise ComputationError(
                "near_constant_feature", "feature is near-constant", feature=name
            )


def _distinct_rows(X: np.ndarray) -> int:
    return int(np.unique(X, axis=0).shape[0])


def _batches(n: int, k: int, p: int):
    size = max(1, DISTANCE_BATCH_ELEMENTS // max(1, k * p))
    for start in range(0, n, size):
        yield start, min(n, start + size)


def _nearest_assignment_ok(Xs: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> bool:
    k, p = centers.shape
    for lo, hi in _batches(len(Xs), k, p):
        block = Xs[lo:hi]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned = d2[np.arange(hi - lo), labels[lo:hi]]
        slack = ASSIGNMENT_TOLERANCE * (1.0 + (block**2).sum(axis=1))
        if np.any(assigned > d2.min(axis=1) + slack):
            return False
    return True


def _inertia(Xs: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for lo, hi in _batches(len(Xs), 1, Xs.shape[1]):
        diff = Xs[lo:hi] - centers[labels[lo:hi]]
        total += float((diff**2).sum())
    return total


def _canonical_order(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Canonical ids ordered by the smallest member ordinal (== ordered membership sets).

    Returns ``(canonical_labels, order)`` where ``order[new_id] = solver_label``.
    """
    first = np.full(k, len(labels), dtype=np.int64)
    for label in range(k):
        idx = np.flatnonzero(labels == label)
        first[label] = idx[0]
    order = np.argsort(first, kind="stable")
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return remap[labels], order


def run_cluster(dataset: Dataset, plan: ClusterPlan, limits: Limits) -> tuple[dict[str, Any], list]:
    cols = [dataset.columns[f] for f in plan.features]
    eligible = [i for i in range(dataset.n_rows) if all(c[i] is not None for c in cols)]
    eligible_set = set(eligible)
    excluded = [i + 1 for i in range(dataset.n_rows) if i not in eligible_set]
    n = len(eligible)
    if n < plan.k:
        raise ComputationError("too_few_rows", "fewer eligible rows than k", n_eligible=n, k=plan.k)
    X = np.array([[c[i] for c in cols] for i in eligible], dtype=np.float64)
    _check_features(X, plan.features)
    distinct = _distinct_rows(X)
    if distinct < plan.k:
        raise ComputationError(
            "k_exceeds_distinct_points",
            "k exceeds distinct eligible points",
            distinct=distinct,
            k=plan.k,
        )

    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    Xs = scaler.fit_transform(X)
    if not np.all(np.isfinite(Xs)) or not np.all(scaler.scale_ > 0):
        raise ComputationError("unsafe_scale", "standardization produced non-finite values")
    if _distinct_rows(Xs) < plan.k:
        raise ComputationError(
            "k_exceeds_distinct_points", "k exceeds distinct standardized points"
        )
    ref_mean = X.mean(axis=0)
    ref_scale = np.sqrt(((X - ref_mean) ** 2).mean(axis=0))
    scaler_ok = bool(
        np.all(np.abs(scaler.scale_ - ref_scale) <= SCALE_RTOL * ref_scale)
        and np.all(np.abs(scaler.mean_ - ref_mean) <= MEAN_TOL_IN_SCALES * ref_scale)
    )

    model = KMeans(
        n_clusters=plan.k,
        init=plan.init,
        n_init=plan.n_init,
        max_iter=plan.max_iter,
        tol=plan.tol,
        algorithm=plan.algorithm,
        copy_x=True,
        random_state=plan.seed,
    )
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(Xs)
    if caught:
        raise ComputationError(
            "solver_warning",
            "solver or NumPy emitted a warning; result rejected",
            categories=sorted({w.category.__name__ for w in caught}),
        )
    if model.n_iter_ >= plan.max_iter:
        raise ComputationError(
            "iteration_cap_reached", "solver hit max_iter; result rejected (conservative)"
        )

    labels, centers = model.labels_.astype(np.int64), np.asarray(model.cluster_centers_)
    sizes = np.bincount(labels, minlength=plan.k)
    if np.any(sizes == 0):
        raise ComputationError("empty_cluster", "solver returned an empty cluster")
    mean_var = float(np.var(Xs, axis=0).mean())
    members_mean = np.vstack([Xs[labels == j].mean(axis=0) for j in range(plan.k)])
    residual = float(((centers - members_mean) ** 2).sum())
    inertia = _inertia(Xs, centers, labels)
    checks = {
        "nearest_center": _nearest_assignment_ok(Xs, centers, labels),
        "nonempty_clusters": True,
        "centroid_fixed_point": residual <= plan.tol * mean_var + CENTROID_ABS_SLACK,
        "inertia_consistent": abs(inertia - float(model.inertia_))
        <= INERTIA_RTOL * max(1.0, inertia),
        "scaler_consistent": scaler_ok,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ComputationError("postcondition_failed", "independent check failed", failed=failed)

    canon, order = _canonical_order(labels, plan.k)
    clusters = []
    for new_id in range(plan.k):
        member_idx = np.flatnonzero(canon == new_id)
        original = {}
        for j, name in enumerate(plan.features):
            vals = X[member_idx, j].tolist()
            original[name] = {
                "mean": statistics.mean(vals) + 0.0,
                "min": min(vals) + 0.0,
                "max": max(vals) + 0.0,
            }
        clusters.append(
            {
                "id": new_id,
                "size": int(member_idx.size),
                "centroid_standardized": [float(v) + 0.0 for v in centers[order[new_id]]],
                "original_units": original,
            }
        )
    cap = limits.max_reported_ordinals
    result = {
        "features": list(plan.features),
        "k": plan.k,
        "n_rows_total": dataset.n_rows,
        "n_eligible": n,
        "n_excluded": len(excluded),
        "excluded_row_ordinals": excluded[:cap],
        "excluded_row_ordinals_truncated": len(excluded) > cap,
        "scaler": {
            "kind": plan.scaling,
            "mean": [float(v) + 0.0 for v in scaler.mean_],
            "scale": [float(v) + 0.0 for v in scaler.scale_],
        },
        "solver": {
            "init": plan.init,
            "algorithm": plan.algorithm,
            "seed": plan.seed,
            "n_init": plan.n_init,
            "max_iter": plan.max_iter,
            "tol": plan.tol,
            "copy_x": True,
            "threadpool_limit": 1,
            "n_iter_diagnostic": int(model.n_iter_),
        },
        "inertia_standardized": inertia + 0.0,
        "clusters": clusters,
        "eligible_row_ordinals": [i + 1 for i in eligible],
        "assignments": canon.tolist(),
        "postconditions": checks,
    }
    warns = [
        {
            "code": "clusters_are_descriptive",
            "detail": "clusters are a descriptive partition of the sample, not validated classes",
        },
        {
            "code": "local_solution",
            "detail": "result is a local K-means solution; no global-optimum claim",
        },
    ]
    return result, warns
