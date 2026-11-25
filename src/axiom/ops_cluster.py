"""Reproducible K-means with independent postcondition checks (ADR 0006).

scikit-learn ``KMeans`` + ``StandardScaler`` (ddof=0) do the fitting. The result is accepted only if
quantities recomputed here with separate code (nearest-center assignment, centroid means, inertia,
scaler statistics) agree with the solver within tolerances fixed *before* any measurement. A passing
check shows internal consistency of a local solution, never a global optimum.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .errors import ComputationError, PlanError
from .ingest import Dataset
from .kernels import NEAR_CONSTANT_RELATIVE_SPREAD, mean_checked
from .limits import Limits
from .plan import ClusterPlan, cluster_work

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


def _nearest_labels(Xs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    k, p = centers.shape
    out = np.empty(len(Xs), dtype=np.int64)
    for lo, hi in _batches(len(Xs), k, p):
        d2 = ((Xs[lo:hi, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[lo:hi] = d2.argmin(axis=1)
    return out


def _check_solver_output(
    model: Any, n: int, k: int, p: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Validate shape, dtype, range and finiteness of the solver's outputs before any indexing."""

    def bad(problem: str) -> ComputationError:
        return ComputationError("invalid_solver_output", "solver output rejected", problem=problem)

    try:
        labels = np.asarray(model.labels_)
        centers = np.asarray(model.cluster_centers_)
        inertia = float(model.inertia_)
        n_iter = int(model.n_iter_)
    except (AttributeError, TypeError, ValueError) as exc:
        raise bad(f"missing or malformed attribute ({type(exc).__name__})") from exc
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise bad("labels have the wrong shape or dtype")
    if n and (int(labels.min()) < 0 or int(labels.max()) >= k):
        raise bad("labels outside [0, k)")
    if centers.shape != (k, p) or centers.dtype.kind != "f" or not np.all(np.isfinite(centers)):
        raise bad("centers have the wrong shape or are not finite floats")
    if not math.isfinite(inertia) or inertia < 0.0:
        raise bad("inertia is not a finite non-negative number")
    if n_iter < 1:
        raise bad("n_iter is not positive")
    return labels.astype(np.int64), centers, inertia, n_iter


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
    n, p = len(eligible), len(plan.features)
    if n < plan.k:
        raise ComputationError("too_few_rows", "fewer eligible rows than k", n_eligible=n, k=plan.k)
    work = cluster_work(plan, n)
    if work > limits.max_cluster_work:  # defensive: also checked at binding and execution
        raise PlanError(
            "cluster_workload_too_large",
            "workload exceeds cap",
            work=work,
            cap=limits.max_cluster_work,
        )
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
    # Standardization is invariant under exact power-of-two scaling, so each feature is scaled to
    # max|v| in [0.5, 1) first: squares and sums cannot overflow or underflow for any representable
    # input, and results are reported back in original units (exactly, by the inverse scaling).
    shifts = np.array([math.frexp(float(np.abs(X[:, j]).max()))[1] for j in range(p)])
    Xp = np.ldexp(X, -shifts[None, :])
    lossy = np.flatnonzero(~np.all(np.ldexp(Xp, shifts[None, :]) == X, axis=0))
    if lossy.size:  # an entry underflowed (or lost bits) while scaling: the scaling is not exact
        raise ComputationError(
            "prescaling_underflow",
            "feature range is too wide for exact power-of-two scaling; result rejected",
            features=[plan.features[int(j)] for j in lossy],
        )

    try:
        with (
            warnings.catch_warnings(record=True) as caught,
            threadpool_limits(limits=1),
            np.errstate(over="raise", invalid="raise", divide="raise"),
        ):
            warnings.simplefilter("always")
            scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
            Xs = scaler.fit_transform(Xp)
            if not np.all(np.isfinite(Xs)) or not np.all(scaler.scale_ > 0):
                raise ComputationError("unsafe_scale", "standardization produced invalid values")
            if _distinct_rows(Xs) < plan.k:
                raise ComputationError(
                    "k_exceeds_distinct_points", "k exceeds distinct standardized points"
                )
            mean_original = np.ldexp(scaler.mean_, shifts)
            scale_original = np.ldexp(scaler.scale_, shifts)
            for j, name in enumerate(plan.features):
                exact = bool(
                    np.isfinite(mean_original[j])
                    and np.isfinite(scale_original[j])
                    and np.ldexp(mean_original[j], -shifts[j]) == scaler.mean_[j]
                    and np.ldexp(scale_original[j], -shifts[j]) == scaler.scale_[j]
                )
                if not exact:
                    raise ComputationError(
                        "original_units_not_representable",
                        "scaler mean or scale cannot be reported exactly in original units",
                        feature=name,
                    )
            ref_mean = Xp.mean(axis=0)
            ref_scale = np.sqrt(((Xp - ref_mean) ** 2).mean(axis=0))
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
            model.fit(Xs)
            labels, centers, solver_inertia, n_iter = _check_solver_output(model, n, plan.k, p)
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
                # Approximate by design: the solver's stopping rule bounds the last center shift
                # (squared) by tol * mean(var); it does not prove an exact Lloyd fixed point.
                "centroid_consistency_approx": residual <= plan.tol * mean_var + CENTROID_ABS_SLACK,
                "inertia_consistent": abs(inertia - solver_inertia)
                <= INERTIA_RTOL * max(1.0, inertia),
                "scaler_consistent": scaler_ok,
            }
            lloyd_fixed_point = bool(np.array_equal(_nearest_labels(Xs, members_mean), labels))
    except FloatingPointError as exc:
        raise ComputationError(
            "numeric_exception", "floating-point exception during standardization or fit"
        ) from exc
    except (ValueError, OverflowError, MemoryError) as exc:
        raise ComputationError(
            "solver_failure", f"solver raised {type(exc).__name__}", exception=type(exc).__name__
        ) from exc
    if caught:
        raise ComputationError(
            "solver_warning",
            "scaler, solver or NumPy emitted a warning; result rejected",
            categories=sorted({w.category.__name__ for w in caught}),
        )
    if n_iter >= plan.max_iter:
        raise ComputationError(
            "iteration_cap_reached", "solver hit max_iter; result rejected (conservative)"
        )
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
            mean, reason = mean_checked(vals)
            if reason is not None:
                raise ComputationError(
                    "original_units_not_representable",
                    "cluster mean cannot be reported in original units",
                    feature=name,
                    cluster=new_id,
                    reason=reason,
                )
            original[name] = {"mean": mean, "min": min(vals) + 0.0, "max": max(vals) + 0.0}
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
            "mean": [float(v) + 0.0 for v in mean_original],
            "scale": [float(v) + 0.0 for v in scale_original],
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
            "n_iter_diagnostic": n_iter,
        },
        "inertia_standardized": inertia + 0.0,
        "clusters": clusters,
        "eligible_row_ordinals": [i + 1 for i in eligible],
        "assignments": canon.tolist(),
        "postconditions": checks,
        "diagnostics": {"lloyd_label_fixed_point": lloyd_fixed_point},
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
