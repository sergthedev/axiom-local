"""Cluster: independent fixtures, postcondition recomputation, fail-closed behavior, determinism.

Acceptance of a solution means internal consistency of a local K-means solution; no test here
claims a global optimum. Thresholds in the engine were fixed before measurement (ADR 0006).
"""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction

import numpy as np
import pytest

import axiom.ops_cluster as ops_cluster
from _helpers import make_dataset
from axiom import ComputationError, parse_plan_json, run
from axiom.limits import Limits
from axiom.partition import membership_sets, same_partition


def plan(features: list[str], k: int, **extra: object) -> object:
    body = {
        "schema_version": "axiom.plan/1",
        "operation": "cluster",
        "features": features,
        "k": k,
        **extra,
    }
    return parse_plan_json(json.dumps(body))


def dataset(rows: list[tuple], names: tuple[str, ...] = ("x", "y")) -> object:
    csv = (
        ",".join(names)
        + "\n"
        + "".join(",".join("" if v is None else repr(v) for v in row) + "\n" for row in rows)
    )
    return make_dataset(csv, dict.fromkeys(names, "numeric"))


def cluster(rows: list[tuple], k: int, names: tuple[str, ...] = ("x", "y"), **extra: object):
    return run(dataset(rows, names), plan(list(names), k, **extra))  # type: ignore[arg-type]


def test_one_dimensional_separated_fixture_has_the_expected_partition_and_canonical_ids() -> None:
    values = [502.0, 1.0, 101.0, 2.0, 500.0, 100.0, 3.0, 501.0, 102.0]
    res = cluster([(v,) for v in values], 3, ("x",)).core["result"]
    expected = frozenset(
        frozenset(i + 1 for i, v in enumerate(values) if lo <= v <= hi)
        for lo, hi in ((0, 10), (90, 110), (400, 600))
    )
    got = membership_sets(res["assignments"], res["eligible_row_ordinals"])
    assert got == expected
    # canonical ids: ordered by smallest member ordinal (row 1 is in the 500s cluster)
    assert res["assignments"][0] == 0
    assert res["assignments"] == [0, 1, 2, 1, 0, 2, 1, 0, 2]
    assert [c["size"] for c in res["clusters"]] == [3, 3, 3]
    first = [
        min(
            o
            for o, a in zip(res["eligible_row_ordinals"], res["assignments"], strict=True)
            if a == i
        )
        for i in range(3)
    ]
    assert first == sorted(first)


def test_asymmetric_two_dimensional_fixture_and_original_unit_summaries() -> None:
    a = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
    b = [(50.0, 80.0), (51.0, 80.0), (50.0, 82.0)]
    rows = [a[0], b[0], a[1], b[1], a[2], b[2], a[3], a[4]]
    res = cluster(rows, 2).core["result"]
    labels = res["assignments"]
    assert same_partition(labels, [0, 1, 0, 1, 0, 1, 0, 0])
    c0, c1 = res["clusters"]
    assert c0["size"] == 5 and c1["size"] == 3
    assert c0["original_units"]["x"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    assert c1["original_units"]["y"]["mean"] == pytest.approx(242 / 3, abs=1e-12)
    assert c1["original_units"]["y"]["min"] == 80.0 and c1["original_units"]["y"]["max"] == 82.0


def test_row_order_changes_ids_but_not_the_partition() -> None:
    rng = random.Random(5)
    pts = [
        (rng.gauss(cx, 0.3), rng.gauss(cy, 0.3))
        for cx, cy in ((0, 0), (9, 0), (0, 9))
        for _ in range(20)
    ]
    base = cluster(pts, 3).core["result"]
    order = list(range(len(pts)))
    rng.shuffle(order)
    shuffled = cluster([pts[i] for i in order], 3).core["result"]
    # map shuffled rows back to the original ordinals
    ordinals_b = [order[o - 1] + 1 for o in shuffled["eligible_row_ordinals"]]
    assert same_partition(
        base["assignments"], shuffled["assignments"], base["eligible_row_ordinals"], ordinals_b
    )
    # canonical ids always start with 0 at the first row, whatever the solver label order was
    assert base["assignments"][0] == 0 and shuffled["assignments"][0] == 0


def test_partition_helper_is_exact_and_label_permutation_invariant() -> None:
    assert same_partition([0, 0, 1, 1], [5, 5, 3, 3])
    assert not same_partition([0, 0, 1, 1], [0, 1, 0, 1])
    assert not same_partition([0, 0, 1], [0, 0, 0])
    assert same_partition([0, 1], [1, 0], [1, 2], [2, 1])  # same sets, ordinals reversed
    with pytest.raises(ValueError):
        membership_sets([0, 1], [1])


def test_symmetric_rectangle_has_no_unique_expected_partition_but_is_valid_and_repeatable() -> None:
    rect = [(0.0, 0.0), (0.0, 1.0), (3.0, 0.0), (3.0, 1.0)]
    first = cluster(rect, 2).core["result"]
    sizes = sorted(c["size"] for c in first["clusters"])
    assert sizes == [2, 2]  # a valid 2-partition; which of the two optimal ones is not asserted
    assert all(first["postconditions"].values())
    for _ in range(10):
        assert cluster(rect, 2).canonical_json == cluster(rect, 2).canonical_json


def test_scaler_uses_population_sd_ddof0_and_matches_exact_arithmetic() -> None:
    res = cluster([(1.0, 10.0), (2.0, 30.0), (3.0, 20.0), (10.0, 50.0)], 2).core["result"]
    xs = [Fraction(v) for v in (1, 2, 3, 10)]
    mean = sum(xs) / 4
    var0 = sum((v - mean) ** 2 for v in xs) / 4  # ddof = 0
    assert res["scaler"]["kind"] == "standard_ddof0"
    assert res["scaler"]["mean"][0] == float(mean)
    assert res["scaler"]["scale"][0] == pytest.approx(math.sqrt(float(var0)), rel=1e-14)
    var1 = sum((v - mean) ** 2 for v in xs) / 3
    assert abs(res["scaler"]["scale"][0] - math.sqrt(float(var1))) > 0.1  # not the sample SD


def test_independent_recomputation_from_the_reported_result() -> None:
    rng = random.Random(11)
    pts = [
        (rng.gauss(cx, 1.0), rng.gauss(cy, 1.0))
        for cx, cy in ((0, 0), (7, 2), (3, 8))
        for _ in range(60)
    ]
    rng.shuffle(pts)
    res = cluster(pts, 3).core["result"]
    X = np.array(pts)
    mean, scale = np.array(res["scaler"]["mean"]), np.array(res["scaler"]["scale"])
    Xs = (X - mean) / scale
    centers = np.array([c["centroid_standardized"] for c in res["clusters"]])
    labels = np.array(res["assignments"])
    d2 = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assert np.array_equal(d2.argmin(axis=1), labels)  # nearest-center assignment
    for j, c in enumerate(res["clusters"]):  # centroids are member means (fixed point)
        assert np.allclose(Xs[labels == j].mean(axis=0), c["centroid_standardized"], atol=1e-3)
        assert c["size"] == int((labels == j).sum()) > 0
        for f, name in enumerate(("x", "y")):
            members = [pts[i][f] for i in np.flatnonzero(labels == j)]
            exact = float(sum(Fraction(v) for v in members) / len(members))
            assert c["original_units"][name]["mean"] == pytest.approx(exact, rel=1e-14)
    inertia = float(((Xs - centers[labels]) ** 2).sum())
    assert res["inertia_standardized"] == pytest.approx(inertia, rel=1e-9)
    assert sum(c["size"] for c in res["clusters"]) == res["n_eligible"] == 180


def test_listwise_exclusions_are_disclosed_and_ordinals_preserved() -> None:
    rows = [
        (0.0, 0.0),
        (None, 5.0),
        (0.1, 0.1),
        (10.0, 10.0),
        (4.0, None),
        (10.1, 9.9),
        (None, None),
    ]
    res = cluster(rows, 2).core["result"]
    assert res["n_rows_total"] == 7 and res["n_eligible"] == 4 and res["n_excluded"] == 3
    assert res["excluded_row_ordinals"] == [2, 5, 7]
    assert res["eligible_row_ordinals"] == [1, 3, 4, 6]
    assert same_partition(res["assignments"], [0, 0, 1, 1])
    capped = run(dataset(rows), plan(["x", "y"], 2), Limits(max_reported_ordinals=2)).core["result"]
    assert capped["excluded_row_ordinals"] == [2, 5] and capped["excluded_row_ordinals_truncated"]


@pytest.mark.parametrize(
    ("rows", "k", "code"),
    [
        ([(1.0, 1.0), (2.0, 2.0)], 3, "too_few_rows"),
        ([(1.0, 1.0)] * 3 + [(2.0, 2.0)] * 3, 3, "k_exceeds_distinct_points"),
        ([(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)], 2, "constant_feature"),
        ([(1.0, 1.0), (2.0, 1.0 + 2.0**-50), (3.0, 1.0 + 2.0**-49)], 2, "near_constant_feature"),
    ],
)
def test_invalid_data_fails_closed_with_a_structured_error(rows: list, k: int, code: str) -> None:
    with pytest.raises(ComputationError) as e:
        cluster(rows, k)
    assert e.value.code == code
    json.dumps(e.value.to_dict())


def test_iteration_cap_is_a_conservative_failure() -> None:
    rng = random.Random(2)
    pts = [(rng.random(), rng.random()) for _ in range(200)]
    with pytest.raises(ComputationError) as e:
        cluster(pts, 8, max_iter=1)
    assert e.value.code == "iteration_cap_reached"


def test_solver_warnings_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import warnings

    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning

    class Noisy(KMeans):
        def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001
            warnings.warn("synthetic", ConvergenceWarning, stacklevel=2)
            return super().fit(X, y, sample_weight)

    monkeypatch.setattr(ops_cluster, "KMeans", Noisy)
    with pytest.raises(ComputationError) as e:
        cluster([(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)], 2)
    assert e.value.code == "solver_warning" and e.value.details["categories"] == [
        "ConvergenceWarning"
    ]


def test_independent_checks_detect_a_corrupted_solver_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.cluster import KMeans

    class Wrong(KMeans):
        def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001
            super().fit(X, y, sample_weight)
            self.labels_ = (self.labels_ + 1) % self.n_clusters  # labels no longer match centers
            return self

    monkeypatch.setattr(ops_cluster, "KMeans", Wrong)
    with pytest.raises(ComputationError) as e:
        cluster([(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)], 2)
    assert e.value.code == "postcondition_failed"
    assert "nearest_center" in e.value.details["failed"]

    class WrongInertia(KMeans):
        def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001
            super().fit(X, y, sample_weight)
            self.inertia_ = self.inertia_ * 1.01 + 1.0
            return self

    monkeypatch.setattr(ops_cluster, "KMeans", WrongInertia)
    with pytest.raises(ComputationError) as e:
        cluster([(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)], 2)
    assert "inertia_consistent" in e.value.details["failed"]

    class EmptyCluster(KMeans):
        def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001
            super().fit(X, y, sample_weight)
            self.labels_ = np.zeros_like(self.labels_)
            return self

    monkeypatch.setattr(ops_cluster, "KMeans", EmptyCluster)
    with pytest.raises(ComputationError) as e:
        cluster([(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)], 2)
    assert e.value.code == "empty_cluster"


def test_twenty_in_process_repeats_are_byte_identical() -> None:
    rng = random.Random(8)
    pts = [
        (rng.gauss(cx, 1.0), rng.gauss(cy, 1.0))
        for cx, cy in ((0, 0), (6, 1), (2, 7))
        for _ in range(100)
    ]
    ds = dataset(pts)
    p = plan(["x", "y"], 3)
    outputs = {run(ds, p).canonical_json for _ in range(20)}  # type: ignore[arg-type]
    assert len(outputs) == 1


def test_seed_and_k_are_explicit_inputs_that_change_the_plan_hash() -> None:
    pts = [(0.0, 0.0), (0.2, 0.1), (5.0, 5.0), (5.1, 4.9), (9.0, 0.0), (9.2, 0.3)]
    a = cluster(pts, 2).core
    b = cluster(pts, 2, seed=7).core
    assert a["inputs"]["plan_sha256"] != b["inputs"]["plan_sha256"]
    assert a["result"]["solver"]["seed"] == 42 and b["result"]["solver"]["seed"] == 7
    assert cluster(pts, 3).core["result"]["k"] == 3


def test_descriptive_warnings_are_present_and_no_semantic_labels() -> None:
    core = cluster([(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)], 2).core
    assert {w["code"] for w in core["warnings"]} >= {"clusters_are_descriptive", "local_solution"}
    assert "label" not in json.dumps(core["result"]["clusters"][0]).lower().replace("labels", "")


# ---- scale handling ----


def _blobs(seed: int = 21) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [
        (rng.gauss(cx, 0.4), rng.gauss(cy, 0.4))
        for cx, cy in ((1, 1), (9, 2), (4, 11))
        for _ in range(25)
    ]


@pytest.mark.parametrize("exponent", [-1000, -600, 600, 900])
def test_exact_power_of_two_rescaling_gives_identical_clusters_in_scaled_units(
    exponent: int,
) -> None:
    base = cluster(_blobs(), 3).core["result"]
    factor = 2.0**exponent
    scaled_rows = [(x * factor, y * factor) for x, y in _blobs()]
    res = cluster(scaled_rows, 3).core["result"]
    assert res["assignments"] == base["assignments"]
    assert [c["centroid_standardized"] for c in res["clusters"]] == [
        c["centroid_standardized"] for c in base["clusters"]
    ]
    assert res["inertia_standardized"] == base["inertia_standardized"]
    # the scaler is reported in original units, exactly (inverse power-of-two scaling)
    assert res["scaler"]["mean"] == [v * factor for v in base["scaler"]["mean"]]
    assert res["scaler"]["scale"] == [v * factor for v in base["scaler"]["scale"]]


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1e100, 1e200])
def test_tiny_and_huge_magnitudes_are_valid_and_give_the_same_partition(scale: float) -> None:
    base = cluster(_blobs(), 3).core["result"]
    rows = [(x * scale, y * scale) for x, y in _blobs()]
    res = cluster(rows, 3).core["result"]
    assert same_partition(res["assignments"], base["assignments"])
    assert all(res["postconditions"].values())
    assert res["scaler"]["scale"][0] > 0.0 and math.isfinite(res["scaler"]["scale"][0])


def test_outlier_that_collapses_other_points_after_scaling_is_rejected() -> None:
    rows = [(1e-300,), (2e-300,), (3e-300,), (1e300,)]
    with pytest.raises(ComputationError) as e:
        cluster(rows, 4, ("x",))
    assert e.value.code == "k_exceeds_distinct_points"
    ok = cluster(rows, 2, ("x",)).core["result"]  # 2 distinct standardized levels still allow k=2
    assert sorted(c["size"] for c in ok["clusters"]) == [1, 3]


# ---- solver contract ----


def _patched(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    from sklearn.cluster import KMeans

    class Faulty(KMeans):
        def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001
            super().fit(X, y, sample_weight)
            for name, value in overrides.items():
                if callable(value):
                    value = value(self)
                if value is _DELETE:
                    delattr(self, name)
                else:
                    setattr(self, name, value)
            return self

    monkeypatch.setattr(ops_cluster, "KMeans", Faulty)


_DELETE = object()
FOUR = [(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)]


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"labels_": lambda m: m.labels_ + 5}, "labels outside"),
        ({"labels_": lambda m: -m.labels_ - 1}, "labels outside"),
        ({"labels_": lambda m: m.labels_[:-1]}, "wrong shape"),
        ({"labels_": lambda m: m.labels_.astype(float)}, "wrong shape or dtype"),
        ({"cluster_centers_": lambda m: m.cluster_centers_[:1]}, "centers"),
        ({"cluster_centers_": lambda m: m.cluster_centers_ * float("nan")}, "centers"),
        ({"inertia_": float("nan")}, "inertia"),
        ({"inertia_": -1.0}, "inertia"),
        ({"n_iter_": 0}, "n_iter"),
        ({"labels_": _DELETE}, "missing or malformed"),
        ({"inertia_": "not a number"}, "missing or malformed"),
    ],
)
def test_corrupted_solver_outputs_fail_closed_before_indexing(
    monkeypatch: pytest.MonkeyPatch, overrides: dict, problem: str
) -> None:
    _patched(monkeypatch, **overrides)
    with pytest.raises(ComputationError) as e:
        cluster(FOUR, 2)
    assert e.value.code == "invalid_solver_output"
    assert problem in e.value.details["problem"]


def test_warnings_from_the_scaler_are_rejected_not_only_from_the_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    from sklearn.preprocessing import StandardScaler

    class NoisyScaler(StandardScaler):
        def fit_transform(self, X, y=None, **kwargs):  # noqa: ANN001
            warnings.warn("synthetic scaler warning", RuntimeWarning, stacklevel=2)
            return super().fit_transform(X, y, **kwargs)

    monkeypatch.setattr(ops_cluster, "StandardScaler", NoisyScaler)
    with pytest.raises(ComputationError) as e:
        cluster(FOUR, 2)
    assert e.value.code == "solver_warning" and e.value.details["categories"] == ["RuntimeWarning"]


def test_floating_point_exceptions_and_solver_errors_are_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    class Overflowing(StandardScaler):
        def fit_transform(self, X, y=None, **kwargs):  # noqa: ANN001
            raise FloatingPointError("overflow encountered")

    monkeypatch.setattr(ops_cluster, "StandardScaler", Overflowing)
    with pytest.raises(ComputationError) as e:
        cluster(FOUR, 2)
    assert e.value.code == "numeric_exception"
    monkeypatch.undo()

    class Broken(KMeans):
        def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001
            raise ValueError("synthetic solver failure")

    monkeypatch.setattr(ops_cluster, "KMeans", Broken)
    with pytest.raises(ComputationError) as e:
        cluster(FOUR, 2)
    assert e.value.code == "solver_failure" and e.value.details["exception"] == "ValueError"


def test_reported_checks_are_explicitly_approximate_and_lloyd_fixed_point_is_a_diagnostic() -> None:
    res = cluster(_blobs(), 3).core["result"]
    assert "centroid_consistency_approx" in res["postconditions"]
    assert "centroid_fixed_point" not in res["postconditions"]
    assert isinstance(res["diagnostics"]["lloyd_label_fixed_point"], bool)
    assert res["diagnostics"]["lloyd_label_fixed_point"] is True  # well-separated blobs


# ---- joint workload guard ----


def test_workload_guard_rejects_before_any_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    from axiom import Limits, PlanError

    class MustNotRun:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("KMeans was constructed although the workload cap was exceeded")

    monkeypatch.setattr(ops_cluster, "KMeans", MustNotRun)
    ds = dataset(_blobs())
    with pytest.raises(PlanError) as e:
        run(ds, plan(["x", "y"], 3), Limits(max_cluster_work=1_000))
    assert e.value.code == "cluster_workload_too_large"
    assert e.value.details["work"] == 75 * 3 * 2 * 10 * 300


def test_documented_worst_cases_exceed_the_default_cap_without_building_data() -> None:
    from axiom import DEFAULT_LIMITS
    from axiom.plan import cluster_work

    worst = parse_plan_json(
        json.dumps(
            {
                "schema_version": "axiom.plan/1",
                "operation": "cluster",
                "features": [f"f{i}" for i in range(64)],
                "k": 64,
                "n_init": 20,
                "max_iter": 1000,
            }
        )
    )
    assert cluster_work(worst, 1_000_000) > DEFAULT_LIMITS.max_cluster_work
    assert cluster_work(worst, 1_000) <= DEFAULT_LIMITS.max_cluster_work  # small data stay allowed
    default_plan = plan(["a", "b", "c", "d"], 8)
    assert cluster_work(default_plan, 100_000) <= DEFAULT_LIMITS.max_cluster_work
