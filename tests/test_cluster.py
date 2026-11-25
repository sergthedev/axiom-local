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
        ([(1.0, 1.0), (2.0, 2.0), (1e300 * 3, 3.0)], 2, "unsafe_scale"),
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


def test_twenty_fresh_repeats_are_byte_identical() -> None:
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
