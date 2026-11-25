"""Comparison criteria (ADR 0008): exact identity, per-operation tolerances, partitions."""

from __future__ import annotations

import copy
import math
import random
from fractions import Fraction

import pytest
from sklearn.metrics import adjusted_rand_score

from _helpers import make_dataset
from _passport_helpers import DATA, PROFILE, example_plan
from axiom import load_csv, parse_plan_json, run
from axiom.agreement import (
    CORRELATION_ABS,
    DESCRIPTIVE_MAX_ULP,
    compare_cores,
    ulp_distance,
)
from axiom.partition import adjusted_rand_index, cluster_partition, same_partition


@pytest.fixture(scope="module")
def cores() -> dict[str, dict]:
    ds = load_csv(DATA, PROFILE)
    names = ("inspect", "relate_pearson", "relate_spearman", "compare", "cluster")
    return {n: run(ds, example_plan(n)).core for n in names}


def _up(x: float, steps: int) -> float:
    for _ in range(steps):
        x = math.nextafter(x, math.inf)
    return x


def test_ulp_distance_basics() -> None:
    assert ulp_distance(1.0, 1.0) == 0
    assert ulp_distance(0.0, -0.0) == 0
    assert ulp_distance(1.0, _up(1.0, 3)) == 3
    assert ulp_distance(-5e-324, 5e-324) == 2  # across zero
    assert ulp_distance(_up(-1.0, 2), -1.0) == 2


@pytest.mark.parametrize(
    "name", ["inspect", "relate_pearson", "relate_spearman", "compare", "cluster"]
)
def test_identical_cores_are_identical(cores, name: str) -> None:
    report = compare_cores(cores[name], copy.deepcopy(cores[name]))
    assert report["verdict"] == "identical" and report["byte_identical"]
    assert report["n_gating_failures"] == 0 and report["n_float_fields"] > 0


def _stats(core: dict, column: str) -> dict:
    return next(c for c in core["result"]["columns"] if c["name"] == column)["stats"]


@pytest.mark.parametrize(
    ("key", "steps", "verdict"),
    [
        ("mean", 1, "within_criteria"),
        ("sample_sd", DESCRIPTIVE_MAX_ULP, "within_criteria"),
        ("q1", DESCRIPTIVE_MAX_ULP + 1, "outside_criteria"),
        ("min", 1, "outside_criteria"),  # order statistics are input values: exact
        ("max", 1, "outside_criteria"),
    ],
)
def test_descriptive_statistics_criteria(cores, key: str, steps: int, verdict: str) -> None:
    obs = copy.deepcopy(cores["inspect"])
    stats = _stats(obs, "gdp_per_capita")
    stats[key] = _up(stats[key], steps)
    report = compare_cores(cores["inspect"], obs)
    assert report["verdict"] == verdict
    (diff,) = report["field_differences"]
    assert diff["path"].endswith(key) and diff["ulp_distance"] == steps


def test_undefined_versus_value_is_a_gating_difference(cores) -> None:
    obs = copy.deepcopy(cores["inspect"])
    _stats(obs, "gdp_per_capita")["mean"] = None
    report = compare_cores(cores["inspect"], obs)
    assert report["verdict"] == "outside_criteria"
    assert report["non_float_differences"][0]["kind"] == "type"


def test_count_and_key_changes_are_exact(cores) -> None:
    obs = copy.deepcopy(cores["compare"])
    obs["result"]["groups"][0]["n_valid"] += 1
    assert compare_cores(cores["compare"], obs)["verdict"] == "outside_criteria"
    obs = copy.deepcopy(cores["compare"])
    del obs["result"]["groups"][0]["n_valid"]
    report = compare_cores(cores["compare"], obs)
    assert report["non_float_differences"][0]["kind"] == "keys"


@pytest.mark.parametrize(
    ("delta", "verdict"),
    [(CORRELATION_ABS / 2, "within_criteria"), (CORRELATION_ABS * 1.5, "outside_criteria")],
)
def test_correlation_criterion(cores, delta: float, verdict: str) -> None:
    obs = copy.deepcopy(cores["relate_pearson"])
    obs["result"]["coefficient"] -= delta
    assert compare_cores(cores["relate_pearson"], obs)["verdict"] == verdict


def test_inputs_and_plan_are_always_exact(cores) -> None:
    obs = copy.deepcopy(cores["cluster"])
    obs["plan"]["tol"] = _up(obs["plan"]["tol"], 1)
    assert compare_cores(cores["cluster"], obs)["verdict"] == "outside_criteria"


def _relabel(core: dict, perm: dict[int, int]) -> dict:
    out = copy.deepcopy(core)
    res = out["result"]
    res["assignments"] = [perm[a] for a in res["assignments"]]
    for c in res["clusters"]:
        c["id"] = perm[c["id"]]
    res["clusters"].sort(key=lambda c: c["id"])
    return out


def test_cluster_comparison_ignores_label_numbering(cores) -> None:
    ref = cores["cluster"]
    k = ref["result"]["k"]
    perm = {i: (i + 1) % k for i in range(k)}
    obs = _relabel(ref, perm)
    assert obs["result"]["assignments"] != ref["result"]["assignments"]
    report = compare_cores(ref, obs)
    assert report["partition"]["same_partition"] is True
    assert report["partition"]["labels_realigned"] is True
    assert report["partition"]["adjusted_rand_index_auxiliary"] == 1.0
    assert report["verdict"] == "within_criteria"  # bytes differ, nothing else does
    assert report["n_float_fields_different"] == 0 and report["n_gating_failures"] == 0


def test_cluster_partition_change_fails_and_iteration_count_is_diagnostic(cores) -> None:
    ref = cores["cluster"]
    obs = copy.deepcopy(ref)
    obs["result"]["solver"]["n_iter_diagnostic"] += 3
    obs["result"]["diagnostics"]["lloyd_label_fixed_point"] = not obs["result"]["diagnostics"][
        "lloyd_label_fixed_point"
    ]
    report = compare_cores(ref, obs)
    assert report["verdict"] == "within_criteria"
    assert {d["path"] for d in report["non_float_differences"] if not d["gating"]} == {
        "result/solver/n_iter_diagnostic",
        "result/diagnostics/lloyd_label_fixed_point",
    }
    moved = copy.deepcopy(ref)
    a = moved["result"]["assignments"]
    i = next(j for j in range(len(a)) if a[j] != a[0])
    a[0], a[i] = a[i], a[0]  # two rows swap clusters: sizes unchanged, partition changed
    report = compare_cores(ref, moved)
    assert report["partition"]["same_partition"] is False
    assert report["verdict"] == "outside_criteria"
    assert report["partition"]["adjusted_rand_index_auxiliary"] < 1.0


def test_cluster_centroid_criterion_derived_from_tolerance(cores) -> None:
    ref = cores["cluster"]
    bound = 2.0 * math.sqrt(ref["plan"]["tol"] + 1e-12)
    for delta, verdict in ((bound * 0.5, "within_criteria"), (bound * 1.5, "outside_criteria")):
        obs = copy.deepcopy(ref)
        obs["result"]["clusters"][0]["centroid_standardized"][0] += delta
        assert compare_cores(ref, obs)["verdict"] == verdict


# --- partitions and the auxiliary adjusted Rand index -------------------------------------------


def test_same_partition_is_label_invariant() -> None:
    assert same_partition([0, 0, 1, 2], [5, 5, 9, 7])
    assert not same_partition([0, 0, 1, 1], [0, 1, 0, 1])
    result = {"assignments": [1, 0, 1], "eligible_row_ordinals": [2, 5, 9]}
    assert cluster_partition(result) == frozenset({frozenset({2, 9}), frozenset({5})})


def test_exact_ari_matches_scikit_learn_reference() -> None:
    rng = random.Random(20261125)
    for _ in range(200):
        n = rng.randint(2, 40)
        a = [rng.randint(0, rng.randint(1, 5)) for _ in range(n)]
        b = [rng.randint(0, rng.randint(1, 5)) for _ in range(n)]
        exact = adjusted_rand_index(a, b)
        ref = adjusted_rand_score(a, b)
        if exact is None:  # sklearn defines the degenerate cases as 1.0 by convention
            assert ref == 1.0
        else:
            assert abs(float(exact) - ref) <= 1e-12


def test_exact_ari_known_answers() -> None:
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == 1
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == Fraction(-1, 2)
    assert adjusted_rand_index([0, 0, 0], [0, 0, 0]) is None  # one cluster each: undefined
    with pytest.raises(ValueError):
        adjusted_rand_index([0], [0, 1])


def test_small_relate_core_from_csv() -> None:
    ds = make_dataset("x,y\n1,2\n2,4\n3,7\n", {"x": "numeric", "y": "numeric"})
    plan = (
        '{"schema_version":"axiom.plan/1","operation":"relate","x":"x","y":"y","method":"pearson"}'
    )
    core = run(ds, parse_plan_json(plan)).core
    assert compare_cores(core, core)["verdict"] == "identical"
