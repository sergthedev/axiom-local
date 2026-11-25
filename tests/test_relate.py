"""Relate: known answers, independent exact oracles, guards, exclusions.

Pre-registered criteria (ADR 0004, fixed before measurement), versus the exact value (Fraction /
60-digit Decimal) of the parsed binary64 inputs: Pearson absolute error <= 1e-12 and Spearman
absolute error <= 1e-12 on seeded random data with offset/spread up to 1e6 and n <= 60.
"""

from __future__ import annotations

import math
import random
import statistics
from decimal import Decimal

import pytest

from _helpers import make_dataset
from axiom import ComputationError, parse_plan_json, run
from axiom.kernels import NEAR_CONSTANT_RELATIVE_SPREAD, _finish_correlation, correlate
from axiom.limits import Limits
from oracles import exact_pearson_decimal, exact_spearman_decimal

SQRT3_OVER_2 = math.sqrt(3) / 2


def pearson(x: list[float], y: list[float]) -> float:
    value, undefined = correlate(x, y, "pearson")
    assert undefined is None, undefined
    assert value is not None
    return value


def spearman(x: list[float], y: list[float]) -> float:
    value, undefined = correlate(x, y, "spearman")
    assert undefined is None, undefined
    assert value is not None
    return value


def test_pearson_known_answers() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0, abs=1e-15)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0, abs=1e-15)
    # exactly dependent yet uncorrelated: x = [-1, 0, 1], y = x^2
    assert pearson([-1, 0, 1], [1, 0, 1]) == pytest.approx(0.0, abs=1e-15)
    assert pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8, abs=1e-15)  # hand: 4/5


def test_spearman_known_answers_including_average_ties() -> None:
    assert spearman([1, 2, 3], [1, 4, 9]) == pytest.approx(1.0, abs=1e-15)  # monotone, not linear
    assert spearman([1, 2, 3], [9, 4, 1]) == pytest.approx(-1.0, abs=1e-15)
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(SQRT3_OVER_2, abs=1e-15)
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8, abs=1e-15)


def test_returned_values_are_clipped_into_range_and_never_negative_zero() -> None:
    for r in (pearson([1, 2, 3], [2, 4, 6]), pearson([1, 2, 3], [6, 4, 2])):
        assert -1.0 <= r <= 1.0


def test_clip_policy() -> None:
    assert _finish_correlation(1.0 + 1e-12) == 1.0
    assert _finish_correlation(-1.0 - 1e-12) == -1.0
    assert str(_finish_correlation(-0.0)) == "0.0"
    for bad in (1.0 + 1e-6, -1.5, math.nan, math.inf):
        with pytest.raises(ComputationError) as e:
            _finish_correlation(bad)
        assert e.value.code == "correlation_postcondition"


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_constant_and_insufficient_inputs_are_undefined_with_reason(method: str) -> None:
    assert correlate([1, 1, 1], [1, 2, 3], method) == (None, ("constant_input", ["x"]))
    assert correlate([1, 2, 3], [5, 5, 5], method) == (None, ("constant_input", ["y"]))
    assert correlate([2, 2, 2], [5, 5, 5], method) == (None, ("constant_input", ["x", "y"]))
    assert correlate([1, 2], [1, 2], method) == (None, ("insufficient_data", []))
    assert correlate([], [], method) == (None, ("insufficient_data", []))
    with pytest.raises(ValueError):
        correlate([1, 2, 3], [1, 2], method)


def test_near_constant_pearson_is_flagged_by_a_documented_relative_criterion() -> None:
    eps = 2.0**-50  # relative spread 2^-49, below 2^-40
    x = [1.0, 1.0 + eps, 1.0 + 2 * eps]
    assert correlate(x, [1.0, 2.0, 3.5], "pearson") == (None, ("near_constant_input", ["x"]))
    # the same data are fine for Spearman: ranks do not depend on magnitudes
    assert spearman(x, [1.0, 2.0, 3.5]) == pytest.approx(1.0)
    # just above the threshold the coefficient is computed, and agrees with the exact oracle
    delta = 2.0**-30
    x2 = [1.0, 1.0 + delta, 1.0 + 3 * delta]
    y2 = [1.0, 2.0, 2.5]
    assert 2 * delta > NEAR_CONSTANT_RELATIVE_SPREAD
    assert abs(Decimal(pearson(x2, y2)) - exact_pearson_decimal(x2, y2)) <= Decimal("1e-12")


def test_small_and_large_magnitudes_are_not_rejected_or_silently_wrong() -> None:
    y = [1.0, 2.0, 3.5, 3.0]
    for scale in (1e-300, 1e-200, 1e-100, 1.0, 1e100, 1e200, 1e300):
        x = [scale * v for v in (1.0, 2.0, 4.0, 3.0)]
        assert abs(Decimal(pearson(x, y)) - exact_pearson_decimal(x, y)) <= Decimal("1e-12")
        assert abs(Decimal(pearson(y, x)) - exact_pearson_decimal(y, x)) <= Decimal("1e-12")
    mixed = [1e-300, 1.0, 1e300, 5.0]
    assert abs(Decimal(pearson(mixed, y)) - exact_pearson_decimal(mixed, y)) <= Decimal("1e-12")


def test_stdlib_alone_is_wrong_at_extreme_scales_which_is_why_the_guard_exists() -> None:
    x = [1e200, 2e200, 4e200]
    y = [1.0, 2.0, 3.5]
    assert statistics.correlation(x, y) == 0.0  # silently wrong (overflow to inf)
    assert pearson(x, y) > 0.99
    with pytest.raises(statistics.StatisticsError):  # false "constant" report (underflow)
        statistics.correlation([1e-200, 2e-200, 4e-200], y)
    assert pearson([1e-200, 2e-200, 4e-200], y) > 0.99


def test_exact_symmetries_and_scaling_invariants() -> None:
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(40)]
    y = [0.5 * a + rng.gauss(0, 1) for a in x]
    base = pearson(x, y)
    assert pearson(y, x) == base  # symmetric
    assert pearson([-v for v in x], y) == -base  # sign flip
    assert pearson([v * 2.0**40 for v in x], [v * 2.0**-33 for v in y]) == base  # exact scaling
    assert spearman(y, x) == spearman(x, y)
    assert spearman([math.exp(v) for v in x], y) == spearman(x, y)  # monotone transform
    order = list(range(40))
    rng.shuffle(order)
    # pairing is preserved, only row order changes; exact-rational ranks make Spearman identical
    assert spearman([x[i] for i in order], [y[i] for i in order]) == spearman(x, y)


def _random_pair(rng: random.Random) -> tuple[list[float], list[float]]:
    n = rng.randint(3, 60)
    sx = 10.0 ** rng.randint(-6, 6)
    sy = 10.0 ** rng.randint(-6, 6)
    ox = rng.choice([0.0, 1.0, 1e3, 1e6]) * sx
    oy = rng.choice([0.0, 1.0, 1e3, 1e6]) * sy
    rho = rng.uniform(-1, 1)
    x = [rng.gauss(0, 1) for _ in range(n)]
    y = [rho * a + math.sqrt(1 - rho * rho) * rng.gauss(0, 1) for a in x]
    if rng.random() < 0.3:  # introduce ties
        x = [round(v, 1) for v in x]
        y = [round(v, 1) for v in y]
    return [ox + sx * v for v in x], [oy + sy * v for v in y]


def test_random_pairs_against_exact_oracles() -> None:
    rng = random.Random(20261125)
    worst_p = worst_s = 0.0
    n_p = n_s = 0
    for _ in range(300):
        x, y = _random_pair(rng)
        value, undefined = correlate(x, y, "pearson")
        if value is not None:
            worst_p = max(worst_p, float(abs(Decimal(value) - exact_pearson_decimal(x, y))))
            n_p += 1
        else:
            assert undefined is not None  # constants created by rounding ties are allowed
        value, undefined = correlate(x, y, "spearman")
        if value is not None:
            worst_s = max(worst_s, float(abs(Decimal(value) - exact_spearman_decimal(x, y))))
            n_s += 1
    assert n_p > 250 and n_s > 250
    assert worst_p <= 1e-12
    assert worst_s <= 1e-12


def run_relate(csv_text: str, method: str = "pearson", limits: Limits | None = None) -> dict:
    ds = make_dataset(csv_text, {"x": "numeric", "y": "numeric"})
    plan = parse_plan_json(
        '{"schema_version":"axiom.plan/1","operation":"relate","x":"x","y":"y","method":"'
        + method
        + '"}'
    )
    return run(ds, plan, limits or Limits()).core


def test_end_to_end_exclusions_are_counted_and_attributed() -> None:
    core = run_relate("x,y\n1,2\n2,\n,3\n,\n3,6\n4,8\n")
    res = core["result"]
    assert res["n_rows_total"] == 6 and res["n_excluded"] == 3
    assert res["n_used"] == 3
    assert (res["excluded_missing_y_only"], res["excluded_missing_x_only"]) == (1, 1)
    assert res["excluded_missing_both"] == 1
    assert res["excluded"]["row_ordinals"] == [2, 3, 4]
    assert res["coefficient"] == pytest.approx(1.0, abs=1e-15)
    assert res["undefined"] is None and res["minimum_n"] == 3
    codes = {w["code"] for w in core["warnings"]}
    assert {"not_causal", "linear_only", "small_sample"} <= codes


def test_end_to_end_undefined_has_structured_reason() -> None:
    res = run_relate("x,y\n1,5\n2,5\n3,5\n")["result"]
    assert res["coefficient"] is None
    assert res["undefined"] == {"code": "constant_input", "columns": ["y"]}
    res = run_relate("x,y\n1,5\n2,\n")["result"]
    assert res["undefined"] == {"code": "insufficient_data", "columns": []}
    assert res["n_used"] == 1


def test_no_pvalues_strength_labels_or_direction_categories() -> None:
    res = run_relate("x,y\n1,2\n2,4\n3,5\n4,9\n5,9\n6,12\n7,13\n8,13\n9,15\n10,19\n")["result"]
    for forbidden in ("p_value", "pvalue", "strength", "direction", "ci", "interpretation"):
        assert forbidden not in res
    assert res["coefficient"] > 0.9


def test_excluded_ordinals_are_capped_and_flagged() -> None:
    rows = "".join(f"{i},\n" for i in range(1, 8)) + "1,1\n2,2\n3,4\n"
    res = run_relate("x,y\n" + rows, limits=Limits(max_reported_ordinals=3))["result"]
    assert res["n_excluded"] == 7
    assert res["excluded"] == {"row_ordinals": [1, 2, 3], "row_ordinals_truncated": True}


def test_spearman_end_to_end_with_ties() -> None:
    res = run_relate("x,y\n1,1\n1,2\n2,3\n", "spearman")["result"]
    assert res["coefficient"] == pytest.approx(SQRT3_OVER_2, abs=1e-15)


def test_repeated_runs_identical_and_hash_changes_with_data() -> None:
    a = run_relate("x,y\n1,2\n2,4\n3,5\n")
    b = run_relate("x,y\n1,2\n2,4\n3,5\n")
    c = run_relate("x,y\n1,2\n2,4\n3,6\n")
    assert a == b
    assert a["inputs"]["dataset_sha256"] != c["inputs"]["dataset_sha256"]
    assert a["result"]["coefficient"] != c["result"]["coefficient"]
