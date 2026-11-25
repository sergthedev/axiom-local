"""Inspect: hand fixtures, exact-arithmetic oracles, extreme-range behavior, invariants.

Accuracy criteria (ADR 0004). mean, sample variance, sample SD: <= 1 ulp from the exact value of
the parsed binary64 inputs. Type-7 quantiles: first registered as <= 2 ulp of the *result*; that
criterion FAILED (4.0 ulp) on seeded random data because it ignores cancellation in
``a + t*(b-a)``. It was replaced by the operand-scaled bound below (<= 1 ulp of
max(|a|, |b|, |result|)), derived from the interpolation formula, not from the failing value. Both
the failure and the revision are recorded in docs/evidence/DEVELOPMENT_LOG.md.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from _helpers import make_dataset
from axiom import parse_plan_json, run
from axiom.kernels import describe
from oracles import (
    exact_mean,
    exact_quantile_type7,
    exact_sample_sd_decimal,
    exact_sample_variance,
    frac_to_decimal,
    ulp_error,
)

QUARTILES = (Fraction(1, 4), Fraction(3, 4))


def inspect(csv_text: str, spec: dict[str, str], columns: list[str]) -> dict:
    plan = parse_plan_json(
        '{"schema_version":"axiom.plan/1","operation":"inspect","columns":'
        + str(columns).replace("'", '"')
        + "}"
    )
    return run(make_dataset(csv_text, spec), plan).core["result"]


def test_hand_fixture_known_answers() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats, undefined = describe(values)
    assert undefined == {}
    assert stats == {
        "min": 2.0,
        "max": 9.0,
        "mean": 5.0,
        "median": 4.5,
        "sample_variance": 32 / 7,
        "sample_sd": math.sqrt(32 / 7),
        "q1": 4.0,
        "q3": 5.5,
        "iqr": 1.5,
    }


def test_end_to_end_counts_missing_and_text_columns() -> None:
    res = inspect("id,v\nA,1\nB,\nC,3\nD,5\n", {"id": "text", "v": "numeric"}, ["v", "id"])
    v, ident = res["columns"]
    assert (v["n_total"], v["n_valid"], v["n_missing"]) == (4, 3, 1)
    assert v["stats"]["mean"] == 3.0 and v["stats"]["median"] == 3.0
    assert v["stats"]["sample_sd"] == 2.0
    assert ident["kind"] == "text" and "stats" not in ident
    assert (ident["n_valid"], ident["n_missing"]) == (4, 0)


def test_empty_and_single_value_columns_have_explicit_reasons() -> None:
    res = inspect("v\n\n,\n".replace(",\n", "\n"), {"v": "numeric"}, ["v"])
    assert res["columns"][0]["n_valid"] == 0
    stats0, und0 = describe([])
    assert all(v is None for v in stats0.values())
    assert set(und0.values()) == {"no_valid_values"}
    stats1, und1 = describe([3.5])
    assert stats1["mean"] == stats1["median"] == stats1["q1"] == stats1["q3"] == 3.5
    assert stats1["sample_sd"] is None and stats1["sample_variance"] is None
    assert und1 == {"sample_variance": "requires_n_ge_2", "sample_sd": "requires_n_ge_2"}


def test_constant_data_has_true_zero_dispersion() -> None:
    stats, undefined = describe([7.25] * 10)
    assert stats["sample_sd"] == 0.0 and stats["sample_variance"] == 0.0 and stats["iqr"] == 0.0
    assert undefined == {}


def _random_dataset(rng: random.Random) -> list[float]:
    n = rng.randint(2, 60)
    scale = 10.0 ** rng.randint(-8, 8)
    offset = rng.choice([0.0, 1.0, 1e3, 1e6]) * scale
    return [offset + rng.gauss(0, 1) * scale for _ in range(n)]


def test_random_datasets_against_exact_oracles() -> None:
    rng = random.Random(20261125)
    worst = {"mean": 0.0, "variance": 0.0, "sd": 0.0, "q": 0.0, "q_result": 0.0}
    for _ in range(300):
        values = _random_dataset(rng)
        stats, undefined = describe(values)
        assert undefined == {}
        mean = exact_mean(values)
        variance = exact_sample_variance(values)
        worst["mean"] = max(worst["mean"], ulp_error(stats["mean"], frac_to_decimal(mean)))
        worst["variance"] = max(
            worst["variance"], ulp_error(stats["sample_variance"], frac_to_decimal(variance))
        )
        worst["sd"] = max(
            worst["sd"], ulp_error(stats["sample_sd"], exact_sample_sd_decimal(values))
        )
        for key, q in zip(("q1", "q3"), QUARTILES, strict=True):
            exact_q = frac_to_decimal(exact_quantile_type7(values, q))
            s = sorted(values)
            h = (len(s) - 1) * q
            lo, hi = math.floor(h), min(math.floor(h) + 1, len(s) - 1)
            scale = max(abs(s[lo]), abs(s[hi]), abs(stats[key]))
            err = abs(Decimal(stats[key]) - exact_q) / Decimal(math.ulp(scale))
            worst["q"] = max(worst["q"], float(err))
            if stats[key] != 0:  # exact rounding: <= 0.5 ulp of the result itself
                worst["q_result"] = max(worst["q_result"], ulp_error(stats[key], exact_q))
    assert worst["mean"] <= 1.0
    assert worst["variance"] <= 1.0
    assert worst["sd"] <= 1.0
    assert worst["q"] <= 1.0
    assert worst["q_result"] <= 0.5  # quartiles are exact values rounded once


def test_median_matches_exact_midpoint() -> None:
    for values in ([1.0, 2.0], [0.1, 0.2], [1e-320, 3e-320], [-5.0, 5.0], [1.0, 2.0, 10.0, 11.0]):
        stats, _ = describe(values)
        s = sorted(values)
        n = len(s)
        exact = (
            Fraction(s[n // 2]) if n % 2 else (Fraction(s[n // 2 - 1]) + Fraction(s[n // 2])) / 2
        )
        assert stats["median"] == float(exact)


def test_extreme_magnitudes_do_not_overflow_the_mean_or_median() -> None:
    big = 1.7976931348623157e308
    stats, undefined = describe([big, big, big])
    assert stats["mean"] == big and stats["median"] == big and stats["sample_sd"] == 0.0
    stats, undefined = describe([big, big])
    assert stats["median"] == big  # naive (a + b) / 2 would be inf
    stats, undefined = describe([-big, big])
    assert stats["mean"] == 0.0 and stats["median"] == 0.0


BIG = 1.7976931348623157e308


def test_full_range_pair_has_exact_representable_quartiles_and_iqr() -> None:
    # Regression for a defect in the first implementation: NumPy's lerp overflowed in b - a, so the
    # first version reported IQR overflow although q1 = -M/2, q3 = M/2 and IQR = M are all
    # representable.
    stats, undefined = describe([-BIG, BIG])
    assert stats["q1"] == -BIG / 2 and stats["q3"] == BIG / 2 and stats["iqr"] == BIG
    assert stats["median"] == 0.0 and stats["mean"] == 0.0
    assert "iqr" not in undefined and "q1" not in undefined and "q3" not in undefined
    # the variance and SD are genuinely unrepresentable: exact SD = sqrt(2) * M > max float
    assert undefined["sample_variance"] == "overflow_exceeds_float64"
    assert undefined["sample_sd"] == "overflow_exceeds_float64"
    for value in stats.values():
        assert value is None or math.isfinite(value)


def test_true_iqr_overflow_is_distinguished_from_intermediate_overflow() -> None:
    stats, undefined = describe([-BIG, -BIG, BIG, BIG])
    assert stats["q1"] == -BIG and stats["q3"] == BIG  # both representable
    assert stats["iqr"] is None and undefined["iqr"] == "overflow_exceeds_float64"  # exact IQR = 2M
    assert stats["median"] == 0.0 and stats["mean"] == 0.0


def test_overflowing_sd_but_representable_variance_boundary() -> None:
    stats, undefined = describe([-1e155, 1e155])
    assert undefined["sample_variance"] == "overflow_exceeds_float64"
    assert "sample_sd" not in undefined
    assert math.isclose(stats["sample_sd"], 1e155 * math.sqrt(2), rel_tol=1e-15)
    stats, undefined = describe([-1e200, 1e200])
    assert undefined["sample_variance"] == "overflow_exceeds_float64"
    assert math.isclose(stats["sample_sd"], 1e200 * math.sqrt(2), rel_tol=1e-15)


@pytest.mark.parametrize(
    "values",
    [
        [-BIG, BIG],
        [-BIG, -BIG, BIG, BIG],
        [BIG, BIG, BIG],
        [-BIG, 0.0, BIG],
        [-1.8041831441410014e-08, 5.1370316991998964e-09],  # cancellation case from EXP-001
        [-1.8041831441410014e-08, 5.1370316991998964e-09, 3.0, -7.0, 1e-300],
        [1e300, -1e300, 1e-300, 2.5e-300, -3.5e300],
    ],
)
def test_quartiles_and_median_are_correctly_rounded_exact_values(values: list[float]) -> None:
    stats, undefined = describe(values)
    s = sorted(values)
    for key, q in (("q1", Fraction(1, 4)), ("q3", Fraction(3, 4))):
        exact = exact_quantile_type7(values, q)
        assert stats[key] == float(exact) and key not in undefined
    n = len(s)
    exact_median = (
        Fraction(s[n // 2]) if n % 2 else (Fraction(s[n // 2 - 1]) + Fraction(s[n // 2])) / 2
    )
    assert stats["median"] == float(exact_median)


def test_quartiles_agree_with_numpy_linear_semantics_on_ordinary_data() -> None:
    # NumPy's method="linear" is the semantic reference for type 7 (REF-009), used here only as a
    # cross-check (its lerp is not exactly rounded, so allow 2 ulp of the operand scale).
    import numpy as np

    rng = random.Random(41)
    for _ in range(200):
        values = [rng.gauss(0, 1) * 10.0 ** rng.randint(-3, 3) for _ in range(rng.randint(1, 40))]
        stats, _ = describe(values)
        expected = np.quantile(np.asarray(values), [0.25, 0.75], method="linear")
        scale = max(abs(min(values)), abs(max(values)))
        for key, ref in zip(("q1", "q3"), expected, strict=True):
            assert abs(stats[key] - float(ref)) <= 2 * math.ulp(scale)


def test_exact_nonzero_values_that_round_to_zero_are_undefined_not_false_zero() -> None:
    tiny = 5e-324  # smallest subnormal
    stats, undefined = describe([0.0, tiny])
    assert stats["median"] is None and undefined["median"] == "underflow_to_zero"  # exact tiny/2
    assert stats["mean"] is None and undefined["mean"] == "underflow_to_zero"
    assert stats["q1"] is None and undefined["q1"] == "underflow_to_zero"  # exact tiny/4
    assert stats["q3"] == tiny  # exact 3*tiny/4 rounds to tiny
    assert stats["iqr"] is None and undefined["iqr"] == "underflow_to_zero"  # exact tiny/2
    assert stats["min"] == 0.0 and stats["max"] == tiny  # exact order statistics are never flagged
    stats, undefined = describe([tiny, 0.0, 0.0])
    assert undefined["mean"] == "underflow_to_zero" and stats["median"] == 0.0  # true zero median
    # a true zero is a value, not an underflow
    stats, undefined = describe([-tiny, tiny])
    assert stats["mean"] == 0.0 and stats["median"] == 0.0 and "mean" not in undefined
    stats, undefined = describe([0.0, 0.0, 0.0])
    assert stats["mean"] == 0.0 and undefined == {}
    # consistent semantics with variance: exact nonzero variance that rounds to zero
    _, undefined = describe([0.0, tiny])
    assert undefined["sample_variance"] == "underflow_to_zero"


def test_subnormal_inputs_with_representable_results() -> None:
    values = [5e-324, 1e-323, 1.5e-323, 2e-323]
    stats, undefined = describe(values)
    assert stats["min"] == 5e-324 and stats["max"] == 2e-323
    assert stats["median"] == float((Fraction(1e-323) + Fraction(1.5e-323)) / 2)
    assert undefined.get("mean") is None


def test_underflow_is_not_reported_as_false_zero() -> None:
    values = [1e-200, 2e-200, 3e-200]
    stats, undefined = describe(values)
    assert undefined["sample_variance"] == "underflow_to_zero"
    assert stats["sample_variance"] is None
    assert ulp_error(stats["sample_sd"], exact_sample_sd_decimal(values)) <= 1.0
    assert stats["sample_sd"] > 0.0


def test_exact_inputs_are_permutation_and_integer_shift_invariant() -> None:
    rng = random.Random(7)
    base = [float(rng.randint(-1000, 1000)) for _ in range(101)]
    ref, _ = describe(base)
    shuffled = base[:]
    rng.shuffle(shuffled)
    assert describe(shuffled)[0] == ref
    shifted, _ = describe([v + 2.0**20 for v in base])  # exact in binary64
    assert shifted["sample_sd"] == ref["sample_sd"] and shifted["iqr"] == ref["iqr"]
    assert shifted["mean"] == ref["mean"] + 2.0**20
    scaled, _ = describe([v * 2.0**30 for v in base])  # exact scaling
    assert scaled["sample_sd"] == ref["sample_sd"] * 2.0**30
    assert scaled["mean"] == ref["mean"] * 2.0**30
    assert scaled["median"] == ref["median"] * 2.0**30


def test_negative_zero_never_appears() -> None:
    stats, _ = describe([-1.0, 1.0])
    assert str(stats["mean"]) == "0.0" and str(stats["median"]) == "0.0"


def test_repeated_runs_are_identical() -> None:
    ds = make_dataset("v\n1.5\n2.5\n4\n", {"v": "numeric"})
    plan = parse_plan_json(
        '{"schema_version":"axiom.plan/1","operation":"inspect","columns":["v"]}'
    )
    first = run(ds, plan)
    for _ in range(5):
        again = run(ds, plan)
        assert again.canonical_json == first.canonical_json
        assert again.result_sha256 == first.result_sha256


def test_decimal_import_is_used_for_decimal_reference() -> None:
    assert Decimal(1) == 1  # guard against accidental removal of the Decimal oracle path
