"""Compare: exact text keys, stable ordering, group counts, shared descriptive kernels."""

from __future__ import annotations

import unicodedata

import pytest

from _helpers import make_dataset
from axiom import ComputationError, parse_plan_json, run
from axiom.kernels import describe
from axiom.limits import Limits

PLAN = '{"schema_version":"axiom.plan/1","operation":"compare","group_by":"g","measure":"v"}'


def compare(csv_text: str, limits: Limits | None = None) -> dict:
    ds = make_dataset(csv_text, {"g": "text", "v": "numeric"})
    return run(ds, parse_plan_json(PLAN), limits or Limits()).core


def groups(core: dict) -> dict[str, dict]:
    return {g["key"]: g for g in core["result"]["groups"]}


def test_hand_calculated_group_summaries_unequal_sizes() -> None:
    csv = "g,v\nb,10\na,1\na,2\na,6\nb,20\nc,5\n"
    g = groups(compare(csv))
    assert g["a"]["n_total"] == 3 and g["a"]["stats"]["mean"] == 3.0
    assert (
        g["a"]["stats"]["median"] == 2.0 and g["a"]["stats"]["sample_variance"] == 7.0
    )  # (4 + 1 + 9) / 2
    assert g["b"]["stats"]["mean"] == 15.0 and g["b"]["stats"]["sample_sd"] == pytest.approx(
        50**0.5
    )
    assert g["c"]["n_valid"] == 1 and g["c"]["stats"]["sample_sd"] is None
    assert g["c"]["undefined"]["sample_sd"] == "requires_n_ge_2"


def test_no_average_of_averages_overall_is_not_reported() -> None:
    res = compare("g,v\na,1\na,1\na,1\nb,10\n")["result"]
    assert "overall" not in res and "mean" not in res  # only per-group summaries


def test_group_order_is_unicode_codepoint_not_input_order_or_locale() -> None:
    csv = "g,v\nz,1\nB,1\nb,1\nÇ,1\nİ,1\nı,1\n_,1\n1,1\n"
    keys = [g["key"] for g in compare(csv)["result"]["groups"]]
    assert keys == sorted(keys) == ["1", "B", "_", "b", "z", "Ç", "İ", "ı"]
    shuffled = "g,v\nı,1\n1,1\nb,1\n_,1\nz,1\nİ,1\nB,1\nÇ,1\n"
    assert [g["key"] for g in compare(shuffled)["result"]["groups"]] == keys


def test_keys_are_exact_no_case_fold_no_strip_no_nfc() -> None:
    nfd = unicodedata.normalize("NFD", "é")
    nfc = unicodedata.normalize("NFC", "é")
    res = compare(f"g,v\nA,1\na,2\n a,3\na ,4\n{nfd},5\n{nfc},6\n")["result"]
    assert res["n_groups"] == 6


def test_missing_group_key_excluded_and_counted_missing_measure_kept_separate() -> None:
    res = compare("g,v\na,1\n,2\na,\nb,\n,\nb,4\n")["result"]
    assert res["n_missing_group_key"] == 2
    assert res["missing_group_key"]["row_ordinals"] == [2, 5]
    a = next(g for g in res["groups"] if g["key"] == "a")
    assert (a["n_total"], a["n_valid"], a["n_missing"]) == (2, 1, 1)
    assert res["n_rows_total"] == 6


def test_group_with_no_valid_values_is_explicitly_undefined() -> None:
    res = compare("g,v\na,\na,\nb,1\n")["result"]
    a = next(g for g in res["groups"] if g["key"] == "a")
    assert a["n_valid"] == 0 and a["n_missing"] == 2
    assert set(a["undefined"].values()) == {"no_valid_values"}


def test_duplicates_are_retained() -> None:
    res = compare("g,v\na,5\na,5\na,5\n")["result"]
    a = res["groups"][0]
    assert a["n_valid"] == 3 and a["stats"]["sample_sd"] == 0.0


def test_group_kernel_is_identical_to_the_inspect_kernel() -> None:
    values = [1.5, 2.25, 9.0, 4.0, 4.0]
    csv = "g,v\n" + "".join(f"a,{v}\n" for v in values)
    assert groups(compare(csv))["a"]["stats"] == describe(values)[0]


def test_small_group_warnings_without_privacy_claim() -> None:
    core = compare("g,v\na,1\na,2\nb,3\n")
    codes = [w["code"] for w in core["warnings"]]
    assert codes.count("small_group") == 2 and "small_groups_not_private" in codes
    big = "g,v\n" + "".join(f"a,{i}\n" for i in range(5))
    assert "small_group" not in [w["code"] for w in compare(big)["warnings"]]
    assert "descriptive_only" in [w["code"] for w in compare(big)["warnings"]]


def test_group_cap_is_enforced_and_causal_language_is_absent() -> None:
    csv = "g,v\n" + "".join(f"k{i},1\n" for i in range(4))
    with pytest.raises(ComputationError) as e:
        compare(csv, Limits(max_groups=3))
    assert e.value.code == "too_many_groups"
    text = str(compare(csv)).lower()
    assert "cause" not in text.replace("not causal", "")


def test_header_only_and_all_missing_groups() -> None:
    res = compare("g,v\n")["result"]
    assert res["n_groups"] == 0 and res["groups"] == []
    res = compare("g,v\n,1\n,2\n")["result"]
    assert res["n_groups"] == 0 and res["n_missing_group_key"] == 2


def test_repeated_runs_identical() -> None:
    csv = "g,v\nb,2\na,1\nb,4\n"
    assert compare(csv) == compare(csv)
