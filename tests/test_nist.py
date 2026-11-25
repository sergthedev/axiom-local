"""NIST StRD univariate summary-statistics datasets (PiDigits, NumAcc1-4).

Pre-registered criteria (ADR 0004; fixed before measurement; see docs/evidence/experiments):

* (B) versus the exact value of the *parsed binary64 inputs*: mean and sample SD within 1 ulp.
* (A) versus the NIST certified *decimal* values: the certified decimals are not exactly
  representable, so the bound is derived from input conversion. With u = 2^-53 and m = max|x|:
  mean error <= u*m + ulp(certified); SD error <= u*m*sqrt(n/(n-1)) + certified precision + ulp.
"""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from _helpers import make_dataset
from axiom import parse_plan_json, run
from oracles import (
    NIST_DIR,
    exact_mean,
    exact_sample_sd_decimal,
    frac_to_decimal,
    load_nist,
    ulp_error,
)

MANIFEST, TOKENS = load_nist()
NAMES = sorted(MANIFEST["datasets"])
U = 2.0**-53


def _certified_precision(text: str, exact: bool) -> Decimal:
    if exact:
        return Decimal(0)
    return Decimal(1).scaleb(Decimal(text).as_tuple().exponent) / 2  # half unit in last digit


def decimal_reference(name: str) -> tuple[Decimal, Decimal]:
    """Exact mean and sample SD of the decimal source tokens (independent of binary64)."""
    with localcontext() as ctx:
        ctx.prec = 80
        vals = [Decimal(t) for t in TOKENS[name]]
        n = len(vals)
        mean = sum(vals) / n
        var = sum(((v - mean) ** 2 for v in vals), Decimal(0)) / (n - 1)
        return mean, var.sqrt()


@pytest.mark.parametrize("name", NAMES)
def test_vendored_files_match_manifest_hashes_and_counts(name: str) -> None:
    meta = MANIFEST["datasets"][name]
    assert hashlib.sha256((NIST_DIR / meta["file"]).read_bytes()).hexdigest() == meta["sha256"]
    assert len(TOKENS[name]) == meta["n_observations"]


@pytest.mark.parametrize("name", NAMES)
def test_decimal_reference_reproduces_the_certified_values(name: str) -> None:
    meta = MANIFEST["datasets"][name]
    mean, sd = decimal_reference(name)
    assert abs(mean - Decimal(meta["certified_mean"])) <= _certified_precision(
        meta["certified_mean"], meta["certified_mean_is_exact"]
    ) + Decimal("1e-40")
    assert abs(sd - Decimal(meta["certified_sd_n_minus_1"])) <= _certified_precision(
        meta["certified_sd_n_minus_1"], meta["certified_sd_is_exact"]
    ) + Decimal("1e-40")


@pytest.mark.parametrize("name", NAMES)
def test_axiom_inspect_against_nist_and_exact_binary64_oracle(name: str) -> None:
    meta = MANIFEST["datasets"][name]
    csv_text = "y\n" + "\n".join(TOKENS[name]) + "\n"
    ds = make_dataset(csv_text, {"y": "numeric"})
    plan = parse_plan_json(
        '{"schema_version":"axiom.plan/1","operation":"inspect","columns":["y"]}'
    )
    stats = run(ds, plan).core["result"]["columns"][0]["stats"]
    values = list(ds.columns["y"])
    n, m = len(values), max(abs(v) for v in values)

    # (B) exact oracle of the parsed binary64 values
    assert ulp_error(stats["mean"], frac_to_decimal(exact_mean(values))) <= 1.0
    assert ulp_error(stats["sample_sd"], exact_sample_sd_decimal(values)) <= 1.0

    # (A) certified decimals, bound derived from input conversion
    c_mean, c_sd = Decimal(meta["certified_mean"]), Decimal(meta["certified_sd_n_minus_1"])
    mean_bound = Decimal(U * m) + Decimal(math.ulp(float(c_mean)))
    sd_bound = (
        Decimal(U * m * math.sqrt(n / (n - 1)))
        + _certified_precision(meta["certified_sd_n_minus_1"], meta["certified_sd_is_exact"])
        + Decimal(math.ulp(float(c_sd)))
    )
    assert abs(Decimal(stats["mean"]) - c_mean) <= mean_bound
    assert abs(Decimal(stats["sample_sd"]) - c_sd) <= sd_bound


def test_all_five_datasets_were_evaluated() -> None:
    assert NAMES == ["NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4", "PiDigits"]
    assert Fraction(1, 3) == Fraction(2, 6)  # sanity: Fraction available for oracles
