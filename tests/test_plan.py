"""Strict versioned plans and binding to a dataset."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from _helpers import make_dataset
from axiom import PlanError, bind_plan, parse_plan_json, parse_plan_object
from axiom.limits import Limits
from axiom.plan import plan_sha256

V = '"schema_version":"axiom.plan/1"'
CSV = "g,x,y\na,1,2\nb,2,4\n"
SPEC = {"g": "text", "x": "numeric", "y": "numeric"}


def plan_text(body: str) -> str:
    return "{" + V + "," + body + "}"


GOOD = {
    "inspect": plan_text('"operation":"inspect","columns":["x","y"]'),
    "relate": plan_text('"operation":"relate","x":"x","y":"y","method":"pearson"'),
    "compare": plan_text('"operation":"compare","group_by":"g","measure":"x"'),
    "cluster": plan_text('"operation":"cluster","features":["x","y"],"k":2'),
}


@pytest.mark.parametrize("op", sorted(GOOD))
def test_valid_plans_parse(op: str) -> None:
    assert parse_plan_json(GOOD[op]).operation == op


@pytest.mark.parametrize(
    "text",
    [
        plan_text('"operation":"inspect","columns":["x"],"extra":1'),  # unknown field
        plan_text('"operation":"inspect","columns":["x"],"dataset_sha256":"abc"'),  # no model hash
        plan_text('"operation":"inspect","columns":[]'),
        plan_text('"operation":"inspect","columns":["x","x"]'),
        plan_text('"operation":"inspect","columns":[""]'),
        plan_text('"operation":"inspect","columns":"x"'),
        plan_text('"operation":"relate","x":"x","y":"x","method":"pearson"'),
        plan_text('"operation":"relate","x":"x","y":"y","method":"kendall"'),
        plan_text('"operation":"compare","group_by":"g","measure":"g"'),
        plan_text('"operation":"cluster","features":["x"],"k":1'),
        plan_text('"operation":"cluster","features":["x"],"k":"2"'),  # strict: no coercion
        plan_text('"operation":"cluster","features":["x"],"k":2.0'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"seed":-1'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"seed":4294967296'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"tol":0'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"tol":1e-9'),  # below v1 policy
        plan_text('"operation":"cluster","features":["x"],"k":2,"tol":0.1'),  # above v1 policy
        plan_text('"operation":"cluster","features":["x"],"k":2,"n_init":21'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"n_init":0'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"max_iter":1001'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"tol":1e999'),
        plan_text('"operation":"cluster","features":["x"],"k":2,"algorithm":"elkan"'),
        plan_text('"operation":"cluster","features":["x","x"],"k":2'),
        plan_text('"operation":"regress","columns":["x"]'),  # unknown operation
        '{"operation":"inspect","columns":["x"]}',  # missing version
        '{"schema_version":"axiom.plan/2","operation":"inspect","columns":["x"]}',
        "[]",
        "not json",
        "",
    ],
)
def test_invalid_plans_are_rejected(text: str) -> None:
    with pytest.raises(PlanError):
        parse_plan_json(text)


def test_duplicate_keys_and_nonfinite_constants_rejected() -> None:
    with pytest.raises(PlanError) as e:
        parse_plan_json(
            '{"schema_version":"axiom.plan/1","operation":"inspect","columns":["x"],"columns":["y"]}'
        )
    assert e.value.code == "duplicate_json_key"
    with pytest.raises(PlanError) as e:
        parse_plan_json(plan_text('"operation":"cluster","features":["x"],"k":2,"tol":NaN'))
    assert e.value.code == "nonfinite_json_constant"


def test_invalid_plan_error_is_structured() -> None:
    with pytest.raises(PlanError) as e:
        parse_plan_json(plan_text('"operation":"inspect","columns":["x"],"extra":1'))
    assert e.value.code == "invalid_plan"
    json.dumps(e.value.to_dict())  # must be JSON-serializable


def test_parse_plan_object_matches_json_path() -> None:
    obj = json.loads(GOOD["cluster"])
    assert parse_plan_object(obj) == parse_plan_json(GOOD["cluster"])
    with pytest.raises(PlanError):
        parse_plan_object(
            {
                "schema_version": "axiom.plan/1",
                "operation": "inspect",
                "columns": ["x"],
                "z": float("nan"),
            }
        )
    with pytest.raises(PlanError):
        parse_plan_object({"k": object()})


def test_defaults_are_made_explicit_and_hash_is_identical() -> None:
    implicit = parse_plan_json(GOOD["cluster"])
    explicit = parse_plan_json(
        plan_text(
            '"operation":"cluster","features":["x","y"],"k":2,"scaling":"standard_ddof0",'
            '"init":"k-means++","algorithm":"lloyd","seed":42,"n_init":10,"max_iter":300,"tol":0.0001'
        )
    )
    assert plan_sha256(implicit) == plan_sha256(explicit)
    assert plan_sha256(implicit) != plan_sha256(
        parse_plan_json(GOOD["cluster"].replace('"k":2', '"k":3'))
    )


def test_plans_are_frozen() -> None:
    plan = parse_plan_json(GOOD["relate"])
    with pytest.raises(ValidationError):
        plan.x = "y"  # type: ignore[misc]


def test_bind_computes_hashes_from_the_snapshot() -> None:
    ds = make_dataset(CSV, SPEC)
    bound = bind_plan(ds, parse_plan_json(GOOD["relate"]))
    assert bound.dataset_raw_sha256 == ds.raw_sha256
    assert bound.dataset_logical_sha256 == ds.logical_sha256
    assert bound.plan_sha256 == plan_sha256(bound.plan)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ('"operation":"inspect","columns":["nope"]', "unknown_column"),
        ('"operation":"relate","x":"g","y":"y","method":"pearson"', "wrong_column_kind"),
        ('"operation":"compare","group_by":"x","measure":"y"', "wrong_column_kind"),
        ('"operation":"compare","group_by":"g","measure":"g"', None),
        ('"operation":"cluster","features":["g"],"k":2', "wrong_column_kind"),
    ],
)
def test_bind_semantic_validation(body: str, code: str | None) -> None:
    ds = make_dataset(CSV, SPEC)
    with pytest.raises(PlanError) as e:
        bind_plan(ds, parse_plan_json(plan_text(body)))
    if code:
        assert e.value.code == code


def test_bind_caps() -> None:
    ds = make_dataset(CSV, SPEC)
    with pytest.raises(PlanError) as e:
        bind_plan(ds, parse_plan_json(GOOD["cluster"]), Limits(max_k=1))
    assert e.value.code == "k_too_large"
    with pytest.raises(PlanError) as e:
        bind_plan(ds, parse_plan_json(GOOD["cluster"]), Limits(max_cluster_features=1))
    assert e.value.code == "too_many_features"
    with pytest.raises(PlanError):
        bind_plan(ds, {"operation": "inspect"})  # type: ignore[arg-type]
