"""Validated-plan boundary: revalidation at bind and execute, execution-time limits, bounded input.

These tests make no claim of protection against arbitrary local Python code; they check that the
engine does not trust unchecked copies, forged bound plans, stale caps or pathological plan input.
"""

from __future__ import annotations

import json

import pytest

from _helpers import make_dataset
from axiom import (
    BoundPlan,
    ClusterPlan,
    InspectPlan,
    Limits,
    PlanError,
    bind_plan,
    execute,
    parse_plan_json,
    parse_plan_object,
    run,
)
from axiom.canonical import canonical_sha256
from axiom.plan import plan_sha256, plan_to_dict

V = '"schema_version":"axiom.plan/1"'
CSV = "g,x,y\na,1,2\nb,2,4\nc,3,5\nd,4,9\ne,5,10\nf,6,11\n"
SPEC = {"g": "text", "x": "numeric", "y": "numeric"}
CLUSTER = "{" + V + ',"operation":"cluster","features":["x","y"],"k":3}'
INSPECT = "{" + V + ',"operation":"inspect","columns":["x","y"]}'


def ds():
    return make_dataset(CSV, SPEC)


def forge(bound: BoundPlan, plan) -> BoundPlan:
    """A forged bound plan whose digest was recomputed to match the forged plan."""
    digest = canonical_sha256({"policy_version": bound.policy_version, "plan": plan_to_dict(plan)})
    return BoundPlan(
        plan=plan,
        plan_sha256=digest,
        dataset_raw_sha256=bound.dataset_raw_sha256,
        dataset_logical_sha256=bound.dataset_logical_sha256,
        policy_version=bound.policy_version,
    )


# ---- unchecked copies and foreign objects ----


def test_unchecked_model_copy_is_rejected_at_binding() -> None:
    valid = parse_plan_json(CLUSTER)
    for update in (
        {"k": 1},
        {"n_init": 10_000},
        {"tol": float("nan")},
        {"seed": -5},
    ):
        bad = valid.model_copy(update=update)  # skips validation
        assert isinstance(bad, ClusterPlan)
        with pytest.raises(PlanError) as e:
            bind_plan(ds(), bad)
        assert e.value.code == "invalid_plan"
    bad_cols = parse_plan_json(INSPECT).model_copy(update={"columns": ("x", "x")})
    with pytest.raises(PlanError) as e:
        bind_plan(ds(), bad_cols)
    assert e.value.code == "invalid_plan"


def test_model_construct_subclass_and_foreign_objects_are_rejected() -> None:
    constructed = ClusterPlan.model_construct(
        schema_version="axiom.plan/1", operation="cluster", features=("x",), k=0
    )
    with pytest.raises(PlanError) as e:
        bind_plan(ds(), constructed)
    assert e.value.code == "invalid_plan"
    incomplete = InspectPlan.model_construct(operation="inspect")  # required fields missing
    with pytest.raises(PlanError):
        bind_plan(ds(), incomplete)

    class Sneaky(InspectPlan):
        pass

    sneaky = Sneaky(schema_version="axiom.plan/1", operation="inspect", columns=("x",))
    with pytest.raises(PlanError):
        bind_plan(ds(), sneaky)
    for foreign in ({"operation": "inspect"}, None, "inspect", 3):
        with pytest.raises(PlanError):
            bind_plan(ds(), foreign)  # type: ignore[arg-type]


def test_binding_returns_a_freshly_validated_plan() -> None:
    plan = parse_plan_json(CLUSTER)
    bound = bind_plan(ds(), plan)
    assert bound.plan == plan and bound.plan is not plan
    # a schema-valid but over-cap k is a cap error, not a schema error
    with pytest.raises(PlanError) as e:
        bind_plan(ds(), plan.model_copy(update={"k": 10_000_000}))
    assert e.value.code == "k_too_large"


# ---- forged bound plans ----


def test_forged_bound_plan_with_recomputed_digest_and_invalid_plan_is_rejected_at_execution() -> (
    None
):
    d = ds()
    bound = bind_plan(d, parse_plan_json(CLUSTER))
    for update in ({"k": 1}, {"max_iter": 10**9}, {"features": ("x", "x")}):
        forged = forge(bound, bound.plan.model_copy(update=update))
        assert forged.plan_sha256 != bound.plan_sha256
        with pytest.raises(PlanError) as e:
            execute(d, forged)
        assert e.value.code == "invalid_plan"


def test_forged_bound_plan_referencing_unknown_columns_is_rejected_at_execution() -> None:
    d = ds()
    bound = bind_plan(d, parse_plan_json(INSPECT))
    forged = forge(bound, bound.plan.model_copy(update={"columns": ("nope",)}))
    with pytest.raises(PlanError) as e:
        execute(d, forged)
    assert e.value.code == "unknown_column"


def test_forged_bound_plan_that_is_valid_executes_because_a_digest_is_not_a_signature() -> None:
    d = ds()
    bound = bind_plan(d, parse_plan_json(INSPECT))
    other = parse_plan_json("{" + V + ',"operation":"inspect","columns":["y"]}')
    result = execute(d, forge(bound, other))
    # documented semantics: the digest identifies the computation, it does not authorize it
    assert result.core["inputs"]["plan_sha256"] == plan_sha256(other)
    assert [c["name"] for c in result.core["result"]["columns"]] == ["y"]


def test_digest_mismatch_and_foreign_arguments_are_rejected() -> None:
    d = ds()
    bound = bind_plan(d, parse_plan_json(INSPECT))
    tampered = BoundPlan(
        plan=bound.plan,
        plan_sha256="0" * 64,
        dataset_raw_sha256=bound.dataset_raw_sha256,
        dataset_logical_sha256=bound.dataset_logical_sha256,
        policy_version=bound.policy_version,
    )
    with pytest.raises(PlanError) as e:
        execute(d, tampered)
    assert e.value.code == "plan_hash_mismatch"
    with pytest.raises(PlanError):
        execute(d, bound.plan)  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        execute("not a dataset", bound)  # type: ignore[arg-type]


# ---- limits supplied at execution ----


def test_stricter_limits_at_execution_cannot_be_bypassed_by_an_earlier_binding() -> None:
    d = ds()
    bound = bind_plan(d, parse_plan_json(CLUSTER))  # default limits
    assert execute(d, bound).core["operation"] == "cluster"
    cases = [
        (Limits(max_k=2), "k_too_large"),
        (Limits(max_cluster_features=1), "too_many_features"),
        (Limits(max_cluster_work=10), "cluster_workload_too_large"),
    ]
    for limits, code in cases:
        with pytest.raises(PlanError) as e:
            execute(d, bound, limits)
        assert e.value.code == code
    inspect_bound = bind_plan(d, parse_plan_json(INSPECT))
    with pytest.raises(PlanError) as e:
        execute(d, inspect_bound, Limits(max_columns=1))
    assert e.value.code == "too_many_columns"
    with pytest.raises(PlanError):
        run(d, parse_plan_json(CLUSTER), Limits(max_k=2))


# ---- bounded plan input ----


def test_oversized_deep_and_wide_plan_json_is_rejected_with_structured_errors() -> None:
    limits = Limits()
    big = "{" + V + ',"operation":"inspect","columns":["' + "a" * limits.max_plan_bytes + '"]}'
    with pytest.raises(PlanError) as e:
        parse_plan_json(big)
    assert e.value.code == "plan_too_large"
    with pytest.raises(PlanError) as e:
        parse_plan_json("[" * 200_000)  # would exhaust a recursive parser
    assert e.value.code in {"plan_too_deep", "plan_too_large"}
    with pytest.raises(PlanError) as e:
        parse_plan_json("[" * 100 + "]" * 100)
    assert e.value.code == "plan_too_deep"
    wide = (
        "{"
        + V
        + ',"operation":"inspect","columns":['
        + ",".join(f'"c{i}"' for i in range(5_000))
        + "]}"
    )
    assert len(wide) < limits.max_plan_bytes
    with pytest.raises(PlanError) as e:
        parse_plan_json(wide)
    assert e.value.code == "plan_too_large"
    with pytest.raises(PlanError) as e:
        parse_plan_json(INSPECT, Limits(max_plan_bytes=10))
    assert e.value.code == "plan_too_large"


def test_pathological_numbers_and_text_do_not_raise_uncaught_exceptions() -> None:
    huge_int = "{" + V + ',"operation":"cluster","features":["x"],"k":1' + "0" * 5000 + "}"
    with pytest.raises(PlanError) as e:
        parse_plan_json(huge_int)
    assert e.value.code in {"plan_input_rejected", "invalid_plan"}
    for text in ("1e999999999", '{"a": 1e999}', "\ud800", "\x00", '{"k": 1,}', "{" * 5):
        with pytest.raises(PlanError):
            parse_plan_json(text)
    with pytest.raises(PlanError):
        parse_plan_json(b'{"operation": "inspect"}')  # type: ignore[arg-type]
    nested: list = []
    node = nested
    for _ in range(100_000):  # deep Python object, built iteratively
        child: list = []
        node.append(child)
        node = child
    with pytest.raises(PlanError):
        parse_plan_object(nested)
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(PlanError):
        parse_plan_object(cyclic)
    with pytest.raises(PlanError):
        parse_plan_object({"k": 10**5000})


def test_brackets_inside_strings_and_escapes_do_not_count_as_nesting() -> None:
    text = "{" + V + ',"operation":"inspect","columns":["[[[[[[[[[[[[[[[[[[[[ \\" [[[["]}'
    plan = parse_plan_json(text)
    assert plan.columns == ('[[[[[[[[[[[[[[[[[[[[ " [[[[',)
    with pytest.raises(PlanError) as e:
        bind_plan(ds(), plan)
    assert e.value.code == "unknown_column"


# ---- result identity ----


def test_result_bytes_are_fixed_at_construction_and_core_is_a_copy() -> None:
    result = run(ds(), parse_plan_json(INSPECT))
    before = (result.canonical_json, result.result_sha256)
    core = result.core
    core["result"]["columns"][0]["stats"]["mean"] = 999.0
    core["operation"] = "tampered"
    assert (result.canonical_json, result.result_sha256) == before
    assert result.core["operation"] == "inspect" and result.core is not result.core
    assert json.loads(result.canonical_json) == result.core
    with pytest.raises(AttributeError):
        result.canonical_json = "{}"  # type: ignore[misc]
