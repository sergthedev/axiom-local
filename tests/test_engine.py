"""Engine-level behavior: plan binding, mutation, serialization, repeatability, isolation."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from _helpers import make_dataset
from axiom import (
    PlanError,
    bind_plan,
    environment_info,
    execute,
    load_csv_bytes,
    parse_plan_json,
    run,
)
from axiom.canonical import canonical_dumps
from axiom.plan import BoundPlan

SPEC = {"g": "text", "x": "numeric", "y": "numeric"}
CSV = "g,x,y\na,1,2\na,2,4\nb,3,5\nb,4,9\nc,5,10\nc,6,11\n"
_V = '"schema_version":"axiom.plan/1"'
PLANS = {
    "inspect": "{" + _V + ',"operation":"inspect","columns":["x","y"]}',
    "relate": "{" + _V + ',"operation":"relate","x":"x","y":"y","method":"pearson"}',
    "compare": "{" + _V + ',"operation":"compare","group_by":"g","measure":"y"}',
    "cluster": "{" + _V + ',"operation":"cluster","features":["x","y"],"k":2}',
}


@pytest.mark.parametrize("op", sorted(PLANS))
def test_every_operation_returns_strict_json_with_four_identities(op: str) -> None:
    result = run(make_dataset(CSV, SPEC), parse_plan_json(PLANS[op]))
    core = json.loads(result.canonical_json, parse_constant=lambda c: pytest.fail(c))
    assert core == json.loads(json.dumps(result.core, allow_nan=False))
    inputs = core["inputs"]
    assert set(inputs) >= {"raw_sha256", "dataset_sha256", "plan_sha256"}
    assert (
        core["schema_version"] == "axiom.result/1"
        and core["serializer"] == "axiom.canonical-json/1"
    )
    assert len(result.result_sha256) == 64
    assert core["operation"] == op


@pytest.mark.parametrize("op", sorted(PLANS))
def test_core_has_no_environment_paths_or_timing(op: str) -> None:
    text = run(make_dataset(CSV, SPEC), parse_plan_json(PLANS[op])).canonical_json.lower()
    for forbidden in ("/users/", "\\users\\", "/tmp", "timestamp", "seconds", "hostname", "pid"):
        assert forbidden not in text
    env = environment_info()
    assert {"python", "numpy", "scikit_learn", "pydantic", "machine"} <= set(env)
    assert "environment" not in run(make_dataset(CSV, SPEC), parse_plan_json(PLANS[op])).core


@pytest.mark.parametrize("op", sorted(PLANS))
def test_same_data_and_plan_give_identical_results_in_process(op: str) -> None:
    ds = make_dataset(CSV, SPEC)
    plan = parse_plan_json(PLANS[op])
    assert len({run(ds, plan).canonical_json for _ in range(10)}) == 1
    # equal input bytes loaded twice are two equal snapshots
    assert run(make_dataset(CSV, SPEC), plan).result_sha256 == run(ds, plan).result_sha256


def test_fresh_processes_reproduce_the_same_result_bytes() -> None:
    script = textwrap.dedent(
        """
        import sys
        from axiom import ColumnSpec, CsvProfile, load_csv_bytes, parse_plan_json, run
        spec = {"g": "text", "x": "numeric", "y": "numeric"}
        profile = CsvProfile(columns=tuple(ColumnSpec(name=n, kind=k) for n, k in spec.items()))
        ds = load_csv_bytes(sys.stdin.buffer.read(), profile)
        for text in sys.argv[1:]:
            print(run(ds, parse_plan_json(text)).result_sha256)
        """
    )
    plans = [PLANS[k] for k in sorted(PLANS)]
    here = [run(make_dataset(CSV, SPEC), parse_plan_json(p)).result_sha256 for p in plans]
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", script, *plans],
            input=CSV.encode(),
            capture_output=True,
            check=True,
            timeout=120,
        )
        assert proc.stdout.decode().split() == here


def test_plan_bound_to_one_snapshot_fails_on_a_changed_snapshot() -> None:
    old = make_dataset(CSV, SPEC)
    plan = parse_plan_json(PLANS["relate"])
    bound = bind_plan(old, plan)
    new = make_dataset(CSV.replace("c,6,11", "c,6,40"), SPEC)
    assert execute(old, bound).core["inputs"]["dataset_sha256"] == old.logical_sha256
    with pytest.raises(PlanError) as e:
        execute(new, bound)  # stale binding must not silently run on changed data
    assert e.value.code == "plan_not_bound_to_dataset"


def test_rebinding_after_mutation_yields_the_independently_expected_change() -> None:
    old = make_dataset(CSV, SPEC)
    new = make_dataset(CSV.replace("c,6,11", "c,6,40"), SPEC)
    plan = parse_plan_json(PLANS["inspect"])
    before = run(old, plan).core["result"]["columns"][1]["stats"]
    after = run(new, plan).core["result"]["columns"][1]["stats"]
    # independent expectation: only the last y changed from 11 to 40
    assert before["max"] == 11.0 and after["max"] == 40.0
    assert before["min"] == after["min"] == 2.0
    assert before["mean"] == pytest.approx(41 / 6) and after["mean"] == pytest.approx(70 / 6)
    assert (
        run(new, plan).core["inputs"]["dataset_sha256"]
        != run(old, plan).core["inputs"]["dataset_sha256"]
    )


def test_same_typed_data_with_different_bytes_changes_only_the_raw_hash() -> None:
    a = make_dataset(CSV, SPEC)
    b = make_dataset(CSV.replace("\n", "\r\n"), SPEC)
    ra = run(a, parse_plan_json(PLANS["inspect"])).core["inputs"]
    rb = run(b, parse_plan_json(PLANS["inspect"])).core["inputs"]
    assert ra["dataset_sha256"] == rb["dataset_sha256"] and ra["raw_sha256"] != rb["raw_sha256"]


def test_tampered_bound_plan_is_rejected() -> None:
    ds = make_dataset(CSV, SPEC)
    bound = bind_plan(ds, parse_plan_json(PLANS["relate"]))
    forged = BoundPlan(
        plan=parse_plan_json(PLANS["relate"].replace("pearson", "spearman")),
        plan_sha256=bound.plan_sha256,
        dataset_raw_sha256=bound.dataset_raw_sha256,
        dataset_logical_sha256=bound.dataset_logical_sha256,
        policy_version=bound.policy_version,
    )
    with pytest.raises(PlanError) as e:
        execute(ds, forged)
    assert e.value.code == "plan_hash_mismatch"
    with pytest.raises(AttributeError):
        bound.plan_sha256 = "x"  # type: ignore[misc]


def test_plans_are_data_never_code() -> None:
    hostile = (
        '{"schema_version":"axiom.plan/1","operation":"inspect","columns":["__import__(\'os\')"]}'
    )
    ds = make_dataset(CSV, SPEC)
    with pytest.raises(PlanError) as e:
        bind_plan(ds, parse_plan_json(hostile))
    assert e.value.code == "unknown_column"


def test_hash_sensitivity_fixtures() -> None:
    base = run(make_dataset(CSV, SPEC), parse_plan_json(PLANS["inspect"])).result_sha256
    variants = [
        CSV.replace("a,1,2", "a,1,3"),
        CSV.replace("g,x,y", "g,x,y").replace("c,5,10", "c,5,10.0000001"),
        CSV + "d,7,12\n",
        CSV.replace("a,1,2\na,2,4", "a,2,4\na,1,2"),
    ]
    hashes = {
        run(make_dataset(v, SPEC), parse_plan_json(PLANS["inspect"])).result_sha256
        for v in variants
    }
    assert base not in hashes and len(hashes) == len(variants)


def test_malformed_input_is_a_structured_error() -> None:
    from axiom import InputError

    with pytest.raises(InputError) as e:
        make_dataset("g,x,y\na,1\n", SPEC)
    assert json.loads(json.dumps(e.value.to_dict()))["code"] == "record_width"


def test_canonical_dumps_of_a_result_is_stable_across_key_insertion_order() -> None:
    core = run(make_dataset(CSV, SPEC), parse_plan_json(PLANS["compare"])).core
    reordered = dict(reversed(list(core.items())))
    assert canonical_dumps(core) == canonical_dumps(reordered)


def test_loading_from_bytes_matches_loading_from_file(tmp_path) -> None:
    from _helpers import make_profile
    from axiom import load_csv

    path = tmp_path / "d.csv"
    path.write_bytes(CSV.encode())
    a = load_csv(path, make_profile(SPEC))
    b = load_csv_bytes(CSV.encode(), make_profile(SPEC))
    assert (a.raw_sha256, a.logical_sha256) == (b.raw_sha256, b.logical_sha256)
