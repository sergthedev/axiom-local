"""Regression tests for the PR #7 technical review (blockers B1, B2, B3).

Each test reproduces a counterexample from the review against the code before the fix and checks
the corrected behaviour at the API and, where relevant, the CLI boundary. Re-sealing (recomputing
every digest after an edit) models an adversary: digests are not signatures, so a re-sealed
passport must be rejected by structure, consistency or policy checks, or exposed by recomputation.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys

import pytest

from _passport_helpers import DATA, PROFILE, ROOT, example_plan, reseal
from axiom import replay as replay_module
from axiom.environment import unavailable
from axiom.limits import DEFAULT_LIMITS, Limits
from axiom.passport import (
    PassportError,
    dumps_passport,
    integrity_checks,
    loads_passport,
    validate_passport,
)
from axiom.replay import (
    MAX_REPLAY_RUNS,
    MAX_TIMEOUT_SECONDS,
    VerifyError,
    analyze,
    compare_passports,
    replay,
    run_worker,
)

EXACT_MODES = ("same_environment_exact", "exact_environment_unverified")
MAX_FLOAT = 1.7976931348623157e308


@pytest.fixture(scope="module")
def passports() -> dict[str, dict]:
    return {
        n: analyze(DATA, PROFILE, example_plan(n)) for n in ("inspect", "relate_pearson", "cluster")
    }


@pytest.fixture
def no_dispatch(monkeypatch):
    """Fail the test if any worker process would be started."""
    calls: list[dict] = []

    def forbidden(request, **kwargs):
        calls.append(request)
        raise AssertionError("a worker was dispatched")

    monkeypatch.setattr(replay_module, "run_worker", forbidden)
    return calls


def _checks(items: list[dict]) -> dict[str, str]:
    return {c["id"]: c["status"] for c in items}


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "axiom", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
        check=False,
    )


# --- B1: untrusted passports cannot raise resource budgets; arguments are bounded ----------------


def test_resealed_oversized_limits_are_rejected_before_any_worker(passports, no_dispatch) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["inputs"]["limits"]["max_bytes"] = DEFAULT_LIMITS.max_bytes * 1000
    p["inputs"]["limits"]["max_cluster_work"] = 2**62  # was 10**30: now outside int64 (audit)
    p = reseal(p)
    assert all(c["status"] == "pass" for c in integrity_checks(p))  # a consistent forgery
    with pytest.raises(VerifyError) as e:
        replay(p, DATA)
    assert e.value.code == "limits_exceed_policy"
    assert set(e.value.details["fields"]) == {"max_bytes", "max_cluster_work"}
    p["inputs"]["limits"]["max_cluster_work"] = 10**30
    with pytest.raises(PassportError) as e2:
        replay(reseal(p), DATA)
    assert e2.value.code == "invalid_passport"
    assert e2.value.details["field"] == "inputs/limits/max_cluster_work"
    assert no_dispatch == []


def test_trusted_policy_override_and_stricter_recorded_limits_are_used_unchanged(
    passports, monkeypatch
) -> None:
    seen: list[dict] = []

    def fake(request, *, timeout):
        seen.append(request)
        return {"status": "timeout", "parent_wall_seconds": 0.0}

    monkeypatch.setattr(replay_module, "run_worker", fake)
    big = copy.deepcopy(passports["inspect"])
    big["inputs"]["limits"]["max_bytes"] = DEFAULT_LIMITS.max_bytes * 2
    big = reseal(big)
    policy = Limits(max_bytes=DEFAULT_LIMITS.max_bytes * 2)  # an explicit, trusted override
    assert replay(big, DATA, policy=policy)["verdict"] == "execution_failure"
    small = copy.deepcopy(passports["inspect"])
    small["inputs"]["limits"]["max_bytes"] = 1_000_000
    replay(reseal(small), DATA)
    assert [r["limits"]["max_bytes"] for r in seen] == [DEFAULT_LIMITS.max_bytes * 2, 1_000_000]


@pytest.mark.parametrize("runs", [0, -1, True, 1.5, "2", MAX_REPLAY_RUNS + 1])
def test_run_count_is_bounded(passports, no_dispatch, runs) -> None:
    with pytest.raises(VerifyError) as e:
        replay(passports["inspect"], DATA, runs=runs)
    assert e.value.code == "invalid_runs"


@pytest.mark.parametrize(
    "timeout", [math.nan, math.inf, -math.inf, -1.0, 0, True, "10", MAX_TIMEOUT_SECONDS + 1]
)
def test_timeout_must_be_finite_positive_and_bounded(passports, no_dispatch, monkeypatch, timeout):
    with pytest.raises(VerifyError) as e:
        replay(passports["inspect"], DATA, timeout=timeout)
    assert e.value.code == "invalid_timeout"

    def no_process(*args, **kwargs):
        raise AssertionError("a process was started")

    monkeypatch.setattr(replay_module.subprocess, "run", no_process)
    for call in (
        lambda: run_worker({}, timeout=timeout),
        lambda: analyze(DATA, PROFILE, example_plan("inspect"), timeout=timeout),
    ):
        with pytest.raises(VerifyError):
            call()


def test_analyze_requires_a_limits_instance(monkeypatch) -> None:
    monkeypatch.setattr(replay_module, "run_worker", lambda *a, **k: pytest.fail("dispatched"))
    with pytest.raises(VerifyError) as e:
        analyze(DATA, PROFILE, example_plan("inspect"), limits={"max_bytes": 10})
    assert e.value.code == "invalid_limits"
    with pytest.raises(VerifyError):
        analyze(DATA, PROFILE, example_plan("inspect"), limits=Limits(max_rows=0))


def test_passport_text_is_bounded_before_parsing(passports) -> None:
    text = dumps_passport(passports["inspect"])
    with pytest.raises(PassportError) as e:
        loads_passport(text, max_bytes=len(text.encode("utf-8")) - 1)
    assert e.value.code == "passport_too_large"
    assert loads_passport(text, max_bytes=len(text.encode("utf-8")))["passport_version"]
    with pytest.raises(PassportError) as e:
        loads_passport("[" * 200_000 + "]" * 200_000)
    assert e.value.code == "invalid_passport"
    deep = copy.deepcopy(passports["inspect"])
    nested: list = []
    for _ in range(40):
        nested = [nested]
    deep["environment"]["numerical_backend"]["extra"] = nested
    with pytest.raises(PassportError, match="nesting deeper"):
        validate_passport(reseal(deep))


def test_cli_rejects_an_oversized_passport_file_unparsed(tmp_path) -> None:
    from axiom.passport import MAX_PASSPORT_BYTES

    big = tmp_path / "big.json"
    with open(big, "wb") as handle:
        handle.truncate(MAX_PASSPORT_BYTES + 1)  # sparse file of zero bytes
    proc = _cli("replay", "--passport", str(big), "--data", str(DATA))
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["error"]["code"] == "passport_too_large"


def test_cli_rejects_invalid_runs_and_timeout(tmp_path, passports) -> None:
    path = tmp_path / "p.json"
    path.write_text(dumps_passport(passports["inspect"]), encoding="utf-8")
    for extra, code in (
        (("--runs", "0"), "invalid_runs"),
        (("--timeout", "nan"), "invalid_timeout"),
    ):
        proc = _cli("replay", "--passport", str(path), "--data", str(DATA), *extra)
        assert proc.returncode == 2
        assert json.loads(proc.stdout)["error"]["code"] == code


# --- B2: strict structure, consistency, structured errors, finite-safe reports -------------------


def _invalid(passport: dict) -> PassportError:
    """The passport must be rejected structurally by every entry point, with the same error."""
    resealed = reseal(passport) if _sealable(passport) else passport
    with pytest.raises(PassportError) as e:
        validate_passport(resealed)
    for call in (
        lambda: integrity_checks(resealed),
        lambda: compare_passports(resealed, resealed),
    ):
        with pytest.raises(PassportError):
            call()
    return e.value


def _sealable(passport: dict) -> bool:
    try:
        reseal(passport)
        return True
    except Exception:  # noqa: BLE001 - some edits make the passport impossible to re-seal
        return False


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (
            lambda p: p.__setitem__(
                "verification",
                [
                    {
                        "id": "x",
                        "kind": "magic",
                        "status": "pass",
                        "evidence": "trust_me",
                        "detail": "",
                    }
                ],
            ),
            "verification/0/kind",
        ),
        (lambda p: p["verification"][0].__setitem__("extra", True), "verification/0"),
        (
            lambda p: p["identity"].__setitem__("dataset_format", "axiom.dataset/999"),
            "identity/dataset_format",
        ),
        (lambda p: p["identity"].__setitem__("raw_sha256", "not-a-digest"), "identity/raw_sha256"),
        (lambda p: p["inputs"].__setitem__("n_rows", -5), "inputs/n_rows"),
        (lambda p: p["inputs"].__setitem__("n_rows", True), "inputs/n_rows"),
        (lambda p: p["result"].__setitem__("result", {"nonsense": 1}), "result/result"),
        (lambda p: p["result"]["result"]["columns"][0].pop("stats"), "result/result/columns/0"),
        (
            lambda p: p["result"]["result"]["columns"][0]["stats"].__setitem__("mean", "1.0"),
            "result/result/columns/0/stats/mean",
        ),
        (lambda p: p["result"].__setitem__("operation", "regress"), "result/operation"),
        (lambda p: p["execution"].__setitem__("mode", "cached"), "execution/mode"),
        (
            lambda p: p["environment"]["threads"].__setitem__("fit_limit_probe", "yes"),
            "environment/threads/fit_limit_probe",
        ),
        (
            lambda p: p["environment"].__setitem__("schema_version", "axiom.environment/9"),
            "environment/schema_version",
        ),
    ],
)
def test_structurally_invalid_passports_are_rejected_with_the_field(passports, edit, field) -> None:
    p = copy.deepcopy(passports["inspect"])
    edit(p)
    error = _invalid(p)
    assert error.code == "invalid_passport"
    assert error.details["field"] == field


def test_identical_invalid_results_are_not_an_exact_match(passports) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["result"]["result"] = {"nonsense": 1}
    p = reseal(p)
    with pytest.raises(PassportError):
        compare_passports(p, copy.deepcopy(p))


def test_malformed_direct_api_objects_raise_structured_errors(passports, no_dispatch) -> None:
    edits = [
        lambda p: p["identity"].pop("raw_sha256"),
        lambda p: p.__setitem__("result", []),
        lambda p: p.pop("environment"),
        lambda p: p["inputs"].__setitem__("limits", None),
    ]
    for edit in edits:
        p = copy.deepcopy(passports["inspect"])
        edit(p)
        for fn, args in ((replay, (p, DATA)), (integrity_checks, (p,))):
            with pytest.raises(PassportError) as e:
                fn(*args)
            assert e.value.code == "invalid_passport"
    for bad in (None, [], "passport", {"passport_version": "axiom.passport/2"}):
        with pytest.raises(PassportError):
            replay(bad, DATA)
    assert no_dispatch == []


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "NaN", "Infinity"])
def test_numbers_outside_binary64_are_rejected(passports, literal) -> None:
    p = passports["inspect"]
    text = dumps_passport(p)
    value = repr(p["result"]["result"]["columns"][0]["stats"]["mean"])
    assert text.count(value) >= 1
    with pytest.raises(PassportError) as e:
        loads_passport(text.replace(value, literal, 1))
    assert e.value.code == "invalid_passport"


def test_api_rejects_non_finite_values_anywhere(passports) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["execution"]["timing_seconds"]["parent_wall"] = math.inf
    with pytest.raises(PassportError, match="non-finite"):
        validate_passport(p)


@pytest.mark.parametrize(
    ("name", "edit", "check_id"),
    [
        (
            "inspect",
            lambda p: p["inputs"].__setitem__("n_rows", p["inputs"]["n_rows"] + 1),
            "identity_consistency",
        ),
        (
            "inspect",
            lambda p: p["result"]["result"]["columns"][0].__setitem__(
                "n_valid", p["result"]["result"]["columns"][0]["n_valid"] + 1
            ),
            "result_consistency",
        ),
        (
            "relate_pearson",
            lambda p: p["result"]["result"].__setitem__("coefficient", 1.5),
            "result_consistency",
        ),
        (
            "relate_pearson",
            lambda p: p["result"]["result"].__setitem__("x", "household_income"),
            "result_consistency",
        ),
        (
            "cluster",
            lambda p: p["result"]["result"]["clusters"][0].__setitem__(
                "size", p["result"]["result"]["clusters"][0]["size"] + 1
            ),
            "result_consistency",
        ),
        ("cluster", lambda p: p["result"]["result"]["assignments"].append(0), "result_consistency"),
        (
            "inspect",
            lambda p: p["verification"][-1].__setitem__("status", "pass"),
            "verification_record",
        ),
    ],
)
def test_internally_inconsistent_records_fail_integrity_and_do_not_execute(
    passports, no_dispatch, name, edit, check_id
) -> None:
    p = copy.deepcopy(passports[name])
    edit(p)
    p = reseal(p)
    statuses = _checks(integrity_checks(p))
    assert statuses[check_id] == "fail"
    assert statuses["passport_digest"] == statuses["result_digest"] == "pass"
    report = replay(p, DATA)
    assert report["verdict"] == "integrity_mismatch" and report["mismatch"] == "passport"
    assert no_dispatch == []


def test_operation_and_plan_must_agree(passports, no_dispatch) -> None:
    p = copy.deepcopy(passports["inspect"])
    relate_plan = copy.deepcopy(passports["relate_pearson"]["inputs"]["plan"])
    p["inputs"]["plan"] = relate_plan
    p["result"]["plan"] = copy.deepcopy(relate_plan)
    p = reseal(p, plan_too=True)
    statuses = _checks(integrity_checks(p))
    assert statuses["plan_digest"] == "fail"
    assert replay(p, DATA)["verdict"] == "integrity_mismatch"


def test_recorded_verification_is_historical_and_never_reported_as_performed(passports) -> None:
    report = replay(passports["inspect"], DATA)
    assert report["reference"]["recorded_verification"].startswith("historical")
    assert _checks(report["checks"])["independent_reference"] == "not_run"


def _overflow_pair(passport: dict) -> tuple[dict, dict]:
    a, b = copy.deepcopy(passport), copy.deepcopy(passport)
    a["result"]["result"]["columns"][0]["stats"]["max"] = MAX_FLOAT
    b["result"]["result"]["columns"][0]["stats"]["max"] = -MAX_FLOAT
    b["environment"]["os"]["system"] = "recorded-on-another-os"
    return reseal(a), reseal(b)


def test_differences_beyond_the_float_range_are_reported_without_infinity(passports) -> None:
    a, b = _overflow_pair(passports["inspect"])
    report = compare_passports(a, b)
    assert report["verdict"] == "numerical_discrepancy"
    (diff,) = report["comparison"]["field_differences"]
    assert diff["abs_diff"] is None and diff["within"] is False
    assert report["comparison"]["max_abs_diff"] is None
    json.dumps(report, allow_nan=False)  # serializable for the CLI


def test_cli_compare_reports_overflowing_difference_and_malformed_input(tmp_path, passports):
    a, b = _overflow_pair(passports["inspect"])
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(dumps_passport(a), encoding="utf-8")
    pb.write_text(dumps_passport(b), encoding="utf-8")
    proc = _cli("compare", "--reference", str(pa), "--observed", str(pb))
    assert proc.returncode == 1, proc.stderr
    assert json.loads(proc.stdout)["verdict"] == "numerical_discrepancy"
    value = repr(passports["inspect"]["result"]["result"]["columns"][0]["stats"]["mean"])
    pb.write_text(dumps_passport(passports["inspect"]).replace(value, "1e999", 1), "utf-8")
    proc = _cli("compare", "--reference", str(pa), "--observed", str(pb))
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["error"]["code"] == "invalid_passport"


# --- B3: unknown environment never weakens exact replay; per-run evidence; partial runs ----------


def _forge_mean(passport: dict, steps: int = 1) -> dict:
    p = copy.deepcopy(passport)
    stats = p["result"]["result"]["columns"][0]["stats"]
    for _ in range(steps):
        stats["mean"] = math.nextafter(stats["mean"], math.inf)
    return p


def test_unknown_environment_metadata_still_requires_byte_identity(passports) -> None:
    p = _forge_mean(passports["inspect"])
    p["environment"]["packages"]["numpy"] = unavailable("not recorded")  # unknown, not different
    p = reseal(p)
    report = replay(p, DATA)
    assert report["environment"]["status"] == "unverified"
    assert report["comparison_mode"] == "exact_environment_unverified"
    assert report["runs"][0]["comparison"]["verdict"] == "within_criteria"  # yet not accepted
    assert report["verdict"] == "numerical_discrepancy"
    checks = _checks(report["checks"])
    assert checks["environment_match"] == "not_run"  # unknown is neither pass nor fail
    assert checks["exact_replay"] == "fail" and checks["numerical_agreement"] == "not_applicable"
    clean = copy.deepcopy(passports["inspect"])
    clean["environment"]["packages"]["numpy"] = unavailable("not recorded")
    assert replay(reseal(clean), DATA)["verdict"] == "exact_match"  # byte identity still observed
    elsewhere = copy.deepcopy(p)
    elsewhere["environment"]["os"]["system"] = "recorded-on-another-os"
    report = compare_passports(reseal(elsewhere), p)
    assert report["environment"]["status"] == "different"


def test_compare_with_unverified_environment_requires_byte_identity(passports) -> None:
    a = copy.deepcopy(passports["inspect"])
    a["environment"]["code"]["commit"] = unavailable("installed package")
    a = reseal(a)
    b = reseal(_forge_mean(a))
    report = compare_passports(a, b)
    assert (
        report["comparison_mode"] in EXACT_MODES and report["environment"]["status"] != "different"
    )
    assert report["verdict"] == "numerical_discrepancy"


def test_each_run_records_its_own_worker_observations(passports) -> None:
    report = replay(passports["cluster"], DATA, runs=2)
    ids = [r["run_id"] for r in report["runs"]]
    assert len(set(ids)) == 2 and all(len(i) == 32 for i in ids)
    for run in report["runs"]:
        threads = run["threads"]
        assert threads["observed"] is True
        assert all(v == "1" for v in threads["environment_variables"].values())
        assert threads["pools"] and all(p["num_threads"] == 1 for p in threads["pools"])
        assert run["environment_status"] in ("verified_same", "unverified")
    statuses = [r["threads"]["native_thread_status"]["status"] for r in report["runs"]]
    expected = "pass" if all(s == "verified_single_thread" for s in statuses) else "not_run"
    assert _checks(report["checks"])["native_thread_policy"] == expected
    if sys.platform == "darwin":  # NumPy wheels for macOS arm64 use Accelerate
        threads = report["runs"][0]["threads"]
        if "accelerate" in str(threads["blas_thread_control"]):
            assert threads["native_thread_status"]["status"] == "unverified"


def test_partial_runs_never_look_fully_passed(passports, monkeypatch) -> None:
    real = replay_module.run_worker
    calls = {"n": 0}

    def flaky(request, *, timeout):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"status": "timeout", "parent_wall_seconds": 0.0}
        return real(request, timeout=timeout)

    monkeypatch.setattr(replay_module, "run_worker", flaky)
    report = replay(passports["cluster"], DATA, runs=3)
    assert report["verdict"] == "execution_failure"
    assert report["runs_completed"] == 2 and report["runs_requested"] == 3
    checks = {c["id"]: c for c in report["checks"]}
    partition = checks["partition_identity"]
    assert partition["status"] == "fail"
    assert partition["runs_compared"] == 2 and partition["equal_in_compared_runs"] is True
    assert checks["exact_replay"]["status"] == "fail"
    assert checks["plan_identity"]["status"] == "not_run"
    assert checks["dataset_identity"]["status"] == "not_run"
