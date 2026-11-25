"""Regression, adversarial and boundary tests from the final review of PR #7 and the final audit.

The residual B1 finding: ``result_problems`` built ``list(range(k))`` and ``[0] * k`` from the
cluster count ``k`` stated in an untrusted passport, so a small re-sealed passport could make the
parent process allocate memory proportional to an arbitrary claimed number, through ``replay`` and
``compare_passports``. These tests reproduce it only at a safe scale and then guard the corrected
behaviour with an instrumented allocation bound (``tracemalloc``) and a no-dispatch guard.
The other tests cover the further audit findings (each reproduced on ``e8a9f57`` before the fix;
see the development log): lone surrogates and 64-bit overflow in untrusted documents, unbounded
error messages, an empty-cluster comparison crash, locale-dependent CLI output, malformed worker
responses, and the tolerance boundaries of ``axiom.agreement/1``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tracemalloc

import pytest

from _passport_helpers import DATA, PROFILE, ROOT, example_plan, reseal
from axiom import replay as replay_module
from axiom.agreement import (
    CORRELATION_ABS,
    SCALER_MEAN_SCALES,
    SCALER_SCALE_RTOL,
    compare_cores,
)
from axiom.limits import DEFAULT_LIMITS
from axiom.passport import (
    PassportError,
    dumps_passport,
    integrity_checks,
    loads_passport,
    validate_passport,
)
from axiom.passport_schema import INT_MAX, INT_MIN
from axiom.replay import VerifyError, analyze, compare_passports, replay, run_worker

# Peak traced allocation allowed for checking one example passport (a few tens of kilobytes of
# JSON). Before the fix a claimed k of 10**6 needed about 40 MB; any k-proportional allocation of
# even 10**5 clusters (about 4 MB) would exceed this bound.
ALLOCATION_BOUND = 1_000_000


@pytest.fixture(scope="module")
def passports() -> dict[str, dict]:
    names = ("inspect", "relate_pearson", "compare", "cluster")
    return {n: analyze(DATA, PROFILE, example_plan(n)) for n in names}


@pytest.fixture
def no_dispatch(monkeypatch):
    calls: list[dict] = []

    def forbidden(request, **kwargs):
        calls.append(request)
        raise AssertionError("a worker was dispatched")

    monkeypatch.setattr(replay_module, "run_worker", forbidden)
    return calls


def _peak(fn, *args, **kwargs):
    """(result or raised exception, peak traced bytes) of one call."""
    tracemalloc.start()
    try:
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - returned for the caller to assert on
            out = exc
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _claim_k(passport: dict, k: int, *, plan: bool, max_k: bool) -> dict:
    """Re-sealed cluster passport whose result (and optionally plan and max_k) claims ``k``."""
    p = copy.deepcopy(passport)
    p["result"]["result"]["k"] = k
    if plan:
        p["inputs"]["plan"]["k"] = k
        p["result"]["plan"]["k"] = k
    if max_k:
        p["inputs"]["limits"]["max_k"] = k
    return reseal(p, plan_too=plan)


def _statuses(checks: list[dict]) -> dict[str, str]:
    return {c["id"]: c["status"] for c in checks}


# --- residual B1: no work is sized by a claimed count ---------------------------------------------

CLAIMED_K = [10**6, 2**62, INT_MAX]


@pytest.mark.parametrize("k", CLAIMED_K)
@pytest.mark.parametrize(("plan", "max_k"), [(False, False), (True, False), (True, True)])
def test_claimed_cluster_count_never_sizes_integrity_work(passports, k, plan, max_k) -> None:
    forged = _claim_k(passports["cluster"], k, plan=plan, max_k=max_k)
    checks, peak = _peak(integrity_checks, forged)
    assert not isinstance(checks, Exception), checks
    assert peak < ALLOCATION_BOUND, peak
    statuses = _statuses(checks)
    assert statuses["result_consistency"] == "fail"
    problems = next(c for c in checks if c["id"] == "result_consistency")["problems"]
    assert "number of clusters != k" in problems
    if not max_k:
        assert "k exceeds the recorded max_k" in problems


@pytest.mark.parametrize("k", CLAIMED_K)
def test_replay_with_claimed_k_is_bounded_and_never_dispatches(passports, no_dispatch, k) -> None:
    # Result-only claim: integrity mismatch, bounded, nothing executed.
    report, peak = _peak(replay, _claim_k(passports["cluster"], k, plan=False, max_k=False), DATA)
    assert not isinstance(report, Exception), report
    assert report["verdict"] == "integrity_mismatch" and report["runs"] == []
    assert peak < ALLOCATION_BOUND, peak
    # Plan and recorded max_k raised consistently: the trusted policy rejects it first.
    forged = _claim_k(passports["cluster"], k, plan=True, max_k=True)
    raised, peak = _peak(replay, forged, DATA)
    assert isinstance(raised, VerifyError) and raised.code == "limits_exceed_policy"
    assert raised.details["fields"] == ["max_k"]
    assert peak < ALLOCATION_BOUND, peak
    assert no_dispatch == []


def test_replay_applies_the_trusted_policy_before_any_integrity_work(
    passports, no_dispatch, monkeypatch
) -> None:
    def forbidden(passport):
        raise AssertionError("integrity work before the policy check")

    monkeypatch.setattr(replay_module, "integrity_checks", forbidden)
    forged = _claim_k(passports["cluster"], 2**62, plan=True, max_k=True)
    with pytest.raises(VerifyError) as e:
        replay(forged, DATA)
    assert e.value.code == "limits_exceed_policy"
    assert no_dispatch == []


@pytest.mark.parametrize("k", CLAIMED_K)
@pytest.mark.parametrize("side", ["reference", "observed"])
def test_compare_with_claimed_k_is_bounded_on_either_side(passports, k, side) -> None:
    forged = _claim_k(passports["cluster"], k, plan=True, max_k=True)
    pair = (forged, passports["cluster"]) if side == "reference" else (passports["cluster"], forged)
    report, peak = _peak(compare_passports, *pair)
    assert not isinstance(report, Exception), report
    assert report["verdict"] == "integrity_mismatch" and report["mismatch"] == "passport"
    assert "comparison" not in report
    assert peak < ALLOCATION_BOUND, peak


def test_cli_compare_with_claimed_k_is_an_integrity_mismatch(tmp_path, passports) -> None:
    forged = tmp_path / "forged.json"
    forged.write_text(
        dumps_passport(_claim_k(passports["cluster"], 2**62, plan=True, max_k=True)), "utf-8"
    )
    ref = tmp_path / "ref.json"
    ref.write_text(dumps_passport(passports["cluster"]), "utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "axiom",
            "compare",
            "--reference",
            str(ref),
            "--observed",
            str(forged),
        ],
        capture_output=True,
        cwd=ROOT,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout.decode("utf-8"))["verdict"] == "integrity_mismatch"


def test_invalid_plan_or_limits_skip_the_dependent_result_checks(passports) -> None:
    bad_plan = copy.deepcopy(passports["cluster"])
    bad_plan["inputs"]["plan"]["k"] = 0  # rejected by the plan schema (k >= 2)
    statuses = _statuses(integrity_checks(reseal(bad_plan)))
    assert statuses["plan_digest"] == "fail"
    assert statuses["result_consistency"] == "not_run"
    bad_limits = copy.deepcopy(passports["cluster"])
    bad_limits["inputs"]["limits"]["max_k"] = 0
    checks = integrity_checks(reseal(bad_limits))
    assert _statuses(checks)["replay_inputs_valid"] == "fail"
    entry = next(c for c in checks if c["id"] == "result_consistency")
    assert entry["status"] == "not_run" and entry["problems"] == ["not checked: invalid limits"]


def test_cluster_ids_sizes_and_labels_are_checked_over_existing_entries(passports) -> None:
    base = passports["cluster"]

    def problems(edit) -> list[str]:
        p = copy.deepcopy(base)
        edit(p["result"]["result"])
        checks = integrity_checks(reseal(p))
        return next(c for c in checks if c["id"] == "result_consistency")["problems"]

    assert problems(lambda r: None) == []
    assert "cluster ids" in problems(lambda r: r["clusters"][0].update(id=1))
    assert "cluster sizes" in problems(lambda r: r["clusters"][0].update(size=1))
    assert "assignment labels" in problems(lambda r: r["assignments"].__setitem__(0, r["k"]))
    assert "assignment labels" in problems(lambda r: r["assignments"].__setitem__(0, INT_MAX))
    assert "number of clusters != k" in problems(lambda r: r["clusters"].pop())


def test_empty_cluster_and_failed_postcondition_are_inconsistent_records(passports) -> None:
    p = copy.deepcopy(passports["cluster"])
    res = p["result"]["result"]
    extra = copy.deepcopy(res["clusters"][-1])
    extra.update(id=res["k"], size=0)
    res["clusters"].append(extra)
    res["k"] += 1
    p["inputs"]["plan"]["k"] = p["result"]["plan"]["k"] = res["k"]
    forged = reseal(p, plan_too=True)
    checks = integrity_checks(forged)
    entry = next(c for c in checks if c["id"] == "result_consistency")
    assert entry["status"] == "fail" and entry["problems"] == ["empty cluster"]
    p = copy.deepcopy(passports["cluster"])
    p["result"]["result"]["postconditions"]["nearest_center"] = False
    entry = next(c for c in integrity_checks(reseal(p)) if c["id"] == "result_consistency")
    assert entry["problems"] == ["recorded postcondition false"]


def test_compare_cores_with_an_empty_cluster_does_not_raise(passports) -> None:
    """The comparison function itself tolerates a record the engine cannot produce."""
    ref = copy.deepcopy(passports["cluster"]["result"])
    res = ref["result"]
    extra = copy.deepcopy(res["clusters"][-1])
    extra.update(id=res["k"], size=0)
    res["clusters"].append(extra)
    obs = copy.deepcopy(ref)
    obs_res = obs["result"]
    obs_res["assignments"] = [1 - a if a in (0, 1) else a for a in obs_res["assignments"]]
    for c in obs_res["clusters"]:
        if c["id"] in (0, 1):
            c["id"] = 1 - c["id"]
    obs_res["clusters"].sort(key=lambda c: c["id"])
    report = compare_cores(ref, obs)  # raised KeyError before the fix
    assert report["partition"]["same_partition"] is True
    assert report["partition"]["labels_realigned"] is False
    assert report["verdict"] == "outside_criteria"


def test_shape_differences_report_sizes_not_whole_containers(passports) -> None:
    ref = passports["cluster"]["result"]
    obs = copy.deepcopy(ref)
    obs["result"]["assignments"].append(0)
    report = compare_cores(ref, obs)
    (diff,) = [d for d in report["non_float_differences"] if d["kind"] == "length"]
    n = len(ref["result"]["assignments"])
    assert diff["reference"] == {"type": "list", "length": n}
    assert diff["observed"] == {"type": "list", "length": n + 1}
    obs = copy.deepcopy(ref)
    del obs["result"]["diagnostics"]["lloyd_label_fixed_point"]
    (diff,) = [d for d in compare_cores(ref, obs)["non_float_differences"] if d["kind"] == "keys"]
    assert diff["keys_only_in_reference"] == ["lloyd_label_fixed_point"]
    assert diff["keys_only_in_observed"] == []


# --- arguments: timeout and data path -------------------------------------------------------------


@pytest.mark.parametrize("timeout", [10**400, -(10**400), 2**1000, 3600.0000000001, 0, -0.0])
def test_oversized_integer_timeouts_are_structured_errors(passports, no_dispatch, timeout) -> None:
    with pytest.raises(VerifyError) as e:
        replay(passports["inspect"], DATA, timeout=timeout)
    assert e.value.code == "invalid_timeout"
    with pytest.raises(VerifyError):
        run_worker({}, timeout=timeout)
    with pytest.raises(VerifyError):
        analyze(DATA, PROFILE, example_plan("inspect"), timeout=timeout)
    assert no_dispatch == []


def test_timeout_boundaries_are_inclusive_at_the_maximum(passports, monkeypatch) -> None:
    seen: list[float] = []

    def fake(request, *, timeout):
        seen.append(timeout)
        return {"status": "timeout", "parent_wall_seconds": 0.0}

    monkeypatch.setattr(replay_module, "run_worker", fake)
    for timeout in (3600, 3600.0, 5e-324):
        replay(passports["inspect"], DATA, timeout=timeout)
    assert seen == [3600.0, 3600.0, 5e-324]


@pytest.mark.parametrize("path", [None, 3, "data\x00.csv"])
def test_invalid_data_paths_are_rejected_before_dispatch(passports, no_dispatch, path) -> None:
    with pytest.raises(VerifyError) as e:
        replay(passports["inspect"], path)
    assert e.value.code == "invalid_data_path"
    with pytest.raises(VerifyError):
        analyze(path, PROFILE, example_plan("inspect"))
    assert no_dispatch == []


# --- untrusted text: encoding, integer range, message size ----------------------------------------


@pytest.mark.parametrize(
    "where", ['"workspace": "', '"result_cache": "', '"name": "', '"blas_thread_control": "']
)
def test_escaped_lone_surrogates_are_rejected_structurally(passports, no_dispatch, where) -> None:
    name = "inspect" if where != '"name": "' else "compare"
    text = dumps_passport(passports[name])
    assert where in text
    forged = text.replace(where, where + "\\udc80", 1)  # valid JSON, not encodable as UTF-8
    with pytest.raises(PassportError) as e:
        loads_passport(forged)
    assert e.value.code == "invalid_passport"
    assert "surrogate" in e.value.message
    with pytest.raises(PassportError):
        loads_passport("\ud800")  # the text itself
    assert no_dispatch == []


def test_lone_surrogate_keys_and_values_in_api_objects_are_rejected(passports) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["environment"]["packages"]["\ud800"] = "1.0"
    with pytest.raises(PassportError) as e:
        integrity_checks(p)  # raised UnicodeEncodeError before the fix
    assert e.value.code == "invalid_passport"
    p = copy.deepcopy(passports["inspect"])
    p["execution"]["workspace"] = "x\udfff"
    with pytest.raises(PassportError):
        compare_passports(p, passports["inspect"])


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        (INT_MAX, True),
        (INT_MIN, True),
        (INT_MAX + 1, False),
        (INT_MIN - 1, False),
        (10**5000, False),
    ],
    ids=["int64_max", "int64_min", "int64_max_plus_1", "int64_min_minus_1", "5001_digits"],
)
def test_integers_are_bounded_to_64_bits_everywhere(passports, value, ok) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["environment"]["numerical_backend"] = {"probe": value}  # a free-form JSON value
    if ok:
        validate_passport(p)
        assert _statuses(integrity_checks(p))["passport_digest"] == "fail"  # edited, not resealed
    else:
        with pytest.raises(PassportError) as e:
            integrity_checks(p)  # ValueError (int-to-text limit) before the fix for 10**5000
        assert e.value.details["field"] == "environment/numerical_backend/probe"
    p = copy.deepcopy(passports["inspect"])
    p["inputs"]["n_bytes"] = value
    if value >= 0 and ok:
        validate_passport(p)
    else:
        with pytest.raises(PassportError):
            validate_passport(p)


def test_booleans_are_not_mistaken_for_integers_by_the_range_check(passports) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["inputs"]["n_bytes"] = True
    with pytest.raises(PassportError) as e:
        validate_passport(p)
    assert "expected nat" in e.value.message


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p["identity"].__setitem__("dataset_format", "x" * 1_000_000),
        lambda p: p["identity"].__setitem__("dataset_format", 10**4000),
        lambda p: p["result"]["result"]["columns"][0].__setitem__("kind", "y" * 1_000_000),
        lambda p: p["environment"]["packages"].__setitem__("k" * 1_000_000, 1),
        lambda p: p.__setitem__("passport_version", "z" * 1_000_000),
    ],
)
def test_error_messages_do_not_echo_large_values(passports, edit) -> None:
    p = copy.deepcopy(passports["inspect"])
    edit(p)
    with pytest.raises(PassportError) as e:
        validate_passport(p)
    assert len(json.dumps(e.value.to_dict())) < 1_000


def test_integrity_problem_messages_are_bounded(passports) -> None:
    p = copy.deepcopy(passports["compare"])
    group = p["result"]["result"]["groups"][0]
    group["key"] = "g" * 1_000_000
    group["n_valid"] += 1
    checks = integrity_checks(reseal(p))
    entry = next(c for c in checks if c["id"] == "result_consistency")
    assert entry["status"] == "fail"
    assert len(json.dumps(entry)) < 2_000


@pytest.mark.parametrize("text", [None, 12, ["{}"]])
def test_passport_text_must_be_text_or_bytes(text) -> None:
    with pytest.raises(PassportError) as e:
        loads_passport(text)
    assert e.value.code == "invalid_passport"


# --- worker responses: failures are reported, never raised in the parent --------------------------


def _fake_proc(stdout: bytes, returncode: int = 0):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=b"")

    return fake_run


@pytest.mark.parametrize(
    ("response", "returncode", "status"),
    [
        ({"status": "ok"}, 0, "bad_output"),
        ({"status": "ok", "result_canonical_json": 5, "fingerprints": {}}, 0, "bad_output"),
        (
            {"status": "ok", "result_canonical_json": "{}", "environment": {}, "fingerprints": {}},
            0,
            "bad_output",
        ),
        ({"status": "bad_request", "error": "unsupported worker protocol"}, 2, "bad_request"),
        ({"status": "error", "error": None, "timing": []}, 0, "error"),
    ],
)
def test_malformed_worker_responses_become_failed_runs(
    passports, monkeypatch, response, returncode, status
) -> None:
    monkeypatch.setattr(
        replay_module.subprocess, "run", _fake_proc(json.dumps(response).encode(), returncode)
    )
    out = run_worker({"run_id": "x"}, timeout=10)
    assert out["status"] == status
    report = replay(passports["inspect"], DATA)
    assert report["verdict"] == "execution_failure"
    assert report["runs_completed"] == 0
    assert all(c["status"] != "pass" for c in report["checks"] if c["id"] == "exact_replay")


def test_unreadable_result_text_is_a_failed_run(passports, monkeypatch) -> None:
    real = replay_module.run_worker

    def corrupt(request, **kwargs):
        response = real(request, **kwargs)
        response["result_canonical_json"] = '{"truncated": '
        return response

    monkeypatch.setattr(replay_module, "run_worker", corrupt)
    report = replay(passports["inspect"], DATA)
    assert report["verdict"] == "execution_failure"
    assert report["runs"][0]["worker_status"] == "bad_output"
    assert report["runs_completed"] == 0


def test_ok_response_with_nonzero_exit_is_a_crash(monkeypatch) -> None:
    monkeypatch.setattr(replay_module.subprocess, "run", _fake_proc(b'{"status": "ok"}', 1))
    assert run_worker({"run_id": "x"}, timeout=10)["status"] == "crashed"


# --- CLI output is UTF-8 with LF on every platform ------------------------------------------------


def test_cli_output_is_utf8_regardless_of_the_locale_encoding(tmp_path) -> None:
    data = tmp_path / "d.csv"
    data.write_bytes("Größe,温度\n1.5,2\n3,4.25\n".encode())
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "columns": [
                    {"name": "Größe", "kind": "numeric"},
                    {"name": "温度", "kind": "numeric"},
                ],
            }
        ),
        "utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"schema_version": "axiom.plan/1", "operation": "inspect", "columns": ["Größe", "温度"]}
        ),
        "utf-8",
    )
    # cp1252 is the locale encoding of a redirected stdout on a Windows runner; 温度 is not in it
    env = dict(os.environ, PYTHONIOENCODING="cp1252")

    def cli(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "axiom", *args],
            capture_output=True,
            cwd=ROOT,
            env=env,
            timeout=300,
            check=False,
        )

    an = cli("analyze", "--data", str(data), "--profile", str(profile), "--plan", str(plan))
    assert an.returncode == 0, an.stdout[:300]  # exit 2 (UnicodeEncodeError) before the fix
    assert b"\r\n" not in an.stdout
    passport = tmp_path / "p.json"
    passport.write_bytes(an.stdout)
    assert loads_passport(an.stdout)["inputs"]["plan"]["columns"] == ["Größe", "温度"]
    out = tmp_path / "p_out.json"
    assert (
        cli(
            "analyze",
            "--data",
            str(data),
            "--profile",
            str(profile),
            "--plan",
            str(plan),
            "--out",
            str(out),
        ).returncode
        == 0
    )
    assert b"\r\n" not in out.read_bytes()
    rep = cli("replay", "--passport", str(passport), "--data", str(data))
    assert rep.returncode == 0
    assert json.loads(rep.stdout.decode("utf-8"))["verdict"] == "exact_match"


# --- tolerance boundaries of axiom.agreement/1 (inclusive, as documented) -------------------------


def _relate(passports) -> dict:
    return copy.deepcopy(passports["relate_pearson"]["result"])


@pytest.mark.parametrize(
    ("ref", "obs", "verdict"),
    [
        (0.0, CORRELATION_ABS, "within_criteria"),  # |difference| == bound exactly
        (0.0, math.nextafter(CORRELATION_ABS, 1.0), "outside_criteria"),
        (0.25, 0.25 + 2**-40, "within_criteria"),  # 9.1e-13, exact in binary64
        (0.25, 0.25 + 2**-38, "outside_criteria"),  # 3.6e-12
    ],
)
def test_correlation_bound_is_inclusive(passports, ref, obs, verdict) -> None:
    a, b = _relate(passports), _relate(passports)
    a["result"]["coefficient"], b["result"]["coefficient"] = ref, obs
    assert compare_cores(a, b)["verdict"] == verdict


def test_signed_zero_is_identical_and_subnormals_across_zero_are_two_ulp(passports) -> None:
    core = copy.deepcopy(passports["inspect"]["result"])
    stats = next(c for c in core["result"]["columns"] if c["kind"] == "numeric")["stats"]
    stats["mean"] = 0.0
    obs = copy.deepcopy(core)
    next(c for c in obs["result"]["columns"] if c["kind"] == "numeric")["stats"]["mean"] = -0.0
    assert compare_cores(core, obs)["byte_identical"] is True
    stats["mean"] = 5e-324
    obs = copy.deepcopy(core)
    next(c for c in obs["result"]["columns"] if c["kind"] == "numeric")["stats"]["mean"] = -5e-324
    report = compare_cores(core, obs)
    assert report["max_ulp_distance"] == 2 and report["verdict"] == "within_criteria"


def _cluster_pair(passports):
    ref = copy.deepcopy(passports["cluster"]["result"])
    return ref, copy.deepcopy(ref)


def test_scaler_scale_bound_is_relative_to_the_reference(passports) -> None:
    ref, obs = _cluster_pair(passports)
    scale = ref["result"]["scaler"]["scale"]
    scale[0] = 1.0
    obs["result"]["scaler"]["scale"][0] = 1.0 + 2**-30  # 9.3e-10 <= 2e-9
    assert compare_cores(ref, obs)["verdict"] == "within_criteria"
    obs["result"]["scaler"]["scale"][0] = 1.0 + 2**-28  # 3.7e-9 > 2e-9
    assert compare_cores(ref, obs)["verdict"] == "outside_criteria"
    assert SCALER_SCALE_RTOL == 2e-9


def test_scaler_mean_bound_uses_the_scale_of_the_same_feature(passports) -> None:
    ref, obs = _cluster_pair(passports)
    scaler = ref["result"]["scaler"]
    assert len(scaler["scale"]) >= 2
    scaler["scale"][0], scaler["scale"][1] = 1.0, 1024.0
    obs["result"]["scaler"]["scale"][:2] = [1.0, 1024.0]
    bound_small, bound_large = SCALER_MEAN_SCALES * 1.0, SCALER_MEAN_SCALES * 1024.0
    scaler["mean"][1] = 0.0
    obs["result"]["scaler"]["mean"][1] = bound_large  # exactly at the bound of feature 1
    assert compare_cores(ref, obs)["verdict"] == "within_criteria"
    scaler["mean"][0] = 0.0
    obs["result"]["scaler"]["mean"][0] = bound_small * 2  # above feature 0's bound
    assert compare_cores(ref, obs)["verdict"] == "outside_criteria"


def test_inertia_bound_scales_with_eligible_rows_and_tolerance(passports) -> None:
    ref, obs = _cluster_pair(passports)
    res = ref["result"]
    tol, n = ref["plan"]["tol"], res["n_eligible"]
    res["inertia_standardized"] = 0.0
    obs["result"]["inertia_standardized"] = n * (tol + 1e-12) * 0.999
    assert compare_cores(ref, obs)["verdict"] == "within_criteria"
    obs["result"]["inertia_standardized"] = n * (tol + 1e-12) * 1.001
    assert compare_cores(ref, obs)["verdict"] == "outside_criteria"


def test_default_policy_rejects_limits_above_every_default(passports, no_dispatch) -> None:
    """Each recorded limit, raised by one above the trusted default, is rejected by name."""
    for name in DEFAULT_LIMITS.__dataclass_fields__:
        p = copy.deepcopy(passports["inspect"])
        p["inputs"]["limits"][name] = getattr(DEFAULT_LIMITS, name) + 1
        with pytest.raises(VerifyError) as e:
            replay(reseal(p), DATA)
        assert e.value.details["fields"] == [name]
    assert no_dispatch == []


def test_timed_out_worker_is_terminated_and_its_workspace_removed(passports, monkeypatch) -> None:
    """A real worker killed by the timeout has exited and left no workspace behind."""
    started: list[subprocess.Popen] = []
    workspaces: list[str] = []
    real_popen, real_tmp = subprocess.Popen, replay_module.tempfile.TemporaryDirectory

    class RecordingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

    class RecordingTmp(real_tmp):
        def __enter__(self):
            workspaces.append(super().__enter__())
            return workspaces[-1]

    monkeypatch.setattr(replay_module.subprocess, "Popen", RecordingPopen)
    monkeypatch.setattr(replay_module.tempfile, "TemporaryDirectory", RecordingTmp)
    report = replay(passports["inspect"], DATA, timeout=0.05)
    assert report["verdict"] == "execution_failure"
    assert report["runs"][0]["worker_status"] == "timeout"
    assert len(started) == 1 and started[0].poll() is not None  # exited, not left running
    assert len(workspaces) == 1 and not os.path.exists(workspaces[0])
