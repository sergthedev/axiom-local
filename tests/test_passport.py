"""Environment manifest and Analysis Passport v1 (ADR 0007): content, privacy, integrity."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from _passport_helpers import DATA, PROFILE, ROOT, example_plan, reseal
from axiom import load_csv, run
from axiom.environment import (
    capture_environment,
    compare_environments,
    environment_differences,
    is_unavailable,
    native_thread_status,
    unavailable,
)
from axiom.limits import DEFAULT_LIMITS
from axiom.passport import (
    CHECK_STATUSES,
    PASSPORT_VERSION,
    PassportError,
    build_passport,
    dumps_passport,
    integrity_checks,
    limits_from_dict,
    loads_passport,
    result_from_passport,
)


@pytest.fixture(scope="module")
def env() -> dict:
    return capture_environment()


def _passport(name: str, env: dict) -> dict:
    ds = load_csv(DATA, PROFILE)
    result = run(ds, example_plan(name))
    return build_passport(
        result=result,
        profile=PROFILE,
        limits=DEFAULT_LIMITS,
        n_bytes=DATA.stat().st_size,
        environment=env,
        execution=EXECUTION,
    )


# Shape of the execution record written by ``axiom.replay.analyze`` (values are illustrative).
EXECUTION = {
    "run_id": "0" * 32,
    "mode": "fresh_process",
    "workspace": "fresh temporary directory, deleted after the run",
    "result_cache": "none: the worker recomputes from the data file",
    "thread_environment": {"OMP_NUM_THREADS": "1"},
    "started_utc": None,
    "finished_utc": None,
    "timing_seconds": {"parent_wall": 1.0},
}


@pytest.fixture(scope="module")
def passports(env) -> dict[str, dict]:
    return {n: _passport(n, env) for n in ("inspect", "relate_pearson", "compare", "cluster")}


# --- environment ---------------------------------------------------------------------------------


def test_environment_records_the_required_information(env) -> None:
    assert env["schema_version"] == "axiom.environment/1"
    assert env["python"]["implementation"] == "CPython" and env["python"]["version"].startswith(
        "3.12."
    )
    assert env["os"]["system"] and env["os"]["machine"]
    for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "pydantic", "axiom-local"):
        assert isinstance(env["packages"][name], str)
    assert len(env["code"]["package_files_sha256"]) == 64
    backend = env["numerical_backend"]
    assert "numpy_blas" in backend and "simd" in backend
    threads = env["threads"]
    assert set(threads["environment_variables"]) >= {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    assert threads["fit_limit_probe"] in ("verified_single_thread", "unverified")
    for key in ("commit", "dirty", "lock_sha256"):  # either a value or an explicit marker
        value = env["code"][key]
        assert is_unavailable(value) or isinstance(value, (str, bool))


def test_environment_contains_no_local_paths_or_host_name(env) -> None:
    import platform

    text = json.dumps(env)
    for path in (str(ROOT), str(Path.home())):
        assert path not in text and json.dumps(path)[1:-1] not in text
    assert "filepath" not in text
    node = platform.node()
    if node:
        assert node not in text


def _fully_known(env: dict) -> dict:
    """A copy of ``env`` in which every compared field is observed (no unknown versions)."""
    e = copy.deepcopy(env)
    e["numerical_backend"] = {
        "numpy_blas": {"name": "openblas", "version": "1.0"},
        "numpy_lapack": {"name": "openblas", "version": "1.0"},
        "simd": {"baseline": ["X"], "found": [], "not_found": []},
    }
    e["threads"]["detected_pools"] = [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "libopenblas",
            "version": "1.0",
            "num_threads": 1,
        }
    ]
    e["threads"]["fit_limit_probe"] = "verified_single_thread"
    e["threads"]["blas_thread_control"] = "introspectable"
    for key in ("commit", "dirty", "lock_sha256"):
        e["code"][key] = {"commit": "a" * 40, "dirty": False, "lock_sha256": "b" * 64}[key]
    return e


def test_environment_classification_never_treats_unknown_as_equal(env) -> None:
    known = _fully_known(env)
    same = compare_environments(known, copy.deepcopy(known))
    assert same["status"] == "verified_same" and environment_differences(known, known) == []
    # review B3: equal unavailable markers, nested anywhere, are not equal
    a, b = copy.deepcopy(known), copy.deepcopy(known)
    a["packages"]["scipy"] = b["packages"]["scipy"] = unavailable("test")
    result = compare_environments(a, b)
    assert result["status"] == "unverified"
    assert [u["field"] for u in result["unverified_fields"]] == ["packages"]
    # two "unverified" thread probes, or an invisible BLAS on both sides, are not equal either
    for field, value in (
        ("fit_limit_probe", "unverified"),
        ("blas_thread_control", "unverified (NumPy BLAS 'accelerate' is not visible)"),
    ):
        a, b = copy.deepcopy(known), copy.deepcopy(known)
        a["threads"][field] = b["threads"][field] = value
        assert compare_environments(a, b)["status"] == "unverified", field
    # versions reported as unknown (Accelerate) or None (libomp) are unknown
    a = copy.deepcopy(known)
    a["numerical_backend"]["numpy_blas"]["version"] = "unknown"
    a["threads"]["detected_pools"][0]["version"] = None
    assert compare_environments(a, copy.deepcopy(a))["status"] == "unverified"
    # effective pools and thread variables are compared
    b = copy.deepcopy(known)
    b["threads"]["detected_pools"][0]["num_threads"] = 8
    assert [d["field"] for d in compare_environments(known, b)["differences"]] == [
        "threads.detected_pools"
    ]
    b = copy.deepcopy(known)
    b["threads"]["environment_variables"]["OMP_NUM_THREADS"] = "4"
    assert compare_environments(known, b)["status"] == "different"
    # an observed difference wins over unknowns elsewhere
    b = copy.deepcopy(a)
    b["packages"]["numpy"] = "0.0.0"
    result = compare_environments(a, b)
    assert result["status"] == "different"
    assert [d["field"] for d in result["differences"]] == ["packages"]
    diffs = environment_differences(a, b)
    assert {d["field"]: d["comparable"] for d in diffs}["packages"] is True
    # a missing manifest is unknown, not equal
    assert compare_environments(known, None)["status"] == "unverified"


def test_native_thread_status_reports_invisible_blas_as_unverified(env) -> None:
    known = _fully_known(env)
    assert native_thread_status(known)["status"] == "verified_single_thread"
    accelerate = copy.deepcopy(known)
    accelerate["threads"]["blas_thread_control"] = (
        "unverified (NumPy BLAS 'accelerate' is not visible to threadpoolctl)"
    )
    status = native_thread_status(accelerate)
    assert status["status"] == "unverified" and "accelerate" in status["reason"]
    busy = copy.deepcopy(known)
    busy["threads"]["detected_pools"][0]["num_threads"] = 4
    assert native_thread_status(busy)["status"] == "multi_threaded"
    assert native_thread_status(None)["status"] == "unverified"


# --- passport content ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["inspect", "relate_pearson", "compare", "cluster"])
def test_passport_carries_four_identities_and_full_precision_result(passports, name) -> None:
    p = passports[name]
    assert p["passport_version"] == PASSPORT_VERSION
    ident = p["identity"]
    for key in ("raw_sha256", "dataset_sha256", "plan_sha256", "result_sha256"):
        assert len(ident[key]) == 64
    fresh = run(load_csv(DATA, PROFILE), example_plan(name))
    assert ident["result_sha256"] == fresh.result_sha256
    # the embedded result is the exact canonical core, not a rounded copy
    assert result_from_passport(p).canonical_json == fresh.canonical_json
    reread = loads_passport(dumps_passport(p))
    assert result_from_passport(reread).canonical_json == fresh.canonical_json
    assert all(c["status"] == "pass" for c in integrity_checks(reread))


def test_passport_never_marks_unperformed_verification_as_passed(passports) -> None:
    for p in passports.values():
        statuses = {c["id"]: c["status"] for c in p["verification"]}
        assert set(statuses.values()) <= set(CHECK_STATUSES)
        assert statuses["fresh_process_replay"] == "not_run"
        assert statuses["cross_environment_agreement"] == "not_run"
        assert statuses["independent_reference"] == "not_run"
    cluster = {c["id"]: c for c in passports["cluster"]["verification"]}
    assert cluster["cluster_postconditions"]["evidence"] == "internal_consistency"


def test_passport_scientific_payload_is_free_of_paths(passports) -> None:
    for p in passports.values():
        text = dumps_passport(p)
        assert str(ROOT) not in text and str(Path.home()) not in text
        assert "synthetic_regions.csv" not in text  # file names are not recorded either


# --- integrity -----------------------------------------------------------------------------------


def _ids_failing(p: dict) -> set[str]:
    return {c["id"] for c in integrity_checks(p) if c["status"] == "fail"}


def test_editing_a_result_value_breaks_result_and_passport_digests(passports) -> None:
    p = copy.deepcopy(passports["relate_pearson"])
    p["result"]["result"]["coefficient"] += 1e-15
    assert _ids_failing(p) == {"passport_digest", "result_digest"}


def test_editing_the_plan_is_detected_even_after_resealing_the_passport(passports) -> None:
    p = copy.deepcopy(passports["cluster"])
    p["inputs"]["plan"]["k"] = 4
    p = reseal(p)  # passport and result digests recomputed, plan digest left stale
    assert _ids_failing(p) == {"plan_digest", "identity_consistency"}


def test_editing_a_fingerprint_is_detected(passports) -> None:
    p = copy.deepcopy(passports["inspect"])
    p["identity"]["raw_sha256"] = "0" * 64
    p = reseal(p)
    assert _ids_failing(p) == {"identity_consistency"}


def test_implicit_plan_defaults_are_rejected(passports) -> None:
    p = copy.deepcopy(passports["cluster"])
    del p["inputs"]["plan"]["tol"]  # valid plan, but no longer the explicit hashed form
    assert "plan_digest" in _ids_failing(reseal(p))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace('"passport_version"', '"passport_version": 1, "passport_version"', 1),
        lambda t: t.replace('"execution"', '"unknown_field": 1, "execution"', 1),
        lambda t: t.replace(PASSPORT_VERSION, "axiom.passport/0"),
        lambda t: "[]",
        lambda t: t[: len(t) // 2],
    ],
    ids=["duplicate_key", "unknown_field", "version", "not_object", "truncated"],
)
def test_malformed_passport_text_is_rejected(passports, mutate) -> None:
    with pytest.raises(PassportError):
        loads_passport(mutate(dumps_passport(passports["inspect"])))


def test_nonfinite_constants_are_rejected(passports) -> None:
    text = dumps_passport(passports["relate_pearson"])
    value = repr(passports["relate_pearson"]["result"]["result"]["coefficient"])
    with pytest.raises(PassportError):
        loads_passport(text.replace(value, "NaN", 1))


def test_limits_round_trip_and_validation() -> None:
    import dataclasses

    data = dataclasses.asdict(DEFAULT_LIMITS)
    assert limits_from_dict(data) == DEFAULT_LIMITS
    for bad in ({**data, "max_k": 0}, {**data, "max_k": 2.0}, {k: data[k] for k in list(data)[1:]}):
        with pytest.raises(PassportError):
            limits_from_dict(bad)
