"""Fresh-process analysis and replay (ADR 0008): exact replay, mutation, adversarial passports.

These tests start real Python subprocesses. Each worker re-reads the data file and recomputes; no
reference result is ever passed to it.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import subprocess
import sys
from fractions import Fraction

import pytest

from _passport_helpers import DATA, EXAMPLES, PROFILE, ROOT, example_plan, reseal
from axiom import load_csv, run
from axiom import replay as replay_module
from axiom.passport import dumps_passport, integrity_checks
from axiom.replay import (
    REFERENCE_THREAD_ENV,
    AnalysisFailed,
    analyze,
    compare_passports,
    replay,
    run_worker,
)

# Byte identity is required in both: environment verified equal, or nothing observed to differ but
# some field unknown (on Apple Silicon NumPy's Accelerate BLAS is invisible to threadpoolctl).
EXACT_MODES = ("same_environment_exact", "exact_environment_unverified")


@pytest.fixture(scope="module")
def passports() -> dict[str, dict]:
    return {n: analyze(DATA, PROFILE, example_plan(n)) for n in ("inspect", "compare", "cluster")}


def _checks(report: dict) -> dict[str, str]:
    return {c["id"]: c["status"] for c in report["checks"]}


def _mutated_csv(tmp_path, old: str, new: str, name: str = "mutated.csv"):
    text = DATA.read_text("utf-8")
    assert text.count(old) == 1
    path = tmp_path / name
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    return path


# --- analysis in a fresh process -----------------------------------------------------------------


def test_fresh_process_passport_matches_in_process_result(passports) -> None:
    ds = load_csv(DATA, PROFILE)
    for name, p in passports.items():
        assert p["execution"]["mode"] == "fresh_process"
        assert p["execution"]["thread_environment"] == REFERENCE_THREAD_ENV
        assert p["identity"]["result_sha256"] == run(ds, example_plan(name)).result_sha256
        assert all(c["status"] == "pass" for c in integrity_checks(p))


def test_worker_threads_are_configured_before_numerical_imports(passports) -> None:
    threads = passports["cluster"]["environment"]["threads"]
    assert all(v == "1" for v in threads["environment_variables"].values())
    pools = threads["detected_pools"]
    assert pools, "expected at least the OpenMP pool used by scikit-learn"
    assert all(p["num_threads"] == 1 for p in pools)  # effective, not merely requested
    assert threads["fit_limit_probe"] == "verified_single_thread"


def test_analysis_failure_issues_no_passport(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_bytes(DATA.read_bytes() + b"R999,Alpha,1\n")  # short record
    with pytest.raises(AnalysisFailed) as e:
        analyze(bad, PROFILE, example_plan("inspect"))
    assert e.value.response["error"]["code"] == "record_width"


# --- same-environment replay ---------------------------------------------------------------------


def test_replay_in_fresh_processes_is_byte_identical(passports) -> None:
    report = replay(passports["cluster"], DATA, runs=2)
    assert report["verdict"] == "exact_match"
    assert report["comparison_mode"] in EXACT_MODES
    assert report["summary"]["distinct_result_sha256"] == 1
    assert report["summary"]["distinct_run_ids"] == 2
    assert report["runs_completed"] == 2
    checks = _checks(report)
    assert checks["exact_replay"] == "pass" and checks["partition_identity"] == "pass"
    assert checks["numerical_agreement"] == "not_applicable"
    assert checks["independent_reference"] == "not_run"  # replay is not an oracle comparison
    assert all(r["timing_seconds"]["compute_seconds"] > 0 for r in report["runs"])


def test_worker_is_never_given_a_reference_result(passports, monkeypatch) -> None:
    seen: list[dict] = []

    def fake(request, *, timeout):
        seen.append(request)
        return {"status": "timeout", "parent_wall_seconds": 0.0}

    monkeypatch.setattr(replay_module, "run_worker", fake)
    report = replay(passports["inspect"], DATA, runs=2)
    assert report["verdict"] == "execution_failure"
    assert len(seen) == 2 and seen[0]["run_id"] != seen[1]["run_id"]
    for request in seen:
        assert set(request) == {"run_id", "data_path", "profile", "plan", "limits", "binding"}
        assert "result" not in json.dumps(request["binding"])


# --- mutation: old binding rejected, rebinding gives the independently expected change -----------


def _column(path, name: str) -> list[str]:
    rows = list(csv.DictReader(io.StringIO(path.read_text("utf-8"), newline="")))
    return [r[name] for r in rows]


def test_old_bound_plan_is_rejected_on_changed_data(passports, tmp_path) -> None:
    changed = _mutated_csv(tmp_path, "R001,Charlie,23400.39", "R001,Charlie,23400.40")
    report = replay(passports["inspect"], changed)
    assert report["verdict"] == "integrity_mismatch" and report["mismatch"] == "dataset"
    (run_report,) = report["runs"]
    assert run_report["error"]["code"] == "plan_not_bound_to_dataset"  # the engine's own check
    assert run_report["dataset"] == {"raw_bytes_match": False, "logical_dataset_match": False}
    checks = _checks(report)
    assert checks["dataset_identity"] == "fail"
    assert checks["plan_identity"] == "not_run"  # never reached; not reported as passed
    assert checks["exact_replay"] == "not_run"


def test_rebinding_after_mutation_gives_the_independently_expected_statistics(
    passports, tmp_path
) -> None:
    changed = _mutated_csv(tmp_path, "R001,Charlie,23400.39", "R001,Charlie,93400.39")
    new = analyze(changed, PROFILE, example_plan("inspect"))
    assert new["identity"]["dataset_sha256"] != passports["inspect"]["identity"]["dataset_sha256"]
    assert new["identity"]["plan_sha256"] == passports["inspect"]["identity"]["plan_sha256"]

    def stats(p: dict, column: str) -> dict:
        return next(c for c in p["result"]["result"]["columns"] if c["name"] == column)["stats"]

    # independent expectation from the CSV text: csv module, float(), exact rational mean
    values = [float(t) for t in _column(changed, "gdp_per_capita") if t != ""]
    exact = sum(map(Fraction, values), Fraction(0)) / len(values)
    mean = stats(new, "gdp_per_capita")["mean"]
    assert abs(Fraction(mean) - exact) <= Fraction(math.ulp(mean))
    assert stats(new, "gdp_per_capita")["max"] == max(values) == 93400.39
    old_mean = stats(passports["inspect"], "gdp_per_capita")["mean"]
    assert mean - old_mean == pytest.approx(70000 / len(values), rel=1e-9)
    for other in ("unemployment_rate", "education_years", "household_income"):
        assert stats(new, other) == stats(passports["inspect"], other)  # untouched columns
    assert replay(new, changed)["verdict"] == "exact_match"


def test_rebinding_after_group_change_moves_exactly_one_row(passports, tmp_path) -> None:
    line = next(ln for ln in DATA.read_text("utf-8").splitlines() if ",Alpha," in ln)
    changed = _mutated_csv(tmp_path, line, line.replace(",Alpha,", ",Bravo,"))
    old = {g["key"]: g["n_total"] for g in passports["compare"]["result"]["result"]["groups"]}
    new_p = analyze(changed, PROFILE, example_plan("compare"))
    new = {g["key"]: g["n_total"] for g in new_p["result"]["result"]["groups"]}
    assert new["Alpha"] == old["Alpha"] - 1 and new["Bravo"] == old["Bravo"] + 1
    assert {k: v for k, v in new.items() if k not in ("Alpha", "Bravo")} == {
        k: v for k, v in old.items() if k not in ("Alpha", "Bravo")
    }


def test_same_typed_data_with_different_bytes_is_still_an_integrity_mismatch(
    passports, tmp_path
) -> None:
    path = tmp_path / "crlf.csv"
    path.write_bytes(DATA.read_bytes().replace(b"\n", b"\r\n"))
    report = replay(passports["inspect"], path)
    assert report["verdict"] == "integrity_mismatch" and report["mismatch"] == "dataset"
    assert report["runs"][0]["dataset"] == {"raw_bytes_match": False, "logical_dataset_match": True}


def test_malformed_replacement_data_is_an_integrity_mismatch(passports, tmp_path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_bytes(b'region_id,macro_region\n"unterminated\n')
    report = replay(passports["inspect"], path)
    assert report["verdict"] == "integrity_mismatch"
    assert report["runs"][0]["error"]["code"] == "schema_mismatch"


# --- adversarial passports -----------------------------------------------------------------------


def test_stale_plan_digest_stops_replay_before_any_execution(passports) -> None:
    p = copy.deepcopy(passports["cluster"])
    p["inputs"]["plan"]["k"] = 4
    p["result"]["plan"]["k"] = 4
    report = replay(reseal(p), DATA)
    assert report["verdict"] == "integrity_mismatch" and report["mismatch"] == "passport"
    assert report["runs_completed"] == 0 and report["runs"] == []
    checks = _checks(report)
    assert checks["passport_integrity"] == "fail"
    assert all(checks[c] == "not_run" for c in ("exact_replay", "numerical_agreement"))


def test_forged_plan_with_inconsistent_result_fails_integrity_without_execution(passports) -> None:
    p = copy.deepcopy(passports["cluster"])
    p["inputs"]["plan"]["k"] = 4
    p["result"]["plan"]["k"] = 4
    forged = reseal(p, plan_too=True)  # every digest is consistent: digests are not signatures
    statuses = {c["id"]: c["status"] for c in integrity_checks(forged)}
    assert statuses["plan_digest"] == "pass" and statuses["result_consistency"] == "fail"
    report = replay(forged, DATA)
    assert report["verdict"] == "integrity_mismatch" and report["runs"] == []


def test_consistently_forged_result_is_exposed_only_by_recomputation(passports) -> None:
    p = copy.deepcopy(passports["cluster"])
    labels = p["result"]["result"]["assignments"]
    i = labels.index(0)
    j = labels.index(1)
    labels[i], labels[j] = 1, 0  # same cluster sizes: internally consistent, different partition
    forged = reseal(p)
    assert all(c["status"] == "pass" for c in integrity_checks(forged))  # not authentication
    report = replay(forged, DATA)
    assert report["verdict"] == "numerical_discrepancy"
    comparison = report["runs"][0]["comparison"]
    assert comparison["verdict"] == "outside_criteria"
    assert comparison["partition"]["same_partition"] is False


def _forge_mean(passport: dict, steps: int) -> dict:
    p = copy.deepcopy(passport)
    stats = p["result"]["result"]["columns"][0]["stats"]
    for _ in range(steps):
        stats["mean"] = math.nextafter(stats["mean"], math.inf)
    return reseal(p)


def test_same_environment_requires_exact_bytes_not_tolerance(passports) -> None:
    report = replay(_forge_mean(passports["inspect"], 1), DATA)
    assert report["comparison_mode"] in EXACT_MODES
    assert report["verdict"] == "numerical_discrepancy"
    assert report["runs"][0]["comparison"]["verdict"] == "within_criteria"  # but not exact
    checks = _checks(report)
    assert checks["exact_replay"] == "fail" and checks["numerical_agreement"] == "not_applicable"


def test_different_environment_uses_numeric_criteria_and_reports_the_difference(passports) -> None:
    p = _forge_mean(passports["inspect"], 1)
    p["environment"]["packages"]["numpy"] = "0.0.0-recorded-elsewhere"
    p = reseal(p)
    report = replay(p, DATA)
    assert report["comparison_mode"] == "cross_environment_numeric"
    assert [d["field"] for d in report["environment"]["differences"]] == ["packages"]
    assert report["verdict"] == "numeric_agreement"
    checks = _checks(report)
    assert checks["environment_match"] == "fail"
    assert checks["exact_replay"] == "not_applicable" and checks["numerical_agreement"] == "pass"
    worse = reseal(dict(_forge_mean(passports["inspect"], 5), environment=p["environment"]))
    assert replay(worse, DATA)["verdict"] == "numerical_discrepancy"


def test_execution_failures_are_not_integrity_or_numerical_findings(passports, tmp_path) -> None:
    report = replay(passports["inspect"], DATA, timeout=0.001)
    assert report["verdict"] == "execution_failure"
    assert report["runs"][0]["error"]["code"] == "timeout"
    assert _checks(report)["exact_replay"] == "not_run"
    missing = replay(passports["inspect"], tmp_path / "missing.csv")
    assert missing["verdict"] == "execution_failure"
    assert missing["runs"][0]["error"]["code"] == "unreadable_file"


def test_worker_rejects_a_bad_request() -> None:
    response = run_worker({"run_id": "x"})  # missing every field
    assert response["status"] == "error"
    assert response["error"]["code"] == "unexpected_exception"


# --- passport comparison (cross-environment) -----------------------------------------------------


def test_compare_passports(passports) -> None:
    a = passports["inspect"]
    assert compare_passports(a, copy.deepcopy(a))["verdict"] == "exact_match"
    elsewhere = copy.deepcopy(_forge_mean(a, 2))
    elsewhere["environment"]["os"]["system"] = "recorded-on-another-os"  # differs on every runner
    elsewhere = reseal(elsewhere)
    report = compare_passports(a, elsewhere)
    assert report["comparison_mode"] == "cross_environment_numeric"
    assert report["verdict"] == "numeric_agreement"
    assert compare_passports(a, passports["compare"])["verdict"] == "integrity_mismatch"
    tampered = copy.deepcopy(a)
    tampered["identity"]["raw_sha256"] = "0" * 64
    assert compare_passports(a, tampered)["mismatch"] == "passport"


# --- command-line interface ----------------------------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "axiom", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
        check=False,
    )


def test_cli_analyze_replay_compare(tmp_path, passports) -> None:
    out = tmp_path / "p.json"
    profile = str(EXAMPLES / "profiles" / "synthetic_regions.json")
    plan = str(EXAMPLES / "plans" / "relate_spearman.json")
    proc = _cli(
        "analyze", "--data", str(DATA), "--profile", profile, "--plan", plan, "--out", str(out)
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = _cli("replay", "--passport", str(out), "--data", str(DATA), "--runs", "2")
    assert proc.returncode == 0 and json.loads(proc.stdout)["verdict"] == "exact_match"
    assert _cli("compare", "--reference", str(out), "--observed", str(out)).returncode == 0
    other = tmp_path / "other.json"
    other.write_text(dumps_passport(passports["inspect"]), encoding="utf-8")
    proc = _cli("compare", "--reference", str(out), "--observed", str(other))
    assert proc.returncode == 1 and json.loads(proc.stdout)["verdict"] == "integrity_mismatch"
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    proc = _cli("replay", "--passport", str(bad), "--data", str(DATA))
    assert proc.returncode == 2 and json.loads(proc.stdout)["verdict"] == "input_error"
    env = _cli("env")
    assert env.returncode == 0 and json.loads(env.stdout)["schema_version"] == "axiom.environment/1"
