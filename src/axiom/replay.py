"""AXIOM Verify: fresh-process analysis, replay and comparison (ADR 0008).

Callable API (also exposed by ``python -m axiom``):

* :func:`analyze` runs a plan on a data file and returns an Analysis Passport.
* :func:`replay` re-executes a passport's plan on a data file in ``runs`` fresh processes and
  classifies the outcome against the passport.
* :func:`compare_passports` compares two passports (for example from two platforms) under the
  documented per-operation criteria.

Every execution happens in a new Python process (``-I`` isolated mode) with thread-control
variables set before any numerical library loads, in its own temporary working directory, with a
fresh run nonce. There is no result cache anywhere: the worker is never given a reference result.
The operating system's file cache may serve the input bytes; that is not a result cache.

Outcomes are kept in separate categories: an **integrity mismatch** (passport altered, data or plan
not the ones the passport describes), an **execution failure** (the computation did not complete),
and a **numerical discrepancy** (it completed but the result differs beyond the applicable
criterion). Same-environment replay is judged by byte identity; different environments by the
numerical criteria of :mod:`axiom.agreement`. Agreement is not correctness.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agreement import CRITERIA_VERSION, compare_cores
from .engine import AnalysisResult
from .environment import environment_differences
from .ingest import CsvProfile
from .limits import DEFAULT_LIMITS, Limits
from .passport import (
    build_passport,
    check,
    integrity_checks,
    limits_to_dict,
    profile_to_dict,
)
from .plan import CandidatePlan, plan_to_dict, revalidate_plan

REPORT_VERSION = "axiom.replay-report/1"
COMPARISON_VERSION = "axiom.passport-comparison/1"
WORKER_PROTOCOL = "axiom.worker/1"
DEFAULT_TIMEOUT_SECONDS = 600.0

# Reference mode: one native thread, configured before the worker interpreter imports anything.
REFERENCE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

VERDICTS = (
    "exact_match",
    "numeric_agreement",
    "numerical_discrepancy",
    "execution_failure",
    "integrity_mismatch",
    "not_run",
)

# Engine error codes that mean "this is not the data/plan the binding describes".
_DATASET_CODES = {"plan_not_bound_to_dataset"}
_PLAN_CODES = {"plan_hash_mismatch", "invalid_plan"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def run_worker(
    request: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Execute one request in a fresh interpreter and return its response plus parent timing.

    Never raises for worker problems: a crash, a timeout or unreadable output becomes a response
    with ``status`` ``"crashed"``/``"timeout"``/``"bad_output"``.
    """
    env = dict(os.environ)
    env.update(REFERENCE_THREAD_ENV)
    payload = json.dumps(dict(request, protocol=WORKER_PROTOCOL), allow_nan=False)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="axiom-run-") as workspace:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-m", "axiom._worker"],
                input=payload.encode("utf-8"),
                capture_output=True,
                cwd=workspace,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "parent_wall_seconds": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    try:
        response = json.loads(proc.stdout.decode("utf-8"))
        if not isinstance(response, dict):
            raise ValueError("response is not an object")
    except (UnicodeDecodeError, ValueError):
        status = "crashed" if proc.returncode != 0 else "bad_output"
        return {"status": status, "exit_code": proc.returncode, "parent_wall_seconds": wall}
    response["exit_code"] = proc.returncode
    response["parent_wall_seconds"] = wall
    response["fresh_workspace"] = True
    return response


def _request(
    data_path: str | Path,
    profile: CsvProfile,
    plan: dict[str, Any],
    limits: Limits,
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex,
        "data_path": os.fspath(Path(data_path).resolve()),
        "profile": profile_to_dict(profile),
        "plan": plan,
        "limits": limits_to_dict(limits),
        "binding": binding,
    }


def analyze(
    data_path: str | Path,
    profile: CsvProfile,
    plan: CandidatePlan,
    limits: Limits = DEFAULT_LIMITS,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run ``plan`` on the file in a fresh process and return an Analysis Passport.

    Raises the engine's structured error (as :class:`~axiom.errors.AxiomError` data inside a
    ``RuntimeError``) if the analysis itself fails; no passport is issued for a failed analysis.
    """
    plan = revalidate_plan(plan)
    request = _request(data_path, profile, plan_to_dict(plan), limits, None)
    response = run_worker(request, timeout=timeout)
    if response.get("status") != "ok":
        raise AnalysisFailed(response)
    result = AnalysisResult(response["result_canonical_json"])
    execution = {
        "run_id": request["run_id"],
        "mode": "fresh_process",
        "workspace": "fresh temporary directory, deleted after the run",
        "result_cache": "none: the worker recomputes from the data file",
        "thread_environment": dict(REFERENCE_THREAD_ENV),
        "started_utc": response.get("started_utc"),
        "finished_utc": response.get("finished_utc"),
        "timing_seconds": dict(
            response.get("timing", {}), parent_wall=response["parent_wall_seconds"]
        ),
    }
    return build_passport(
        result=result,
        profile=profile,
        limits=limits,
        n_bytes=response["fingerprints"]["n_bytes"],
        environment=response["environment"],
        execution=execution,
    )


class AnalysisFailed(RuntimeError):
    """The fresh-process analysis did not produce a result; ``response`` holds the details."""

    def __init__(self, response: dict[str, Any]) -> None:
        error = response.get("error") or {}
        super().__init__(f"analysis failed: {response.get('status')} {error.get('code', '')}")
        self.response = response


def _classify_run(
    passport: dict[str, Any], response: dict[str, Any], same_environment: bool
) -> dict[str, Any]:
    """Outcome of one replay run. Keeps integrity, execution and numerical categories apart."""
    identity = passport["identity"]
    run: dict[str, Any] = {
        "run_id": response.get("run_id"),
        "worker_status": response.get("status"),
        "timing_seconds": dict(
            response.get("timing", {}), parent_wall=response.get("parent_wall_seconds")
        ),
    }
    fp = response.get("fingerprints") or {}
    raw_ok = fp.get("raw_sha256") == identity["raw_sha256"] if fp else None
    logical_ok = (
        fp.get("dataset_sha256") == identity["dataset_sha256"] if "dataset_sha256" in fp else None
    )
    run["dataset"] = {"raw_bytes_match": raw_ok, "logical_dataset_match": logical_ok}
    error = response.get("error") or {}
    if response.get("status") != "ok":
        run["error"] = error or {"code": response.get("status")}
        code = error.get("code")
        if raw_ok is False or logical_ok is False or code in _DATASET_CODES:
            run["outcome"] = "integrity_mismatch"
            run["mismatch"] = "dataset"
        elif code in _PLAN_CODES:
            run["outcome"] = "integrity_mismatch"
            run["mismatch"] = "plan"
        else:
            run["outcome"] = "execution_failure"
        return run
    text = response["result_canonical_json"]
    observed = AnalysisResult(text)
    run["result_sha256"] = observed.result_sha256
    run["byte_identical"] = observed.result_sha256 == identity["result_sha256"]
    comparison = compare_cores(passport["result"], observed.core)
    run["comparison"] = _summarize(comparison)
    if run["byte_identical"]:
        run["outcome"] = "exact_match"
    elif same_environment:
        run["outcome"] = "numerical_discrepancy"  # exact replay failed in the same environment
    elif comparison["verdict"] == "within_criteria":
        run["outcome"] = "numeric_agreement"
    else:
        run["outcome"] = "numerical_discrepancy"
    return run


def _summarize(comparison: dict[str, Any], keep: int = 20) -> dict[str, Any]:
    """Comparison report with the difference lists truncated (counts are always complete)."""
    out = dict(comparison)
    for key in ("field_differences", "non_float_differences"):
        items = comparison.get(key, [])
        out[key] = items[:keep]
        out[f"{key}_truncated"] = len(items) > keep
    return out


def _timing_stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def replay(
    passport: dict[str, Any],
    data_path: str | Path,
    *,
    runs: int = 1,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Re-execute a passport's bound plan ``runs`` times, each in a fresh process, and judge it.

    The passport's own digests are checked first; if any fails nothing is executed (the verdict is
    ``integrity_mismatch``). Each run presents the *old* binding to the engine, so changed data is
    rejected by the engine itself, and a changed plan is rejected by its digest.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    started = _now()
    integrity = integrity_checks(passport)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "reference": {
            "passport_sha256": passport.get("passport_sha256"),
            "result_sha256": passport["identity"].get("result_sha256"),
            "plan_sha256": passport["identity"].get("plan_sha256"),
            "dataset_sha256": passport["identity"].get("dataset_sha256"),
            "raw_sha256": passport["identity"].get("raw_sha256"),
            "operation": passport["result"].get("operation"),
            "run_id": passport["execution"].get("run_id"),
        },
        "started_utc": started,
        "runs_requested": runs,
        "integrity": integrity,
    }
    if any(c["status"] == "fail" for c in integrity):
        report.update(
            runs_completed=0,
            runs=[],
            verdict="integrity_mismatch",
            mismatch="passport",
            comparison_mode=None,
            environment_differences=None,
        )
        report["checks"] = _report_checks(report, None)
        report["finished_utc"] = _now()
        return report

    from .passport import limits_from_dict, profile_from_dict

    profile = profile_from_dict(passport["inputs"]["csv_profile"])
    limits = limits_from_dict(passport["inputs"]["limits"])
    identity = passport["identity"]
    binding = {
        "plan_sha256": identity["plan_sha256"],
        "raw_sha256": identity["raw_sha256"],
        "dataset_sha256": identity["dataset_sha256"],
        "policy_version": identity["policy_version"],
    }
    run_reports: list[dict[str, Any]] = []
    env_diffs: list[dict[str, Any]] | None = None
    for _ in range(runs):
        request = _request(data_path, profile, passport["inputs"]["plan"], limits, binding)
        response = run_worker(request, timeout=timeout)
        observed_env = response.get("environment")
        diffs = (
            environment_differences(passport["environment"], observed_env)
            if isinstance(observed_env, dict)
            else [{"field": "environment", "comparable": False}]
        )
        if env_diffs is None or (diffs and not env_diffs):
            env_diffs = diffs
        run_reports.append(_classify_run(passport, response, same_environment=not diffs))
    same_env = not env_diffs
    outcomes = [r["outcome"] for r in run_reports]
    for verdict in ("integrity_mismatch", "execution_failure", "numerical_discrepancy"):
        if verdict in outcomes:
            break
    else:
        verdict = (
            "exact_match" if all(o == "exact_match" for o in outcomes) else "numeric_agreement"
        )
    report.update(
        runs_completed=sum(1 for r in run_reports if r["worker_status"] == "ok"),
        runs=run_reports,
        verdict=verdict,
        comparison_mode="same_environment_exact" if same_env else "cross_environment_numeric",
        environment_differences=env_diffs,
        summary=_run_summary(run_reports),
    )
    if verdict == "integrity_mismatch":
        report["mismatch"] = next(r["mismatch"] for r in run_reports if "mismatch" in r)
    report["checks"] = _report_checks(report, same_env)
    report["finished_utc"] = _now()
    return report


def _run_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def times(key: str) -> list[float]:
        return [
            r["timing_seconds"][key]
            for r in runs
            if isinstance(r["timing_seconds"].get(key), (int, float))
        ]

    outcomes: dict[str, int] = {}
    for r in runs:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    hashes = {r["result_sha256"] for r in runs if "result_sha256" in r}
    return {
        "outcome_counts": outcomes,
        "distinct_result_sha256": len(hashes),
        "distinct_run_ids": len({r["run_id"] for r in runs}),
        "timing_seconds": {
            "parent_wall": _timing_stats(times("parent_wall")),
            "compute": _timing_stats(times("compute_seconds")),
            "load": _timing_stats(times("load_seconds")),
        },
    }


def _report_checks(report: dict[str, Any], same_env: bool | None) -> list[dict[str, Any]]:
    """Summary checks in the shared vocabulary. Unperformed checks are ``not_run``, never pass."""
    runs = report.get("runs", [])
    passport_ok = all(c["status"] == "pass" for c in report["integrity"])
    checks = [
        check(
            "passport_integrity",
            "exact",
            "integrity",
            "pass" if passport_ok else "fail",
            "all digests stated in the passport recomputed",
        )
    ]
    if not runs:
        for cid, kind, ev in (
            ("dataset_identity", "exact", "integrity"),
            ("plan_identity", "exact", "integrity"),
            ("environment_match", "exact", "environment"),
            ("exact_replay", "exact", "repeatability"),
            ("numerical_agreement", "numeric", "cross_environment"),
        ):
            checks.append(check(cid, kind, ev, "not_run", "no replay was executed"))
    else:
        data_bad = any(r.get("mismatch") == "dataset" for r in runs)
        data_ok = all(
            r["dataset"]["raw_bytes_match"] and r["dataset"]["logical_dataset_match"] for r in runs
        )
        plan_bad = any(r.get("mismatch") == "plan" for r in runs)
        plan_ok = all(r["worker_status"] == "ok" for r in runs)  # engine accepted the binding
        checks.append(
            check(
                "dataset_identity",
                "exact",
                "integrity",
                "fail" if data_bad else ("pass" if data_ok else "not_run"),
                "data file bytes and typed dataset match the passport fingerprints",
            )
        )
        checks.append(
            check(
                "plan_identity",
                "exact",
                "integrity",
                "fail" if plan_bad else ("pass" if plan_ok else "not_run"),
                "the engine accepted the passport's bound plan digest in every run",
            )
        )
        checks.append(
            check(
                "environment_match",
                "exact",
                "environment",
                "pass" if same_env else "fail",
                "comparable environment fields equal (unknown counts as different)",
            )
        )
        ok_runs = [r for r in runs if r["worker_status"] == "ok"]
        if not ok_runs:
            exact_status = numeric_status = "not_run"
        elif same_env:
            exact_status = "pass" if all(r["byte_identical"] for r in ok_runs) else "fail"
            if len(ok_runs) < len(runs):
                exact_status = "fail"
            numeric_status = "not_applicable"
        else:
            exact_status = "not_applicable"
            numeric_status = (
                "pass"
                if len(ok_runs) == len(runs)
                and all(r["outcome"] in ("exact_match", "numeric_agreement") for r in ok_runs)
                else "fail"
            )
        checks.append(
            check(
                "exact_replay",
                "exact",
                "repeatability",
                exact_status,
                "canonical result bytes identical in every run (same environment only)",
                runs=len(runs),
            )
        )
        checks.append(
            check(
                "numerical_agreement",
                "numeric",
                "cross_environment",
                numeric_status,
                f"per-operation criteria {CRITERIA_VERSION} (different environment only)",
            )
        )
        if report["reference"]["operation"] == "cluster" and ok_runs:
            same = [r["comparison"].get("partition", {}).get("same_partition") for r in ok_runs]
            checks.append(
                check(
                    "partition_identity",
                    "semantic",
                    "repeatability" if same_env else "cross_environment",
                    "pass" if all(s is True for s in same) else "fail",
                    "membership sets of source rows equal; label numbering ignored; no "
                    "global-optimum claim",
                )
            )
    checks.append(
        check(
            "independent_reference",
            "numeric",
            "independent_reference",
            "not_run",
            "replay compares a result with itself; independent oracles are in the test suite",
        )
    )
    return checks


def compare_passports(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Compare two passports' results (for example macOS arm64 versus Windows x64).

    Both passports are integrity-checked first. The inputs (raw bytes, typed dataset, plan) must be
    identical, otherwise the comparison is an integrity mismatch and the results are not compared.
    """
    report: dict[str, Any] = {
        "report_version": COMPARISON_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "reference_integrity": integrity_checks(reference),
        "observed_integrity": integrity_checks(observed),
    }
    bad = [
        c
        for c in report["reference_integrity"] + report["observed_integrity"]
        if c["status"] == "fail"
    ]
    if bad:
        report["verdict"] = "integrity_mismatch"
        report["mismatch"] = "passport"
        return report
    ri, oi = reference["identity"], observed["identity"]
    same_inputs = {
        key: ri[key] == oi[key] for key in ("raw_sha256", "dataset_sha256", "plan_sha256")
    }
    report["same_inputs"] = same_inputs
    if not all(same_inputs.values()):
        report["verdict"] = "integrity_mismatch"
        report["mismatch"] = "inputs"
        return report
    diffs = environment_differences(reference["environment"], observed["environment"])
    report["environment_differences"] = diffs
    report["comparison_mode"] = (
        "same_environment_exact" if not diffs else "cross_environment_numeric"
    )
    comparison = compare_cores(reference["result"], observed["result"])
    report["comparison"] = comparison
    report["byte_identical"] = comparison["byte_identical"]
    if comparison["byte_identical"]:
        report["verdict"] = "exact_match"
    elif not diffs or comparison["verdict"] != "within_criteria":
        report["verdict"] = "numerical_discrepancy"
    else:
        report["verdict"] = "numeric_agreement"
    return report
