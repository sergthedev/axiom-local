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

import dataclasses
import json
import math
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
from .environment import compare_environments, native_thread_status
from .errors import AxiomError
from .ingest import CsvProfile
from .limits import DEFAULT_LIMITS, Limits
from .passport import (
    build_passport,
    check,
    integrity_checks,
    limits_from_dict,
    limits_to_dict,
    profile_from_dict,
    profile_to_dict,
    validate_passport,
)
from .plan import CandidatePlan, plan_to_dict, revalidate_plan

REPORT_VERSION = "axiom.replay-report/1"
COMPARISON_VERSION = "axiom.passport-comparison/1"
WORKER_PROTOCOL = "axiom.worker/1"
DEFAULT_TIMEOUT_SECONDS = 600.0
# Bounds on caller-supplied arguments of the public entry points (review B1). 100 runs is the
# release target per golden scenario (DR-001); one hour per run is far above any measured run.
MAX_REPLAY_RUNS = 100
MAX_TIMEOUT_SECONDS = 3600.0

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


class VerifyError(AxiomError):
    """A verification request was rejected before anything was executed."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _check_timeout(timeout: Any) -> float:
    # No float conversion before the range check: math.isfinite(10**400) raises OverflowError,
    # while comparing an int with a float is exact in Python (final audit).
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or (isinstance(timeout, float) and not math.isfinite(timeout))
        or not 0 < timeout <= MAX_TIMEOUT_SECONDS
    ):
        raise VerifyError(
            "invalid_timeout",
            "timeout must be a finite number of seconds in (0, max]",
            max=MAX_TIMEOUT_SECONDS,
        )
    return float(timeout)


def _check_data_path(data_path: Any) -> Path:
    """A str or path-like naming the data file (read by the worker, never stored in a report)."""
    try:
        path = Path(data_path)
    except TypeError:
        raise VerifyError("invalid_data_path", "data path must be a str or path-like") from None
    if "\x00" in str(path):
        raise VerifyError("invalid_data_path", "data path contains a NUL character")
    return path


def _check_runs(runs: Any) -> int:
    if type(runs) is not int or not 1 <= runs <= MAX_REPLAY_RUNS:
        raise VerifyError(
            "invalid_runs", "runs must be an integer in [1, max]", max=MAX_REPLAY_RUNS
        )
    return runs


def _check_limits(limits: Any) -> Limits:
    """A ``Limits`` instance whose fields are all positive integers."""
    if not isinstance(limits, Limits):
        raise VerifyError("invalid_limits", "limits must be an axiom Limits instance")
    try:
        return limits_from_dict(limits_to_dict(limits))
    except AxiomError as exc:
        raise VerifyError("invalid_limits", exc.message) from None


def limits_within_policy(recorded: Limits, policy: Limits) -> list[str]:
    """Names of the recorded limits that exceed the trusted local policy (empty if none)."""
    return [
        f.name
        for f in dataclasses.fields(Limits)
        if getattr(recorded, f.name) > getattr(policy, f.name)
    ]


def run_worker(
    request: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Execute one request in a fresh interpreter and return its response plus parent timing.

    Never raises for worker problems: a crash, a timeout or unreadable output becomes a response
    with ``status`` ``"crashed"``/``"timeout"``/``"bad_output"``. An invalid ``timeout`` is
    rejected (:class:`VerifyError`) before any process is started.
    """
    timeout = _check_timeout(timeout)
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
    except (UnicodeDecodeError, ValueError, RecursionError):
        status = "crashed" if proc.returncode != 0 else "bad_output"
        return {"status": status, "exit_code": proc.returncode, "parent_wall_seconds": wall}
    if not isinstance(response.get("timing", {}), dict) or not isinstance(
        response.get("error") or {}, dict
    ):
        # e.g. the worker's bad_request response carries its error as text
        response["error"] = {"code": str(response.get("status")), "message": "worker error"}
        response["timing"] = {}
    if response.get("status") == "ok" and (proc.returncode != 0 or not _ok_shape(response)):
        # A successful response must carry everything the parent reads (final audit), so that a
        # malformed worker response is a reported failure, never an exception in the parent.
        status = "crashed" if proc.returncode != 0 else "bad_output"
        return {"status": status, "exit_code": proc.returncode, "parent_wall_seconds": wall}
    response["exit_code"] = proc.returncode
    response["parent_wall_seconds"] = wall
    response["fresh_workspace"] = True
    return response


def _ok_shape(response: dict[str, Any]) -> bool:
    fp = response.get("fingerprints")
    return (
        isinstance(response.get("result_canonical_json"), str)
        and isinstance(response.get("environment"), dict)
        and isinstance(fp, dict)
        and isinstance(fp.get("raw_sha256"), str)
        and isinstance(fp.get("dataset_sha256"), str)
        and type(fp.get("n_bytes")) is int
    )


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
    timeout = _check_timeout(timeout)
    limits = _check_limits(limits)  # the caller's own, trusted limits
    data_path = _check_data_path(data_path)
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


def _thread_evidence(environment: Any) -> dict[str, Any]:
    """What this worker itself observed about native threads (stored per run; review B3)."""
    threads = environment.get("threads") if isinstance(environment, dict) else None
    if not isinstance(threads, dict):
        return {"observed": False, "native_thread_status": native_thread_status(environment)}
    pools = threads.get("detected_pools")
    return {
        "observed": True,
        "environment_variables": threads.get("environment_variables"),
        "pools": (
            [
                {k: p.get(k) for k in ("user_api", "internal_api", "prefix", "num_threads")}
                for p in pools
                if isinstance(p, dict)
            ]
            if isinstance(pools, list)
            else pools
        ),
        "fit_limit_probe": threads.get("fit_limit_probe"),
        "blas_thread_control": threads.get("blas_thread_control"),
        "native_thread_status": native_thread_status(environment),
    }


def _classify_run(
    passport: dict[str, Any], response: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    """Outcome of one replay run. Keeps integrity, execution and numerical categories apart.

    ``environment`` is :func:`~axiom.environment.compare_environments` of the passport and this
    worker. Byte identity is required unless the environments were *observed* to differ.
    """
    identity = passport["identity"]
    exact_required = environment["status"] != "different"
    run: dict[str, Any] = {
        "run_id": response.get("run_id"),
        "worker_status": response.get("status"),
        "environment_status": environment["status"],
        "threads": _thread_evidence(response.get("environment")),
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
    try:
        observed_core = observed.core
        if not isinstance(observed_core, dict):
            raise ValueError("result is not an object")
    except (ValueError, RecursionError):
        # unreadable result text: a failed run, never a crash of the parent (final audit)
        run["worker_status"] = "bad_output"
        run["error"] = {"code": "bad_output"}
        run["outcome"] = "execution_failure"
        return run
    run["result_sha256"] = observed.result_sha256
    run["byte_identical"] = observed.result_sha256 == identity["result_sha256"]
    comparison = compare_cores(passport["result"], observed_core)
    run["comparison"] = _summarize(comparison)
    if run["byte_identical"]:
        run["outcome"] = "exact_match"
    elif exact_required:
        # exact replay failed where no environment difference was observed (verified or unknown)
        run["outcome"] = "numerical_discrepancy"
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


_ENV_ORDER = {"verified_same": 0, "unverified": 1, "different": 2}
_COMPARISON_MODES = {
    "verified_same": "same_environment_exact",
    "unverified": "exact_environment_unverified",
    "different": "cross_environment_numeric",
}


def replay(
    passport: dict[str, Any],
    data_path: str | Path,
    *,
    runs: int = 1,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    policy: Limits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Re-execute a passport's bound plan ``runs`` times, each in a fresh process, and judge it.

    Before anything executes: the passport's structure is validated and the arguments are bounded
    (:class:`~axiom.passport.PassportError` / :class:`VerifyError`); the limits the passport
    *records* are compared with ``policy``, the trusted local caps, and a passport asking for more
    is rejected (``limits_exceed_policy``), never clamped, since different limits can be a
    different analysis; then the passport's digests and internal consistency are checked, and if
    any fails nothing is executed (verdict ``integrity_mismatch``). Each run then presents the
    *old* binding to the engine, so changed data is rejected by the engine itself, and a changed
    plan by its digest.
    """
    validate_passport(passport)
    runs = _check_runs(runs)
    timeout = _check_timeout(timeout)
    policy = _check_limits(policy)
    data_path = _check_data_path(data_path)
    # Trusted policy first, before any integrity work (final review B1). Limits that cannot be
    # reconstructed are reported by the integrity check replay_inputs_valid instead.
    try:
        recorded_limits: Limits | None = limits_from_dict(passport["inputs"]["limits"])
    except AxiomError:
        recorded_limits = None
    if recorded_limits is not None:
        over = limits_within_policy(recorded_limits, policy)
        if over:
            raise VerifyError(
                "limits_exceed_policy",
                "the passport records resource limits above the local policy; nothing was executed",
                fields=over,
            )
    started = _now()
    integrity = integrity_checks(passport)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "reference": {
            "passport_sha256": passport["passport_sha256"],
            "result_sha256": passport["identity"]["result_sha256"],
            "plan_sha256": passport["identity"]["plan_sha256"],
            "dataset_sha256": passport["identity"]["dataset_sha256"],
            "raw_sha256": passport["identity"]["raw_sha256"],
            "operation": passport["result"]["operation"],
            "run_id": passport["execution"]["run_id"],
            "recorded_verification": "historical record from the passport; not performed again",
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
            environment=None,
            environment_differences=None,
        )
        report["checks"] = _report_checks(report, None)
        report["finished_utc"] = _now()
        return report

    limits = limits_from_dict(passport["inputs"]["limits"])  # valid: replay_inputs_valid passed
    profile = profile_from_dict(passport["inputs"]["csv_profile"])
    identity = passport["identity"]
    binding = {
        "plan_sha256": identity["plan_sha256"],
        "raw_sha256": identity["raw_sha256"],
        "dataset_sha256": identity["dataset_sha256"],
        "policy_version": identity["policy_version"],
    }
    run_reports: list[dict[str, Any]] = []
    worst_env: dict[str, Any] | None = None
    for _ in range(runs):
        request = _request(data_path, profile, passport["inputs"]["plan"], limits, binding)
        response = run_worker(request, timeout=timeout)
        env = compare_environments(passport["environment"], response.get("environment"))
        if worst_env is None or _ENV_ORDER[env["status"]] > _ENV_ORDER[worst_env["status"]]:
            worst_env = env
        run_reports.append(_classify_run(passport, response, env))
    assert worst_env is not None  # runs >= 1
    env_status = worst_env["status"]
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
        comparison_mode=_COMPARISON_MODES[env_status],
        environment={
            **worst_env,
            "run_status_counts": _counts(r["environment_status"] for r in run_reports),
        },
        environment_differences=[dict(d, comparable=True) for d in worst_env["differences"]]
        + [dict(d, comparable=False) for d in worst_env["unverified_fields"]],
        native_threads=_counts(r["threads"]["native_thread_status"]["status"] for r in run_reports),
        summary=_run_summary(run_reports),
    )
    if verdict == "integrity_mismatch":
        report["mismatch"] = next(r["mismatch"] for r in run_reports if "mismatch" in r)
    report["checks"] = _report_checks(report, env_status)
    report["finished_utc"] = _now()
    return report


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def _run_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def times(key: str) -> list[float]:
        return [
            r["timing_seconds"][key]
            for r in runs
            if isinstance(r["timing_seconds"].get(key), (int, float))
        ]

    hashes = {r["result_sha256"] for r in runs if "result_sha256" in r}
    return {
        "outcome_counts": _counts(r["outcome"] for r in runs),
        "distinct_result_sha256": len(hashes),
        "distinct_run_ids": len({r["run_id"] for r in runs}),
        "timing_seconds": {
            "parent_wall": _timing_stats(times("parent_wall")),
            "compute": _timing_stats(times("compute_seconds")),
            "load": _timing_stats(times("load_seconds")),
        },
    }


def _report_checks(report: dict[str, Any], env_status: str | None) -> list[dict[str, Any]]:
    """Summary checks in the shared vocabulary. Unperformed checks are ``not_run``, never pass.

    Every check that is about "every run" passes only if *all requested* runs completed and
    satisfied it; agreement observed on a successful subset is reported as such, not as a pass.
    """
    runs = report.get("runs", [])
    n_requested = report.get("runs_requested", len(runs))
    passport_ok = all(c["status"] == "pass" for c in report["integrity"])
    checks = [
        check(
            "passport_integrity",
            "exact",
            "integrity",
            "pass" if passport_ok else "fail",
            "all digests stated in the passport recomputed and its record is internally "
            "consistent (not authentication)",
        )
    ]
    if not runs:
        for cid, kind, ev in (
            ("dataset_identity", "exact", "integrity"),
            ("plan_identity", "exact", "integrity"),
            ("environment_match", "exact", "environment"),
            ("native_thread_policy", "exact", "environment"),
            ("exact_replay", "exact", "repeatability"),
            ("numerical_agreement", "numeric", "cross_environment"),
        ):
            checks.append(check(cid, kind, ev, "not_run", "no replay was executed"))
    else:
        ok_runs = [r for r in runs if r["worker_status"] == "ok"]
        complete = len(ok_runs) == n_requested == len(runs)
        data_bad = any(r.get("mismatch") == "dataset" for r in runs)
        data_ok = all(
            r["dataset"]["raw_bytes_match"] and r["dataset"]["logical_dataset_match"] for r in runs
        )
        plan_bad = any(r.get("mismatch") == "plan" for r in runs)
        checks.append(
            check(
                "dataset_identity",
                "exact",
                "integrity",
                "fail" if data_bad else ("pass" if data_ok else "not_run"),
                "data file bytes and typed dataset match the passport fingerprints in every run",
            )
        )
        checks.append(
            check(
                "plan_identity",
                "exact",
                "integrity",
                "fail" if plan_bad else ("pass" if complete else "not_run"),
                "the engine accepted the passport's bound plan digest in every run",
            )
        )
        env = report["environment"]
        checks.append(
            check(
                "environment_match",
                "exact",
                "environment",
                {"verified_same": "pass", "different": "fail"}.get(env_status or "", "not_run"),
                "every compared environment field observed on both sides and equal; unknown "
                "counts as not verified, never as equal",
                unverified_fields=[u["field"] for u in env["unverified_fields"]],
                differing_fields=[d["field"] for d in env["differences"]],
            )
        )
        thread_states = [r["threads"]["native_thread_status"]["status"] for r in runs]
        checks.append(
            check(
                "native_thread_policy",
                "exact",
                "environment",
                "fail"
                if "multi_threaded" in thread_states
                else (
                    "pass"
                    if all(t == "verified_single_thread" for t in thread_states)
                    else "not_run"
                ),
                "every worker verified one native thread for every library, NumPy's BLAS "
                "included (a BLAS invisible to threadpoolctl, such as Accelerate, is unverified)",
                run_status_counts=_counts(thread_states),
            )
        )
        exact_required = env_status != "different"
        if not ok_runs:
            exact_status = numeric_status = "not_run"
        elif exact_required:
            identical = complete and all(r["byte_identical"] for r in ok_runs)
            exact_status = "pass" if identical else "fail"
            numeric_status = "not_applicable"
        else:
            exact_status = "not_applicable"
            numeric_status = (
                "pass"
                if complete
                and all(r["outcome"] in ("exact_match", "numeric_agreement") for r in ok_runs)
                else "fail"
            )
        checks.append(
            check(
                "exact_replay",
                "exact",
                "repeatability",
                exact_status,
                "canonical result bytes identical in every requested run (required unless the "
                "environments were observed to differ"
                + ("; environment not fully verified)" if env_status == "unverified" else ")"),
                runs_requested=n_requested,
                runs_completed=len(ok_runs),
            )
        )
        checks.append(
            check(
                "numerical_agreement",
                "numeric",
                "cross_environment",
                numeric_status,
                f"per-operation criteria {CRITERIA_VERSION} (observed different environment only)",
            )
        )
        if report["reference"]["operation"] == "cluster" and ok_runs:
            same = [r["comparison"].get("partition", {}).get("same_partition") for r in ok_runs]
            all_same = all(s is True for s in same)
            checks.append(
                check(
                    "partition_identity",
                    "semantic",
                    "cross_environment" if env_status == "different" else "repeatability",
                    "pass" if complete and all_same else "fail",
                    "membership sets of source rows equal in every requested run; label "
                    "numbering ignored; no global-optimum claim",
                    runs_requested=n_requested,
                    runs_compared=len(ok_runs),
                    equal_in_compared_runs=all_same,
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

    Both passports are validated (raises :class:`~axiom.passport.PassportError`) and
    integrity-checked first. No local resource policy applies here, since nothing is executed;
    validation, integrity checks and comparison are linear in the size of the two documents and
    never sized by a count the documents state. The inputs (raw bytes, typed dataset, plan) must
    be identical, otherwise the comparison is an integrity mismatch and the results are not
    compared. Byte identity is required unless the two environments were observed to differ.
    """
    validate_passport(reference)
    validate_passport(observed)
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
    env = compare_environments(reference["environment"], observed["environment"])
    report["environment"] = env
    report["environment_differences"] = [dict(d, comparable=True) for d in env["differences"]] + [
        dict(d, comparable=False) for d in env["unverified_fields"]
    ]
    report["comparison_mode"] = _COMPARISON_MODES[env["status"]]
    report["native_threads"] = {
        "reference": native_thread_status(reference["environment"]),
        "observed": native_thread_status(observed["environment"]),
    }
    comparison = compare_cores(reference["result"], observed["result"])
    report["comparison"] = comparison
    report["byte_identical"] = comparison["byte_identical"]
    if comparison["byte_identical"]:
        report["verdict"] = "exact_match"
    elif env["status"] != "different" or comparison["verdict"] != "within_criteria":
        report["verdict"] = "numerical_discrepancy"
    else:
        report["verdict"] = "numeric_agreement"
    return report
