"""EXP-006: confirmation of the PR #7 review corrections in fresh processes (Issue #3).

Run from the repository root on a clean checkout of the commit that contains the pre-registration:

    uv run python benchmarks/issue3_confirmation.py --runs 20 > results.json

1. Replay: for each of the ten scenarios of ``issue3_common``, one fresh-process analysis creates
   the reference passport and ``--runs`` replays recompute it. Unlike EXP-004, each run's own
   evidence is stored: run id, environment status, the worker's thread observations and outcome.
2. Mutation and malformed-input checks: the EXP-004 functions, unchanged, on the revised code.
3. Adversarial checks with real worker processes (A1-A3, see the pre-registration).

Writes JSON to stdout; no absolute paths. Timings include interpreter start-up and imports.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from issue3_common import EXAMPLE_PROFILE, EXAMPLES, example_plan, file_sha256, git_state, scenarios  # noqa: E402
from issue3_replay_benchmark import malformed, mutations  # noqa: E402  (EXP-004 code, unchanged)

from _passport_helpers import reseal  # noqa: E402
from axiom import load_csv, run  # noqa: E402
from axiom import replay as replay_module  # noqa: E402
from axiom.environment import capture_environment, unavailable  # noqa: E402
from axiom.limits import DEFAULT_LIMITS  # noqa: E402
from axiom.replay import VerifyError, analyze, replay  # noqa: E402

EXAMPLE = EXAMPLES / "synthetic_regions.csv"


def _per_run(r: dict) -> dict:
    threads = r["threads"]
    return {
        "run_id": r["run_id"],
        "outcome": r["outcome"],
        "worker_status": r["worker_status"],
        "environment_status": r["environment_status"],
        "result_sha256": r.get("result_sha256"),
        "thread_environment_variables": threads.get("environment_variables"),
        "pools": threads.get("pools"),
        "fit_limit_probe": threads.get("fit_limit_probe"),
        "blas_thread_control": threads.get("blas_thread_control"),
        "native_thread_status": threads["native_thread_status"]["status"],
        "parent_wall_seconds": r["timing_seconds"].get("parent_wall"),
        "compute_seconds": r["timing_seconds"].get("compute_seconds"),
        "n_iter_diagnostic": (
            None
            if r.get("comparison") is None or "partition" not in r["comparison"]
            else r["comparison"]["partition"]["n_iter_observed"]
        ),
    }


def replay_scenarios(workdir: Path, runs: int) -> list[dict]:
    out = []
    for sc in scenarios(workdir):
        passport = analyze(sc["data"], sc["profile"], sc["plan"])
        in_process = run(load_csv(sc["data"], sc["profile"]), sc["plan"]).result_sha256
        report = replay(passport, sc["data"], runs=runs)
        out.append(
            {
                "scenario": sc["id"],
                "operation": sc["plan"].operation,
                "data_sha256": file_sha256(sc["data"]),
                "reference_run_id": passport["execution"]["run_id"],
                "reference_result_sha256": passport["identity"]["result_sha256"],
                "in_process_result_sha256_equal": in_process
                == passport["identity"]["result_sha256"],
                "verdict": report["verdict"],
                "comparison_mode": report["comparison_mode"],
                "environment_status": report["environment"]["status"],
                "environment_unverified_fields": [
                    u["field"] for u in report["environment"]["unverified_fields"]
                ],
                "environment_differing_fields": [
                    d["field"] for d in report["environment"]["differences"]
                ],
                "runs_requested": report["runs_requested"],
                "runs_completed": report["runs_completed"],
                "summary": report["summary"],
                "native_threads": report["native_threads"],
                "checks": {c["id"]: c["status"] for c in report["checks"]},
                "per_run": [_per_run(r) for r in report["runs"]],
            }
        )
        print(
            f"{sc['id']}: {report['verdict']} {report['summary']['outcome_counts']}",
            file=sys.stderr,
        )
    return out


def adversarial() -> dict:
    """A1-A3 with real worker processes (pre-registered)."""
    results: dict = {}
    inspect = analyze(EXAMPLE, EXAMPLE_PROFILE, example_plan("inspect"))

    # A1: re-sealed passport raising resource limits: rejected, no worker started.
    forged = copy.deepcopy(inspect)
    forged["inputs"]["limits"]["max_bytes"] = DEFAULT_LIMITS.max_bytes * 1000
    forged = reseal(forged)
    started: list[int] = []
    real = replay_module.run_worker

    def counting(request, **kwargs):
        started.append(1)
        return real(request, **kwargs)

    replay_module.run_worker = counting
    try:
        replay(forged, EXAMPLE)
        a1 = {"rejected": False}
    except VerifyError as exc:
        a1 = {"rejected": True, "code": exc.code, "fields": exc.details.get("fields")}
    finally:
        replay_module.run_worker = real
    a1["workers_started"] = len(started)
    results["A1_oversized_limits"] = a1

    # A2: mean moved by one ulp, plus an unknown (not different) environment field.
    forged = copy.deepcopy(inspect)
    stats = forged["result"]["result"]["columns"][0]["stats"]
    stats["mean"] = math.nextafter(stats["mean"], math.inf)
    forged["environment"]["packages"]["numpy"] = unavailable("removed for EXP-006 A2")
    rep = replay(reseal(forged), EXAMPLE)
    results["A2_one_ulp_unknown_environment"] = {
        "verdict": rep["verdict"],
        "comparison_mode": rep["comparison_mode"],
        "environment_status": rep["environment"]["status"],
        "within_criteria_but_not_accepted": rep["runs"][0]["comparison"]["verdict"],
    }

    # A3: two assignments swapped between clusters (sizes unchanged, digests re-sealed).
    cluster = analyze(EXAMPLE, EXAMPLE_PROFILE, example_plan("cluster"))
    forged = copy.deepcopy(cluster)
    labels = forged["result"]["result"]["assignments"]
    i, j = labels.index(0), labels.index(1)
    labels[i], labels[j] = 1, 0
    forged = reseal(forged)
    rep = replay(forged, EXAMPLE)
    results["A3_consistent_partition_forgery"] = {
        "integrity": {c["id"]: c["status"] for c in rep["integrity"]},
        "verdict": rep["verdict"],
        "same_partition": rep["runs"][0]["comparison"]["partition"]["same_partition"],
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="axiom-exp006-") as tmp:
        workdir = Path(tmp)
        report = {
            "experiment": "EXP-006",
            "git": git_state(),
            "parent_environment": capture_environment(),
            "runs_per_scenario": args.runs,
            "scenarios": replay_scenarios(workdir, args.runs),
            "mutations": mutations(workdir),
            "malformed_input": malformed(workdir),
            "adversarial": adversarial(),
        }
    report["total_wall_seconds"] = time.time() - started
    json.dump(report, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
