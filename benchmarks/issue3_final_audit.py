"""EXP-007: confirmation for the final review of PR #7 and the final audit (Issue #3).

Run from the repository root on a clean checkout of the commit that contains the pre-registration:

    uv run python benchmarks/issue3_final_audit.py --runs 3 > results.json

Part S: fresh-process replay smoke of the ten ``issue3_common`` scenarios (all four operation
families) on the final code, with each run's own evidence.
Part B: bounded work for claimed cluster counts: traced peak allocation of ``integrity_checks``,
``compare_passports`` (forgery on either side) and ``replay`` for re-sealed passports claiming
``k`` up to 2**63 - 1, the number of worker processes started, and whether any integrity work ran
before the trusted policy rejected a passport.
Part P: memory amplification of parsing passport text (``loads_passport``) for worst-case JSON
shapes at two sizes, a measurement with a linearity criterion only.

Writes JSON to stdout; no absolute paths. Timings include interpreter start-up and imports.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from issue3_common import EXAMPLE_PROFILE, EXAMPLES, example_plan, git_state, scenarios  # noqa: E402

from _passport_helpers import reseal  # noqa: E402
from axiom import replay as replay_module  # noqa: E402
from axiom.environment import capture_environment  # noqa: E402
from axiom.passport import dumps_passport, integrity_checks, loads_passport  # noqa: E402
from axiom.replay import VerifyError, analyze, compare_passports, replay  # noqa: E402

EXAMPLE = EXAMPLES / "synthetic_regions.csv"
CLAIMED_K = (10**3, 10**6, 2**62, 2**63 - 1)
MIB = 1024 * 1024


def _per_run(r: dict) -> dict:
    threads = r["threads"]
    return {
        "run_id": r["run_id"],
        "outcome": r["outcome"],
        "worker_status": r["worker_status"],
        "environment_status": r["environment_status"],
        "result_sha256": r.get("result_sha256"),
        "thread_environment_variables": threads.get("environment_variables"),
        "native_thread_status": threads["native_thread_status"]["status"],
        "parent_wall_seconds": r["timing_seconds"].get("parent_wall"),
    }


def smoke(workdir: Path, runs: int) -> list[dict]:
    out = []
    for sc in scenarios(workdir):
        passport = analyze(sc["data"], sc["profile"], sc["plan"])
        report = replay(passport, sc["data"], runs=runs)
        out.append(
            {
                "scenario": sc["id"],
                "operation": sc["plan"].operation,
                "reference_run_id": passport["execution"]["run_id"],
                "reference_result_sha256": passport["identity"]["result_sha256"],
                "verdict": report["verdict"],
                "comparison_mode": report["comparison_mode"],
                "environment_status": report["environment"]["status"],
                "runs_requested": report["runs_requested"],
                "runs_completed": report["runs_completed"],
                "checks": {c["id"]: c["status"] for c in report["checks"]},
                "per_run": [_per_run(r) for r in report["runs"]],
            }
        )
        print(f"{sc['id']}: {report['verdict']}", file=sys.stderr)
    return out


def _peak(fn, *args, **kwargs):
    tracemalloc.start()
    try:
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded
            out = exc
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _claim_k(passport: dict, k: int, *, plan: bool, max_k: bool) -> dict:
    p = copy.deepcopy(passport)
    p["result"]["result"]["k"] = k
    if plan:
        p["inputs"]["plan"]["k"] = k
        p["result"]["plan"]["k"] = k
    if max_k:
        p["inputs"]["limits"]["max_k"] = k
    return reseal(p, plan_too=plan)


def _describe(out) -> dict:
    if isinstance(out, Exception):
        return {"raised": type(out).__name__, "code": getattr(out, "code", None)}
    if isinstance(out, list):  # integrity checks
        return {"failed": sorted(c["id"] for c in out if c["status"] == "fail")}
    return {"verdict": out.get("verdict"), "runs": len(out.get("runs", []))}


def bounded_claims() -> dict:
    cluster = analyze(EXAMPLE, EXAMPLE_PROFILE, example_plan("cluster"))
    started: list[int] = []
    integrity_calls: list[int] = []
    real_worker, real_integrity = replay_module.run_worker, replay_module.integrity_checks

    def counting_worker(request, **kwargs):
        started.append(1)
        return real_worker(request, **kwargs)

    def counting_integrity(passport):
        integrity_calls.append(1)
        return real_integrity(passport)

    _, baseline = _peak(integrity_checks, cluster)
    rows = []
    replay_module.run_worker = counting_worker
    replay_module.integrity_checks = counting_integrity
    try:
        for k in CLAIMED_K:
            for plan, max_k in ((False, False), (True, False), (True, True)):
                forged = _claim_k(cluster, k, plan=plan, max_k=max_k)
                row = {"k": str(k), "plan_claims_k": plan, "max_k_claims_k": max_k}
                out, row["integrity_peak_bytes"] = _peak(integrity_checks, forged)
                row["integrity"] = _describe(out)
                out, row["compare_reference_peak_bytes"] = _peak(compare_passports, forged, cluster)
                row["compare_reference"] = _describe(out)
                out, row["compare_observed_peak_bytes"] = _peak(compare_passports, cluster, forged)
                row["compare_observed"] = _describe(out)
                before_w, before_i = len(started), len(integrity_calls)
                out, row["replay_peak_bytes"] = _peak(replay, forged, EXAMPLE)
                row["replay"] = _describe(out)
                row["replay_workers_started"] = len(started) - before_w
                row["replay_integrity_calls"] = len(integrity_calls) - before_i
                rows.append(row)
    finally:
        replay_module.run_worker = real_worker
        replay_module.integrity_checks = real_integrity
    return {"baseline_integrity_peak_bytes": baseline, "rows": rows}


def _pad_shapes() -> dict:
    return {
        "empty_lists": ("[", "[],", "[]]"),
        "empty_objects": ("[", "{},", "{}]"),
        "small_ints": ("[", "0,", "0]"),
        "seven_digit_ints": ("[", "1234567,", "1234567]"),
        "floats": ("[", "1.5,", "1.5]"),
        "one_string": ('"', "a", 'a"'),
    }


def parse_amplification(sizes_mib: tuple[int, ...]) -> dict:
    base = analyze(EXAMPLE, EXAMPLE_PROFILE, example_plan("inspect"))
    marker = '"__PAD__"'
    template = copy.deepcopy(base)
    template["environment"]["numerical_backend"] = {"pad": "__PAD__"}
    text = dumps_passport(template)
    assert text.count(marker) == 1
    rows = []
    for shape, (head, unit, tail) in _pad_shapes().items():
        for size in sizes_mib:
            target = size * MIB
            n = max(0, (target - len(text) - len(head) - len(tail)) // len(unit))
            doc = text.replace(marker, head + unit * n + tail).encode("utf-8")
            t0 = time.perf_counter()
            out, peak = _peak(loads_passport, doc)
            wall = time.perf_counter() - t0
            rows.append(
                {
                    "shape": shape,
                    "target_mib": size,
                    "text_bytes": len(doc),
                    "parsed": not isinstance(out, Exception),
                    "error": None if not isinstance(out, Exception) else getattr(out, "code", None),
                    "peak_traced_bytes": peak,
                    "peak_per_text_byte": peak / len(doc),
                    "wall_seconds_traced": wall,
                }
            )
            print(f"P {shape} {size} MiB: {peak / len(doc):.1f}x", file=sys.stderr)
    # A large legitimate passport for reference (synthetic 5000-row cluster scenario).
    reference = []
    with tempfile.TemporaryDirectory(prefix="axiom-exp007-") as tmp:
        for sc in scenarios(Path(tmp)):
            if sc["id"] in ("synthetic5000/cluster", "synthetic5000/compare"):
                doc = dumps_passport(analyze(sc["data"], sc["profile"], sc["plan"])).encode()
                _, peak = _peak(loads_passport, doc)
                reference.append(
                    {
                        "scenario": sc["id"],
                        "text_bytes": len(doc),
                        "peak_traced_bytes": peak,
                        "peak_per_text_byte": peak / len(doc),
                    }
                )
    return {"adversarial": rows, "legitimate": reference}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="axiom-exp007-") as tmp:
        report = {
            "experiment": "EXP-007",
            "git": git_state(),
            "parent_environment": capture_environment(),
            "runs_per_scenario": args.runs,
            "smoke": smoke(Path(tmp), args.runs),
        }
    report["bounded_claims"] = bounded_claims()
    report["parse_amplification"] = parse_amplification((4, 8))
    report["total_wall_seconds"] = time.time() - started
    json.dump(report, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except VerifyError as exc:  # pragma: no cover - an unexpected rejection is a result too
        print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        raise
