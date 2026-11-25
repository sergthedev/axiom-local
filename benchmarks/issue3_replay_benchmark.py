"""EXP-004: fresh-process replay, mutation and malformed-input checks (Issue #3).

Run from the repository root:

    uv run python benchmarks/issue3_replay_benchmark.py --runs 20 > results.json

For each of the ten scenarios in ``issue3_common``: one fresh-process analysis creates the reference
passport, then ``--runs`` replays each start a new interpreter that re-reads the data file and
recomputes (no result cache). Every run's outcome, result hash and timings are recorded; nothing is
summarized away. Then the pre-registered mutation and malformed-input checks are run. The criteria
are in docs/evidence/experiments/EXP-004-replay-benchmark/PREREGISTRATION.md.

Writes JSON to stdout; no absolute paths. Timings include interpreter start-up and imports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from issue3_common import (  # noqa: E402
    EXAMPLE_PROFILE,
    EXAMPLES,
    example_plan,
    file_sha256,
    git_state,
    scenarios,
)

from axiom import load_csv, run  # noqa: E402
from axiom.environment import capture_environment  # noqa: E402
from axiom.partition import cluster_partition  # noqa: E402
from axiom.replay import AnalysisFailed, analyze, replay  # noqa: E402
from oracles import exact_pearson_decimal  # noqa: E402

EXAMPLE = EXAMPLES / "synthetic_regions.csv"


def replay_scenarios(workdir: Path, runs: int) -> list[dict]:
    out = []
    for sc in scenarios(workdir):
        t0 = time.perf_counter()
        passport = analyze(sc["data"], sc["profile"], sc["plan"])
        analyze_wall = time.perf_counter() - t0
        in_process = run(load_csv(sc["data"], sc["profile"]), sc["plan"]).result_sha256
        report = replay(passport, sc["data"], runs=runs)
        out.append(
            {
                "scenario": sc["id"],
                "operation": sc["plan"].operation,
                "data_sha256": file_sha256(sc["data"]),
                "reference_result_sha256": passport["identity"]["result_sha256"],
                "in_process_result_sha256_equal": in_process
                == passport["identity"]["result_sha256"],
                "analyze_parent_wall_seconds": analyze_wall,
                "verdict": report["verdict"],
                "comparison_mode": report["comparison_mode"],
                "environment_differences": report["environment_differences"],
                "runs_requested": report["runs_requested"],
                "runs_completed": report["runs_completed"],
                "summary": report["summary"],
                "checks": {c["id"]: c["status"] for c in report["checks"]},
                "per_run": [
                    {
                        "outcome": r["outcome"],
                        "result_sha256": r.get("result_sha256"),
                        "parent_wall_seconds": r["timing_seconds"].get("parent_wall"),
                        "compute_seconds": r["timing_seconds"].get("compute_seconds"),
                        "load_seconds": r["timing_seconds"].get("load_seconds"),
                        "n_iter_diagnostic": (
                            None
                            if r.get("comparison") is None or "partition" not in r["comparison"]
                            else r["comparison"]["partition"]["n_iter_observed"]
                        ),
                    }
                    for r in report["runs"]
                ],
                "environment_threads": passport["environment"]["threads"],
            }
        )
        print(
            f"{sc['id']}: {report['verdict']} {report['summary']['outcome_counts']}",
            file=sys.stderr,
        )
    return out


def _write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


def _stats(passport: dict, column: str) -> dict:
    return next(c for c in passport["result"]["result"]["columns"] if c["name"] == column)["stats"]


def mutations(workdir: Path) -> list[dict]:
    """Pre-registered mutations: stale binding rejected; rebinding gives the expected change."""
    text = EXAMPLE.read_text("utf-8")
    records: list[dict] = []

    def old_replay(name: str, mutated: Path) -> dict:
        ref = analyze(EXAMPLE, EXAMPLE_PROFILE, example_plan(name))
        rep = replay(ref, mutated)
        return ref, {
            "verdict": rep["verdict"],
            "mismatch": rep.get("mismatch"),
            "engine_error_code": (rep["runs"][0].get("error") or {}).get("code"),
            "dataset": rep["runs"][0]["dataset"],
        }

    # M1 Inspect: one gdp_per_capita value raised by 70000 (R001).
    m1 = _write(workdir / "m1.csv", text.replace("R001,Charlie,23400.39", "R001,Charlie,93400.39"))
    ref, stale = old_replay("inspect", m1)
    new = analyze(m1, EXAMPLE_PROFILE, example_plan("inspect"))
    rows = list(csv.DictReader(io.StringIO(m1.read_text("utf-8"), newline="")))
    values = [float(r["gdp_per_capita"]) for r in rows if r["gdp_per_capita"] != ""]
    exact = sum(map(Fraction, values), Fraction(0)) / len(values)
    mean = _stats(new, "gdp_per_capita")["mean"]
    others_unchanged = all(
        _stats(new, c) == _stats(ref, c)
        for c in ("unemployment_rate", "education_years", "household_income")
    )
    records.append(
        {
            "id": "M1-inspect-value",
            "stale_binding": stale,
            "expected": "mean = exact rational mean of the parsed values within 1 ulp; "
            "max = 93400.39; other columns unchanged",
            "mean_error_ulp": float(abs(Fraction(mean) - exact) / Fraction(math.ulp(mean))),
            "max": _stats(new, "gdp_per_capita")["max"],
            "other_columns_unchanged": others_unchanged,
            "rebound_replay_verdict": replay(new, m1)["verdict"],
        }
    )

    # M2 Compare: the first Alpha row moved to Bravo.
    line = next(ln for ln in text.splitlines() if ",Alpha," in ln)
    m2 = _write(workdir / "m2.csv", text.replace(line, line.replace(",Alpha,", ",Bravo,")))
    ref, stale = old_replay("compare", m2)
    new = analyze(m2, EXAMPLE_PROFILE, example_plan("compare"))
    old_n = {g["key"]: g["n_total"] for g in ref["result"]["result"]["groups"]}
    new_n = {g["key"]: g["n_total"] for g in new["result"]["result"]["groups"]}
    records.append(
        {
            "id": "M2-compare-group",
            "stale_binding": stale,
            "expected": "Alpha n_total - 1, Bravo n_total + 1, other groups unchanged",
            "old_counts": old_n,
            "new_counts": new_n,
            "rebound_replay_verdict": replay(new, m2)["verdict"],
        }
    )

    # M3 Relate (Pearson): one education_years value changed; oracle = exact rational Pearson.
    m3 = _write(
        workdir / "m3.csv",
        text.replace("R003,Bravo,14116.92,6.84,10.05", "R003,Bravo,14116.92,6.84,19.05"),
    )
    ref, stale = old_replay("relate_pearson", m3)
    new = analyze(m3, EXAMPLE_PROFILE, example_plan("relate_pearson"))
    rows = list(csv.DictReader(io.StringIO(m3.read_text("utf-8"), newline="")))
    pairs = [
        (float(r["education_years"]), float(r["gdp_per_capita"]))
        for r in rows
        if r["education_years"] != "" and r["gdp_per_capita"] != ""
    ]
    oracle = exact_pearson_decimal([a for a, _ in pairs], [b for _, b in pairs])
    coefficient = new["result"]["result"]["coefficient"]
    records.append(
        {
            "id": "M3-relate-value",
            "stale_binding": stale,
            "expected": "|r - exact rational Pearson| <= 1e-12 (ADR 0004 criterion); r changes",
            "abs_error_vs_oracle": float(abs(oracle - Decimal(coefficient))),
            "coefficient_before": ref["result"]["result"]["coefficient"],
            "coefficient_after": coefficient,
            "rebound_replay_verdict": replay(new, m3)["verdict"],
        }
    )

    # M4 Cluster: change a column that is not a cluster feature (household_income of R001).
    m4 = _write(
        workdir / "m4.csv",
        text.replace(
            "R001,Charlie,23400.39,6.54,10.81,13551.96", "R001,Charlie,23400.39,6.54,10.81,99999.99"
        ),
    )
    ref, stale = old_replay("cluster", m4)
    new = analyze(m4, EXAMPLE_PROFILE, example_plan("cluster"))
    records.append(
        {
            "id": "M4-cluster-nonfeature",
            "stale_binding": stale,
            "expected": "dataset identities change; result block byte-identical; same partition",
            "dataset_sha256_changed": new["identity"]["dataset_sha256"]
            != ref["identity"]["dataset_sha256"],
            "result_block_identical": new["result"]["result"] == ref["result"]["result"],
            "same_partition": cluster_partition(new["result"]["result"])
            == cluster_partition(ref["result"]["result"]),
            "rebound_replay_verdict": replay(new, m4)["verdict"],
        }
    )
    return records


def malformed(workdir: Path) -> dict:
    """Short record appended: analysis fails structurally; the old passport does not replay."""
    bad = _write(workdir / "malformed.csv", EXAMPLE.read_text("utf-8") + "R999,Alpha,1\n")
    try:
        analyze(bad, EXAMPLE_PROFILE, example_plan("inspect"))
        analysis = {"failed": False}
    except AnalysisFailed as exc:
        analysis = {"failed": True, "code": exc.response["error"]["code"]}
    ref = analyze(EXAMPLE, EXAMPLE_PROFILE, example_plan("inspect"))
    rep = replay(ref, bad)
    return {
        "analysis": analysis,
        "old_passport_replay": {
            "verdict": rep["verdict"],
            "mismatch": rep.get("mismatch"),
            "error_code": (rep["runs"][0].get("error") or {}).get("code"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="axiom-exp004-") as tmp:
        workdir = Path(tmp)
        report = {
            "experiment": "EXP-004",
            "git": git_state(),
            "parent_environment": capture_environment(),
            "runs_per_scenario": args.runs,
            "scenarios": replay_scenarios(workdir, args.runs),
            "mutations": mutations(workdir),
            "malformed_input": malformed(workdir),
        }
    report["total_wall_seconds"] = time.time() - started
    json.dump(report, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
