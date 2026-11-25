"""Measure K-means wall time against the workload upper bound used by the cap (ADR 0006 amendment).

Run from the repository root; JSON on stdout:
  uv run python benchmarks/issue2_cluster_workload.py > results.json
Single runs on one machine; indicative only, not a performance claim.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from axiom import ColumnSpec, ComputationError, CsvProfile, load_csv_bytes, parse_plan_json, run  # noqa: E402
from axiom.plan import cluster_work  # noqa: E402

CONFIGS = [
    # (rows, features, k, n_init, max_iter, data)
    (10_000, 4, 8, 10, 300, "blobs"),
    (100_000, 4, 8, 10, 300, "blobs"),
    (100_000, 8, 32, 3, 300, "uniform"),
    (200_000, 8, 16, 3, 1000, "uniform"),
    (400_000, 4, 8, 5, 300, "uniform"),
]


def make_dataset(n: int, p: int, kind: str, seed: int = 11):
    rng = np.random.default_rng(seed)
    if kind == "blobs":
        centers = rng.normal(0, 6, size=(8, p))
        X = centers[rng.integers(0, 8, n)] + rng.normal(0, 1, size=(n, p))
    else:
        X = rng.random((n, p))
    names = [f"f{i}" for i in range(p)]
    csv = (
        ",".join(names) + "\n" + "\n".join(",".join(f"{v:.6f}" for v in row) for row in X) + "\n"
    ).encode()
    profile = CsvProfile(columns=tuple(ColumnSpec(name=m, kind="numeric") for m in names))
    return load_csv_bytes(csv, profile), names


def main() -> None:
    rows = []
    for n, p, k, n_init, max_iter, kind in CONFIGS:
        ds, names = make_dataset(n, p, kind)
        plan = parse_plan_json(
            json.dumps(
                {
                    "schema_version": "axiom.plan/1",
                    "operation": "cluster",
                    "features": names,
                    "k": k,
                    "n_init": n_init,
                    "max_iter": max_iter,
                }
            )
        )
        bound = cluster_work(plan, n)
        t0 = time.perf_counter()
        entry = {
            "rows": n,
            "features": p,
            "k": k,
            "n_init": n_init,
            "max_iter": max_iter,
            "data": kind,
            "work_upper_bound": bound,
        }
        try:
            res = run(ds, plan).core["result"]
            entry["outcome"] = "result"
            entry["n_iter_diagnostic"] = res["solver"]["n_iter_diagnostic"]
        except ComputationError as exc:
            entry["outcome"] = exc.code
        entry["seconds"] = round(time.perf_counter() - t0, 3)
        entry["upper_bound_work_per_second"] = round(bound / entry["seconds"])
        rows.append(entry)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()
    )
    json.dump(
        {
            "commit": commit,
            "working_tree_dirty": dirty,
            "python": sys.version.split()[0],
            "runs": rows,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
