"""Measure the Issue #2 engine against independent oracles and record the raw numbers.

Run from the repository root:  uv run python benchmarks/issue2_evidence.py > results.json

Writes a JSON report to stdout (no absolute paths). Every number is measured by this script on the
machine that runs it. Nothing here is a certification; see docs/evidence/experiments/.
"""

from __future__ import annotations

import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from axiom import (  # noqa: E402
    ColumnSpec,
    CsvProfile,
    environment_info,
    load_csv_bytes,
    parse_plan_json,
    run,
)
from axiom.kernels import correlate, describe  # noqa: E402
from oracles import (  # noqa: E402
    exact_mean,
    exact_pearson_decimal,
    exact_quantile_type7,
    exact_sample_sd_decimal,
    exact_sample_variance,
    exact_spearman_decimal,
    frac_to_decimal,
    load_nist,
    ulp_error,
)

U = 2.0**-53
V = '"schema_version":"axiom.plan/1"'


def profile(spec: dict[str, str]) -> CsvProfile:
    return CsvProfile(columns=tuple(ColumnSpec(name=n, kind=k) for n, k in spec.items()))


def nist() -> list[dict]:
    manifest, tokens = load_nist()
    rows = []
    for name in sorted(manifest["datasets"]):
        meta = manifest["datasets"][name]
        csv = ("y\n" + "\n".join(tokens[name]) + "\n").encode()
        ds = load_csv_bytes(csv, profile({"y": "numeric"}))
        plan = parse_plan_json("{" + V + ',"operation":"inspect","columns":["y"]}')
        stats = run(ds, plan).core["result"]["columns"][0]["stats"]
        vals = list(ds.columns["y"])
        n, m = len(vals), max(abs(v) for v in vals)
        c_mean, c_sd = Decimal(meta["certified_mean"]), Decimal(meta["certified_sd_n_minus_1"])
        oracle_mean = frac_to_decimal(exact_mean(vals))
        oracle_sd = exact_sample_sd_decimal(vals)
        rows.append(
            {
                "dataset": name,
                "n": n,
                "difficulty": meta["difficulty"],
                "mean": stats["mean"],
                "sample_sd": stats["sample_sd"],
                "certified_mean": meta["certified_mean"],
                "certified_sd": meta["certified_sd_n_minus_1"],
                "abs_error_vs_certified_mean": float(abs(Decimal(stats["mean"]) - c_mean)),
                "abs_error_vs_certified_sd": float(abs(Decimal(stats["sample_sd"]) - c_sd)),
                "input_conversion_bound_mean": U * m,
                "input_conversion_bound_sd": U * m * math.sqrt(n / (n - 1)),
                "ulp_error_vs_binary64_oracle_mean": ulp_error(stats["mean"], oracle_mean),
                "ulp_error_vs_binary64_oracle_sd": ulp_error(stats["sample_sd"], oracle_sd),
                "binary64_oracle_vs_certified_sd_abs": float(abs(oracle_sd - c_sd)),
                "binary64_oracle_vs_certified_mean_abs": float(abs(oracle_mean - c_mean)),
            }
        )
    return rows


def inspect_random(n_sets: int = 300) -> dict:
    rng = random.Random(20261125)
    worst = {
        "mean_ulp": 0.0,
        "variance_ulp": 0.0,
        "sd_ulp": 0.0,
        "quantile_ulp_of_result": 0.0,
        "quantile_ulp_of_operand_scale": 0.0,
    }
    for _ in range(n_sets):
        n = rng.randint(2, 60)
        scale = 10.0 ** rng.randint(-8, 8)
        offset = rng.choice([0.0, 1.0, 1e3, 1e6]) * scale
        values = [offset + rng.gauss(0, 1) * scale for _ in range(n)]
        stats, _ = describe(values)
        worst["mean_ulp"] = max(
            worst["mean_ulp"], ulp_error(stats["mean"], frac_to_decimal(exact_mean(values)))
        )
        worst["variance_ulp"] = max(
            worst["variance_ulp"],
            ulp_error(stats["sample_variance"], frac_to_decimal(exact_sample_variance(values))),
        )
        worst["sd_ulp"] = max(
            worst["sd_ulp"], ulp_error(stats["sample_sd"], exact_sample_sd_decimal(values))
        )
        s = sorted(values)
        for key, q in (("q1", 0.25), ("q3", 0.75)):
            from fractions import Fraction

            exact_q = frac_to_decimal(exact_quantile_type7(values, Fraction(q)))
            h = (len(s) - 1) * q
            lo, hi = math.floor(h), min(math.floor(h) + 1, len(s) - 1)
            err = abs(Decimal(stats[key]) - exact_q)
            if stats[key] != 0:
                worst["quantile_ulp_of_result"] = max(
                    worst["quantile_ulp_of_result"], float(err / Decimal(math.ulp(stats[key])))
                )
            sc = max(abs(s[lo]), abs(s[hi]), abs(stats[key]))
            worst["quantile_ulp_of_operand_scale"] = max(
                worst["quantile_ulp_of_operand_scale"], float(err / Decimal(math.ulp(sc)))
            )
    return {"n_datasets": n_sets, "seed": 20261125, "worst": worst}


def relate_random(n_sets: int = 300) -> dict:
    rng = random.Random(20261125)
    worst_p = worst_s = 0.0
    counts = {"pearson_defined": 0, "spearman_defined": 0, "pearson_undefined": 0}
    for _ in range(n_sets):
        n = rng.randint(3, 60)
        sx, sy = 10.0 ** rng.randint(-6, 6), 10.0 ** rng.randint(-6, 6)
        ox = rng.choice([0.0, 1.0, 1e3, 1e6]) * sx
        oy = rng.choice([0.0, 1.0, 1e3, 1e6]) * sy
        rho = rng.uniform(-1, 1)
        x = [rng.gauss(0, 1) for _ in range(n)]
        y = [rho * a + math.sqrt(1 - rho * rho) * rng.gauss(0, 1) for a in x]
        if rng.random() < 0.3:
            x, y = [round(v, 1) for v in x], [round(v, 1) for v in y]
        x, y = [ox + sx * v for v in x], [oy + sy * v for v in y]
        value, _ = correlate(x, y, "pearson")
        if value is None:
            counts["pearson_undefined"] += 1
        else:
            counts["pearson_defined"] += 1
            worst_p = max(worst_p, float(abs(Decimal(value) - exact_pearson_decimal(x, y))))
        value, _ = correlate(x, y, "spearman")
        if value is not None:
            counts["spearman_defined"] += 1
            worst_s = max(worst_s, float(abs(Decimal(value) - exact_spearman_decimal(x, y))))
    return {
        "n_pairs": n_sets,
        "seed": 20261125,
        "counts": counts,
        "worst_abs_error_pearson": worst_p,
        "worst_abs_error_spearman": worst_s,
    }


def near_constant_sweep() -> list[dict]:
    """Pearson error versus relative spread of x, against the exact oracle of the same inputs."""
    rng = random.Random(1)
    rows = []
    for exponent in (-60, -52, -48, -44, -41, -40, -39, -36, -30, -20, -10):
        spread = 2.0**exponent
        x = [1.0 + spread * k for k in (0, 1, 3, 4, 6, 7)]
        y = [rng.gauss(0, 1) + 0.5 * i for i, _ in enumerate(x)]
        value, undefined = correlate(x, y, "pearson")
        entry = {"relative_spread_exponent": exponent, "flagged": undefined is not None}
        if value is not None:
            entry["abs_error_vs_exact_oracle"] = float(
                abs(Decimal(value) - exact_pearson_decimal(x, y))
            )
        else:
            entry["reason"] = undefined[0]
            entry["abs_error_if_forced_stdlib"] = None
            try:
                forced = statistics.correlation(x, y)
                entry["abs_error_if_forced_stdlib"] = float(
                    abs(Decimal(forced) - exact_pearson_decimal(x, y))
                )
            except statistics.StatisticsError as exc:
                entry["stdlib_error"] = str(exc)
        rows.append(entry)
    return rows


def kmeans_thread_experiment() -> dict:
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(7)
    X = rng.normal(size=(60_000, 8))
    X[:20_000] += 3
    X[20_000:40_000] -= 3
    kwargs = {
        "n_clusters": 8,
        "init": "k-means++",
        "n_init": 4,
        "max_iter": 300,
        "tol": 1e-4,
        "algorithm": "lloyd",
        "random_state": 42,
    }

    def fingerprint(limit: int | None) -> set[str]:
        seen = set()
        for _ in range(20):
            if limit is None:
                model = KMeans(**kwargs).fit(X)
            else:
                with threadpool_limits(limits=limit):
                    model = KMeans(**kwargs).fit(X)
            seen.add(
                model.cluster_centers_.tobytes().hex()[:64] + f"|{model.n_iter_}|{model.inertia_!r}"
            )
        return seen

    return {
        "n_points": int(X.shape[0]),
        "runs_each": 20,
        "distinct_outputs_default_threads": len(fingerprint(None)),
        "distinct_outputs_single_thread": len(fingerprint(1)),
        "logical_cpus": __import__("os").cpu_count(),
    }


def timings() -> list[dict]:
    rng = random.Random(3)
    out = []
    for n in (10_000, 100_000, 1_000_000):
        rows = "".join(f"{rng.gauss(0, 1):.6f},{rng.gauss(0, 1):.6f}\n" for _ in range(n))
        data = ("x,y\n" + rows).encode()
        t0 = time.perf_counter()
        ds = load_csv_bytes(data, profile({"x": "numeric", "y": "numeric"}))
        t_load = time.perf_counter() - t0
        entry = {"n_rows": n, "bytes": len(data), "load_s": round(t_load, 3)}
        for label, body in (
            ("inspect_s", '"operation":"inspect","columns":["x","y"]'),
            ("pearson_s", '"operation":"relate","x":"x","y":"y","method":"pearson"'),
            ("spearman_s", '"operation":"relate","x":"x","y":"y","method":"spearman"'),
        ):
            plan = parse_plan_json("{" + V + "," + body + "}")
            t0 = time.perf_counter()
            run(ds, plan)
            entry[label] = round(time.perf_counter() - t0, 3)
        if n <= 100_000:
            plan = parse_plan_json("{" + V + ',"operation":"cluster","features":["x","y"],"k":4}')
            t0 = time.perf_counter()
            run(ds, plan)
            entry["cluster_s"] = round(time.perf_counter() - t0, 3)
        out.append(entry)
    return out


def fresh_process_replay(runs: int = 20) -> dict:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, 'examples')\n"
        "from run_example import PROFILE\n"
        "from axiom import load_csv, parse_plan_json, run\n"
        "ds = load_csv(Path('examples/synthetic_regions.csv'), PROFILE)\n"
        "for p in sorted(Path('examples/plans').glob('*.json')):\n"
        "    print(p.stem, run(ds, parse_plan_json(p.read_text('utf-8'))).result_sha256)\n"
    )
    outputs = []
    for _ in range(runs):
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=ROOT
        )
        outputs.append(proc.stdout)
    return {
        "fresh_processes": runs,
        "plans_per_process": len(outputs[0].strip().splitlines()),
        "distinct_outputs": len(set(outputs)),
        "result_hashes": dict(line.split() for line in outputs[0].strip().splitlines()),
    }


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()
    )
    report = {
        "commit": commit,
        "working_tree_dirty": dirty,
        "environment": environment_info(),
        "hardware": {"machine": platform.machine(), "processor": platform.processor()},
        "nist": nist(),
        "inspect_random": inspect_random(),
        "relate_random": relate_random(),
        "pearson_near_constant_sweep": near_constant_sweep(),
        "kmeans_threads": kmeans_thread_experiment(),
        "timings_single_run": timings(),
        "fresh_process_replay": fresh_process_replay(),
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
