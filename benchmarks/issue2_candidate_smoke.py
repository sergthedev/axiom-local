"""Candidate-kernel smoke comparison against exact oracles (records why kernels were chosen).

Run from the repository root; output is JSON on stdout:
  uv run python benchmarks/issue2_candidate_smoke.py > candidate_smoke.json
Compares, on the same inputs: stdlib vs SciPy vs NumPy kernels, against exact rational oracles.
Not a performance benchmark and not an accuracy guarantee.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import sys
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import scipy
import scipy.stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from axiom.kernels import correlate  # noqa: E402
from oracles import (  # noqa: E402
    exact_mean,
    exact_pearson_decimal,
    exact_sample_sd_decimal,
    exact_spearman_decimal,
    frac_to_decimal,
    load_nist,
)


def err(value: float, ref: Decimal) -> float:
    return float(abs(Decimal(value) - ref))


def descriptive_candidates() -> list[dict]:
    manifest, tokens = load_nist()
    rows = []
    for name in sorted(manifest["datasets"]):
        vals = [float(t) for t in tokens[name]]
        mean_ref, sd_ref = frac_to_decimal(exact_mean(vals)), exact_sample_sd_decimal(vals)
        arr = np.array(vals)
        rows.append(
            {
                "dataset": name,
                "abs_error_mean": {
                    "statistics.mean": err(statistics.mean(vals), mean_ref),
                    "numpy.mean": err(float(arr.mean()), mean_ref),
                    "naive_sum_over_n": err(sum(vals) / len(vals), mean_ref),
                },
                "abs_error_sd": {
                    "statistics.stdev": err(statistics.stdev(vals), sd_ref),
                    "numpy.std_ddof1": err(float(arr.std(ddof=1)), sd_ref),
                    "naive_one_pass": err(
                        math.sqrt(
                            max(
                                0.0,
                                (sum(v * v for v in vals) - sum(vals) ** 2 / len(vals))
                                / (len(vals) - 1),
                            )
                        ),
                        sd_ref,
                    ),
                },
            }
        )
    return rows


def correlation_candidates() -> dict:
    rng = random.Random(20261125)
    worst = {
        k: 0.0
        for k in (
            "axiom_guarded_pearson",
            "stdlib_raw_pearson",
            "scipy_pearsonr",
            "axiom_spearman",
            "scipy_spearmanr",
        )
    }
    for _ in range(300):
        n = rng.randint(3, 60)
        sx, sy = 10.0 ** rng.randint(-6, 6), 10.0 ** rng.randint(-6, 6)
        ox = rng.choice([0.0, 1.0, 1e3, 1e6]) * sx
        oy = rng.choice([0.0, 1.0, 1e3, 1e6]) * sy
        rho = rng.uniform(-1, 1)
        x = [rng.gauss(0, 1) for _ in range(n)]
        y = [rho * a + math.sqrt(1 - rho * rho) * rng.gauss(0, 1) for a in x]
        x, y = [ox + sx * v for v in x], [oy + sy * v for v in y]
        rp, rs = exact_pearson_decimal(x, y), exact_spearman_decimal(x, y)
        worst["axiom_guarded_pearson"] = max(
            worst["axiom_guarded_pearson"], err(correlate(x, y, "pearson")[0], rp)
        )
        worst["stdlib_raw_pearson"] = max(
            worst["stdlib_raw_pearson"], err(statistics.correlation(x, y), rp)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            worst["scipy_pearsonr"] = max(
                worst["scipy_pearsonr"], err(float(st.pearsonr(x, y).statistic), rp)
            )
            worst["scipy_spearmanr"] = max(
                worst["scipy_spearmanr"], err(float(st.spearmanr(x, y).statistic), rs)
            )
        worst["axiom_spearman"] = max(
            worst["axiom_spearman"], err(correlate(x, y, "spearman")[0], rs)
        )
    extreme = []
    y = [1.0, 2.0, 3.5]
    for scale in (1e-200, 1e200):
        x = [scale, 2 * scale, 4 * scale]
        ref = exact_pearson_decimal(x, y)
        row = {"scale": scale, "exact": float(ref)}
        try:
            row["stdlib_raw"] = statistics.correlation(x, y)
        except statistics.StatisticsError as exc:
            row["stdlib_raw"] = f"StatisticsError: {exc}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["scipy_pearsonr"] = float(st.pearsonr(x, y).statistic)
        row["axiom_guarded"] = correlate(x, y, "pearson")[0]
        extreme.append(row)
    return {
        "n_pairs": 300,
        "seed": 20261125,
        "worst_abs_error": worst,
        "extreme_scale_cases": extreme,
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
    json.dump(
        {
            "commit": commit,
            "working_tree_dirty": dirty,
            "versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "python": sys.version.split()[0],
            },
            "descriptive_candidates": descriptive_candidates(),
            "correlation_candidates": correlation_candidates(),
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
