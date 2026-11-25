"""EXP-002: confirm exact descriptive statistics on held-back data (frozen in PREREGISTRATION.md).

Run from the repository root; JSON on stdout:
  uv run python benchmarks/issue2_quartile_confirmation.py > results.json
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from axiom.kernels import describe  # noqa: E402
from oracles import (  # noqa: E402
    classify_exact,
    exact_mean,
    exact_median_weighted,
    exact_quantile_type7_weighted,
)

BIG = sys.float_info.max
TINY = 5e-324
REASON = {"overflow_exceeds_float64": "overflow", "underflow_to_zero": "underflow"}


def f1(rng: random.Random) -> list[float]:
    n = rng.randint(2, 60)
    scale = 10.0 ** rng.randint(-8, 8)
    offset = rng.choice([0.0, 1.0, 1e3, 1e6]) * scale
    return [offset + rng.gauss(0, 1) * scale for _ in range(n)]


def f2(rng: random.Random) -> list[float]:
    n = rng.randint(2, 9)
    s = 10.0 ** rng.randint(-200, 200)
    values = [rng.choice([-1.0, 1.0]) * s * (1 + rng.uniform(-1e-3, 1e-3)) for _ in range(n)]
    if all(v > 0 for v in values) or all(v < 0 for v in values):
        values[0] = -values[0]
    return values


def f3(rng: random.Random) -> list[float]:
    n = rng.randint(2, 30)
    return [rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-300, 300) for _ in range(n)]


def f4(rng: random.Random) -> list[float]:
    n = rng.randint(2, 12)
    out = []
    for _ in range(n):
        pick = rng.random()
        if pick < 0.2:
            out.append(rng.choice([-BIG, BIG]))
        elif pick < 0.8:
            out.append(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * BIG)
        elif pick < 0.9:
            out.append(0.0)
        else:
            out.append(rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-10, 10))
    return out


def f5(rng: random.Random) -> list[float]:
    n = rng.randint(2, 20)
    out = []
    for _ in range(n):
        pick = rng.random()
        if pick < 0.75:
            out.append(rng.randint(-50, 50) * TINY)
        elif pick < 0.85:
            out.append(0.0)
        else:
            out.append(rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 5.0) * 2.2250738585072014e-308)
    return out


def f6(rng: random.Random) -> list[float]:
    n = rng.randint(200, 2000)
    scale = 10.0 ** rng.randint(-6, 6)
    palette = [scale * rng.gauss(0, 1) for _ in range(20)]
    return [rng.choice(palette) for _ in range(n)]


FAMILIES = [
    ("F1_gaussian_scales", 7300001, 300, f1),
    ("F2_cancellation", 7300002, 300, f2),
    ("F3_opposite_signs_wide", 7300003, 300, f3),
    ("F4_extreme_magnitude", 7300004, 300, f4),
    ("F5_subnormal", 7300005, 300, f5),
    ("F6_ties_large_n", 7300006, 100, f6),
]


def ulp_ratio(value: float, exact: Fraction, scale: float) -> float | None:
    if scale == 0.0 or not math.isfinite(scale):
        return None
    return float(abs(Fraction(value) - exact) / Fraction(math.ulp(scale)))


def run_family(name: str, seed: int, n_sets: int, gen) -> dict:
    rng = random.Random(seed)
    rec = {
        "family": name,
        "seed": seed,
        "sets": n_sets,
        "production_evaluations": 0,
        "c1_value_mismatches": 0,
        "c1_classification_mismatches": 0,
        "c1_examples": [],
        "c3_oracle_outcomes": {
            k: {"defined": 0, "overflow": 0, "underflow": 0}
            for k in ("q1", "q3", "iqr", "median", "mean")
        },
        "numpy_evaluations": 0,
        "numpy_nonfinite_where_oracle_finite": 0,
        "numpy_value_mismatch_bitwise": 0,
        "numpy_max_result_ulp": 0.0,
        "numpy_max_operand_ulp": 0.0,
        "numpy_result_ulp_gt_2": 0,
        "numpy_operand_ulp_gt_1": 0,
        "numpy_operand_ulp_gt_4": 0,
        "numpy_examples": [],
    }
    for _ in range(n_sets):
        values = gen(rng)
        stats, undefined = describe(values)
        s = sorted(values)
        exact = {
            "q1": exact_quantile_type7_weighted(values, 1),
            "q3": exact_quantile_type7_weighted(values, 3),
            "median": exact_median_weighted(values),
            "mean": exact_mean(values),
        }
        exact["iqr"] = exact["q3"] - exact["q1"]
        for key, frac in exact.items():
            ref_value, ref_class = classify_exact(frac)
            rec["c3_oracle_outcomes"][key][ref_class] += 1
            rec["production_evaluations"] += 1
            if key in undefined:
                got_class = REASON.get(undefined[key], "other")
                got_value = None
            else:
                got_class, got_value = "defined", stats[key]
            if got_class != ref_class:
                rec["c1_classification_mismatches"] += 1
                if len(rec["c1_examples"]) < 5:
                    rec["c1_examples"].append(
                        {
                            "stat": key,
                            "kind": "classification",
                            "got": got_class,
                            "want": ref_class,
                            "values": [v.hex() for v in values[:12]],
                        }
                    )
            elif ref_class == "defined" and got_value != ref_value:
                rec["c1_value_mismatches"] += 1
                if len(rec["c1_examples"]) < 5:
                    rec["c1_examples"].append(
                        {
                            "stat": key,
                            "kind": "value",
                            "got": got_value.hex(),
                            "want": ref_value.hex(),
                            "values": [v.hex() for v in values[:12]],
                        }
                    )
        with np.errstate(all="ignore"):
            nq1, nq3 = (
                float(x) for x in np.quantile(np.asarray(values), [0.25, 0.75], method="linear")
            )
            niqr = nq3 - nq1
        n = len(s)
        for key, nval in (("q1", nq1), ("q3", nq3), ("iqr", niqr)):
            ref_value, ref_class = classify_exact(exact[key])
            if ref_class != "defined":
                continue
            rec["numpy_evaluations"] += 1
            if not math.isfinite(nval):
                rec["numpy_nonfinite_where_oracle_finite"] += 1
                if len(rec["numpy_examples"]) < 5:
                    rec["numpy_examples"].append(
                        {
                            "stat": key,
                            "numpy": repr(nval),
                            "want": ref_value.hex(),
                            "values": [v.hex() for v in values[:12]],
                        }
                    )
                continue
            if nval != ref_value:
                rec["numpy_value_mismatch_bitwise"] += 1
            num = (n - 1) * (1 if key != "q3" else 3)
            lo = num // 4
            hi = min(lo + 1, n - 1)
            operand = (
                max(abs(s[lo]), abs(s[hi]), abs(nval))
                if key != "iqr"
                else max(abs(nq1), abs(nq3), abs(nval))
            )
            r_res = ulp_ratio(nval, exact[key], nval)
            r_op = ulp_ratio(nval, exact[key], operand)
            if r_res is not None:
                rec["numpy_max_result_ulp"] = max(rec["numpy_max_result_ulp"], r_res)
                rec["numpy_result_ulp_gt_2"] += r_res > 2
            if r_op is not None:
                rec["numpy_max_operand_ulp"] = max(rec["numpy_max_operand_ulp"], r_op)
                rec["numpy_operand_ulp_gt_1"] += r_op > 1
                rec["numpy_operand_ulp_gt_4"] += r_op > 4
    return rec


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def main() -> None:
    families = [run_family(*f) for f in FAMILIES]
    json.dump(
        {
            "experiment": "EXP-002",
            "commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "families": families,
            "totals": {
                "production_evaluations": sum(f["production_evaluations"] for f in families),
                "c1_value_mismatches": sum(f["c1_value_mismatches"] for f in families),
                "c1_classification_mismatches": sum(
                    f["c1_classification_mismatches"] for f in families
                ),
                "numpy_evaluations": sum(f["numpy_evaluations"] for f in families),
                "numpy_nonfinite_where_oracle_finite": sum(
                    f["numpy_nonfinite_where_oracle_finite"] for f in families
                ),
            },
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
