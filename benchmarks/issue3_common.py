"""Scenarios shared by the Issue #3 experiments (EXP-004 replay, EXP-005 cross-platform).

Two fully synthetic datasets: the committed example table (150 rows) and a 5,000-row table
generated here from integers only (``random.Random`` with ``randrange``; no floating-point
arithmetic, so the CSV bytes are identical on every platform). Each dataset gets one plan per
operation (Relate with both methods): ten representative operation plans covering all four
operation families.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from axiom import ColumnSpec, CsvProfile
from axiom.passport import profile_from_dict
from axiom.plan import parse_plan_object

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
EXAMPLE_PROFILE = profile_from_dict(
    json.loads((EXAMPLES / "profiles" / "synthetic_regions.json").read_text("utf-8"))
)
SYNTHETIC_SEED = 20261126
SYNTHETIC_ROWS = 5000
SYNTHETIC_PROFILE = CsvProfile(
    columns=(
        ColumnSpec(name="id", kind="text"),
        ColumnSpec(name="segment", kind="text"),
        ColumnSpec(name="f1", kind="numeric"),
        ColumnSpec(name="f2", kind="numeric"),
        ColumnSpec(name="f3", kind="numeric"),
        ColumnSpec(name="y", kind="numeric"),
    )
)
_SEGMENTS = ("North", "South", "East", "West", "Ümran")
_CENTERS = (
    (10_000, 2_000, 50_000),
    (30_000, 9_000, 20_000),
    (55_000, 4_000, 80_000),
    (80_000, 12_000, 35_000),
)


def _milli(value: int) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}{value // 1000}.{value % 1000:03d}"


def synthetic_bytes(rows: int = SYNTHETIC_ROWS, seed: int = SYNTHETIC_SEED) -> bytes:
    """Seeded integer-only table: four point groups, about 1 % missing f2 and 0.5 % missing y."""
    rng = random.Random(seed)
    lines = ["id,segment,f1,f2,f3,y"]
    for i in range(1, rows + 1):
        cx, cy, cz = _CENTERS[rng.randrange(len(_CENTERS))]
        f1 = cx + rng.randrange(-6_000, 6_001)
        f2 = cy + rng.randrange(-2_500, 2_501)
        f3 = cz + rng.randrange(-9_000, 9_001)
        y = 3 * f1 - 2 * f2 + rng.randrange(-40_000, 40_001)
        f2_text = "" if rng.randrange(100) == 0 else _milli(f2)
        y_text = "" if rng.randrange(200) == 0 else _milli(y)
        segment = _SEGMENTS[rng.randrange(len(_SEGMENTS))]
        lines.append(f"S{i:05d},{segment},{_milli(f1)},{f2_text},{_milli(f3)},{y_text}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _plan(obj: dict):
    return parse_plan_object({"schema_version": "axiom.plan/1", **obj})


def example_plan(name: str):
    return parse_plan_object(json.loads((EXAMPLES / "plans" / f"{name}.json").read_text("utf-8")))


SYNTHETIC_PLANS = {
    "inspect": _plan({"operation": "inspect", "columns": ["f1", "f2", "f3", "y", "segment"]}),
    "relate_pearson": _plan({"operation": "relate", "x": "f1", "y": "y", "method": "pearson"}),
    "relate_spearman": _plan({"operation": "relate", "x": "f2", "y": "y", "method": "spearman"}),
    "compare": _plan({"operation": "compare", "group_by": "segment", "measure": "y"}),
    "cluster": _plan({"operation": "cluster", "features": ["f1", "f2", "f3"], "k": 4}),
}
EXAMPLE_PLAN_NAMES = ("inspect", "relate_pearson", "relate_spearman", "compare", "cluster")


def scenarios(workdir: Path) -> list[dict]:
    """The ten scenarios; the synthetic table is written into ``workdir``."""
    synthetic = workdir / "synthetic_5000.csv"
    synthetic.write_bytes(synthetic_bytes())
    out = []
    for name in EXAMPLE_PLAN_NAMES:
        out.append(
            {
                "id": f"example/{name}",
                "data": EXAMPLES / "synthetic_regions.csv",
                "profile": EXAMPLE_PROFILE,
                "plan": example_plan(name),
            }
        )
    for name, plan in SYNTHETIC_PLANS.items():
        out.append(
            {
                "id": f"synthetic5000/{name}",
                "data": synthetic,
                "profile": SYNTHETIC_PROFILE,
                "plan": plan,
            }
        )
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    import subprocess

    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}
