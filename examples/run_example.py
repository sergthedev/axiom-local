"""Run one plan file against the synthetic example dataset and print the canonical result.

Usage: uv run python examples/run_example.py examples/plans/compare.json
This is a demonstration of the library API, not a command-line product.
"""

from __future__ import annotations

import sys
from pathlib import Path

from axiom import ColumnSpec, CsvProfile, load_csv, parse_plan_json, run

HERE = Path(__file__).parent
PROFILE = CsvProfile(
    columns=(
        ColumnSpec(name="region_id", kind="text"),
        ColumnSpec(name="macro_region", kind="text"),
        ColumnSpec(name="gdp_per_capita", kind="numeric"),
        ColumnSpec(name="unemployment_rate", kind="numeric"),
        ColumnSpec(name="education_years", kind="numeric"),
        ColumnSpec(name="household_income", kind="numeric"),
    )
)


def main(plan_path: str) -> None:
    dataset = load_csv(HERE / "synthetic_regions.csv", PROFILE)
    plan = parse_plan_json(Path(plan_path).read_text(encoding="utf-8"))
    result = run(dataset, plan)
    sys.stdout.write(result.canonical_json + "\n")
    sys.stderr.write(f"result_sha256={result.result_sha256}\n")


if __name__ == "__main__":
    main(sys.argv[1])
