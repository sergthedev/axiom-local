"""The synthetic example dataset and plans keep working and keep their documented properties."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axiom import ColumnSpec, CsvProfile, load_csv, parse_plan_json, run

EXAMPLES = Path(__file__).parent.parent / "examples"
PROFILE = CsvProfile(
    columns=tuple(
        ColumnSpec(name=n, kind=k)
        for n, k in (
            ("region_id", "text"),
            ("macro_region", "text"),
            ("gdp_per_capita", "numeric"),
            ("unemployment_rate", "numeric"),
            ("education_years", "numeric"),
            ("household_income", "numeric"),
        )
    )
)
PLAN_FILES = sorted((EXAMPLES / "plans").glob("*.json"))


@pytest.fixture(scope="module")
def dataset():
    return load_csv(EXAMPLES / "synthetic_regions.csv", PROFILE)


def test_example_file_is_the_committed_synthetic_table(dataset) -> None:
    raw = (EXAMPLES / "synthetic_regions.csv").read_bytes()
    assert dataset.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert dataset.n_rows == 150
    assert "Şahin" in dataset.columns["macro_region"]  # non-ASCII keys survive exactly


@pytest.mark.parametrize("path", PLAN_FILES, ids=lambda p: p.stem)
def test_every_example_plan_runs_and_is_repeatable(dataset, path: Path) -> None:
    plan = parse_plan_json(path.read_text(encoding="utf-8"))
    first = run(dataset, plan)
    assert json.loads(first.canonical_json)["operation"] == plan.operation
    assert run(dataset, plan).canonical_json == first.canonical_json


def test_example_results_have_the_expected_structure(dataset) -> None:
    def result(name: str) -> dict:
        return run(dataset, parse_plan_json((EXAMPLES / "plans" / name).read_text("utf-8"))).core[
            "result"
        ]

    inspect = result("inspect.json")
    by_name = {c["name"]: c for c in inspect["columns"]}
    assert by_name["unemployment_rate"]["n_missing"] == 8
    assert by_name["education_years"]["n_missing"] == 5
    pearson = result("relate_pearson.json")
    assert pearson["n_used"] == 145 and -1.0 <= pearson["coefficient"] <= 1.0
    compare = result("compare.json")
    assert [g["key"] for g in compare["groups"]] == ["Alpha", "Bravo", "Charlie", "Delta", "Şahin"]
    assert sum(g["n_total"] for g in compare["groups"]) == 150
    cluster = result("cluster.json")
    assert cluster["n_eligible"] == 137 and cluster["n_excluded"] == 13
    assert sum(c["size"] for c in cluster["clusters"]) == 137
    assert all(cluster["postconditions"].values())
