"""Real economic observations, kept apart from the supervised intent examples.

Only sources whose licence permits redistribution (registry ``derivatives_allowed: true``) are
copied into ``data/axbench-tr/economic/``: the World Bank WDI slice (CC BY 4.0), the Eurostat
Turkey series (reuse with source acknowledgement) and the İzmir municipal tables (CC BY 4.0).
ECB, OECD and TCMB files stay in the raw directory until their terms are verified.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .fetch import raw_dir, sha256_file
from .schemas import WB_TR

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "axbench-tr" / "economic"


def build_worldbank_csv(out: Path) -> int:
    """Wide country-year table: one row per (country, year), one column per indicator (English
    names)."""
    cells: dict[tuple[str, int], dict[str, float | None]] = {}
    names: dict[str, str] = {}
    for ind in WB_TR:
        payload = json.loads((raw_dir() / "worldbank" / f"{ind}.json").read_text("utf-8"))
        for row in payload[1]:
            names[ind] = row["indicator"]["value"]
            key = (row["country"]["value"], int(row["date"]))
            cells.setdefault(key, {})[ind] = row["value"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Country", "Year", *[names[i] for i in WB_TR]])
        for (country, year), vals in sorted(cells.items()):
            writer.writerow(
                [country, year, *["" if vals.get(i) is None else vals[i] for i in WB_TR]]
            )
    return len(cells)


def build() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    rows = build_worldbank_csv(OUT_DIR / "worldbank_wdi_turkiye_and_peers.csv")
    files.append(
        {"path": "worldbank_wdi_turkiye_and_peers.csv", "source_id": "SRC-O01", "rows": rows}
    )
    for src in sorted((raw_dir() / "eurostat").glob("*.csv")):
        shutil.copyfile(src, OUT_DIR / f"eurostat_{src.name}")
        files.append({"path": f"eurostat_{src.name}", "source_id": "SRC-O02"})
    (OUT_DIR / "izmir").mkdir(exist_ok=True)
    for src in sorted((raw_dir() / "ckan" / "data").glob("*.csv")):
        shutil.copyfile(src, OUT_DIR / "izmir" / src.name)
        files.append({"path": f"izmir/{src.name}", "source_id": "SRC-O06"})
    for rec in files:
        rec["sha256"] = sha256_file(OUT_DIR / str(rec["path"]))
        rec["bytes"] = (OUT_DIR / str(rec["path"])).stat().st_size
    manifest = {"schema": "axbench-tr.economic-observations/1", "files": files}
    (OUT_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    return manifest
