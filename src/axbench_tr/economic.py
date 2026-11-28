"""Real economic observations, kept apart from the supervised intent examples.

Only sources whose licence permits redistribution (registry ``derivatives_allowed: true``) are
copied into ``data/axbench-tr/economic/``: the World Bank WDI slice (CC BY 4.0), the Eurostat
Turkey series (reuse with source acknowledgement) and the İzmir municipal tables (CC BY 4.0).
ECB, OECD and TCMB files stay in the raw directory until their terms are verified.
"""

from __future__ import annotations

import csv
import json
import re
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


# Committed copies exclude files that carry contact details (the portal licence would allow them;
# a training pipeline has no use for them, and addresses and phone numbers are better left out).
EXCLUDED = {
    "turizm-bilgilendirme-ofisleri-konum-verisi.csv": "office street addresses and phone numbers",
}
# AI-made decisions on pattern hits in the files that are kept (a human has not confirmed them).
HIT_NOTES = {
    "izmir/imalat-sanayi-ihracati.csv": "phone-like hits are 10-digit trade values",
    "izmir/imalat-sanayi-ithalati.csv": "phone-like hits are 10-digit trade values",
    "worldbank_wdi_turkiye_and_peers.csv": "hits are 11-digit GDP and similar magnitudes",
    "izmir/kablosuz-internet-baglanti-noktalari.csv": "'ADI' is a street name (public Wi-Fi point)",
    "izmir/balik-hal-fiyatlari.csv": "'MAL_ADI' is a commodity name",
    "izmir/sebze-ve-meyve-hal-fiyatlari.csv": "'MAL_ADI' is a commodity name",
    "izmir/izmir-ilcelerine-ait-konut-ve-ozel-is-yeri-istatistikleri.csv": (
        "ILCE_ADI and MAHALLE_ADI are district and neighbourhood names"
    ),
}


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
        if src.name in EXCLUDED:
            (OUT_DIR / "izmir" / src.name).unlink(missing_ok=True)
            continue
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


# ------------------------------------------------------------ privacy and licence audit

_PERSONAL_HEADER = re.compile(
    r"(^|[^a-z])(ad|adi|soyad|soyadi|isim|ad_soyad|tc|tckn|kimlik|telefon|tel|gsm|eposta|e_posta"
    r"|email|mail|adres|plaka|iban|dogum|name|surname|phone|address|passport)([^a-z]|$)",
    re.I,
)
_VALUE_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "tr_phone": re.compile(
        r"(?<!\d)(?:\+?90|0)?\s?\(?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
    ),
    "tc_identity_number": re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)"),
    "iban": re.compile(r"TR\d{2}\s?(?:\d{4}\s?){5}\d{2}", re.I),
}


def privacy_and_licence_audit(out_dir: Path = OUT_DIR) -> dict[str, object]:
    """Scan the committed CSVs for direct identifiers and record each İzmir dataset's CKAN licence.

    Header names are matched against personal-field words; every cell is matched against e-mail,
    Turkish mobile phone, 11-digit identity-number and IBAN patterns. A hit is a candidate for
    review, not proof of personal data (a 11-digit number may be a count or an id).
    """
    packages = {}
    catalog = raw_dir() / "ckan" / "catalog.json"
    if catalog.exists():
        for pkg in json.loads(catalog.read_text("utf-8"))["result"]["results"]:
            packages[pkg["name"]] = {
                "license_id": pkg.get("license_id"),
                "license_title": pkg.get("license_title"),
            }
    report = []
    for path in sorted(out_dir.rglob("*.csv")):
        text = path.read_text("utf-8-sig", errors="replace")
        first = text.splitlines()[0] if text else ""
        delim = ";" if first.count(";") > first.count(",") else ","
        reader = csv.reader(text.splitlines(), delimiter=delim)
        header = next(reader, [])
        hits = {k: 0 for k in _VALUE_PATTERNS}
        n_rows = 0
        for row in reader:
            n_rows += 1
            for cell in row:
                for kind, pat in _VALUE_PATTERNS.items():
                    hits[kind] += bool(pat.search(cell))
        entry: dict[str, object] = {
            "file": path.relative_to(out_dir).as_posix(),
            "rows": n_rows,
            "personal_looking_headers": [h for h in header if _PERSONAL_HEADER.search(h)],
            "value_pattern_hits": {k: v for k, v in hits.items() if v},
        }
        if path.parent.name == "izmir":
            entry["ckan_package"] = packages.get(path.stem)
        if entry["personal_looking_headers"] or entry["value_pattern_hits"]:
            entry["ai_note"] = HIT_NOTES.get(str(entry["file"]), "unreviewed")
        report.append(entry)
    flagged = [
        r["file"] for r in report if r["personal_looking_headers"] or r["value_pattern_hits"]
    ]
    return {
        "schema": "axbench-tr.economic-privacy-audit/1",
        "files_scanned": len(report),
        "files_with_candidates": flagged,
        "excluded_from_commit": EXCLUDED,
        "files": report,
        "limits": "pattern scan only; a human must still decide on every candidate",
    }
