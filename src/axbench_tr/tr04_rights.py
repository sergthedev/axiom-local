"""EXP-TR04 rights evidence (``python -m axbench_tr.tr04 rights-fetch|rights-check``).

Status: **automated screening, not a legal opinion.** EXP-TR03 left three open rights caveats:
the upstream terms of World Bank indicators, Eurostat's dataset-specific notes, and the free-text
notes of the İzmir open-data packages. EXP-TR04 is an automated exploratory run, so it does not
ask a human to resolve them; it uses only layouts whose caveat could be checked here and excludes
everything else (see ``tr04.rights_decision``).

* İzmir (SRC-O06): the package records of every table used by the corpus are fetched once from the municipality's public CKAN
  API (``resource_show`` to find the package of each table, then ``package_show``). Only text fields are kept (title, notes, extras, resource names and
  descriptions) together with the SHA-256 of the raw API response; contact fields are dropped.
  A deterministic, documented pattern list screens the text for restrictions, consent or
  permission wording, copyright reservations, commercial limits, personal-data notes, third-party
  ownership and other licences. A package with any hit is excluded as uncertain. No hit means only
  that this screen found nothing, not that no restriction exists.
* World Bank (SRC-O01): every indicator in the committed licence addendum names an upstream
  compiler (IMF, ILO, UN, national statistical offices). Their terms were never read, so every
  layout containing an indicator column is uncertain; the evidence table below is derived from the
  committed addendum, nothing new is fetched.
* Eurostat (SRC-O02), ECB/OECD/TCMB (SRC-O03 to O05) and external adapted rows are excluded as in
  EXP-TR03; Eurostat additionally because its dataset-specific copyright notes were not found and
  the reading of the candidate-country exception is a legal judgement this run does not make.

``rights-fetch`` needs network access and is run once; the result is committed
(``data/axbench-tr/tr04/rights/IZMIR_PACKAGE_NOTES.json``) and every later build and check is
offline.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from axiom_router.schema import turkish_lower

from .build import OUT_ROOT
from .fetch import FetchError, download

TR04_DIR = OUT_ROOT / "tr04"
RIGHTS_DIR = TR04_DIR / "rights"
IZMIR_NOTES_PATH = RIGHTS_DIR / "IZMIR_PACKAGE_NOTES.json"
ADDENDUM_PATH = OUT_ROOT / "sources" / "LICENCE_EVIDENCE_ADDENDUM_1.json"
TABLES_PATH = OUT_ROOT / "schemas" / "real_table_schemas.json"
IZMIR_API = "https://acikveri.bizizmir.com/api/3/action/"
SCHEMA = "axbench-tr.tr04-izmir-package-notes/2"

# Screening patterns, applied to Turkish-aware lower-cased text. They are deliberately broad: a
# hit excludes the package (uncertain), it does not assert a restriction. ``sınırlı``-type words
# are matched without the plain noun ``sınır`` ("boundary"), which every geographic description
# uses ("İzmir ili sınırları içerisinde").
PATTERNS: dict[str, str] = {
    "permission_or_consent": r"\bizin|\bonay\b|permission|consent|authori[sz]",
    "copyright_reserved": r"telif|copyright|©|all rights|hakları saklı|haklari sakli",
    "commercial_wording": r"ticari|commercial",
    "restriction_wording": (
        r"yasak|kısıt|kisit|sınırlı\b|sınırlama|sınırlandır|sinirli\b|sinirlama"
        r"|restrict|prohibit|forbid|not allowed|paylaşılamaz|çoğaltılamaz|kullanılamaz"
    ),
    "personal_data": r"kişisel|kisisel|kvkk|personal data|gdpr",
    "third_party_ownership": (
        r"üçüncü|ucuncu|third.part|veri sahibi|data owner|başka kurum|baska kurum"
    ),
    "other_licence": (
        r"cc[ -]by[- ]?(nc|nd|sa)|non-?commercial|ticari olmayan|\bgnu\b|\bgpl\b|odbl|\bodc\b"
    ),
}
_COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}

# Fields kept from a package record. Contact fields (author, maintainer, e-mail) are never kept.
KEPT = ("title", "notes", "license_id", "license_title", "metadata_modified")


def scan_text(text: str) -> list[str]:
    """Names of the screening patterns that match ``text`` (sorted; empty means no hit)."""
    folded = turkish_lower(text)
    return sorted(name for name, rx in _COMPILED.items() if rx.search(folded))


def package_text(rec: Mapping[str, Any]) -> str:
    """Every free-text field of one kept package record, joined for screening."""
    parts = [str(rec.get(k) or "") for k in ("title", "notes")]
    parts += [f"{e['key']}: {e['value']}" for e in rec.get("extras", [])]
    parts += [f"{r['name']} {r['description']}" for r in rec.get("resources", [])]
    return "\n".join(parts)


def extract(api_json: Mapping[str, Any], raw_sha256: str, raw_bytes: int) -> dict[str, Any]:
    """The kept view of a ``package_show`` response (text fields only; no contact data)."""
    res = api_json["result"]
    rec: dict[str, Any] = {k: res.get(k) for k in KEPT}
    rec["name"] = res["name"]
    rec["extras"] = [
        {"key": str(e.get("key", "")), "value": str(e.get("value", ""))}
        for e in res.get("extras", [])
    ]
    rec["resources"] = [
        {
            "id": str(r.get("id", "")),
            "name": str(r.get("name") or ""),
            "description": str(r.get("description") or ""),
            "format": str(r.get("format") or ""),
        }
        for r in res.get("resources", [])
    ]
    rec["raw_sha256"] = raw_sha256
    rec["raw_bytes"] = raw_bytes
    rec["flags"] = scan_text(package_text(rec))
    return rec


def izmir_tables(tables_path: Path = TABLES_PATH) -> dict[str, str]:
    """``{table id: CKAN resource id}`` of every SRC-O06 table of the committed layout file.

    The licence addendum of EXP-TR02 covered only the 93 packages of the probe and smoke subsets;
    the full corpus uses more, so every table is screened here, resolved through its resource id.
    """
    tables = json.loads(tables_path.read_text("utf-8"))["tables"]
    return {t["table_id"]: t["meta"]["resource_id"] for t in tables if t["source_id"] == "SRC-O06"}


def table_group(table_id: str) -> str:
    """``izmir/<package group>/1`` -> ``izmir/<package group>`` (the corpus ``family``)."""
    return table_id.rsplit("/", 1)[0]


def wdi_evidence(addendum_path: Path = ADDENDUM_PATH) -> list[dict[str, Any]]:
    """Per-indicator upstream compilers recorded in the committed addendum (nothing is fetched)."""
    entries = json.loads(addendum_path.read_text("utf-8"))["entries"]
    out = []
    for e in sorted(entries, key=lambda e: e["id"]):
        if e.get("scope") != "indicator":
            continue
        org = " ".join(str(e["source_organization"]).split())
        out.append(
            {
                "id": e["id"],
                "indicator": e["indicator_name"],
                "upstream_compiler": org[:160],
                "third_party_compiler": "world bank" not in org.lower().split(";")[0],
                "addendum_sha256": e["sha256"],
            }
        )
    return out


def _get_json(url: str, tmp: Path, name: str) -> tuple[dict[str, Any], str, int]:
    rec = download(url, name, root=tmp, max_bytes=4 << 20)
    raw = (tmp / name).read_bytes()
    return json.loads(raw), rec.sha256, rec.bytes


def fetch_izmir(dest: Path = IZMIR_NOTES_PATH, *, today: str | None = None) -> dict[str, Any]:
    """Fetch the package records once and write the kept view (network; run by a person)."""
    resources = izmir_tables()
    stamp = today or _dt.datetime.now(_dt.UTC).date().isoformat()
    kept: dict[str, Any] = {}
    table_package: dict[str, dict[str, str]] = {}
    package_of_resource: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for table_id, resource_id in sorted(resources.items()):
            if resource_id not in package_of_resource:
                try:
                    res, _, _ = _get_json(
                        f"{IZMIR_API}resource_show?id={resource_id}", tmp, f"r-{resource_id}.json"
                    )
                except FetchError as exc:
                    raise SystemExit(f"resource lookup failed for {table_id}: {exc}") from exc
                package_of_resource[resource_id] = res["result"]["package_id"]
            table_package[table_id] = {
                "resource_id": resource_id,
                "package_id": package_of_resource[resource_id],
            }
        for pid in sorted(set(package_of_resource.values())):
            try:
                api, sha, size = _get_json(
                    f"{IZMIR_API}package_show?id={pid}", tmp, f"p-{pid}.json"
                )
            except FetchError as exc:
                raise SystemExit(f"package fetch failed for {pid}: {exc}") from exc
            rec = extract(api, sha, size)
            rec["package_id"] = pid
            kept[rec["name"]] = rec
    by_id = {r["package_id"]: name for name, r in kept.items()}
    for info in table_package.values():
        info["package"] = by_id[info["package_id"]]
    for name, rec in kept.items():
        rec["tables"] = sorted(t for t, i in table_package.items() if i["package"] == name)
    doc = {
        "schema": SCHEMA,
        "retrieved": stamp,
        "source": "SRC-O06",
        "endpoints": [
            IZMIR_API + "resource_show?id=<resource id>",
            IZMIR_API + "package_show?id=<package id>",
        ],
        "patterns": PATTERNS,
        "tables": dict(sorted(table_package.items())),
        "packages": dict(sorted(kept.items())),
        "limits": [
            "automated screen of text fields; not a legal opinion",
            "no hit means this screen found nothing, not that no restriction exists",
            "records can change after the retrieval date; raw_sha256 identifies the bytes read",
            "contact fields of the API record are not kept",
        ],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")
    return doc


def load_izmir(path: Path = IZMIR_NOTES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def izmir_status(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{group: {package, flags, licence_ok}}``; the flags are re-computed from the kept text.

    A group is usable only if the committed record's licence id is one of the two the addendum
    allowed, its stored flags equal a fresh screen, and the screen is empty.
    """
    allowed = {"Izmir-Metropolitan-Municipality-License", "cc-by"}
    out: dict[str, dict[str, Any]] = {}
    for name, rec in doc["packages"].items():
        fresh = scan_text(package_text(rec))
        problems = []
        if rec["license_id"] not in allowed:
            problems.append(f"licence:{rec['license_id']}")
        if fresh != rec["flags"]:
            problems.append("stored_flags_differ_from_rescan")
        problems += [f"flag:{f}" for f in fresh]
        for table in rec["tables"]:
            out[table_group(table)] = {
                "package": name,
                "problems": problems,
                "usable": not problems,
            }
    return out


def check_evidence(doc: Mapping[str, Any]) -> list[str]:
    """Structural problems of the committed evidence file (offline)."""
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append("schema_differs")
    expected = izmir_tables()
    if sorted(doc.get("tables", {})) != sorted(expected):
        problems.append("table_set_differs_from_the_layout_file")
    for table_id, info in doc.get("tables", {}).items():
        if info.get("resource_id") != expected.get(table_id):
            problems.append(f"resource_differs:{table_id}")
        if info.get("package") not in doc.get("packages", {}):
            problems.append(f"table_without_package_record:{table_id}")
    for name, rec in doc.get("packages", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(rec.get("raw_sha256", ""))):
            problems.append(f"bad_raw_hash:{name}")
        if any(k in rec for k in ("author", "author_email", "maintainer", "maintainer_email")):
            problems.append(f"contact_field_kept:{name}")
        if not rec.get("tables"):
            problems.append(f"package_without_tables:{name}")
    if doc.get("patterns") != PATTERNS:
        problems.append("patterns_differ_from_the_code")
    return problems


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over fetch/check
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["fetch"]:
        doc = fetch_izmir()
        flagged = sum(bool(r["flags"]) for r in doc["packages"].values())
        print(json.dumps({"packages": len(doc["packages"]), "flagged": flagged}))
        return 0
    if args == ["check"]:
        problems = check_evidence(load_izmir())
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    raise SystemExit("usage: python -m axbench_tr.tr04_rights fetch|check")


if __name__ == "__main__":
    sys.exit(main())
