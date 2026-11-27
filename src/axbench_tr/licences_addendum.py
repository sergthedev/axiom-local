"""Licence evidence addendum 1: terms for the exact datasets behind the probe and smoke subsets.

``python -m axbench_tr licences-addendum`` reads the corpus, selects the probe and smoke subsets
(``export.select_subsets``), lists the tables they use per schema source, fetches the licence
pages and the dataset-level records for those tables, and writes
``data/axbench-tr/sources/LICENCE_EVIDENCE_ADDENDUM_1.json``: URL, retrieval date, byte count,
SHA-256 of the bytes read, the extracted licence fields or a verbatim excerpt. The original
``LICENCE_EVIDENCE.json`` is not touched. The excerpts are evidence for a human reader, not legal
advice. Pages can change; the hash identifies what was read on the retrieval date.

Only the table and column names of these sources reach the examples (no observations), so the
terms matter for the schema text that the subsets reproduce.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

from .export import RIGHTS_VERIFIED_SCHEMA_SOURCES, load_corpus, select_subsets
from .fetch import FetchError, download, raw_dir
from .licences import _plain
from .registry import REGISTRY_PATH

ADDENDUM_PATH = REGISTRY_PATH.parent / "LICENCE_EVIDENCE_ADDENDUM_1.json"
TABLES_PATH = REGISTRY_PATH.parents[1] / "schemas" / "real_table_schemas.json"
IZMIR_ALLOWED_LICENSE_IDS = ("Izmir-Metropolitan-Municipality-License", "cc-by")
IZMIR_API = "https://acikveri.bizizmir.com/api/3/action"
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT"
WB_API = "https://api.worldbank.org/v2"

PAGES = (
    (
        "AD-01",
        "SRC-O01",
        "portal",
        "https://datacatalog.worldbank.org/search/dataset/0037712/World-Development-Indicators",
        r"License : Creative Commons Attribution 4\.0 This dataset is licensed under Creative "
        r"Commons Attribution 4\.0",
        "WDI dataset page of the World Bank Data Catalog: licensed under CC BY 4.0.",
    ),
    (
        "AD-02",
        "SRC-O01",
        "portal",
        "https://datacatalog.worldbank.org/public-licenses",
        r"licenses datasets under the Creative Commons Attribution 4\.0 International license "
        r"\(CC-BY 4\.0\)\. Many datasets are available under other licenses\. They are labeled "
        r"accordingly.{0,120}",
        "World Bank Data Catalog licensing page: CC BY 4.0 is the default; datasets under other "
        "licences are labelled. The WDI page (AD-01) carries the CC BY 4.0 label.",
    ),
    (
        "AD-03",
        "SRC-O01",
        "portal",
        "https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets",
        r"The content available on this platform is provided for informational and non-commercial "
        r"purposes only, unless otherwise stated\.",
        "World Bank site terms: the general default is non-commercial, 'unless otherwise stated'. "
        "Datasets carry their own licence on their catalogue page, which for WDI is CC BY 4.0 "
        "(AD-01). Recorded as a limit: the WDI use rests on that dataset-specific label.",
    ),
    (
        "AD-04",
        "SRC-O02",
        "portal",
        "https://ec.europa.eu/eurostat/web/main/help/copyright-notice",
        r"Data for countries other than: Member States of the European Union \(EU\) Member States "
        r"of the European Free Trade Association \(EFTA\) official EU acceding and candidate "
        r"countries\.",
        "Eurostat copyright notice, list of data that may not be reused commercially: data for "
        "countries other than EU and EFTA member states and official EU acceding and candidate "
        "countries. Turkey is an official EU candidate country, so the Turkey series are not in "
        "that exception list.",
    ),
    (
        "AD-05",
        "SRC-O02",
        "portal",
        "https://ec.europa.eu/eurostat/web/main/help/copyright-notice",
        r"Reuse of statistical data, metadata, publications.{0,220}",
        "Eurostat copyright notice: general permission to reuse statistical data for commercial "
        "or non-commercial purposes with the source acknowledged.",
    ),
    (
        "AD-06",
        "SRC-O06",
        "portal",
        "https://acikveri.bizizmir.com/tr/license",
        r"İzmir Büyükşehir Belediyesi Açık Veri Lisansı Bu site altında yer alan tüm materyaller "
        r"aksi belirtilmedikçe CC BY 4\.0 uluslararası lisansı ile İzmir Büyükşehir Belediyesi "
        r"tarafından lisanslanmıştır\..{0,260}",
        "The page that every used CKAN package names as its licence URL: 'İzmir Metropolitan "
        "Municipality Open Data Licence', all materials CC BY 4.0 unless stated otherwise; "
        "copying, editing and distribution allowed with attribution, commercial use included.",
    ),
)


def _fetch(url: str, rel: str) -> tuple[dict[str, Any], str]:
    root = raw_dir() / "licence_addendum"
    rec = download(url, rel, root=root, max_bytes=8_000_000)
    text = (root / rel).read_text("utf-8", errors="replace")
    return (
        {
            "url": url,
            "retrieved": rec.retrieved,
            "bytes": rec.bytes,
            "sha256": rec.sha256,
        },
        text,
    )


def tables_used() -> dict[str, list[str]]:
    """Schema ids used by the probe and smoke subsets, per schema source."""
    train, valid, lengths = load_corpus()
    out: dict[str, set[str]] = defaultdict(set)
    for tr, va in select_subsets(train, valid, lengths).values():
        for r in (*tr, *va):
            out[r["provenance"]["schema_source_id"]].add(r["schema_id"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def _table_meta() -> dict[str, dict[str, Any]]:
    tables = json.loads(TABLES_PATH.read_text("utf-8"))["tables"]
    return {t["table_id"]: t for t in tables}


def _page_entries() -> list[dict[str, Any]]:
    entries = []
    for ident, source, scope, url, pattern, supports in PAGES:
        entry: dict[str, Any] = {
            "id": ident,
            "source": source,
            "scope": scope,
            "supports": supports,
        }
        try:
            info, text = _fetch(url, f"{ident}.txt")
            match = re.search(pattern, _plain(text))
            entry |= info | {
                "excerpt": match.group(0)[:500] if match else None,
                "status": "excerpt_found" if match else "excerpt_not_found",
            }
        except (FetchError, OSError) as exc:
            entry |= {"url": url, "status": "fetch_failed", "error": str(exc)[:200]}
        entries.append(entry)
    return entries


def _izmir_entries(table_ids: list[str], meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_package: dict[str, list[str]] = defaultdict(list)
    for tid in table_ids:
        by_package[meta[tid]["group"].removeprefix("izmir/")].append(tid)
    entries: list[dict[str, Any]] = []
    for name, tids in sorted(by_package.items()):
        entry: dict[str, Any] = {
            "id": f"IZ-{name}",
            "source": "SRC-O06",
            "scope": "dataset",
            "tables": tids,
        }
        url = f"{IZMIR_API}/package_show?id={quote(name)}"
        try:
            info, text = _fetch(url, f"izmir_{hashlib.sha256(name.encode()).hexdigest()[:16]}.json")
            result = json.loads(text)["result"]
            resource_ids = {r["id"] for r in result.get("resources", [])}
            wanted = {meta[t]["meta"].get("resource_id") for t in tids}
            entry |= info | {
                "package_name": result.get("name"),
                "license_id": result.get("license_id"),
                "license_title": result.get("license_title"),
                "license_url": result.get("license_url"),
                "organization": (result.get("organization") or {}).get("title"),
                "metadata_modified": result.get("metadata_modified"),
                "resource_ids_found": sorted(wanted & resource_ids),
                "resource_ids_missing": sorted(wanted - resource_ids),
            }
            ok = (
                result.get("license_id") in IZMIR_ALLOWED_LICENSE_IDS
                and not entry["resource_ids_missing"]
            )
            entry["status"] = "terms_documented" if ok else "needs_review"
        except (FetchError, OSError, KeyError, ValueError) as exc:
            entry |= {"url": url, "status": "fetch_failed", "error": str(exc)[:200]}
        entries.append(entry)
    return entries


def _eurostat_entries(table_ids: list[str]) -> list[dict[str, Any]]:
    entries = []
    for tid in table_ids:
        code = tid.split("/")[1].removesuffix("_TR.csv")
        entry: dict[str, Any] = {
            "id": f"EU-{code}",
            "source": "SRC-O02",
            "scope": "dataset",
            "tables": [tid],
            "geo": "TR",
        }
        url = f"{EUROSTAT_API}/{code}/latest?detail=full"
        try:
            info, text = _fetch(url, f"eurostat_{code}.xml")
            root = ElementTree.fromstring(text)
            annotations = "".join(root.itertext())
            creator = re.search(r"corporate-body/([A-Z]+)", text)
            doi = re.search(r"https://doi\.org/(10\.\d+/[A-Za-z0-9_]+)", text)
            entry |= info | {
                "dataflow": f"ESTAT:{code.upper()}",
                "dct_creator": creator.group(1) if creator else None,
                "doi": doi.group(1) if doi else None,
                "annotation_has_other_source_marker": bool(
                    re.search(r"other[- ]source|third[- ]party", annotations, re.I)
                ),
            }
            entry["status"] = (
                "terms_documented"
                if entry["dct_creator"] == "ESTAT"
                and not entry["annotation_has_other_source_marker"]
                else "needs_review"
            )
        except (FetchError, OSError, ElementTree.ParseError) as exc:
            entry |= {"url": url, "status": "fetch_failed", "error": str(exc)[:200]}
        entries.append(entry)
    return entries


def _wdi_entries(table_ids: list[str]) -> list[dict[str, Any]]:
    entries = []
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))
    sources = registry["sources"] if isinstance(registry, dict) else registry
    wdi = next(s for s in sources if s["id"] == "SRC-O01")
    for code in wdi["acquire"]["indicators"]:
        entry: dict[str, Any] = {
            "id": f"WB-{code}",
            "source": "SRC-O01",
            "scope": "indicator",
            "tables_using_the_dataset": len(table_ids),
        }
        url = f"{WB_API}/indicator/{code}?format=json"
        try:
            info, text = _fetch(url, f"wdi_{code}.json")
            rec = json.loads(text)[1][0]
            entry |= info | {
                "indicator_name": rec.get("name"),
                "wdi_source": (rec.get("source") or {}).get("value"),
                "source_organization": re.sub(
                    r"\s*uri: \S+", "", rec.get("sourceOrganization") or ""
                ),
                "licence_field_in_api_record": any("licen" in k.lower() for k in rec),
            }
            entry["status"] = (
                "recorded"
                if entry["wdi_source"] == "World Development Indicators"
                else "needs_review"
            )
        except (FetchError, OSError, KeyError, IndexError, ValueError) as exc:
            entry |= {"url": url, "status": "fetch_failed", "error": str(exc)[:200]}
        entries.append(entry)
    return entries


def collect() -> dict[str, Any]:
    used = tables_used()
    meta = _table_meta()
    entries = _page_entries()
    entries += _wdi_entries(used.get("SRC-O01", []))
    entries += _eurostat_entries(used.get("SRC-O02", []))
    entries += _izmir_entries(used.get("SRC-O06", []), meta)
    return {
        "schema": "axbench-tr.licence-evidence-addendum/1",
        "addendum": 1,
        "retrieved_on": date.today().isoformat(),
        "supplements": "data/axbench-tr/sources/LICENCE_EVIDENCE.json (unchanged)",
        "note": (
            "Evidence for a human reader, not legal advice; hashes identify the bytes read. Only "
            "table and column names of these sources reach the examples; no observations do."
        ),
        "rule": {
            "probe_and_smoke_schema_sources": list(RIGHTS_VERIFIED_SCHEMA_SOURCES),
            "excluded_unverified": ["SRC-O03", "SRC-O04", "SRC-O05"],
            "izmir_allowed_license_ids": list(IZMIR_ALLOWED_LICENSE_IDS),
        },
        "limits": [
            "SRC-O01: no indicator-level licence override was found (the API record has no licence "
            "field and the WDI catalogue page carries one dataset label, CC BY 4.0). Several "
            "indicators are compiled from IMF, ILO, UN, OECD or Eurostat; their upstream terms "
            "were not read separately, so the use rests on the WDI dataset label (AD-01). Only "
            "indicator names reach the examples as column headers.",
            "SRC-O02: terms rest on the Eurostat notice (AD-04, AD-05) and on the dataflow record "
            "naming Eurostat as creator; a dataset-specific copyright note was not found for any "
            "of the three datasets.",
            "SRC-O06: the licence field is the CKAN record's licence_id on the retrieval date; "
            "the per-package notes were not read for free-text exceptions beyond that field.",
            "Not legal advice. Pages and records can change after the retrieval date.",
        ],
        "tables_used": used,
        "entries": entries,
    }


def coverage_problems(addendum: dict[str, Any], used: dict[str, list[str]]) -> list[str]:
    """Problems that stop the probe and smoke subsets from resting on the addendum."""
    problems: list[str] = []
    entries = addendum["entries"]
    for source in used:
        if source not in RIGHTS_VERIFIED_SCHEMA_SOURCES:
            problems.append(f"{source} is not a rights-verified schema source")
    pages = {e["id"]: e for e in entries if e["scope"] == "portal"}
    for ident in ("AD-01", "AD-04", "AD-06"):
        if pages.get(ident, {}).get("status") != "excerpt_found":
            problems.append(f"{ident}: licence page excerpt missing")
    covered: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        if e["scope"] == "dataset" and e["status"] == "terms_documented":
            covered[e["source"]].update(e["tables"])
    for source in ("SRC-O02", "SRC-O06"):
        for tid in used.get(source, []):
            if tid not in covered[source]:
                problems.append(f"{tid}: no documented dataset-level terms")
    return problems


def write() -> Path:
    ADDENDUM_PATH.write_text(
        json.dumps(collect(), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ADDENDUM_PATH
