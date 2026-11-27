"""Official and international source methods (World Bank, municipal CKAN).

Eurostat, ECB and TCMB are plain HTTPS files and use the generic ``http_files`` method.
These methods only fetch what the registry names; they never log in, solve challenges or use keys.
"""

from __future__ import annotations

import datetime as _dt
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .acquire import register_method
from .fetch import FetchError, FileRecord, download, raw_dir, sha256_bytes

ECONOMIC_KEYWORDS = (
    "fiyat",
    "ücret",
    "ucret",
    "gelir",
    "gider",
    "bütçe",
    "butce",
    "vergi",
    "işsizlik",
    "issizlik",
    "istihdam",
    "enflasyon",
    "ihracat",
    "ithalat",
    "ekonomi",
    "tarife",
    "yardım",
    "yardim",
    "hal-fiyat",
    "borç",
    "borc",
    "maaş",
    "maas",
    "kira",
    "konut",
    "turizm",
    "sanayi",
    "tarım",
    "tarim",
)


def _write_generated(rel_path: str, payload: Any, url: str) -> FileRecord:
    data = (json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    dest = raw_dir() / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    today = _dt.datetime.now(_dt.UTC).date().isoformat()
    return FileRecord(rel_path, url, len(data), sha256_bytes(data), today)


def _worldbank_indicators(spec: dict[str, Any]) -> list[FileRecord]:
    countries = ";".join(spec["countries"])
    out = []
    for ind in spec["indicators"]:
        url = (
            f"https://api.worldbank.org/v2/country/{countries}/indicator/{ind}"
            f"?format=json&per_page=20000&date={spec['date']}"
        )
        out.append(download(url, f"worldbank/{ind}.json"))
    return out


def _get_json(url: str, rel: str) -> tuple[Any, FileRecord]:
    rec = download(url, rel, max_bytes=64 * 1024 * 1024)
    return json.loads((raw_dir() / rel).read_text("utf-8")), rec


def _ckan_catalog(spec: dict[str, Any]) -> list[FileRecord]:
    base = spec["base"]
    allowed = set(spec["allowed_license_ids"])
    catalog, rec = _get_json(f"{base}/api/3/action/package_search?rows=1000", "ckan/catalog.json")
    records = [rec]
    packages = catalog["result"]["results"]
    resources = [
        (p, r)
        for p in packages
        if p.get("license_id") in allowed
        for r in p["resources"]
        if r.get("datastore_active")
    ]

    def fields(pr: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        p, r = pr
        url = f"{base}/api/3/action/datastore_search?resource_id={r['id']}&limit=0"
        try:
            rec = download(url, f"ckan/_fields/{r['id']}.json", max_bytes=8 * 1024 * 1024)
            result = json.loads((raw_dir() / rec.path).read_text("utf-8"))["result"]
            cols = [
                {"name": f["id"], "ckan_type": f["type"]}
                for f in result["fields"]
                if f["id"] != "_id"
            ]
            err = None
        except (FetchError, KeyError, ValueError) as exc:
            cols, err = [], f"{type(exc).__name__}: {exc}"
        return {
            "package": p["name"],
            "title": p.get("title"),
            "license_id": p.get("license_id"),
            "resource_id": r["id"],
            "resource_name": r.get("name"),
            "format": r.get("format"),
            "metadata_modified": p.get("metadata_modified"),
            "total_records": None,
            "columns": cols,
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        tables = list(pool.map(fields, resources))
    payload = {
        "portal": base,
        "packages_in_catalog": len(packages),
        "packages_by_license": {
            lic: sum(1 for p in packages if p.get("license_id") == lic)
            for lic in sorted({p.get("license_id") for p in packages}, key=str)
        },
        "tables": tables,
    }
    records.append(
        _write_generated("ckan/izmir_tables.json", payload, f"{base}/api/3/action/datastore_search")
    )
    # A small number of economic CSV tables are downloaded as real observations.
    wanted = []
    for p in packages:
        if p.get("license_id") not in allowed:
            continue
        text = (p["name"] + " " + (p.get("title") or "")).lower()
        if not any(k in text for k in ECONOMIC_KEYWORDS):
            continue
        for r in p["resources"]:
            if r.get("format", "").upper() == "CSV" and r.get("url", "").startswith("https://"):
                wanted.append((p["name"], r))
                break
    for name, r in wanted[: spec.get("max_downloads", 25)]:
        try:
            records.append(
                download(
                    r["url"], f"ckan/data/{name}.csv", max_bytes=spec.get("max_bytes", 5 << 20)
                )
            )
        except FetchError:
            continue  # oversize or unavailable files are skipped: absent from the manifest
    return records


register_method("worldbank_indicators", _worldbank_indicators)
register_method("ckan_catalog", _ckan_catalog)
