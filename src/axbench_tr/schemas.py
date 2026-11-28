"""Real table schemas for AXBENCH-TR (column names and Core kinds only; never observations).

Schemas come from the files acquired by :mod:`axbench_tr.acquire`. Where a Turkish rendering of an
official English header was written by the implementing AI tool (the World Bank indicator glossary
and the TCMB header variant), the table carries ``localized=True`` and the corpus records it; those
renderings are not official translations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from axiom_router.schema import deasciify

from .fetch import raw_dir
from .official import ECONOMIC_KEYWORDS

MAX_COLUMNS = 64


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    kind: str  # "numeric" | "text"
    slot_ok: bool = (
        True  # may be chosen as a plan column (id, flag and date columns are distractors)
    )


@dataclass(frozen=True, slots=True)
class RealTable:
    table_id: str
    source_id: str
    group: str  # partition unit: all tables of one group fall in one split
    title: str
    domain: str  # "economic" | "non_economic"
    header_language: str  # "tr" | "en"
    columns: tuple[Column, ...]
    localized: bool = False
    meta: dict[str, str] = field(default_factory=dict, hash=False, compare=False)


_NOT_SLOT_TOKENS = frozenset(
    "id kod kodu no numara enlem boylam lat lon latitude longitude tarih date time period unit "
    "mult decimals flag status conf key dataflow structure action version adres aciklama telefon "
    "email url link adi ad isim name last update freq formatted comment note".split()
)


def tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", deasciify(name)) if t]


_TIME_TOKENS = frozenset("yil year ay month donem period quarter ceyrek hafta week gun day".split())


def time_like(name: str) -> bool:
    """Calendar columns (year, month, quarter): summarizable, but not meaningful measures."""
    toks = tokens(name)
    return bool(toks) and toks[-1] in _TIME_TOKENS and len(toks) <= 2


def slot_ok(name: str, kind: str) -> bool:
    toks = tokens(name)
    if not toks or any(t in _NOT_SLOT_TOKENS for t in toks):
        return False
    return not (kind == "numeric" and toks[-1].endswith("id") and len(toks[-1]) > 2)


def _col(name: str, kind: str) -> Column:
    name = name.lstrip("\ufeff\u200b").strip()  # a UTF-8 byte-order mark can lead a CSV header
    return Column(name, kind, slot_ok(name, kind))


def _is_economic(text: str) -> bool:
    low = deasciify(text)
    return any(deasciify(k) in low for k in ECONOMIC_KEYWORDS)


def partition_of(group: str) -> str:
    """Deterministic split of a table group: about one fifth go to validation."""
    digest = hashlib.sha256(f"axbench-tr/0.2/table-group/{group}".encode()).digest()
    return "validation" if digest[0] < 51 else "train"


# ------------------------------------------------------------------------------------- İzmir

_CKAN_KIND = {"numeric": "numeric", "int4": "numeric", "float8": "numeric", "text": "text"}


def izmir_tables() -> list[RealTable]:
    path = raw_dir() / "ckan" / "izmir_tables.json"
    payload = json.loads(path.read_text("utf-8"))
    out, seen = [], set()
    for t in payload["tables"]:
        cols = tuple(_col(c["name"], _CKAN_KIND.get(c["ckan_type"], "text")) for c in t["columns"])
        if not cols or len(cols) > MAX_COLUMNS:
            continue
        key = (t["package"], tuple((c.name, c.kind) for c in cols))
        if key in seen:  # the CSV and XLSX resources of one table share their schema
            continue
        seen.add(key)
        n = sum(1 for k in seen if k[0] == t["package"])
        out.append(
            RealTable(
                table_id=f"izmir/{t['package']}/{n}",
                source_id="SRC-O06",
                group=f"izmir/{t['package']}",
                title=t.get("title") or t["package"],
                domain="economic"
                if _is_economic(t["package"] + " " + (t.get("title") or ""))
                else "non_economic",
                header_language="tr",
                columns=cols,
                meta={"license_id": str(t["license_id"]), "resource_id": t["resource_id"]},
            )
        )
    return out


# ------------------------------------------------------------------------------ TCMB (XML)

TCMB_EN = (
    "Kod",
    "Unit",
    "Isim",
    "CurrencyName",
    "ForexBuying",
    "ForexSelling",
    "BanknoteBuying",
    "BanknoteSelling",
)
TCMB_TR = (
    "Kod",
    "Birim",
    "Para Birimi",
    "Para Birimi (İngilizce)",
    "Döviz Alış",
    "Döviz Satış",
    "Efektif Alış",
    "Efektif Satış",
)
_TCMB_KINDS = ("text", "numeric", "text", "text", "numeric", "numeric", "numeric", "numeric")


def tcmb_tables() -> list[RealTable]:
    files = sorted((raw_dir() / "tcmb").glob("kurlar_*.xml"))
    if not files:
        return []
    root = ET.parse(files[0]).getroot()
    cur = root.find("Currency")
    assert cur is not None
    child_tags = [c.tag for c in cur]
    expected = [
        "Unit",
        "Isim",
        "CurrencyName",
        "ForexBuying",
        "ForexSelling",
        "BanknoteBuying",
        "BanknoteSelling",
    ]
    if child_tags[:7] != expected:
        raise ValueError(f"unexpected TCMB layout: {child_tags}")
    out = []
    for lang, names, localized in (("en", TCMB_EN, False), ("tr", TCMB_TR, True)):
        cols = tuple(Column(n, k, k == "numeric") for n, k in zip(names, _TCMB_KINDS, strict=True))
        out.append(
            RealTable(
                f"tcmb/kurlar/{lang}",
                "SRC-O05",
                "tcmb/kurlar",
                "TCMB günlük döviz kurları",
                "economic",
                lang,
                cols,
                localized,
            )
        )
    return out


# ----------------------------------------------------------------- SDMX / CSV files (inferred)


def _infer_kinds(text: str, delimiter: str = ",") -> list[tuple[str, str]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = next(reader)
    rows = [r for _, r in zip(range(400), reader, strict=False)]
    out = []
    for i, name in enumerate(header):
        vals = [r[i].strip() for r in rows if i < len(r) and r[i].strip()]
        numeric = bool(vals) and all(
            re.fullmatch(r"-?\d+([.,]\d+)?([eE][-+]?\d+)?", v) for v in vals
        )
        out.append((name, "numeric" if numeric else "text"))
    return out


def _sdmx_table(source_id: str, rel: str, title: str) -> RealTable | None:
    path = raw_dir() / rel
    if not path.exists():
        return None
    cols = tuple(_col(n, k) for n, k in _infer_kinds(path.read_text("utf-8-sig")) if n)
    if len(cols) > MAX_COLUMNS:
        return None
    return RealTable(f"{rel}", source_id, rel, title, "economic", "en", cols)


def sdmx_tables() -> list[RealTable]:
    specs = [
        ("SRC-O02", "eurostat/une_rt_m_TR.csv", "Eurostat monthly unemployment (Turkey)"),
        ("SRC-O02", "eurostat/prc_hicp_manr_TR.csv", "Eurostat HICP annual rate (Turkey)"),
        ("SRC-O02", "eurostat/namq_10_gdp_TR.csv", "Eurostat quarterly GDP growth (Turkey)"),
        ("SRC-O03", "ecb/exr_try_eur_daily.csv", "ECB TRY per EUR reference rate"),
        ("SRC-O04", "oecd/une_m_TUR.csv", "OECD monthly unemployment (Turkiye)"),
        ("SRC-O04", "oecd/prices_TUR.csv", "OECD consumer prices (Turkiye)"),
    ]
    return [t for s in specs if (t := _sdmx_table(*s))]


# ---------------------------------------------------------------------------- World Bank wide

WB_TR = {
    "FP.CPI.TOTL.ZG": "Enflasyon, tüketici fiyatları (yıllık %)",
    "SL.UEM.TOTL.ZS": "İşsizlik, toplam (işgücünün %'si) (ILO model tahmini)",
    "NY.GDP.MKTP.CD": "GSYH (cari ABD doları)",
    "NY.GDP.MKTP.KD.ZG": "GSYH büyümesi (yıllık %)",
    "NY.GDP.PCAP.CD": "Kişi başına GSYH (cari ABD doları)",
    "NE.EXP.GNFS.ZS": "Mal ve hizmet ihracatı (GSYH'nin %'si)",
    "NE.IMP.GNFS.ZS": "Mal ve hizmet ithalatı (GSYH'nin %'si)",
    "PA.NUS.FCRF": "Resmi döviz kuru (ABD doları başına yerel para birimi, dönem ortalaması)",
    "FR.INR.LEND": "Kredi faiz oranı (%)",
    "SP.POP.TOTL": "Toplam nüfus",
    "GC.DOD.TOTL.GD.ZS": "Merkezi yönetim borcu, toplam (GSYH'nin %'si)",
    "BN.CAB.XOKA.GD.ZS": "Cari işlemler dengesi (GSYH'nin %'si)",
    "SL.TLF.CACT.ZS": "İşgücüne katılım oranı, toplam (15+ yaş nüfusun %'si) (ILO model tahmini)",
}
# Disjoint indicator pools keep the two splits' World Bank column names apart.
WB_POOLS = {
    "train": (
        "FP.CPI.TOTL.ZG",
        "NY.GDP.MKTP.KD.ZG",
        "NE.EXP.GNFS.ZS",
        "NE.IMP.GNFS.ZS",
        "BN.CAB.XOKA.GD.ZS",
        "SP.POP.TOTL",
        "PA.NUS.FCRF",
    ),
    "validation": (
        "SL.UEM.TOTL.ZS",
        "NY.GDP.PCAP.CD",
        "FR.INR.LEND",
        "GC.DOD.TOTL.GD.ZS",
        "SL.TLF.CACT.ZS",
        "NY.GDP.MKTP.CD",
    ),
}


def worldbank_tables(variants_per_pool: int = 14) -> list[RealTable]:
    names: dict[str, str] = {}
    for ind in WB_TR:
        path = raw_dir() / "worldbank" / f"{ind}.json"
        if not path.exists():
            return []
        names[ind] = json.loads(path.read_text("utf-8"))[1][0]["indicator"]["value"]
    out = []
    for split, pool in WB_POOLS.items():
        rng = random.Random(f"axbench-tr/0.2/worldbank/{split}")
        for i in range(variants_per_pool):
            chosen = rng.sample(pool, rng.randint(3, len(pool)))
            lang = "tr" if i % 4 else "en"  # three Turkish renderings for each English one
            head = (
                [("Ülke", "text"), ("Yıl", "numeric")]
                if lang == "tr"
                else [("Country", "text"), ("Year", "numeric")]
            )
            cols = [Column(n, k, True) for n, k in head]
            cols += [
                Column(WB_TR[c] if lang == "tr" else names[c], "numeric", True) for c in chosen
            ]
            out.append(
                RealTable(
                    f"worldbank/{split}/{i}",
                    "SRC-O01",
                    f"worldbank/{split}",
                    "World Bank WDI country-year table",
                    "economic",
                    lang,
                    tuple(cols),
                    localized=(lang == "tr"),
                )
            )
    return out


INVENTORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "axbench-tr"
    / "schemas"
    / "real_table_schemas.json"
)
INVENTORY_SCHEMA = "axbench-tr.table-schemas/1"


def tables_from_raw() -> list[RealTable]:
    """Parse the acquired raw files (needs ``python -m axbench_tr acquire`` first)."""
    tables = izmir_tables() + tcmb_tables() + sdmx_tables() + worldbank_tables()
    ids = [t.table_id for t in tables]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate table ids")
    return tables


def export_inventory(path: Path = INVENTORY_PATH) -> int:
    """Write the committed schema inventory (column names and kinds only, no observations)."""
    tables = tables_from_raw()
    payload = {
        "schema": INVENTORY_SCHEMA,
        "tables": [
            {
                "table_id": t.table_id,
                "source_id": t.source_id,
                "group": t.group,
                "title": t.title,
                "domain": t.domain,
                "header_language": t.header_language,
                "localized": t.localized,
                "meta": t.meta,
                "columns": [
                    {"name": c.name, "kind": c.kind, "slot_ok": c.slot_ok} for c in t.columns
                ],
            }
            for t in tables
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return len(tables)


def all_tables(path: Path = INVENTORY_PATH) -> list[RealTable]:
    """The committed schema inventory (no network or raw files needed)."""
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unexpected schema inventory version")
    return [
        RealTable(
            t["table_id"],
            t["source_id"],
            t["group"],
            t["title"],
            t["domain"],
            t["header_language"],
            tuple(Column(c["name"], c["kind"], c["slot_ok"]) for c in t["columns"]),
            t["localized"],
            t["meta"],
        )
        for t in payload["tables"]
    ]


def schema_key(columns: Iterable[tuple[str, str]]) -> str:
    """Order-free identity of a column list (folded names and kinds)."""
    pairs = sorted((re.sub(r"\W+", "", deasciify(n).lower()), k) for n, k in columns)
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest()[:16]


def components(links: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Union-find over (a, b) links: every node maps to the smallest node of its component."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in list(parent)}


@lru_cache(maxsize=1)
def _group_roots() -> dict[str, str]:
    """Table groups that share an identical schema are one family and go to one split."""
    by_key: dict[str, str] = {}
    links: list[tuple[str, str]] = []
    for t in all_tables():
        links.append((t.group, t.group))
        key = schema_key((c.name, c.kind) for c in t.columns)
        links.append((by_key.setdefault(key, t.group), t.group))
    return components(links)


def table_partition(table: RealTable) -> str:
    if table.source_id == "SRC-O01":
        return table.group.rsplit("/", 1)[1]
    if table.source_id == "SRC-O05":
        return "train"
    return partition_of(_group_roots().get(table.group, table.group))


def usable(table: RealTable) -> bool:
    """Enough plan-eligible columns to host at least an inspect, compare or relate example."""
    num = sum(1 for c in table.columns if c.kind == "numeric" and c.slot_ok)
    return num >= 1 and len(table.columns) >= 2


def summary(tables: Iterable[RealTable]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tables:
        key = f"{table_partition(t)}:{t.source_id}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
