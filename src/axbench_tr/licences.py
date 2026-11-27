"""Licence evidence: fetch each licence page or card, hash it and keep a short excerpt.

``python -m axbench_tr licences`` downloads the pages below into the ignored raw directory and
writes ``data/axbench-tr/sources/LICENCE_EVIDENCE.json`` (URL, access date, SHA-256 of the fetched
bytes, a short verbatim excerpt, and what the excerpt supports). Pages and cards can change; the
hash identifies what was read on the access date. The excerpts are evidence for a human reader, not
legal advice and not a clearance.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .fetch import FetchError, download, raw_dir
from .registry import REGISTRY_PATH

EVIDENCE_PATH = REGISTRY_PATH.parent / "LICENCE_EVIDENCE.json"


@dataclass(frozen=True)
class Check:
    id: str
    sources: tuple[str, ...]
    url: str
    pattern: str  # regular expression locating the supporting sentence
    supports: str
    kind: str = "html"  # "html" strips markup first; "json" searches the raw text


CHECKS = (
    Check(
        "LE-01",
        ("SRC-N01",),
        "https://huggingface.co/api/datasets/Cognipeer/synthetic-text-to-sql-tr",
        r'"license:apache-2\.0"',
        "Hub metadata tag for the Turkish translation: Apache-2.0. Apache-2.0 permits "
        "modification and redistribution with notice and a statement of changes.",
        "json",
    ),
    Check(
        "LE-02",
        ("SRC-N01",),
        "https://huggingface.co/api/datasets/gretelai/synthetic_text_to_sql",
        r'"license:apache-2\.0"',
        "The upstream English dataset that Cognipeer translated is Apache-2.0 on the Hub.",
        "json",
    ),
    Check(
        "LE-03",
        ("SRC-N03",),
        "https://huggingface.co/api/datasets/AlicanKiraz0/Turkish-Finance-SFT-Dataset",
        r'"license:mit"',
        "Hub metadata tag: MIT (the README badge says Apache-2.0; both are permissive). Used only "
        "for short out-of-scope questions that become REJECT negatives.",
        "json",
    ),
    Check(
        "LE-04",
        ("SRC-O06",),
        "https://acikveri.bizizmir.com/license",
        r"tüm materyaller aksi belirtilmedikçe CC BY 4\.0.{0,320}",
        "İzmir open-data licence page: CC BY 4.0, copying, editing and distribution allowed with "
        "attribution, commercial use included. Datasets labelled otherwise were excluded.",
    ),
    Check(
        "LE-05",
        ("SRC-O01",),
        "https://datacatalog.worldbank.org/search/dataset/0037712/World-Development-Indicators",
        r"License : Creative Commons Attribution 4\.0.{0,80}",
        "WDI-specific catalogue page: licensed under Creative Commons Attribution 4.0.",
    ),
    Check(
        "LE-06",
        ("SRC-O01",),
        "https://datacatalog.worldbank.org/public-licenses",
        r"The Creative Commons Attribution 4\.0 International license allows.{0,200}",
        "CC BY 4.0 allows copying, modification and distribution for any purpose with credit.",
    ),
    Check(
        "LE-07",
        ("SRC-O02",),
        "https://ec.europa.eu/eurostat/web/main/help/copyright-notice",
        r"Reuse of statistical data, metadata, publications.{0,220}",
        "Eurostat copyright notice: reuse of statistical data is authorised for commercial and "
        "non-commercial purposes provided the source is acknowledged.",
    ),
    Check(
        "LE-08",
        ("SRC-M01",),
        "https://huggingface.co/api/models/Qwen/Qwen3-1.7B",
        r'"license:apache-2\.0"',
        "Qwen3-1.7B: Apache-2.0. Only the tokenizer and chat template were downloaded.",
        "json",
    ),
)


def _plain(text: str) -> str:
    text = re.sub(r"(?s)<(script|style).*?</\1>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))


def collect() -> dict[str, object]:
    root = raw_dir() / "licence_pages"
    entries = []
    for check in CHECKS:
        entry: dict[str, object] = {
            "id": check.id,
            "sources": list(check.sources),
            "url": check.url,
            "supports": check.supports,
        }
        try:
            record = download(check.url, f"{check.id}.txt", root=root, max_bytes=5_000_000)
            text = (root / f"{check.id}.txt").read_text("utf-8", errors="replace")
            match = re.search(check.pattern, text if check.kind == "json" else _plain(text))
            entry |= {
                "retrieved": record.retrieved,
                "bytes": record.bytes,
                "sha256": record.sha256,
                "excerpt": match.group(0)[:400] if match else None,
                "status": "excerpt_found" if match else "excerpt_not_found",
            }
        except (FetchError, OSError) as exc:
            entry |= {"status": "fetch_failed", "error": str(exc)[:200], "retrieved": None}
        entries.append(entry)
    return {
        "schema": "axbench-tr.licence-evidence/1",
        "access_date": date.today().isoformat(),
        "note": "Evidence for a human reader, not legal advice; hashes identify the bytes read.",
        "entries": entries,
    }


def write() -> Path:
    EVIDENCE_PATH.write_text(
        json.dumps(collect(), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return EVIDENCE_PATH
