"""Adapters that turn external examples into AXBENCH-TR records (ADR-style rules in DATASET_CARD).

Only two kinds of external content are reused, both from accepted sources:

* ``SRC-N01`` (Cognipeer/synthetic-text-to-sql-tr, Apache-2.0): the *table definition* and the
  *question* of rows whose SQL maps losslessly onto Compare (one GROUP BY with one aggregate),
  Inspect (one aggregate over one column) or REJECT ``data_modification`` (INSERT/UPDATE/DELETE).
  The SQL is parsed only to derive the gold plan and is never stored, taught or emitted.
* ``SRC-N03`` (AlicanKiraz0/Turkish-Finance-SFT-Dataset): the *user questions* only, as Turkish
  ``out_of_scope`` negatives.

Only the registered training-side files are read; no external test split exists in the raw
directory.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from axiom_router.contract import CANDIDATE_VERSION
from axiom_router.schema import project_columns

from .fetch import raw_dir
from .schemas import RealTable, table_partition, usable
from .synth import CORPUS_VERSION, normalize_request

COGNIPEER_FILE = (
    "hf/Cognipeer/synthetic-text-to-sql-tr/99734807e190/data/train-00000-of-00001.parquet"
)
FINANCE_FILE = "hf/AlicanKiraz0/Turkish-Finance-SFT-Dataset/d106acf808bf/finance_sft_dataset.jsonl"

_CREATE = re.compile(r"CREATE TABLE(?: IF NOT EXISTS)?\s+([\w.\"]+)\s*\((.*?)\)\s*;", re.S | re.I)
_GROUP = re.compile(
    r"^SELECT\s+(\w+)\s*,\s*(AVG|SUM|MIN|MAX)\(\s*(\w+)\s*\)(?:\s+AS\s+\w+)?"
    r"\s+FROM\s+([\w.]+)\s+GROUP BY\s+(\w+)\s*;?$",
    re.I,
)
_AGG = re.compile(
    r"^SELECT\s+(AVG|SUM|MIN|MAX)\(\s*(\w+)\s*\)(?:\s+AS\s+\w+)?\s+FROM\s+([\w.]+)\s*;?$", re.I
)
_MOD = re.compile(r"^\s*(INSERT|UPDATE|DELETE)\b", re.I)
_NUMERIC_TYPES = re.compile(
    r"^(INT|INTEGER|BIGINT|SMALLINT|TINYINT|DECIMAL|NUMERIC|FLOAT|DOUBLE|REAL|NUMBER)\b", re.I
)
_SKIP_COLUMN = re.compile(r"^(PRIMARY|FOREIGN|UNIQUE|CONSTRAINT|CHECK|KEY|INDEX)\b", re.I)
_ECON_DOMAIN = (
    "financ",
    "bank",
    "insur",
    "econom",
    "trade",
    "retail",
    "commerce",
    "real estate",
    "invest",
    "tax",
    "wage",
    "labor",
    "labour",
    "employ",
)
ANALYSIS_WORDS = re.compile(
    r"korelasyon|ortalama|medyan|küme|regresyon|tablo|sütun|veri seti|veri kümesi|dataset"
    r"|istatistik|dağılım|csv",
    re.I,
)


def _split_columns(body: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in body:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return [*parts, cur]


def parse_create(context: str) -> tuple[str, list[tuple[str, str]]] | None:
    """First CREATE TABLE of an SQL context: (bare table name, [(column, kind)])."""
    m = _CREATE.search(context)
    if not m:
        return None
    name = m.group(1).strip('"').split(".")[-1].lower()
    cols = []
    for part in _split_columns(m.group(2)):
        bits = part.strip().split(None, 1)
        if len(bits) != 2 or _SKIP_COLUMN.match(bits[0]):
            continue
        col = bits[0].strip('"`[]')
        if not re.fullmatch(r"[A-Za-z_]\w*", col):
            return None  # a malformed or unusual definition: skip the whole table, never guess
        cols.append((col, "numeric" if _NUMERIC_TYPES.match(bits[1]) else "text"))
    return name, cols


def _domain_split(domain: str) -> str:
    digest = hashlib.sha256(f"axbench-tr/0.2/external-domain/{domain}".encode()).digest()
    return "validation" if digest[0] < 51 else "train"


def _economic(domain: str) -> str:
    d = domain.lower()
    return "economic" if any(k in d for k in _ECON_DOMAIN) else "non_economic"


def cognipeer_records(want: dict[str, int], seed: str) -> dict[str, list[dict[str, Any]]]:
    """Adapted records keyed by split. ``want``: counts of {compare, inspect, data_modification}
    Turkish examples; one quarter as many English ones are drawn from different rows."""
    import pyarrow.parquet as pq  # optional dependency, needed only to rebuild the corpus

    rows = pq.read_table(raw_dir() / COGNIPEER_FILE).to_pylist()
    rng = random.Random(f"{seed}/cognipeer")
    rng.shuffle(rows)
    pools: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        "compare": [],
        "inspect": [],
        "data_modification": [],
    }
    for row in rows:
        parsed = parse_create(row["sql_context"])
        sql = row["sql"].strip()
        if not parsed:
            continue
        table, cols = parsed
        kinds = dict(cols)
        if not 2 <= len(cols) <= 64 or len({c for c, _ in cols}) != len(cols):
            continue
        if (
            (m := _GROUP.match(sql))
            and m.group(1).lower() == m.group(5).lower()
            and m.group(4).split(".")[-1].lower() == table
        ):
            g, m_col = m.group(1), m.group(3)
            if kinds.get(g) == "text" and kinds.get(m_col) == "numeric" and g != m_col:
                pools["compare"].append(
                    (
                        row,
                        {
                            "op": "compare",
                            "group_by": g,
                            "measure": m_col,
                            "agg": m.group(2).upper(),
                            "cols": cols,
                        },
                    )
                )
        elif (m := _AGG.match(sql)) and m.group(3).split(".")[-1].lower() == table:
            if kinds.get(m.group(2)) == "numeric":
                pools["inspect"].append(
                    (
                        row,
                        {
                            "op": "inspect",
                            "column": m.group(2),
                            "agg": m.group(1).upper(),
                            "cols": cols,
                        },
                    )
                )
        elif _MOD.match(sql):
            pools["data_modification"].append((row, {"op": "data_modification", "cols": cols}))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen: set[str] = set()
    for frame, n_tr in want.items():
        picked = pools[frame]
        n_en = n_tr // 4
        for lang, count, start in (("tr", n_tr, 0), ("en", n_en, n_tr * 3)):  # disjoint row ranges
            taken = 0
            for row, basis in picked[start:]:
                if taken >= count:
                    break
                text = (row["tr_prompt"] if lang == "tr" else row["sql_prompt"]).strip()
                if not 12 <= len(text) <= 260 or (
                    lang == "tr" and text == row["sql_prompt"].strip()
                ):
                    continue
                key = normalize_request(text)
                if key in seen:
                    continue
                seen.add(key)
                out[_domain_split(row["domain"])].append(
                    _cognipeer_record(row, basis, lang, text, frame)
                )
                taken += 1
    return out


def _cognipeer_record(
    row: dict[str, Any], basis: dict[str, Any], lang: str, text: str, frame: str
) -> dict[str, Any]:
    cols: list[tuple[str, str]] = basis["cols"]
    view = project_columns(cols)
    ids = {name: f"c{i + 1}" for i, (name, _) in enumerate(cols)}
    head = {"version": CANDIDATE_VERSION}
    if frame == "compare":
        gold: dict[str, Any] = {
            **head,
            "action": "PLAN",
            "operation": "compare",
            "group_by": ids[basis["group_by"]],
            "measure": ids[basis["measure"]],
        }
        derivation = {
            "rule": "one GROUP BY with one aggregate -> compare",
            "aggregate": basis["agg"],
        }
    elif frame == "inspect":
        gold = {**head, "action": "PLAN", "operation": "inspect", "columns": [ids[basis["column"]]]}
        derivation = {
            "rule": "one aggregate over one column -> inspect that column",
            "aggregate": basis["agg"],
        }
    else:
        gold = {**head, "action": "REJECT", "reason": "data_modification"}
        derivation = {"rule": "INSERT/UPDATE/DELETE -> reject data_modification"}
    rid = int(row["id"])
    return {
        "id": f"t2-x-cog-{rid}-{lang}",
        "corpus_version": CORPUS_VERSION,
        "split": _domain_split(row["domain"]),
        "stratum": "external_adapted",
        "language": lang,
        "family": f"cognipeer/{row['domain']}",
        "domain": _economic(row["domain"]),
        "schema_id": f"cognipeer/{rid}",
        "lineage_id": f"X-cog-{rid}",
        "frame": "reject:data_modification" if frame == "data_modification" else frame,
        "request": text,
        "schema": view.to_prompt_payload(),
        "gold": gold,
        "accepted_variants": [],
        "tags": sorted({"machine_translated"} if lang == "tr" else {"original_english_synthetic"}),
        "provenance": {
            "origin": "external_adapted",
            "source_id": "SRC-N01",
            "upstream_row_id": rid,
            "upstream_license": "Apache-2.0",
            "request_authoring": "machine_translation_of_llm_generated_prompt"
            if lang == "tr"
            else "llm_generated_prompt",
            "schema_origin": "synthetic_create_table_context",
            "gold_derivation": derivation,
            "human_review": "none",
            "generator": "axbench_tr.adapters/0.2",
        },
    }


def finance_negatives(
    tables: list[RealTable], count: int, seed: str
) -> dict[str, list[dict[str, Any]]]:
    """Turkish out_of_scope examples from user questions of SRC-N03, paired with real table
    schemas."""
    path = raw_dir() / FINANCE_FILE
    rng = random.Random(f"{seed}/finance")
    questions: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text("utf-8").splitlines()):
        q = " ".join(json.loads(line)["user"].split())
        q = q.replace('\\"', '"').replace("\\'", "'").replace("**", "")
        if (
            25 <= len(q) <= 200
            and q.endswith("?")
            and not ANALYSIS_WORDS.search(q)
            and "<" not in q
            and "http" not in q
        ):
            questions.append((i, q))
    rng.shuffle(questions)
    pools = {
        s: [t for t in tables if table_partition(t) == s and usable(t)]
        for s in ("train", "validation")
    }
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen: set[str] = set()
    for i, q in questions:
        if sum(map(len, out.values())) >= count:
            break
        key = normalize_request(q)
        if key in seen:
            continue
        seen.add(key)
        split = _domain_split(f"fin-{i}")
        table = rng.choice(pools[split])
        view = project_columns([(c.name, c.kind) for c in table.columns])
        out[split].append(
            {
                "id": f"t2-x-fin-{i}",
                "corpus_version": CORPUS_VERSION,
                "split": split,
                "stratum": "external_adapted",
                "language": "tr",
                "family": table.group,
                "domain": table.domain,
                "schema_id": table.table_id,
                "lineage_id": f"X-fin-{i}",
                "frame": "reject:out_of_scope",
                "request": q,
                "schema": view.to_prompt_payload(),
                "gold": {
                    "version": CANDIDATE_VERSION,
                    "action": "REJECT",
                    "reason": "out_of_scope",
                },
                "accepted_variants": [],
                "tags": ["llm_generated_question"],
                "provenance": {
                    "origin": "external_adapted",
                    "source_id": "SRC-N03",
                    "upstream_row_index": i,
                    "request_authoring": "llm_generated_question",
                    "schema_origin": "real_table_headers_localized"
                    if table.localized
                    else "real_table_headers",
                    "schema_source_id": table.source_id,
                    "human_review": "none",
                    "generator": "axbench_tr.adapters/0.2",
                },
            }
        )
    return out
