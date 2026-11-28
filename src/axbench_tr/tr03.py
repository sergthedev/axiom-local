"""EXP-TR03 preparation tooling, data side (``python -m axbench_tr.tr03 <command>``).

Status: **prepared, deterministic, nothing trained.** This module reads only the committed corpus
``data/axbench-tr/v0.2-rev1/`` (train and validation, plus the per-example token file and the
committed source records) and writes ``data/axbench-tr/tr03-prep/``. It loads no model, scores no
system, and never opens a test split (none exists in this corpus; the sealed AXBENCH test is not
referenced).

What it does:

* ``inventory``: which records are eligible for EXP-TR03 under the rights rule (project-authored
  template text on table layouts from SRC-O01, SRC-O02 or SRC-O06 only), what is excluded and why,
  the 122 templates, distributions, and token and loss-mask statistics.
* ``audit``: independent train and validation checks on the eligible records (shared lineages,
  table families, schema fingerprints, request texts, lexical near duplicates, overlap with
  AXBENCH v0.1 requests), gold consistency, descriptive shortcut statistics, and which licensed
  tables are available for independently authored held-out tasks.
* ``worklist``: empty human worklists (template review, stratified row review, rights, held-out
  schema candidates). No cell for a human decision is filled; an AI judgement is never recorded
  as one.
* ``sizing``: exact binomial sizing tables for the proposed review and evaluation sizes.
* ``gates`` (``tr03_gates``): validates filled-in ledgers (if any) against the frozen worklists and
  reports each gate as absent, invalid, incomplete or consistent-but-unverified. It never opens
  execution and never declares research quality.

Everything written is a pure function of the committed inputs (sorted keys, LF endings).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from axbench.audit import gold_problems
from axbench.stats import wilson

from . import contamination
from .build import CURRENT_VERSION, OUT_ROOT
from .export import CLEARED_ORIGIN, RIGHTS_VERIFIED_SCHEMA_SOURCES
from .fetch import sha256_bytes
from .registry import REGISTRY_PATH
from .synth import normalize_request
from .templates import LINEAGES
from .validate import schema_fingerprint

PREP_DIR = OUT_ROOT / "tr03-prep"
SCHEMAS_PATH = OUT_ROOT / "schemas" / "real_table_schemas.json"
ADDENDUM_PATH = OUT_ROOT / "sources" / "LICENCE_EVIDENCE_ADDENDUM_1.json"
ELIGIBLE_ORIGIN = CLEARED_ORIGIN
ELIGIBLE_SOURCES = tuple(RIGHTS_VERIFIED_SCHEMA_SOURCES)
REVIEW_SEED = "axbench-tr-0.2-rev1-tr03-review"
MAX_SEQ_LENGTH = 2048  # the sequence cap proposed in training/tr03/config.json (TR02 value)
SOFT_LENGTH = 1024
PAD_TO = 32

# Proposed (not approved) review sizes; the arithmetic is in ``sizing``.
TR_ROWS_PER_STRATUM = 59  # smallest n at which zero failures bound the failure rate below 5%
EN_ROWS_PER_STRATUM = 15  # English is reported, not gated on its own
SECOND_REVIEW_FRACTION = 0.10
FRAME_CLASSES = ("inspect", "inspect_all", "relate", "compare", "cluster", "clarify", "reject")
HIGH_RISK_FRAMES = (
    "reject:instruction_override",
    "reject:code_or_system_request",
    "clarify:ambiguous_column",
    "clarify:column_type_mismatch",
    "clarify:invalid_parameter",
)

# Proposed gates (technical lead, comments 7222653458 and 7221592809; owner confirmation pending).
GOLD_CORRECT_MIN = 0.95
NATURAL_OR_FIXABLE_MIN = 0.90
GATE_MIN_N = TR_ROWS_PER_STRATUM


# ------------------------------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def load(
    version: str = CURRENT_VERSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Train and validation records, per-example token rows by id, and the SHA-256 of each input."""
    root = OUT_ROOT / version
    train = read_jsonl(root / "train.jsonl")
    valid = read_jsonl(root / "validation.jsonl")
    per_path = root / "token_lengths_per_example.jsonl"
    per = {r["id"]: r for r in read_jsonl(per_path)}
    hashes = {
        f"{version}/{p.name}": sha256_bytes(p.read_bytes())
        for p in (root / "train.jsonl", root / "validation.jsonl", per_path)
    }
    return train, valid, per, hashes


def frame_class(frame: str) -> str:
    head = frame.split(":")[0]
    return head if head in ("clarify", "reject") else frame


def is_eligible(rec: dict[str, Any]) -> tuple[bool, str]:
    """The rights rule: this project's own template text on a layout from a documented source."""
    prov = rec["provenance"]
    if prov["origin"] != ELIGIBLE_ORIGIN:
        return False, f"origin:{prov['origin']}"
    source = prov.get("schema_source_id")
    if source not in ELIGIBLE_SOURCES:
        return False, f"schema_source:{source}"
    if "translation_unverified" in rec.get("tags", []):
        return False, "tag:translation_unverified"
    return True, ""


def split_eligible(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    keep: list[dict[str, Any]] = []
    why: Counter[str] = Counter()
    for r in rows:
        ok, reason = is_eligible(r)
        if ok:
            keep.append(r)
        else:
            why[reason] += 1
    return keep, why


def padded_tokens(n: int, max_seq_length: int = MAX_SEQ_LENGTH, pad_to: int = PAD_TO) -> int:
    """Same rule as ``training/tr02/tr02.py``: tokens the model computes on for one row."""
    return min(1 + pad_to * ((n + pad_to - 1) // pad_to), max_seq_length) - 1


# --------------------------------------------------------------------------------- statistics


def _nearest_rank(values: Sequence[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": _nearest_rank(values, 0.5),
        "p90": _nearest_rank(values, 0.9),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
        "min": min(values),
    }


def token_stats(rows: Sequence[dict[str, Any]], per: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Full-sequence, prompt and supervised-answer token counts (per-example file, pinned)."""
    seq = [int(per[r["id"]]["separate_tokens"]) for r in rows]
    prompt = [int(per[r["id"]]["prompt_tokens"]) for r in rows]
    answer = [int(per[r["id"]]["completion_tokens"]) for r in rows]
    padded = [padded_tokens(n) for n in seq]
    return {
        "full_sequence_tokens": summary(seq),
        "prompt_tokens": summary(prompt),
        "supervised_answer_tokens": summary(answer),
        "padded_compute_tokens_total": sum(padded),
        "rows_over_1024": sum(n > SOFT_LENGTH for n in seq),
        "rows_over_2048": sum(n > MAX_SEQ_LENGTH for n in seq),
        "supervised_share_of_full_tokens": round(sum(answer) / max(1, sum(seq)), 5),
        "boundary_straddle_rows": sum(bool(per[r["id"]].get("boundary_straddle")) for r in rows),
    }


def distributions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def count(key) -> dict[str, int]:
        return dict(sorted(Counter(key(r) for r in rows).items()))

    return {
        "records": len(rows),
        "language": count(lambda r: r["language"]),
        "action": count(lambda r: r["gold"]["action"]),
        "language_x_action": count(lambda r: f"{r['language']}:{r['gold']['action']}"),
        "frame": count(lambda r: r["frame"]),
        "frame_class": count(lambda r: frame_class(r["frame"])),
        "domain": count(lambda r: r["domain"]),
        "schema_source": count(lambda r: r["provenance"]["schema_source_id"]),
        "header_language": count(lambda r: r["provenance"]["header_language"]),
        "request_authoring": count(lambda r: r["provenance"]["request_authoring"]),
        "human_review": count(lambda r: r["provenance"]["human_review"]),
        "tags": dict(sorted(Counter(t for r in rows for t in r.get("tags", [])).items())),
        "lineages": len({r["lineage_id"] for r in rows}),
        "table_families": len({r["family"] for r in rows}),
        "schema_ids": len({r["schema_id"] for r in rows}),
        "schema_fingerprints": len({schema_fingerprint(r) for r in rows}),
    }


def concentration(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How unevenly records spread over templates and table families (diversity, not quality)."""
    by_lineage = Counter(r["lineage_id"] for r in rows)
    by_family = Counter(r["family"] for r in rows)
    n = max(1, len(rows))
    lineage_counts = sorted(by_lineage.values(), reverse=True)
    family_counts = sorted(by_family.values(), reverse=True)
    return {
        "records_per_template": summary(lineage_counts),
        "top_10_templates_share": round(sum(lineage_counts[:10]) / n, 4),
        "records_per_table_family": summary(family_counts),
        "top_1_table_family": {
            "family": by_family.most_common(1)[0][0] if by_family else None,
            "share": round(family_counts[0] / n, 4) if family_counts else 0.0,
        },
        "top_5_table_families_share": round(sum(family_counts[:5]) / n, 4),
        "effective_templates_inverse_simpson": round(
            1 / sum((c / n) ** 2 for c in lineage_counts), 2
        )
        if lineage_counts
        else 0.0,
    }


# ------------------------------------------------------------------------------------ inventory


def template_table(
    rows: Sequence[dict[str, Any]], per: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One entry per project-authored template (lineage) with its eligible records."""
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lineage[r["lineage_id"]].append(r)
    out = []
    for lin in sorted(LINEAGES, key=lambda x: x.id):
        recs = by_lineage.get(lin.id, [])
        if not recs:
            continue
        lang = Counter(r["language"] for r in recs)
        out.append(
            {
                "lineage_id": lin.id,
                "frame": lin.frame,
                "split": recs[0]["split"],
                "records": len(recs),
                "records_tr": lang.get("tr", 0),
                "records_en": lang.get("en", 0),
                "schema_sources": sorted({r["provenance"]["schema_source_id"] for r in recs}),
                "table_families": len({r["family"] for r in recs}),
                "mean_full_tokens": round(
                    sum(int(per[r["id"]]["separate_tokens"]) for r in recs) / len(recs), 2
                ),
                "template_text_sha256": sha256_bytes(
                    json.dumps([lin.tr, lin.en], ensure_ascii=False).encode("utf-8")
                ),
                "authored_by": "ai_tool (implementing assistant); no human wrote or reviewed it",
            }
        )
    return out


def inventory(version: str = CURRENT_VERSION) -> dict[str, Any]:
    train, valid, per, hashes = load(version)
    elig_t, why_t = split_eligible(train)
    elig_v, why_v = split_eligible(valid)
    return {
        "schema": "axbench-tr.tr03-inventory/1",
        "status": "measured counts of a draft corpus; AI-authored, no human review of any record",
        "corpus_version": f"axbench-tr/{version.removeprefix('v')}",
        "input_sha256": hashes,
        "eligibility_rule": {
            "origin": ELIGIBLE_ORIGIN,
            "schema_sources": list(ELIGIBLE_SOURCES),
            "excluded_always": [
                "external_adapted records (upstream terms unresolved)",
                "translation_unverified or machine-translated text",
                "layouts from SRC-O03 (ECB), SRC-O04 (OECD), SRC-O05 (TCMB): terms unverified",
                "any record of AXBENCH v0.1 (comparator evidence) and the sealed AXBENCH test",
            ],
            "meaning": "rights-documented for resource use only; not a legal clearance; not a "
            "human-reviewed corpus (see PROTOCOL.md, gates G1 to G6)",
        },
        "totals": {
            "corpus": {"train": len(train), "validation": len(valid)},
            "eligible": {"train": len(elig_t), "validation": len(elig_v)},
            "excluded_by_reason": {
                "train": dict(sorted(why_t.items())),
                "validation": dict(sorted(why_v.items())),
            },
        },
        "eligible_train": distributions(elig_t),
        "eligible_validation": distributions(elig_v),
        "eligible_train_concentration": concentration(elig_t),
        "eligible_validation_concentration": concentration(elig_v),
        "tokens": {
            "train": token_stats(elig_t, per),
            "validation": token_stats(elig_v, per),
        },
        "human_reviewed_records": {
            "train": sum(r["provenance"]["human_review"] != "none" for r in elig_t),
            "validation": sum(r["provenance"]["human_review"] != "none" for r in elig_v),
        },
        "templates": template_table(elig_t + elig_v, per),
        "source_attribution": source_attribution(),
    }


def source_attribution() -> list[dict[str, Any]]:
    """Per accepted schema source: publisher, licence id, evidence ids and open caveats."""
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))["sources"]
    by_id = {s["id"]: s for s in registry}
    addendum = json.loads(ADDENDUM_PATH.read_text("utf-8"))
    entries: dict[str, list[str]] = defaultdict(list)
    for e in addendum["entries"]:
        entries[e["source"]].append(e["id"])
    caveats = {c.split(":", 1)[0]: c for c in addendum["limits"] if c.startswith("SRC-")}
    out = []
    for sid in ELIGIBLE_SOURCES:
        s = by_id[sid]
        out.append(
            {
                "source_id": sid,
                "name": s["name"],
                "publisher": s["publisher"],
                "url": s["url"],
                "licence_id": s["license"]["id"],
                "licence_evidence_in_registry": s["license"]["evidence"],
                "addendum_evidence_ids": sorted(entries[sid]),
                "open_caveat": caveats.get(sid, ""),
                "what_reaches_examples": "table and column names only; no observations",
            }
        )
    return out


# ---------------------------------------------------------------------------------------- audit


def _near_pairs(queries: list[str], refs: list[str], threshold: float) -> int:
    return len(contamination.near_duplicates(queries, refs, threshold))


def split_audit(
    elig_t: Sequence[dict[str, Any]], elig_v: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Independence of the eligible train and validation records, measured four ways."""

    def keys(rows, fn) -> set[str]:
        return {fn(r) for r in rows}

    kinds = {
        "lineages": lambda r: r["lineage_id"],
        "template_texts": lambda r: next(
            (
                lin.tr if r["language"] == "tr" else lin.en
                for lin in LINEAGES
                if lin.id == r["lineage_id"]
            ),
            "",
        ),
        "table_families": lambda r: r["family"],
        "table_ids": lambda r: r["schema_id"],
        "schema_fingerprints": schema_fingerprint,
        "normalized_requests": lambda r: normalize_request(r["request"]),
    }
    shared = {k: len(keys(elig_t, fn) & keys(elig_v, fn)) for k, fn in kinds.items()}
    req_t = [r["request"] for r in elig_t]
    req_v = [r["request"] for r in elig_v]
    v01 = contamination.load_v01_requests()
    return {
        "shared_between_train_and_validation": shared,
        "lexical_near_duplicate_pairs_validation_vs_train": {
            "char_4gram_jaccard_ge_0.8": _near_pairs(req_v, req_t, 0.8),
            "char_4gram_jaccard_ge_0.9": _near_pairs(req_v, req_t, 0.9),
            "note": "lexical proxy only; not semantic deduplication",
        },
        "overlap_with_axbench_v0.1_train_and_validation_requests": {
            "train": contamination.request_report(req_t, v01),
            "validation": contamination.request_report(req_v, v01),
            "note": "v0.1 train and validation are public comparator data; the sealed test was "
            "not read",
        },
        "ids_unique": len({r["id"] for r in (*elig_t, *elig_v)}) == len(elig_t) + len(elig_v),
    }


def _gold_columns(gold: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    for key in ("x", "y", "group_by", "measure"):
        if key in gold:
            cols.append(gold[key])
    for key in ("features", "columns"):
        cols += list(gold.get(key, []))
    return cols


def _label_by_name(rec: dict[str, Any]) -> str:
    """The gold label with column ids replaced by names; column lists (an unordered set in the
    plan semantics) are sorted."""
    names = {c["id"]: c["name"] for c in rec["schema"]["columns"]}
    out: dict[str, Any] = {}
    for key, value in rec["gold"].items():
        if key in ("columns", "features"):
            out[key] = sorted(names[c] for c in value)
        elif key in ("x", "y", "group_by", "measure"):
            out[key] = names[value]
        else:
            out[key] = value
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def shortcut_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive statistics that could let a model succeed without reading the request.

    These are shares in the data, not results of any system.
    """
    plan = [r for r in rows if r["gold"]["action"] == "PLAN" and r["frame"] != "inspect_all"]
    first_numeric = 0
    first_text = 0
    numeric_total = 0
    text_total = 0
    verbatim = 0
    for r in plan:
        cols = r["schema"]["columns"]
        kind = {c["id"]: c["type"] for c in cols}
        numeric_ids = [c["id"] for c in cols if c["type"] == "numeric"]
        text_ids = [c["id"] for c in cols if c["type"] == "text"]
        names = {c["id"]: c["name"] for c in cols}
        gold = r["gold"]
        for key in ("x", "y", "measure"):
            if key in gold and kind.get(gold[key]) == "numeric":
                numeric_total += 1
                first_numeric += gold[key] == numeric_ids[0]
        if "group_by" in gold and kind.get(gold["group_by"]) == "text":
            text_total += 1
            first_text += gold["group_by"] == text_ids[0]
        request = normalize_request(r["request"])
        used = _gold_columns(gold)
        verbatim += bool(used) and all(
            normalize_request(names[c]) and normalize_request(names[c]) in request for c in used
        )
    by_action_len: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_action_len[r["gold"]["action"]].append(len(r["request"]))
    # The schema fingerprint ignores column order, so two records with the same request and layout
    # can number their columns differently; compare the label by column *names* (readable_gold).
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(normalize_request(r["request"]), schema_fingerprint(r))].append(r)
    id_numbering_differs = sum(
        len({json.dumps(r["gold"], sort_keys=True) for r in v}) > 1 for v in by_key.values()
    )
    name_label_conflicts = sum(len({_label_by_name(r) for r in v}) > 1 for v in by_key.values())
    request_counts = Counter(normalize_request(r["request"]) for r in rows)
    n = len(rows)
    actions = Counter(r["gold"]["action"] for r in rows)
    frames = Counter(r["frame"] for r in rows)
    return {
        "plan_rows_considered": len(plan),
        "share_of_numeric_gold_columns_that_are_the_first_numeric_column": round(
            first_numeric / max(1, numeric_total), 4
        ),
        "share_of_gold_group_columns_that_are_the_first_text_column": round(
            first_text / max(1, text_total), 4
        ),
        "share_of_plan_requests_naming_every_gold_column_verbatim": round(
            verbatim / max(1, len(plan)), 4
        ),
        "largest_action_share": round(max(actions.values()) / max(1, n), 4),
        "largest_frame_share": round(max(frames.values()) / max(1, n), 4),
        "mean_request_characters_by_action": {
            a: round(sum(v) / len(v), 1) for a, v in sorted(by_action_len.items())
        },
        "same_request_and_layout_with_different_gold_by_column_name": name_label_conflicts,
        "same_request_and_layout_differing_only_in_column_id_numbering": id_numbering_differs
        - name_label_conflicts,
        "records_whose_normalized_request_repeats_an_earlier_record": sum(
            c - 1 for c in request_counts.values()
        ),
    }


def gold_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    problems = Counter()
    for r in rows:
        for p in gold_problems(r):
            problems[p] += 1
    return {"records_checked": len(rows), "problems": dict(sorted(problems.items()))}


def table_capabilities(columns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(c["kind"] for c in columns)
    numeric, text = kinds.get("numeric", 0), kinds.get("text", 0)
    return {
        "numeric_columns": numeric,
        "text_columns": text,
        "supports_inspect": numeric + text >= 1,
        "supports_relate": numeric >= 2,
        "supports_compare": numeric >= 1 and text >= 1,
        "supports_cluster": numeric >= 2,
    }


def heldout_candidates(
    corpus_records: Sequence[dict[str, Any]], train_ids: set[str]
) -> list[dict[str, Any]]:
    """Licensed tables (accepted sources) with how they are used by the committed corpus.

    ``role`` says whether the layout appears in an eligible train record, only in validation or
    other records, or in no record at all. Independent authors should draw from tables whose
    layout the training rows never show.
    """
    used_splits: dict[str, set[str]] = defaultdict(set)
    seen_names: set[str] = set()
    for r in corpus_records:
        used_splits[r["schema_id"]].add(r["split"])
        if r["schema_id"] in train_ids:
            seen_names |= {normalize_request(c["name"]) for c in r["schema"]["columns"]}
    tables = json.loads(SCHEMAS_PATH.read_text("utf-8"))["tables"]
    out = []
    for t in sorted(tables, key=lambda t: t["table_id"]):
        if t["source_id"] not in ELIGIBLE_SOURCES:
            continue
        splits = used_splits.get(t["table_id"], set())
        if t["table_id"] in train_ids:
            role = "in_eligible_train_records"
        elif splits:
            role = "validation_only_or_other_records"
        else:
            role = "in_no_record"
        out.append(
            {
                "table_id": t["table_id"],
                "source_id": t["source_id"],
                "domain": t["domain"],
                "header_language": t["header_language"],
                "title": t["title"],
                "role": role,
                "columns_seen_in_train_share": round(
                    sum(normalize_request(c["name"]) in seen_names for c in t["columns"])
                    / max(1, len(t["columns"])),
                    3,
                ),
                **table_capabilities(t["columns"]),
            }
        )
    return out


def column_vocabulary_overlap(
    elig_t: Sequence[dict[str, Any]],
    elig_v: Sequence[dict[str, Any]],
    tables_in_no_record: Sequence[str],
) -> dict[str, Any]:
    """How much of the held-out column vocabulary the training rows already show.

    A new table layout with the same column names is a weak held-out test of schema handling. Names
    are folded with the corpus normaliser; shares are over distinct names per source.
    """
    seen = {normalize_request(c["name"]) for r in elig_t for c in r["schema"]["columns"]}
    by_source: dict[str, set[str]] = defaultdict(set)
    for r in elig_v:
        for c in r["schema"]["columns"]:
            by_source[r["provenance"]["schema_source_id"]].add(normalize_request(c["name"]))
    val = {
        src: {
            "distinct_names": len(names),
            "seen_in_train_schemas": sum(n in seen for n in names),
            "share": round(sum(n in seen for n in names) / len(names), 4),
        }
        for src, names in sorted(by_source.items())
    }
    wanted = set(tables_in_no_record)
    tables = [
        t for t in json.loads(SCHEMAS_PATH.read_text("utf-8"))["tables"] if t["table_id"] in wanted
    ]
    shares = [
        sum(normalize_request(c["name"]) in seen for c in t["columns"]) / len(t["columns"])
        for t in tables
        if t["columns"]
    ]
    return {
        "validation_column_names_seen_in_train_schemas_by_source": val,
        "tables_in_no_record": {
            "tables": len(tables),
            "mean_share_of_columns_seen_in_train": round(sum(shares) / max(1, len(shares)), 4),
            "tables_with_no_column_seen_in_train": sum(x == 0 for x in shares),
        },
    }


def audit(version: str = CURRENT_VERSION) -> dict[str, Any]:
    train, valid, per, hashes = load(version)
    elig_t, _ = split_eligible(train)
    elig_v, _ = split_eligible(valid)
    train_tables = {r["schema_id"] for r in elig_t}
    cands = heldout_candidates([*train, *valid], train_tables)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    for c in cands:
        by_role[c["role"]][f"{c['source_id']}:{c['domain']}"] += 1
    unused = [c for c in cands if c["role"] == "in_no_record"]
    return {
        "schema": "axbench-tr.tr03-audit/1",
        "input_sha256": hashes,
        "split_independence": split_audit(elig_t, elig_v),
        "gold_consistency": {
            "train": gold_audit(elig_t),
            "validation": gold_audit(elig_v),
        },
        "shortcut_statistics": {
            "train": shortcut_audit(elig_t),
            "validation": shortcut_audit(elig_v),
        },
        "heldout_schema_availability": {
            "licensed_tables_by_role": {
                role: dict(sorted(c.items())) for role, c in sorted(by_role.items())
            },
            "tables_in_no_record": len(unused),
            "tables_in_no_record_that_support_compare": sum(c["supports_compare"] for c in unused),
            "tables_in_no_record_that_support_relate": sum(c["supports_relate"] for c in unused),
            "tables_in_no_record_economic_domain": sum(c["domain"] == "economic" for c in unused),
            "column_vocabulary_overlap": column_vocabulary_overlap(
                elig_t, elig_v, [c["table_id"] for c in unused]
            ),
            "note": "a layout absent from every corpus record is the cleanest held-out schema, but "
            "authors still see only column names and kinds; economic coverage is the limit",
        },
        "limitations": [
            "lexical near-duplicate counts are a proxy, not semantic deduplication",
            "shortcut statistics describe the data; they are not results of any system",
            "gold consistency checks the label against the contract, not against human judgement",
            "the eligible records are AI-authored; no human has judged any record",
        ],
    }


# -------------------------------------------------------------------------------------- sizing


def zero_failure_n(max_failure_rate: float, alpha: float = 0.05) -> int:
    """Smallest n such that zero failures in n bound the failure rate below ``max_failure_rate``
    at one-sided confidence 1 - alpha: (1 - p) ** n <= alpha."""
    return math.ceil(math.log(alpha) / math.log(1 - max_failure_rate))


def wilson_half_width(p: float, n: int) -> float:
    lo, hi = wilson(round(p * n), n) or (0.0, 0.0)
    return round((hi - lo) / 2, 4)


def sizing() -> dict[str, Any]:
    ns = (30, 59, 100, 150, 200, 300, 400)
    return {
        "schema": "axbench-tr.tr03-sizing/1",
        "method": "exact arithmetic; Wilson score interval from axbench.stats.wilson",
        "zero_failure_sample_size": {
            "gold_correct_gate_failure_rate_0.05": zero_failure_n(0.05),
            "natural_or_fixable_gate_failure_rate_0.10": zero_failure_n(0.10),
            "meaning": "with zero failures in n reviewed rows, the one-sided 95% upper bound of "
            "the failure rate is at most the stated rate; any failure needs a larger sample",
        },
        "wilson_half_width_at_p_0.5": {str(n): wilson_half_width(0.5, n) for n in ns},
        "wilson_half_width_at_p_0.1": {str(n): wilson_half_width(0.1, n) for n in ns},
        "proposed_review_sizes": {
            "turkish_rows_per_stratum": TR_ROWS_PER_STRATUM,
            "english_rows_per_stratum": EN_ROWS_PER_STRATUM,
            "strata": list(FRAME_CLASSES),
            "second_review_fraction_of_rest": SECOND_REVIEW_FRACTION,
            "second_review_also": "every row a first reviewer marks gold_correct no or "
            "serious_error yes",
        },
        "proposed_heldout_evaluation_size": {
            "turkish_plan": 120,
            "turkish_clarify": 90,
            "turkish_reject": 90,
            "turkish_total": 300,
            "english_secondary": 100,
            "near_miss_subset_within_turkish": 60,
            "basis": "at least 30 per supported operation and 10 per CLARIFY or REJECT reason; "
            "the owner decides the final numbers (HELDOUT_AUTHORING_KIT.md)",
        },
    }


# ------------------------------------------------------------------------------------ worklists

# Human-judgement blocks. Every reviewed item carries a first review, an optional independent
# second review and an optional adjudication by a third person. The judgement fields differ by
# ledger; the block layout is the same so that one checker (``tr03_gates``) validates both.
TEMPLATE_JUDGMENTS = ("tr_natural", "en_natural", "gold_rule_correct", "serious_error", "decision")
ROW_JUDGMENTS = ("request_natural", "gold_correct", "serious_error", "decision")


def review_blocks(
    judgments: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    first = ("reviewer_id", "reviewer_type", "reviewed_on", *judgments, "notes")
    second = (
        "second_reviewer_id",
        "second_reviewer_type",
        "second_reviewed_on",
        *(f"second_{j}" for j in judgments),
        "second_notes",
    )
    adjudication = (
        "adjudicator_id",
        "adjudicator_type",
        "adjudicated_on",
        *(f"adjudicated_{j}" for j in judgments),
        "adjudication_notes",
    )
    return first, second, adjudication


TEMPLATE_IDENTITY = (
    "lineage_id",
    "frame",
    "split",
    "records_eligible",
    "template_tr",
    "template_en",
    "example_record_ids",
    "authored_by",
    "batch",
    "second_review_planned",
)
ROW_IDENTITY = (
    "sample_id",
    "record_id",
    "language",
    "frame",
    "risk",
    "lineage_id",
    "schema_source_id",
    "schema_columns",
    "request",
    "gold_readable",
    "gold_json",
    "batch",
    "second_review_planned",
)
TEMPLATE_COLUMNS = tuple(
    c for block in (TEMPLATE_IDENTITY, *review_blocks(TEMPLATE_JUDGMENTS)) for c in block
)
ROW_COLUMNS = tuple(c for block in (ROW_IDENTITY, *review_blocks(ROW_JUDGMENTS)) for c in block)
RIGHTS_COLUMNS = (
    "source_id",
    "name",
    "licence_id",
    "evidence_ids",
    "retrieved_on",
    "status_in_repository",
    "open_caveat",
    "attribution_required",
    "owner_decision",
    "decided_by",
    "decided_on",
    "notes",
)
HELDOUT_COLUMNS = (
    "table_id",
    "source_id",
    "domain",
    "header_language",
    "role",
    "columns_seen_in_train_share",
    "numeric_columns",
    "text_columns",
    "supports_relate",
    "supports_compare",
    "supports_cluster",
    "author_id",
    "use_for_heldout",
)


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def readable_gold(rec: dict[str, Any]) -> str:
    names = {c["id"]: c["name"] for c in rec["schema"]["columns"]}
    g = rec["gold"]
    if g["action"] == "REJECT":
        return f"REJECT ({g['reason']})"
    if g["action"] == "CLARIFY":
        extra = f" [{g['operation']}]" if "operation" in g else ""
        return f"CLARIFY ({g['reason']}){extra}"
    op = g["operation"]
    if op == "relate":
        return f"relate {g['method']}: {names[g['x']]} ~ {names[g['y']]}"
    if op == "compare":
        return f"compare: {names[g['measure']]} by {names[g['group_by']]}"
    if op == "cluster":
        return f"cluster k={g['k']}: " + ", ".join(names[c] for c in g["features"])
    return "inspect: " + ", ".join(names[c] for c in g["columns"])


def select_row_sample(
    rows: Sequence[dict[str, Any]],
    tr_per: int = TR_ROWS_PER_STRATUM,
    en_per: int = EN_ROWS_PER_STRATUM,
    seed: str = REVIEW_SEED,
) -> list[dict[str, Any]]:
    """Stratified review sample: (language, frame class), seed-ranked, one record per template
    first so that no template dominates; high-risk frames get extra floors inside their stratum
    by taking those templates first."""
    out: list[dict[str, Any]] = []
    for lang, per in (("tr", tr_per), ("en", en_per)):
        for cls in FRAME_CLASSES:
            pool = [r for r in rows if r["language"] == lang and frame_class(r["frame"]) == cls]
            ranked = sorted(
                pool,
                key=lambda r: (
                    r["frame"] not in HIGH_RISK_FRAMES,
                    _rank(seed, r["id"]),
                ),
            )
            chosen: list[dict[str, Any]] = []
            seen: set[str] = set()
            for r in ranked:  # first pass: one record per template
                if len(chosen) < per and r["lineage_id"] not in seen:
                    seen.add(r["lineage_id"])
                    chosen.append(r)
            for r in ranked:  # second pass fills the stratum
                if len(chosen) < per and r not in chosen:
                    chosen.append(r)
            out += chosen
    return out


ROW_BATCHES = {"tr": 8, "en": 2}  # about 52 rows per batch, never mixing the two languages
TEMPLATE_BATCHES = 6  # about 20 templates per batch
IDENTITY_SEED = REVIEW_SEED + "/batch"


def assign_batches(keys: Sequence[str], count: int, prefix: str) -> dict[str, str]:
    """Deterministic, seed-ranked split of ``keys`` into ``count`` near-equal batches. The batch
    is an immutable workload label for the owner to allocate people; it carries no judgement."""
    ranked = sorted(keys, key=lambda k: _rank(IDENTITY_SEED, k))
    return {k: f"{prefix}{i % count + 1:02d}" for i, k in enumerate(ranked)}


def _second_planned(key: str) -> str:
    share = int(_rank(REVIEW_SEED + "/second", key)[:8], 16) / 0xFFFFFFFF
    return "yes" if share < SECOND_REVIEW_FRACTION else ""


def _blank(columns: Sequence[str], first_blank: str) -> dict[str, str]:
    return {c: "" for c in columns[columns.index(first_blank) :]}


def row_worklist(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    sample = sorted(select_row_sample(rows), key=lambda r: (r["language"], r["id"]))
    batch: dict[str, str] = {}
    for lang, n in ROW_BATCHES.items():
        batch.update(
            assign_batches([r["id"] for r in sample if r["language"] == lang], n, f"RW-{lang}-")
        )
    items = []
    for i, r in enumerate(sample, 1):
        items.append(
            {
                "sample_id": f"tr03-{i:04d}",
                "record_id": r["id"],
                "language": r["language"],
                "frame": r["frame"],
                "risk": "high" if r["frame"] in HIGH_RISK_FRAMES else "standard",
                "lineage_id": r["lineage_id"],
                "schema_source_id": r["provenance"]["schema_source_id"],
                "schema_columns": " | ".join(
                    f"{c['id']}:{c['name']}({c['type']})" for c in r["schema"]["columns"]
                ),
                "request": r["request"],
                "gold_readable": readable_gold(r),
                "gold_json": json.dumps(r["gold"], ensure_ascii=False, sort_keys=True),
                "batch": batch[r["id"]],
                "second_review_planned": _second_planned(r["id"]),
                **_blank(ROW_COLUMNS, "reviewer_id"),
            }
        )
    return items


def template_worklist(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lineage[r["lineage_id"]].append(r)
    lineages = [lin for lin in sorted(LINEAGES, key=lambda x: x.id) if by_lineage.get(lin.id)]
    batch = assign_batches([lin.id for lin in lineages], TEMPLATE_BATCHES, "TW-")
    items = []
    for lin in lineages:
        recs = by_lineage[lin.id]
        examples = sorted(recs, key=lambda r: _rank(REVIEW_SEED + "/example", r["id"]))[:2]
        items.append(
            {
                "lineage_id": lin.id,
                "frame": lin.frame,
                "split": recs[0]["split"],
                "records_eligible": str(len(recs)),
                "template_tr": lin.tr,
                "template_en": lin.en,
                "example_record_ids": ";".join(r["id"] for r in examples),
                "authored_by": "ai_tool",
                "batch": batch[lin.id],
                "second_review_planned": _second_planned(lin.id),
                **_blank(TEMPLATE_COLUMNS, "reviewer_id"),
            }
        )
    return items


def rights_worklist() -> list[dict[str, str]]:
    registry = {s["id"]: s for s in json.loads(REGISTRY_PATH.read_text("utf-8"))["sources"]}
    addendum = json.loads(ADDENDUM_PATH.read_text("utf-8"))
    entries: dict[str, list[str]] = defaultdict(list)
    for e in addendum["entries"]:
        entries[e["source"]].append(e["id"])
    caveats = {c.split(":", 1)[0]: c for c in addendum["limits"] if c.startswith("SRC-")}
    items = []
    unverified = addendum["rule"]["excluded_unverified"]
    for sid in (*ELIGIBLE_SOURCES, *unverified):
        s = registry[sid]
        accepted = sid in ELIGIBLE_SOURCES
        items.append(
            {
                "source_id": sid,
                "name": s["name"],
                "licence_id": s["license"]["id"],
                "evidence_ids": ";".join(sorted(entries.get(sid, []))) or "none",
                "retrieved_on": addendum["retrieved_on"] if accepted else "",
                "status_in_repository": "documented for the exact datasets used; not legal advice"
                if accepted
                else "terms unverified; excluded from EXP-TR03",
                "open_caveat": caveats.get(sid, "terms not verified for the datasets used"),
                "attribution_required": "yes",
                "owner_decision": "",
                "decided_by": "",
                "decided_on": "",
                "notes": "",
            }
        )
    items.append(
        {
            "source_id": "external_adapted (all upstream sources)",
            "name": "Cognipeer and other adapted external records",
            "licence_id": "see DATA_SOURCES.md",
            "evidence_ids": "none",
            "retrieved_on": "",
            "status_in_repository": "unresolved upstream generation terms; quarantined",
            "open_caveat": "excluded from EXP-TR03 by rule; not reviewed here",
            "attribution_required": "yes",
            "owner_decision": "",
            "decided_by": "",
            "decided_on": "",
            "notes": "",
        }
    )
    return items


def heldout_worklist(cands: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    items = []
    for c in cands:
        if c["role"] == "in_eligible_train_records":
            continue  # a layout the training rows show is not held out
        items.append(
            {
                **{k: str(c[k]) for k in HELDOUT_COLUMNS[:11]},
                "author_id": "",
                "use_for_heldout": "",
            }
        )
    return items


def _csv_bytes(columns: Sequence[str], rows: Iterable[dict[str, str]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


ROSTER_STATEMENT = (
    "I designate each person listed as a real human being, not an AI system, script or automated "
    "service, and I take responsibility for the independence each role requires. This file is a "
    "record of my designation. The checking tool cannot authenticate anyone from a string."
)
ROSTER_ROLES = {
    "reviewer": "first or second reviewer of templates and rows; judges gold and naturalness",
    "adjudicator": "decides a row or template on which two reviewers differ; never one of them",
    "author": "writes held-out evaluation tasks; native Turkish; has not seen the training data",
    "checker": "checks a held-out task and its gold; never the author of that task",
    "owner": "records rights decisions; the project owner",
}
ROSTER_PERSON_FIELDS = (
    "id",
    "type",
    "roles",
    "native_turkish",
    "fluent_english",
    "independent_of_corpus_authoring",
)
ROSTER_TEMPLATE = {
    "schema": "axbench-tr.tr03-roster/2",
    "status": "TEMPLATE: empty. Only the owner designates people. An AI cannot be listed as human.",
    "designated_by": "",
    "designated_on": "",
    "attestation": {
        "statement": ROSTER_STATEMENT,
        "statement_accepted": False,
        "attested_by": "",
        "attested_on": "",
    },
    "people": [],
    "id_rule": "a short private code, 2 to 32 characters of letters, digits, '_' or '-', starting "
    "with a letter, unique ignoring case; keep the mapping to real names outside the repository",
    "person_entry_fields": {
        "id": "private code",
        "type": "must be exactly 'human'",
        "roles": sorted(ROSTER_ROLES),
        "native_turkish": "true or false, never left out",
        "fluent_english": "true or false, never left out",
        "independent_of_corpus_authoring": "true only if the person did not write the corpus "
        "templates, this tooling or this protocol",
    },
    "roles": ROSTER_ROLES,
    "dates": "ISO calendar dates, YYYY-MM-DD",
}


# ------------------------------------------------------------------------------------ writing


def build_outputs(version: str = CURRENT_VERSION) -> dict[str, bytes]:
    train, valid, _, _ = load(version)
    elig_t, _ = split_eligible(train)
    elig_v, _ = split_eligible(valid)
    cands = heldout_candidates([*train, *valid], {r["schema_id"] for r in elig_t})
    return {
        "INVENTORY.json": _json_bytes(inventory(version)),
        "AUDIT.json": _json_bytes(audit(version)),
        "SIZING.json": _json_bytes(sizing()),
        "TEMPLATE_REVIEW_WORKLIST.csv": _csv_bytes(
            TEMPLATE_COLUMNS, template_worklist(elig_t + elig_v)
        ),
        "ROW_REVIEW_WORKLIST.csv": _csv_bytes(ROW_COLUMNS, row_worklist(elig_t)),
        "RIGHTS_WORKLIST.csv": _csv_bytes(RIGHTS_COLUMNS, rights_worklist()),
        "HELDOUT_SCHEMA_CANDIDATES.csv": _csv_bytes(HELDOUT_COLUMNS, heldout_worklist(cands)),
        "HUMAN_ROSTER.template.json": _json_bytes(ROSTER_TEMPLATE),
    }


def write_outputs(out_dir: Path = PREP_DIR, version: str = CURRENT_VERSION) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = build_outputs(version)
    hashes = {}
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        hashes[name] = sha256_bytes(data)
    manifest = {
        "schema": "axbench-tr.tr03-prep-manifest/1",
        "generator": "python -m axbench_tr.tr03 all",
        "corpus_version": f"axbench-tr/{version.removeprefix('v')}",
        "files_sha256": hashes,
        "status": "prepared; generated deterministically from committed inputs; contains no "
        "human decision",
    }
    (out_dir / "MANIFEST.json").write_bytes(_json_bytes(manifest))
    return hashes


def check_outputs(out_dir: Path = PREP_DIR, version: str = CURRENT_VERSION) -> list[str]:
    """Regenerate in memory and compare byte for byte with the committed files."""
    problems = []
    for name, data in build_outputs(version).items():
        path = out_dir / name
        if not path.exists():
            problems.append(f"missing:{name}")
        elif path.read_bytes() != data:
            problems.append(f"differs:{name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axbench_tr.tr03", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all", help="write every file under data/axbench-tr/tr03-prep/")
    sub.add_parser("check", help="regenerate in memory and compare with the committed files")
    sub.add_parser("gates", help="report the state of each gate; never opens execution")
    a = ap.parse_args(argv)
    if a.cmd == "all":
        print(json.dumps(write_outputs(), indent=1, sort_keys=True))
    elif a.cmd == "check":
        problems = check_outputs()
        print("\n".join(problems) or "committed files equal a fresh deterministic rebuild")
        return 1 if problems else 0
    else:
        from . import tr03_gates  # late: tr03_gates imports this module

        report = tr03_gates.gates_report()
        print(json.dumps(report, indent=1, sort_keys=True))
        return tr03_gates.exit_code(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
