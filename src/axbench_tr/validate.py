"""Deterministic validation of AXBENCH-TR records and of the corpus as a whole.

Per record: the gold label is a valid ``axiom.candidate/1`` object and is semantically valid for the
record's own schema (``axbench.audit.gold_problems``, which delegates to the ``axiom_router``
parser and validator and the Core plan model); for template-generated PLAN/CLARIFY records an
independent check confirms that the request text really names the gold columns and no other
column. Per corpus: splits share no lineage, no table group, no normalized request text.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from axbench.audit import gold_problems

from .schemas import tokens
from .synth import normalize_request

REQUIRED_KEYS = (
    "id",
    "corpus_version",
    "split",
    "stratum",
    "language",
    "family",
    "domain",
    "schema_id",
    "lineage_id",
    "frame",
    "request",
    "schema",
    "gold",
    "accepted_variants",
    "tags",
    "provenance",
)
ORIGINS = frozenset({"synthetic_template", "external_adapted"})
LANGUAGES = frozenset({"tr", "en"})


def _word_seq(text: str) -> str:
    return " " + " ".join(tokens(text)) + " "


def record_problems(rec: dict[str, Any]) -> list[str]:
    problems = [f"missing_key:{k}" for k in REQUIRED_KEYS if k not in rec]
    if problems:
        return problems
    if rec["language"] not in LANGUAGES:
        problems.append("bad_language")
    if rec["split"] not in ("train", "validation"):
        problems.append("bad_split")
    if rec["provenance"].get("origin") not in ORIGINS:
        problems.append("bad_origin")
    if rec["provenance"].get("human_review") != "none":
        problems.append("human_review_claim_without_evidence")
    if not isinstance(rec["request"], str) or not 3 <= len(rec["request"]) <= 2000:
        problems.append("bad_request_length")
    problems += gold_problems(rec)
    if rec["provenance"].get("origin") == "synthetic_template":
        problems += _mention_problems(rec)
    elif (
        "gold_derivation" not in rec["provenance"]
        and rec["provenance"].get("source_id") == "SRC-N01"
    ):
        problems.append("adapted_record_without_derivation")
    return problems


def _mention_problems(rec: dict[str, Any]) -> list[str]:
    """Independent check that the request names the gold columns (and no other column)."""
    mentions: dict[str, list[str]] = rec.get("mentions", {})
    request = rec["request"]
    problems = [
        f"mention_not_in_request:{slot}"
        for slot, ms in mentions.items()
        for m in ms
        if m not in request
    ]
    cols = rec["schema"]["columns"]
    gold = rec["gold"]
    if gold["action"] == "PLAN":
        ids: list[str]
        op = gold["operation"]
        if op == "inspect":
            ids = gold["columns"]
        elif op == "relate":
            ids = [gold["x"], gold["y"]]
        elif op == "compare":
            ids = [gold["group_by"], gold["measure"]]
        else:
            ids = gold["features"]
        if rec["frame"] != "inspect_all":
            slots = sum(len(v) for k, v in mentions.items() if k != "a")
            if slots != len(ids):
                problems.append("mention_count_differs_from_gold_columns")
        remaining = request
        for ms in mentions.values():
            for m in ms:
                remaining = remaining.replace(m, " ")
        seq = _word_seq(remaining)
        gold_names = {c["name"] for c in cols if c["id"] in ids}
        for c in cols:
            if c["name"] in gold_names:
                continue
            name_seq = _word_seq(c["name"]).strip()
            if len(name_seq) >= 4 and f" {name_seq} " in seq:
                problems.append(f"request_names_other_column:{c['id']}")
    elif gold["action"] == "CLARIFY" and gold["reason"] == "ambiguous_column":
        token = mentions["a"][0]
        members = [c["id"] for c in cols if token in tokens(c["name"])]
        if sorted(members) != sorted(gold["columns"]):
            problems.append("ambiguity_set_differs_from_token_matches")
    elif gold["action"] == "CLARIFY" and gold["reason"] == "column_not_found":
        absent_tokens = {t for t in tokens(mentions["n"][0]) if len(t) >= 4}
        if any(absent_tokens & set(tokens(c["name"])) for c in cols):
            problems.append("absent_column_overlaps_schema")
    return problems


def corpus_report(corpus: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Validate every record and the cross-split constraints; returns problems and counts."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    for split, recs in corpus.items():
        for rec in recs:
            if rec["split"] != split:
                problems.append(f"{rec['id']}:split_field_differs_from_file")
            if rec["id"] in seen_ids:
                problems.append(f"{rec['id']}:duplicate_id")
            seen_ids.add(rec["id"])
            problems += [f"{rec['id']}:{p}" for p in record_problems(rec)]
    lineages: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = defaultdict(set)
    for split, recs in corpus.items():
        for rec in recs:
            lineages[rec["lineage_id"]].add(split)
            if rec["provenance"]["origin"] == "synthetic_template":
                groups[rec["family"]].add(split)
            texts[normalize_request(rec["request"])].add(split)
    shared = {
        "lineages": sorted(k for k, v in lineages.items() if len(v) > 1),
        "table_groups": sorted(k for k, v in groups.items() if len(v) > 1),
        "request_texts": sorted(k for k, v in texts.items() if len(v) > 1),
    }
    problems += [f"cross_split_{kind}:{x}" for kind, xs in shared.items() for x in xs]
    n = sum(map(len, corpus.values()))
    return {
        "records": n,
        "problems": problems,
        "problem_count": len(problems),
        "cross_split_shared": {k: len(v) for k, v in shared.items()},
        "distinct_request_texts": len(texts),
        "records_by_split": {s: len(r) for s, r in corpus.items()},
        "problem_kinds": dict(
            Counter(p.split(":", 2)[1] if p.count(":") >= 1 else p for p in problems)
        ),
    }
