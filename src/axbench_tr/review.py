"""Reproducible, stratified Turkish human-review sample (``python -m axbench_tr review-sample``).

The sample is a deterministic function of the committed corpus and a seed: candidates are ranked by
``sha256(seed|record id)``, strata are the origin (template or external) and the frame, and the
ranking prefers one record per lineage inside each stratum. External (machine-translated or
LLM-written) records are oversampled because they carry the highest quality risk. Each row holds the
request, the schema projection, the gold plan, a readable form of that plan, provenance with
the upstream licence, and empty reviewer columns. Nothing in this file is a review result.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build import CURRENT_VERSION, OUT_ROOT
from .fetch import sha256_bytes
from .registry import REGISTRY_PATH

SAMPLE_SEED = "axbench-tr-0.2-rev1-review-sample"
SAMPLE_SIZE = 390
# (language, origin) -> records. External (machine-translated or upstream-LLM) rows are
# oversampled, and English gets its own slice for an English reviewer.
ALLOCATION = {
    ("tr", "external_adapted"): 100,
    ("tr", "synthetic_template"): 200,
    ("en", "external_adapted"): 30,
    ("en", "synthetic_template"): 60,
}
HIGH_RISK_FRAMES = (
    "reject:instruction_override",
    "reject:code_or_system_request",
    "clarify:ambiguous_column",
    "clarify:column_type_mismatch",
    "clarify:invalid_parameter",
)
RISK_FLOOR = 5  # a mis-labelled safety or ambiguity row is worse than a mis-labelled summary
MIN_PER_STRATUM = 3
REVIEW_DIR = OUT_ROOT / CURRENT_VERSION / "review"
REVIEWER_COLUMNS = (
    "reviewer_request_natural",  # yes / no / fixable
    "reviewer_gold_correct",  # yes / no
    "reviewer_serious_error",  # yes if the plan could mislead or act unsafely, else no
    "reviewer_decision",  # keep / fix / drop
    "reviewer_notes",
    "second_reviewer_gold_correct",  # independent second reviewer on a subset (blank elsewhere)
    "second_reviewer_decision",
    "adjudicated_decision",  # final decision after the two disagree
)
CSV_COLUMNS = (
    "sample_id",
    "record_id",
    "split",
    "origin",
    "language",
    "risk",
    "frame",
    "request",
    "schema_columns",
    "gold_readable",
    "gold_json",
    "source_id",
    "source_licence",
    "schema_source",
    "request_authoring",
    *REVIEWER_COLUMNS,
)


def _rank(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|{record_id}".encode()).hexdigest()


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation with a per-stratum floor, bounded by availability."""
    pop = sum(sizes.values())
    floor = {
        k: min(RISK_FLOOR if k in HIGH_RISK_FRAMES else MIN_PER_STRATUM, n)
        for k, n in sizes.items()
    }
    if sum(floor.values()) > total:  # a small slice: one record per stratum instead
        floor = {k: min(1, n) for k, n in sizes.items()}
    spare = total - sum(floor.values())
    if spare < 0:
        raise ValueError("sample too small for the stratum floor")
    quota = {k: spare * n / pop for k, n in sizes.items()}
    alloc = {k: floor[k] + int(quota[k]) for k in sizes}
    for k in sorted(sizes, key=lambda k: (-(quota[k] - int(quota[k])), k)):
        if sum(alloc.values()) >= total:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    # Rounding can leave a stratum above its size; move the surplus to the largest strata.
    while sum(alloc.values()) < total:
        k = max((k for k in sizes if alloc[k] < sizes[k]), key=lambda k: sizes[k] - alloc[k])
        alloc[k] += 1
    return {k: min(v, sizes[k]) for k, v in alloc.items()}


def _pick(rows: list[dict[str, Any]], n: int, seed: str) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: _rank(seed, r["id"]))
    chosen: list[dict[str, Any]] = []
    seen_lineages: set[str] = set()
    for r in ranked:  # first pass: one record per lineage
        if len(chosen) == n:
            break
        if r["lineage_id"] not in seen_lineages:
            seen_lineages.add(r["lineage_id"])
            chosen.append(r)
    for r in ranked:  # second pass only if lineages ran out
        if len(chosen) == n:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen


def readable_gold(rec: dict[str, Any]) -> str:
    names = {c["id"]: c["name"] for c in rec["schema"]["columns"]}

    def col(i: str) -> str:
        return f"{names.get(i, '?')} ({i})"

    g = rec["gold"]
    if g["action"] == "CLARIFY":
        return f"CLARIFY: {g['reason']}"
    if g["action"] == "REJECT":
        return f"REJECT: {g['reason']}"
    op = g["operation"]
    if op == "inspect":
        return "PLAN inspect: " + ", ".join(col(i) for i in g["columns"])
    if op == "relate":
        return f"PLAN relate ({g['method']}): {col(g['x'])} vs {col(g['y'])}"
    if op == "compare":
        return f"PLAN compare: measure {col(g['measure'])} by {col(g['group_by'])}"
    if op == "cluster":
        return f"PLAN cluster (k={g['k']}): " + ", ".join(col(i) for i in g["features"])
    return json.dumps(g, ensure_ascii=False)


def _source_index() -> dict[str, dict[str, Any]]:
    registry = json.loads(REGISTRY_PATH.read_text("utf-8"))
    return {s["id"]: s for s in registry["sources"]}


def load_corpus() -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for split in ("train", "validation"):
        raw = (OUT_ROOT / CURRENT_VERSION / f"{split}.jsonl").read_bytes()
        hashes[f"{split}.jsonl"] = sha256_bytes(raw)
        rows += [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    return rows, hashes


def select(rows: list[dict[str, Any]], seed: str = SAMPLE_SEED) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for (lang, origin), total in ALLOCATION.items():
        by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r["language"] == lang and r["provenance"]["origin"] == origin:
                by_frame[r["frame"]].append(r)
        total = min(total, sum(map(len, by_frame.values())))
        alloc = _allocate({f: len(v) for f, v in by_frame.items()}, total)
        for frame in sorted(by_frame):
            picked += _pick(by_frame[frame], alloc[frame], f"{seed}/{lang}/{origin}/{frame}")
    picked.sort(key=lambda r: _rank(seed, r["id"]))
    return picked


def sample_rows(picked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = _source_index()
    out = []
    for n, r in enumerate(picked, 1):
        prov = r["provenance"]
        src = sources.get(prov.get("source_id") or prov.get("schema_source_id", ""), {})
        lic = src.get("license", {})
        lic_id = lic.get("id") if isinstance(lic, dict) else lic
        out.append(
            {
                "sample_id": f"R2-{n:03d}",
                "record_id": r["id"],
                "split": r["split"],
                "origin": prov["origin"],
                "frame": r["frame"],
                "request": r["request"],
                "schema_columns": [
                    f"{c['id']}={c['name']} [{c['type']}]" for c in r["schema"]["columns"]
                ],
                "gold_readable": readable_gold(r),
                "gold_json": r["gold"],
                "source_id": prov.get("source_id") or prov.get("schema_source_id"),
                "source_revision": src.get("revision"),
                "source_licence": lic_id,
                "schema_source": prov.get("schema_source_id") or prov.get("schema_origin"),
                "language": r["language"],
                "risk": "high" if r["frame"] in HIGH_RISK_FRAMES else "standard",
                "request_authoring": prov["request_authoring"],
                "upstream_row": prov.get("upstream_row_id", prov.get("upstream_row_index")),
                "lineage_id": r["lineage_id"],
                **{c: "" for c in REVIEWER_COLUMNS},
            }
        )
    return out


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    text = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n"
    return text.encode("utf-8")


def _csv(rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        flat = dict(r)
        flat["schema_columns"] = " | ".join(r["schema_columns"])
        flat["gold_json"] = json.dumps(r["gold_json"], ensure_ascii=False, sort_keys=True)
        w.writerow(flat)
    return ("﻿" + buf.getvalue()).encode("utf-8")  # BOM so spreadsheets open UTF-8 correctly


def build(seed: str = SAMPLE_SEED) -> dict[str, Any]:
    rows, hashes = load_corpus()
    out = sample_rows(select(rows, seed))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    jsonl, csv_bytes = _jsonl(out), _csv(out)
    (REVIEW_DIR / "REVIEW_SAMPLE.jsonl").write_bytes(jsonl)
    (REVIEW_DIR / "REVIEW_SAMPLE.csv").write_bytes(csv_bytes)
    manifest = {
        "schema": "axbench-tr.review-sample/1",
        "seed": seed,
        "records": len(out),
        "selection": (
            "Records of train and validation; strata language x origin x frame; allocation "
            f"{ {f'{k[0]}/{k[1]}': v for k, v in ALLOCATION.items()} }; floor "
            f"{MIN_PER_STRATUM} per stratum, {RISK_FLOOR} for high-risk frames (1 in a small "
            "slice; or all if fewer); largest-remainder "
            "allocation; inside a stratum records are ranked by sha256(seed/language/origin/"
            "frame|id) with one record per lineage preferred"
        ),
        "ai_review": "none: no AI review decisions are recorded here; human and AI review "
        "results, when they exist, go in separate files",
        "corpus_files_sha256": hashes,
        "by_origin": dict(Counter(r["origin"] for r in out)),
        "by_language": dict(Counter(r["language"] for r in out)),
        "by_risk": dict(Counter(r["risk"] for r in out)),
        "by_split": dict(Counter(r["split"] for r in out)),
        "by_frame": dict(sorted(Counter(r["frame"] for r in out).items())),
        "by_source": dict(sorted(Counter(str(r["source_id"]) for r in out).items())),
        "distinct_lineages": len({r["lineage_id"] for r in out}),
        "files": {
            "REVIEW_SAMPLE.jsonl": sha256_bytes(jsonl),
            "REVIEW_SAMPLE.csv": sha256_bytes(csv_bytes),
        },
        "status": "unreviewed: reviewer columns are empty",
    }
    (REVIEW_DIR / "REVIEW_SAMPLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


GATES = {
    "gold_correct_min_per_origin": 0.95,
    "natural_or_fixable_min_per_origin": 0.90,
    "serious_errors_max": 0,
}


def _rate(rows: list[dict[str, str]], column: str, ok: tuple[str, ...]) -> dict[str, Any]:
    filled = [r for r in rows if r[column]]
    n_ok = sum(r[column] in ok for r in filled)
    return {"n": len(filled), "ok": n_ok, "rate": round(n_ok / len(filled), 3) if filled else None}


def reviewed_summary(path: Path) -> dict[str, Any]:
    """Decisions in a filled-in CSV, reported by origin, frame, language and risk.

    The gates are the technical lead's proposed predeclared ones; the summary reports them and
    never declares a pass on its own (rows still blank, or a sample alone, prove nothing).
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    def block(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for value in sorted({r[key] for r in rows}):
            part = [r for r in rows if r[key] == value]
            out[value] = {
                "rows": len(part),
                "gold_correct": _rate(part, "reviewer_gold_correct", ("yes",)),
                "natural_or_fixable": _rate(part, "reviewer_request_natural", ("yes", "fixable")),
                "serious_errors": sum(r["reviewer_serious_error"] == "yes" for r in part),
                "decision": dict(Counter(r["reviewer_decision"] or "(blank)" for r in part)),
            }
        return out

    by_origin = block("origin")
    gates = {
        origin: {
            "gold_correct_ok": (b["gold_correct"]["rate"] or 0)
            >= GATES["gold_correct_min_per_origin"]
            and b["gold_correct"]["n"] == b["rows"],
            "natural_or_fixable_ok": (b["natural_or_fixable"]["rate"] or 0)
            >= GATES["natural_or_fixable_min_per_origin"]
            and b["natural_or_fixable"]["n"] == b["rows"],
            "no_serious_errors": b["serious_errors"] <= GATES["serious_errors_max"],
        }
        for origin, b in by_origin.items()
    }
    return {
        "rows": len(rows),
        "blank_rows": sum(not r["reviewer_decision"] for r in rows),
        "by_origin": by_origin,
        "by_language": block("language"),
        "by_risk": block("risk"),
        "by_frame": block("frame"),
        "second_review": _rate(rows, "second_reviewer_gold_correct", ("yes",)),
        "predeclared_gates": GATES,
        "gates_by_origin": gates,
        "note": "A sample alone is not a passing claim; gates need every row decided, a second "
        "independent review on a subset and adjudication.",
    }
