"""Build, audit and write the AXBENCH-TR corpus (``data/axbench-tr/<version>/``)."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from . import acquire, adapters, contamination, dedup, validate
from .registry import ACQUISITION_PATH, REGISTRY_PATH
from .schemas import all_tables, table_partition, usable
from .synth import CORPUS_VERSION, GENERATOR, generate

OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "axbench-tr"
CURRENT_VERSION = "v0.2-rev1"  # v0.2-draft is preserved unchanged for the record
DEFAULT_SEED = "axbench-tr-0.2-rev1"
CROSS_SPLIT_COSINE = 0.9  # char n-gram TF-IDF cosine; a lexical proxy, not semantic equivalence
SYNTHETIC_SIZES = {"train": 5400, "validation": 1300}
COGNIPEER_WANT = {
    "compare": 1000,  # upper bounds on upstream rows; the intent check decides what qualifies
    "inspect": 1000,
    "data_modification": 300,  # Turkish rows; a quarter as many English ones from other rows
}


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records) + "\n"
    ).encode("utf-8")


ADAPTER_AUDIT: dict[str, Any] = {}
DEDUP_AUDIT: dict[str, Any] = {}


def build_corpus(seed: str = DEFAULT_SEED) -> dict[str, list[dict[str, Any]]]:
    problems = acquire.verify()  # the raw files must match the acquisition manifest
    if problems:
        raise SystemExit("raw files do not match ACQUISITION.json:\n" + "\n".join(problems[:10]))
    tables = all_tables()
    corpus: dict[str, list[dict[str, Any]]] = {
        s: generate(tables, s, n, seed) for s, n in SYNTHETIC_SIZES.items()
    }
    for split, recs in adapters.cognipeer_records(COGNIPEER_WANT, seed).items():
        corpus[split] += recs
    ADAPTER_AUDIT.clear()
    ADAPTER_AUDIT.update(
        {
            "qualified_upstream_rows": dict(adapters.LAST_CANDIDATES),
            "excluded_by_reason": dict(Counter(adapters.LAST_EXCLUSIONS.values())),
            "rule": "an adapted PLAN is kept only if the whole English request is a plain "
            "aggregate (average/min/max, no sum or count) question matching the bound columns; "
            "see axbench_tr/intent.py",
        }
    )
    # SRC-N03 (finance questions) is quarantined: its licence evidence conflicts (MIT tag, Apache
    # badge) and is unresolved. ``adapters.finance_negatives`` is kept but not called.
    # Final deterministic order and cross-source duplicate removal (first occurrence wins).
    seen: set[str] = set()
    for split, recs in corpus.items():
        kept = []
        for rec in sorted(recs, key=lambda r: r["id"]):
            key = f"{normalize(rec['request'])}|{rec['schema_id']}"
            if key not in seen:
                seen.add(key)
                kept.append(rec)
        corpus[split] = kept
    # Near-duplicates across the split boundary (lexical proxy, cosine >= CROSS_SPLIT_COSINE)
    # are resolved on the validation side, whole lineage at a time.
    drop = dedup.cross_split_conflicts(corpus, CROSS_SPLIT_COSINE)
    DEDUP_AUDIT.clear()
    DEDUP_AUDIT.update(
        {
            "rule": f"validation lineages with a record at cosine >= {CROSS_SPLIT_COSINE} to a "
            "train record of different text are removed whole (lexical proxy only)",
            "dropped_validation_lineages": sorted(drop),
        }
    )
    for split in corpus:
        corpus[split] = [r for r in corpus[split] if r["lineage_id"] not in drop]
    return corpus


def normalize(text: str) -> str:
    from .synth import normalize_request

    return normalize_request(text)


def distributions(corpus: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def count(recs: list[dict[str, Any]], fn) -> dict[str, int]:
        return dict(sorted(Counter(fn(r) for r in recs).items()))

    out: dict[str, Any] = {}
    for split, recs in corpus.items():
        gold = [r["gold"] for r in recs]
        out[split] = {
            "records": len(recs),
            "language": count(recs, lambda r: r["language"]),
            "origin": count(recs, lambda r: r["provenance"]["origin"]),
            "action": count(gold, lambda g: g["action"]),
            "operation_or_reason": count(
                gold,
                lambda g: (
                    g["operation"] if g["action"] == "PLAN" else f"{g['action']}:{g['reason']}"
                ),
            ),
            "frame": count(recs, lambda r: r["frame"]),
            "domain": count(recs, lambda r: r["domain"]),
            "header_language": count(recs, lambda r: r["provenance"].get("header_language", "n/a")),
            "schema_source": count(
                recs,
                lambda r: r["provenance"].get(
                    "schema_source_id", r["provenance"].get("source_id", "n/a")
                ),
            ),
            "tags": count([{"t": t} for r in recs for t in r["tags"]], lambda r: r["t"]),
            "distinct_schemas": len({r["schema_id"] for r in recs}),
            "distinct_lineages": len({r["lineage_id"] for r in recs}),
            "distinct_request_texts": len({normalize(r["request"]) for r in recs}),
            "request_chars_median": statistics.median(len(r["request"]) for r in recs),
            "schema_columns_median": statistics.median(len(r["schema"]["columns"]) for r in recs),
        }
    total = sum(len(r) for r in corpus.values())
    tr = sum(1 for r in corpus.values() for x in r if x["language"] == "tr")
    out["overall"] = {
        "records": total,
        "turkish_share": round(tr / total, 4),
        "english_share": round(1 - tr / total, 4),
    }
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def write(corpus: dict[str, list[dict[str, Any]]], version: str, seed: str = DEFAULT_SEED) -> Path:
    report = validate.corpus_report(corpus)
    if report["problem_count"]:
        raise SystemExit(
            f"refusing to write: {report['problem_count']} validation problems, "
            f"e.g. {report['problems'][:5]}"
        )
    out = OUT_ROOT / version
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, recs in corpus.items():
        data = _jsonl(recs)
        (out / f"{split}.jsonl").write_bytes(data)
        files[f"{split}.jsonl"] = {
            "records": len(recs),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    all_requests = [r["request"] for recs in corpus.values() for r in recs]
    v01 = contamination.request_report(all_requests, contamination.load_v01_requests())
    audit = {
        "validation": {k: v for k, v in report.items() if k != "problems"},
        "contamination_against_axbench_v0.1_train_and_validation": v01,
        "template_similarity_against_axbench_v0.1_train_and_validation_lineages": (
            contamination.template_report()
        ),
        "near_duplicates_in_corpus": dedup.near_duplicate_report(corpus),
        "external_adapter_audit": ADAPTER_AUDIT,
        "cross_split_near_duplicate_resolution": DEDUP_AUDIT,
        "sealed_test_check": (
            "not performed: this project's tooling has not been run on the sealed AXBENCH "
            "test split"
        ),
        "distributions": distributions(corpus),
    }
    (out / "AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        "utf-8",
        newline="\n",
    )
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "status": "draft: AI-authored, no human review (human_review is none on every record)",
        "generator": GENERATOR,
        "seed": seed,
        "files": files,
        "synthetic_sizes": SYNTHETIC_SIZES,
        "external_adapted": {
            "SRC-N01 row caps": COGNIPEER_WANT,
            "SRC-N03": "quarantined: licence evidence conflicts, no records used",
        },
        "tables_in_inventory": len(all_tables()),
        "tables_usable": sum(1 for t in all_tables() if usable(t)),
        "table_partition_counts": dict(
            Counter(table_partition(t) for t in all_tables() if usable(t))
        ),
        "registry_sha256": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
        "acquisition_manifest_sha256": hashlib.sha256(ACQUISITION_PATH.read_bytes()).hexdigest(),
        "base_commit": _git("merge-base", "HEAD", "origin/main"),  # stable across branch commits
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    return out
