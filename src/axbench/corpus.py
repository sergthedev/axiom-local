"""Versioned AXBENCH development corpora: verified loading and freezing (ADR 0011, revision 2).

After the Phase 3A review, every consumer (audit, review batch, rule baseline, scoring, MLX export,
sealing) reads the development splits from a **committed corpus directory** whose ``MANIFEST.json``
pins the SHA-256 and size of each file; nothing regenerates them from templates any more. The
template generator only creates a draft (``python -m axbench generate``).

A **frozen** version is made from a parent version by :func:`freeze`, optionally applying a human
reviewer's decision ledger (the filled-in review CSV). Each decision is ``approve``, ``correct``
(a corrected gold label and/or request) or ``reject`` (the example is removed). Provenance is
per record: only examples with a decision carry ``human_review: "owner_reviewed"``; every other
example keeps ``"none"``. The ledger must be attested as human (``human_review_attested``): an AI
review is not a decision and must never be passed here. The new version is audited and written
only if the audit passes; the parent is never modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .audit import audit, gold_problems

DEV_SPLITS = ("train", "validation")
FREEZE_VERSION = "axbench.freeze/0.1"
DECISIONS = ("approve", "correct", "reject")


class CorpusError(Exception):
    pass


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    return text.encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus(
    corpus_dir: str | os.PathLike,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Read a corpus directory; every file must match its manifest hash and count."""
    root = Path(corpus_dir)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise CorpusError("corpus directory has no MANIFEST.json")
    if (root / "test.jsonl").exists():
        raise CorpusError("a development corpus directory must not contain test.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in DEV_SPLITS:
        meta = files.get(f"{split}.jsonl")
        if meta is None:
            raise CorpusError(f"manifest lists no {split}.jsonl")
        data = (root / f"{split}.jsonl").read_bytes()
        if _sha(data) != meta["sha256"]:
            raise CorpusError(f"{split}.jsonl does not match its manifest hash")
        records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(records) != meta["n"]:
            raise CorpusError(
                f"{split}.jsonl has {len(records)} records, manifest says {meta['n']}"
            )
        if any(r.get("split") != split for r in records):
            raise CorpusError(f"{split}.jsonl contains a record of another split")
        if any(r.get("corpus_version") != manifest.get("corpus_version") for r in records):
            raise CorpusError(f"{split}.jsonl records do not carry the manifest corpus version")
        splits[split] = records
    return manifest, splits


def manifest_sha256(corpus_dir: str | os.PathLike) -> str:
    return _sha((Path(corpus_dir) / "MANIFEST.json").read_bytes())


def parse_decisions(data: bytes) -> dict[str, dict[str, Any]]:
    """Decisions from the review CSV: rows with an empty ``decision`` are not decisions."""
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if rows.fieldnames is None or not {"id", "decision"} <= set(rows.fieldnames):
        raise CorpusError("decision ledger needs at least the columns id and decision")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        decision = (row.get("decision") or "").strip().lower()
        if not decision:
            continue
        rid = (row.get("id") or "").strip()
        if decision not in DECISIONS:
            raise CorpusError(f"{rid}: unknown decision {decision!r}")
        if rid in out:
            raise CorpusError(f"{rid}: more than one decision")
        entry: dict[str, Any] = {"decision": decision}
        corrected_gold = (row.get("corrected_gold") or "").strip()
        corrected_request = (row.get("corrected_request") or "").strip()
        if decision == "correct":
            if not corrected_gold and not corrected_request:
                raise CorpusError(f"{rid}: 'correct' needs a corrected gold label or request")
            if corrected_gold:
                try:
                    entry["gold"] = json.loads(corrected_gold)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{rid}: corrected gold is not JSON") from exc
            if corrected_request:
                entry["request"] = corrected_request
        elif corrected_gold or corrected_request:
            raise CorpusError(f"{rid}: a correction is given but the decision is {decision!r}")
        out[rid] = entry
    return out


def freeze(
    source_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    version: str,
    decisions_csv: bytes | None = None,
    reviewer: str | None = None,
    human_review_attested: bool = False,
) -> dict[str, Any]:
    """Write a frozen corpus version derived from ``source_dir``; returns its manifest.

    Nothing is written unless every decision applies, every gold label is valid and the
    train/validation audit passes.
    """
    parent_manifest, splits = load_corpus(source_dir)
    if not version or version == parent_manifest.get("corpus_version"):
        raise CorpusError("a frozen corpus needs a new version label")
    decisions: dict[str, dict[str, Any]] = {}
    if decisions_csv is not None:
        if not human_review_attested:
            raise CorpusError(
                "decisions must come from a human reviewer; an AI review is not a decision"
            )
        if not reviewer:
            raise CorpusError("name the human reviewer (a label, not contact details)")
        decisions = parse_decisions(decisions_csv)
    known = {r["id"] for records in splits.values() for r in records}
    unknown = sorted(set(decisions) - known)
    if unknown:
        raise CorpusError(f"decisions for unknown example ids: {unknown[:5]}")

    counts: dict[str, Counter[str]] = {s: Counter() for s in DEV_SPLITS}
    frozen: dict[str, list[dict[str, Any]]] = {}
    removed: list[str] = []
    for split in DEV_SPLITS:
        out_records = []
        for record in splits[split]:
            r = json.loads(json.dumps(record))  # deep copy
            r["corpus_version"] = version
            d = decisions.get(r["id"])
            if d is None:
                counts[split]["unreviewed"] += 1
                out_records.append(r)
                continue
            counts[split][d["decision"]] += 1
            if d["decision"] == "reject":
                removed.append(r["id"])
                continue
            prov = r["provenance"]
            prov["human_review"] = "owner_reviewed"
            prov["review"] = {"decision": d["decision"], "reviewer": reviewer}
            if "gold" in d:
                prov["original_gold"] = r["gold"]
                prov["label_source"] = "human_corrected"
                r["gold"] = d["gold"]
            if "request" in d:
                prov["original_request"] = r["request"]
                prov["request_source"] = "human_corrected"
                r["request"] = d["request"]
            problems = gold_problems(r)
            if problems:
                raise CorpusError(f"{r['id']}: corrected label is invalid: {problems}")
            out_records.append(r)
        frozen[split] = out_records

    report = audit(frozen)
    if not report["checks"]["passed"]:
        raise CorpusError(f"audit failed; nothing written: {report['checks']['failures']}")

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise CorpusError("output directory must be empty")
    data = {split: _jsonl(frozen[split]) for split in DEV_SPLITS}
    reviewed = sum(c["approve"] + c["correct"] + c["reject"] for c in counts.values())
    ledger = [{"id": rid, **{k: v for k, v in d.items()}} for rid, d in sorted(decisions.items())]
    ledger_bytes = _jsonl(ledger)
    manifest = {
        "corpus_version": version,
        "status": "frozen",
        "freeze_version": FREEZE_VERSION,
        "parent": {
            "corpus_version": parent_manifest.get("corpus_version"),
            "manifest_sha256": manifest_sha256(source_dir),
            "files": parent_manifest["files"],
        },
        "files": {f"{s}.jsonl": {"sha256": _sha(data[s]), "n": len(frozen[s])} for s in DEV_SPLITS},
        "human_review": {
            "reviewer": reviewer if decisions else None,
            "attested_human": bool(decisions),
            "decision_ledger_sha256": _sha(ledger_bytes) if decisions else None,
            "source_csv_sha256": _sha(decisions_csv) if decisions_csv is not None else None,
            "examples_with_a_human_decision": reviewed,
            "by_split": {s: dict(sorted(counts[s].items())) for s in DEV_SPLITS},
            "removed_ids": sorted(removed),
            "note": "Only examples with a decision were reviewed by a human; all others are "
            "AI-generated from templates and unreviewed.",
        },
        "audit_passed": True,
        "test": "not in the repository; sealed by the owner against this version",
    }
    out.mkdir(parents=True, exist_ok=True)
    for split in DEV_SPLITS:
        (out / f"{split}.jsonl").write_bytes(data[split])
    if decisions:
        (out / "DECISIONS.jsonl").write_bytes(ledger_bytes)
    (out / "MANIFEST.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return manifest


__all__ = [
    "CorpusError",
    "DEV_SPLITS",
    "FREEZE_VERSION",
    "freeze",
    "load_corpus",
    "manifest_sha256",
    "parse_decisions",
]
