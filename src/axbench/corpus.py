"""Versioned AXBENCH development corpora: verified loading and freezing (ADR 0011, revision 3).

After the Phase 3A review, every consumer (audit, review batch, rule baseline, scoring, MLX export,
sealing) reads the development splits from a **committed corpus directory** whose ``MANIFEST.json``
pins the SHA-256 and size of each file; nothing regenerates them from templates any more. The
template generator only creates a draft (``python -m axbench generate``).

A **frozen** version is made from a parent version by :func:`freeze`, optionally applying a decision
ledger (the filled-in review CSV). Each decision is ``approve``, ``correct`` (a corrected gold label
and/or request, with the reviewed perturbation tags of a corrected request) or ``reject`` (the
example is removed). A ledger is applied either as a human review, which must be attested
(``human_review_attested``), or, since revision 3, as an **owner-authorized AI review**
(``ai_review_authorization``), which never counts as human review: every example keeps
``human_review: "none"`` and the decision is recorded with ``reviewer_type: "ai"`` and the
authorization reference. Under a human review only decided examples carry
``human_review: "owner_reviewed"``. The new version is audited and written only if the audit
passes; the parent is never modified.
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
FREEZE_VERSION = "axbench.freeze/0.2"
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


def _tags(text: str) -> list[str]:
    return sorted({t.strip() for t in text.replace(",", ";").split(";") if t.strip()})


def parse_decisions(data: bytes) -> dict[str, dict[str, Any]]:
    """Decisions from the review CSV: rows with an empty ``decision`` are not decisions.

    Optional correction columns: ``corrected_gold`` (candidate JSON), ``corrected_request`` and
    ``corrected_tags`` (perturbation tags of the corrected request, separated by ``;``).
    """
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
        corrected_tags = row.get("corrected_tags")
        has_tags = corrected_tags is not None and corrected_tags.strip() != ""
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
                if not has_tags:
                    raise CorpusError(
                        f"{rid}: a corrected request needs its reviewed tags in corrected_tags "
                        "(use 'none' for no tag)"
                    )
            if has_tags:
                if not corrected_request:
                    raise CorpusError(f"{rid}: corrected_tags without a corrected request")
                tags = _tags(corrected_tags)
                entry["tags"] = [] if tags == ["none"] else tags
        elif corrected_gold or corrected_request or has_tags:
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
    ai_review_authorization: str | None = None,
) -> dict[str, Any]:
    """Write a frozen corpus version derived from ``source_dir``; returns its manifest.

    A decision ledger is applied either as a **human** review (``human_review_attested``) or as
    an **owner-authorized AI review** (``ai_review_authorization``: a reference to the owner's
    authorization and the published decisions). The two are mutually exclusive. An AI review never
    sets ``human_review``: every example keeps ``"none"`` and the review is recorded with
    ``reviewer_type: "ai"``. Nothing is written unless every decision applies, every gold label is
    valid and the train/validation audit passes.
    """
    parent_manifest, splits = load_corpus(source_dir)
    if not version or version == parent_manifest.get("corpus_version"):
        raise CorpusError("a frozen corpus needs a new version label")
    if human_review_attested and ai_review_authorization:
        raise CorpusError("a ledger is either a human review or an AI review, not both")
    reviewer_type: str | None = None
    decisions: dict[str, dict[str, Any]] = {}
    if decisions_csv is not None:
        if human_review_attested:
            reviewer_type = "human"
        elif ai_review_authorization and ai_review_authorization.strip():
            reviewer_type = "ai"
        else:
            raise CorpusError(
                "decisions must be attested as a human review, or recorded as an AI review "
                "with a reference to the owner's authorization"
            )
        if not reviewer:
            raise CorpusError("name the reviewer (a label, not contact details)")
        decisions = parse_decisions(decisions_csv)
    elif human_review_attested or ai_review_authorization:
        raise CorpusError("a review type was given without a decision ledger")
    known = {r["id"] for records in splits.values() for r in records}
    unknown = sorted(set(decisions) - known)
    if unknown:
        raise CorpusError(f"decisions for unknown example ids: {unknown[:5]}")
    known_tags = {t for records in splits.values() for r in records for t in r["tags"]}
    for rid, d in decisions.items():
        bad = sorted(set(d.get("tags", [])) - known_tags)
        if bad:
            raise CorpusError(f"{rid}: unknown tags {bad}")

    source = "human_corrected" if reviewer_type == "human" else "ai_review_corrected"
    counts: dict[str, Counter[str]] = {s: Counter() for s in DEV_SPLITS}
    frozen: dict[str, list[dict[str, Any]]] = {}
    removed: list[str] = []
    corrections: list[dict[str, Any]] = []
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
            review = {
                "decision": d["decision"],
                "reviewer": reviewer,
                "reviewer_type": reviewer_type,
            }
            if reviewer_type == "human":
                prov["human_review"] = "owner_reviewed"
            else:
                review["authorization"] = ai_review_authorization
            prov["review"] = review
            changed = []
            if "gold" in d:
                prov["original_gold"] = r["gold"]
                prov["label_source"] = source
                r["gold"] = d["gold"]
                changed.append("gold")
            if "request" in d:
                prov["original_request"] = r["request"]
                prov["request_source"] = source
                r["request"] = d["request"]
                changed.append("request")
            if "tags" in d:
                prov["original_tags"] = r["tags"]
                r["tags"] = d["tags"]
                if d["tags"] != prov["original_tags"]:
                    changed.append("tags")
            if changed:
                corrections.append({"id": r["id"], "split": split, "changed": changed})
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
    decided = sum(c["approve"] + c["correct"] + c["reject"] for c in counts.values())
    ledger = [{"id": rid, **d} for rid, d in sorted(decisions.items())]
    ledger_bytes = _jsonl(ledger)
    if reviewer_type == "ai":
        note = (
            "Decisions are an owner-authorized AI review, not human annotation or independent "
            "human validation. No example has been reviewed by a human."
        )
    elif reviewer_type == "human":
        note = (
            "Only examples with a decision were reviewed by a human; all others are "
            "AI-generated from templates and unreviewed."
        )
    else:
        note = "No review applied; all examples are AI-generated from templates and unreviewed."
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
        "review": {
            "reviewer_type": reviewer_type,
            "reviewer": reviewer if decisions else None,
            "authorization": ai_review_authorization if reviewer_type == "ai" else None,
            "attested_human": reviewer_type == "human",
            "decision_ledger_sha256": _sha(ledger_bytes) if decisions else None,
            "source_csv_sha256": _sha(decisions_csv) if decisions_csv is not None else None,
            "examples_with_a_decision": decided,
            "examples_with_a_human_decision": decided if reviewer_type == "human" else 0,
            "by_split": {s: dict(sorted(counts[s].items())) for s in DEV_SPLITS},
            "corrections": corrections,
            "removed_ids": sorted(removed),
            "note": note,
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
