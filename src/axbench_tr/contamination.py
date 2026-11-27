"""Leakage and contamination measurements (DR-005 style, structural).

* ``near_duplicates``: character 4-gram Jaccard between two text lists, via an inverted index.
* Against ``axbench/0.1-ai-reviewed`` train and validation (public, committed): exact normalized
  matches and near-duplicates of every AXBENCH-TR request.
* Against template text of the ``axbench/0.1`` train and validation lineages.
* ``check_external_file``: the same request check against any JSONL file of requests supplied
  locally by the owner (for example the sealed AXBENCH test split). This module prints and returns
  counts only; nobody who is not the owner needs to open that file, and this project's tooling has
  never been run on it.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .synth import normalize_request

THRESHOLD = 0.8
V01_DIR = Path(__file__).resolve().parents[2] / "data" / "axbench" / "v0.1-ai-reviewed"


def _shingles(text: str, n: int = 4) -> frozenset[str]:
    t = f" {normalize_request(text)} "
    return frozenset(t[i : i + n] for i in range(max(1, len(t) - n + 1)))


def near_duplicates(
    queries: list[str], references: list[str], threshold: float = THRESHOLD
) -> list[tuple[int, int, float]]:
    """(query index, reference index, Jaccard) for all pairs at or above ``threshold``."""
    ref_sh = [_shingles(r) for r in references]
    index: dict[str, list[int]] = defaultdict(list)
    for j, sh in enumerate(ref_sh):
        for g in sh:
            index[g].append(j)
    limit = max(50, len(references) // 20)  # ignore very common 4-grams when proposing candidates
    out = []
    for i, q in enumerate(queries):
        qs = _shingles(q)
        votes: dict[int, int] = defaultdict(int)
        for g in qs:
            posting = index.get(g, ())
            if len(posting) <= limit:
                for j in posting:
                    votes[j] += 1
        for j, v in votes.items():
            if v / max(1, len(qs)) < threshold * 0.5:
                continue
            union = len(qs | ref_sh[j])
            sim = len(qs & ref_sh[j]) / union if union else 1.0
            if sim >= threshold:
                out.append((i, j, round(sim, 4)))
    return out


def load_v01_requests() -> list[str]:
    out = []
    for name in ("train.jsonl", "validation.jsonl"):
        for line in (V01_DIR / name).read_text("utf-8").splitlines():
            out.append(json.loads(line)["request"])
    return out


def request_report(requests: list[str], references: list[str]) -> dict[str, Any]:
    ref_keys = {normalize_request(r) for r in references}
    exact = [i for i, r in enumerate(requests) if normalize_request(r) in ref_keys]
    near = near_duplicates(requests, references)
    return {
        "requests": len(requests),
        "references": len(references),
        "exact_normalized_matches": len(exact),
        "near_duplicates_at_or_above": THRESHOLD,
        "near_duplicate_pairs": len(near),
        "near_duplicate_requests": len({i for i, _, _ in near}),
    }


_SLOT = re.compile(r"\{\w+\}")


def template_report() -> dict[str, Any]:
    """Similarity of every AXBENCH-TR lineage template to the v0.1 train/validation lineage
    templates."""
    from axbench.lineages import LINEAGES as V01

    from .templates import LINEAGES as V02

    refs = [(lin.id, t) for lin in V01 if lin.split != "test" for t in (lin.tr, lin.en)]
    ref_texts = [_SLOT.sub("X", t) for _, t in refs]
    worst: dict[str, tuple[float, str]] = {}
    for lin in V02:
        for text in (lin.tr, lin.en):
            hits = near_duplicates([_SLOT.sub("X", text)], ref_texts, threshold=0.5)
            for _, j, sim in hits:
                if sim > worst.get(lin.id, (0.0, ""))[0]:
                    worst[lin.id] = (sim, refs[j][0])
    return {
        "v02_templates": len(V02),
        "v01_train_validation_lineages_compared": len({r[0] for r in refs}),
        "pairs_at_or_above_0.5": len(worst),
        "pairs_at_or_above_0.8": sum(1 for s, _ in worst.values() if s >= THRESHOLD),
        "closest": sorted(
            ({"lineage": k, "similarity": v[0], "v01_lineage": v[1]} for k, v in worst.items()),
            key=lambda d: -d["similarity"],
        )[:5],
    }


def check_external_file(
    corpus_requests: list[str], path: str | Path, field: str = "request"
) -> dict[str, Any]:
    """Count exact and near-duplicate overlaps with an owner-supplied JSONL file (counts only)."""
    refs = [
        json.loads(line)[field]
        for line in Path(path).read_text("utf-8").splitlines()
        if line.strip()
    ]
    return request_report(corpus_requests, refs)
