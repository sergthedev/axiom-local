"""Near-duplicate report on request text: character n-gram TF-IDF cosine (scikit-learn, locked).

This is a lexical-semantic proxy for paraphrase detection that works for Turkish morphology and
column-name echoes; it is not an embedding model and does not understand meaning. Pairs whose
normalised text is identical are counted separately (repeated request over different tables).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .synth import normalize_request

THRESHOLDS = (0.8, 0.9, 0.95)
CHUNK = 500


def near_duplicate_report(corpus: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [r for split in sorted(corpus) for r in corpus[split]]
    texts = [normalize_request(r["request"]) for r in rows]
    splits = np.array([r["split"] for r in rows])
    lineages = np.array([r["lineage_id"] for r in rows])
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
    ).fit_transform(texts)
    matrix = matrix.tocsr()
    cross = dict.fromkeys(THRESHOLDS, 0)
    within_other = dict.fromkeys(THRESHOLDS, 0)
    within_same = dict.fromkeys(THRESHOLDS, 0)
    identical_text_pairs = 0
    examples: list[dict[str, Any]] = []
    for start in range(0, len(rows), CHUNK):
        sim = (matrix[start : start + CHUNK] @ matrix.T).toarray()
        for i in range(sim.shape[0]):
            sim[i, : start + i + 1] = 0.0  # each unordered pair once
        for i, j in np.argwhere(sim >= THRESHOLDS[0]):
            a, b, s = start + int(i), int(j), float(sim[i, j])
            if texts[a] == texts[b]:
                identical_text_pairs += 1
                continue
            bucket = (
                cross
                if splits[a] != splits[b]
                else within_same
                if lineages[a] == lineages[b]
                else within_other
            )
            for t in THRESHOLDS:
                bucket[t] += s >= t
            if bucket is cross and len(examples) < 10:
                examples.append({"ids": [rows[a]["id"], rows[b]["id"]], "cosine": round(s, 3)})
    fmt = lambda d: {f">={t}": n for t, n in d.items()}  # noqa: E731
    return {
        "method": "char_wb 3-5-gram TF-IDF cosine over diacritic-folded normalised requests",
        "pairs_with_identical_normalised_text_excluded": identical_text_pairs,
        "cross_split_pairs": fmt(cross),
        "cross_split_examples": examples,
        "within_split_same_lineage_pairs": fmt(within_same),
        "within_split_different_lineage_pairs": fmt(within_other),
        "note": "pairs of different text at or above the first threshold; templated requests "
        "that differ only in a column name score high by construction",
    }
