"""AXBENCH split, duplicate and leakage audit (ADR 0011, DR-005).

Checks are *structural*; they need no predictions and never look at system outputs:

* every gold label is a valid ``axiom.candidate/1`` candidate and is semantically valid for the
  example's own schema projection;
* lineage identifiers are disjoint across splits (all strata); held-out-family strata use only
  families absent from training; novel-instance strata use only instances absent from training;
* no normalized request text occurs in two splits; the connected components of the graph that
  links examples sharing a lineage or a normalized text never span two splits;
* near-duplicate requests across splits are measured (character 4-gram Jaccard) and reported;
* distributions (language, stratum, action, frame, operation, domain, tags) and group sizes are
  reported so confounds are visible.

A passing audit shows that these specific constraints hold for this corpus version. It does not
show that the benchmark measures the intended capability or that labels are correct.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from axiom_router.contract import CandidateError, parse_candidate
from axiom_router.schema import project_columns
from axiom_router.semantic import SemanticError, validate_candidate

from .generate import normalize_request

NEAR_DUPLICATE_THRESHOLD = 0.8


def _shingles(text: str, n: int = 4) -> frozenset[str]:
    t = f" {normalize_request(text)} "
    return frozenset(t[i : i + n] for i in range(max(1, len(t) - n + 1)))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def gold_problems(record: dict[str, Any]) -> list[str]:
    """Problems with one record's gold label (empty when valid)."""
    import json

    view = project_columns([(c["name"], c["type"]) for c in record["schema"]["columns"]])
    try:
        cand = parse_candidate(json.dumps(record["gold"], ensure_ascii=False))
    except CandidateError as exc:
        return [f"gold_not_a_candidate:{exc.code}"]
    try:
        validate_candidate(cand, view)
    except SemanticError as exc:
        return [f"gold_semantically_invalid:{exc.code}"]
    if record["frame"].split(":")[0] in ("clarify", "reject"):
        if record["gold"]["action"] != record["frame"].split(":")[0].upper():
            return ["gold_action_does_not_match_frame"]
    return []


def _share(counts: Counter, n: int) -> float:
    return round(max(counts.values(), default=0) / max(1, n), 4)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def audit(corpus: dict[str, list[dict[str, Any]]], *, include_texts: bool = True) -> dict[str, Any]:
    """Audit split records. Request texts are never included for pairs involving ``test``."""
    from .families import FAMILY_BY_KEY, INSTANCE_BY_ID

    report: dict[str, Any] = {"splits": {}, "checks": {}, "near_duplicates": {}}
    failures: list[str] = []
    lineages: dict[str, set[str]] = {}
    texts: dict[str, set[str]] = {}
    for split, records in corpus.items():
        lineages[split] = {r["lineage_id"] for r in records}
        texts[split] = {normalize_request(r["request"]) for r in records}
        problems = Counter(p for r in records for p in gold_problems(r))
        if problems:
            failures.append(f"{split}: invalid gold labels {dict(problems)}")
        if any(r["split"] != split for r in records):
            failures.append(f"{split}: record with another split label")
        by_lineage = Counter(r["lineage_id"] for r in records)
        by_instance = Counter(r["schema_id"] for r in records)
        lang_action = Counter(f"{r['language']}/{r['gold']['action']}" for r in records)
        report["splits"][split] = {
            "n": len(records),
            "language": dict(sorted(Counter(r["language"] for r in records).items())),
            "stratum": dict(sorted(Counter(r["stratum"] for r in records).items())),
            "action": dict(sorted(Counter(r["gold"]["action"] for r in records).items())),
            "language_x_action": dict(sorted(lang_action.items())),
            "frame": dict(sorted(Counter(r["frame"] for r in records).items())),
            "plan_operation": dict(
                sorted(
                    Counter(
                        r["gold"].get("operation") for r in records if r["gold"]["action"] == "PLAN"
                    ).items()
                )
            ),
            "domain": dict(sorted(Counter(r["domain"] for r in records).items())),
            "tags": dict(sorted(Counter(t for r in records for t in r["tags"]).items())),
            "groups": {
                "lineages": len(by_lineage),
                "largest_lineage_share": _share(by_lineage, len(records)),
                "schema_instances": len(by_instance),
                "largest_instance_share": _share(by_instance, len(records)),
                "families": len({r["family"] for r in records}),
            },
            "distinct_request_texts": len(texts[split]),
            "max_repeats_of_one_text": max(
                Counter(normalize_request(r["request"]) for r in records).values(), default=0
            ),
            "invalid_gold": sum(problems.values()),
        }

    names = list(corpus)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared_lineages = lineages[a] & lineages[b]
            shared_texts = texts[a] & texts[b]
            if shared_lineages:
                failures.append(f"lineages shared by {a} and {b}: {sorted(shared_lineages)}")
            if shared_texts:
                failures.append(f"{len(shared_texts)} request texts shared by {a} and {b}")

    train_families = {r["family"] for r in corpus.get("train", [])}
    train_instances = {r["schema_id"] for r in corpus.get("train", [])}
    for split, records in corpus.items():
        for r in records:
            inst = INSTANCE_BY_ID.get(r["schema_id"])
            if inst is None or inst.family != r["family"]:
                failures.append(f"{r['id']}: unknown schema instance")
                continue
            if r["stratum"] == "A_heldout_family":
                if r["family"] in train_families or FAMILY_BY_KEY[r["family"]].partition != split:
                    failures.append(f"{r['id']}: held-out family is not held out")
            if r["stratum"] == "B_novel_instance" and r["schema_id"] in train_instances:
                failures.append(f"{r['id']}: novel instance appears in training")
            if r["stratum"] == "C_familiar_instance" and r["schema_id"] not in train_instances:
                failures.append(f"{r['id']}: familiar instance is not a training instance")

    uf = _UnionFind()
    for split, records in corpus.items():
        for r in records:
            node = f"{split}:{r['id']}"
            uf.union(node, f"lineage:{r['lineage_id']}")
            uf.union(node, f"text:{normalize_request(r['request'])}")
    component_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in corpus.items():
        for r in records:
            component_splits[uf.find(f"{split}:{r['id']}")].add(split)
    mixed_components = sum(1 for s in component_splits.values() if len(s) > 1)
    if mixed_components:
        failures.append(f"{mixed_components} lineage/text components span splits")

    shingles = {
        s: [(r["id"], _shingles(r["request"]), r["request"]) for r in recs]
        for s, recs in corpus.items()
    }
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            best = (0.0, None, None)
            over = 0
            for ida, sa, ta in shingles[a]:
                for idb, sb, tb in shingles[b]:
                    j = _jaccard(sa, sb)
                    if j >= NEAR_DUPLICATE_THRESHOLD:
                        over += 1
                    if j > best[0]:
                        best = (j, (ida, ta), (idb, tb))
            entry: dict[str, Any] = {
                "max_jaccard": round(best[0], 4),
                "pairs_at_or_above_threshold": over,
                "threshold": NEAR_DUPLICATE_THRESHOLD,
            }
            if include_texts and best[1] and "test" not in (a, b):
                entry["closest_pair"] = [best[1], best[2]]
            report["near_duplicates"][f"{a}~{b}"] = entry

    report["checks"] = {
        "passed": not failures,
        "failures": failures,
        "connected_components": len(component_splits),
        "components_spanning_splits": mixed_components,
    }
    return report
