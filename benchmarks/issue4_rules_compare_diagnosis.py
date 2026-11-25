"""EXP-012: train-only diagnosis of the rule baseline's Compare behaviour (Issue #4, PR #8 review).

    uv run python benchmarks/issue4_rules_compare_diagnosis.py > \
        docs/evidence/experiments/EXP-012-rules-compare-train-diagnosis/results.json

Reads only the committed ``train.jsonl`` (hash verified). No validation or test file, template
or prediction is read; no rule is changed. For every train example whose gold involves Compare
(a ``compare`` PLAN or a ``CLARIFY`` with operation hint ``compare``) it records which operation
cues of ``axiom.rules/0.1`` fire, and asks, leaving one request lineage out at a time, whether the
example would still contain a Compare cue that occurs in some *other* train Compare lineage. A low
leave-one-lineage-out rate means the Compare vocabulary is lineage-specific: it reproduces the
training phrasings rather than covering new ones. Writes JSON to stdout, no request texts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue3_common import git_state  # noqa: E402

from axiom_router.rules import OP_CUES, RULES_VERSION, _has, _norm_words  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "axbench" / "v0.1-draft" / "train.jsonl"
TRAIN_SHA256 = "fd885f1535d31983771493b6aa8e27ea3e9e0182ec58f385d96f1c686f9c2f3e"


def _cues(words: list[str], op: str) -> set[str]:
    return {" ".join(c) for c in OP_CUES[op] if _has(words, (c,))}


def main() -> int:
    data = TRAIN.read_bytes()
    if hashlib.sha256(data).hexdigest() != TRAIN_SHA256:
        raise SystemExit("train.jsonl does not match the pinned hash")
    train = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    involved = [r for r in train if r["gold"].get("operation") == "compare"]
    per_example = []
    for r in involved:
        words = _norm_words(r["request"])
        per_example.append(
            {
                "lineage": r["lineage_id"],
                "gold_action": r["gold"]["action"],
                "language": r["language"],
                "compare_cues": sorted(_cues(words, "compare")),
                "inspect_cues": sorted(_cues(words, "inspect")),
            }
        )
    by_lineage: dict[str, list[dict]] = defaultdict(list)
    for e in per_example:
        by_lineage[e["lineage"]].append(e)
    cue_lineages: dict[str, set[str]] = defaultdict(set)
    for e in per_example:
        for cue in e["compare_cues"]:
            cue_lineages[cue].add(e["lineage"])
    lineages = {}
    lolo_hits = 0
    for lin, exs in sorted(by_lineage.items()):
        others = {c for c, lins in cue_lineages.items() if lins - {lin}}
        covered = sum(1 for e in exs if set(e["compare_cues"]) & others)
        lolo_hits += covered
        lineages[lin] = {
            "n": len(exs),
            "gold_action": dict(Counter(e["gold_action"] for e in exs)),
            "language": dict(sorted(Counter(e["language"] for e in exs).items())),
            "with_any_compare_cue": sum(1 for e in exs if e["compare_cues"]),
            "with_an_inspect_cue": sum(1 for e in exs if e["inspect_cues"]),
            "compare_cues_used": dict(
                sorted(Counter(c for e in exs for c in e["compare_cues"]).items())
            ),
            "inspect_cues_seen": dict(
                sorted(Counter(c for e in exs for c in e["inspect_cues"]).items())
            ),
            "still_covered_by_other_lineages_cues": covered,
        }
    n = len(per_example)
    result = {
        "experiment": "EXP-012",
        "git": git_state(),
        "rules_version": RULES_VERSION,
        "inputs": {"train.jsonl": TRAIN_SHA256, "validation_or_test_read": False},
        "n_train_examples_with_compare_gold": n,
        "compare_cue_list_size": len(OP_CUES["compare"]),
        "compare_cues_matched_in_train": len(cue_lineages),
        "cue_to_lineage_count": {c: len(lins) for c, lins in sorted(cue_lineages.items())},
        "cues_in_only_one_lineage": sorted(c for c, lins in cue_lineages.items() if len(lins) == 1),
        "leave_one_lineage_out_covered": {"count": lolo_hits, "n": n},
        "with_an_inspect_cue": sum(1 for e in per_example if e["inspect_cues"]),
        "lineages": lineages,
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
