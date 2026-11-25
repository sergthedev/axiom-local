"""EXP-011: rescore the EXP-008 rule-baseline predictions with ``axbench.score/0.2`` (Issue #4).

Run from the repository root on a clean checkout of the commit that contains the pre-registration:

    EXP=docs/evidence/experiments/EXP-011-axbench-rescore-v02
    uv run python benchmarks/issue4_rescore_v02.py > $EXP/results.json

Reads the committed validation and train files and the committed EXP-008 predictions and results;
verifies their SHA-256 against the values pinned in the pre-registration; checks that the frozen
rule router ``axiom.rules/0.1`` still produces the same raw validation outputs; scores the EXP-008
predictions (not new ones) with scorer 0.2 and harness 0.2; and compares every reported metric with
the EXP-008 values (scorer 0.1). EXP-008 files are only read. The final test split is not
generated, read or scored. Writes JSON to stdout; no absolute paths and no test texts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue3_common import git_state  # noqa: E402

from axbench.evaluate import HARNESS_VERSION, evaluate, rule_router, run_router  # noqa: E402
from axbench.score import SCORE_VERSION, load_jsonl  # noqa: E402
from axiom_router.rules import RULES_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data" / "axbench" / "v0.1-draft"
EXP008 = ROOT / "docs" / "evidence" / "experiments" / "EXP-008-axbench-rule-baseline"
PINNED = {  # from the EXP-011 pre-registration
    "validation.jsonl": "87771e489e186e4b4753adcf35916f41541355ed159c8d9bfe321d9d100ed9e4",
    "train.jsonl": "fd885f1535d31983771493b6aa8e27ea3e9e0182ec58f385d96f1c686f9c2f3e",
    "rules_validation.jsonl": "40c9ba2208d016120a6683640e5dd9ce2daf4f79f8fa74afd3ab3c3655fb2e65",
    "results.json": "2105fb23959a6b6f8987ec38e926fa7378ef23235e07d507a22703319e7ffaf4",
}
COMPARED = (  # (label, path in raw_candidate)
    ("exact_match", ("primary_semantic_exact_match", "count")),
    ("json_valid", ("json_valid", "count")),
    ("candidate_schema_valid", ("candidate_schema_valid", "count")),
    ("semantic_valid", ("semantic_valid", "count")),
    ("action_accuracy", ("action_accuracy", "count")),
    ("action_macro_f1", ("action_macro_f1",)),
    ("plan_f1", ("action_per_class", "PLAN", "f1")),
    ("clarify_f1", ("action_per_class", "CLARIFY", "f1")),
    ("reject_f1", ("action_per_class", "REJECT", "f1")),
    ("plan_operation_accuracy", ("plan_operation_accuracy", "count")),
    ("clarify_recall", ("clarify_recall", "count")),
    ("clarify_reason_accuracy", ("clarify_reason_accuracy", "count")),
    ("reject_recall", ("reject_recall", "count")),
    ("reject_reason_accuracy", ("reject_reason_accuracy", "count")),
    ("false_reject_among_supported", ("false_reject_among_supported", "count")),
    ("plan_coverage_all", ("plan_coverage_all", "count")),
    ("wrong_executable_proposals", ("wrong_executable_proposals", "per_all", "count")),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def main() -> int:
    argparse.ArgumentParser().parse_args()
    paths = {
        "validation.jsonl": DRAFT / "validation.jsonl",
        "train.jsonl": DRAFT / "train.jsonl",
        "rules_validation.jsonl": EXP008 / "rules_validation.jsonl",
        "results.json": EXP008 / "results.json",
    }
    hashes = {k: {"observed": _sha(p), "pinned": PINNED[k]} for k, p in paths.items()}
    if any(v["observed"] != v["pinned"] for v in hashes.values()):
        json.dump({"experiment": "EXP-011", "input_hashes": hashes, "aborted": True}, sys.stdout)
        sys.stdout.write("\n")
        return 1
    validation = load_jsonl(paths["validation.jsonl"])
    train = load_jsonl(paths["train.jsonl"])
    committed = load_jsonl(paths["rules_validation.jsonl"])
    old = json.loads(paths["results.json"].read_text(encoding="utf-8"))["part_r"]

    # C2: the frozen rules still produce the committed raw outputs (timings excluded).
    rerun = run_router(rule_router(), validation, system="rules")
    fields = ("example_id", "status", "raw_output")
    same = sum(
        all(a.get(f) == b.get(f) for f in fields) for a, b in zip(rerun, committed, strict=True)
    )

    new_val = evaluate(validation, committed, system=f"rules {RULES_VERSION}", split="validation")
    new_val["role"] = "EXP-008 predictions rescored with scorer 0.2"
    train_preds = run_router(rule_router(), train, system="rules")
    new_train = evaluate(train, train_preds, system=f"rules {RULES_VERSION}", split="train")
    new_train["role"] = "development reference (the rules were tuned on train)"
    new_train.pop("per_example")

    comparison = {}
    for split, new in (("validation", new_val), ("train", new_train)):
        rows = {}
        for label, path in COMPARED:
            a = _get(old[split]["raw_candidate"], path)
            b = _get(new["raw_candidate"], path)
            rows[label] = {"score_0.1": a, "score_0.2": b, "changed": a != b}
        rows["after_guard_awaiting_correct"] = {
            "score_0.1": old[split]["after_guard"]["awaiting_confirmation_correct"],
            "score_0.2": new["after_guard"]["awaiting_confirmation_correct"],
        }
        rows["after_guard_awaiting_correct"]["changed"] = (
            rows["after_guard_awaiting_correct"]["score_0.1"]
            != rows["after_guard_awaiting_correct"]["score_0.2"]
        )
        comparison[split] = rows
    old_pe = {r["id"]: r for r in old["validation"]["per_example"]}
    exact_flips = sorted(
        r["id"]
        for r in new_val["per_example"]
        if r["exact_match"] != old_pe[r["id"]]["exact_match"]
    )
    result = {
        "experiment": "EXP-011",
        "git": git_state(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "rules_version": RULES_VERSION,
        "harness_version": HARNESS_VERSION,
        "score_version": SCORE_VERSION,
        "input_hashes": hashes,
        "rules_regenerate_committed_outputs": {"identical": same, "n": len(committed)},
        "validation_examples_with_changed_exact_match": exact_flips,
        "comparison_with_exp008": comparison,
        "validation": new_val,
        "train": new_train,
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
