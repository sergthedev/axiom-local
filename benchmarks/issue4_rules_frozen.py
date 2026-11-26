"""EXP-013: the frozen rule baseline on the frozen AI-reviewed AXBENCH corpus (Issue #4).

Run from the repository root on a clean checkout of the commit that contains the pre-registration:

    EXP=docs/evidence/experiments/EXP-013-rules-frozen-ai-reviewed
    uv run python benchmarks/issue4_rules_frozen.py \
        --predictions $EXP/rules_validation.jsonl > $EXP/results.json

Loads ``axbench/0.1-ai-reviewed`` through its verified manifest (manifest SHA-256 pinned below),
runs ``axiom.rules/0.1`` once on validation and train, scores with ``axbench.score/0.2`` and
harness 0.2, and compares each validation example with EXP-011 (the same rules and scorer on the
draft). The final test split is not generated, read or scored. JSON to stdout, no test texts.
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

from axbench.corpus import load_corpus  # noqa: E402
from axbench.evaluate import HARNESS_VERSION, evaluate, rule_router, run_router  # noqa: E402
from axbench.score import SCORE_VERSION  # noqa: E402
from axiom_router.rules import RULES_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "axbench" / "v0.1-ai-reviewed"
MANIFEST_SHA256 = "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"
EXP011 = ROOT / "docs" / "evidence" / "experiments" / "EXP-011-axbench-rescore-v02" / "results.json"


def _jsonl(records) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    observed = hashlib.sha256((CORPUS / "MANIFEST.json").read_bytes()).hexdigest()
    if observed != MANIFEST_SHA256:
        raise SystemExit("frozen corpus manifest does not match the pre-registered hash")
    manifest, corpus = load_corpus(CORPUS)  # verifies file hashes and counts
    router = rule_router()
    out = {}
    for split in ("validation", "train"):
        preds = run_router(router, corpus[split], system="rules")
        if split == "validation":
            Path(args.predictions).write_bytes(_jsonl(preds))
        report = evaluate(corpus[split], preds, system=f"rules {RULES_VERSION}", split=split)
        report["role"] = (
            "pre-registered measurement" if split == "validation" else "development reference"
        )
        if split == "train":
            report.pop("per_example")
        out[split] = report
    old = json.loads(EXP011.read_text(encoding="utf-8"))["validation"]["per_example"]
    old_by_id = {r["id"]: r for r in old}
    changed = [
        {
            "id": r["id"],
            "exp011_exact": old_by_id[r["id"]]["exact_match"],
            "exp013_exact": r["exact_match"],
        }
        for r in out["validation"]["per_example"]
        if r["exact_match"] != old_by_id[r["id"]]["exact_match"]
        or r["predicted"] != old_by_id[r["id"]]["predicted"]
    ]
    result = {
        "experiment": "EXP-013",
        "git": git_state(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": observed,
        "corpus_files": manifest["files"],
        "rules_version": RULES_VERSION,
        "harness_version": HARNESS_VERSION,
        "score_version": SCORE_VERSION,
        "validation_examples_differing_from_exp011": changed,
        "validation": out["validation"],
        "train": out["train"],
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
