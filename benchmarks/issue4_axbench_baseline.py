"""EXP-008: AXBENCH v0.1-draft audit and the frozen rule baseline on validation (Issue #4).

Run from the repository root on a clean checkout of the commit that contains the pre-registration:

    EXP=docs/evidence/experiments/EXP-008-axbench-rule-baseline
    uv run python benchmarks/issue4_axbench_baseline.py \
        --predictions $EXP/rules_validation.jsonl > $EXP/results.json

Part A: regenerate the development corpus and compare it byte for byte with the committed files;
run the split/leakage audit (test split in memory with the public development seed, no test text
in the output) and the rule-baseline contamination audit.
Part R: one run of the frozen rule router (``axiom.rules/0.1``) on the 160 validation examples,
scored with ``axbench.score`` and passed through the real Guard (never confirmed). The training
split is scored too, labelled as the development reference (the rules were tuned on it).
The final test split is not generated, read or scored.

Writes JSON to stdout; no absolute paths and no test texts.

This script reproduces the EXP-008 record only at commit ``857d4ee`` (scorer and harness 0.1). At
later commits it scores with the corrected scorer 0.2; the EXP-008 predictions rescored under 0.2
are EXP-011 (``benchmarks/issue4_rescore_v02.py``).
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

from axbench.audit import audit  # noqa: E402
from axbench.contamination import contamination_findings  # noqa: E402
from axbench.evaluate import HARNESS_VERSION, evaluate, rule_router, run_router  # noqa: E402
from axbench.generate import generate_all_for_audit, generate_development_corpus  # noqa: E402
from axiom_router.rules import RULES_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data" / "axbench" / "v0.1-draft"


def _jsonl(records) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ).encode("utf-8")


def part_a() -> dict:
    corpus = {s: [e.to_record() for e in exs] for s, exs in generate_development_corpus().items()}
    manifest = json.loads((DRAFT / "MANIFEST.json").read_text(encoding="utf-8"))
    files = {}
    for split, records in corpus.items():
        regenerated = hashlib.sha256(_jsonl(records)).hexdigest()
        committed = hashlib.sha256((DRAFT / f"{split}.jsonl").read_bytes()).hexdigest()
        files[split] = {
            "n": len(records),
            "regenerated_sha256": regenerated,
            "committed_sha256": committed,
            "manifest_sha256": manifest["files"][f"{split}.jsonl"]["sha256"],
        }
    full = {s: [e.to_record() for e in exs] for s, exs in generate_all_for_audit().items()}
    report = audit(full)
    return {
        "files": files,
        "test_file_in_repository": (DRAFT / "test.jsonl").exists(),
        "audit_checks": report["checks"],
        "near_duplicates": {
            k: {kk: vv for kk, vv in v.items() if kk != "closest_pair"}
            for k, v in report["near_duplicates"].items()
        },
        "split_summaries": report["splits"],
        "contamination_findings": contamination_findings(),
    }


def part_r(predictions_path: Path | None) -> dict:
    corpus = {s: [e.to_record() for e in exs] for s, exs in generate_development_corpus().items()}
    router = rule_router()
    out = {}
    for split in ("validation", "train"):
        preds = run_router(router, corpus[split], system="rules")
        if split == "validation" and predictions_path is not None:
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_path.write_bytes(_jsonl(preds))
        report = evaluate(corpus[split], preds, system=f"rules {RULES_VERSION}", split=split)
        report["role"] = (
            "pre-registered measurement" if split == "validation" else "development reference"
        )
        if split == "train":
            report.pop("per_example")
        out[split] = report
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="where to write raw validation predictions")
    args = parser.parse_args()
    result = {
        "experiment": "EXP-008",
        "git": git_state(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "rules_version": RULES_VERSION,
        "harness_version": HARNESS_VERSION,
        "part_a": part_a(),
        "part_r": part_r(Path(args.predictions) if args.predictions else None),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
