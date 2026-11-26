"""EXP-010 analysis (PRERUN D18): verify, score and decide, exactly as frozen before the pilot.

Run from the repository root in the Core environment, after the pilot's training records and the
evaluation's raw records have been committed unchanged (nothing is scored before that):

    EXP=docs/evidence/experiments/EXP-010-lora-pilot
    uv run python benchmarks/issue4_exp010_analyze.py --approved-commit <cleared sha> \\
        --train-run $EXP/runs/pilot --eval-run $EXP/runs/pilot_eval > $EXP/results.json

This script only reads files and Git facts and passes them to :func:`axbench.exp010analysis.analyze`
(the logic and its tests live there). Inputs: the training sequence directory (``exit_status.txt``,
``sequence_done.txt``, ``monitor_stop.json`` if any, ``train/measurements.json``,
``train/training_log.jsonl``), the evaluation sequence directory (the same sequence files and
``eval/main_measurements.json``, ``eval/main_warmup.jsonl``, ``eval/main_validation.jsonl``), the
pinned EXP-009 comparators and results, and the frozen corpus (train and validation only; the
sealed final test is never read). No MLX is needed. Integrity failures are reported in the result
and give ``invalid_run``; nothing is repaired, dropped or retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue3_common import git_state  # noqa: E402

from axbench import exp010analysis as ax  # noqa: E402
from axbench.corpus import load_corpus  # noqa: E402
from axbench.evaluate import rule_router, run_router  # noqa: E402
from axbench.runwatch import warmup_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "axbench" / "v0.1-ai-reviewed"
SEQUENCE_FILES = ("exit_status.txt", "sequence_done.txt", "sequence_log.txt", "baseline.json",
                  "monitor.jsonl")  # fmt: skip


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _sha(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def committed_sha(path: Path) -> str | None:
    """SHA-256 of the file's bytes in ``HEAD``, or ``None`` if it is not committed."""
    proc = _git("show", f"HEAD:{_rel(path)}")
    return _sha(proc.stdout) if proc.returncode == 0 else None


def changed_since(approved: str, commit: str | None) -> list[str] | None:
    """Paths changed from the approved commit to ``commit``; ``None`` if it is not an ancestor."""
    if not commit or _git("merge-base", "--is-ancestor", approved, commit).returncode != 0:
        return None
    proc = _git("diff", "--name-only", approved, commit)
    return proc.stdout.decode().split() if proc.returncode == 0 else None


def _json(data: bytes | None) -> Any:
    return {} if data is None else json.loads(data.decode("utf-8"))


def _jsonl(data: bytes | None) -> Any:
    return None if data is None else ax.parse_jsonl(data)


def _sequence(run: Path) -> tuple[str | None, bool, bool]:
    status = _read(run / "exit_status.txt")
    text = None if status is None else status.decode().strip()
    return text, (run / "monitor_stop.json").exists(), (run / "sequence_done.txt").is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-commit", required=True, help="the head cleared for the pilot")
    parser.add_argument("--train-run", required=True, help="the pilot's sequence directory")
    parser.add_argument("--eval-run", required=True, help="the evaluation's sequence directory")
    args = parser.parse_args()
    train_run, eval_run = Path(args.train_run).resolve(), Path(args.eval_run).resolve()

    raw = {
        "train_meas": train_run / "train" / "measurements.json",
        "train_log": train_run / "train" / "training_log.jsonl",
        "eval_meas": eval_run / "eval" / "main_measurements.json",
        "warmup": eval_run / "eval" / "main_warmup.jsonl",
        "validation": eval_run / "eval" / "main_validation.jsonl",
    }
    data = {k: _read(p) for k, p in raw.items()}
    evidence = [*raw.values(), *(r / f for r in (train_run, eval_run) for f in SEQUENCE_FILES)]
    preserved = {_rel(p): (_sha(_read(p)), committed_sha(p)) for p in evidence}

    manifest_bytes = (CORPUS / "MANIFEST.json").read_bytes()
    manifest, corpus = load_corpus(CORPUS)
    val = corpus["validation"]
    comparator_bytes = {name: _read(ROOT / path) for name, (path, _) in ax.COMPARATORS.items()}
    results_bytes = _read(ROOT / ax.EXP009_RESULTS[0])
    eval_meas = _json(data["eval_meas"])

    head = git_state()
    changed = {
        "training commit": changed_since(args.approved_commit, ((_json(data["train_meas"])
                                         .get("environment") or {}).get("git") or {})
                                         .get("commit")),
        "evaluation commit": changed_since(args.approved_commit, ((eval_meas.get("environment")
                                           or {}).get("git") or {}).get("commit")),
        "analysis HEAD": changed_since(args.approved_commit, head["commit"]),
    }  # fmt: skip
    if head["dirty"] is not False:
        changed["analysis HEAD"] = None  # tracked files modified: not the committed code

    inputs = ax.Inputs(
        approved_commit=args.approved_commit,
        corpus_splits=sorted(corpus),
        corpus_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        validation=val,
        warmup_ids=warmup_ids(corpus["train"]),
        train_meas=_json(data["train_meas"]),
        train_log=_jsonl(data["train_log"]) or [],
        train_sequence=_sequence(train_run),
        eval_meas=eval_meas,
        eval_sequence=_sequence(eval_run),
        warmup_records=_jsonl(data["warmup"]),
        adapter_records=_jsonl(data["validation"]),
        comparator_bytes=comparator_bytes,
        exp009_results_bytes=results_bytes,
        rules_rerun=run_router(rule_router(), val, system=ax.RULES),
        changed_since_approved=changed,
        preserved=preserved,
        input_hashes={**{p: w for p, (w, _) in preserved.items()},
                      **{path: _sha(comparator_bytes[n]) for n, (path, _) in
                         ax.COMPARATORS.items()},
                      ax.EXP009_RESULTS[0]: _sha(results_bytes),
                      "data/axbench/v0.1-ai-reviewed/MANIFEST.json": _sha(manifest_bytes)},
    )  # fmt: skip
    result = ax.analyze(inputs)
    result["git"] = head
    result["corpus_version"] = manifest["corpus_version"]
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
