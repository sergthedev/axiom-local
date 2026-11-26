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

Revision 1 of the freeze (lead's D18 review): an unreadable or malformed input never ends the
script with a traceback or an empty file. Every JSONL line is kept (a malformed line is recorded in
place), an unreadable JSON file is reported as a problem of its section, and the result is a
structured ``invalid_run`` report; if the analysis itself raised, a minimal ``invalid_run`` report
with the error and the input hashes is written instead. The output is strict JSON (no NaN or
infinity) and is written in one piece after it has been fully built.

Revision 2 (lead's re-review): every physical line is kept, each decoded on its own (an undecodable
or blank line is a marker); a failure report keeps the safety counts and coverage already computed,
or recomputes them from the parsed records.
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
                  "monitor.jsonl", "monitor_ready.json")  # fmt: skip


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _rel(path: Path) -> str | None:
    """Repository-relative path, or ``None`` outside the repository (never an absolute path)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _label(path: Path) -> str:
    return _rel(path) or f"<outside the repository>/{path.parent.name}/{path.name}"


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _sha(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def committed_sha(path: Path) -> str | None:
    """SHA-256 of the file's bytes in ``HEAD``, or ``None`` if it is not committed."""
    rel = _rel(path)
    if rel is None:
        return None
    proc = _git("show", f"HEAD:{rel}")
    return _sha(proc.stdout) if proc.returncode == 0 else None


def changed_since(approved: str, commit: Any) -> list[str] | None:
    """Paths changed from the approved commit to ``commit``; ``None`` if it is not an ancestor."""
    if not isinstance(commit, str) or not commit:
        return None
    if _git("merge-base", "--is-ancestor", approved, commit).returncode != 0:
        return None
    proc = _git("diff", "--name-only", approved, commit)
    return proc.stdout.decode().split() if proc.returncode == 0 else None


def read_json(data: bytes | None, label: str, problems: list[str]) -> Any:
    """The parsed file, or ``None`` with a problem (missing, not UTF-8 or not JSON)."""
    if data is None:
        problems.append(f"{label}: missing")
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        problems.append(f"{label}: unreadable JSON ({type(exc).__name__})")
        return None


def read_jsonl(data: bytes | None, label: str, problems: list[str]) -> list[Any] | None:
    """Every line kept in order (malformed lines as ``Malformed``), or ``None`` if missing."""
    if data is None:
        problems.append(f"{label}: missing")
        return None
    return ax.parse_jsonl_lines(data)


def read_sequence(run: Path, problems: list[str]) -> ax.SequenceRecord:
    status = _read(run / "exit_status.txt")
    try:
        text = None if status is None else status.decode("utf-8").strip()
    except UnicodeDecodeError:
        text = None
    return ax.SequenceRecord(
        exit_status=text,
        monitor_stop=(run / "monitor_stop.json").exists(),
        done=(run / "sequence_done.txt").is_file(),
        baseline=read_json(_read(run / "baseline.json"), _label(run / "baseline.json"), problems),
        monitor=read_jsonl(_read(run / "monitor.jsonl"), _label(run / "monitor.jsonl"), problems),
        monitor_ready=read_json(_read(run / "monitor_ready.json"),
                                _label(run / "monitor_ready.json"), problems),
    )  # fmt: skip


def _commit(meas: Any) -> Any:
    return ax._obj(ax._obj(ax._obj(meas).get("environment")).get("git")).get("commit")


def build_inputs(approved: str, train_run: Path, eval_run: Path) -> ax.Inputs:
    raw = {
        "train_meas": train_run / "train" / "measurements.json",
        "train_log": train_run / "train" / "training_log.jsonl",
        "eval_meas": eval_run / "eval" / "main_measurements.json",
        "warmup": eval_run / "eval" / "main_warmup.jsonl",
        "validation": eval_run / "eval" / "main_validation.jsonl",
    }
    data = {k: _read(p) for k, p in raw.items()}
    evidence = [*raw.values(), *(r / f for r in (train_run, eval_run) for f in SEQUENCE_FILES)]
    preserved = {_label(p): (_sha(_read(p)), committed_sha(p)) for p in evidence}
    found: dict[str, list[str]] = {"training": [], "evaluation": [], "adapter_records": []}
    train_meas = read_json(data["train_meas"], _label(raw["train_meas"]), found["training"])
    train_log = read_jsonl(data["train_log"], _label(raw["train_log"]), found["training"])
    eval_meas = read_json(data["eval_meas"], _label(raw["eval_meas"]), found["evaluation"])
    warmup = read_jsonl(data["warmup"], _label(raw["warmup"]), found["evaluation"])
    records = read_jsonl(data["validation"], _label(raw["validation"]), found["adapter_records"])
    train_seq = read_sequence(train_run, found["training"])
    eval_seq = read_sequence(eval_run, found["evaluation"])

    manifest_bytes = (CORPUS / "MANIFEST.json").read_bytes()
    _, corpus = load_corpus(CORPUS)
    val = corpus["validation"]
    comparator_bytes = {name: _read(ROOT / path) for name, (path, _) in ax.COMPARATORS.items()}
    results_bytes = _read(ROOT / ax.EXP009_RESULTS[0])
    head = git_state()
    changed = {
        "training commit": changed_since(approved, _commit(train_meas)),
        "evaluation commit": changed_since(approved, _commit(eval_meas)),
        "analysis HEAD": changed_since(approved, head["commit"]),
    }
    if head["dirty"] is not False:
        changed["analysis HEAD"] = None  # tracked files modified: not the committed code
    return ax.Inputs(
        approved_commit=approved,
        corpus_splits=sorted(corpus),
        corpus_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        validation=val,
        warmup_ids=warmup_ids(corpus["train"]),
        train_meas=train_meas,
        train_log=train_log,
        train_sequence=train_seq,
        eval_meas=eval_meas,
        eval_sequence=eval_seq,
        warmup_records=warmup,
        adapter_records=records,
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
        input_problems=found,
    )  # fmt: skip


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      default=str)  # fmt: skip


def report(approved: str, train_run: Path, eval_run: Path) -> str:
    """The analysis as strict JSON text. :func:`axbench.exp010analysis.analyze` never raises; if
    reading the inputs or finishing the report raised, the result is a structured ``invalid_run``
    that keeps any safety counts already computed (never discarded, never NaN)."""
    inputs = result = None
    try:
        inputs = build_inputs(approved, train_run, eval_run)
        result = ax.analyze(inputs)
        result["git"] = git_state()
        result["corpus_version"] = "axbench/0.1-ai-reviewed"
        return _dump(result)
    except Exception as exc:  # noqa: BLE001 - reported as an invalid run, never a traceback
        safety = ax._obj(ax._obj(ax._obj(result).get("safety")).get("adapter")) or None
        if safety is None and inputs is not None:
            try:
                safety = ax.safety_scan({e["id"]: e for e in inputs.validation},
                                        inputs.adapter_records)  # fmt: skip
            except Exception:  # noqa: BLE001 - the failure report must still be written
                safety = None
        hashes = inputs.input_hashes if inputs is not None else None
        return _dump(ax.failure_report(approved, hashes, safety, f"{type(exc).__name__}: {exc}"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-commit", required=True, help="the head cleared for the pilot")
    parser.add_argument("--train-run", required=True, help="the pilot's sequence directory")
    parser.add_argument("--eval-run", required=True, help="the evaluation's sequence directory")
    args = parser.parse_args()
    text = report(args.approved_commit, Path(args.train_run), Path(args.eval_run))
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
