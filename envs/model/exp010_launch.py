"""EXP-010 detached launcher (model environment, macOS): a thin wrapper over ``axbench.supervise``.

    uv run --locked python exp010_launch.py start --phase smoke --model-dir <dir> \\
        --data-dir <export dir> --adapter-dir <new dir outside the repository> \\
        --out <new run dir under docs/evidence/experiments/EXP-010-lora-pilot/runs> \\
        --expected-commit <reviewed sha>

``start`` checks the arguments, then returns at once. The supervised sequence runs in a new
session (``start_new_session``), under ``nohup`` (hang-ups ignored) and ``caffeinate -i`` (no idle
sleep; no experimental setting changes), with stdin closed and output appended to
``sequence_console.txt``, so the run survives the end of the Claude Code session that started it
(EXP-009 attempt-1 lesson).

Revision 1 (lead pre-run review, correction A), implemented and tested in ``axbench.supervise``:
one common pre-load baseline and preflight before anything starts (``baseline.json``); the
external monitor (30 s sampling, the unchanged EXP-009/EXP-010 thresholds) must be ready before
the first step; the training process and, after a smoke, the fresh-process reload check run as
supervised steps that both receive ``--baseline``; a monitor stop, a monitor failure (malformed or
missing sample, crash, stale heartbeat) or an exception terminates the running step (SIGTERM, a
bounded grace period, then a forced kill); a stop marker always overrides success; nothing is
retried; ``exit_status.txt`` is written last in every case.

Records: ``<run>/train/`` (training) and ``<run>/train/reload/`` (smoke reload). File names end in
``.txt``/``.json``/``.jsonl`` so that the repository's ``*.log`` ignore rule cannot hide evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from axbench import supervise  # noqa: E402

RUNS = REPO / "docs" / "evidence" / "experiments" / "EXP-010-lora-pilot" / "runs"
TRAIN = HERE / "exp010_train.py"


def steps(args, out: Path) -> list[tuple[str, list[str]]]:
    baseline = str(out / supervise.BASELINE)
    train_dir = out / "train"  # the training process requires a new, empty directory
    train = [sys.executable, str(TRAIN), "train", "--phase", args.phase,
             "--model-dir", args.model_dir, "--data-dir", args.data_dir,
             "--adapter-dir", args.adapter_dir, "--out", str(train_dir),
             "--expected-commit", args.expected_commit, "--baseline", baseline]  # fmt: skip
    if args.grad_checkpoint:
        train += ["--grad-checkpoint", "--declared-change", args.declared_change]
    out_steps = [("train", train)]
    if args.phase == "smoke":
        out_steps.append(("reload", [sys.executable, str(TRAIN), "reload",
                                     "--model-dir", args.model_dir,
                                     "--adapter-dir", args.adapter_dir,
                                     "--run-dir", str(train_dir),
                                     "--expected-commit", args.expected_commit,
                                     "--baseline", baseline]))  # fmt: skip
    return out_steps


def cmd_start(args) -> int:
    out = Path(args.out).resolve()
    if RUNS not in out.parents:
        raise SystemExit("--out must be a new directory under the EXP-010 runs directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit("--out is not empty; a new run needs a new directory")
    adapter_dir = Path(args.adapter_dir).resolve()
    if REPO in adapter_dir.parents or (adapter_dir.exists() and any(adapter_dir.iterdir())):
        raise SystemExit("--adapter-dir must be a new or empty directory outside the repository")
    if bool(args.grad_checkpoint) != (args.declared_change == "C1"):
        raise SystemExit("--grad-checkpoint is predeclared change C1 only; pass both or neither")
    out.mkdir(parents=True, exist_ok=True)
    argv = supervise.sequence_argv(out, steps(args, out), "macos", supervise.INTERVAL_SECONDS,
                                   supervise.GRACE_SECONDS,
                                   supervise.READY_TIMEOUT_SECONDS)  # fmt: skip
    pid = supervise.start_detached(argv, out, supervise.child_env())
    print(json.dumps({"started": True, "sequence_pid": pid, "phase": args.phase,
                      "run_dir": str(out.relative_to(REPO))}))  # fmt: skip
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("--phase", required=True, choices=("smoke", "pilot"))
    for opt in ("--model-dir", "--data-dir", "--adapter-dir", "--out", "--expected-commit"):
        p.add_argument(opt, required=True)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="predeclared change C1 only (PRERUN D9), never for the initial smoke",
    )
    p.add_argument("--declared-change", choices=("C1",))
    args = parser.parse_args()
    return cmd_start(args)


if __name__ == "__main__":
    raise SystemExit(main())
