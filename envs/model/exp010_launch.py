"""EXP-010 detached launcher and external resource monitor (model environment, macOS).

    uv run --locked python exp010_launch.py start --phase smoke --model-dir <dir> \\
        --data-dir <export dir> --adapter-dir <new dir outside the repository> \\
        --out <new run dir under docs/evidence/experiments/EXP-010-lora-pilot/runs> \\
        --expected-commit <reviewed sha>

``start`` returns at once. It starts ``sequence`` in a new session (``start_new_session``), under
``nohup`` (hang-ups ignored) and ``caffeinate -i`` (no idle sleep; no experimental setting
changes), with stdin closed and output appended to ``sequence_console.txt`` in the run directory,
so the run survives the end of the Claude Code session that started it (EXP-009 attempt-1 lesson).

``sequence`` (never called by hand) writes ``sequence_log.txt``, starts the training process
(records in ``<run>/train/``) and the monitor, waits, stops the monitor, runs the fresh-process
adapter reload check after a completed smoke (``<run>/train/reload/``), and writes
``exit_status.txt`` (exit code and UTC time) last. It never retries.

``monitor`` samples every 30 s, in its own process: swap used, memory-pressure level, the
``memory_pressure`` free percentage, power source and the training process's resident size
(``ps``). It writes one JSON line per sample to ``monitor.jsonl`` and, on a protocol stop
condition (swap growth above 2 GB since its first sample, critical pressure, loss of mains power),
writes ``monitor_stop.json`` and sends SIGTERM to the training process, which records the stop.
The training process also checks the same rules after every optimizer update and evaluation.

File names end in ``.txt``/``.json``/``.jsonl`` so that the repository's ``*.log`` ignore rule
cannot hide evidence (the EXP-009 packaging correction).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from exp009_run import snapshot  # noqa: E402

from axbench.runwatch import ProcessMonitor  # noqa: E402

INTERVAL_SECONDS = 30
RUNS = REPO / "docs" / "evidence" / "experiments" / "EXP-010-lora-pilot" / "runs"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A finished child that was not reaped yet is a zombie; treat it as gone.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def _rss_bytes(pid: int) -> int | None:
    proc = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = proc.stdout.strip()
    return int(text) * 1024 if text.isdigit() else None  # ps reports KiB


def cmd_monitor(args) -> int:
    out = Path(args.out)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    first = snapshot()
    monitor = ProcessMonitor(baseline={k: first[k] for k in ("swap_used", "pressure_level",
                                                              "power_source")})  # fmt: skip
    with open(out / "monitor.jsonl", "ab") as fh:
        n = 0
        while True:
            snap = first if n == 0 else snapshot()
            rec: dict[str, Any] = {"sample": n, **snap, "trainer_rss_bytes": _rss_bytes(args.pid)}
            reason = monitor.check("monitor", {k: snap[k] for k in ("swap_used", "pressure_level",
                                                                     "power_source")})  # fmt: skip
            rec["stop"] = reason
            fh.write((json.dumps(rec, sort_keys=True) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
            if reason:
                (out / "monitor_stop.json").write_bytes(
                    (
                        json.dumps(
                            {"time_utc": _utc(), "reason": reason, "sample": n}, sort_keys=True
                        )
                        + "\n"
                    ).encode()  # fmt: skip
                )
                if _alive(args.pid):
                    os.kill(args.pid, signal.SIGTERM)
                return 0
            n += 1
            for _ in range(INTERVAL_SECONDS * 10):
                if stop["flag"] or not _alive(args.pid):
                    break
                time.sleep(0.1)
            if stop["flag"] or not _alive(args.pid):
                final = snapshot()
                fh.write((json.dumps({"sample": n, "final": True, **final}, sort_keys=True)
                          + "\n").encode())  # fmt: skip
                return 0


def _log(out: Path, line: str) -> None:
    with open(out / "sequence_log.txt", "a", encoding="utf-8") as fh:
        fh.write(f"== {_utc()} {line}\n")


def _train_argv(args, out: Path) -> list[str]:
    argv = [sys.executable, str(HERE / "exp010_train.py"), "train", "--phase", args.phase,
            "--model-dir", args.model_dir, "--data-dir", args.data_dir,
            "--adapter-dir", args.adapter_dir, "--out", str(out),
            "--expected-commit", args.expected_commit]  # fmt: skip
    if args.grad_checkpoint:
        argv.append("--grad-checkpoint")
    return argv


# Plain console text in the evidence files (MLX-LM's progress display honours NO_COLOR).
CHILD_ENV = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}


def cmd_sequence(args) -> int:
    out = Path(args.out)
    _log(out, f"start phase {args.phase} pid {os.getpid()} commit {args.expected_commit}")
    rc = 0
    console = open(out / "train_console.txt", "ab")  # noqa: SIM115 - closed below
    train_dir = out / "train"  # the training process requires a new, empty directory
    trainer = subprocess.Popen(_train_argv(args, train_dir), cwd=HERE, env=CHILD_ENV,
                               stdin=subprocess.DEVNULL, stdout=console,
                               stderr=subprocess.STDOUT)  # fmt: skip
    _log(out, f"train started pid {trainer.pid}")
    monitor = subprocess.Popen(
        [sys.executable, str(HERE / "exp010_launch.py"), "monitor", "--pid", str(trainer.pid),
         "--out", str(out)],
        cwd=HERE, env=CHILD_ENV, stdin=subprocess.DEVNULL, stdout=console,
        stderr=subprocess.STDOUT,
    )  # fmt: skip
    rc = trainer.wait()
    _log(out, f"train exited rc={rc}")
    if monitor.poll() is None:
        monitor.send_signal(signal.SIGTERM)
    monitor.wait()
    if rc == 0 and args.phase == "smoke":
        _log(out, "reload check (fresh process, 5 train examples)")
        rc = subprocess.run(
            [sys.executable, str(HERE / "exp010_train.py"), "reload", "--model-dir",
             args.model_dir, "--adapter-dir", args.adapter_dir, "--run-dir", str(train_dir),
             "--expected-commit", args.expected_commit],
            cwd=HERE, env=CHILD_ENV, stdin=subprocess.DEVNULL, stdout=console,
            stderr=subprocess.STDOUT,
        ).returncode  # fmt: skip
        _log(out, f"reload exited rc={rc}")
    console.close()
    _log(out, f"end rc={rc}")
    (out / "exit_status.txt").write_text(f"{rc} {_utc()}\n", encoding="utf-8")
    return rc


def cmd_start(args) -> int:
    out = Path(args.out).resolve()
    if RUNS not in out.parents:
        raise SystemExit("--out must be a new directory under the EXP-010 runs directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit("--out is not empty; a new run needs a new directory")
    out.mkdir(parents=True, exist_ok=True)
    argv = ["nohup", "caffeinate", "-i", sys.executable, str(HERE / "exp010_launch.py"),
            "sequence", "--phase", args.phase, "--model-dir", args.model_dir,
            "--data-dir", args.data_dir, "--adapter-dir", args.adapter_dir, "--out", str(out),
            "--expected-commit", args.expected_commit]  # fmt: skip
    if args.grad_checkpoint:
        argv.append("--grad-checkpoint")
    with open(out / "sequence_console.txt", "ab") as console:
        proc = subprocess.Popen(argv, cwd=HERE, env=CHILD_ENV, stdin=subprocess.DEVNULL,
                                stdout=console, stderr=subprocess.STDOUT,
                                start_new_session=True)  # fmt: skip
    print(json.dumps({"started": True, "sequence_pid": proc.pid, "phase": args.phase,
                      "run_dir": str(out.relative_to(REPO))}))  # fmt: skip
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("start", "sequence"):
        p = sub.add_parser(name)
        p.add_argument("--phase", required=True, choices=("smoke", "pilot"))
        for opt in ("--model-dir", "--data-dir", "--adapter-dir", "--out", "--expected-commit"):
            p.add_argument(opt, required=True)
        p.add_argument("--grad-checkpoint", action="store_true")
    p = sub.add_parser("monitor")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    return {"start": cmd_start, "sequence": cmd_sequence, "monitor": cmd_monitor}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
