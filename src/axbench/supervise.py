"""Supervised, detached execution of EXP-010 processes with an external resource monitor.

No MLX here; the model-environment launcher (``envs/model/exp010_launch.py``) supplies the real
training and reload commands, and the tests supply harmless dummy processes and scripted samples.

Lifecycle of one sequence (PRERUN D10, revision 1 after the lead's pre-run review, correction A):

1. One **common pre-load baseline** (swap used, memory-pressure level, power source) is sampled
   before anything starts, written to ``baseline.json`` and checked by the preflight (mains power,
   normal pressure); a refusal ends the sequence with exit code 4 and is recorded.
2. The **monitor** starts next, in its own process, and must write its first valid sample and
   ``monitor_ready.json`` within ``ready_timeout``; otherwise the sequence ends (exit code 5)
   before any step starts.
3. Steps (training, then the smoke reload check) run one at a time. The step's process ID is
   written to ``current_pid.txt`` so that the monitor can signal it. The sequence supervises both
   processes: if the monitor writes ``monitor_stop.json`` (a stop rule, or a sampling failure,
   which fails closed), exits, or stops writing samples (stale heartbeat), the running step is
   terminated: SIGTERM, a bounded grace period, then a forced kill.
4. A stop marker always overrides success: the sequence exit code is 6 if ``monitor_stop.json``
   exists, whatever the steps returned; 7 for a monitor failure; the first non-zero step code
   otherwise; 8 for an exception in the sequence itself. ``exit_status.txt`` is written last, in
   every case, and nothing is retried.

The monitor samples every ``interval`` seconds (30 s for EXP-010), validates each sample (a missing
or malformed field is a sampling failure), applies the EXP-009/EXP-010 resource rules against the
common baseline (swap growth above 2 GB, critical pressure, loss of mains power), and appends every
sample to ``monitor.jsonl``. Thresholds come from :mod:`axbench.runwatch` unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .runwatch import ProcessMonitor, parse_free_percentage, parse_swapusage, preflight

SUPERVISE_VERSION = "exp010.supervise/1"
INTERVAL_SECONDS = 30.0
READY_TIMEOUT_SECONDS = 120.0
GRACE_SECONDS = 120.0
POLL_SECONDS = 0.2

BASELINE = "baseline.json"
READY = "monitor_ready.json"
STOP = "monitor_stop.json"
PID = "current_pid.txt"
DONE = "sequence_done.txt"
SAMPLES = "monitor.jsonl"
LOG = "sequence_log.txt"
EXIT = "exit_status.txt"

RC_OK = 0
RC_STEP_FAILED = 3  # also a step ended by a signal that was not the sequence's own escalation
RC_REFUSED = 4
RC_MONITOR_NOT_READY = 5
RC_MONITOR_STOP = 6
RC_MONITOR_FAILURE = 7
RC_SEQUENCE_ERROR = 8

RULE_KEYS = ("swap_used", "pressure_level", "power_source")
PRESSURE_LEVELS = (1, 2, 4)


class MalformedSample(Exception):
    pass


class SequenceInterrupted(Exception):
    """SIGTERM to the sequence itself: the running step is stopped and the evidence written."""


def _raise_interrupted(signum, frame):  # noqa: ARG001
    raise SequenceInterrupted("SIGTERM received by the sequence")


# ------------------------------------------------------------------ samples


def validate_sample(sample: Any) -> dict[str, Any]:
    """A sample must carry the three rule fields with sane types, or it is a sampling failure."""
    if not isinstance(sample, dict):
        raise MalformedSample(f"sample is {type(sample).__name__}, not a mapping")
    swap, level, power = (sample.get(k) for k in RULE_KEYS)
    if not isinstance(swap, int) or isinstance(swap, bool) or swap < 0:
        raise MalformedSample(f"swap_used {swap!r}")
    if level not in PRESSURE_LEVELS or isinstance(level, bool):
        raise MalformedSample(f"pressure_level {level!r}")
    if not isinstance(power, str) or not power:
        raise MalformedSample(f"power_source {power!r}")
    return sample


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20).stdout


def macos_sample() -> dict[str, Any]:
    """swap, pressure level, power source and the free percentage (macOS commands)."""
    first = _run("pmset", "-g", "batt").splitlines()[0]
    m = re.search(r"'([^']+)'", first)
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
    swap = parse_swapusage(_run("sysctl", "-n", "vm.swapusage"))
    return {
        "time_utc": _utc(),
        "swap_used": swap["used_bytes"],
        "swap": swap,
        "pressure_level": int(_run("sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
        "power_source": m.group(1) if m else first.strip(),
        "memory_free_percentage": parse_free_percentage(mp),
    }


class ScriptedSampler:
    """Test sampler: one JSON value per line, used in order and the last one repeated.

    ``{"raise": "text"}`` raises, ``{"exit": n}`` ends the process at once with code ``n``
    (a crashing monitor), ``{"hang": seconds}`` blocks before the next item (a hung sampler, seen
    by the sequence as a stale heartbeat); anything else is returned as the sample (and validated
    by the caller).
    """

    def __init__(self, path: str) -> None:
        self.items = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        self.i = 0

    def __call__(self) -> Any:
        item = self.items[min(self.i, len(self.items) - 1)]
        self.i += 1
        if isinstance(item, dict) and "raise" in item:
            raise RuntimeError(item["raise"])
        if isinstance(item, dict) and "exit" in item:
            os._exit(int(item["exit"]))
        if isinstance(item, dict) and "hang" in item:
            time.sleep(float(item["hang"]))
            return self()
        return {"time_utc": _utc(), **item} if isinstance(item, dict) else item


def make_sampler(spec: str) -> Callable[[], Any]:
    if spec == "macos":
        return macos_sample
    if spec.startswith("file:"):
        return ScriptedSampler(spec[5:])
    raise ValueError(f"unknown sampler {spec!r}")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(obj, sort_keys=True, indent=2) + "\n").encode())
    os.replace(tmp, path)


def _append(path: Path, obj: Any) -> None:
    with open(path, "ab") as fh:
        fh.write((json.dumps(obj, sort_keys=True) + "\n").encode())
        fh.flush()
        os.fsync(fh.fileno())


def _rss_bytes(pid: int | None) -> int | None:
    if pid is None or sys.platform == "win32":
        return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=10).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) * 1024 if out.isdigit() else None


def _current_pid(out: Path) -> int | None:
    try:
        return int((out / PID).read_text().strip())
    except (OSError, ValueError):
        return None


def _signal(pid: int | None, sig: int) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# ------------------------------------------------------------------ monitor process


def monitor_main(out: Path, sampler: Callable[[], Any], interval: float) -> int:
    """Sample until the sequence is done or a stop/failure occurs (runs in its own process)."""
    term = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: term.__setitem__("flag", True))
    baseline = json.loads((out / BASELINE).read_text())
    monitor = ProcessMonitor(baseline={k: baseline[k] for k in RULE_KEYS})
    n = 0
    while True:
        pid = _current_pid(out)
        try:
            sample = validate_sample(sampler())
            reason = monitor.check("monitor", {k: sample[k] for k in RULE_KEYS})
        except Exception as exc:  # noqa: BLE001 - any sampling problem fails closed
            sample, reason = {"time_utc": _utc()}, f"monitor sampling failed: {exc!r}"
        _append(out / SAMPLES, {"sample": n, **sample, "target_pid_rss_bytes": _rss_bytes(pid),
                                "stop": reason})  # fmt: skip
        if n == 0 and reason is None:
            _write_json(out / READY, {"time_utc": _utc(), "interval_seconds": interval})
        if reason:
            _write_json(out / STOP, {"time_utc": _utc(), "reason": reason, "sample": n})
            _signal(_current_pid(out), signal.SIGTERM)
            return 2
        n += 1
        deadline = time.monotonic() + interval
        while time.monotonic() < deadline:
            if term["flag"] or (out / DONE).exists():
                _append(out / SAMPLES, {"sample": n, "final": True, "time_utc": _utc()})
                return 0
            time.sleep(min(POLL_SECONDS, interval))


# ------------------------------------------------------------------ sequence


class _SequenceLog:
    def __init__(self, out: Path) -> None:
        self.out = out

    def log(self, line: str) -> None:
        with open(self.out / LOG, "a", encoding="utf-8") as fh:
            fh.write(f"== {_utc()} {line}\n")


def terminate(proc: subprocess.Popen, grace: float, log: Callable[[str], None], name: str) -> None:
    """SIGTERM, wait up to ``grace`` seconds, then kill; every step is logged."""
    if proc.poll() is not None:
        return
    log(f"{name}: SIGTERM pid {proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=grace)
        log(f"{name}: exited after SIGTERM rc={proc.returncode}")
    except subprocess.TimeoutExpired:
        log(f"{name}: still running after {grace:g} s grace; killing")
        proc.kill()
        proc.wait()
        log(f"{name}: killed rc={proc.returncode}")


def _heartbeat_age(out: Path) -> float | None:
    try:
        return time.time() - (out / SAMPLES).stat().st_mtime
    except OSError:
        return None


def run_sequence(
    out: Path,
    steps: Sequence[tuple[str, list[str]]],
    *,
    sampler: Callable[[], Any],
    monitor_argv: list[str],
    interval: float = INTERVAL_SECONDS,
    grace: float = GRACE_SECONDS,
    ready_timeout: float = READY_TIMEOUT_SECONDS,
    stale_after: float | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run ``steps`` under the monitor; returns the exit code, also written to ``exit_status.txt``.

    ``stale_after`` (default three sampling intervals plus 5 s) is the heartbeat age at which a
    silent monitor counts as failed.
    """
    seq = _SequenceLog(out)
    procs: dict[str, subprocess.Popen] = {}
    stale_after = 3 * interval + 5.0 if stale_after is None else stale_after

    def body() -> int:
        seq.log(f"start {SUPERVISE_VERSION} pid {os.getpid()} steps {[s for s, _ in steps]}")
        try:
            base = validate_sample(sampler())
        except Exception as exc:  # noqa: BLE001
            seq.log(f"baseline sampling failed: {exc!r}")
            return RC_REFUSED
        _write_json(out / BASELINE, base)
        refusal = preflight({k: base[k] for k in RULE_KEYS})
        if refusal:
            _write_json(out / "preflight_refusal.json", {"reason": refusal, "baseline": base})
            seq.log(f"preflight refused: {refusal}")
            return RC_REFUSED
        mon_console = open(out / "monitor_console.txt", "ab")  # noqa: SIM115 - closed by the OS
        monitor = procs["monitor"] = subprocess.Popen(
            monitor_argv, stdin=subprocess.DEVNULL, stdout=mon_console, stderr=subprocess.STDOUT,
            env=env)  # fmt: skip
        seq.log(f"monitor started pid {monitor.pid}")
        deadline = time.monotonic() + ready_timeout
        while not (out / READY).exists():
            if monitor.poll() is not None or (out / STOP).exists() or time.monotonic() > deadline:
                seq.log("monitor not ready; no step started")
                return RC_MONITOR_STOP if (out / STOP).exists() else RC_MONITOR_NOT_READY
            time.sleep(POLL_SECONDS)
        seq.log("monitor ready")
        for name, argv in steps:
            if (out / STOP).exists():
                return RC_MONITOR_STOP
            console = open(out / f"{name}_console.txt", "ab")  # noqa: SIM115
            child = procs["step"] = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                env=env)  # fmt: skip
            (out / PID).write_text(f"{child.pid}\n")
            seq.log(f"{name} started pid {child.pid}")
            failure = None
            while child.poll() is None:
                if (out / STOP).exists():
                    failure = "monitor stop marker"
                elif monitor.poll() is not None:
                    failure = f"monitor exited rc={monitor.returncode}"
                else:
                    age = _heartbeat_age(out)
                    if age is None or age > stale_after:
                        failure = f"monitor heartbeat stale ({age})"
                if failure:
                    seq.log(f"{name}: {failure}")
                    terminate(child, grace, seq.log, name)
                    break
                time.sleep(POLL_SECONDS)
            child.wait()
            console.close()
            (out / PID).unlink(missing_ok=True)
            seq.log(f"{name} exited rc={child.returncode}")
            if (out / STOP).exists():
                return RC_MONITOR_STOP
            if failure:
                return RC_MONITOR_FAILURE
            if child.returncode != 0:
                return child.returncode if child.returncode > 0 else RC_STEP_FAILED
        return RC_OK

    rc = RC_SEQUENCE_ERROR
    try:
        rc = body()
    except BaseException as exc:  # noqa: BLE001 - evidence first, whatever happened
        seq.log(f"sequence exception: {type(exc).__name__}: {exc}")
        seq.log(traceback.format_exc().replace("\n", " | "))
        rc = RC_SEQUENCE_ERROR
        if "step" in procs:
            terminate(procs["step"], grace, seq.log, "step")
    finally:
        (out / DONE).write_text(f"{_utc()}\n")
        monitor = procs.get("monitor")
        if monitor is not None:
            terminate(monitor, min(grace, 3 * interval + 5.0), seq.log, "monitor")
            seq.log(f"monitor rc={monitor.returncode}")
            if monitor.returncode not in (0, 2, -signal.SIGTERM) and rc == RC_OK:
                rc = RC_MONITOR_FAILURE
        if (out / STOP).exists():
            rc = RC_MONITOR_STOP  # a stop marker always overrides success
        seq.log(f"end rc={rc}")
        (out / EXIT).write_text(f"{rc} {_utc()}\n", encoding="utf-8")
    return rc


# ------------------------------------------------------------------ detached start


def detached_prefix() -> list[str]:
    """``nohup`` (hang-ups ignored) and, where present, ``caffeinate -i`` (no idle sleep)."""
    prefix = ["nohup"]
    if shutil.which("caffeinate"):
        prefix += ["caffeinate", "-i"]
    return prefix


def start_detached(argv: list[str], out: Path, env: dict[str, str] | None = None) -> int:
    """Start ``argv`` in a new session with stdin closed; returns its process ID at once."""
    with open(out / "sequence_console.txt", "ab") as console:
        proc = subprocess.Popen([*detached_prefix(), *argv], stdin=subprocess.DEVNULL,
                                stdout=console, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)  # fmt: skip
    return proc.pid


def child_env() -> dict[str, str]:
    """Plain console text in the evidence files (MLX-LM's progress display honours NO_COLOR)."""
    return {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}


def sequence_argv(out: Path, steps: Sequence[tuple[str, list[str]]], sampler: str,
                  interval: float, grace: float, ready_timeout: float) -> list[str]:  # fmt: skip
    argv = [sys.executable, "-m", "axbench.supervise", "sequence", "--out", str(out),
            "--sampler", sampler, "--interval", str(interval), "--grace", str(grace),
            "--ready-timeout", str(ready_timeout)]  # fmt: skip
    for name, step in steps:
        argv += ["--step", f"{name}={json.dumps(step)}"]
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m axbench.supervise", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("monitor", "sequence", "start"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--sampler", default="macos")
        p.add_argument("--interval", type=float, default=INTERVAL_SECONDS)
        if name != "monitor":
            p.add_argument("--grace", type=float, default=GRACE_SECONDS)
            p.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_SECONDS)
            p.add_argument("--step", action="append", default=[], help="name=<JSON argv list>")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "monitor":
        return monitor_main(out, make_sampler(args.sampler), args.interval)
    steps = []
    for item in args.step:
        name, _, js = item.partition("=")
        steps.append((name, json.loads(js)))
    if args.cmd == "start":
        out.mkdir(parents=True, exist_ok=True)
        pid = start_detached(sequence_argv(out, steps, args.sampler, args.interval, args.grace,
                                           args.ready_timeout), out, child_env())  # fmt: skip
        print(json.dumps({"started": True, "sequence_pid": pid}))
        return 0
    signal.signal(signal.SIGTERM, _raise_interrupted)
    monitor_argv = [sys.executable, "-m", "axbench.supervise", "monitor", "--out", str(out),
                    "--sampler", args.sampler, "--interval", str(args.interval)]  # fmt: skip
    return run_sequence(out, steps, sampler=make_sampler(args.sampler), monitor_argv=monitor_argv,
                        interval=args.interval, grace=args.grace,
                        ready_timeout=args.ready_timeout, env=child_env())  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
