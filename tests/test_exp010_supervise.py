"""EXP-010 supervised execution (lead pre-run review, correction A) with harmless dummy processes.

Every process here is a short Python one-liner (sleep, exit, ignore SIGTERM) and every resource
sample is scripted; nothing loads a model, reads the corpus or samples the real machine. The
process tests need POSIX signals and sessions, so they are skipped on Windows (the launcher is
macOS-only by protocol).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from axbench import supervise as sv

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and sessions")

OK = {"swap_used": 1_000_000_000, "pressure_level": 1, "power_source": "AC Power"}
SWAP_STOP = {**OK, "swap_used": OK["swap_used"] + 2 * 10**9 + 1}
INTERVAL = 0.2
GRACE = 1.0


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


SLEEP = _py("import time; time.sleep(0.6)")
LONG = _py("import time; time.sleep(60)")
IGNORE_TERM = _py(
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
)
EXIT0_ON_TERM = _py(
    "import signal, sys, time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); "
    "time.sleep(60)"
)


def _script(tmp: Path, items: list) -> str:
    path = tmp / "samples.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in items))
    return f"file:{path}"


def _run(tmp: Path, items: list, steps, monitor_argv=None, **kw) -> tuple[int, Path]:
    out = tmp / "run"
    out.mkdir()
    spec = _script(tmp, items)
    argv = monitor_argv or [sys.executable, "-m", "axbench.supervise", "monitor", "--out",
                            str(out), "--sampler", spec, "--interval", str(INTERVAL)]  # fmt: skip
    t0 = time.monotonic()
    rc = sv.run_sequence(out, steps, sampler=sv.make_sampler(spec), monitor_argv=argv,
                         interval=INTERVAL, grace=GRACE, ready_timeout=10.0, **kw)  # fmt: skip
    assert time.monotonic() - t0 < 30
    return rc, out


def _exit_code(out: Path) -> int:
    return int((out / sv.EXIT).read_text().split()[0])


def _log(out: Path) -> str:
    return (out / sv.LOG).read_text()


def _samples(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / sv.SAMPLES).read_text().splitlines()]


# ------------------------------------------------------------------ sample validation


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {**OK, "swap_used": None},
        {**OK, "swap_used": -1},
        {**OK, "swap_used": "1G"},
        {**OK, "swap_used": True},
        {**OK, "pressure_level": 3},
        {**OK, "pressure_level": None},
        {**OK, "power_source": ""},
        {k: v for k, v in OK.items() if k != "power_source"},
    ],
)
def test_malformed_samples_are_rejected(bad):
    with pytest.raises(sv.MalformedSample):
        sv.validate_sample(bad)


def test_valid_sample_passes():
    assert sv.validate_sample(dict(OK)) == OK


# ------------------------------------------------------------------ sequences


@posix
def test_normal_completion(tmp_path):
    rc, out = _run(tmp_path, [OK], [("train", SLEEP), ("reload", SLEEP)])
    assert rc == 0 and _exit_code(out) == 0
    assert (out / sv.BASELINE).exists() and (out / sv.READY).exists()
    assert not (out / sv.STOP).exists()
    log = _log(out)
    assert "monitor ready" in log and log.index("monitor ready") < log.index("train started")
    assert "train exited rc=0" in log and "reload exited rc=0" in log
    samples = _samples(out)
    assert samples[0]["sample"] == 0 and samples[-1].get("final") is True


@posix
def test_preflight_refusal_starts_nothing(tmp_path):
    rc, out = _run(tmp_path, [{**OK, "power_source": "Battery Power"}], [("train", SLEEP)])
    assert rc == sv.RC_REFUSED == _exit_code(out)
    assert json.loads((out / "preflight_refusal.json").read_text())["reason"]
    assert "monitor started" not in _log(out) and "train started" not in _log(out)


@posix
def test_unreadable_baseline_is_a_refusal(tmp_path):
    rc, out = _run(tmp_path, [{"raise": "sysctl failed"}], [("train", SLEEP)])
    assert rc == sv.RC_REFUSED == _exit_code(out)
    assert "baseline sampling failed" in _log(out)


@posix
def test_monitor_that_never_becomes_ready_starts_no_step(tmp_path):
    rc, out = _run(tmp_path, [OK], [("train", SLEEP)], monitor_argv=_py("pass"))
    assert rc == sv.RC_MONITOR_NOT_READY == _exit_code(out)
    assert "train started" not in _log(out)


@posix
def test_malformed_first_monitor_sample_blocks_the_start(tmp_path):
    bad = tmp_path / "monitor_samples.jsonl"
    bad.write_text(json.dumps({**OK, "pressure_level": 7}) + "\n")
    out = tmp_path / "run"
    argv = [sys.executable, "-m", "axbench.supervise", "monitor", "--out", str(out),
            "--sampler", f"file:{bad}", "--interval", str(INTERVAL)]  # fmt: skip
    rc, out = _run(tmp_path, [OK], [("train", SLEEP)], monitor_argv=argv)
    assert rc == sv.RC_MONITOR_STOP == _exit_code(out)
    assert not (out / sv.READY).exists() and "train started" not in _log(out)


@posix
def test_injected_stop_terminates_the_step_and_skips_the_rest(tmp_path):
    rc, out = _run(tmp_path, [OK, OK, OK, SWAP_STOP], [("train", LONG), ("reload", SLEEP)])
    assert rc == sv.RC_MONITOR_STOP == _exit_code(out)
    stop = json.loads((out / sv.STOP).read_text())
    assert "swap growth" in stop["reason"]
    log = _log(out)
    assert "train: monitor stop marker" in log or "train exited rc=-15" in log
    assert "reload started" not in log


@posix
def test_malformed_sample_fails_closed(tmp_path):
    rc, out = _run(tmp_path, [OK, OK, {**OK, "swap_used": "garbage"}], [("train", LONG)])
    assert rc == sv.RC_MONITOR_STOP
    assert "sampling failed" in json.loads((out / sv.STOP).read_text())["reason"]


@posix
def test_sampler_exception_fails_closed(tmp_path):
    rc, out = _run(tmp_path, [OK, OK, {"raise": "memory_pressure missing"}], [("train", LONG)])
    assert rc == sv.RC_MONITOR_STOP
    assert "memory_pressure missing" in json.loads((out / sv.STOP).read_text())["reason"]


@posix
def test_monitor_crash_stops_the_step(tmp_path):
    rc, out = _run(tmp_path, [OK, OK, {"exit": 9}], [("train", LONG), ("reload", SLEEP)])
    assert rc == sv.RC_MONITOR_FAILURE == _exit_code(out)
    log = _log(out)
    assert "train: monitor exited rc=9" in log and "reload started" not in log


@posix
def test_hung_monitor_is_a_stale_heartbeat(tmp_path):
    rc, out = _run(tmp_path, [OK, {"hang": 60}], [("train", LONG)], stale_after=1.5)
    assert rc == sv.RC_MONITOR_FAILURE == _exit_code(out)
    assert "monitor heartbeat stale" in _log(out)


@posix
def test_sigterm_escalates_to_kill_after_the_grace_period(tmp_path):
    rc, out = _run(tmp_path, [OK, OK, OK, OK, OK, SWAP_STOP], [("train", IGNORE_TERM)])
    assert rc == sv.RC_MONITOR_STOP
    log = _log(out)
    assert "still running after 1 s grace; killing" in log
    assert f"killed rc=-{int(signal.SIGKILL)}" in log


@posix
def test_stop_marker_overrides_a_step_that_exits_zero(tmp_path):
    rc, out = _run(tmp_path, [OK, OK, OK, SWAP_STOP], [("train", EXIT0_ON_TERM)])
    assert "train exited rc=0" in _log(out)
    assert rc == sv.RC_MONITOR_STOP == _exit_code(out)


@posix
def test_failed_step_ends_the_sequence(tmp_path):
    rc, out = _run(tmp_path, [OK], [("train", _py("raise SystemExit(3)")), ("reload", SLEEP)])
    assert rc == 3 == _exit_code(out)
    assert "reload started" not in _log(out)


@posix
def test_launcher_exception_still_writes_exit_status(tmp_path):
    missing = str(tmp_path / "no-such-program")
    rc, out = _run(tmp_path, [OK], [("train", [missing])])
    assert rc == sv.RC_SEQUENCE_ERROR == _exit_code(out)
    log = _log(out)
    assert "sequence exception: FileNotFoundError" in log and "monitor rc=" in log


@posix
def test_sigterm_to_the_sequence_stops_the_step_and_keeps_evidence(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    spec = _script(tmp_path, [OK])
    argv = sv.sequence_argv(out, [("train", LONG)], spec, INTERVAL, GRACE, 10.0)
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while not (out / sv.PID).exists():
        assert time.monotonic() < deadline and proc.poll() is None
        time.sleep(0.05)
    child = int((out / sv.PID).read_text())
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) == sv.RC_SEQUENCE_ERROR
    assert _exit_code(out) == sv.RC_SEQUENCE_ERROR
    assert "SequenceInterrupted" in _log(out)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


@posix
def test_detached_start_survives_its_parent_and_completes(tmp_path):
    out = tmp_path / "run"
    spec = _script(tmp_path, [OK])
    step = _py("import time; time.sleep(1.5)")
    argv = [sys.executable, "-m", "axbench.supervise", "start", "--out", str(out), "--sampler",
            spec, "--interval", str(INTERVAL), "--grace", str(GRACE), "--ready-timeout", "10",
            "--step", f"train={json.dumps(step)}"]  # fmt: skip
    t0 = time.monotonic()
    started = json.loads(subprocess.run(argv, capture_output=True, text=True, check=True,
                                        timeout=20).stdout)  # fmt: skip
    assert time.monotonic() - t0 < 10  # returns at once, the sequence keeps running
    pid = started["sequence_pid"]
    assert os.getsid(pid) == pid != os.getsid(0)  # its own session: no hang-up from ours
    deadline = time.monotonic() + 30
    while not (out / sv.EXIT).exists():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert _exit_code(out) == 0
    assert "train exited rc=0" in _log(out)
