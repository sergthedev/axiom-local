"""Stop rules, preflight and fixed request subsets for EXP-009 (pure logic, no MLX, no OS calls).

The model runner (``envs/model/exp009_run.py``) samples the operating system and passes the values
in, so every decision here is unit-tested in Core CI on every platform. Operational definitions are
in the EXP-009 pre-run record (``PRERUN.md``, revision 1 after the lead's pre-run review); they are
fixed before the first validation prediction and are not changed after observing predictions.

Stop rules (protocol revision 3): allocation error; swap growth above 2 GB; memory pressure level
"critical"; sustained slowdown of more than 50 % against the first 30 measured requests. Mains
power is a run condition: a process does not start without it and a loss of mains power stops it.

Revision 1 (pre-run review): swap growth is measured from **one baseline per process, sampled
before the model is loaded**, and checked after loading, after every warm-up and after every
request in every phase (main and fresh processes); a process does not start unless the memory
pressure level is normal. Sampling is discrete: it does not guarantee that a short peak between
two samples is seen.
"""

from __future__ import annotations

import random
import re
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SWAP_GROWTH_LIMIT_BYTES = 2 * 10**9  # "2 GB", decimal
PRESSURE_NORMAL = 1  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
PRESSURE_CRITICAL = 4
AC_POWER = "AC Power"
BASELINE_REQUESTS = 30
WINDOW = 10
SUSTAINED_WINDOWS = 3
SLOWDOWN_FRACTION = 0.5  # throughput below half of the baseline
REPEAT_PER_STRATUM = 10
REPEAT_SEED = 2026
WARMUP_COUNT = 5

_UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}
_ALLOCATION_TEXT = re.compile(r"malloc|alloc|out of memory|insufficient memory", re.I)


class AllocationStop(Exception):
    """An allocation failure: the protocol stops the run. Carries the failed request's record."""

    def __init__(self, message: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = record


# ------------------------------------------------------------------ parsing


def parse_swapusage(text: str) -> dict[str, int]:
    """Parse ``sysctl vm.swapusage`` (``total = 4096.00M  used = 2947.88M  free = ...``)."""
    out = {}
    for key in ("total", "used", "free"):
        m = re.search(rf"{key} = ([0-9.]+)([KMG])", text)
        if not m:
            raise ValueError(f"cannot parse swap {key!r}")
        out[f"{key}_bytes"] = round(float(m.group(1)) * _UNITS[m.group(2)])
    return out


def parse_free_percentage(text: str) -> int | None:
    """``memory_pressure``'s 'System-wide memory free percentage: 33%' line, if present."""
    m = re.search(r"System-wide memory free percentage:\s*([0-9]+)%", text)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ fixed subsets


def repeat_subset(examples: Sequence[dict[str, Any]]) -> list[str]:
    """30 stratified validation ids: 10 per stratum, seeded sample of the sorted ids."""
    rng = random.Random(REPEAT_SEED)
    by_stratum: dict[str, list[str]] = {}
    for e in examples:
        by_stratum.setdefault(e["stratum"], []).append(e["id"])
    chosen: list[str] = []
    for stratum in sorted(by_stratum):
        ids = sorted(by_stratum[stratum])
        if len(ids) < REPEAT_PER_STRATUM:
            raise ValueError(f"stratum {stratum} has fewer than {REPEAT_PER_STRATUM} examples")
        chosen.extend(sorted(rng.sample(ids, REPEAT_PER_STRATUM)))
    return chosen


def warmup_ids(train: Sequence[dict[str, Any]]) -> list[str]:
    """The first five train examples in file order (warm-up only; never scored)."""
    return [e["id"] for e in train[:WARMUP_COUNT]]


# ------------------------------------------------------------------ preflight and resources


def preflight(sample: dict[str, Any]) -> str | None:
    """Reason to refuse starting a process, or ``None``. ``sample``: swap_used, pressure, power."""
    if sample["power_source"] != AC_POWER:
        return f"power source is {sample['power_source']!r}, mains power is required"
    if sample["pressure_level"] != PRESSURE_NORMAL:
        return f"memory pressure level {sample['pressure_level']} is not normal"
    return None


@dataclass
class ProcessMonitor:
    """Resource stop rules for one process, from one baseline sampled before the model loads."""

    baseline: dict[str, Any]
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    samples: int = 0
    max_growth: int = 0
    max_pressure: int = 0
    stopped: str | None = None

    def __post_init__(self) -> None:
        self.max_pressure = self.baseline["pressure_level"]

    def check(self, phase: str, sample: dict[str, Any]) -> str | None:
        if self.stopped:
            return self.stopped
        self.samples += 1
        growth = sample["swap_used"] - self.baseline["swap_used"]
        self.max_growth = max(self.max_growth, growth)
        self.max_pressure = max(self.max_pressure, sample["pressure_level"])
        p = self.phases.setdefault(
            phase,
            {
                "samples": 0,
                "first_swap_used": sample["swap_used"],
                "max_swap_used": sample["swap_used"],
                "max_pressure_level": sample["pressure_level"],
                "power_sources": [],
            },
        )
        p["samples"] += 1
        p["last_swap_used"] = sample["swap_used"]
        p["max_swap_used"] = max(p["max_swap_used"], sample["swap_used"])
        p["max_pressure_level"] = max(p["max_pressure_level"], sample["pressure_level"])
        if sample["power_source"] not in p["power_sources"]:
            p["power_sources"].append(sample["power_source"])
        if growth > SWAP_GROWTH_LIMIT_BYTES:
            self.stopped = (
                f"{phase}: swap growth {growth} bytes since the pre-load baseline "
                f"is above {SWAP_GROWTH_LIMIT_BYTES}"
            )
        elif sample["pressure_level"] >= PRESSURE_CRITICAL:
            self.stopped = f"{phase}: memory pressure level {sample['pressure_level']} (critical)"
        elif sample["power_source"] != AC_POWER:
            self.stopped = f"{phase}: mains power interrupted ({sample['power_source']!r})"
        return self.stopped

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_before_load": self.baseline,
            "samples": self.samples,
            "max_cumulative_swap_growth_bytes": self.max_growth,
            "max_pressure_level": self.max_pressure,
            "phases": self.phases,
            "stopped": self.stopped,
            "note": "discrete samples after load, each warm-up and each request; not a "
            "continuous peak guarantee",
        }


@dataclass
class SlowdownWatch:
    """Sustained-slowdown rule (D11) for one run phase: its own baseline of 30 requests."""

    throughputs: list[float] = field(default_factory=list)
    slow_windows: int = 0
    stopped: str | None = None

    def baseline(self) -> float | None:
        if len(self.throughputs) < BASELINE_REQUESTS:
            return None
        return statistics.median(self.throughputs[:BASELINE_REQUESTS])

    def check(self, throughput: float) -> str | None:
        if self.stopped:
            return self.stopped
        self.throughputs.append(throughput)
        n = len(self.throughputs)
        base = self.baseline()
        if base is None or n <= BASELINE_REQUESTS or (n - BASELINE_REQUESTS) % WINDOW:
            return None
        window = statistics.median(self.throughputs[n - WINDOW :])
        self.slow_windows = self.slow_windows + 1 if window < SLOWDOWN_FRACTION * base else 0
        if self.slow_windows >= SUSTAINED_WINDOWS:
            self.stopped = (
                f"sustained slowdown: {SUSTAINED_WINDOWS} consecutive windows of {WINDOW} "
                f"requests below {SLOWDOWN_FRACTION:.0%} of the baseline {base:.2f} tokens/s"
            )
        return self.stopped

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_median_generation_tps": self.baseline(),
            "requests": len(self.throughputs),
            "stopped": self.stopped,
        }


# ------------------------------------------------------------------ requests


def is_allocation_error(exc: BaseException) -> bool:
    """``MemoryError`` always; otherwise an allocation message (MLX reports ``[metal::malloc]``)."""
    return isinstance(exc, MemoryError) or bool(_ALLOCATION_TEXT.search(str(exc)))


def timed_call(record: dict[str, Any], call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one router call as ``axbench.evaluate.run_router`` does, plus generation metadata.

    An allocation error raises :class:`AllocationStop` (the run stops); any other exception becomes
    an ``error`` status for this request, which stays in the denominator.
    """
    start = time.perf_counter()
    try:
        meta = call()
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - MemoryError is an Exception subclass
        if is_allocation_error(exc):
            record.update(
                status="error",
                raw_output=None,
                error=f"allocation:{type(exc).__name__}",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
            message = f"allocation error: {type(exc).__name__}: {exc}"
            raise AllocationStop(message, record) from exc
        meta, status = {"raw_output": None}, "error"
        record["error"] = type(exc).__name__
    record.update(status=status, elapsed_ms=(time.perf_counter() - start) * 1000.0, **meta)
    return record


def run_requests(
    items: Iterable[Any],
    request: Callable[[Any], dict[str, Any]],
    sample: Callable[[], dict[str, Any]],
    monitor: ProcessMonitor,
    phase: str,
    emit: Callable[[dict[str, Any]], None],
    slowdown: SlowdownWatch | None = None,
) -> str | None:
    """Run requests one by one; emit each record before checking the stop rules.

    Returns the stop reason, or ``None`` when every item ran. Exceptions other than allocation
    failures inside ``request`` are the caller's (the runner records the process as failed).
    """
    for item in items:
        try:
            rec = request(item)
        except AllocationStop as exc:
            if exc.record is not None:
                emit(exc.record)
            return f"{phase}: {exc}"
        emit(rec)
        reason = monitor.check(phase, sample())
        if reason is None and slowdown is not None:
            reason = slowdown.check(float(rec.get("generation_tps") or 0.0))
        if reason:
            return reason
    return None
