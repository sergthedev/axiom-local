"""Stop-rule bookkeeping and fixed request subsets for EXP-009 (pure logic, no MLX, no I/O).

The model runner (``envs/model/exp009_run.py``) samples the operating system and feeds the values
here, so the decisions are unit-tested in Core CI on every platform. Operational definitions are
recorded in the EXP-009 pre-run record (``PRERUN.md``); they are fixed before the first validation
prediction and are not changed after observing predictions.

Stop rules (protocol revision 3): allocation error; swap growth above 2 GB; memory pressure level
"critical"; sustained slowdown of more than 50 % against the first 30 measured requests.
"""

from __future__ import annotations

import random
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

SWAP_GROWTH_LIMIT_BYTES = 2 * 10**9  # "2 GB", decimal
PRESSURE_CRITICAL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
BASELINE_REQUESTS = 30
WINDOW = 10
SUSTAINED_WINDOWS = 3
SLOWDOWN_FRACTION = 0.5  # throughput below half of the baseline
REPEAT_PER_STRATUM = 10
REPEAT_SEED = 2026
WARMUP_COUNT = 5

_UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}


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


@dataclass
class StopWatch:
    """Per-request stop-rule checks for one run. ``check`` returns a reason or ``None``."""

    swap_used_before: int
    throughputs: list[float] = field(default_factory=list)
    slow_windows: int = 0
    max_swap_growth: int = 0
    max_pressure_level: int = 0
    stopped: str | None = None

    def baseline(self) -> float | None:
        if len(self.throughputs) < BASELINE_REQUESTS:
            return None
        return statistics.median(self.throughputs[:BASELINE_REQUESTS])

    def check(self, *, throughput: float, swap_used: int, pressure_level: int) -> str | None:
        if self.stopped:
            return self.stopped
        self.throughputs.append(throughput)
        growth = swap_used - self.swap_used_before
        self.max_swap_growth = max(self.max_swap_growth, growth)
        self.max_pressure_level = max(self.max_pressure_level, pressure_level)
        if growth > SWAP_GROWTH_LIMIT_BYTES:
            self.stopped = f"swap growth {growth} bytes above {SWAP_GROWTH_LIMIT_BYTES}"
        elif pressure_level >= PRESSURE_CRITICAL:
            self.stopped = f"memory pressure level {pressure_level} (critical)"
        else:
            self.stopped = self._slowdown()
        return self.stopped

    def _slowdown(self) -> str | None:
        n = len(self.throughputs)
        base = self.baseline()
        if base is None or n <= BASELINE_REQUESTS or (n - BASELINE_REQUESTS) % WINDOW:
            return None
        window = statistics.median(self.throughputs[n - WINDOW :])
        if window < SLOWDOWN_FRACTION * base:
            self.slow_windows += 1
        else:
            self.slow_windows = 0
        if self.slow_windows >= SUSTAINED_WINDOWS:
            return (
                f"sustained slowdown: {SUSTAINED_WINDOWS} consecutive windows of {WINDOW} "
                f"requests below {SLOWDOWN_FRACTION:.0%} of the baseline {base:.2f} tokens/s"
            )
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "stopped": self.stopped,
            "swap_used_before_bytes": self.swap_used_before,
            "max_swap_growth_bytes": self.max_swap_growth,
            "max_pressure_level": self.max_pressure_level,
            "baseline_median_generation_tps": self.baseline(),
            "requests_checked": len(self.throughputs),
        }
