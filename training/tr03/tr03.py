"""EXP-TR03 training-plan tooling (no model path; nothing here trains or loads a model).

Status: **prepared, not executed, not approved.** This file holds only the pure, testable parts of
the bounded single-run LoRA plan: the config checks, the exposure schedule with its partial-batch
rule, the time budget, the fixed evaluation and snapshot points, the sustained-run monitor and the
thermal-status parser. There is deliberately no driver here: a driver that loads weights is a
separate piece of work (an extension of ``training/tr02/tr02.py``'s ``run_phase``, which stays
unchanged) and needs its own review and clearance. The frozen EXP-TR02 and EXP-010 code is not
modified; EXP-TR02's loss mask, rendering and event audit are reused as they are.

Commands (``uv run python training/tr03/tr03.py <command>`` from the repository root; none needs
the model environment):

* ``verify-config``: the config is internally consistent and equal to TR02 where it must be.
* ``plan``: the exposure schedule summary and time budget for all eligible train rows.
* ``check-therm``: parse ``pmset -g therm`` text from standard input (run on the Mac in a no-weight
  preflight; it has been tested only against the text recorded in the TR02 run).
* ``readiness``: which human gates are open (all closed today); exit status 1 unless all are open.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_PATH = HERE / "config.json"
TR02_PATH = REPO / "training" / "tr02" / "tr02.py"
TR02_CONFIG_PATH = REPO / "training" / "tr02" / "config.json"
PLAN_PATH = REPO / "data" / "axbench-tr" / "tr03-prep" / "TRAINING_PLAN.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

try:  # the repository's source tree is importable when run through ``uv run`` from its root
    from axbench_tr import export, tr03
except ModuleNotFoundError:  # pragma: no cover - plain ``python`` without the project
    sys.path.insert(0, str(REPO / "src"))
    from axbench_tr import export, tr03


def _load_tr02() -> Any:
    """The frozen TR02 harness as a module (read only; its source is never edited)."""
    if "tr02" in sys.modules:
        return sys.modules["tr02"]
    spec = importlib.util.spec_from_file_location("tr02", TR02_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["tr02"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


tr02 = _load_tr02()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


# ------------------------------------------------------------------------------ config checks


def config_problems(cfg: Mapping[str, Any], tr02_cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Problems that would make the plan inconsistent with EXP-TR02 or internally contradictory."""
    tr02_cfg = tr02_cfg or json.loads(TR02_CONFIG_PATH.read_text("utf-8"))
    problems: list[str] = []
    if cfg["inherits"]["tr02_config_sha256"] != sha256_file(TR02_CONFIG_PATH):
        problems.append("tr02_config_hash_differs")
    if cfg["model"] != tr02_cfg["model"]:
        problems.append("model_pin_differs_from_tr02")
    for key in ("packages", "python"):
        if cfg[key] != tr02_cfg[key]:
            problems.append(f"{key}_differ_from_tr02")
    hp, hp02 = cfg["hyperparameters"], tr02_cfg["hyperparameters"]
    for key, value in hp02.items():
        if hp.get(key) != value:
            problems.append(f"hyperparameter_{key}_differs_from_tr02")
    if hp.get("init_adapter") is not None:
        problems.append("init_adapter_must_be_null")
    if hp.get("partial_batch") != "drop_remainder":
        problems.append("partial_batch_rule_changed")
    if hp.get("micro_batch") != 1 or hp.get("grad_accumulation") != 4:
        problems.append("effective_batch_is_not_1_by_4")
    inputs = tr02_cfg["inputs"]
    for key, tr02_key in (
        ("corpus_version", "corpus_version"),
        ("corpus_files_sha256", "corpus_files_sha256"),
        ("per_example_path", "per_example_path"),
        ("per_example_sha256", "per_example_sha256"),
    ):
        if cfg["data"][key] != inputs[tr02_key]:
            problems.append(f"data_{key}_differs_from_tr02")
    for name in ("mlx_peak_gb", "rss_gb", "swap_growth_gb", "pressure_critical"):
        if cfg["monitor"][name] != tr02_cfg["limits"][name]:
            problems.append(f"monitor_{name}_differs_from_tr02")
    if "not approved" not in cfg["status"] or "no clearance" not in cfg["status"]:
        problems.append("status_must_say_not_approved_and_no_clearance")
    if cfg["data"]["eligibility"] != {
        "origin": tr03.ELIGIBLE_ORIGIN,
        "schema_sources": list(tr03.ELIGIBLE_SOURCES),
    }:
        problems.append("eligibility_differs_from_the_data_tooling")
    mon = cfg["monitor"]
    if not (0 < mon["floor_ratio"] < mon["slow_ratio"] < 1):
        problems.append("monitor_ratios_must_satisfy_0<floor<slow<1")
    if cfg["arms"]["research"]["needs_human_gates"] is not True:
        problems.append("research_arm_must_need_human_gates")
    if cfg["arms"]["resource"]["needs_human_gates"] is not False:
        problems.append("resource_arm_flag_changed")
    return problems


# --------------------------------------------------------------------------------- schedule


def exposure_order(n_rows: int, seed: int, epochs: int) -> list[list[int]]:
    """One permutation per epoch from one seeding, the stock iterator's behaviour for batch size
    1 (``np.random.seed(seed)`` once, then ``permutation(n)`` per epoch). For one epoch this is
    exactly ``tr02.planned_order`` (a test asserts it)."""
    import numpy as np

    np.random.seed(seed)
    return [[int(i) for i in np.random.permutation(n_rows)] for _ in range(epochs)]


def exposure_schedule(
    ids: Sequence[str],
    lengths: Mapping[str, int],
    *,
    seed: int,
    grad_accumulation: int,
    epochs: int,
    max_seq_length: int,
    full: bool = False,
) -> dict[str, Any]:
    """The exact rows exposed, in order, with the partial-batch rule ``drop_remainder``.

    Rows are taken in permutation order in groups of ``grad_accumulation``; the last
    ``n mod grad_accumulation`` rows of each epoch's permutation are not exposed in that epoch,
    because a partial accumulation would change the gradient scale. They are listed, never
    silently dropped. Over-length rows are an error (nothing is truncated).
    """
    if grad_accumulation < 1 or epochs < 1:
        raise ValueError("grad_accumulation and epochs must be at least 1")
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids")
    too_long = [i for i in ids if lengths[i] > max_seq_length]
    if too_long:
        raise ValueError(f"{len(too_long)} rows exceed {max_seq_length} tokens; none is truncated")
    n = len(ids)
    updates_per_epoch = n // grad_accumulation
    if updates_per_epoch == 0:
        raise ValueError("fewer rows than one effective batch")
    exposed: list[str] = []
    per_epoch: list[list[str]] = []
    unexposed: list[list[str]] = []
    for perm in exposure_order(n, seed, epochs):
        keep = updates_per_epoch * grad_accumulation
        per_epoch.append([ids[i] for i in perm[:keep]])
        exposed += per_epoch[-1]
        unexposed.append([ids[i] for i in perm[keep:]])
    full_tokens = sum(lengths[i] for i in exposed)
    padded = sum(tr03.padded_tokens(lengths[i], max_seq_length) for i in exposed)
    out: dict[str, Any] = {
        "rows": n,
        "seed": seed,
        "epochs": epochs,
        "grad_accumulation": grad_accumulation,
        "optimizer_updates": updates_per_epoch * epochs,
        "microsteps": len(exposed),
        "partial_batch_rule": "drop_remainder",
        "unexposed_rows_per_epoch": [len(u) for u in unexposed],
        "unexposed_row_ids": unexposed,
        "full_tokens_exposed": full_tokens,
        "padded_tokens_exposed": padded,
        "exposure_order_sha256": hashlib.sha256("\n".join(exposed).encode()).hexdigest(),
        "row_set_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "each_row_exposed_at_most_once_per_epoch": all(
            len(set(rows)) == len(rows) for rows in per_epoch
        ),
    }
    if full:
        out["steps"] = [
            {"microstep": k + 1, "id": i, "tokens": lengths[i]} for k, i in enumerate(exposed)
        ]
    return out


def eval_points(updates: int, count: int = 4) -> list[int]:
    """Updates after which the fixed diagnostic validation runs: ``floor(k * U / count)``, the
    last equal to ``U``. A schedule too short for distinct points is an error."""
    points = [updates * k // count for k in range(1, count + 1)]
    if points[-1] != updates or len(set(points)) != len(points) or points[0] < 1:
        raise ValueError("too few updates for distinct evaluation points")
    return points


def diagnostic_validation_ids(
    valid_rows: Sequence[Mapping[str, Any]], count: int, key: str
) -> list[str]:
    """A fixed validation subset spread over frames (the export module's round-robin rule)."""
    chosen = export.stratified_subset(list(valid_rows), count, key)  # type: ignore[arg-type]
    return [r["id"] for r in chosen]


# ----------------------------------------------------------------------------------- budget


def budget(
    schedule: Mapping[str, Any],
    *,
    diagnostic_rows: int,
    full_validation_rows: int,
    rates: Mapping[str, Any],
    eval_count: int = 4,
) -> dict[str, Any]:
    """Seconds, from the measured TR02 rates and a conservative thermal allowance.

    ``nominal`` uses the slowest rate observed in TR02. ``conservative`` divides the compute rate
    by ``1 - thermal_allowance`` and multiplies the validation time likewise. The wall-time cap is
    ``wall_cap_factor`` times the conservative total, rounded up to ``wall_cap_round_to_seconds``.
    Training, validation and load are reported separately. Nothing here is measured at this scale.
    """
    rate = float(rates["full_tokens_per_s_slowest_observed"])
    per_row = float(rates["validation_seconds_per_row_slowest_observed"])
    load = float(rates["model_load_seconds"])
    allowance = float(rates["thermal_allowance_fraction"])
    slow = 1.0 - allowance
    train_nominal = schedule["full_tokens_exposed"] / rate
    val_passes = [diagnostic_rows] * (eval_count + 1) + [full_validation_rows]
    val_nominal = per_row * sum(val_passes)
    total_nominal = load + train_nominal + val_nominal
    train_cons, val_cons = train_nominal / slow, val_nominal / slow
    total_cons = load + train_cons + val_cons
    step = int(rates["wall_cap_round_to_seconds"])
    cap = math.ceil(total_cons * float(rates["wall_cap_factor"]) / step) * step
    return {
        "inputs": {
            "full_tokens_exposed": schedule["full_tokens_exposed"],
            "diagnostic_validation_rows_per_pass": diagnostic_rows,
            "diagnostic_passes": eval_count + 1,
            "full_validation_rows": full_validation_rows,
            "rate_full_tokens_per_s": rate,
            "validation_seconds_per_row": per_row,
            "thermal_allowance_fraction": allowance,
        },
        "seconds": {
            "training_nominal": round(train_nominal),
            "validation_nominal": round(val_nominal),
            "total_nominal": round(total_nominal),
            "training_conservative": round(train_cons),
            "validation_conservative": round(val_cons),
            "total_conservative": round(total_cons),
            "hard_wall_cap": cap,
        },
        "minutes": {
            "training_nominal": round(train_nominal / 60, 1),
            "validation_nominal": round(val_nominal / 60, 1),
            "total_nominal": round(total_nominal / 60, 1),
            "total_conservative": round(total_cons / 60, 1),
            "hard_wall_cap": round(cap / 60, 1),
        },
        "validation_overhead_fraction_of_training_nominal": round(val_nominal / train_nominal, 4),
        "status": "extrapolated from EXP-TR02 rates (2.5 minute smoke); not measured at this scale",
    }


def alternatives(
    ids: Sequence[str], lengths: Mapping[str, int], cfg: Mapping[str, Any], diag: int, full_val: int
) -> list[dict[str, Any]]:
    """Durations of other epoch counts, shown to document that the epoch count was not chosen to
    fill time. Not selected."""
    out = []
    hp = cfg["hyperparameters"]
    for epochs in (1, 2, 3):
        sched = exposure_schedule(
            ids,
            lengths,
            seed=int(hp["seed"]),
            grad_accumulation=int(hp["grad_accumulation"]),
            epochs=epochs,
            max_seq_length=int(hp["max_seq_length"]),
        )
        b = budget(sched, diagnostic_rows=diag, full_validation_rows=full_val, rates=cfg["rates"])
        out.append(
            {
                "epochs": epochs,
                "optimizer_updates": sched["optimizer_updates"],
                "minutes_nominal": b["minutes"]["total_nominal"],
                "minutes_conservative": b["minutes"]["total_conservative"],
                "selected": epochs == int(hp["epochs"]),
            }
        )
    return out


# ------------------------------------------------------------------------------------ monitor


class Stop(Exception):
    """A stop rule fired; the run is a failed run (no retry, no resume)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


THERMAL_OK = (
    re.compile(r"^no thermal warning level has been recorded$"),
    re.compile(r"^no performance warning level has been recorded$"),
    re.compile(r"^no cpu power status has been recorded$"),
)


def parse_therm(text: str) -> list[str]:
    """Warnings in ``pmset -g therm`` text; empty only if every statement is a known 'no ...' line.

    Accepts the sampler's joined form (``;``-separated, as recorded in the TR02 run logs) and raw
    lines. Fails closed: empty or unrecognised text yields a problem. Only the three 'no warning'
    statements seen in the TR02 run are treated as normal.
    """
    statements = [
        re.sub(r"^note:\s*", "", s.strip().lower()).strip()
        for s in re.split(r"[;\n]", text)
        if s.strip()
    ]
    if not statements:
        return ["empty_thermal_text"]
    problems = [s for s in statements if not any(p.match(s) for p in THERMAL_OK)]
    seen = {i for i, p in enumerate(THERMAL_OK) if any(p.match(s) for s in statements)}
    if not problems and {0, 1} - seen:
        problems.append("thermal_or_performance_statement_missing")
    return problems


@dataclass(frozen=True)
class MonitorLimits:
    mlx_peak_gb: float
    rss_gb: float
    swap_growth_gb: float
    pressure_critical: int
    reference_windows: int
    slow_ratio: float
    slow_windows: int
    floor_ratio: float
    floor_windows: int
    pressure_warning_level: int
    pressure_warning_samples: int
    max_wall_s: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], max_wall_s: float) -> MonitorLimits:
        m = cfg["monitor"]
        return cls(
            mlx_peak_gb=float(m["mlx_peak_gb"]),
            rss_gb=float(m["rss_gb"]),
            swap_growth_gb=float(m["swap_growth_gb"]),
            pressure_critical=int(m["pressure_critical"]),
            reference_windows=int(m["reference_windows"]),
            slow_ratio=float(m["slow_ratio"]),
            slow_windows=int(m["slow_windows"]),
            floor_ratio=float(m["floor_ratio"]),
            floor_windows=int(m["floor_windows"]),
            pressure_warning_level=int(m["pressure_warning_level"]),
            pressure_warning_samples=int(m["pressure_warning_samples"]),
            max_wall_s=float(max_wall_s),
        )


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class LongRunMonitor:
    """Fail-closed rules for a run of about an hour, fed one event at a time.

    Events (dicts): ``window`` (``index``, ``full_tokens_per_s``, ``mlx_peak_gb``, ``train_loss``,
    ``elapsed_s``), ``os`` (``elapsed_s``, ``rss_gb``, ``swap_used_gb``, ``pressure_level``,
    ``on_battery``, ``therm``) and ``val`` (``name``, ``loss``, ``elapsed_s``). ``feed`` raises
    ``Stop``; nothing is retried. The first window is not special; the reference rate is the median
    of the first ``reference_windows`` windows and the throughput rules apply after it exists.
    """

    def __init__(self, limits: MonitorLimits) -> None:
        self.lim = limits
        self.rates: list[float] = []
        self.next_window = 1
        self.swap_baseline: float | None = None
        self.slow_run = 0
        self.floor_run = 0
        self.pressure_run = 0
        self.peak_seen = 0.0
        self.val_names: list[str] = []

    @property
    def reference_rate(self) -> float | None:
        if len(self.rates) < self.lim.reference_windows:
            return None
        head = sorted(self.rates[: self.lim.reference_windows])
        mid = len(head) // 2
        return head[mid] if len(head) % 2 else (head[mid - 1] + head[mid]) / 2

    def _wall(self, ev: Mapping[str, Any]) -> None:
        elapsed = ev.get("elapsed_s")
        if not _finite(elapsed) or elapsed < 0:
            raise Stop("malformed_event", f"elapsed_s={elapsed!r}")
        if elapsed > self.lim.max_wall_s:
            raise Stop("max_wall_time", f"{elapsed:.0f}s > {self.lim.max_wall_s:.0f}s")

    def feed(self, ev: Mapping[str, Any]) -> None:
        kind = ev.get("kind")
        self._wall(ev)
        if kind == "window":
            self._window(ev)
        elif kind == "os":
            self._os(ev)
        elif kind == "val":
            if not _finite(ev.get("loss")) or ev["loss"] < 0:
                raise Stop("nonfinite_validation_loss", str(ev.get("name")))
            self.val_names.append(str(ev.get("name")))
        else:
            raise Stop("unknown_event", repr(kind))

    def _window(self, ev: Mapping[str, Any]) -> None:
        if ev.get("index") != self.next_window:
            raise Stop("event_sequence", f"window {ev.get('index')!r}, expected {self.next_window}")
        self.next_window += 1
        rate, peak, loss = ev.get("full_tokens_per_s"), ev.get("mlx_peak_gb"), ev.get("train_loss")
        if not (_finite(rate) and rate > 0):
            raise Stop("nonfinite_or_nonpositive_rate", repr(rate))
        if not _finite(loss) or loss < 0:
            raise Stop("nonfinite_training_loss", repr(loss))
        if not _finite(peak) or peak < self.peak_seen:
            raise Stop("mlx_peak_invalid_or_decreasing", repr(peak))
        self.peak_seen = float(peak)
        if peak > self.lim.mlx_peak_gb:
            raise Stop("mlx_peak_limit", f"{peak} GB > {self.lim.mlx_peak_gb} GB")
        self.rates.append(float(rate))
        ref = self.reference_rate
        if ref is None or len(self.rates) <= self.lim.reference_windows:
            return
        self.slow_run = self.slow_run + 1 if rate < self.lim.slow_ratio * ref else 0
        self.floor_run = self.floor_run + 1 if rate < self.lim.floor_ratio * ref else 0
        if self.floor_run >= self.lim.floor_windows:
            raise Stop("throughput_floor", f"{self.floor_run} windows below {self.lim.floor_ratio}")
        if self.slow_run >= self.lim.slow_windows:
            raise Stop(
                "sustained_throughput_loss",
                f"{self.slow_run} windows below {self.lim.slow_ratio} of {ref:.0f} tokens/s",
            )

    def _os(self, ev: Mapping[str, Any]) -> None:
        rss, swap, level = ev.get("rss_gb"), ev.get("swap_used_gb"), ev.get("pressure_level")
        if not (_finite(swap) and swap >= 0) or not isinstance(level, int):
            raise Stop("unreadable_os_sample", repr((swap, level)))
        if rss is not None and not (_finite(rss) and rss >= 0):
            raise Stop("unreadable_os_sample", f"rss={rss!r}")
        if ev.get("on_battery") is not False:
            raise Stop("not_on_ac_power", repr(ev.get("on_battery")))
        problems = parse_therm(str(ev.get("therm", "")))
        if problems:
            raise Stop("thermal_or_performance_warning", "; ".join(problems)[:200])
        if self.swap_baseline is None:
            self.swap_baseline = float(swap)
        if swap - self.swap_baseline > self.lim.swap_growth_gb:
            raise Stop("swap_growth", f"{swap - self.swap_baseline:.2f} GB")
        if rss is not None and rss > self.lim.rss_gb:
            raise Stop("rss_limit", f"{rss} GB > {self.lim.rss_gb} GB")
        if level >= self.lim.pressure_critical:
            raise Stop("memory_pressure_critical", str(level))
        # Warning level is tolerated at model load (TR02 saw it in 3 of 12 probe samples) but not
        # as a steady state once training has started.
        if self.next_window > 1 and level >= self.lim.pressure_warning_level:
            self.pressure_run += 1
            if self.pressure_run >= self.lim.pressure_warning_samples:
                raise Stop("sustained_memory_pressure_warning", f"{self.pressure_run} samples")
        else:
            self.pressure_run = 0

    def complete(self, expected_windows: int, expected_val: Sequence[str]) -> list[str]:
        """Problems that make a finished run invalid (empty means every event arrived)."""
        problems = []
        if self.next_window - 1 != expected_windows:
            problems.append(f"windows_{self.next_window - 1}_expected_{expected_windows}")
        if list(self.val_names) != list(expected_val):
            problems.append("validation_events_differ_from_the_plan")
        return problems


def window_count(updates: int, updates_per_window: int) -> int:
    """Number of monitor windows: a final partial window is its own window."""
    return math.ceil(updates / updates_per_window)


# ----------------------------------------------------------------------------------- plan


def build_plan(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Schedule and budget for all eligible train rows (arm R exactly; arm T's upper bound)."""
    cfg = cfg or load_config()
    train, valid, per, hashes = tr03.load()
    elig_t, _ = tr03.split_eligible(train)
    elig_v, _ = tr03.split_eligible(valid)
    lengths = {r["id"]: int(per[r["id"]]["separate_tokens"]) for r in (*elig_t, *elig_v)}
    ids = sorted(r["id"] for r in elig_t)  # file order is irrelevant: the row set is sorted by id
    hp = cfg["hyperparameters"]
    sched = exposure_schedule(
        ids,
        lengths,
        seed=int(hp["seed"]),
        grad_accumulation=int(hp["grad_accumulation"]),
        epochs=int(hp["epochs"]),
        max_seq_length=int(hp["max_seq_length"]),
    )
    diag = diagnostic_validation_ids(
        elig_v,
        int(cfg["data"]["diagnostic_validation_rows"]),
        cfg["data"]["diagnostic_validation_key"],
    )
    updates = sched["optimizer_updates"]
    points = eval_points(updates)
    b = budget(
        sched,
        diagnostic_rows=len(diag),
        full_validation_rows=len(elig_v),
        rates=cfg["rates"],
        eval_count=len(points),
    )
    win = int(cfg["monitor"]["updates_per_window"])
    return {
        "schema": "axbench-tr.tr03-training-plan/1",
        "status": "proposal; nothing executed; applies to arm R exactly and to arm T as an upper "
        "bound (T uses a human-approved subset of these rows)",
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_sha256": hashes,
        "rows": {"train_eligible": len(ids), "validation_eligible": len(elig_v)},
        "schedule": {k: v for k, v in sched.items() if k != "unexposed_row_ids"},
        "unexposed_row_ids": sched["unexposed_row_ids"],
        "eval_points_after_updates": [0, *points],
        "snapshot_points_after_updates": points,
        "diagnostic_validation": {
            "rows": len(diag),
            "ids_sha256": hashlib.sha256("\n".join(diag).encode()).hexdigest(),
            "tokens": sum(lengths[i] for i in diag),
        },
        "full_validation_tokens": sum(lengths[r["id"]] for r in elig_v),
        "monitor_windows": {
            "updates_per_window": win,
            "windows": window_count(updates, win),
        },
        "budget": b,
        "epoch_alternatives_not_selected": alternatives(ids, lengths, cfg, len(diag), len(elig_v)),
        "epoch_choice": "one epoch: each eligible row is exposed once; no measurement supports "
        "a second epoch, 87 templates already repeat about 58 times per epoch, checkpoint "
        "selection is forbidden, so longer training would add overfitting risk without a stop "
        "criterion; a different schedule is a different experiment",
    }


def plan_bytes(plan: Mapping[str, Any]) -> bytes:
    return (json.dumps(plan, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# ------------------------------------------------------------------------------------ CLI


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tr03", description="EXP-TR03 plan tooling; trains nothing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-config")
    p = sub.add_parser("plan")
    p.add_argument("--write", action="store_true", help=f"write {PLAN_PATH.name}")
    sub.add_parser("check-plan", help="the committed plan equals a fresh rebuild")
    sub.add_parser("check-therm", help="read pmset -g therm text from stdin")
    sub.add_parser("readiness")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "verify-config":
        problems = config_problems(load_config())
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "plan":
        data = plan_bytes(build_plan())
        if a.write:
            PLAN_PATH.write_bytes(data)
        sys.stdout.write(data.decode("utf-8"))
        return 0
    if a.cmd == "check-plan":
        same = PLAN_PATH.exists() and PLAN_PATH.read_bytes() == plan_bytes(build_plan())
        print("committed plan equals a fresh rebuild" if same else "committed plan differs")
        return 0 if same else 1
    if a.cmd == "check-therm":
        problems = parse_therm(sys.stdin.read())
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    report = tr03.gates_report()
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["research_training_open"] else 1


if __name__ == "__main__":
    sys.exit(main())
