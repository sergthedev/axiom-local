"""EXP-TR02 feasibility-smoke harness (independent of the EXP-010 harness; nothing here trains).

Status: **prepared, not executed.** No model weights have been loaded by this code. The model path
(``run_phase``) is reviewed by reading only; the data path, schedule, stop rules, parsers and
supervisor are unit-tested in ``tests/test_tr02_harness.py``. The first execution needs the owner's
separate approval (see ``PROPOSED_EXP-TR02.md`` in the EXP-TR01 evidence folder).

Why a harness instead of ``python -m mlx_lm lora`` (all read from mlx-lm 0.32.2 source, see README):

* the stock trainer *truncates* sequences over ``max_seq_length`` with only a printed warning;
* its ``Tokens/sec`` counts loss tokens (the supervised completion), not the tokens it computes on;
* its last in-loop validation runs *before* the last optimizer step;
* its chat dataset supervises the template's empty thinking block and a trailing newline, which
  neither the EXP-010 rendering nor the inference prompt contain.

Commands (``python training/tr02/tr02.py <command> ...``): ``pin`` (write input hashes into
``config.json``), ``verify-inputs``, ``schedule``, ``verify-model``, ``check-data``, ``supervise``,
``run`` (model; needs Apple Silicon, ``envs/model`` and owner approval).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_PATH = HERE / "config.json"
EVENTS = "events.jsonl"  # one JSON object per line, written by the driver, read by the supervisor


# ------------------------------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


# ------------------------------------------------------------------------- immutable inputs


def verify_inputs(export_dir: Path, config: dict[str, Any]) -> list[str]:
    """Problems found comparing an export directory with the hashes pinned in the config."""
    problems: list[str] = []
    pinned = config["inputs"]["export_files"]
    for name, digest in pinned.items():
        path = export_dir / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256_file(path) != digest:
            problems.append(f"hash differs: {name}")
    manifest = export_dir / "EXPORT_MANIFEST.json"
    if not manifest.is_file():
        problems.append("missing: EXPORT_MANIFEST.json")
    elif sha256_file(manifest) != config["inputs"]["export_manifest_sha256"]:
        problems.append("hash differs: EXPORT_MANIFEST.json")
    names = sorted(
        p.relative_to(export_dir).as_posix() for p in export_dir.rglob("*") if p.is_file()
    )
    extra = [n for n in names if n not in pinned and n != "EXPORT_MANIFEST.json"]
    problems += [f"unpinned file: {n}" for n in extra]
    problems += [f"test file present: {n}" for n in names if "test" in Path(n).name.lower()]
    return problems


def row_lengths(export_dir: Path, per_example: Path) -> dict[str, dict[str, int]]:
    """``{file: {row id: sequence tokens}}`` from the exported ids and the per-example lengths."""
    ids = json.loads((export_dir / "ROW_IDS.json").read_text("utf-8"))
    by_id = {r["id"]: int(r["separate_tokens"]) for r in read_jsonl(per_example)}
    return {name: {i: by_id[i] for i in rows} for name, rows in ids.items()}


# ----------------------------------------------------------------------------------- schedule


def planned_order(n_rows: int, seed: int) -> list[int]:
    """Row order of the stock trainer for ``batch_size=1`` after ``np.random.seed(seed)``.

    The stock ``iterate_batches`` sorts rows by ``len(dict)`` (equal for all rows, so the sort keeps
    the file order) and draws one ``np.random.permutation`` per epoch; one epoch is every row once.
    A test checks this against the installed ``mlx_lm`` when it is importable.
    """
    import numpy as np

    np.random.seed(seed)
    return [int(i) for i in np.random.permutation(n_rows)]


def schedule(
    export_dir: Path, per_example: Path, phase: dict[str, Any], seed: int
) -> dict[str, Any]:
    """The exact rows of a phase in order, one per microstep, with their token lengths."""
    lengths = row_lengths(export_dir, per_example)[f"{phase['data']}/train.jsonl"]
    ids = list(lengths)
    order = planned_order(len(ids), seed)
    microsteps = int(phase["microsteps"])
    if microsteps != len(ids):
        raise ValueError(f"{microsteps} microsteps do not expose {len(ids)} rows exactly once")
    steps = [
        {"microstep": k + 1, "id": ids[i], "tokens": lengths[ids[i]]} for k, i in enumerate(order)
    ]
    return {
        "phase": phase["name"],
        "seed": seed,
        "rows": len(ids),
        "microsteps": microsteps,
        "optimizer_updates": microsteps // int(phase["grad_accumulation"]),
        "tokens_total": sum(s["tokens"] for s in steps),
        "each_row_exposed_exactly_once": sorted(s["id"] for s in steps) == sorted(ids),
        "steps": steps,
    }


# ------------------------------------------------------------------------ dataset and checks


class TR02ChatDataset:
    """mlx-lm compatible chat dataset with the EXP-010 rendering and no truncation.

    ``process`` returns ``(tokens, offset)``: the chat template applied to the whole conversation
    (system, user, assistant), cut at the ``<|im_end|>`` that ends the answer, and the length of the
    prompt rendered with ``enable_thinking=False`` plus the generation prompt. Loss is therefore on
    the compact JSON and ``<|im_end|>`` only, exactly the EXP-010 convention. A sequence longer than
    ``max_seq_length`` raises; it is never truncated.
    """

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> None:
        self._data = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def process(self, d: dict[str, Any]) -> tuple[list[int], int]:
        messages = d["messages"]
        full = list(self.tokenizer.apply_chat_template(messages, return_dict=False))
        prompt = list(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )
        if full[: len(prompt)] != prompt:
            raise ValueError("the thinking-disabled prompt is not a prefix of the full sequence")
        if self.tokenizer.decode(full[-2:]) == "<|im_end|>\n":
            full = full[:-1]  # the template's trailing newline is not part of the EXP-010 sequence
        if self.tokenizer.decode(full[-1:]) != "<|im_end|>":
            raise ValueError("the sequence does not end with <|im_end|>")
        if len(full) > self.max_seq_length:
            raise ValueError(f"sequence of {len(full)} tokens exceeds {self.max_seq_length}")
        return full, len(prompt)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def check_rows(
    rows: list[dict[str, Any]],
    ids: list[str],
    expected: dict[str, int],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Every row: rendering, supervised span, length equal to the measured one, no truncation."""
    ds = TR02ChatDataset(rows, tokenizer, max_seq_length)
    problems: list[str] = []
    supervised: list[int] = []
    for rid, row in zip(ids, rows, strict=True):
        try:
            tokens, offset = ds.process(row)
        except ValueError as exc:
            problems.append(f"{rid}: {exc}")
            continue
        target = row["messages"][-1]["content"] + "<|im_end|>"
        if tokenizer.decode(tokens[offset:]) != target:
            problems.append(f"{rid}: supervised span is not the answer")
        if len(tokens) != expected[rid]:
            problems.append(f"{rid}: {len(tokens)} tokens, measured {expected[rid]}")
        if not 0 < offset < len(tokens):
            problems.append(f"{rid}: empty prompt or empty answer")
        supervised.append(len(tokens) - offset)
    return {
        "rows": len(rows),
        "problems": problems,
        "supervised_tokens": sum(supervised),
        "supervised_min": min(supervised, default=0),
        "supervised_max": max(supervised, default=0),
    }


class ExposureChecker:
    """Compares the rows the trainer actually draws with the declared schedule, step by step."""

    def __init__(self, planned: list[dict[str, Any]], hash_to_id: dict[str, str]) -> None:
        self.planned = planned
        self.hash_to_id = hash_to_id
        self.seen: list[str] = []

    def observe(self, token_hash: str, length: int) -> None:
        k = len(self.seen)
        if k >= len(self.planned):
            raise RuntimeError("the trainer asked for more rows than the schedule holds")
        row_id = self.hash_to_id.get(token_hash)
        want = self.planned[k]
        if row_id != want["id"]:
            raise RuntimeError(f"microstep {k + 1}: drew {row_id}, schedule says {want['id']}")
        if length != want["tokens"]:
            raise RuntimeError(f"microstep {k + 1}: {length} tokens, expected {want['tokens']}")
        self.seen.append(row_id)

    def complete(self) -> bool:
        return [s["id"] for s in self.planned] == self.seen


def tokens_hash(tokens: Iterable[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


# ------------------------------------------------------------------------------ system probes


def parse_swapusage(text: str) -> float:
    """``vm.swapusage`` -> used swap in GB (decimal). ``total = 2048.00M  used = 1024.50M ...``"""
    m = re.search(r"used\s*=\s*([\d.]+)\s*([KMGT])", text)
    if not m:
        raise ValueError(f"unrecognised swap usage: {text!r}")
    scale = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[m.group(2)]
    return float(m.group(1)) * scale / 1e9


def parse_pressure_level(text: str) -> int:
    """``kern.memorystatus_vm_pressure_level``: 1 normal, 2 warning, 4 critical."""
    return int(text.strip())


def parse_on_battery(text: str) -> bool:
    """``pmset -g batt`` first line: ``Now drawing from 'Battery Power'`` or ``'AC Power'``."""
    m = re.search(r"drawing from '([^']+)'", text)
    if not m:
        raise ValueError(f"unrecognised power source: {text!r}")
    return m.group(1).lower().startswith("battery")


def parse_rss_kb(text: str) -> float:
    """``ps -o rss= -p PID`` (kilobytes) -> GB (decimal)."""
    return int(text.strip()) * 1024 / 1e9


def macos_sampler(pid: int) -> dict[str, Any]:
    """One OS sample on macOS. Failures raise: an unreadable probe stops the run, never ignored."""

    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout

    return {
        "kind": "os",
        "t": time.time(),
        "rss_gb": parse_rss_kb(run("ps", "-o", "rss=", "-p", str(pid))),
        "swap_used_gb": parse_swapusage(run("sysctl", "-n", "vm.swapusage")),
        "pressure_level": parse_pressure_level(
            run("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
        ),
        "on_battery": parse_on_battery(run("pmset", "-g", "batt")),
    }


# ----------------------------------------------------------------------------------- stop rules


@dataclass(frozen=True)
class Limits:
    mlx_peak_gb: float = 10.0  # MLX allocator peak (mx.get_peak_memory), decimal GB
    rss_gb: float = 11.0  # process resident set, decimal GB; separate from the MLX peak
    swap_growth_gb: float = 2.0
    pressure_critical: int = 4
    window_ratio_floor: float = 0.5  # of the first window's full-token rate
    window_count: int = 3  # consecutive windows below the floor
    reports_per_window: int = 10  # one report per optimizer update: a window is 10 updates

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Limits:
        return cls(**cfg["limits"])


@dataclass
class StopRules:
    """Fed with driver events and OS samples; returns the first stop reason. No retry, ever."""

    limits: Limits
    microsteps_per_report: int
    planned_tokens: list[int]  # tokens of the row drawn at each microstep, in order
    swap_baseline: float | None = None
    _window: list[tuple[float, float]] = field(default_factory=list)  # (full tokens, seconds)
    window_rates: list[float] = field(default_factory=list)
    below: int = 0

    def feed(self, ev: dict[str, Any]) -> str | None:
        kind = ev["kind"]
        if kind == "os":
            return self._os(ev)
        if kind == "train":
            return self._train(ev)
        if kind == "error":
            return f"error:{ev.get('message', '')[:120]}"
        return None

    def _os(self, ev: dict[str, Any]) -> str | None:
        if self.swap_baseline is None:
            self.swap_baseline = ev["swap_used_gb"]
        if ev["on_battery"]:
            return "battery_power"
        if ev["pressure_level"] >= self.limits.pressure_critical:
            return "memory_pressure_critical"
        if ev["rss_gb"] > self.limits.rss_gb:
            return f"rss_above_limit:{ev['rss_gb']:.2f}GB"
        if ev["swap_used_gb"] - self.swap_baseline > self.limits.swap_growth_gb:
            return "swap_growth_above_limit"
        return None

    def _train(self, ev: dict[str, Any]) -> str | None:
        if not math.isfinite(ev["train_loss"]):
            return "non_finite_loss"
        if ev["peak_memory_gb"] > self.limits.mlx_peak_gb:
            return f"mlx_peak_above_limit:{ev['peak_memory_gb']:.2f}GB"
        it = int(ev["iteration"])
        first = it - self.microsteps_per_report
        tokens = sum(self.planned_tokens[first:it])  # full tokens computed on, not loss tokens
        seconds = self.microsteps_per_report / float(ev["iterations_per_second"])
        self._window.append((tokens, seconds))
        if len(self._window) < self.limits.reports_per_window:
            return None
        rate = sum(t for t, _ in self._window) / sum(s for _, s in self._window)
        self._window = []
        self.window_rates.append(rate)
        if len(self.window_rates) == 1:
            return None
        if rate < self.limits.window_ratio_floor * self.window_rates[0]:
            self.below += 1
        else:
            self.below = 0
        if self.below >= self.limits.window_count:
            return "throughput_below_half_of_first_window_for_3_windows"
        return None


# ----------------------------------------------------------------------------------- supervisor


def supervise(
    cmd: list[str],
    out_dir: Path,
    rules: StopRules,
    sampler: Callable[[int], dict[str, Any]],
    poll_s: float = 5.0,
    kill_after_s: float = 30.0,
) -> dict[str, Any]:
    """Run the driver as a child, apply the stop rules to its events and to OS samples.

    The child writes ``events.jsonl`` into ``out_dir`` (which it requires to be new and empty). The
    supervisor's own files go beside it, in ``<out_dir>.supervisor``. On the first stop reason the
    child is terminated and ``STOP.json`` records the reason; the run is over (no retry, no resume).
    """
    side = out_dir.with_name(out_dir.name + ".supervisor")
    side.mkdir(parents=True, exist_ok=True)
    events = out_dir / EVENTS
    log = side / "supervisor.jsonl"
    with (side / "child.log").open("wb") as child_log, log.open("w", encoding="utf-8") as sup:
        proc = subprocess.Popen(cmd, stdout=child_log, stderr=subprocess.STDOUT)
        offset = 0
        reason: str | None = None

        def drain() -> str | None:
            nonlocal offset
            if not events.exists():
                return None
            with events.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
            complete = chunk[: chunk.rfind(b"\n") + 1]  # never parse a half-written line
            offset += len(complete)
            for line in complete.decode("utf-8").splitlines():
                ev = json.loads(line)
                sup.write(json.dumps(ev, sort_keys=True) + "\n")
                if stop := rules.feed(ev):
                    return stop
            return None

        while reason is None:
            try:
                sample = sampler(proc.pid)
            except Exception as exc:  # noqa: BLE001 - an unreadable probe is a stop, not a skip
                sample = None
                reason = f"probe_failed:{type(exc).__name__}"
            if sample is not None:
                sup.write(json.dumps(sample, sort_keys=True) + "\n")
                reason = rules.feed(sample)
            reason = reason or drain()
            if reason is None and proc.poll() is not None:
                reason = drain()
                if reason is None:
                    break
            if reason is None:
                time.sleep(poll_s)
        code = proc.poll()
        if reason is not None and code is None:
            proc.terminate()
            try:
                proc.wait(kill_after_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        code = proc.returncode
    if reason is None and code != 0:
        reason = f"child_exit_code_{code}"
    result = {"stopped": reason is not None, "reason": reason, "child_exit_code": code}
    if reason is not None:
        (side / "STOP.json").write_text(json.dumps(result, indent=1) + "\n", "utf-8")
    return result


# -------------------------------------------------------------------------------- driver (model)


def run_phase(
    phase_name: str, model_dir: Path, tokenizer_dir: Path, export_dir: Path, out: Path
) -> None:
    """Train one phase in this process. NOT EXECUTED during preparation: no model has been run.

    A phase is one fresh process with its own adapter directory (the MLX peak counter and the
    optimizer are never shared between phases). The data are checked first; the schedule is
    enforced while training; the trainer's validation is not used (see the module docstring).
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import (
        TrainingArgs,
        TrainingCallback,
        default_loss,
        evaluate,
        iterate_batches,
        train,
    )
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = load_config()
    phase = next(p for p in cfg["phases"] if p["name"] == phase_name)
    hp = cfg["hyperparameters"]
    if out.exists() and any(out.iterdir()):
        raise SystemExit("the adapter/output directory must be fresh and empty")
    out.mkdir(parents=True, exist_ok=True)
    events = (out / EVENTS).open("a", encoding="utf-8")

    def emit(ev: dict[str, Any]) -> None:
        events.write(json.dumps(ev, sort_keys=True) + "\n")
        events.flush()

    try:
        problems = verify_inputs(export_dir, cfg)
        if problems:
            raise SystemExit("inputs differ from the pinned hashes: " + "; ".join(problems[:5]))
        mx.random.seed(int(hp["seed"]))
        model, tokenizer = load(str(model_dir))
        rows = read_jsonl(export_dir / phase["data"] / "train.jsonl")
        valid_rows = read_jsonl(export_dir / phase["data"] / "valid.jsonl")
        train_ds = CacheDataset(TR02ChatDataset(rows, tokenizer, hp["max_seq_length"]))
        valid_ds = CacheDataset(TR02ChatDataset(valid_rows, tokenizer, hp["max_seq_length"]))
        per_example = REPO / cfg["inputs"]["per_example_path"]
        sched = schedule(export_dir, per_example, phase, int(hp["seed"]))
        hash_to_id = {}
        for i, rid in enumerate(
            row_lengths(export_dir, per_example)[f"{phase['data']}/train.jsonl"]
        ):
            hash_to_id[tokens_hash(train_ds[i][0])] = rid
        checker = ExposureChecker(sched["steps"], hash_to_id)

        def checked_batches(**kwargs: Any):  # type: ignore[no-untyped-def]
            for batch, lengths in iterate_batches(**kwargs):
                n = int(lengths[0, 1])
                if kwargs["dataset"] is train_ds:
                    checker.observe(tokens_hash(batch[0, :n].tolist()), n)
                yield batch, lengths

        model.freeze()
        linear_to_lora_layers(
            model,
            hp["lora_layers"],
            {
                "rank": hp["lora_rank"],
                "scale": hp["lora_scale"],
                "dropout": hp["lora_dropout"],
                "keys": hp["lora_keys"],
            },
        )
        (out / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "num_layers": hp["lora_layers"],
                    "lora_parameters": {
                        "rank": hp["lora_rank"],
                        "scale": hp["lora_scale"],
                        "dropout": hp["lora_dropout"],
                        "keys": hp["lora_keys"],
                    },
                }
            ),
            "utf-8",
        )

        def val_loss(tag: str, updates_done: int) -> float:
            tic = time.perf_counter()
            loss = evaluate(
                model,
                valid_ds,
                batch_size=1,
                num_batches=-1,  # every validation row; the stock default of 25 would skip rows
                max_seq_length=hp["max_seq_length"],
                loss=default_loss,
            )
            emit(
                {
                    "kind": "val",
                    "tag": tag,
                    "updates_done": updates_done,
                    "val_loss": loss,
                    "seconds": time.perf_counter() - tic,
                    "diagnostic_only": phase.get("loss_is_diagnostic", False),
                }
            )
            return loss

        class Reporter(TrainingCallback):
            def on_train_loss_report(self, info: dict[str, Any]) -> None:
                emit(
                    {
                        "kind": "train",
                        "iteration": info["iteration"],
                        "train_loss": float(info["train_loss"]),
                        "iterations_per_second": float(info["iterations_per_second"]),
                        "loss_tokens_per_second": float(info["tokens_per_second"]),
                        "peak_memory_gb": float(info["peak_memory"]),
                        "t": time.time(),
                    }
                )

        val_loss("before_training", 0)
        args = TrainingArgs(
            batch_size=1,
            iters=int(phase["microsteps"]),
            val_batches=-1,
            steps_per_report=int(phase["microsteps_per_report"]),
            steps_per_eval=10**9,  # never inside the loop
            steps_per_save=10**9,  # no intermediate checkpoints: no checkpoint selection
            max_seq_length=int(hp["max_seq_length"]),
            adapter_file=str(out / "adapters.safetensors"),
            grad_checkpoint=bool(hp["grad_checkpoint"]),
            grad_accumulation_steps=int(phase["grad_accumulation"]),
        )
        np.random.seed(int(hp["seed"]))  # the stock iterator draws its order from the global state
        tic = time.perf_counter()
        train(
            model=model,
            optimizer=optim.Adam(learning_rate=hp["learning_rate"]),
            train_dataset=train_ds,
            val_dataset=None,
            args=args,
            iterate_batches=checked_batches,
            training_callback=Reporter(),
        )
        emit(
            {"kind": "train_total", "seconds_including_report_overhead": time.perf_counter() - tic}
        )
        if not checker.complete():
            raise RuntimeError("the rows drawn differ from the declared schedule")
        val_loss("after_training", int(phase["optimizer_updates"]))
        emit({"kind": "done", "peak_memory_gb": mx.get_peak_memory() / 1e9})
    except BaseException as exc:
        emit({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        events.close()


def run_reload(model_dir: Path, export_dir: Path, adapter_dir: Path, out: Path) -> None:
    """Load the saved smoke adapter in a fresh process and re-evaluate the 40 validation rows.

    NOT EXECUTED during preparation. The loss must match the smoke run's ``after_training``
    validation loss; the difference is reported, nothing is scored or selected.
    """
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import default_loss, evaluate

    cfg = load_config()
    hp = cfg["hyperparameters"]
    if out.exists() and any(out.iterdir()):
        raise SystemExit("the output directory must be fresh and empty")
    out.mkdir(parents=True, exist_ok=True)
    if verify_inputs(export_dir, cfg):
        raise SystemExit("inputs differ from the pinned hashes")
    model, tokenizer = load(str(model_dir), adapter_path=str(adapter_dir))
    valid = CacheDataset(
        TR02ChatDataset(
            read_jsonl(export_dir / "smoke" / "valid.jsonl"), tokenizer, hp["max_seq_length"]
        )
    )
    loss = evaluate(
        model, valid, batch_size=1, num_batches=-1, max_seq_length=hp["max_seq_length"],
        loss=default_loss,
    )  # fmt: skip
    after = [
        e
        for e in read_jsonl(adapter_dir / EVENTS)
        if e["kind"] == "val" and e["tag"] == "after_training"
    ]
    record = {
        "reload_val_loss": loss,
        "smoke_after_training_val_loss": after[-1]["val_loss"] if after else None,
    }
    (out / "reload.json").write_text(json.dumps(record, indent=1) + "\n", "utf-8")


# ------------------------------------------------------------------------------------------ CLI


def _export_pins(export_dir: Path) -> dict[str, Any]:
    manifest = json.loads((export_dir / "EXPORT_MANIFEST.json").read_text("utf-8"))
    return {
        "export_files": {name: meta["sha256"] for name, meta in manifest["files"].items()},
        "export_manifest_sha256": sha256_file(export_dir / "EXPORT_MANIFEST.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pin", "verify-inputs", "schedule", "check-data"):
        p = sub.add_parser(name)
        p.add_argument("--export", required=True, help="directory written by `export-mlx`")
        if name == "schedule":
            p.add_argument("--phase", required=True)
            p.add_argument("--per-example", required=True)
        if name == "check-data":
            p.add_argument("--tokenizer-dir", required=True)
            p.add_argument("--per-example", required=True)
    p = sub.add_parser("verify-model")
    p.add_argument("--model-dir", required=True)
    for name in ("run", "supervise"):
        p = sub.add_parser(name)
        for a in ("phase", "model-dir", "tokenizer-dir", "export", "out"):
            p.add_argument(f"--{a}", required=True)
        p.add_argument("--adapter-dir", help="phase reload only: the smoke output directory")
        if name == "supervise":
            p.add_argument("--poll-seconds", type=float, default=5.0)
    a = parser.parse_args(argv)
    cfg = load_config()
    if a.cmd == "pin":
        corpus = REPO / "data" / "axbench-tr" / cfg["inputs"]["corpus_version"]
        manifest = json.loads((corpus / "MANIFEST.json").read_text("utf-8"))
        cfg["inputs"]["corpus_files_sha256"] = {
            n: m["sha256"] for n, m in manifest["files"].items()
        }
        cfg["inputs"]["per_example_sha256"] = sha256_file(REPO / cfg["inputs"]["per_example_path"])
        cfg["inputs"] |= _export_pins(Path(a.export))
        CONFIG_PATH.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n", "utf-8")
        print("pinned", len(cfg["inputs"]["export_files"]), "files")
        return 0
    if a.cmd == "verify-inputs":
        problems = verify_inputs(Path(a.export), cfg)
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "schedule":
        phase = next(p for p in cfg["phases"] if p["name"] == a.phase)
        out = schedule(
            Path(a.export), Path(a.per_example), phase, int(cfg["hyperparameters"]["seed"])
        )
        print(json.dumps({k: v for k, v in out.items() if k != "steps"}, indent=1))
        return 0
    if a.cmd == "check-data":
        from transformers import AutoTokenizer  # only the tokenizer; mlx-lm brings transformers

        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
        export_dir = Path(a.export)
        lengths = row_lengths(export_dir, Path(a.per_example))
        report: dict[str, Any] = {}
        for name, expected in lengths.items():
            if not name.endswith(".jsonl"):
                continue
            rows = read_jsonl(export_dir / name)
            report[name] = check_rows(
                rows, list(expected), expected, tok, int(cfg["hyperparameters"]["max_seq_length"])
            )
        print(
            json.dumps(
                {k: {**v, "problems": v["problems"][:5]} for k, v in report.items()}, indent=1
            )
        )
        return 1 if any(v["problems"] for v in report.values()) else 0
    if a.cmd == "verify-model":
        model_dir = Path(a.model_dir)
        prov = json.loads((model_dir / "PROVENANCE.json").read_text("utf-8"))
        problems = []
        if prov["revision"] != cfg["model"]["revision"] or prov.get("tokenizer_only"):
            problems.append("revision differs or weights not provisioned")
        for name, meta in prov["files"].items():
            if sha256_file(model_dir / name) != meta["sha256"]:
                problems.append(f"hash differs: {name}")
            if name.endswith(".safetensors") and meta["verified_against"] != "published hash":
                problems.append(f"weights not verified against a published hash: {name}")
        for name, digest in cfg["model"]["tokenizer_sha256"].items():
            if prov["files"].get(name, {}).get("sha256") != digest:
                problems.append(f"tokenizer file differs from the pin: {name}")
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "supervise":
        if sys.platform != "darwin":
            raise SystemExit("the supervisor reads macOS probes: run it on the Apple Silicon Mac")
        phase = next(p for p in cfg["phases"] if p["name"] == a.phase)
        plan = schedule(
            Path(a.export),
            REPO / cfg["inputs"]["per_example_path"],
            phase,
            cfg["hyperparameters"]["seed"],
        )
        rules = StopRules(
            Limits.from_config(cfg),
            int(phase["microsteps_per_report"]),
            [s["tokens"] for s in plan["steps"]],
        )
        child = [sys.executable, str(Path(__file__).resolve()), "run"]
        for key in ("phase", "model_dir", "tokenizer_dir", "export", "out"):
            child += [f"--{key.replace('_', '-')}", getattr(a, key)]
        result = supervise(child, Path(a.out), rules, macos_sampler, poll_s=a.poll_seconds)
        print(json.dumps(result, indent=1))
        return 1 if result["stopped"] else 0
    if a.cmd == "run" and a.phase == "reload":
        run_reload(Path(a.model_dir), Path(a.export), Path(a.adapter_dir), Path(a.out))
        return 0
    if a.cmd == "run":
        run_phase(a.phase, Path(a.model_dir), Path(a.tokenizer_dir), Path(a.export), Path(a.out))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
