"""EXP-TR04 LoRA training driver: one exploratory epoch on the audited, rights-screened rows.

Status: **prepared, not executed, not approved.** No model weights have been loaded by this code,
nothing has been trained and no technical clearance exists. The driver is exploratory: the data it
trains on is machine-audited and machine-screened, **not human-reviewed**, and the result is
neither a human-validated nor a deployment-ready model.

The driver reuses the frozen EXP-TR02 harness as it is (``training/tr02/tr02.py`` is imported and
never edited): the chat dataset and exact loss mask, the exposure checker, the LoRA name contract,
``run_phase`` (train, save, validate before and after), the supervisor, the event audit
(``StopRules``), evidence collection and sanitizing. What TR04 adds around it:

* a config (``config.json``) that pins the corpus, the audit outputs, the MLX export, the held-out
  benchmark and the model, and is checked against TR02 where it must be equal;
* ``phases``: ``train`` (every kept row exactly once, one epoch) and ``reload`` (a fresh process
  re-evaluates the saved adapter); there is no probe and no smoke phase here because TR02 already
  ran them (their rates are the basis of the budget);
* ``TR04Rules``: the TR02 event audit plus the sustained-run rules of ``training/tr03``
  (``LongRunMonitor``'s thermal parser, throughput reference and floor, sustained memory-pressure
  warning) and a hard wall-clock cap derived from the budget;
* a macOS sampler that also records ``pmset -g therm``.

Commands (``uv run --locked --project envs/model python training/tr04/tr04.py <command>`` from the
repository root, except where noted): ``verify-config``, ``pin``, ``plan``, ``verify-inputs``,
``schedule``, ``verify-model``, ``check-data``, ``check-iterator``, ``check-lora``, ``preflight``
(no weights are loaded), ``supervise`` and ``run`` (both would train; not to be started before
technical clearance), ``collect-evidence`` and ``sanitize-evidence``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_PATH = HERE / "config.json"
TR02_PATH = REPO / "training" / "tr02" / "tr02.py"
TR02_CONFIG_PATH = REPO / "training" / "tr02" / "config.json"
TR03_PATH = REPO / "training" / "tr03" / "tr03.py"
DATA_DIR = REPO / "data" / "axbench-tr" / "tr04"

try:  # the repository's source tree is importable when run through ``uv run`` from its root
    from axbench_tr import tr04 as tr04data
    from axbench_tr import tr04_heldout
except ModuleNotFoundError:  # pragma: no cover - plain ``python`` without the project
    sys.path.insert(0, str(REPO / "src"))
    from axbench_tr import tr04 as tr04data
    from axbench_tr import tr04_heldout


def _load_module(name: str, path: Path) -> Any:
    """A frozen sibling harness as a module (read only; its source is never edited)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


tr02 = _load_module("tr02", TR02_PATH)
tr03_tools = _load_module("tr03_tools", TR03_PATH)

TRAIN_PHASE, RELOAD_PHASE = "train", "reload"
DATA_FILES = (
    "AUDIT.json",
    "EXCLUSIONS.csv",
    "KEEP_TRAIN.txt",
    "KEEP_VALIDATION.txt",
    "MANIFEST.json",
    "MUTATION_REPORT.json",
    "TEMPLATE_CONTRACT.json",
    "rights/IZMIR_PACKAGE_NOTES.json",
)
HELDOUT_FILES = ("HELDOUT.jsonl", "HELDOUT_AUDIT.json", "HELDOUT_MANIFEST.json")
VALIDATION_PASSES = 3  # before training, after training, and the reload process


def sha256_file(path: Path) -> str:
    return tr02.sha256_file(path)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def write_config(cfg: Mapping[str, Any], path: Path = CONFIG_PATH) -> None:
    path.write_bytes((json.dumps(cfg, indent=1, sort_keys=True) + "\n").encode("utf-8"))


# ------------------------------------------------------------------------------ pinned inputs


def data_hashes(root: Path = DATA_DIR) -> dict[str, dict[str, str]]:
    """SHA-256 of every audit output and of the held-out benchmark files."""
    return {
        "audit_files_sha256": {name: sha256_file(root / name) for name in DATA_FILES},
        "heldout_files_sha256": {
            name: sha256_file(root / "heldout" / name) for name in HELDOUT_FILES
        },
    }


def data_problems(cfg: Mapping[str, Any], root: Path = DATA_DIR) -> list[str]:
    """The committed audit outputs and held-out files against the hashes pinned in the config."""
    problems: list[str] = []
    for key, sub in (("audit_files_sha256", ""), ("heldout_files_sha256", "heldout")):
        pinned = cfg["inputs"][key]
        for name, digest in pinned.items():
            path = root / sub / name if sub else root / name
            if not path.is_file():
                problems.append(f"{key}: missing {name}")
            elif sha256_file(path) != digest:
                problems.append(f"{key}: hash differs {name}")
    return problems


def export_keep_problems(export_dir: Path, root: Path = DATA_DIR) -> list[str]:
    """The exported rows are exactly the kept ids, in order (the export is only a rendering)."""
    ids = json.loads((export_dir / "ROW_IDS.json").read_text("utf-8"))
    keep = {
        "tr04/train.jsonl": (root / "KEEP_TRAIN.txt").read_text("utf-8").split(),
        "tr04/valid.jsonl": (root / "KEEP_VALIDATION.txt").read_text("utf-8").split(),
    }
    problems = [
        f"export rows differ from the keep list: {n}" for n in keep if ids.get(n) != keep[n]
    ]
    problems += [f"unexpected export part: {n}" for n in sorted(set(ids) - set(keep))]
    heldout = {r["id"] for r in tr04_heldout.load_rows(root / "heldout" / "HELDOUT.jsonl")}
    exported = {i for rows in ids.values() for i in rows}
    if heldout & exported:
        problems.append("a held-out row id is in the training export")
    return problems


def pin_inputs(cfg: dict[str, Any], export_dir: Path) -> dict[str, Any]:
    """Fill ``inputs`` (corpus, per-example, export, audit and held-out hashes) from the files."""
    corpus = REPO / "data" / "axbench-tr" / cfg["inputs"]["corpus_version"]
    manifest = json.loads((corpus / "MANIFEST.json").read_text("utf-8"))
    cfg["inputs"]["corpus_files_sha256"] = {
        n: m["sha256"]
        for n, m in manifest["files"].items()
        if n in ("train.jsonl", "validation.jsonl")
    }
    cfg["inputs"]["per_example_sha256"] = sha256_file(REPO / cfg["inputs"]["per_example_path"])
    cfg["inputs"] |= tr02._export_pins(export_dir)
    cfg["inputs"] |= data_hashes()
    return cfg


# ----------------------------------------------------------------------------------- budget


def kept_lengths(cfg: Mapping[str, Any]) -> dict[str, list[int]]:
    """Measured sequence length of every kept row, in keep-list order (per-example file, hashed)."""
    raw = (REPO / cfg["inputs"]["per_example_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg["inputs"]["per_example_sha256"]:
        raise ValueError("the per-example length file differs from the pinned hash")
    by_id = {
        r["id"]: int(r["separate_tokens"])
        for r in (json.loads(line) for line in raw.decode("utf-8").splitlines())
    }
    keep = tr04data.kept_ids()
    return {k: [by_id[i] for i in ids] for k, ids in keep.items()}


def budget(
    cfg: Mapping[str, Any], lengths: Mapping[str, Sequence[int]] | None = None
) -> dict[str, Any]:
    """Seconds from the measured TR02 rates and a conservative thermal allowance (extrapolated).

    Same arithmetic as ``training/tr03`` ``budget``, with this experiment's pass structure: the
    full validation set is evaluated three times (before, after, in the reload process) and no
    intermediate evaluation exists. Nothing here is measured at this scale.
    """
    lengths = lengths or kept_lengths(cfg)
    hp, rates = cfg["hyperparameters"], cfg["rates"]
    cap = int(hp["max_seq_length"])
    train_tokens = sum(lengths["train"])
    padded = sum(tr02.padded_tokens(n, cap) for n in lengths["train"])
    rate = float(rates["full_tokens_per_s_slowest_observed"])
    per_row = float(rates["validation_seconds_per_row_slowest_observed"])
    load = float(rates["model_load_seconds"])
    slow = 1.0 - float(rates["thermal_allowance_fraction"])
    train_nominal = train_tokens / rate
    val_nominal = per_row * len(lengths["validation"]) * VALIDATION_PASSES
    total_nominal = load * 2 + train_nominal + val_nominal  # two process loads: train and reload
    total_cons = load * 2 + (train_nominal + val_nominal) / slow
    step = int(rates["wall_cap_round_to_seconds"])
    cap_s = math.ceil(total_cons * float(rates["wall_cap_factor"]) / step) * step
    train_cap = (
        math.ceil(
            (load + (train_nominal + per_row * len(lengths["validation"]) * 2) / slow)
            * float(rates["wall_cap_factor"])
            / step
        )
        * step
    )
    return {
        "rows": {"train": len(lengths["train"]), "validation": len(lengths["validation"])},
        "full_tokens_exposed": train_tokens,
        "padded_tokens_exposed": padded,
        "seconds": {
            "training_nominal": round(train_nominal),
            "validation_nominal_three_passes": round(val_nominal),
            "total_nominal": round(total_nominal),
            "total_conservative": round(total_cons),
            "hard_wall_cap_train_phase": train_cap,
            "hard_wall_cap_reload_phase": 900,
            "hard_wall_cap_total": cap_s,
        },
        "minutes": {
            "total_nominal": round(total_nominal / 60, 1),
            "total_conservative": round(total_cons / 60, 1),
            "hard_wall_cap_train_phase": round(train_cap / 60, 1),
        },
        "status": "extrapolated from EXP-TR02 rates (2.5 minute smoke); not measured at this scale",
    }


# ---------------------------------------------------------------------------- config checks


def config_problems(cfg: Mapping[str, Any], tr02_cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Inconsistencies with EXP-TR02 (where equality is required) or inside the config itself."""
    tr02_cfg = tr02_cfg or json.loads(TR02_CONFIG_PATH.read_text("utf-8"))
    problems: list[str] = []
    if cfg.get("schema") != "axbench-tr.exp-tr04-config/1":
        problems.append("schema_is_not_exp_tr04_config_1")
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
    if hp.get("epochs") != 1:
        problems.append("epochs_must_be_one")
    if hp.get("init_adapter") is not None:
        problems.append("init_adapter_must_be_null")
    if hp.get("micro_batch") != 1 or hp.get("grad_accumulation") != 4:
        problems.append("effective_batch_is_not_1_by_4")
    inputs02 = tr02_cfg["inputs"]
    for key in ("corpus_version", "per_example_path", "per_example_sha256"):
        if cfg["inputs"][key] != inputs02[key]:
            problems.append(f"inputs_{key}_differs_from_tr02")
    if cfg["inputs"]["corpus_files_sha256"] != inputs02["corpus_files_sha256"]:
        problems.append("corpus_files_differ_from_tr02")
    lim, lim02 = cfg["limits"], tr02_cfg["limits"]
    for key in (
        "mlx_peak_gb",
        "rss_gb",
        "swap_growth_gb",
        "pressure_critical",
        "reload_abs_tolerance",
    ):
        if lim[key] != lim02[key]:
            problems.append(f"limit_{key}_differs_from_tr02")
    for key in ("window_count", "window_ratio_floor"):
        if lim[key] != lim02[key]:
            problems.append(f"limit_{key}_differs_from_tr02")
    mon = cfg["monitor"]
    if lim["reports_per_window"] != mon["updates_per_window"]:
        problems.append("reports_per_window_differs_from_monitor_updates_per_window")
    if not (0 < mon["floor_ratio"] < mon["slow_ratio"] < 1):
        problems.append("monitor_ratios_must_satisfy_0<floor<slow<1")
    status = cfg["status"]
    for phrase in ("not approved", "no clearance", "exploratory", "not human-validated"):
        if phrase not in status:
            problems.append(f"status_must_say:{phrase}")
    names = [p["name"] for p in cfg["phases"]]
    if names != [TRAIN_PHASE, RELOAD_PHASE]:
        problems.append("phases_must_be_train_then_reload")
    else:
        train, reload_ = cfg["phases"]
        keep = tr04data.kept_ids()
        n = len(keep["train"])
        if train["data"] != "tr04" or train["microsteps"] != n:
            problems.append("train_microsteps_differ_from_the_kept_train_rows")
        if (
            n % train["grad_accumulation"]
            or train["optimizer_updates"] != n // train["grad_accumulation"]
        ):
            problems.append("train_optimizer_updates_inconsistent")
        if train["grad_accumulation"] != hp.get("grad_accumulation"):
            problems.append("phase_accumulation_differs_from_hyperparameters")
        if train["microsteps_per_report"] != train["grad_accumulation"]:
            problems.append("one_report_per_optimizer_update_required")
        if reload_.get("requires") != TRAIN_PHASE or reload_.get("kind") != "reload":
            problems.append("reload_must_require_train")
        if train.get("requires") is not None:
            problems.append("train_phase_must_not_require_a_prior_phase")
    never = " ".join(cfg["data"]["never_used"]).lower()
    for term in ("sealed", "axbench v0.1", "held-out"):
        if term not in never:
            problems.append(f"never_used_must_name:{term}")
    if (
        cfg["evaluation"]["human_validated"] is not False
        or cfg["evaluation"]["deployment_ready"] is not False
    ):
        problems.append("evaluation_must_say_not_human_validated_not_deployment_ready")
    return problems


def consistency_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Config checks that read the committed data (hashes, lengths, the frozen budget)."""
    problems = config_problems(cfg) + data_problems(cfg)
    fresh = budget(cfg)
    if cfg["budget"] != fresh:
        problems.append("frozen_budget_differs_from_a_fresh_computation")
    train, reload_ = cfg["phases"]
    if train.get("max_wall_s") != fresh["seconds"]["hard_wall_cap_train_phase"]:
        problems.append("train_wall_cap_differs_from_budget")
    if reload_.get("max_wall_s") != fresh["seconds"]["hard_wall_cap_reload_phase"]:
        problems.append("reload_wall_cap_differs_from_budget")
    return problems


# ------------------------------------------------------------------------- preflight (no weights)


def preflight(
    config: dict[str, Any],
    phase_name: str,
    export_dir: Path,
    model_dir: Path,
    out: Path,
    after: Path | None,
    expect_commit: str,
    *,
    repo: Path = REPO,
    env: Mapping[str, str] | None = None,
    versions: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """The TR02 preflight plus the TR04 checks; loads no weights."""
    record, problems = _tr02_preflight(
        config, phase_name, export_dir, model_dir, out, after, expect_commit,
        repo=repo, env=env, versions=versions,
    )  # fmt: skip
    problems = list(problems)
    problems += [f"config: {p}" for p in config_problems(config)]
    problems += [f"data: {p}" for p in data_problems(config, repo / "data" / "axbench-tr" / "tr04")]
    if (export_dir / "ROW_IDS.json").is_file():
        problems += export_keep_problems(export_dir, repo / "data" / "axbench-tr" / "tr04")
    record = {
        **record,
        "experiment": "EXP-TR04",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "audit_files_sha256": config["inputs"]["audit_files_sha256"],
        "heldout_files_sha256": config["inputs"]["heldout_files_sha256"],
    }
    return record, problems


_tr02_preflight = tr02.preflight  # captured before the child patches it


# ------------------------------------------------------------------------------ stop rules


def therm_text(output: str) -> str:
    """``pmset -g therm`` output as the one-line, ``;``-joined form ``parse_therm`` accepts."""
    return "; ".join(line.strip() for line in output.splitlines() if line.strip())


def macos_sampler(pid: int | None) -> dict[str, Any]:
    """The TR02 sample plus the thermal status (an unreadable probe stops the run)."""
    import subprocess

    sample = tr02.macos_sampler(pid)
    out = subprocess.run(
        ("pmset", "-g", "therm"), capture_output=True, text=True, check=True, timeout=10
    ).stdout
    sample["therm"] = therm_text(out)
    return sample


@dataclass
class TR04Rules(tr02.StopRules):
    """TR02's event audit plus the sustained-run rules of EXP-TR03 and a wall-clock cap.

    Added on top of ``StopRules`` (which stays unchanged and still fires first): any thermal or
    performance statement other than the three known 'no ...' lines (``parse_therm``, fail
    closed), a steady-state memory-pressure warning, a throughput below ``slow_ratio`` of the
    reference (median of the first ``reference_windows`` windows) for ``slow_windows`` windows or
    below ``floor_ratio`` for ``floor_windows`` windows, and a wall-clock time past ``max_wall_s``
    measured from the pre-launch baseline.
    """

    mon: Mapping[str, Any] | None = None
    max_wall_s: float = 0.0
    t0: float | None = None
    pressure_run: int = 0
    slow_run: int = 0
    floor_run: int = 0
    _seen_windows: int = 0

    def _wall(self, t: Any) -> str | None:
        if not isinstance(t, int | float) or isinstance(t, bool) or not math.isfinite(t):
            return "malformed_timestamp"
        if self.t0 is None:
            self.t0 = float(t)
        elapsed = float(t) - self.t0
        if self.max_wall_s and elapsed > self.max_wall_s:
            return f"max_wall_time:{elapsed:.0f}s>{self.max_wall_s:.0f}s"
        return None

    def prelaunch(self, sample: Mapping[str, Any]) -> str | None:
        self.t0 = None
        reason = super().prelaunch(sample)
        reason = reason or self._thermal(sample)
        return reason or self._wall(sample.get("t"))

    @staticmethod
    def _thermal(ev: Mapping[str, Any]) -> str | None:
        problems = tr03_tools.parse_therm(str(ev.get("therm", "")))
        return f"thermal_or_performance_warning:{'; '.join(problems)[:120]}" if problems else None

    def _os(self, ev: Mapping[str, Any]) -> str | None:
        reason = super()._os(ev)
        reason = reason or self._thermal(ev) or self._wall(ev.get("t"))
        if reason:
            return reason
        assert self.mon is not None
        if self.pos >= 4 and int(ev["pressure_level"]) >= int(self.mon["pressure_warning_level"]):
            self.pressure_run += 1  # a warning level is tolerated at load, not as a steady state
            if self.pressure_run >= int(self.mon["pressure_warning_samples"]):
                return f"sustained_memory_pressure_warning:{self.pressure_run}_samples"
        else:
            self.pressure_run = 0
        return None

    def _ev_train(self, ev: Mapping[str, Any]) -> None:
        super()._ev_train(ev)
        if len(self.window_rates) == self._seen_windows:
            return
        self._seen_windows = len(self.window_rates)
        assert self.mon is not None
        refs = int(self.mon["reference_windows"])
        if len(self.window_rates) <= refs:
            return  # the reference is the median of the first windows; rules apply after it
        head = sorted(self.window_rates[:refs])
        mid = len(head) // 2
        ref = head[mid] if len(head) % 2 else (head[mid - 1] + head[mid]) / 2
        rate = self.window_rates[-1]
        self.slow_run = self.slow_run + 1 if rate < float(self.mon["slow_ratio"]) * ref else 0
        self.floor_run = self.floor_run + 1 if rate < float(self.mon["floor_ratio"]) * ref else 0
        if self.floor_run >= int(self.mon["floor_windows"]):
            raise tr02.Stop(
                f"throughput_floor:{self.floor_run}_windows_below_{self.mon['floor_ratio']}"
            )
        if self.slow_run >= int(self.mon["slow_windows"]):
            raise tr02.Stop(
                f"sustained_throughput_loss:{self.slow_run}_windows_below_{self.mon['slow_ratio']}"
            )


def build_rules(
    cfg: dict[str, Any],
    phase_name: str,
    export_dir: Path,
    model_dir: Path,
    prior: dict[str, Any],
    expect_commit: str,
) -> TR04Rules:
    phase = tr02.phase_config(cfg, phase_name)
    hp = cfg["hyperparameters"]
    shapes = tr02.expected_lora_shapes(
        json.loads((model_dir / "config.json").read_text("utf-8")), hp
    )
    common: dict[str, Any] = {
        "limits": tr02.Limits.from_config(cfg),
        "phase": phase_name,
        "expect_commit": expect_commit,
        "lora_tensors": len(shapes),
        "lora_params": tr02.trainable_params(shapes),
        "mon": cfg["monitor"],
        "max_wall_s": float(phase["max_wall_s"]),
    }
    valid_rows = len(tr02.read_jsonl(export_dir / "tr04" / "valid.jsonl"))
    if phase.get("kind") == "reload":
        return TR04Rules(
            **common,
            kind="reload",
            optimizer_updates=int(tr02.phase_config(cfg, TRAIN_PHASE)["optimizer_updates"]),
            valid_rows=valid_rows,
            reference_loss=float(prior["reference_val_loss"]),
            reference_adapter_sha256=prior["adapter_sha256"],
            reference_config_sha256=prior["adapter_config_sha256"],
        )
    plan = tr02.schedule(export_dir, cfg, phase)
    return TR04Rules(
        **common,
        kind="train",
        microsteps=int(phase["microsteps"]),
        microsteps_per_report=int(phase["microsteps_per_report"]),
        optimizer_updates=int(phase["optimizer_updates"]),
        full_tokens=[s["tokens"] for s in plan["steps"]],
        padded_tokens=[s["padded_tokens"] for s in plan["steps"]],
        valid_rows=valid_rows,
    )


# ------------------------------------------------------------------------------ model path


def run_reload(
    model_dir: Path, export_dir: Path, after: Path, out: Path, expect_commit: str
) -> None:
    """Load the saved adapter in a fresh process and re-evaluate every validation row.

    NOT EXECUTED during preparation. Adapted from ``tr02.run_reload`` (which hard-codes the smoke
    phase and its 40 rows): the adapter and its config must hash to the values the train phase's
    ``COMPLETE.json`` recorded; the loss must equal the post-training loss within
    ``limits.reload_abs_tolerance`` or the run exits non-zero. Nothing is generated or scored.
    """
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import evaluate

    cfg = load_config()
    hp = cfg["hyperparameters"]
    tol = tr02.Limits.from_config(cfg).reload_abs_tolerance
    loss_fn = tr02.make_exact_loss()
    emit, _ = tr02._start(cfg, RELOAD_PHASE, export_dir, model_dir, out, after, expect_commit)
    try:
        prior, problems = tr02.require_complete(after, TRAIN_PHASE)
        if problems:
            raise RuntimeError("; ".join(problems))
        adapter_path, config_path = after / tr02.ADAPTER_FILE, after / tr02.ADAPTER_CONFIG
        tic = time.perf_counter()
        model, tokenizer = load(str(model_dir), adapter_path=str(after))
        emit(
            {
                "kind": "load",
                "seconds": time.perf_counter() - tic,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
        )
        saved = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(adapter_path)).items()}
        want = tr02.expected_lora_shapes(
            json.loads((model_dir / "config.json").read_text("utf-8")), hp
        )
        bad = tr02.lora_problems({k: v.shape for k, v in saved.items()}, want)
        emit(
            {
                "kind": "adapter_check",
                "adapter_sha256": sha256_file(adapter_path),
                "config_sha256": sha256_file(config_path),
                "tensors": len(saved),
                "shapes_ok": not bad,
                "finite": all(bool(np.isfinite(a).all()) for a in saved.values()),
            }
        )
        live = {
            k: np.array(v.astype(mx.float32))
            for k, v in tree_flatten(model.parameters())
            if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
        }
        mismatch = tr02.live_adapter_problems(saved, live)
        if mismatch:
            raise RuntimeError(
                "the reloaded adapter is not the saved file: " + "; ".join(mismatch[:3])
            )
        valid_rows = tr02.read_jsonl(export_dir / "tr04" / "valid.jsonl")
        valid = CacheDataset(tr02.TR02ChatDataset(valid_rows, tokenizer, int(hp["max_seq_length"])))
        tic = time.perf_counter()
        loss = float(
            evaluate(
                model,
                valid,
                batch_size=1,
                num_batches=-1,
                max_seq_length=int(hp["max_seq_length"]),
                loss=loss_fn,
            )  # fmt: skip
        )
        emit(
            {
                "kind": "val",
                "tag": "reload",
                "updates_done": int(tr02.phase_config(cfg, TRAIN_PHASE)["optimizer_updates"]),
                "rows": len(valid_rows),
                "val_loss": loss,
                "seconds": time.perf_counter() - tic,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
        )
        ref = float(prior["reference_val_loss"])
        emit(
            {
                "kind": "reload",
                "reference_loss": ref,
                "reload_loss": loss,
                "abs_diff": abs(loss - ref),
                "tolerance": tol,
            }
        )
        if not math.isfinite(loss) or abs(loss - ref) > tol:
            raise RuntimeError(f"reload loss {loss} differs from the post-training loss {ref}")
        emit({"kind": "done", "peak_memory_gb": mx.get_peak_memory() / 1e9})
    except BaseException as exc:
        emit({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        emit.close()


def _patch_tr02_for_child() -> None:
    """In the supervised child only: TR02's ``run_phase`` reads *this* config and preflight."""
    tr02.load_config = load_config
    tr02.preflight = preflight


# ------------------------------------------------------------------------------------- CLI


def _check_data(export_dir: Path, cfg: dict[str, Any], tokenizer_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer  # only the tokenizer; mlx-lm brings transformers

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    lengths = tr02.row_lengths(
        export_dir, REPO / cfg["inputs"]["per_example_path"], cfg["inputs"]["per_example_sha256"]
    )
    cap = int(cfg["hyperparameters"]["max_seq_length"])
    return {
        name: tr02.check_rows(
            tr02.read_jsonl(export_dir / name), list(expected), expected, tok, cap
        )
        for name, expected in lengths.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-config")
    sub.add_parser("plan")
    for name in ("pin", "verify-inputs", "schedule", "check-data", "check-iterator"):
        p = sub.add_parser(name)
        p.add_argument("--export", required=True, help="directory written by `tr04 export-mlx`")
        if name in ("check-data", "check-iterator"):
            p.add_argument("--tokenizer-dir", required=True)
    p = sub.add_parser("verify-model")
    p.add_argument("--model-dir", required=True)
    for name in ("run", "supervise", "preflight"):
        p = sub.add_parser(name)
        for a in ("phase", "model-dir", "export", "out", "expect-commit"):
            p.add_argument(f"--{a}", required=True)
        p.add_argument("--after", help="the train phase's output directory (reload)")
        if name == "supervise":
            p.add_argument("--poll-seconds", type=float, default=5.0)
    sub.add_parser("check-lora")
    p = sub.add_parser("sanitize-evidence")
    p.add_argument("--src", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--home", default=str(Path.home()))
    p.add_argument("--term", action="append", default=[], help="NAME=PLACEHOLDER")
    p = sub.add_parser("collect-evidence")
    p.add_argument("--dest", required=True)
    p.add_argument("--run", action="append", required=True, metavar="NAME=OUT_DIR")
    a = parser.parse_args(argv)
    cfg = load_config()
    if a.cmd == "verify-config":
        problems = consistency_problems(cfg)
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "plan":
        print(json.dumps(budget(cfg), indent=1, sort_keys=True))
        return 0
    if a.cmd == "pin":
        pin_inputs(cfg, Path(a.export))
        cfg["budget"] = budget(cfg)
        train, reload_ = cfg["phases"]
        train["max_wall_s"] = cfg["budget"]["seconds"]["hard_wall_cap_train_phase"]
        reload_["max_wall_s"] = cfg["budget"]["seconds"]["hard_wall_cap_reload_phase"]
        write_config(cfg)
        print("pinned", len(cfg["inputs"]["export_files"]), "export files")
        return 0
    if a.cmd == "verify-inputs":
        problems = (
            tr02.verify_inputs(Path(a.export), cfg)
            + tr02.verify_corpus(cfg)
            + data_problems(cfg)
            + export_keep_problems(Path(a.export))
        )
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "schedule":
        out = tr02.schedule(Path(a.export), cfg, tr02.phase_config(cfg, TRAIN_PHASE))
        print(json.dumps({k: v for k, v in out.items() if k != "steps"}, indent=1))
        return 0
    if a.cmd == "check-data":
        report = _check_data(Path(a.export), cfg, Path(a.tokenizer_dir))
        print(
            json.dumps(
                {k: {**v, "problems": v["problems"][:5]} for k, v in report.items()}, indent=1
            )
        )
        return 1 if any(v["problems"] for v in report.values()) else 0
    if a.cmd == "check-iterator":
        result = tr02.check_iterator(
            Path(a.export), cfg, tr02.phase_config(cfg, TRAIN_PHASE), Path(a.tokenizer_dir)
        )
        print(json.dumps(result, indent=1))
        return 1 if result["problem_count"] else 0
    if a.cmd == "verify-model":
        problems = tr02.verify_model_dir(Path(a.model_dir), cfg) + tr02.verify_packages(
            cfg, tr02.installed_versions()
        )
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "check-lora":
        with tempfile.TemporaryDirectory() as tmp:
            result = tr02.lora_fixture_check(cfg["hyperparameters"], Path(tmp))
        print(json.dumps(result, indent=1))
        return 1 if result["problems"] else 0
    if a.cmd == "sanitize-evidence":
        dest = Path(a.dest)
        if not dest.resolve().is_relative_to(REPO.resolve()):
            raise SystemExit("the sanitized directory must be inside the repository")
        terms = dict(t.split("=", 1) for t in a.term)
        manifest = tr02.sanitize_evidence(Path(a.src), dest, a.home, terms)
        print(json.dumps({"files": len(manifest["files"])}, indent=1))
        return 0
    if a.cmd == "collect-evidence":
        dest = Path(a.dest)
        if not dest.resolve().is_relative_to(REPO.resolve()):
            raise SystemExit("the evidence directory must be inside the repository")
        runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.run]
        index = tr02.collect_evidence(runs, dest)
        print(json.dumps({"phases": list(index["phases"])}, indent=1))
        return 0
    after = Path(a.after) if a.after else None
    if a.cmd == "preflight":
        record, problems = preflight(
            cfg, a.phase, Path(a.export), Path(a.model_dir), Path(a.out), after, a.expect_commit
        )
        print(json.dumps({"record": record, "problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "supervise":
        if sys.platform != "darwin":
            raise SystemExit("the supervisor reads macOS probes: run it on the Apple Silicon Mac")
        out = Path(a.out)
        record, problems = preflight(
            cfg, a.phase, Path(a.export), Path(a.model_dir), out, after, a.expect_commit
        )
        if problems:
            print(json.dumps({"preflight_problems": problems}, indent=1))
            return 1
        need = tr02.phase_config(cfg, a.phase).get("requires")
        prior = tr02.require_complete(after, need)[0] if after and need else {}
        rules = build_rules(cfg, a.phase, Path(a.export), Path(a.model_dir), prior, a.expect_commit)
        child = [sys.executable, str(Path(__file__).resolve()), "run"]
        for key in ("phase", "model_dir", "export", "out", "expect_commit"):
            child += [f"--{key.replace('_', '-')}", getattr(a, key)]
        if after:
            child += ["--after", str(after)]
        result = tr02.supervise(
            child, out, rules, macos_sampler, poll_s=a.poll_seconds, preflight_record=record
        )
        print(json.dumps(result, indent=1))
        return 1 if result["stopped"] else 0
    if a.cmd == "run":
        _patch_tr02_for_child()
        if a.phase == RELOAD_PHASE:
            if after is None:
                raise SystemExit("reload needs --after <train output directory>")
            run_reload(Path(a.model_dir), Path(a.export), after, Path(a.out), a.expect_commit)
        elif a.phase == TRAIN_PHASE:
            tr02.run_phase(
                a.phase, Path(a.model_dir), Path(a.export), Path(a.out), after, a.expect_commit
            )
        else:
            raise SystemExit(f"unknown phase {a.phase!r}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
