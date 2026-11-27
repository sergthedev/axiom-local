"""EXP-TR02 feasibility-smoke harness (independent of the EXP-010 harness; nothing here trains).

Status: **prepared, not executed.** No model weights have been loaded by this code. The model path
(``run_phase``, ``run_reload``) is reviewed by reading only; everything else (verification,
schedule, event audit, stop rules, supervisor, evidence collection) is unit-tested without
weights in ``tests/test_tr02_harness.py``. The first execution needs the owner's separate
approval (see ``PROPOSED_EXP-TR02.md`` in the EXP-TR01 evidence folder).

Why a harness instead of ``python -m mlx_lm lora`` (all read from mlx-lm 0.32.2 source, see README):

* the stock trainer *truncates* sequences over ``max_seq_length`` with only a printed warning;
* its ``Tokens/sec`` counts loss tokens (the supervised completion), not the tokens it computes on;
* its last in-loop validation runs *before* the last optimizer step;
* its chat dataset supervises the template's empty thinking block and a trailing newline, which
  neither the EXP-010 rendering nor the inference prompt contain.

Phases are separate processes, each started by ``supervise`` and chained with ``--after``: probe,
then smoke (needs a COMPLETE probe), then reload (needs a COMPLETE smoke). A phase that stopped or
failed leaves ``STOP.json`` and no ``COMPLETE.json``, so the next phase refuses to start.

Commands (``uv run --locked --project envs/model python training/tr02/tr02.py <command> ...`` from
the repository root): ``pin``, ``verify-inputs``, ``schedule``, ``verify-model``, ``check-data``,
``check-iterator`` (no weights; real mlx-lm iterator and loss mask), ``preflight``, ``supervise``,
``run`` (the supervised child; not started by hand) and ``collect-evidence``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_PATH = HERE / "config.json"
EVENTS = "events.jsonl"  # one JSON object per line, written by the driver, read by the supervisor
COMPLETE = "COMPLETE.json"  # written by the supervisor only for a valid, finished phase
STOP = "STOP.json"
ADAPTER_FILE = "adapters.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def side_dir(out_dir: Path) -> Path:
    """The supervisor's own directory for a phase, beside (never inside) the driver's output."""
    return out_dir.with_name(out_dir.name + ".supervisor")


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


def verify_corpus(config: dict[str, Any], repo: Path = REPO) -> list[str]:
    """The committed corpus files and the per-example length file against the pinned hashes."""
    problems: list[str] = []
    corpus = repo / "data" / "axbench-tr" / config["inputs"]["corpus_version"]
    for name, digest in config["inputs"]["corpus_files_sha256"].items():
        path = corpus / name
        if not path.is_file():
            problems.append(f"corpus file missing: {name}")
        elif sha256_file(path) != digest:
            problems.append(f"corpus file hash differs: {name}")
    per_example = repo / config["inputs"]["per_example_path"]
    if not per_example.is_file():
        problems.append("per-example length file missing")
    elif sha256_file(per_example) != config["inputs"]["per_example_sha256"]:
        problems.append("per-example length file hash differs")
    return problems


def verify_model_dir(model_dir: Path, config: dict[str, Any]) -> list[str]:
    """Exact inventory of the checkpoint against the ten pinned files (size and SHA-256 of each).

    Missing, extra, resized or changed files are all problems. ``PROVENANCE.json`` (the fetch
    record) and a ``.cache/`` directory of the hub client are the only tolerated extras.
    """
    problems: list[str] = []
    pinned: Mapping[str, Mapping[str, Any]] = config["model"]["files"]
    if not model_dir.is_dir():
        return [f"model directory missing: {model_dir.name}"]
    present = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()}
    for name in sorted(present - set(pinned) - {"PROVENANCE.json"}):
        if not name.startswith(".cache/"):
            problems.append(f"extra file: {name}")
    for name, meta in pinned.items():
        path = model_dir / name
        if not path.is_file():
            problems.append(f"missing file: {name}")
        elif path.stat().st_size != int(meta["bytes"]):
            problems.append(f"size differs: {name}")
        elif sha256_file(path) != meta["sha256"]:
            problems.append(f"hash differs: {name}")
    prov_path = model_dir / "PROVENANCE.json"
    if not prov_path.is_file():
        problems.append("missing file: PROVENANCE.json")
    else:
        prov = json.loads(prov_path.read_text("utf-8"))
        if prov.get("revision") != config["model"]["revision"] or prov.get("tokenizer_only"):
            problems.append("revision differs or weights not provisioned")
        for name, meta in prov.get("files", {}).items():
            if name.endswith(".safetensors") and meta.get("verified_against") != "published hash":
                problems.append(f"weights not verified against a published hash: {name}")
    for name, digest in config["model"]["tokenizer_sha256"].items():
        if config["model"]["files"].get(name, {}).get("sha256") != digest:
            problems.append(f"tokenizer pin differs from the file pin: {name}")
    return problems


def installed_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": ".".join(map(str, sys.version_info[:2]))}
    for pkg in ("mlx", "mlx-lm"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def verify_packages(config: dict[str, Any], versions: Mapping[str, str]) -> list[str]:
    problems = [
        f"{pkg} is {versions.get(pkg)}, pinned {want}" for pkg, want in config["packages"].items()
        if versions.get(pkg) != want
    ]  # fmt: skip
    want_py = ".".join(map(str, config["python"]))
    if versions.get("python") != want_py:
        problems.append(f"python is {versions.get('python')}, pinned {want_py}")
    return problems


def verify_offline(env: Mapping[str, str]) -> list[str]:
    return [
        f"{name} must be 1 (offline mode)"
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        if env.get(name) != "1"
    ]


def git_state(repo: Path = REPO) -> tuple[str, list[str]]:
    """``(HEAD commit, dirty paths)``; untracked files count as dirty."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=60
        ).stdout

    head = run("rev-parse", "HEAD").strip()
    return head, run("status", "--porcelain", "--untracked-files=all").splitlines()


def verify_clean_tree(expect_commit: str, repo: Path = REPO) -> list[str]:
    commit, dirty = git_state(repo)
    problems = []
    if not re.fullmatch(r"[0-9a-f]{40}", expect_commit):
        problems.append("the owner-approved commit must be a full 40-character SHA")
    if commit != expect_commit:
        problems.append(f"HEAD is {commit[:12]}, the owner-approved commit is {expect_commit[:12]}")
    if dirty:
        problems.append(f"working tree is not clean ({len(dirty)} paths)")
    return problems


def _hash_tree(root: Path, skip: Collection[str] = ()) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() not in skip
    }


def require_complete(prior_out: Path, required_phase: str) -> tuple[dict[str, Any], list[str]]:
    """The previous phase must have a ``COMPLETE.json`` whose recorded file hashes still hold."""
    side = side_dir(prior_out)
    path = side / COMPLETE
    if not path.is_file():
        return {}, [f"{required_phase}: no {COMPLETE} (the phase did not finish cleanly)"]
    rec = json.loads(path.read_text("utf-8"))
    problems: list[str] = []
    if rec.get("phase") != required_phase:
        problems.append(f"previous phase is {rec.get('phase')}, need {required_phase}")
    now = {f"out/{k}": v for k, v in _hash_tree(prior_out).items()}
    now |= {f"supervisor/{k}": v for k, v in _hash_tree(side, {COMPLETE}).items()}
    if now != rec.get("files"):
        problems.append(f"{required_phase}: files changed since {COMPLETE} was written")
    return rec, problems


def phase_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    return next(p for p in config["phases"] if p["name"] == name)


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
    """Every check that must hold at execution time; run by the supervisor and again by the child.

    Returns ``(record, problems)``. Nothing here loads weights.
    """
    env = os.environ if env is None else env
    versions = installed_versions() if versions is None else versions
    phase = phase_config(config, phase_name)
    problems = verify_inputs(export_dir, config)
    problems += verify_corpus(config, repo)
    problems += verify_model_dir(model_dir, config)
    problems += verify_packages(config, versions)
    problems += verify_offline(env)
    problems += verify_clean_tree(expect_commit, repo)
    for d in (out, side_dir(out)):
        if d.resolve().is_relative_to(repo.resolve()):
            problems.append(f"output directory is inside the repository: {d.name}")
    prior: dict[str, Any] = {}
    need = phase.get("requires")
    if need:
        if after is None:
            problems.append(f"phase {phase_name} needs --after <{need} output directory>")
        else:
            prior, more = require_complete(after, need)
            problems += more
    record = {
        "commit": expect_commit,
        "phase": phase_name,
        "versions": dict(versions),
        "offline": True,
        "export_manifest_sha256": config["inputs"]["export_manifest_sha256"],
        "per_example_sha256": config["inputs"]["per_example_sha256"],
        "model_revision": config["model"]["revision"],
        "after_complete_sha256": prior and hashlib.sha256(
            json.dumps(prior, sort_keys=True).encode()
        ).hexdigest(),
    }  # fmt: skip
    return record, problems


# ----------------------------------------------------------------------------------- schedule


def planned_order(n_rows: int, seed: int) -> list[int]:
    """Row order of the stock trainer for ``batch_size=1`` after ``np.random.seed(seed)``.

    The stock ``iterate_batches`` sorts rows by ``len(dict)`` (equal for all rows, so the sort keeps
    the file order) and draws one ``np.random.permutation`` per epoch; one epoch is every row once.
    A test and ``check-iterator`` compare this with the real ``mlx_lm`` iterator.
    """
    import numpy as np

    np.random.seed(seed)
    return [int(i) for i in np.random.permutation(n_rows)]


def padded_tokens(n: int, max_seq_length: int, pad_to: int = 32) -> int:
    """Tokens the model computes on for one row: the stock iterator pads a batch to
    ``1 + 32*ceil(n/32)`` (capped at ``max_seq_length``) and the model sees all but the last."""
    return min(1 + pad_to * ((n + pad_to - 1) // pad_to), max_seq_length) - 1


def row_lengths(
    export_dir: Path, per_example: Path, expected_sha256: str
) -> dict[str, dict[str, int]]:
    """``{file: {row id: sequence tokens}}``; the file is hashed before it is read."""
    raw = per_example.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("the per-example length file differs from the pinned hash")
    by_id = {
        r["id"]: int(r["separate_tokens"])
        for r in (json.loads(line) for line in raw.decode("utf-8").splitlines())
    }
    ids = json.loads((export_dir / "ROW_IDS.json").read_text("utf-8"))
    return {name: {i: by_id[i] for i in rows} for name, rows in ids.items()}


def schedule(
    export_dir: Path, config: dict[str, Any], phase: dict[str, Any], repo: Path = REPO
) -> dict[str, Any]:
    """The exact rows of a phase in order, one per microstep, with full and padded token counts."""
    inputs, hp = config["inputs"], config["hyperparameters"]
    lengths = row_lengths(
        export_dir, repo / inputs["per_example_path"], inputs["per_example_sha256"]
    )[f"{phase['data']}/train.jsonl"]
    ids = list(lengths)
    order = planned_order(len(ids), int(hp["seed"]))
    microsteps = int(phase["microsteps"])
    if microsteps != len(ids):
        raise ValueError(f"{microsteps} microsteps do not expose {len(ids)} rows exactly once")
    accum = int(phase["grad_accumulation"])
    if microsteps % accum or microsteps // accum != int(phase["optimizer_updates"]):
        raise ValueError("microsteps, accumulation and optimizer updates disagree")
    cap = int(hp["max_seq_length"])
    steps = [
        {
            "microstep": k + 1,
            "id": ids[i],
            "tokens": lengths[ids[i]],
            "padded_tokens": padded_tokens(lengths[ids[i]], cap),
        }
        for k, i in enumerate(order)
    ]
    return {
        "phase": phase["name"],
        "seed": int(hp["seed"]),
        "rows": len(ids),
        "microsteps": microsteps,
        "optimizer_updates": microsteps // accum,
        "tokens_total": sum(s["tokens"] for s in steps),
        "padded_tokens_total": sum(s["padded_tokens"] for s in steps),
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


# ------------------------------------------------------------------------- LoRA shape checks


def expected_lora_shapes(
    model_config: Mapping[str, Any], hp: Mapping[str, Any]
) -> dict[str, tuple]:
    """``{tensor name: shape}`` the adapter must hold, from the checkpoint's own ``config.json``.

    ``lora_a`` is ``(in, rank)`` and ``lora_b`` is ``(rank, out)`` for q/v projections of the last
    ``lora_layers`` blocks. q out = heads * head_dim, v out = kv_heads * head_dim.
    """
    hidden = int(model_config["hidden_size"])
    heads = int(model_config["num_attention_heads"])
    kv = int(model_config["num_key_value_heads"])
    head_dim = int(model_config.get("head_dim", hidden // heads))
    n_layers = int(model_config["num_hidden_layers"])
    rank = int(hp["lora_rank"])
    outs = {"q_proj": heads * head_dim, "v_proj": kv * head_dim}
    shapes: dict[str, tuple] = {}
    for layer in range(n_layers - int(hp["lora_layers"]), n_layers):
        for key in hp["lora_keys"]:
            proj = key.split(".")[-1]
            base = f"layers.{layer}.{key}"
            shapes[f"{base}.lora_a"] = (hidden, rank)
            shapes[f"{base}.lora_b"] = (rank, outs[proj])
    return shapes


def lora_problems(shapes: Mapping[str, tuple], expected: Mapping[str, tuple]) -> list[str]:
    problems = [f"missing tensor: {n}" for n in expected if n not in shapes]
    problems += [f"unexpected tensor: {n}" for n in shapes if n not in expected]
    problems += [
        f"shape of {n} is {tuple(shapes[n])}, expected {tuple(want)}"
        for n, want in expected.items()
        if n in shapes and tuple(shapes[n]) != tuple(want)
    ]
    return problems


def trainable_params(shapes: Mapping[str, tuple]) -> int:
    return sum(math.prod(s) for s in shapes.values())


def adapter_summary(initial: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    """Finite, changed, ``lora_b`` no longer zero. Arrays are NumPy (float32) arrays."""
    import numpy as np

    finite = all(bool(np.isfinite(a).all()) for a in final.values())
    changed, max_change = 0, 0.0
    for name, after in final.items():
        delta = float(np.max(np.abs(after - initial[name]))) if after.size else 0.0
        if delta > 0:
            changed += 1
        max_change = max(max_change, delta)
    b_sum = sum(float(np.abs(a).sum()) for n, a in final.items() if n.endswith("lora_b"))
    return {
        "finite": finite,
        "changed_tensors": changed,
        "max_abs_change": max_change,
        "lora_b_abs_sum": b_sum,
    }


def make_exact_loss() -> Callable[..., Any]:
    """Cross-entropy over exactly the answer and ``<|im_end|>`` (``length - offset`` targets).

    The stock ``default_loss`` masks ``offset <= step <= length``, one step too many: the last
    target is the zero padding after ``<|im_end|>`` (found by ``check-iterator`` against mlx-lm
    0.32.2). This version stops at ``step < length``. Needs MLX; used for training, every
    validation and the reload, so all of them measure the same quantity.
    """
    import mlx.core as mx
    import mlx.nn as nn

    def tr02_loss(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
        targets = batch[:, 1:]
        logits = model(batch[:, :-1])
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / ntoks, ntoks

    return tr02_loss


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


def macos_sampler(pid: int | None) -> dict[str, Any]:
    """One OS sample on macOS. Failures raise: an unreadable probe stops the run, never ignored.

    ``pid=None`` is the pre-launch baseline (no process yet, so no RSS).
    """

    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout

    return {
        "kind": "os",
        "t": time.time(),
        "rss_gb": None if pid is None else parse_rss_kb(run("ps", "-o", "rss=", "-p", str(pid))),
        "swap_used_gb": parse_swapusage(run("sysctl", "-n", "vm.swapusage")),
        "pressure_level": parse_pressure_level(
            run("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
        ),
        "on_battery": parse_on_battery(run("pmset", "-g", "batt")),
    }


# ----------------------------------------------------------------------------------- stop rules


class Stop(Exception):
    """Raised inside the event audit; carries the stop reason."""


@dataclass(frozen=True)
class Limits:
    mlx_peak_gb: float = 10.0  # MLX allocator peak (mx.get_peak_memory), decimal GB
    rss_gb: float = 11.0  # process resident set, decimal GB; separate from the MLX peak
    swap_growth_gb: float = 2.0
    pressure_critical: int = 4
    window_ratio_floor: float = 0.5  # of the first window's full-token rate
    window_count: int = 3  # consecutive windows below the floor
    reports_per_window: int = 10  # one report per optimizer update: a window is 10 updates
    reload_abs_tolerance: float = 1e-4  # |reload loss - post-training loss|

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Limits:
        return cls(**cfg["limits"])


def train_sequence(microsteps: int, per_report: int) -> list[str]:
    return (
        ["start", "load", "lora", "val:before_training"]
        + ["train"] * (microsteps // per_report)
        + ["train_total", "adapter", "val:after_training", "done"]
    )


RELOAD_SEQUENCE = ["start", "load", "adapter_check", "val:reload", "reload", "done"]


@dataclass
class StopRules:
    """Audits every driver event and OS sample; returns the first stop reason. No retry, ever.

    ``kind`` is ``train`` (probe, smoke) or ``reload``. The exact event sequence is enforced, every
    metric must be finite and in range, and ``finished`` is true only after the final ``done``.
    """

    limits: Limits
    kind: str = "train"
    phase: str = ""
    expect_commit: str | None = None
    microsteps: int = 0
    microsteps_per_report: int = 4
    optimizer_updates: int = 0
    full_tokens: list[int] = field(default_factory=list)  # per microstep, in schedule order
    padded_tokens: list[int] = field(default_factory=list)
    valid_rows: int = 0
    lora_tensors: int = 0
    lora_params: int = 0
    reference_loss: float | None = None  # reload: the smoke's post-training validation loss
    reference_adapter_sha256: str | None = None
    reference_config_sha256: str | None = None
    swap_baseline: float | None = None
    window_rates: list[float] = field(default_factory=list)  # full tokens / compute second
    window_rates_padded: list[float] = field(default_factory=list)
    below: int = 0
    pos: int = 0
    finished: bool = False
    last_peak: float = 0.0
    last_step: int = 0
    adapter: dict[str, Any] = field(default_factory=dict)
    val_after: float | None = None
    reload_loss: float | None = None
    _window: list[tuple[int, int, float]] = field(default_factory=list)
    _expected: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind == "reload":
            self._expected = list(RELOAD_SEQUENCE)
        else:
            self._expected = train_sequence(self.microsteps, self.microsteps_per_report)
            if (
                len(self.full_tokens) != self.microsteps
                or len(self.padded_tokens) != self.microsteps
            ):
                raise ValueError("token schedule does not cover every microstep")

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _num(ev: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
        value = ev[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise Stop(f"malformed_{key}")
        if not math.isfinite(value):
            raise Stop("non_finite_loss" if "loss" in key else f"non_finite_{key}")
        if value < 0 or (positive and value == 0):
            raise Stop(f"out_of_range_{key}")
        return float(value)

    def _peak(self, ev: Mapping[str, Any]) -> None:
        peak = self._num(ev, "peak_memory_gb")
        if peak > self.limits.mlx_peak_gb:
            raise Stop(f"mlx_peak_above_limit:{peak:.2f}GB")
        if peak + 1e-9 < self.last_peak:
            raise Stop("mlx_peak_decreased")
        self.last_peak = peak

    # -- OS samples ----------------------------------------------------------------------------
    def prelaunch(self, sample: Mapping[str, Any]) -> str | None:
        """Baseline before the child exists: AC power and *normal* pressure; fixes the swap base."""
        self.swap_baseline = float(sample["swap_used_gb"])
        if sample["on_battery"]:
            return "battery_power_before_launch"
        if int(sample["pressure_level"]) != 1:
            return f"memory_pressure_not_normal_before_launch:{sample['pressure_level']}"
        return None

    def _os(self, ev: Mapping[str, Any]) -> str | None:
        if self.swap_baseline is None:
            self.swap_baseline = float(ev["swap_used_gb"])
        if ev["on_battery"]:
            return "battery_power"
        if ev["pressure_level"] >= self.limits.pressure_critical:
            return "memory_pressure_critical"
        rss = ev.get("rss_gb")
        if rss is not None and rss > self.limits.rss_gb:
            return f"rss_above_limit:{rss:.2f}GB"
        if ev["swap_used_gb"] - self.swap_baseline > self.limits.swap_growth_gb:
            return "swap_growth_above_limit"
        return None

    # -- driver events -------------------------------------------------------------------------
    def feed(self, ev: dict[str, Any]) -> str | None:
        try:
            return self._feed(ev)
        except Stop as stop:
            return str(stop)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            return f"malformed_event:{type(exc).__name__}:{str(exc)[:80]}"

    def _feed(self, ev: dict[str, Any]) -> str | None:
        kind = ev["kind"]
        if kind == "os":
            return self._os(ev)
        if kind == "error":
            return f"error:{str(ev.get('message', ''))[:120]}"
        label = f"val:{ev['tag']}" if kind == "val" else kind
        if self.finished:
            raise Stop(f"event_after_done:{label}")
        if self.pos >= len(self._expected) or label != self._expected[self.pos]:
            want = self._expected[self.pos] if self.pos < len(self._expected) else "nothing"
            raise Stop(f"unexpected_event:{label}_expected_{want}")
        getattr(self, "_ev_" + kind)(ev)
        self.pos += 1
        self.finished = self.pos == len(self._expected)
        return None

    def _ev_start(self, ev: Mapping[str, Any]) -> None:
        if ev["phase"] != self.phase:
            raise Stop("start_phase_mismatch")
        if self.expect_commit is not None and ev["commit"] != self.expect_commit:
            raise Stop("start_commit_mismatch")

    def _ev_load(self, ev: Mapping[str, Any]) -> None:
        self._num(ev, "seconds", positive=True)
        self._peak(ev)

    def _ev_lora(self, ev: Mapping[str, Any]) -> None:
        if ev["shapes_ok"] is not True:
            raise Stop("lora_shapes_not_ok")
        if ev["tensors"] != self.lora_tensors or ev["trainable_params"] != self.lora_params:
            raise Stop("lora_tensor_count_or_parameters_differ")

    def _ev_val(self, ev: Mapping[str, Any]) -> None:
        loss = self._num(ev, "val_loss")
        self._num(ev, "seconds", positive=True)
        self._peak(ev)
        if ev["rows"] != self.valid_rows:
            raise Stop("validation_did_not_cover_every_row")
        tag = ev["tag"]
        if tag == "before_training" and ev["updates_done"] != 0:
            raise Stop("val_before_training_updates_not_zero")
        if tag == "after_training":
            if ev["updates_done"] != self.optimizer_updates:
                raise Stop("val_after_training_updates_differ")
            self.val_after = loss
        if tag == "reload":
            self.reload_loss = loss

    def _ev_train(self, ev: Mapping[str, Any]) -> None:
        k = self.pos - 4  # train events start after start, load, lora, val
        it = int(ev["iteration"])
        if it != (k + 1) * self.microsteps_per_report:
            raise Stop(f"iteration_index_wrong:{it}")
        self._num(ev, "train_loss")
        ips = self._num(ev, "iterations_per_second", positive=True)
        self._num(ev, "loss_tokens_per_second")
        self._peak(ev)
        step = int(ev["optimizer_step"])
        if step != self.last_step + 1:
            raise Stop(f"optimizer_step_not_incrementing:{step}")
        self.last_step = step
        first = it - self.microsteps_per_report
        seconds = self.microsteps_per_report / ips
        self._window.append(
            (
                sum(self.full_tokens[first:it]),
                sum(self.padded_tokens[first:it]),
                seconds,
            )
        )
        if len(self._window) < self.limits.reports_per_window:
            return
        secs = sum(s for _, _, s in self._window)
        rate = sum(t for t, _, _ in self._window) / secs  # full tokens: the stop-rule rate
        self.window_rates_padded.append(sum(p for _, p, _ in self._window) / secs)
        self._window = []
        self.window_rates.append(rate)
        if len(self.window_rates) == 1:
            return
        self.below = (
            self.below + 1 if rate < self.limits.window_ratio_floor * self.window_rates[0] else 0
        )
        if self.below >= self.limits.window_count:
            raise Stop("throughput_below_half_of_first_window_for_3_windows")

    def _ev_train_total(self, ev: Mapping[str, Any]) -> None:
        self._num(ev, "seconds", positive=True)
        if ev["optimizer_steps"] != self.optimizer_updates:
            raise Stop(f"optimizer_steps_{ev['optimizer_steps']}_not_{self.optimizer_updates}")
        if ev["rows_exposed"] != self.microsteps or ev["schedule_complete"] is not True:
            raise Stop("rows_exposed_differ_from_schedule")

    def _adapter_fields(self, ev: Mapping[str, Any]) -> None:
        for key in ("adapter_sha256", "config_sha256"):
            if not isinstance(ev[key], str) or not SHA256_RE.match(ev[key]):
                raise Stop(f"malformed_{key}")
        if ev["tensors"] != self.lora_tensors or ev["shapes_ok"] is not True:
            raise Stop("adapter_tensors_or_shapes_differ")
        if ev["finite"] is not True:
            raise Stop("adapter_not_finite")
        self.adapter = {
            "adapter_sha256": ev["adapter_sha256"],
            "config_sha256": ev["config_sha256"],
        }

    def _ev_adapter(self, ev: Mapping[str, Any]) -> None:
        self._adapter_fields(ev)
        if int(ev["changed_tensors"]) < 1 or self._num(ev, "max_abs_change") <= 0:
            raise Stop("adapter_did_not_change")
        if self._num(ev, "lora_b_abs_sum") <= 0:
            raise Stop("lora_b_still_zero")

    def _ev_adapter_check(self, ev: Mapping[str, Any]) -> None:
        self._adapter_fields(ev)
        if ev["adapter_sha256"] != self.reference_adapter_sha256:
            raise Stop("reload_adapter_hash_differs")
        if ev["config_sha256"] != self.reference_config_sha256:
            raise Stop("reload_adapter_config_hash_differs")

    def _ev_reload(self, ev: Mapping[str, Any]) -> None:
        ref = self._num(ev, "reference_loss")
        got = self._num(ev, "reload_loss")
        if self.reference_loss is None or not math.isclose(ref, self.reference_loss, abs_tol=1e-12):
            raise Stop("reload_reference_loss_differs")
        if self.reload_loss is None or not math.isclose(got, self.reload_loss, abs_tol=1e-12):
            raise Stop("reload_loss_differs_from_validation_event")
        if abs(got - ref) > self.limits.reload_abs_tolerance:
            raise Stop(f"reload_loss_mismatch:{abs(got - ref):.3g}")

    def _ev_done(self, ev: Mapping[str, Any]) -> None:
        self._peak(ev)

    # -- end of run ----------------------------------------------------------------------------
    def finish(self, out_dir: Path) -> str | None:
        """After the child exited: the sequence is complete, the saved files are the logged ones."""
        if not self.finished:
            return f"incomplete_sequence:{self.pos}_of_{len(self._expected)}"
        if self.kind == "train":
            for name, key in ((ADAPTER_FILE, "adapter_sha256"), (ADAPTER_CONFIG, "config_sha256")):
                path = out_dir / name
                if not path.is_file() or sha256_file(path) != self.adapter.get(key):
                    return f"saved_{name}_differs_from_the_logged_hash"
        return None


# ----------------------------------------------------------------------------------- supervisor


def _write_json(path: Path, obj: Any) -> None:
    with path.open("x", encoding="utf-8") as fh:  # never overwrite an earlier record
        fh.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def supervise(
    cmd: list[str],
    out_dir: Path,
    rules: StopRules,
    sampler: Callable[[int | None], dict[str, Any]],
    poll_s: float = 5.0,
    kill_after_s: float = 30.0,
    preflight_record: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the driver as a child, apply the stop rules to its events and to OS samples.

    * Both ``out_dir`` and its supervisor directory must not exist: no reuse, no overwrite.
    * The baseline (AC power, normal pressure, swap) is taken *before* the child is started.
    * The child is terminated on any stop, malformed event, failed probe, error or interrupt.
    * A ``ps`` failure after the child exited is a normal exit, not a probe failure.
    * ``STOP.json`` is written on any stop; ``COMPLETE.json`` only for a finished, valid phase.
    """
    side = side_dir(out_dir)
    if out_dir.exists() or side.exists():
        raise FileExistsError("the output and supervisor directories must be new")
    side.mkdir(parents=True)
    events = out_dir / EVENTS
    reason: str | None = None
    proc: subprocess.Popen[bytes] | None = None
    code: int | None = None
    try:
        with (side / "supervisor.jsonl").open("x", encoding="utf-8") as sup:
            try:
                baseline = sampler(None)
            except Exception as exc:  # noqa: BLE001 - no baseline, no launch
                baseline, reason = None, f"baseline_probe_failed:{type(exc).__name__}"
            if baseline is not None:
                sup.write(json.dumps({**baseline, "baseline": True}, sort_keys=True) + "\n")
                reason = rules.prelaunch(baseline)
            if reason is None:
                with (side / "child.log").open("xb") as child_log:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=child_log,
                        stderr=subprocess.STDOUT,
                        env=None if env is None else dict(env),
                    )
                    offset = 0

                    def drain() -> str | None:
                        nonlocal offset
                        if not events.exists():
                            return None
                        with events.open("rb") as fh:
                            fh.seek(offset)
                            chunk = fh.read()
                        complete = chunk[: chunk.rfind(b"\n") + 1]  # never parse a half line
                        offset += len(complete)
                        for line in complete.decode("utf-8", "replace").splitlines():
                            try:
                                ev = json.loads(line)
                            except ValueError:
                                return "malformed_event:not_json"
                            if not isinstance(ev, dict):
                                return "malformed_event:not_an_object"
                            sup.write(json.dumps(ev, sort_keys=True) + "\n")
                            if stop := rules.feed(ev):
                                return stop
                        return None

                    try:
                        while reason is None:
                            try:
                                sample = sampler(proc.pid)
                            except Exception as exc:  # noqa: BLE001
                                sample = None
                                if proc.poll() is None:  # alive and unreadable: a real failure
                                    reason = f"probe_failed:{type(exc).__name__}"
                            if sample is not None:
                                sup.write(json.dumps(sample, sort_keys=True) + "\n")
                                reason = rules.feed(sample)
                            reason = reason or drain()
                            if reason is None and proc.poll() is not None:
                                reason = drain()
                                break
                            if reason is None:
                                time.sleep(poll_s)
                    except KeyboardInterrupt:
                        reason = "interrupted"
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(kill_after_s)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                        code = proc.returncode
    finally:
        if proc is not None and proc.poll() is None:  # belt and braces: never leave a child
            proc.kill()
    if reason is None and code != 0:
        reason = f"child_exit_code_{code}"
    if reason is None:
        reason = rules.finish(out_dir)
    result: dict[str, Any] = {
        "stopped": reason is not None,
        "reason": reason,
        "child_exit_code": code,
        "events_accepted": rules.pos,
    }
    if reason is not None:
        _write_json(side / STOP, result)
        return result
    files = {f"out/{k}": v for k, v in _hash_tree(out_dir).items()}
    files |= {f"supervisor/{k}": v for k, v in _hash_tree(side, {COMPLETE}).items()}
    complete = {
        "schema": "axbench-tr.exp-tr02-complete/1",
        "phase": rules.phase,
        "files": files,
        "window_rates_full_tokens_per_s": rules.window_rates,
        "window_rates_padded_tokens_per_s": rules.window_rates_padded,
        "reference_val_loss": rules.val_after,
        "adapter_sha256": rules.adapter.get("adapter_sha256"),
        "adapter_config_sha256": rules.adapter.get("config_sha256"),
        "preflight": preflight_record,
    }
    _write_json(side / COMPLETE, complete)
    return result | {"complete": True}


def collect_evidence(runs: list[tuple[str, Path]], dest: Path) -> dict[str, Any]:
    """Copy the raw records of finished or stopped phases into ``dest`` (a repository directory).

    Per phase: ``events.jsonl``, supervisor log, child log and STOP/COMPLETE. Adapters are hashed
    but not copied. ``EVIDENCE.json`` lists the SHA-256 of every copied file and the adapters.
    """
    dest.mkdir(parents=True, exist_ok=False)
    index: dict[str, Any] = {"schema": "axbench-tr.exp-tr02-evidence/1", "phases": {}}
    for name, out in runs:
        side = side_dir(out)
        got: dict[str, Any] = {"files": {}, "not_copied": {}}
        target = dest / name
        target.mkdir()
        for src in (
            out / EVENTS,
            out / ADAPTER_CONFIG,
            side / "supervisor.jsonl",
            side / "child.log",
            side / STOP,
            side / COMPLETE,
        ):
            if src.is_file():
                shutil.copy2(src, target / src.name)
                got["files"][src.name] = sha256_file(target / src.name)
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name not in got["files"]:
                got["not_copied"][path.relative_to(out).as_posix()] = {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
        index["phases"][name] = got
    _write_json(dest / "EVIDENCE.json", index)
    return index


# -------------------------------------------------------------------------------- driver (model)


def _np_params(mx: Any, tree_flatten: Any, params: Any) -> dict[str, Any]:
    import numpy as np

    flat = dict(tree_flatten(params))
    mx.eval(list(flat.values()))
    return {k: np.array(v.astype(mx.float32)) for k, v in flat.items()}


class _Events:
    """Append-only event log for the supervisor; flushed per line."""

    def __init__(self, out: Path) -> None:
        self.fh = (out / EVENTS).open("x", encoding="utf-8")

    def __call__(self, ev: dict[str, Any]) -> None:
        self.fh.write(json.dumps(ev, sort_keys=True) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def _start(
    cfg: dict[str, Any],
    phase_name: str,
    export_dir: Path,
    model_dir: Path,
    out: Path,
    after: Path | None,
    expect_commit: str,
) -> tuple[_Events, dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=False)  # the supervisor never pre-creates the driver's output
    emit = _Events(out)
    record, problems = preflight(cfg, phase_name, export_dir, model_dir, out, after, expect_commit)
    if problems:
        emit({"kind": "error", "message": "preflight failed: " + "; ".join(problems[:6])})
        raise SystemExit("preflight failed: " + "; ".join(problems[:6]))
    emit({"kind": "start", "phase": phase_name, "commit": expect_commit, "preflight": record})
    return emit, record


def run_phase(
    phase_name: str,
    model_dir: Path,
    export_dir: Path,
    out: Path,
    after: Path | None,
    expect_commit: str,
) -> None:
    """Train one phase in this process. NOT EXECUTED during preparation: no model has been run.

    A phase is one fresh process with its own adapter directory (the MLX peak counter and the
    optimizer are never shared between phases). Everything is re-verified first; the schedule is
    enforced while training; the trainer's own validation and checkpointing are not used.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import (
        TrainingArgs,
        TrainingCallback,
        evaluate,
        iterate_batches,
        train,
    )
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = load_config()
    phase = phase_config(cfg, phase_name)
    hp = cfg["hyperparameters"]
    emit, _ = _start(cfg, phase_name, export_dir, model_dir, out, after, expect_commit)
    try:
        loss_fn = make_exact_loss()
        mx.random.seed(int(hp["seed"]))
        tic = time.perf_counter()
        model, tokenizer = load(str(model_dir))
        emit(
            {
                "kind": "load",
                "seconds": time.perf_counter() - tic,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
        )
        rows = read_jsonl(export_dir / phase["data"] / "train.jsonl")
        valid_rows = read_jsonl(export_dir / phase["data"] / "valid.jsonl")
        cap = int(hp["max_seq_length"])
        train_ds = CacheDataset(TR02ChatDataset(rows, tokenizer, cap))
        valid_ds = CacheDataset(TR02ChatDataset(valid_rows, tokenizer, cap))
        sched = schedule(export_dir, cfg, phase)
        ids = [s["id"] for s in sched["steps"]]
        file_ids = list(
            row_lengths(
                export_dir,
                REPO / cfg["inputs"]["per_example_path"],
                cfg["inputs"]["per_example_sha256"],
            )[f"{phase['data']}/train.jsonl"]
        )
        assert sorted(ids) == sorted(file_ids)
        hash_to_id = {tokens_hash(train_ds[i][0]): rid for i, rid in enumerate(file_ids)}
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
        initial = _np_params(mx, tree_flatten, model.trainable_parameters())
        want = expected_lora_shapes(json.loads((model_dir / "config.json").read_text("utf-8")), hp)
        bad = lora_problems({k: v.shape for k, v in initial.items()}, want)
        emit(
            {
                "kind": "lora",
                "tensors": len(initial),
                "trainable_params": trainable_params({k: v.shape for k, v in initial.items()}),
                "shapes_ok": not bad,
                "problems": bad[:5],
            }
        )
        if bad:
            raise RuntimeError(
                "LoRA tensors differ from the expected shapes: " + "; ".join(bad[:3])
            )
        (out / ADAPTER_CONFIG).write_text(
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
                max_seq_length=cap,
                loss=loss_fn,
            )
            emit(
                {
                    "kind": "val",
                    "tag": tag,
                    "updates_done": updates_done,
                    "rows": len(valid_rows),
                    "val_loss": float(loss),
                    "seconds": time.perf_counter() - tic,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                    "diagnostic_only": phase.get("loss_is_diagnostic", False),
                }
            )
            return float(loss)

        optimizer = optim.Adam(learning_rate=hp["learning_rate"])

        class Reporter(TrainingCallback):
            def on_train_loss_report(self, info: dict[str, Any]) -> None:
                emit(
                    {
                        "kind": "train",
                        "iteration": int(info["iteration"]),
                        "optimizer_step": int(optimizer.step.item()),
                        "train_loss": float(info["train_loss"]),
                        "iterations_per_second": float(info["iterations_per_second"]),
                        "loss_tokens_per_second": float(info["tokens_per_second"]),
                        "peak_memory_gb": float(info["peak_memory"]),
                        "t": time.time(),
                    }
                )

        val_loss("before_training", 0)
        adapter_path = out / ADAPTER_FILE
        args = TrainingArgs(
            batch_size=1,
            iters=int(phase["microsteps"]),
            val_batches=-1,
            steps_per_report=int(phase["microsteps_per_report"]),
            steps_per_eval=10**9,  # never inside the loop
            steps_per_save=10**9,  # no intermediate checkpoints: no checkpoint selection
            max_seq_length=cap,
            adapter_file=str(adapter_path),
            grad_checkpoint=bool(hp["grad_checkpoint"]),
            grad_accumulation_steps=int(phase["grad_accumulation"]),
        )
        np.random.seed(int(hp["seed"]))  # the stock iterator draws its order from the global state
        tic = time.perf_counter()
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=None,
            args=args,
            loss=loss_fn,
            iterate_batches=checked_batches,
            training_callback=Reporter(),
        )
        emit(
            {
                "kind": "train_total",
                "seconds": time.perf_counter() - tic,
                "optimizer_steps": int(optimizer.step.item()),
                "rows_exposed": len(checker.seen),
                "schedule_complete": checker.complete(),
            }
        )
        if not checker.complete():
            raise RuntimeError("the rows drawn differ from the declared schedule")
        if not adapter_path.is_file():
            raise RuntimeError("the trainer did not save the adapter")
        saved = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(adapter_path)).items()}
        live = _np_params(mx, tree_flatten, model.trainable_parameters())
        bad = lora_problems({k: v.shape for k, v in saved.items()}, want)
        if (
            bad
            or set(saved) != set(live)
            or any(not np.array_equal(saved[k], live[k]) for k in saved)
        ):
            raise RuntimeError("the saved adapter is not the trained adapter")
        emit(
            {
                "kind": "adapter",
                "adapter_sha256": sha256_file(adapter_path),
                "config_sha256": sha256_file(out / ADAPTER_CONFIG),
                "bytes": adapter_path.stat().st_size,
                "tensors": len(saved),
                "shapes_ok": True,
                **adapter_summary(initial, saved),
            }
        )
        val_loss("after_training", int(phase["optimizer_updates"]))
        emit({"kind": "done", "peak_memory_gb": mx.get_peak_memory() / 1e9})
    except BaseException as exc:
        emit({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        emit.close()


def run_reload(
    model_dir: Path, export_dir: Path, after: Path, out: Path, expect_commit: str
) -> None:
    """Load the saved smoke adapter in a fresh process and re-evaluate the 40 validation rows.

    NOT EXECUTED during preparation. Supervised like every phase. The adapter and its config must
    hash to the values the smoke's ``COMPLETE.json`` recorded; the loss must equal the smoke's
    post-training loss within ``limits.reload_abs_tolerance`` or the run exits non-zero. Nothing is
    generated, scored or selected.
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import evaluate

    cfg = load_config()
    hp = cfg["hyperparameters"]
    tol = Limits.from_config(cfg).reload_abs_tolerance
    loss_fn = make_exact_loss()
    emit, _ = _start(cfg, "reload", export_dir, model_dir, out, after, expect_commit)
    try:
        prior, problems = require_complete(after, "smoke")
        if problems:
            raise RuntimeError("; ".join(problems))
        adapter_path, config_path = after / ADAPTER_FILE, after / ADAPTER_CONFIG
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
        want = expected_lora_shapes(json.loads((model_dir / "config.json").read_text("utf-8")), hp)
        bad = lora_problems({k: v.shape for k, v in saved.items()}, want)
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
        valid_rows = read_jsonl(export_dir / "smoke" / "valid.jsonl")
        valid = CacheDataset(TR02ChatDataset(valid_rows, tokenizer, int(hp["max_seq_length"])))
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
                "updates_done": int(phase_config(cfg, "smoke")["optimizer_updates"]),
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


# ------------------------------------------------------------------------- no-weight iterator check


def check_iterator(
    export_dir: Path, config: dict[str, Any], phase: dict[str, Any], tokenizer_dir: Path,
    mask_rows: int = 8,
) -> dict[str, Any]:  # fmt: skip
    """Real mlx-lm iterator and loss mask on the exposed rows; loads no weights (tokenizer only).

    Compares the order, lengths and padded widths the stock ``iterate_batches`` produces with the
    declared schedule, and the loss-mask token count of ``make_exact_loss`` (run on a stub model
    returning zero logits) with the supervised tokens of the harness dataset. The stock
    ``default_loss`` count is reported beside it: it is one token higher per row (the padding).
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import default_loss, iterate_batches
    from transformers import AutoTokenizer

    hp = config["hyperparameters"]
    cap = int(hp["max_seq_length"])
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    rows = read_jsonl(export_dir / phase["data"] / "train.jsonl")
    ds = CacheDataset(TR02ChatDataset(rows, tok, cap))
    sched = schedule(export_dir, config, phase)
    file_ids = list(
        row_lengths(
            export_dir,
            REPO / config["inputs"]["per_example_path"],
            config["inputs"]["per_example_sha256"],
        )[f"{phase['data']}/train.jsonl"]
    )
    hash_to_id = {tokens_hash(ds[i][0]): rid for i, rid in enumerate(file_ids)}
    offsets = {rid: ds[i][1] for i, rid in enumerate(file_ids)}
    exact = make_exact_loss()
    np.random.seed(int(hp["seed"]))
    stream = iterate_batches(dataset=ds, batch_size=1, max_seq_length=cap, loop=True)
    problems: list[str] = []
    masks: list[dict[str, int]] = []
    for k, step in enumerate(sched["steps"]):
        batch, lengths = next(stream)
        n = int(lengths[0, 1])
        got = hash_to_id.get(tokens_hash(batch[0, :n].tolist()))
        if got != step["id"] or n != step["tokens"]:
            problems.append(f"microstep {k + 1}: iterator drew {got} ({n}), schedule {step['id']}")
        if batch.shape != (1, step["padded_tokens"] + 1):
            problems.append(
                f"microstep {k + 1}: padded width {batch.shape[1]}, "
                f"expected {step['padded_tokens'] + 1}"
            )
        if k < mask_rows:
            vocab = int(batch.max().item()) + 1

            def stub(x: Any, vocab: int = vocab) -> Any:
                return mx.zeros((*x.shape, vocab))

            _, ntoks = exact(stub, batch, lengths)
            _, stock_ntoks = default_loss(stub, batch, lengths)
            want = n - offsets[step["id"]]
            masks.append(
                {
                    "microstep": k + 1,
                    "mask_tokens": int(ntoks.item()),
                    "supervised_tokens": want,
                    "stock_mask_tokens": int(stock_ntoks.item()),
                }
            )
            if int(ntoks.item()) != want:
                problems.append(
                    f"microstep {k + 1}: loss mask {int(ntoks.item())} tokens, supervised {want}"
                )
    return {
        "phase": phase["name"],
        "rows": len(rows),
        "problems": problems[:10],
        "problem_count": len(problems),
        "mask_samples": masks,
    }


# ------------------------------------------------------------------------------------------ CLI


def _export_pins(export_dir: Path) -> dict[str, Any]:
    manifest = json.loads((export_dir / "EXPORT_MANIFEST.json").read_text("utf-8"))
    return {
        "export_files": {name: meta["sha256"] for name, meta in manifest["files"].items()},
        "export_manifest_sha256": sha256_file(export_dir / "EXPORT_MANIFEST.json"),
    }


def build_rules(
    cfg: dict[str, Any],
    phase_name: str,
    export_dir: Path,
    model_dir: Path,
    prior: dict[str, Any],
    expect_commit: str,
) -> StopRules:
    phase = phase_config(cfg, phase_name)
    limits = Limits.from_config(cfg)
    hp = cfg["hyperparameters"]
    shapes = expected_lora_shapes(json.loads((model_dir / "config.json").read_text("utf-8")), hp)
    common = {
        "limits": limits,
        "phase": phase_name,
        "expect_commit": expect_commit,
        "lora_tensors": len(shapes),
        "lora_params": trainable_params(shapes),
    }
    if phase.get("kind") == "reload":
        smoke = phase_config(cfg, "smoke")
        return StopRules(
            **common,
            kind="reload",
            optimizer_updates=int(smoke["optimizer_updates"]),
            valid_rows=len(read_jsonl(export_dir / "smoke" / "valid.jsonl")),
            reference_loss=float(prior["reference_val_loss"]),
            reference_adapter_sha256=prior["adapter_sha256"],
            reference_config_sha256=prior["adapter_config_sha256"],
        )
    plan = schedule(export_dir, cfg, phase)
    return StopRules(
        **common,
        kind="train",
        microsteps=int(phase["microsteps"]),
        microsteps_per_report=int(phase["microsteps_per_report"]),
        optimizer_updates=int(phase["optimizer_updates"]),
        full_tokens=[s["tokens"] for s in plan["steps"]],
        padded_tokens=[s["padded_tokens"] for s in plan["steps"]],
        valid_rows=len(read_jsonl(export_dir / phase["data"] / "valid.jsonl")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pin", "verify-inputs", "schedule", "check-data", "check-iterator"):
        p = sub.add_parser(name)
        p.add_argument("--export", required=True, help="directory written by `export-mlx`")
        if name in ("schedule", "check-iterator"):
            p.add_argument("--phase", required=True)
        if name in ("check-data", "check-iterator"):
            p.add_argument("--tokenizer-dir", required=True)
    p = sub.add_parser("verify-model")
    p.add_argument("--model-dir", required=True)
    for name in ("run", "supervise", "preflight"):
        p = sub.add_parser(name)
        for a in ("phase", "model-dir", "export", "out", "expect-commit"):
            p.add_argument(f"--{a}", required=True)
        p.add_argument("--after", help="the previous phase's output directory (smoke, reload)")
        if name == "supervise":
            p.add_argument("--poll-seconds", type=float, default=5.0)
    p = sub.add_parser("collect-evidence")
    p.add_argument("--dest", required=True, help="new directory inside the repository")
    p.add_argument("--run", action="append", required=True, metavar="NAME=OUT_DIR")
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
        problems = verify_inputs(Path(a.export), cfg) + verify_corpus(cfg)
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "schedule":
        out = schedule(Path(a.export), cfg, phase_config(cfg, a.phase))
        print(json.dumps({k: v for k, v in out.items() if k != "steps"}, indent=1))
        return 0
    if a.cmd == "check-data":
        from transformers import AutoTokenizer  # only the tokenizer; mlx-lm brings transformers

        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
        export_dir = Path(a.export)
        lengths = row_lengths(
            export_dir,
            REPO / cfg["inputs"]["per_example_path"],
            cfg["inputs"]["per_example_sha256"],
        )
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
    if a.cmd == "check-iterator":
        result = check_iterator(
            Path(a.export), cfg, phase_config(cfg, a.phase), Path(a.tokenizer_dir)
        )
        print(json.dumps(result, indent=1))
        return 1 if result["problem_count"] else 0
    if a.cmd == "verify-model":
        problems = verify_model_dir(Path(a.model_dir), cfg) + verify_packages(
            cfg, installed_versions()
        )
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "collect-evidence":
        runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.run]
        dest = Path(a.dest)
        if not dest.resolve().is_relative_to(REPO.resolve()):
            raise SystemExit("the evidence directory must be inside the repository")
        index = collect_evidence(runs, dest)
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
        prior = require_complete(after, phase_config(cfg, a.phase)["requires"])[0] if after else {}
        rules = build_rules(cfg, a.phase, Path(a.export), Path(a.model_dir), prior, a.expect_commit)
        child = [sys.executable, str(Path(__file__).resolve()), "run"]
        for key in ("phase", "model_dir", "export", "out", "expect_commit"):
            child += [f"--{key.replace('_', '-')}", getattr(a, key)]
        if after:
            child += ["--after", str(after)]
        result = supervise(
            child, out, rules, macos_sampler, poll_s=a.poll_seconds, preflight_record=record
        )
        print(json.dumps(result, indent=1))
        return 1 if result["stopped"] else 0
    if a.cmd == "run":
        if a.phase == "reload":
            if after is None:
                raise SystemExit("reload needs --after <smoke output directory>")
            run_reload(Path(a.model_dir), Path(a.export), after, Path(a.out), a.expect_commit)
        else:
            run_phase(
                a.phase, Path(a.model_dir), Path(a.export), Path(a.out), after, a.expect_commit
            )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
