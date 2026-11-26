"""EXP-010 harness: completion-only LoRA training of Qwen3-1.7B on AXBENCH train (MLX-LM 0.32.2).

Model environment only (ADR 0012). Nothing here runs unless the owner starts it after the EXP-010
pre-run record (``docs/evidence/experiments/EXP-010-lora-pilot/PRERUN.md``) has been accepted.
Run from ``envs/model``:

    uv run --locked python exp010_train.py tokens   --model-dir <dir> --data-dir <export dir>
    uv run --locked python exp010_train.py params   --model-dir <dir>
    uv run --locked python exp010_launch.py start   --phase smoke ...   (detached; see that file)

* ``tokens``: offline tokenizer check of every exported example (no weights): the export equals the
  inference rendering of the frozen corpus, the completion boundary (PRERUN D2) holds token by
  token, and every length; applies the sequence-length rule. Writes JSON, never texts.
* ``params``: loads the checkpoint lazily (no weights are read), applies the LoRA configuration
  with the seed and reports the trainable parameter count and the adapted modules. No training.
* ``train``: one training process (``--phase smoke`` or ``pilot``), normally started by the
  launcher. Preflight (pinned commit, clean tree, mains power, normal memory pressure), re-hash of
  the checkpoint and of the exported data, LoRA set-up, loss at update 0, the MLX-LM 0.32.2
  trainer with the completion-only loss, per-update records and stop rules, evaluation and adapter
  saves at update boundaries, adapter hashes.
* ``reload``: a fresh process that loads the base with a saved adapter (hash-checked) and runs real
  inference on the five warm-up train examples (the smoke check; never validation).
* ``evaluate``: the EXP-009 harness frame with the adapter on validation (five train warm-ups,
  then the 160 validation requests). It is run only after the pre-registered gates are accepted.

Never reads, exports or scores the final test split. Writes no absolute paths. Adapters go to a
directory outside the repository; records go to a new run directory under the EXP-010 evidence.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import exp009_run as base  # noqa: E402
from exp009_run import (  # noqa: E402
    CORPUS,
    EXPECTED_EOS,
    MANIFEST_SHA256,
    MAX_NEW_TOKENS,
    Sink,
    StopRun,
    git_state,
    rss_peak_bytes,
    sample,
    snapshot,
    verify_model_dir,
    write_json,
)

from axbench.corpus import load_corpus  # noqa: E402
from axbench.runwatch import (  # noqa: E402
    ProcessMonitor,
    SlowdownWatch,
    is_allocation_error,
    preflight,
    run_requests,
    warmup_ids,
)
from axbench.supervise import RULE_KEYS, validate_sample  # noqa: E402
from axbench.supervise import STOP as MONITOR_STOP_FILE  # noqa: E402
from axbench.trainwatch import (  # noqa: E402
    LEARNING_RATE,
    LORA_PARAMETERS,
    NUM_LAYERS,
    SEED,
    TRAIN_PLAN_VERSION,
    PlanError,
    TrainingExample,
    UpdateLedger,
    UpdateThroughputWatch,
    check_adapter_tensors,
    check_lengths,
    check_request_records,
    check_training_records,
    completion_example,
    length_summary,
    max_seq_length,
    mlx_default_mask_positions,
    mlx_padded_length,
    non_finite,
    schedule,
    target_positions,
)
from axiom_router.prompt import PROMPT_VERSION, build_router_input, chat_messages  # noqa: E402
from axiom_router.schema import project_columns  # noqa: E402

HARNESS_VERSION = "exp010.harness/3"  # D18 freeze: supervised evaluation (training path unchanged)
MODEL = "qwen3-1.7b"
SYSTEM = "qwen3-1.7b-lora-exp010"
EXPERIMENT_DIR = REPO / "docs" / "evidence" / "experiments" / "EXP-010-lora-pilot"
# Byte-exact `python -m axbench export-mlx --corpus data/axbench/v0.1-ai-reviewed` (PRERUN D1).
EXPORT_SHA256 = {
    "train.jsonl": "437c2ef0ce44bdedcb865f7f115ef2b7eda461063e127b5e4cb77da20ec99b31",
    "valid.jsonl": "40f83a3ca85a1802d6877c4d09d6d620dcb416b4ee9169b1417239d1e85d4361",
}
EXPORT_SPLITS = {"train.jsonl": "train", "valid.jsonl": "validation"}
# Measured before any training (PRERUN D3, D6); the harness refuses if they differ.
PINNED_MAX_SEQ_LENGTH = 1024
PINNED_TRAINABLE_PARAMETERS = 458_752
SMOKE_EVAL_IDS = 16  # smoke "loss check" set: the first 16 train examples (never validation)
DECLARED_CHANGES = ("C1",)  # PRERUN D9: gradient checkpointing, only after a memory stop
UPDATE_SECONDS_DEFINITION = (
    "MLX-LM 0.32.2 trainer compute time of the update's micro-steps (forward, backward, optimizer "
    "update, mx.eval, cache clearing), summed between its perf_counter marks; excludes batch "
    "construction, the progress display and everything in the callback (evaluation, adapter "
    "saves, resource sampling, records); the first window includes graph compilation"
)


# ---------------------------------------------------------------- data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expected_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """The export's rendering of one corpus record (same code path as ``axbench export-mlx``)."""
    view = project_columns([(c["name"], c["type"]) for c in record["schema"]["columns"]])
    msgs = chat_messages(build_router_input(record["request"], view))
    target = json.dumps(record["gold"], ensure_ascii=False, separators=(",", ":"))
    return [*msgs, {"role": "assistant", "content": target}]


def load_export(data_dir: Path) -> tuple[dict[str, list[tuple[str, list]]], dict[str, Any]]:
    """Exported train/valid chat files: pinned bytes, and equal to the frozen corpus rendering.

    Returns ``{split: [(example_id, messages), ...]}`` in file order and a provenance record.
    """
    manifest_sha = _sha256_bytes((CORPUS / "MANIFEST.json").read_bytes())
    if manifest_sha != MANIFEST_SHA256:
        raise PlanError("frozen corpus manifest does not match the pinned hash")
    manifest, corpus = load_corpus(CORPUS)
    if sorted(p.name for p in data_dir.iterdir() if p.suffix == ".jsonl") != sorted(EXPORT_SHA256):
        raise PlanError("export directory must hold exactly train.jsonl and valid.jsonl")
    out: dict[str, list[tuple[str, list]]] = {}
    prov: dict[str, Any] = {
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": manifest_sha,
        "files": {},
    }
    for name, split in EXPORT_SPLITS.items():
        raw = (data_dir / name).read_bytes()
        digest = _sha256_bytes(raw)
        if digest != EXPORT_SHA256[name]:
            raise PlanError(f"{name} differs from the pinned export hash")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        records = corpus[split]
        if len(rows) != len(records):
            raise PlanError(f"{name}: {len(rows)} lines, corpus {split} has {len(records)}")
        pairs = []
        for row, rec in zip(rows, records, strict=True):
            if row != {"messages": _expected_messages(rec)}:
                raise PlanError(f"{name}: line for {rec['id']} differs from the corpus rendering")
            pairs.append((rec["id"], row["messages"]))
        out[split] = pairs
        prov["files"][name] = {"sha256": digest, "bytes": len(raw), "examples": len(rows)}
    return out, prov


def render(tokenizer, example_id: str, messages: list[dict[str, str]]) -> TrainingExample:
    """Inference prompt (exactly as EXP-009 renders it) and the full rendering, as token ids."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(messages, enable_thinking=False)
    return completion_example(example_id, prompt, full)


def build_examples(tokenizer, export) -> dict[str, list[TrainingExample]]:
    return {s: [render(tokenizer, i, m) for i, m in pairs] for s, pairs in export.items()}


# ---------------------------------------------------------------- tokens (no weights)


def cmd_tokens(args) -> int:
    from mlx_lm.utils import load_tokenizer

    model = verify_model_dir(MODEL, Path(args.model_dir))
    tokenizer = load_tokenizer(Path(args.model_dir))
    export, prov = load_export(Path(args.data_dir))
    examples = build_examples(tokenizer, export)
    every = [e.length for split in examples.values() for e in split]
    limit = max_seq_length(every)
    check_lengths([e for split in examples.values() for e in split], limit)
    mask = mask_self_check([e for split in examples.values() for e in split], limit)
    report: dict[str, Any] = {
        "harness_version": HARNESS_VERSION,
        "train_plan_version": TRAIN_PLAN_VERSION,
        "tokenizer": {
            "class": type(tokenizer._tokenizer).__name__,
            "checkpoint_revision": model["revision"],
            "tokenizer_files_sha256": {
                k: v["sha256"]
                for k, v in model["files"].items()
                if k.startswith("tokenizer") or k == "vocab.json" or k == "merges.txt"
            },
        },
        "export": prov,
        "prompt_version": PROMPT_VERSION,
        "rendering": "prompt: chat template, enable_thinking=False, generation prompt; "
        "sequence: prompt + compact gold JSON + <|im_end|> (template newline dropped)",
        "max_seq_length_rule": "smallest power of two >= longest sequence (train and validation); "
        "stop above 1024; no truncation",
        "longest_sequence": max(every),
        "max_seq_length": limit,
        "loss_mask_check": mask,
        "splits": {},
    }
    rows = []
    for split, items in examples.items():
        default_extra = []
        for e, (_, messages) in zip(items, export[split], strict=True):
            # What `mlx_lm.lora --mask-prompt` on the chat export would train (PRERUN E4, E5):
            # its prompt boundary omits enable_thinking, it keeps the template newline, and its
            # loss mask includes one padding position.
            default_offset = len(
                tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
            )
            full_len = e.length + 1
            default = mlx_default_mask_positions(
                default_offset, full_len, mlx_padded_length(full_len, 2048)
            )
            default_extra.append(len(default) - len(target_positions(e.offset, e.length)))
            rows.append(
                {
                    "id": e.example_id,
                    "split": split,
                    "prompt_tokens": e.offset,
                    "completion_tokens": e.completion_tokens,
                    "sequence_tokens": e.length,
                }
            )
        report["splits"][split] = {
            "examples": len(items),
            "prompt_tokens": length_summary([e.offset for e in items]),
            "completion_tokens_incl_im_end": length_summary([e.completion_tokens for e in items]),
            "sequence_tokens": length_summary([e.length for e in items]),
            "boundary_checks_passed": len(items),
            "mlx_lm_chat_default_extra_targets_per_example": sorted(set(default_extra)),
        }
    if args.per_example:
        Path(args.per_example).write_bytes(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
        )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------- model set-up


def apply_lora(model) -> dict[str, Any]:
    """Freeze the base, seed, convert the configured layers; return the parameter accounting."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx.random.seed(SEED)
    model.freeze()
    if NUM_LAYERS > len(model.layers):
        raise PlanError("model has fewer layers than NUM_LAYERS")
    linear_to_lora_layers(model, NUM_LAYERS, dict(LORA_PARAMETERS))
    trainable = tree_flatten(model.trainable_parameters())
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    count = sum(v.size for _, v in trainable)
    first = len(model.layers) - NUM_LAYERS
    expected = {
        f"model.layers.{i}.self_attn.{proj}.{ab}"
        for i in range(first, len(model.layers))
        for proj in ("q_proj", "v_proj")
        for ab in ("lora_a", "lora_b")
    }
    names = {k for k, _ in trainable}
    if names != expected:
        raise PlanError(f"trainable parameters differ from the plan: {sorted(names ^ expected)}")
    return {
        "trainable_parameters": count,
        "total_parameters_including_lora": total,
        "trainable_fraction": count / total,
        "adapted_layers": list(range(first, len(model.layers))),
        "trainable_tensors": {k: list(v.shape) for k, v in sorted(trainable)},
        "lora_parameters": LORA_PARAMETERS,
        "num_layers": NUM_LAYERS,
        "seed": SEED,
    }


def cmd_params(args) -> int:
    from mlx_lm import load

    verify_model_dir(MODEL, Path(args.model_dir))
    model, _ = load(str(Path(args.model_dir)), lazy=True)
    info = apply_lora(model)
    info["note"] = "lazy load: shapes only, no weights evaluated, nothing trained"
    json.dump(info, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def completion_loss(model, batch, lengths):
    """MLX-LM 0.32.2 ``default_loss`` with the end bound made exclusive (PRERUN D2/E4).

    ``lengths[:, 0]`` is the prompt length, ``lengths[:, 1]`` the sequence length; the targets are
    positions ``offset .. length - 1`` (completion tokens and ``<|im_end|>``), no padding target.
    """
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks
    return ce, ntoks


class SeqDataset:
    """``(tokens, offset)`` items, as MLX-LM's ``iterate_batches`` expects."""

    def __init__(self, examples: list[TrainingExample]) -> None:
        self.examples = examples

    def __getitem__(self, idx: int):
        e = self.examples[idx]
        return list(e.tokens), e.offset

    def __len__(self) -> int:
        return len(self.examples)


def mask_self_check(examples: list[TrainingExample], limit: int) -> dict[str, Any]:
    """Run the harness loss mask on real batches and compare with the planned target positions."""
    import mlx.core as mx
    import numpy as np
    from mlx_lm.tuner.trainer import iterate_batches

    state = np.random.get_state()
    try:
        checked = 0
        for e in examples:
            batch, lengths = next(iterate_batches(SeqDataset([e]), 1, limit))
            steps = mx.arange(1, batch.shape[1])
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
            got = [int(i) + 1 for i in np.nonzero(np.array(mask[0]))[0]]
            if got != list(target_positions(e.offset, e.length)):
                raise PlanError(f"{e.example_id}: loss mask differs from the planned targets")
            if int(batch[0, e.length - 1]) != 151645:
                raise PlanError(f"{e.example_id}: last target is not <|im_end|>")
            checked += 1
    finally:
        np.random.set_state(state)
    return {"examples_checked": checked, "mask_equals_completion_positions": True}


# ---------------------------------------------------------------- training process


class ExternalStop(Exception):
    pass


def _on_sigterm(signum, frame):  # noqa: ARG001
    raise ExternalStop("SIGTERM received (external monitor or operator)")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return "<outside the repository>"


def _new_dir(path: Path, inside_repo: bool) -> Path:
    path = path.resolve()
    if (REPO in path.parents) != inside_repo:
        where = "inside" if inside_repo else "outside"
        raise SystemExit(f"{path.name}: must be {where} the repository")
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path.name} is not empty; a new run needs a new directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": _sha256_bytes(data)}


def _check_commit(args) -> dict[str, Any]:
    git = git_state()
    if git["dirty"]:
        raise SystemExit("refusing to run on a dirty tree (tracked files modified)")
    if git["commit"] != args.expected_commit:
        raise SystemExit("HEAD is not the reviewed commit given with --expected-commit")
    return git


def _is_ancestor(commit: str) -> bool:
    import subprocess

    proc = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO,
                          capture_output=True)  # fmt: skip
    return proc.returncode == 0


def _environment(git: dict[str, Any]) -> dict[str, Any]:
    env = base.environment(git)
    env["runner_version"] = HARNESS_VERSION
    env["train_plan_version"] = TRAIN_PLAN_VERSION
    return env


def _baseline(args, pre: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The stop-rule baseline: the sequence's common pre-load baseline when the launcher passes
    ``--baseline`` (correction A), otherwise this process's own pre-load sample."""
    if getattr(args, "baseline", None):
        b = validate_sample(json.loads(Path(args.baseline).read_text(encoding="utf-8")))
        return {k: b[k] for k in RULE_KEYS}, {"source": "sequence baseline.json", **b}
    return {k: pre[k] for k in RULE_KEYS}, {"source": "this process, before loading", **pre}


def _monitor_stop(args) -> dict[str, Any] | None:
    """The external monitor's stop record, if the sequence's monitor wrote one."""
    if not getattr(args, "baseline", None):
        return None
    path = Path(args.baseline).parent / MONITOR_STOP_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _finalize(meas: dict[str, Any], steps: list[tuple[str, Any]]) -> None:
    """Run each finalization step; an error is recorded and never replaces the original outcome.

    A completed run whose evidence cannot be finalized becomes ``failed``; a stopped or failed run
    keeps its status and reason (correction C)."""
    errors = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        meas["finalization_errors"] = errors
        if meas.get("status") == "completed":
            meas["status"] = "failed"
            meas["failure"] = "finalization: " + "; ".join(errors)


def tensor_stats(path: Path) -> dict[str, dict[str, Any]]:
    """Shape, finiteness and largest magnitude of every tensor in a saved adapter file."""
    import mlx.core as mx

    out = {}
    for name, value in mx.load(str(path)).items():
        v = value.astype(mx.float32)
        out[name] = {"shape": list(value.shape), "finite": bool(mx.all(mx.isfinite(v)).item()),
                     "max_abs": float(mx.abs(v).max().item())}  # fmt: skip
    return out


def verify_training_outputs(
    out: Path, adapter_dir: Path, sched, expected_shapes: dict[str, list[int]]
) -> dict[str, Any]:
    """Correction C: the persisted log and adapter files, checked against the plan.

    Every planned update, evaluation and save record exactly once, in order, with finite losses;
    exactly the required adapter files; every save record's hash equal to its file; the final
    adapter equal to the last update save; every saved tensor with the planned name and shape,
    finite, and every ``lora_b`` non-zero."""
    records = [json.loads(x) for x in (out / "training_log.jsonl").read_text().splitlines()]
    problems = check_training_records(records, sched)
    files = {p.name: _file_record(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
    required = {"adapter_config.json", "adapters.safetensors",
                *(f"update_{u:04d}.safetensors" for u in sched.save_updates)}  # fmt: skip
    if set(files) != required:
        problems.append(f"adapter files {sorted(files)} differ from {sorted(required)}")
    for r in (r for r in records if r.get("kind") == "save"):
        if files.get(r.get("file"), {}).get("sha256") != r.get("sha256"):
            problems.append(f"{r.get('file')}: file hash differs from its save record")
    last = f"update_{sched.updates:04d}.safetensors"
    final_equal = (
        "adapters.safetensors" in files
        and last in files
        and files["adapters.safetensors"]["sha256"] == files[last]["sha256"]
    )
    if not final_equal:
        problems.append("final adapter does not equal the last update save")
    lora_b = {}
    for name in sorted(n for n in files if n.endswith(".safetensors")):
        stats = tensor_stats(adapter_dir / name)
        problems += [f"{name}: {p}" for p in check_adapter_tensors(stats, expected_shapes)]
        b = [t["max_abs"] for k, t in stats.items() if k.endswith("lora_b")]
        lora_b[name] = {"min_tensor_max_abs": min(b, default=None),
                        "max_tensor_max_abs": max(b, default=None)}  # fmt: skip
    return {
        "passed": not problems,
        "problems": problems,
        "records": len(records),
        "files": files,
        "final_adapter_equals_last_update_save": final_equal,
        "lora_b_magnitudes": lora_b,
    }


def run_training(
    model,
    train_set,
    eval_set,
    eval_name,
    sched,
    limit,
    adapter_dir,
    log,
    monitor,
    throughput,
    ledger,
    meas,
    mpath,
    grad_checkpoint: bool,
    eval_delay_seconds: float = 0.0,
) -> None:
    """Loss at update 0, the installed MLX-LM trainer with the completion-only loss, per-update
    records and stop rules, evaluation and saves at update boundaries (PRERUN D4-D8).

    ``eval_delay_seconds`` is for the self-test only: extra callback time that must not appear in
    ``update_seconds`` (lead decision 6/7)."""
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train

    def run_eval(update: int) -> None:
        state = np.random.get_state()  # evaluation must not move the data-order stream
        t0 = time.perf_counter()
        try:
            loss = evaluate(model, eval_set, batch_size=1, num_batches=-1,
                            max_seq_length=limit, loss=completion_loss)  # fmt: skip
        finally:
            np.random.set_state(state)
            model.train()
        if eval_delay_seconds:
            time.sleep(eval_delay_seconds)
        rec = {"kind": "eval", "update": update, "set": eval_name, "loss": loss,
               "seconds": time.perf_counter() - t0, "mlx_peak_bytes": mx.get_peak_memory(),
               "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
        log.emit(rec)
        stop = non_finite(f"{eval_name} loss at update {update}", loss)
        stop = stop or monitor.check(f"eval@{update}", sample())
        if stop:
            raise StopRun(stop)

    def save(update: int) -> None:
        weights = dict(tree_flatten(model.trainable_parameters()))
        path = adapter_dir / f"update_{update:04d}.safetensors"
        mx.save_safetensors(str(path), weights)
        log.emit({"kind": "save", "update": update, "file": path.name, **_file_record(path)})

    last_callback = {"t": 0.0}

    class Callback(TrainingCallback):
        def on_train_loss_report(self, info: dict) -> None:
            now = time.perf_counter()
            interval, last_callback["t"] = now - last_callback["t"], now
            update = ledger.advance(int(info["iteration"]))
            seconds = sched.grad_accum / info["iterations_per_second"]
            s = sample()
            rec = {
                "kind": "update",
                "update": update,
                "micro_steps": int(info["iteration"]),
                "train_loss": info["train_loss"],
                "learning_rate": info["learning_rate"],
                "update_seconds": seconds,
                "callback_interval_seconds": interval,
                "tokens_per_second": info["tokens_per_second"],
                "trained_tokens": info["trained_tokens"],
                "mlx_peak_bytes": mx.get_peak_memory(),
                "mlx_active_bytes": mx.get_active_memory(),
                "rss_peak_bytes": rss_peak_bytes(),
                **s,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            log.emit(rec)
            stop = non_finite(f"train loss at update {update}", info["train_loss"])
            stop = stop or monitor.check(f"update@{update}", s)
            stop = stop or throughput.add(seconds)
            if stop:
                raise StopRun(stop)
            if update in sched.save_updates:
                save(update)
            if update in sched.eval_updates:
                run_eval(update)

    targs = TrainingArgs(
        batch_size=1,
        iters=sched.micro_steps,
        val_batches=-1,
        steps_per_report=sched.grad_accum,
        steps_per_eval=sched.micro_steps + 1,  # built-in evaluation unused (val_dataset None)
        steps_per_save=sched.micro_steps + 1,  # built-in periodic save never fires
        max_seq_length=limit,
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        grad_checkpoint=grad_checkpoint,
        grad_accumulation_steps=sched.grad_accum,
    )
    meas["mlx_lm_training_args"] = {k: (v if k != "adapter_file" else "adapters.safetensors")
                                    for k, v in vars(targs).items()}  # fmt: skip
    meas["update_seconds_definition"] = UPDATE_SECONDS_DEFINITION
    write_json(mpath, meas)
    run_eval(0)
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    t0 = time.perf_counter()
    last_callback["t"] = t0
    try:
        train(model=model, optimizer=optimizer, train_dataset=train_set, val_dataset=None,
              args=targs, loss=completion_loss, training_callback=Callback())  # fmt: skip
    except Exception as exc:
        if is_allocation_error(exc):
            raise StopRun(f"train: allocation error: {type(exc).__name__}: {exc}") from exc
        raise
    meas["train_loop_seconds"] = time.perf_counter() - t0
    meas["optimizer_steps"] = int(optimizer.step.item())
    if not ledger.complete():
        raise PlanError(f"{ledger.updates} updates recorded, {sched.updates} planned")
    if meas["optimizer_steps"] != sched.updates:
        raise PlanError(f"optimizer applied {meas['optimizer_steps']} updates, "
                        f"{sched.updates} planned")  # fmt: skip
    lora_b = [v for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]
    meas["lora_b_max_abs_after_training"] = max(float(mx.abs(v).max()) for v in lora_b)


def adapter_config(phase: str, sched, model: dict[str, Any], prov: dict[str, Any], git, limit,
                   lora: dict[str, Any], env: dict[str, Any], grad_checkpoint: bool,
                   declared_change: str | None) -> dict[str, Any]:  # fmt: skip
    """``adapter_config.json``: what MLX-LM's ``load_adapters`` reads, plus the provenance of the
    adapter (correction C): base and its file hashes, phase and schedule, configuration, data,
    commit and package versions."""
    return {
        "fine_tune_type": "lora",
        "num_layers": NUM_LAYERS,
        "lora_parameters": LORA_PARAMETERS,
        "harness_version": HARNESS_VERSION,
        "train_plan_version": TRAIN_PLAN_VERSION,
        "experiment": "EXP-010",
        "phase": phase,
        "schedule": sched.as_dict(),
        "base": model["repo_id"],
        "revision": model["revision"],
        "base_files_sha256": {k: v["sha256"] for k, v in model["files"].items()},
        "commit": git["commit"],
        "data": prov,
        "max_seq_length": limit,
        "trainable_parameters": lora["trainable_parameters"],
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam (installed defaults)",
        "grad_checkpoint": grad_checkpoint,
        "declared_change": declared_change,
        "packages": env["packages"],
        "python": env["python"],
    }


def cmd_train(args) -> int:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    if bool(args.grad_checkpoint) != (args.declared_change == "C1"):
        raise SystemExit("--grad-checkpoint is predeclared change C1 only; pass both or neither")
    git = _check_commit(args)
    sched = schedule(args.phase)
    out = _new_dir(Path(args.out), inside_repo=True)
    adapter_dir = _new_dir(Path(args.adapter_dir), inside_repo=False)
    pre = snapshot()
    refusal = preflight(pre)
    if refusal:
        write_json(out / "preflight_refusal.json", {"reason": refusal, "snapshot": pre})
        print(json.dumps({"status": "refused", "reason": refusal}))
        return 4
    signal.signal(signal.SIGTERM, _on_sigterm)
    baseline, baseline_rec = _baseline(args, pre)
    mpath = out / "measurements.json"
    env = _environment(git)
    meas: dict[str, Any] = {
        "experiment": "EXP-010",
        "phase": args.phase,
        "system": SYSTEM,
        "status": "running",
        "started_utc": pre["time_utc"],
        "preflight": pre,
        "stop_rule_baseline": baseline_rec,
        "environment": env,
        "schedule": sched.as_dict(),
        "configuration": {
            "base": MODEL,
            "precision": "bfloat16 as published",
            "fine_tune_type": "lora",
            "lora_parameters": LORA_PARAMETERS,
            "num_layers": NUM_LAYERS,
            "optimizer": "mlx.optimizers.Adam (installed defaults: betas [0.9, 0.999], "
            "eps 1e-8, bias_correction False), constant learning rate",
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "grad_checkpoint": bool(args.grad_checkpoint),
            "declared_change": args.declared_change,
            "loss": "completion-only cross-entropy, mean over completion tokens of one example; "
            "an update averages the 4 per-example means (PRERUN D4)",
            "report_to": None,
        },
        "adapter_dir": "<outside the repository>",
    }
    write_json(mpath, meas)
    log = Sink(out / "training_log.jsonl")
    monitor = ProcessMonitor(baseline=baseline)
    throughput = UpdateThroughputWatch()
    ledger: UpdateLedger | None = None
    t_start = time.perf_counter()
    try:
        if monitor.check("before load", sample()):
            raise StopRun(monitor.stopped)
        export, prov = load_export(Path(args.data_dir))
        meas["data"] = prov
        np.random.seed(SEED)
        t0 = time.perf_counter()
        meas["model"] = verify_model_dir(MODEL, Path(args.model_dir))
        meas["model_rehash_seconds"] = time.perf_counter() - t0
        mx.reset_peak_memory()
        try:
            t0 = time.perf_counter()
            model, tokenizer = load(str(Path(args.model_dir)))
            mx.synchronize()
        except Exception as exc:
            if is_allocation_error(exc):
                raise StopRun(f"load: allocation error: {type(exc).__name__}: {exc}") from exc
            raise
        meas["load_seconds"] = time.perf_counter() - t0
        meas["load_mlx_peak_bytes"] = mx.get_peak_memory()
        if sorted(tokenizer.eos_token_ids) != EXPECTED_EOS:
            raise PlanError("stop-token set differs from the reviewed EXP-009 set")
        examples = build_examples(tokenizer, export)
        limit = max_seq_length([e.length for s in examples.values() for e in s])
        if limit != PINNED_MAX_SEQ_LENGTH:
            raise PlanError(
                f"max_seq_length {limit} differs from the pinned {PINNED_MAX_SEQ_LENGTH}"
            )
        check_lengths([e for s in examples.values() for e in s], limit)
        train_set = SeqDataset(examples["train"])
        if args.phase == "pilot":
            eval_name, eval_set = "validation", SeqDataset(examples["validation"])
        else:
            eval_name, eval_set = "train_first_16", SeqDataset(examples["train"][:SMOKE_EVAL_IDS])
        meas["max_seq_length"] = limit
        meas["mask_self_check"] = mask_self_check(examples["train"] + examples["validation"], limit)
        meas["eval_set"] = {"name": eval_name, "examples": len(eval_set),
                            "completion_tokens": sum(e.completion_tokens
                                                     for e in eval_set.examples)}  # fmt: skip
        meas["lora"] = apply_lora(model)
        if meas["lora"]["trainable_parameters"] != PINNED_TRAINABLE_PARAMETERS:
            raise PlanError("trainable parameter count differs from the pinned value")
        config = adapter_config(args.phase, sched, meas["model"], prov, git, limit, meas["lora"],
                                env, bool(args.grad_checkpoint), args.declared_change)  # fmt: skip
        (adapter_dir / "adapter_config.json").write_bytes(
            (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
        )
        reason = monitor.check("load", sample())
        write_json(mpath, meas)
        if reason:
            raise StopRun(reason)

        ledger = UpdateLedger(sched)
        run_training(
            model,
            train_set,
            eval_set,
            eval_name,
            sched,
            limit,
            adapter_dir,
            log,
            monitor,
            throughput,
            ledger,
            meas,
            mpath,
            bool(args.grad_checkpoint),
        )
        checks = verify_training_outputs(out, adapter_dir, sched, meas["lora"]["trainable_tensors"])
        meas["completion_checks"] = checks
        meas["adapter_files"] = checks["files"]
        if not checks["passed"]:
            raise PlanError("completion checks failed: " + "; ".join(checks["problems"][:5]))
        final = sample()
        meas["final_sample"] = final
        reason = monitor.check("final", final)
        if reason:
            raise StopRun(reason)
        meas["status"] = "completed"
    except (StopRun, ExternalStop) as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
        external = _monitor_stop(args)
        if external:
            meas["external_monitor_stop"] = external
    except PlanError as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = f"plan: {exc}"
    except BaseException as exc:
        meas["status"] = "failed"
        meas["failure"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception | SystemExit):
            raise
    finally:
        last = f"update_{sched.updates:04d}.safetensors"

        def files() -> None:
            if "adapter_files" not in meas:
                found = sorted(p for p in adapter_dir.iterdir() if p.is_file())
                meas["adapter_files"] = {p.name: _file_record(p) for p in found}
            f = meas["adapter_files"]
            if "adapters.safetensors" in f and last in f:
                meas["final_adapter_equals_last_update_save"] = (
                    f["adapters.safetensors"]["sha256"] == f[last]["sha256"]
                )

        _finalize(meas, [
            ("log", log.close),
            ("updates", lambda: meas.__setitem__("updates_recorded",
                                                 ledger.updates if ledger else 0)),
            ("wall", lambda: meas.__setitem__("wall_seconds", time.perf_counter() - t_start)),
            ("monitor", lambda: meas.__setitem__("resource_monitor", monitor.summary())),
            ("throughput", lambda: meas.__setitem__("throughput", throughput.summary())),
            ("mlx_peak", lambda: meas.__setitem__("mlx_peak_bytes", mx.get_peak_memory())),
            ("rss", lambda: meas.__setitem__("rss_peak_process_bytes", rss_peak_bytes())),
            ("adapter_files", files),
            ("after", lambda: meas.__setitem__("after", snapshot())),
        ])  # fmt: skip
        write_json(mpath, meas)
    print(json.dumps({k: meas.get(k) for k in ("phase", "status", "stop_reason", "failure",
                                               "updates_recorded")}))  # fmt: skip
    return 0 if meas["status"] == "completed" else 3


# ---------------------------------------------------------------- adapter inference


def _json_equal(a: Any, b: Any) -> bool:
    return json.loads(json.dumps(a, sort_keys=True)) == json.loads(json.dumps(b, sort_keys=True))


def verify_adapter(
    adapter_dir: Path,
    run_dir: Path,
    expected_phase: str,
    commit: str,
    same_commit: bool,
    model: dict[str, Any],
) -> dict[str, Any]:
    """Correction B: the adapter's provenance, re-derived from the training records, not from a
    status flag. Phase, schedule, configuration, trainable count, commit (the same reviewed head
    for the smoke reload; an ancestor of HEAD for evaluation), base revision and file hashes,
    pinned export and corpus hashes, the training log re-checked against the plan, and every
    adapter file equal to the training record."""
    meas = json.loads((run_dir / "measurements.json").read_text(encoding="utf-8"))
    sched = schedule(expected_phase)
    problems = []
    if meas.get("status") != "completed":
        problems.append(f"training status {meas.get('status')!r}")
    if meas.get("phase") != expected_phase:
        problems.append(f"phase {meas.get('phase')!r}, expected {expected_phase!r}")
    if not _json_equal(meas.get("schedule"), sched.as_dict()):
        problems.append("schedule differs from the plan")
    cfg = meas.get("configuration", {})
    planned = {"lora_parameters": LORA_PARAMETERS, "num_layers": NUM_LAYERS,
               "learning_rate": LEARNING_RATE, "seed": SEED, "base": MODEL}  # fmt: skip
    if any(not _json_equal(cfg.get(k), v) for k, v in planned.items()):
        problems.append("configuration differs from the plan")
    if meas.get("lora", {}).get("trainable_parameters") != PINNED_TRAINABLE_PARAMETERS:
        problems.append("trainable parameter count differs from the pinned value")
    trained_at = meas.get("environment", {}).get("git", {}).get("commit")
    if same_commit and trained_at != commit:
        problems.append("training commit differs from the current reviewed head")
    if not same_commit and not (isinstance(trained_at, str) and _is_ancestor(trained_at)):
        problems.append("training commit is not an ancestor of the current head")
    m = meas.get("model", {})
    if m.get("revision") != model["revision"] or not _json_equal(
        {k: v["sha256"] for k, v in m.get("files", {}).items()},
        {k: v["sha256"] for k, v in model["files"].items()},
    ):
        problems.append("base checkpoint differs from the one used in training")
    data = meas.get("data", {})
    if (
        data.get("corpus_manifest_sha256") != MANIFEST_SHA256
        or {k: v.get("sha256") for k, v in data.get("files", {}).items()} != EXPORT_SHA256
    ):
        problems.append("training data hashes differ from the pinned export")
    if not meas.get("completion_checks", {}).get("passed"):
        problems.append("training completion checks did not pass")
    log = run_dir / "training_log.jsonl"
    records = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    problems += [f"log: {p}" for p in check_training_records(records, sched)]
    rec = meas.get("adapter_files", {})
    on_disk = {p.name for p in adapter_dir.iterdir() if p.is_file()}
    if set(rec) != on_disk:
        problems.append("adapter directory differs from the recorded file set")
    for name in sorted(set(rec) & on_disk):
        if _file_record(adapter_dir / name) != rec[name]:
            problems.append(f"{name} differs from the training record")
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    for key, want in (("phase", expected_phase), ("commit", trained_at),
                      ("revision", model["revision"]), ("schedule", sched.as_dict())):  # fmt: skip
        if not _json_equal(config.get(key), want):
            problems.append(f"adapter_config.json {key} differs")
    if problems:
        raise PlanError("adapter verification failed: " + "; ".join(problems[:6]))
    return {"phase": expected_phase, "file": "adapters.safetensors",
            **rec["adapters.safetensors"],
            "config_sha256": rec["adapter_config.json"]["sha256"],
            "training_commit": trained_at,
            "training_log_records": len(records)}  # fmt: skip


class AdapterRouter(base.Router):
    """EXP-009 router (same template, greedy, 256 tokens, D1 stop set) with a LoRA adapter.

    The MLX peak is reset once, before loading, and never again in the process, so a later
    ``mx.get_peak_memory()`` covers loading and every request since (lead decision 9, G12)."""

    def __init__(self, model_dir: Path, adapter_dir: Path, file_name: str) -> None:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import load_adapters

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(str(model_dir))
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg["lora_parameters"] != LORA_PARAMETERS or cfg["num_layers"] != NUM_LAYERS:
            raise PlanError("adapter configuration differs from the plan")
        if file_name != "adapters.safetensors":
            raise PlanError("only the final adapter is evaluated (no checkpoint picking)")
        load_adapters(self.model, str(adapter_dir))
        self.model.eval()
        mx.synchronize()
        self.load_seconds = time.perf_counter() - t0
        self.load_mlx_peak_bytes = mx.get_peak_memory()
        self.eos_token_ids = sorted(self.tokenizer.eos_token_ids)
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.eos_token_ids != EXPECTED_EOS or self.end_of_turn_id != 151645:
            raise PlanError("stop-token set is not the reviewed D1 set")


def reload_requests(router, warm, emit, monitor, sample_fn=sample) -> str | None:
    """The smoke reload requests: the EXP-009 request path (``make_request``, ``timed_call``),
    every record emitted before the stop rules are checked; returns a stop reason or ``None``."""
    request = base.make_request(router, f"{SYSTEM}-smoke")
    return run_requests(warm, request, sample_fn, monitor, "reload", emit)


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def cmd_reload(args) -> int:
    """Smoke check (correction B): fresh process, base + verified smoke adapter, real inference on
    exactly the 5 warm-up train examples, resource checks from the pre-load baseline through load
    and every request, persisted records re-read and checked. Never validation."""
    import mlx.core as mx

    git = _check_commit(args)
    run_dir = Path(args.run_dir).resolve()
    out = _new_dir(run_dir / "reload", inside_repo=True)
    pre = snapshot()
    refusal = preflight(pre)
    if refusal:
        write_json(out / "preflight_refusal.json", {"reason": refusal, "snapshot": pre})
        print(json.dumps({"status": "refused", "reason": refusal}))
        return 4
    signal.signal(signal.SIGTERM, _on_sigterm)
    baseline, baseline_rec = _baseline(args, pre)
    monitor = ProcessMonitor(baseline=baseline)
    mpath = out / "reload_measurements.json"
    meas: dict[str, Any] = {"experiment": "EXP-010", "phase": "smoke-reload", "status": "running",
                            "preflight": pre, "stop_rule_baseline": baseline_rec,
                            "environment": _environment(git)}  # fmt: skip
    write_json(mpath, meas)
    sink: Sink | None = None
    try:
        if monitor.check("before load", sample()):
            raise StopRun(monitor.stopped)
        if _sha256_bytes((CORPUS / "MANIFEST.json").read_bytes()) != MANIFEST_SHA256:
            raise PlanError("frozen corpus manifest does not match the pinned hash")
        meas["model"] = verify_model_dir(MODEL, Path(args.model_dir))
        meas["adapter"] = verify_adapter(Path(args.adapter_dir), run_dir, "smoke", git["commit"],
                                         True, meas["model"])  # fmt: skip
        _, corpus = load_corpus(CORPUS)
        by_id = {e["id"]: e for e in corpus["train"]}
        expected = warmup_ids(corpus["train"])
        if len(expected) != 5 or len(set(expected)) != 5:
            raise PlanError("the reload check needs exactly 5 unique warm-up IDs")
        meas["warmup_ids"] = expected
        write_json(mpath, meas)
        sink = Sink(out / "reload_train5.jsonl")
        try:
            router = AdapterRouter(Path(args.model_dir), Path(args.adapter_dir),
                                   "adapters.safetensors")  # fmt: skip
        except Exception as exc:
            if is_allocation_error(exc):
                raise StopRun(f"load: allocation error: {type(exc).__name__}: {exc}") from exc
            raise
        meas.update(load_seconds=router.load_seconds,
                    load_mlx_peak_bytes=router.load_mlx_peak_bytes,
                    rss_peak_after_load_bytes=rss_peak_bytes())  # fmt: skip
        reason = monitor.check("load", sample())
        reason = reason or reload_requests(router, [by_id[i] for i in expected], sink.emit,
                                           monitor)  # fmt: skip
        if reason:
            raise StopRun(reason)
        sink.close()
        problems = check_request_records(read_records(out / "reload_train5.jsonl"), expected)
        meas["record_checks"] = {"passed": not problems, "problems": problems}
        if problems:
            raise PlanError("reload records: " + "; ".join(problems[:5]))
        meas["status"] = "completed"
    except (StopRun, ExternalStop) as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
        external = _monitor_stop(args)
        if external:
            meas["external_monitor_stop"] = external
    except PlanError as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = f"plan: {exc}"
    except BaseException as exc:
        meas["status"] = "failed"
        meas["failure"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception | SystemExit):
            raise
    finally:
        _finalize(meas, [
            ("records", lambda: sink.close() if sink else None),
            ("count", lambda: meas.__setitem__("records", sink.count if sink else 0)),
            ("monitor", lambda: meas.__setitem__("resource_monitor", monitor.summary())),
            ("mlx_peak", lambda: meas.__setitem__("mlx_peak_bytes_load_and_requests",
                                                  mx.get_peak_memory())),
            ("rss", lambda: meas.__setitem__("rss_peak_process_bytes", rss_peak_bytes())),
            ("after", lambda: meas.__setitem__("after", snapshot())),
        ])  # fmt: skip
        write_json(mpath, meas)
    print(json.dumps({"phase": "smoke-reload", "status": meas["status"],
                      "records": meas.get("records")}))  # fmt: skip
    return 0 if meas["status"] == "completed" else 3


def cmd_evaluate(args) -> int:
    """The EXP-009 ``main`` frame without the repeats, with the final pilot adapter: preflight,
    re-hash, load, 5 train warm-ups (never scored), then the 160 validation requests in file order,
    with the EXP-009 decoding, stop rules and record format (``main_*.jsonl``).

    D18 freeze: started by ``exp010_launch.py start --phase evaluate`` as a supervised step with the
    sequence's common baseline (``--baseline``), like training; a monitor stop is copied into the
    record. Not run before the pilot analysis is frozen and reviewed (lead decision 10)."""
    import mlx.core as mx

    git = _check_commit(args)
    if _sha256_bytes((CORPUS / "MANIFEST.json").read_bytes()) != MANIFEST_SHA256:
        raise SystemExit("frozen corpus manifest does not match the pinned hash")
    model = verify_model_dir(MODEL, Path(args.model_dir))
    adapter = verify_adapter(Path(args.adapter_dir), Path(args.run_dir).resolve(), "pilot",
                             git["commit"], False, model)  # fmt: skip
    manifest, corpus = load_corpus(CORPUS)
    out = _new_dir(Path(args.out), inside_repo=True)
    pre = snapshot()
    refusal = preflight(pre)
    if refusal:
        write_json(out / "preflight_refusal.json", {"reason": refusal, "snapshot": pre})
        print(json.dumps({"status": "refused", "reason": refusal}))
        return 4
    signal.signal(signal.SIGTERM, _on_sigterm)
    baseline, baseline_rec = _baseline(args, pre)
    mpath = out / "main_measurements.json"
    meas: dict[str, Any] = {
        "experiment": "EXP-010",
        "phase": "main",
        "system": SYSTEM,
        "status": "running",
        "preflight": pre,
        "stop_rule_baseline": baseline_rec,
        "environment": _environment(git),
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": MANIFEST_SHA256,
        "prompt_version": PROMPT_VERSION,
        "decoding": {
            "sampler": "greedy (argmax)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
            "batch_size": 1,
            "stop_token_ids": EXPECTED_EOS,
        },  # fmt: skip
        "adapter": adapter,
        "model": model,
    }
    monitor = ProcessMonitor(baseline=baseline)
    write_json(mpath, meas)
    sinks: dict[str, Sink] = {}
    try:
        try:
            router = AdapterRouter(Path(args.model_dir), Path(args.adapter_dir),
                                   "adapters.safetensors")  # fmt: skip
        except Exception as exc:
            if is_allocation_error(exc):
                raise StopRun(f"load: allocation error: {type(exc).__name__}: {exc}") from exc
            raise
        meas.update(cold_load_seconds=router.load_seconds,
                    load_mlx_peak_bytes=router.load_mlx_peak_bytes,
                    eos_token_ids=router.eos_token_ids,
                    rss_peak_after_load_bytes=rss_peak_bytes())  # fmt: skip
        reason = monitor.check("load", sample())
        write_json(mpath, meas)
        if reason:
            raise StopRun(reason)
        val = corpus["validation"]
        by_id = {e["id"]: e for e in corpus["train"]}
        warm = [by_id[i] for i in warmup_ids(corpus["train"])]
        meas["warmup_ids"] = [e["id"] for e in warm]
        request = base.make_request(router, SYSTEM)
        base._phase("warmup", warm, request, out, "main", monitor, sinks)
        meas["after_warmup"] = snapshot()
        meas["mlx_peak_bytes_load_and_warmup"] = mx.get_peak_memory()
        write_json(mpath, meas)
        slow = SlowdownWatch()
        try:
            base._phase("validation", val, request, out, "main", monitor, sinks, slow)
        finally:
            meas["validation_slowdown"] = slow.summary()
            # no reset since loading: this peak covers load, warm-up and validation (G12)
            meas["mlx_peak_bytes_load_warmup_validation"] = mx.get_peak_memory()
            meas["rss_peak_after_validation_bytes"] = rss_peak_bytes()
            meas["after_validation"] = snapshot()
            write_json(mpath, meas)
        meas["validation_completed"] = sinks["validation"].count == len(val)
        meas["status"] = "completed"
    except (StopRun, ExternalStop) as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
        external = _monitor_stop(args)
        if external:
            meas["external_monitor_stop"] = external
    except BaseException as exc:
        meas["status"] = "failed"
        meas["failure"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception | SystemExit):
            raise
    finally:
        _finalize(meas, [
            ("sinks", lambda: [s.close() for s in sinks.values()]),
            ("counts", lambda: meas.__setitem__("counts", {k: s.count for k, s in sinks.items()})),
            ("monitor", lambda: meas.__setitem__("resource_monitor", monitor.summary())),
            ("rss", lambda: meas.__setitem__("rss_peak_process_bytes", rss_peak_bytes())),
            ("after", lambda: meas.__setitem__("after", snapshot())),
        ])  # fmt: skip
        write_json(mpath, meas)
    print(json.dumps({k: meas.get(k) for k in ("phase", "status", "stop_reason", "failure",
                                               "counts")}))  # fmt: skip
    return 0 if meas["status"] == "completed" else 3


# ---------------------------------------------------------------- harness self-test


def _tiny_model(layers: int = 10):
    """A randomly initialised Qwen3-architecture model with tiny dimensions (not a checkpoint)."""
    import mlx.core as mx
    from mlx_lm.models import qwen3

    mx.random.seed(7)
    margs = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=layers,
                            intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
                            vocab_size=151936, num_key_value_heads=2, max_position_embeddings=1024,
                            head_dim=16, tie_word_embeddings=True, rope_theta=10000.0)  # fmt: skip
    model = qwen3.Model(margs)
    mx.eval(model.parameters())
    return model


def _synthetic_examples(n: int, seed: int) -> list[TrainingExample]:
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        prompt = [int(t) for t in rng.integers(0, 5000, int(rng.integers(30, 60)))]
        completion = [int(t) for t in rng.integers(0, 5000, int(rng.integers(4, 12)))]
        full = prompt + completion + [151645, 198]
        out.append(completion_example(f"synthetic-{i:03d}", prompt, full))
    return out


class _FakeRouter:
    """Scripted router for the reload error paths: no model, no generation."""

    def __init__(self, fail_at: int | None = None, exc: type[Exception] = RuntimeError) -> None:
        self.calls, self.fail_at, self.exc = 0, fail_at, exc

    def generate(self, router_input) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        if self.calls == self.fail_at:
            raise self.exc("scripted failure")
        return {"raw_output": '{"kind":"REJECT"}', "generation_tps": 1.0}


def _reload_error_paths() -> dict[str, Any]:
    """Correction B regression: the reload request path with scripted routers on the 5 warm-up
    train examples (no model): a clean run passes, a request error and an allocation failure do
    not, and a missing record is caught by the persisted-record check."""
    _, corpus = load_corpus(CORPUS)
    by_id = {e["id"]: e for e in corpus["train"]}
    expected = warmup_ids(corpus["train"])
    warm = [by_id[i] for i in expected]
    ok_sample = {"swap_used": 0, "pressure_level": 1, "power_source": "AC Power"}

    def run(router) -> tuple[str | None, list[str]]:
        records: list[dict[str, Any]] = []
        monitor = ProcessMonitor(baseline=dict(ok_sample))
        reason = reload_requests(router, warm, records.append, monitor, lambda: dict(ok_sample))
        return reason, check_request_records(records, expected)

    clean = run(_FakeRouter())
    error = run(_FakeRouter(fail_at=3))
    alloc = run(_FakeRouter(fail_at=2, exc=MemoryError))
    return {
        "clean_run_passes": clean == (None, []),
        "request_error_fails_record_check": error[0] is None
        and any("status 'error'" in p for p in error[1]),
        "allocation_error_stops": alloc[0] is not None
        and "allocation" in alloc[0]
        and len(alloc[1]) > 0,
        "missing_record_detected": bool(check_request_records([], expected)),
    }


def _adapter_fixture_checks(root: Path) -> dict[str, Any]:
    """Correction B regression for ``verify_adapter`` on synthetic run directories (placeholder
    bytes, not adapters; nothing is loaded): a consistent smoke record is accepted, and each
    single defect is rejected."""
    from axbench.trainwatch import expected_record_sequence

    sched = schedule("smoke")
    commit = "0" * 40
    model = {"repo_id": "fixture", "revision": "fixture-revision",
             "files": {"model.safetensors": {"sha256": "a" * 64}}}  # fmt: skip

    def build(name: str, before=None, after=None) -> tuple[Path, Path]:
        run = _new_dir(root / f"fixture_{name}" / "run", inside_repo=False)
        adapters = _new_dir(root / f"fixture_{name}" / "adapters", inside_repo=False)
        records: list[dict[str, Any]] = []
        for kind, u in expected_record_sequence(sched):
            if kind == "update":
                records.append({"kind": kind, "update": u, "micro_steps": u * sched.grad_accum,
                                "train_loss": 1.0, "update_seconds": 0.5})  # fmt: skip
            elif kind == "eval":
                records.append({"kind": kind, "update": u, "loss": 1.0})
            else:
                f = adapters / f"update_{u:04d}.safetensors"
                f.write_bytes(f"placeholder {u}".encode())
                records.append({"kind": kind, "update": u, "file": f.name, **_file_record(f)})
        (adapters / "adapters.safetensors").write_bytes(f"placeholder {sched.updates}".encode())
        (adapters / "adapter_config.json").write_text(json.dumps(
            {"phase": "smoke", "commit": commit, "revision": model["revision"],
             "schedule": sched.as_dict()}))  # fmt: skip
        meas: dict[str, Any] = {
            "status": "completed", "phase": "smoke", "schedule": sched.as_dict(),
            "configuration": {"lora_parameters": LORA_PARAMETERS, "num_layers": NUM_LAYERS,
                              "learning_rate": LEARNING_RATE, "seed": SEED, "base": MODEL},
            "lora": {"trainable_parameters": PINNED_TRAINABLE_PARAMETERS},
            "environment": {"git": {"commit": commit}}, "model": model,
            "data": {"corpus_manifest_sha256": MANIFEST_SHA256,
                     "files": {k: {"sha256": v} for k, v in EXPORT_SHA256.items()}},
            "completion_checks": {"passed": True},
        }  # fmt: skip
        if before:
            before(meas, records)
        meas["adapter_files"] = {p.name: _file_record(p) for p in adapters.iterdir()}
        (run / "training_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (run / "measurements.json").write_text(json.dumps(meas))
        if after:
            after(adapters)
        return run, adapters

    def accepted(name: str, before=None, after=None, use_model=None) -> bool:
        run, adapters = build(name, before, after)
        try:
            verify_adapter(adapters, run, "smoke", commit, True, use_model or model)
        except PlanError:
            return False
        return True

    def drop_eval(m, r):
        r.remove(next(x for x in r if x["kind"] == "eval" and x["update"] == sched.updates))

    def nan_loss(m, r):
        next(x for x in r if x["kind"] == "update")["train_loss"] = float("nan")

    other_model = {**model, "revision": "other-revision"}
    rejected = {
        "status_not_completed": not accepted("status", lambda m, r: m.update(status="stopped")),
        "wrong_phase": not accepted("phase", lambda m, r: m.update(phase="pilot")),
        "missing_eval_record": not accepted("eval", drop_eval),
        "non_finite_loss_record": not accepted("nan", nan_loss),
        "other_training_commit": not accepted(
            "commit", lambda m, r: m["environment"]["git"].update(commit="1" * 40)),
        "data_hash_differs": not accepted(
            "data", lambda m, r: m["data"]["files"]["train.jsonl"].update(sha256="b" * 64)),
        "completion_checks_failed": not accepted(
            "checks", lambda m, r: m.update(completion_checks={"passed": False})),
        "adapter_modified_after_training": not accepted(
            "modified", after=lambda d: (d / "adapters.safetensors").write_bytes(b"changed")),
        "extra_adapter_file": not accepted(
            "extra", after=lambda d: (d / "update_0010.safetensors").write_bytes(b"extra")),
        "other_base_checkpoint": not accepted("base", use_model=other_model),
    }  # fmt: skip
    return {"valid_fixture_accepted": accepted("valid"), "rejected": rejected}


def _finalization_checks() -> dict[str, bool]:
    """Correction C: a finalization error never replaces a stop, and fails a completed run."""

    def boom() -> None:
        raise OSError("scripted finalization failure")

    stopped = {"status": "stopped", "stop_reason": "monitor stop"}
    _finalize(stopped, [("ok", lambda: None), ("boom", boom)])
    completed = {"status": "completed"}
    _finalize(completed, [("boom", boom)])
    return {
        "stop_reason_preserved": stopped["status"] == "stopped"
        and stopped["stop_reason"] == "monitor stop"
        and len(stopped["finalization_errors"]) == 1,
        "completed_becomes_failed": completed["status"] == "failed",
    }


def cmd_selftest(args) -> int:
    """Harness self-test on a tiny randomly initialised Qwen3-architecture model and synthetic token
    sequences: no checkpoint, no export, nothing of AX-01 is trained. It exercises the update
    ledger, the completion-only loss with gradient accumulation, evaluation and saves at update
    boundaries, that evaluation does not change the training data order, that ``update_seconds``
    excludes callback time, the optimizer step count, the completion checks (and that they reject
    a tampered adapter), the adapter file written by the trainer and its reload through
    ``load_adapters``, and the reload request error paths with scripted routers. Not a training
    result."""
    import mlx.core as mx
    import numpy as np
    from mlx_lm.tuner.utils import load_adapters

    from axbench.trainwatch import Schedule

    root = Path(args.out).resolve()
    if REPO in root.parents:
        raise SystemExit("self-test outputs must be outside the repository")
    train_ex, eval_ex = _synthetic_examples(24, 1), _synthetic_examples(6, 2)
    limit = max_seq_length([e.length for e in train_ex + eval_ex])
    report: dict[str, Any] = {"mask_self_check": mask_self_check(train_ex + eval_ex, limit)}
    delay = 1.0
    runs = {}
    for name, evals, saves, eval_delay in (("a", (0, 3, 6), (3, 6), delay),
                                           ("b", (0, 6), (6,), 0.0)):  # fmt: skip
        adapter_dir = _new_dir(root / f"adapters_{name}", inside_repo=False)
        out = _new_dir(root / f"run_{name}", inside_repo=False)
        sched = Schedule("selftest", 6, 4, 24, evals, saves, 1.0)
        np.random.seed(SEED)
        model = _tiny_model()
        lora = apply_lora(model)
        (adapter_dir / "adapter_config.json").write_text(json.dumps(
            {"fine_tune_type": "lora", "num_layers": NUM_LAYERS,
             "lora_parameters": LORA_PARAMETERS}))  # fmt: skip
        log = Sink(out / "training_log.jsonl")
        pre = snapshot()
        monitor = ProcessMonitor(baseline={k: pre[k] for k in RULE_KEYS})
        ledger = UpdateLedger(sched)
        meas: dict[str, Any] = {}
        try:
            run_training(model, SeqDataset(train_ex), SeqDataset(eval_ex), "synthetic", sched,
                         limit, adapter_dir, log, monitor, UpdateThroughputWatch(), ledger, meas,
                         out / "measurements.json", False, eval_delay)  # fmt: skip
        finally:
            log.close()
        checks = verify_training_outputs(out, adapter_dir, sched, lora["trainable_tensors"])
        recs = read_records(out / "training_log.jsonl")
        ups = [r for r in recs if r["kind"] == "update"]
        runs[name] = {
            "updates": ledger.updates,
            "optimizer_steps": meas["optimizer_steps"],
            "micro_steps": [r["micro_steps"] for r in ups],
            "train_loss": [r["train_loss"] for r in ups],
            "update_seconds": [r["update_seconds"] for r in ups],
            "callback_interval_seconds": [r["callback_interval_seconds"] for r in ups],
            "eval_updates": [r["update"] for r in recs if r["kind"] == "eval"],
            "save_updates": [r["update"] for r in recs if r["kind"] == "save"],
            "completion_checks_passed": checks["passed"],
            "completion_problems": checks["problems"],
            "final_equals_last_save": checks["final_adapter_equals_last_update_save"],
            "lora_b_max_abs": meas["lora_b_max_abs_after_training"],
        }
        # Reload into a fresh copy of the same base and compare outputs with the trained model.
        model.eval()
        fresh = _tiny_model()
        load_adapters(fresh, str(adapter_dir))
        fresh.eval()
        x = mx.array([list(train_ex[0].tokens)])
        runs[name]["reload_max_abs_logit_difference"] = float(mx.abs(model(x) - fresh(x)).max())
        if name == "a":
            # Tamper with a copy of the final adapter: one zero lora_b tensor must be rejected.
            weights = dict(mx.load(str(adapter_dir / "adapters.safetensors")))
            key = sorted(k for k in weights if k.endswith("lora_b"))[0]
            weights[key] = mx.zeros_like(weights[key])
            tampered = root / "tampered.safetensors"
            mx.save_safetensors(str(tampered), weights)
            runs[name]["tampered_adapter_problems"] = check_adapter_tensors(
                tensor_stats(tampered), lora["trainable_tensors"])  # fmt: skip
    a, b = runs["a"], runs["b"]
    report["runs"] = runs
    report["reload_error_paths"] = _reload_error_paths()
    report["adapter_verification_fixtures"] = _adapter_fixture_checks(root)
    report["finalization"] = _finalization_checks()
    report["checks"] = {
        "updates_complete": a["updates"] == 6 and b["updates"] == 6,
        "optimizer_step_count_equals_updates": a["optimizer_steps"] == 6
        and b["optimizer_steps"] == 6,
        "reports_at_update_boundaries": a["micro_steps"] == [4, 8, 12, 16, 20, 24],
        "evaluation_at_planned_updates": a["eval_updates"] == [0, 3, 6]
        and b["eval_updates"] == [0, 6],
        "saves_at_planned_updates": a["save_updates"] == [3, 6],
        "completion_checks_pass": a["completion_checks_passed"] and b["completion_checks_passed"],
        "tampered_adapter_rejected": bool(a["tampered_adapter_problems"]),
        "final_adapter_equals_last_save": a["final_equals_last_save"]
        and b["final_equals_last_save"],
        "adapter_trained": a["lora_b_max_abs"] > 0,
        "data_order_independent_of_evaluation": a["train_loss"] == b["train_loss"],
        # update 4 follows the update-3 callback, which spent >= `delay` s in evaluation
        "update_seconds_exclude_callback_time": a["callback_interval_seconds"][3] >= delay
        and a["update_seconds"][3] < delay,
        "reload_reproduces_outputs": a["reload_max_abs_logit_difference"] == 0.0,
        "reload_error_paths": all(report["reload_error_paths"].values()),
        "adapter_verification_accepts_valid_and_rejects_defects": report[
            "adapter_verification_fixtures"
        ]["valid_fixture_accepted"]
        and all(report["adapter_verification_fixtures"]["rejected"].values()),
        "finalization_never_masks_the_outcome": all(report["finalization"].values()),
    }
    report["passed"] = all(report["checks"].values())
    report["note"] = "tiny random model and synthetic tokens; not a training result"
    write_json(root / "selftest_report.json", report)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tokens")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--per-example", help="write per-example token counts (JSONL, no texts)")
    p = sub.add_parser("params")
    p.add_argument("--model-dir", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--out", required=True, help="new directory outside the repository")
    p = sub.add_parser("train")
    p.add_argument("--phase", required=True, choices=("smoke", "pilot"))
    for name in ("--model-dir", "--data-dir", "--adapter-dir", "--out", "--expected-commit"):
        p.add_argument(name, required=True)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="predeclared change C1 only (PRERUN D9); off by default")  # fmt: skip
    p.add_argument("--declared-change", choices=DECLARED_CHANGES,
                   help="required with --grad-checkpoint")  # fmt: skip
    p.add_argument("--baseline", help="the sequence's common pre-load baseline (baseline.json)")
    for name in ("reload", "evaluate"):
        p = sub.add_parser(name)
        for opt in ("--model-dir", "--adapter-dir", "--run-dir", "--expected-commit"):
            p.add_argument(opt, required=True)
        if name == "evaluate":
            p.add_argument("--out", required=True)
        p.add_argument("--baseline", help="the sequence's common pre-load baseline")
    args = parser.parse_args()
    commands = {"tokens": cmd_tokens, "params": cmd_params, "train": cmd_train,
                "reload": cmd_reload, "evaluate": cmd_evaluate,
                "selftest": cmd_selftest}  # fmt: skip
    return commands[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
