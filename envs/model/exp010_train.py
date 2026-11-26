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
    warmup_ids,
)
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
    check_lengths,
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

HARNESS_VERSION = "exp010.harness/1"
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


def _environment(git: dict[str, Any]) -> dict[str, Any]:
    env = base.environment(git)
    env["runner_version"] = HARNESS_VERSION
    env["train_plan_version"] = TRAIN_PLAN_VERSION
    return env


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
) -> None:
    """Loss at update 0, the installed MLX-LM trainer with the completion-only loss, per-update
    records and stop rules, evaluation and saves at update boundaries (PRERUN D4-D8)."""
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

    class Callback(TrainingCallback):
        def on_train_loss_report(self, info: dict) -> None:
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
    write_json(mpath, meas)
    run_eval(0)
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    t0 = time.perf_counter()
    try:
        train(model=model, optimizer=optimizer, train_dataset=train_set, val_dataset=None,
              args=targs, loss=completion_loss, training_callback=Callback())  # fmt: skip
    except Exception as exc:
        if is_allocation_error(exc):
            raise StopRun(f"train: allocation error: {type(exc).__name__}: {exc}") from exc
        raise
    meas["train_loop_seconds"] = time.perf_counter() - t0
    if not ledger.complete():
        raise PlanError(f"{ledger.updates} updates recorded, {sched.updates} planned")
    lora_b = [v for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]
    meas["lora_b_max_abs_after_training"] = max(float(mx.abs(v).max()) for v in lora_b)


def cmd_train(args) -> int:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

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
    mpath = out / "measurements.json"
    meas: dict[str, Any] = {
        "experiment": "EXP-010",
        "phase": args.phase,
        "system": SYSTEM,
        "status": "running",
        "started_utc": pre["time_utc"],
        "preflight": pre,
        "environment": _environment(git),
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
            "loss": "completion-only cross-entropy, mean over completion tokens of one example; "
            "an update averages the 4 per-example means (PRERUN D4)",
            "report_to": None,
        },
        "adapter_dir": "<outside the repository>",
    }
    write_json(mpath, meas)
    log = Sink(out / "training_log.jsonl")
    monitor = ProcessMonitor(baseline={k: pre[k] for k in ("swap_used", "pressure_level",
                                                            "power_source")})  # fmt: skip
    throughput = UpdateThroughputWatch()
    t_start = time.perf_counter()
    try:
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
        config = {"fine_tune_type": "lora", "num_layers": NUM_LAYERS,
                  "lora_parameters": LORA_PARAMETERS, "harness_version": HARNESS_VERSION,
                  "base": meas["model"]["repo_id"], "revision": meas["model"]["revision"],
                  "phase": args.phase, "commit": git["commit"], "data": prov}  # fmt: skip
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
        meas["status"] = "completed"
    except (StopRun, ExternalStop) as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
    except PlanError as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = f"plan: {exc}"
    except BaseException as exc:
        meas["status"] = "failed"
        meas["failure"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            raise
    finally:
        log.close()
        meas["updates_recorded"] = ledger.updates if "ledger" in locals() else 0
        meas["wall_seconds"] = time.perf_counter() - t_start
        meas["resource_monitor"] = monitor.summary()
        meas["throughput"] = throughput.summary()
        meas["mlx_peak_bytes"] = mx.get_peak_memory()
        meas["rss_peak_process_bytes"] = rss_peak_bytes()
        meas["adapter_files"] = {
            p.name: _file_record(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()
        }
        files = meas["adapter_files"]
        last = f"update_{sched.updates:04d}.safetensors"
        if "adapters.safetensors" in files and last in files:
            meas["final_adapter_equals_last_update_save"] = (
                files["adapters.safetensors"]["sha256"] == files[last]["sha256"]
            )
        meas["after"] = snapshot()
        write_json(mpath, meas)
    print(json.dumps({k: meas.get(k) for k in ("phase", "status", "stop_reason", "failure",
                                               "updates_recorded")}))  # fmt: skip
    return 0 if meas["status"] == "completed" else 3


# ---------------------------------------------------------------- adapter inference


def verify_adapter(adapter_dir: Path, run_dir: Path, file_name: str) -> dict[str, Any]:
    """The adapter files must equal the hashes recorded by the completed training process."""
    meas = json.loads((run_dir / "measurements.json").read_text(encoding="utf-8"))
    if meas.get("status") != "completed":
        raise SystemExit("the training run is not completed")
    rec = meas["adapter_files"]
    for name in (file_name, "adapter_config.json"):
        if name not in rec or _file_record(adapter_dir / name) != rec[name]:
            raise SystemExit(f"{name} differs from the training record")
    return {"phase": meas["phase"], "file": file_name, **rec[file_name],
            "config_sha256": rec["adapter_config.json"]["sha256"],
            "training_commit": meas["environment"]["git"]["commit"]}  # fmt: skip


class AdapterRouter(base.Router):
    """EXP-009 router (same template, greedy, 256 tokens, D1 stop set) with a LoRA adapter."""

    def __init__(self, model_dir: Path, adapter_dir: Path, file_name: str) -> None:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import load_adapters

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(str(model_dir))
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg["lora_parameters"] != LORA_PARAMETERS or cfg["num_layers"] != NUM_LAYERS:
            raise SystemExit("adapter configuration differs from the plan")
        if file_name != "adapters.safetensors":
            raise SystemExit("only the final adapter is evaluated (no checkpoint picking)")
        load_adapters(self.model, str(adapter_dir))
        self.model.eval()
        mx.synchronize()
        self.load_seconds = time.perf_counter() - t0
        self.load_mlx_peak_bytes = mx.get_peak_memory()
        self.eos_token_ids = sorted(self.tokenizer.eos_token_ids)
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.eos_token_ids != EXPECTED_EOS or self.end_of_turn_id != 151645:
            raise SystemExit("stop-token set is not the reviewed D1 set")


def cmd_reload(args) -> int:
    """Smoke check: fresh process, base + saved adapter, real inference on 5 train examples."""
    git = _check_commit(args)
    run_dir = Path(args.run_dir).resolve()
    out = run_dir / "reload"
    _new_dir(out, inside_repo=True)
    pre = snapshot()
    refusal = preflight(pre)
    if refusal:
        write_json(out / "preflight_refusal.json", {"reason": refusal, "snapshot": pre})
        return 4
    meas: dict[str, Any] = {"experiment": "EXP-010", "phase": "smoke-reload", "status": "running",
                            "preflight": pre, "environment": _environment(git)}  # fmt: skip
    meas["model"] = verify_model_dir(MODEL, Path(args.model_dir))
    meas["adapter"] = verify_adapter(Path(args.adapter_dir), run_dir, "adapters.safetensors")
    _, corpus = load_corpus(CORPUS)
    by_id = {e["id"]: e for e in corpus["train"]}
    warm = [by_id[i] for i in warmup_ids(corpus["train"])]
    sink = Sink(out / "reload_train5.jsonl")
    try:
        router = AdapterRouter(Path(args.model_dir), Path(args.adapter_dir), "adapters.safetensors")
        meas.update(
            load_seconds=router.load_seconds, load_mlx_peak_bytes=router.load_mlx_peak_bytes
        )
        request = base.make_request(router, f"{SYSTEM}-smoke")
        for e in warm:
            sink.emit(request(e))
        meas["status"] = "completed"
    except BaseException as exc:
        meas["status"] = "failed"
        meas["failure"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            raise
    finally:
        sink.close()
        meas["records"] = sink.count
        meas["rss_peak_process_bytes"] = rss_peak_bytes()
        meas["after"] = snapshot()
        write_json(out / "reload_measurements.json", meas)
    print(json.dumps({"phase": "smoke-reload", "status": meas["status"], "records": sink.count}))
    return 0 if meas["status"] == "completed" else 3


def cmd_evaluate(args) -> int:
    """The EXP-009 ``main`` frame without the repeats, with the final pilot adapter: preflight,
    re-hash, load, 5 train warm-ups (never scored), then the 160 validation requests in file order,
    with the EXP-009 decoding, stop rules and record format (``main_*.jsonl``)."""
    import mlx.core as mx

    adapter = verify_adapter(Path(args.adapter_dir), Path(args.run_dir).resolve(),
                             "adapters.safetensors")  # fmt: skip
    if adapter["phase"] != "pilot":
        raise SystemExit("only the pilot adapter is evaluated on validation")
    git = _check_commit(args)
    if _sha256_bytes((CORPUS / "MANIFEST.json").read_bytes()) != MANIFEST_SHA256:
        raise SystemExit("frozen corpus manifest does not match the pinned hash")
    manifest, corpus = load_corpus(CORPUS)
    out = _new_dir(Path(args.out), inside_repo=True)
    pre = snapshot()
    refusal = preflight(pre)
    if refusal:
        write_json(out / "preflight_refusal.json", {"reason": refusal, "snapshot": pre})
        print(json.dumps({"status": "refused", "reason": refusal}))
        return 4
    mpath = out / "main_measurements.json"
    meas: dict[str, Any] = {
        "experiment": "EXP-010",
        "phase": "main",
        "system": SYSTEM,
        "status": "running",
        "preflight": pre,
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
    }
    t0 = time.perf_counter()
    meas["model"] = verify_model_dir(MODEL, Path(args.model_dir))
    meas["model_rehash_seconds"] = time.perf_counter() - t0
    monitor = ProcessMonitor(baseline={k: pre[k] for k in ("swap_used", "pressure_level",
                                                            "power_source")})  # fmt: skip
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
        write_json(mpath, meas)
        mx.reset_peak_memory()
        slow = SlowdownWatch()
        try:
            base._phase("validation", val, request, out, "main", monitor, sinks, slow)
        finally:
            meas["validation_slowdown"] = slow.summary()
            meas["validation_mlx_peak_bytes"] = mx.get_peak_memory()
            meas["rss_peak_after_validation_bytes"] = rss_peak_bytes()
            meas["after_validation"] = snapshot()
            write_json(mpath, meas)
        meas["validation_completed"] = sinks["validation"].count == len(val)
        meas["status"] = "completed"
    except StopRun as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
    except BaseException as exc:
        meas["status"] = "failed"
        meas["failure"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            raise
    finally:
        for s in sinks.values():
            s.close()
        meas["counts"] = {k: s.count for k, s in sinks.items()}
        meas["resource_monitor"] = monitor.summary()
        meas["rss_peak_process_bytes"] = rss_peak_bytes()
        meas["after"] = snapshot()
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


def cmd_selftest(args) -> int:
    """Harness self-test on a tiny randomly initialised Qwen3-architecture model and synthetic token
    sequences: no checkpoint, no corpus, nothing of AX-01 is trained. It exercises the update
    ledger, the completion-only loss with gradient accumulation, evaluation and saves at update
    boundaries, that evaluation does not change the training data order, the adapter file written
    by the trainer and its reload through ``load_adapters``. Not a training result."""
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
    runs = {}
    for name, evals, saves in (("a", (0, 3, 6), (3, 6)), ("b", (0, 6), (6,))):
        adapter_dir = _new_dir(root / f"adapters_{name}", inside_repo=False)
        out = _new_dir(root / f"run_{name}", inside_repo=False)
        sched = Schedule("selftest", 6, 4, 24, evals, saves, 1.0)
        np.random.seed(SEED)
        model = _tiny_model()
        apply_lora(model)
        (adapter_dir / "adapter_config.json").write_text(json.dumps(
            {"fine_tune_type": "lora", "num_layers": NUM_LAYERS,
             "lora_parameters": LORA_PARAMETERS}))  # fmt: skip
        log = Sink(out / "training_log.jsonl")
        pre = snapshot()
        monitor = ProcessMonitor(baseline={k: pre[k] for k in ("swap_used", "pressure_level",
                                                                "power_source")})  # fmt: skip
        ledger = UpdateLedger(sched)
        meas: dict[str, Any] = {}
        try:
            run_training(model, SeqDataset(train_ex), SeqDataset(eval_ex), "synthetic", sched,
                         limit, adapter_dir, log, monitor, UpdateThroughputWatch(), ledger, meas,
                         out / "measurements.json", False)  # fmt: skip
        finally:
            log.close()
        recs = [json.loads(x) for x in (out / "training_log.jsonl").read_text().splitlines()]
        files = {p.name: _file_record(p) for p in adapter_dir.iterdir()}
        runs[name] = {
            "updates": ledger.updates,
            "micro_steps": [r["micro_steps"] for r in recs if r["kind"] == "update"],
            "train_loss": [r["train_loss"] for r in recs if r["kind"] == "update"],
            "eval_updates": [r["update"] for r in recs if r["kind"] == "eval"],
            "save_updates": [r["update"] for r in recs if r["kind"] == "save"],
            "final_equals_last_save": files["adapters.safetensors"]["sha256"]
            == files["update_0006.safetensors"]["sha256"],
            "lora_b_max_abs": meas["lora_b_max_abs_after_training"],
        }
        # Reload into a fresh copy of the same base and compare outputs with the trained model.
        model.eval()
        fresh = _tiny_model()
        load_adapters(fresh, str(adapter_dir))
        fresh.eval()
        x = mx.array([list(train_ex[0].tokens)])
        runs[name]["reload_max_abs_logit_difference"] = float(mx.abs(model(x) - fresh(x)).max())
    a, b = runs["a"], runs["b"]
    report["runs"] = runs
    report["checks"] = {
        "updates_complete": a["updates"] == 6 and b["updates"] == 6,
        "reports_at_update_boundaries": a["micro_steps"] == [4, 8, 12, 16, 20, 24],
        "evaluation_at_planned_updates": a["eval_updates"] == [0, 3, 6]
        and b["eval_updates"] == [0, 6],
        "saves_at_planned_updates": a["save_updates"] == [3, 6],
        "final_adapter_equals_last_save": a["final_equals_last_save"]
        and b["final_equals_last_save"],
        "adapter_trained": a["lora_b_max_abs"] > 0,
        "data_order_independent_of_evaluation": a["train_loss"] == b["train_loss"],
        "reload_reproduces_outputs": a["reload_max_abs_logit_difference"] == 0.0,
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
    for name in ("reload", "evaluate"):
        p = sub.add_parser(name)
        for opt in ("--model-dir", "--adapter-dir", "--run-dir", "--expected-commit"):
            p.add_argument(opt, required=True)
        if name == "evaluate":
            p.add_argument("--out", required=True)
    args = parser.parse_args()
    commands = {"tokens": cmd_tokens, "params": cmd_params, "train": cmd_train,
                "reload": cmd_reload, "evaluate": cmd_evaluate,
                "selftest": cmd_selftest}  # fmt: skip
    return commands[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
