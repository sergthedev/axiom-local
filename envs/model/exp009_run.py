"""EXP-009 model runner: untuned Qwen3 inference on AXBENCH validation with resource measurement.

Model environment only (ADR 0012). Run from ``envs/model`` on a clean checkout of the commit that
contains the EXP-009 pre-run record, on mains power with other applications closed:

    uv run --locked python exp009_run.py verify --model qwen3-0.6b --model-dir <dir>
    uv run --locked python exp009_run.py main   --model qwen3-0.6b --model-dir <dir> --out <run dir>
    uv run --locked python exp009_run.py fresh  --model qwen3-0.6b --model-dir <dir> \
        --out <run dir> --index 1

* ``verify``: SHA-256 of every provisioned weight file against the pinned published hashes.
* ``main``: one fresh process: cold load, five train warm-ups (never scored), the 160 validation
  requests in file order (timed, stop rules checked after every request), then the 30 stratified
  repeatability requests 20 times in the same process.
* ``fresh``: one fresh process: cold load, then the 30 repeatability requests once.

Inference: ``HF_HUB_OFFLINE=1`` (never downloads), prompt ``axiom.prompt/1``, the pinned chat
template with ``enable_thinking=False`` and the generation prompt, greedy decoding (the MLX-LM
default sampler is argmax), at most 256 new tokens, stop at the end-of-turn token (the tokenizer's
EOS set as loaded from the checkpoint ``config.json``), batch size 1, no constrained decoding, no
output repair: the raw generated text is written as is. Raw outputs are scored later, in the Core
environment, by ``benchmarks/issue4_exp009_analyze.py``.

Never reads, generates or scores the final test split. Writes no absolute paths.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from provision import CHECKPOINTS  # noqa: E402

from axbench.corpus import load_corpus  # noqa: E402
from axbench.runwatch import (  # noqa: E402
    StopWatch,
    parse_free_percentage,
    parse_swapusage,
    repeat_subset,
    warmup_ids,
)
from axiom_router.prompt import (  # noqa: E402
    PROMPT_VERSION,
    RouterInputError,
    build_router_input,
    chat_messages,
)
from axiom_router.schema import project_columns  # noqa: E402

RUNNER_VERSION = "exp009.runner/1"
CORPUS = REPO / "data" / "axbench" / "v0.1-ai-reviewed"
MANIFEST_SHA256 = "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"
MAX_NEW_TOKENS = 256
REPEATS_IN_PROCESS = 20
SYSTEM_NAMES = {"qwen3-0.6b": "qwen3-0.6b-untuned", "qwen3-1.7b": "qwen3-1.7b-untuned"}
ALLOCATION_ERROR = re.compile(r"malloc|alloc|out of memory|insufficient memory", re.I)


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- system sampling (macOS)


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def swap_used() -> int:
    return parse_swapusage(_run("sysctl", "-n", "vm.swapusage"))["used_bytes"]


def pressure_level() -> int:
    return int(_run("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip())


def system_snapshot() -> dict[str, Any]:
    swap = parse_swapusage(_run("sysctl", "-n", "vm.swapusage"))
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "swap": swap,
        "pressure_level": pressure_level(),
        "memory_free_percentage": parse_free_percentage(mp),
        "power_source": _power_source(),
    }


def _power_source() -> str:
    first = _run("pmset", "-g", "batt").splitlines()[0]
    m = re.search(r"'([^']+)'", first)
    return m.group(1) if m else first.strip()


def rss_peak_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def environment() -> dict[str, Any]:
    import mlx.core as mx

    return {
        "runner_version": RUNNER_VERSION,
        "git": git_state(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "hardware_model": _run("sysctl", "-n", "hw.model").strip(),
        "cpu": _run("sysctl", "-n", "machdep.cpu.brand_string").strip(),
        "memory_bytes": int(_run("sysctl", "-n", "hw.memsize")),
        "packages": {
            p: version(p) for p in ("mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers")
        },
        "mlx_device": str(mx.default_device()),
        "mlx_device_info": {
            k: v
            for k, v in mx.device_info().items()
            if k in ("device_name", "memory_size", "max_recommended_working_set_size")
        },
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


# ---------------------------------------------------------------- model


def check_model_dir(model: str, model_dir: Path) -> dict[str, Any]:
    """Provenance and sizes must match the pinned checkpoint (hashes: ``verify``)."""
    spec = CHECKPOINTS[model]
    prov = json.loads((model_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    if prov["repo_id"] != spec["repo_id"] or prov["revision"] != spec["revision"]:
        raise SystemExit("provisioned directory is not the pinned checkpoint")
    if prov["tokenizer_only"]:
        raise SystemExit("provisioned directory has no weights")
    for name, (size, sha) in spec["weights"].items():
        rec = prov["files"][name]
        if rec["bytes"] != size or (sha and rec["sha256"] != sha):
            raise SystemExit(f"provenance mismatch for {name}")
        if (model_dir / name).stat().st_size != size:
            raise SystemExit(f"size mismatch on disk for {name}")
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "files": {
            k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in prov["files"].items()
        },
    }


def cmd_verify(args) -> int:
    spec = CHECKPOINTS[args.model]
    model_dir = Path(args.model_dir)
    check_model_dir(args.model, model_dir)
    out = {}
    for name, (_size, sha) in spec["weights"].items():
        h = hashlib.sha256()
        with open(model_dir / name, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        digest = h.hexdigest()
        ok = sha is None or digest == sha
        out[name] = {"sha256": digest, "expected": sha, "match": ok}
        if not ok:
            json.dump(out, sys.stdout, indent=2)
            raise SystemExit(f"SHA-256 mismatch for {name}")
    json.dump(
        {"model": args.model, "revision": spec["revision"], "files": out},
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


class Router:
    def __init__(self, model_dir: Path) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(str(model_dir))
        mx.synchronize()
        self.load_seconds = time.perf_counter() - t0
        self.load_mlx_peak_bytes = mx.get_peak_memory()
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_token_ids = sorted(self.tokenizer.eos_token_ids)
        if im_end not in self.eos_token_ids:
            raise SystemExit("the end-of-turn token is not a stop token")
        self.end_of_turn_id = im_end

    def prompt_ids(self, router_input) -> list[int]:
        return self.tokenizer.apply_chat_template(
            chat_messages(router_input), add_generation_prompt=True, enable_thinking=False
        )

    def generate(self, router_input) -> dict[str, Any]:
        from mlx_lm import stream_generate

        ids = self.prompt_ids(router_input)
        text, last = "", None
        for resp in stream_generate(self.model, self.tokenizer, ids, max_tokens=MAX_NEW_TOKENS):
            text += resp.text
            last = resp
        return {
            "raw_output": text,
            "prompt_tokens": len(ids),
            "generation_tokens": last.generation_tokens,
            "finish_reason": last.finish_reason,
            "generation_tps": last.generation_tps,
            "prompt_tps": last.prompt_tps,
        }


def one_request(router: Router, example: dict[str, Any], system: str) -> dict[str, Any]:
    """Mirror of ``axbench.evaluate.run_router`` plus generation metadata.

    An allocation error stops the run (protocol stop rule); other exceptions become ``error``.
    """
    view = project_columns([(c["name"], c["type"]) for c in example["schema"]["columns"]])
    record: dict[str, Any] = {"example_id": example["id"], "system": system}
    try:
        router_input = build_router_input(example["request"], view)
    except RouterInputError as exc:
        return {
            **record,
            "status": "error",
            "raw_output": None,
            "elapsed_ms": None,
            "error": exc.code,
        }
    start = time.perf_counter()
    try:
        meta = router.generate(router_input)
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        if ALLOCATION_ERROR.search(str(exc)):
            raise StopRun(f"allocation error: {type(exc).__name__}: {exc}") from exc
        meta, status = {"raw_output": None}, "error"
        record["error"] = type(exc).__name__
    record.update(status=status, elapsed_ms=(time.perf_counter() - start) * 1000.0, **meta)
    return record


def _jsonl(records) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ).encode("utf-8")


def _write_json(path: Path, obj) -> None:
    path.write_bytes(
        (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    )


def run_phase(router, examples, system, watch: StopWatch, sink: list, extra=None) -> str | None:
    for e in examples:
        rec = one_request(router, e, system)
        if extra:
            rec.update(extra)
        sink.append(rec)
        tps = rec.get("generation_tps") or 0.0
        reason = watch.check(throughput=tps, swap_used=swap_used(), pressure_level=pressure_level())
        if reason:
            return reason
    return None


def _prepare(args, make_out: bool = True):
    if git_state()["dirty"]:
        raise SystemExit("refusing to run on a dirty tree (tracked files modified)")
    power = _power_source()
    if power != "AC Power" and not args.allow_battery:
        raise SystemExit(f"protocol requires mains power; power source is {power!r}")
    observed = hashlib.sha256((CORPUS / "MANIFEST.json").read_bytes()).hexdigest()
    if observed != MANIFEST_SHA256:
        raise SystemExit("frozen corpus manifest does not match the pinned hash")
    manifest, corpus = load_corpus(CORPUS)
    if not make_out:
        return manifest, corpus, None
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    return manifest, corpus, out


def cmd_main(args) -> int:
    manifest, corpus, out = _prepare(args)
    if any(out.glob("main_*")):
        raise SystemExit("main run outputs already exist; a rerun needs a new run directory")
    system = SYSTEM_NAMES[args.model]
    model_info = check_model_dir(args.model, Path(args.model_dir))
    val = corpus["validation"]
    train_by_id = {e["id"]: e for e in corpus["train"]}
    warm = [train_by_id[i] for i in warmup_ids(corpus["train"])]
    rep_ids = repeat_subset(val)
    val_by_id = {e["id"]: e for e in val}

    meas: dict[str, Any] = {
        "experiment": "EXP-009",
        "phase": "main",
        "system": system,
        "model": model_info,
        "environment": environment(),
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": MANIFEST_SHA256,
        "prompt_version": PROMPT_VERSION,
        "decoding": {
            "sampler": "greedy (argmax)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
            "batch_size": 1,
        },
        "warmup_ids": [e["id"] for e in warm],
        "repeat_ids": rep_ids,
        "before": system_snapshot(),
        "status": "running",
        "validation_completed": False,
    }
    if meas["before"]["power_source"] != "AC Power":
        meas["deviation_power"] = meas["before"]["power_source"]
    warm_recs: list = []
    val_recs: list = []
    rep_recs: list = []
    import mlx.core as mx

    try:
        router = Router(Path(args.model_dir))
        meas["cold_load_seconds"] = router.load_seconds
        meas["load_mlx_peak_bytes"] = router.load_mlx_peak_bytes
        meas["eos_token_ids"] = router.eos_token_ids
        meas["end_of_turn_id"] = router.end_of_turn_id
        meas["rss_peak_after_load_bytes"] = rss_peak_bytes()
        for e in warm:
            warm_recs.append(one_request(router, e, system))
        meas["after_warmup"] = system_snapshot()

        mx.reset_peak_memory()
        watch = StopWatch(swap_used_before=swap_used())
        reason = run_phase(router, val, system, watch, val_recs)
        meas["validation_watch"] = watch.summary()
        meas["validation_mlx_peak_bytes"] = mx.get_peak_memory()
        meas["rss_peak_after_validation_bytes"] = rss_peak_bytes()
        meas["after_validation"] = system_snapshot()
        if reason:
            raise StopRun(f"validation run: {reason}")
        meas["validation_completed"] = len(val_recs) == len(val)

        mx.reset_peak_memory()
        watch = StopWatch(swap_used_before=swap_used())
        reps = [val_by_id[i] for i in rep_ids]
        for r in range(1, REPEATS_IN_PROCESS + 1):
            reason = run_phase(router, reps, system, watch, rep_recs, {"repeat": r})
            if reason:
                break
        meas["repeat_watch"] = watch.summary()
        meas["repeat_mlx_peak_bytes"] = mx.get_peak_memory()
        if reason:
            raise StopRun(f"in-process repeatability: {reason}")
        meas["status"] = "completed"
    except StopRun as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
    finally:
        meas["rss_peak_process_bytes"] = rss_peak_bytes()
        meas["after"] = system_snapshot()
        (out / "main_warmup.jsonl").write_bytes(_jsonl(warm_recs))
        (out / "main_validation.jsonl").write_bytes(_jsonl(val_recs))
        (out / "main_repeat.jsonl").write_bytes(_jsonl(rep_recs))
        meas["counts"] = {
            "warmup": len(warm_recs),
            "validation": len(val_recs),
            "repeat": len(rep_recs),
        }
        _write_json(out / "main_measurements.json", meas)
    print(
        json.dumps(
            {
                "status": meas["status"],
                "stop_reason": meas.get("stop_reason"),
                "counts": meas["counts"],
            }
        )
    )
    return 0 if meas["status"] == "completed" else 3


def cmd_fresh(args) -> int:
    manifest, corpus, out = _prepare(args)
    stem = f"fresh_{args.index}"
    if (out / f"{stem}.jsonl").exists():
        raise SystemExit("fresh run output already exists")
    system = SYSTEM_NAMES[args.model]
    check_model_dir(args.model, Path(args.model_dir))
    val_by_id = {e["id"]: e for e in corpus["validation"]}
    reps = [val_by_id[i] for i in repeat_subset(corpus["validation"])]
    meas: dict[str, Any] = {
        "experiment": "EXP-009",
        "phase": stem,
        "system": system,
        "environment": environment(),
        "before": system_snapshot(),
        "status": "running",
    }
    recs: list = []
    try:
        router = Router(Path(args.model_dir))
        meas["cold_load_seconds"] = router.load_seconds
        watch = StopWatch(swap_used_before=swap_used())
        reason = run_phase(router, reps, system, watch, recs, {"fresh_process": args.index})
        meas["watch"] = watch.summary()
        if reason:
            raise StopRun(reason)
        meas["status"] = "completed"
    except StopRun as exc:
        meas["status"] = "stopped"
        meas["stop_reason"] = str(exc)
    finally:
        meas["rss_peak_process_bytes"] = rss_peak_bytes()
        meas["after"] = system_snapshot()
        (out / f"{stem}.jsonl").write_bytes(_jsonl(recs))
        _write_json(out / f"{stem}_measurements.json", meas)
    print(json.dumps({"status": meas["status"], "n": len(recs)}))
    return 0 if meas["status"] == "completed" else 3


def cmd_smoke(args) -> int:
    """Harness check on the first train examples only (never validation); prints, writes nothing."""
    _, corpus, _ = _prepare(args, make_out=False)
    router = Router(Path(args.model_dir))
    recs = [one_request(router, e, "smoke") for e in corpus["train"][: args.n]]
    summary = {
        "cold_load_seconds": router.load_seconds,
        "eos_token_ids": router.eos_token_ids,
        "end_of_turn_id": router.end_of_turn_id,
        "requests": [
            {
                k: r.get(k)
                for k in (
                    "example_id",
                    "status",
                    "elapsed_ms",
                    "prompt_tokens",
                    "generation_tokens",
                    "finish_reason",
                    "generation_tps",
                )
            }
            for r in recs
        ],
    }
    if args.show_text:
        summary["raw_outputs"] = [r.get("raw_output") for r in recs]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "smoke", "main", "fresh"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
        p.add_argument("--model-dir", required=True)
        if name in ("main", "fresh"):
            p.add_argument("--out", required=True)
        if name != "verify":
            p.add_argument(
                "--allow-battery",
                action="store_true",
                help="harness checks only; a protocol run requires mains power",
            )
        if name == "smoke":
            p.add_argument("--n", type=int, default=3)
            p.add_argument("--show-text", action="store_true")
        if name == "fresh":
            p.add_argument("--index", type=int, required=True, choices=range(1, 6))
    args = parser.parse_args()
    commands = {"verify": cmd_verify, "smoke": cmd_smoke, "main": cmd_main, "fresh": cmd_fresh}
    return commands[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
