"""EXP-009 model runner: untuned Qwen3 inference on AXBENCH validation with resource measurement.

Model environment only (ADR 0012). Run from ``envs/model`` on a clean checkout of the reviewed
commit that contains the EXP-009 pre-run record, on mains power with other applications closed:

    uv run --locked python exp009_run.py verify --model qwen3-0.6b --model-dir <dir>
    uv run --locked python exp009_run.py main   --model qwen3-0.6b --model-dir <dir> \\
        --out <run dir> --expected-commit <sha>
    uv run --locked python exp009_run.py fresh  --model qwen3-0.6b --model-dir <dir> \\
        --out <run dir> --expected-commit <sha> --index 1

* ``verify``: SHA-256 of every provisioned file against the pinned and recorded hashes.
* ``smoke``: harness check on the first train examples only; prints, writes nothing.
* ``check``: the full ``main`` process frame on a few train examples, written outside the
  repository (harness check only; never validation).
* ``main``: one fresh process: preflight, re-hash of every provisioned file, cold load, five train
  warm-ups (never scored), the 160 validation requests in file order, then the 30 stratified
  repeatability requests 20 times in the same process.
* ``fresh``: one fresh process: preflight, re-hash, cold load, the 30 repeatability requests once.

Inference: ``HF_HUB_OFFLINE=1`` (never downloads), prompt ``axiom.prompt/1``, the pinned chat
template with ``enable_thinking=False`` and the generation prompt, greedy decoding (the MLX-LM
default sampler is argmax), at most 256 new tokens, generation stops on the checkpoint's EOS set,
asserted to be exactly {151643, 151645} (D1), batch size 1, no constrained decoding, no output
repair: the raw generated text is written as is. Every record is appended and flushed to disk as
soon as it exists, and the measurement file is rewritten at every phase boundary, so a crash
leaves its partial evidence with status ``running`` (never ``completed``).

Safeguards (pre-run review, PRERUN revision 1): pinned commit and clean tree (a git failure
refuses), mains power and normal memory pressure before every process (no battery override for
protocol runs), one swap baseline per process sampled before the model loads, resource checks
after loading, after every warm-up and after every request, ``MemoryError`` and allocation
failures stop the run (at load or generation), any other unexpected failure marks the process
``failed``. Never reads, generates or scores the final test split. Writes no absolute paths.
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

from provision import CHECKPOINTS, CONFIG_SHA256, KNOWN_SHA256  # noqa: E402

from axbench.corpus import load_corpus  # noqa: E402
from axbench.runwatch import (  # noqa: E402
    ProcessMonitor,
    SlowdownWatch,
    is_allocation_error,
    parse_free_percentage,
    parse_swapusage,
    preflight,
    repeat_subset,
    run_requests,
    timed_call,
    warmup_ids,
)
from axiom_router.prompt import (  # noqa: E402
    PROMPT_VERSION,
    RouterInputError,
    build_router_input,
    chat_messages,
)
from axiom_router.schema import project_columns  # noqa: E402

RUNNER_VERSION = "exp009.runner/2"
CORPUS = REPO / "data" / "axbench" / "v0.1-ai-reviewed"
MANIFEST_SHA256 = "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"
MAX_NEW_TOKENS = 256
REPEATS_IN_PROCESS = 20
EXPECTED_EOS = [151643, 151645]  # <|endoftext|>, <|im_end|> (D1)
SYSTEM_NAMES = {"qwen3-0.6b": "qwen3-0.6b-untuned", "qwen3-1.7b": "qwen3-1.7b-untuned"}


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- system sampling (macOS)


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def power_source() -> str:
    first = _run("pmset", "-g", "batt").splitlines()[0]
    m = re.search(r"'([^']+)'", first)
    return m.group(1) if m else first.strip()


def sample() -> dict[str, Any]:
    """The values the stop rules need (cheap; taken after every request)."""
    return {
        "swap_used": parse_swapusage(_run("sysctl", "-n", "vm.swapusage"))["used_bytes"],
        "pressure_level": int(_run("sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
        "power_source": power_source(),
    }


def snapshot() -> dict[str, Any]:
    """A fuller reading for phase boundaries (adds swap totals and the free percentage)."""
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **sample(),
        "swap": parse_swapusage(_run("sysctl", "-n", "vm.swapusage")),
        "memory_free_percentage": parse_free_percentage(mp),
    }


def rss_peak_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"git {args[0]} failed; refusing to run")
        return proc.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status)}


def environment(git: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    return {
        "runner_version": RUNNER_VERSION,
        "git": git,
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


# ---------------------------------------------------------------- provisioned files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_dir(model: str, model_dir: Path) -> dict[str, Any]:
    """Provenance must name the pinned checkpoint; every file is re-hashed now.

    Each file must equal its ``PROVENANCE.json`` hash, and the pinned hash where one exists
    (weights, ``config.json``, tokenizer files). Refuses on any difference.
    """
    spec = CHECKPOINTS[model]
    prov = json.loads((model_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    if prov["repo_id"] != spec["repo_id"] or prov["revision"] != spec["revision"]:
        raise SystemExit("provisioned directory is not the pinned checkpoint")
    if prov["tokenizer_only"]:
        raise SystemExit("provisioned directory has no weights")
    pinned = {**KNOWN_SHA256, "config.json": CONFIG_SHA256[model]}
    pinned |= {name: sha for name, (_, sha) in spec["weights"].items() if sha}
    missing = [n for n in [*spec["weights"], *pinned] if n not in prov["files"]]
    if missing:
        raise SystemExit(f"provenance lacks {missing}")
    files = {}
    for name, rec in sorted(prov["files"].items()):
        path = model_dir / name
        digest = _sha256(path)
        size = path.stat().st_size
        if digest != rec["sha256"] or size != rec["bytes"]:
            raise SystemExit(f"{name} differs from its provenance record")
        if name in pinned and digest != pinned[name]:
            raise SystemExit(f"{name} differs from the pinned hash")
        if name in spec["weights"] and size != spec["weights"][name][0]:
            raise SystemExit(f"{name} differs from the pinned size")
        files[name] = {
            "bytes": size,
            "sha256": digest,
            "checked_against": "pinned and provenance" if name in pinned else "provenance",
        }
    return {"repo_id": spec["repo_id"], "revision": spec["revision"], "files": files}


def cmd_verify(args) -> int:
    t0 = time.perf_counter()
    info = verify_model_dir(args.model, Path(args.model_dir))
    info["seconds"] = time.perf_counter() - t0
    json.dump(info, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------- model


class Router:
    def __init__(self, model_dir: Path) -> None:
        import mlx.core as mx
        from mlx_lm import load

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(str(model_dir))
        mx.synchronize()
        self.load_seconds = time.perf_counter() - t0
        self.load_mlx_peak_bytes = mx.get_peak_memory()
        self.eos_token_ids = sorted(self.tokenizer.eos_token_ids)
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.eos_token_ids != EXPECTED_EOS or self.end_of_turn_id != 151645:
            raise SystemExit(f"stop-token set {self.eos_token_ids} is not the reviewed D1 set")

    def generate(self, router_input) -> dict[str, Any]:
        from mlx_lm import stream_generate

        ids = self.tokenizer.apply_chat_template(
            chat_messages(router_input), add_generation_prompt=True, enable_thinking=False
        )
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


def make_request(router: Router, system: str, extra: dict[str, Any] | None = None):
    def request(example: dict[str, Any]) -> dict[str, Any]:
        view = project_columns([(c["name"], c["type"]) for c in example["schema"]["columns"]])
        record: dict[str, Any] = {"example_id": example["id"], "system": system, **(extra or {})}
        try:
            router_input = build_router_input(example["request"], view)
        except RouterInputError as exc:
            return {**record, "status": "error", "raw_output": None, "elapsed_ms": None,
                    "error": exc.code}  # fmt: skip
        return timed_call(record, lambda: router.generate(router_input))

    return request


# ---------------------------------------------------------------- output


class Sink:
    """Append-only JSONL, created exclusively, flushed and synced after every record."""

    def __init__(self, path: Path) -> None:
        self.fh = open(path, "xb")  # noqa: SIM115 - closed in close()
        self.count = 0

    def emit(self, record: dict[str, Any]) -> None:
        self.fh.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode())
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.count += 1

    def close(self) -> None:
        self.fh.close()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())
    os.replace(tmp, path)


# ---------------------------------------------------------------- processes


def _prepare(args, stem: str):
    """Commit, corpus, output directory and preflight. Returns None after a refusal."""
    git = git_state()
    if git["dirty"]:
        raise SystemExit("refusing to run on a dirty tree (tracked files modified)")
    if git["commit"] != args.expected_commit:
        raise SystemExit("HEAD is not the reviewed commit given with --expected-commit")
    observed = hashlib.sha256((CORPUS / "MANIFEST.json").read_bytes()).hexdigest()
    if observed != MANIFEST_SHA256:
        raise SystemExit("frozen corpus manifest does not match the pinned hash")
    manifest, corpus = load_corpus(CORPUS)
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob(f"{stem}*")):
        raise SystemExit(f"{stem} outputs already exist; a rerun needs a new run directory")
    pre = snapshot()
    refusal = preflight(pre)
    if refusal:
        with open(out / "preflight_refusals.jsonl", "ab") as fh:
            line = {"process": stem, "reason": refusal, "snapshot": pre, "commit": git["commit"]}
            fh.write((json.dumps(line, sort_keys=True) + "\n").encode())
        print(json.dumps({"status": "refused", "reason": refusal}))
        return None
    return git, manifest, corpus, out, pre


def run_process(args, stem: str, body) -> int:
    """Common frame for ``main`` and ``fresh``: measurements, statuses and failure handling."""
    prepared = _prepare(args, stem)
    if prepared is None:
        return 4
    git, manifest, corpus, out, pre = prepared
    meas: dict[str, Any] = {
        "experiment": "EXP-009",
        "phase": stem,
        "system": SYSTEM_NAMES[args.model],
        "status": "running",
        "preflight": pre,
        "environment": environment(git),
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": MANIFEST_SHA256,
        "prompt_version": PROMPT_VERSION,
        "decoding": {
            "sampler": "greedy (argmax)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
            "batch_size": 1,
            "stop_token_ids": EXPECTED_EOS,
        },
    }
    mpath = out / f"{stem}_measurements.json"
    t0 = time.perf_counter()
    meas["model"] = verify_model_dir(args.model, Path(args.model_dir))
    meas["model_rehash_seconds"] = time.perf_counter() - t0
    # One swap baseline per process, before the model loads (D8 revision 1).
    monitor = ProcessMonitor(baseline={k: pre[k] for k in ("swap_used", "pressure_level",
                                                            "power_source")})  # fmt: skip
    write_json(mpath, meas)
    sinks: dict[str, Sink] = {}
    try:
        try:
            router = Router(Path(args.model_dir))
        except Exception as exc:
            if is_allocation_error(exc):
                raise StopRun(f"load: allocation error: {type(exc).__name__}: {exc}") from exc
            raise
        meas.update(
            cold_load_seconds=router.load_seconds,
            load_mlx_peak_bytes=router.load_mlx_peak_bytes,
            eos_token_ids=router.eos_token_ids,
            end_of_turn_id=router.end_of_turn_id,
            rss_peak_after_load_bytes=rss_peak_bytes(),
        )
        reason = monitor.check("load", sample())
        write_json(mpath, meas)
        if reason:
            raise StopRun(reason)
        body(router, corpus, out, meas, mpath, monitor, sinks)
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


def _phase(name, items, request, out, stem, monitor, sinks, slowdown=None) -> None:
    if name not in sinks:
        sinks[name] = Sink(out / f"{stem}_{name}.jsonl")
    sink = sinks[name]
    reason = run_requests(items, request, sample, monitor, name, sink.emit, slowdown)
    if reason:
        raise StopRun(reason)


def cmd_main(args) -> int:
    def body(router, corpus, out, meas, mpath, monitor, sinks):
        import mlx.core as mx

        system = SYSTEM_NAMES[args.model]
        val = corpus["validation"]
        train_by_id = {e["id"]: e for e in corpus["train"]}
        warm = [train_by_id[i] for i in warmup_ids(corpus["train"])]
        rep_ids = repeat_subset(val)
        val_by_id = {e["id"]: e for e in val}
        meas["warmup_ids"] = [e["id"] for e in warm]
        meas["repeat_ids"] = rep_ids
        meas["validation_completed"] = False

        _phase("warmup", warm, make_request(router, system), out, "main", monitor, sinks)
        meas["after_warmup"] = snapshot()
        write_json(mpath, meas)

        mx.reset_peak_memory()
        slow = SlowdownWatch()
        try:
            _phase("validation", val, make_request(router, system), out, "main", monitor, sinks,
                   slow)  # fmt: skip
        finally:
            meas["validation_slowdown"] = slow.summary()
            meas["validation_mlx_peak_bytes"] = mx.get_peak_memory()
            meas["rss_peak_after_validation_bytes"] = rss_peak_bytes()
            meas["after_validation"] = snapshot()
            write_json(mpath, meas)
        meas["validation_completed"] = sinks["validation"].count == len(val)

        mx.reset_peak_memory()
        slow = SlowdownWatch()
        reps = [val_by_id[i] for i in rep_ids]
        try:
            for r in range(1, REPEATS_IN_PROCESS + 1):
                _phase("repeat", reps, make_request(router, system, {"repeat": r}), out, "main",
                       monitor, sinks, slow)  # fmt: skip
        finally:
            meas["repeat_slowdown"] = slow.summary()
            meas["repeat_mlx_peak_bytes"] = mx.get_peak_memory()
            write_json(mpath, meas)

    return run_process(args, "main", body)


def cmd_fresh(args) -> int:
    stem = f"fresh_{args.index}"

    def body(router, corpus, out, meas, mpath, monitor, sinks):
        system = SYSTEM_NAMES[args.model]
        val_by_id = {e["id"]: e for e in corpus["validation"]}
        rep_ids = repeat_subset(corpus["validation"])
        meas["repeat_ids"] = rep_ids
        reps = [val_by_id[i] for i in rep_ids]
        request = make_request(router, system, {"fresh_process": args.index})
        _phase("repeat", reps, request, out, stem, monitor, sinks)

    return run_process(args, stem, body)


def cmd_check(args) -> int:
    """Harness check of the full process frame on train examples only, into a directory outside
    the repository: preflight, re-hash, load, resource checks, sinks, measurement file. Never
    touches validation; not evidence of any result."""
    if REPO in Path(args.out).resolve().parents:
        raise SystemExit("check outputs must be outside the repository")

    def body(router, corpus, out, meas, mpath, monitor, sinks):
        train = corpus["train"]
        request = make_request(router, "check")
        _phase("warmup", train[:2], request, out, "check", monitor, sinks)
        _phase("validation", train[2:6], request, out, "check", monitor, sinks, SlowdownWatch())
        for r in (1, 2):
            req = make_request(router, "check", {"repeat": r})
            _phase("repeat", train[2:4], req, out, "check", monitor, sinks, SlowdownWatch())

    return run_process(args, "check", body)


def cmd_smoke(args) -> int:
    """Harness check on the first train examples only (never validation); prints, writes nothing."""
    _, corpus = load_corpus(CORPUS)
    router = Router(Path(args.model_dir))
    request = make_request(router, "smoke")
    recs = [request(e) for e in corpus["train"][: args.n]]
    keys = ("example_id", "status", "elapsed_ms", "prompt_tokens", "generation_tokens",
            "finish_reason", "generation_tps")  # fmt: skip
    summary = {
        "power_source": power_source(),
        "cold_load_seconds": router.load_seconds,
        "eos_token_ids": router.eos_token_ids,
        "requests": [{k: r.get(k) for k in keys} for r in recs],
    }
    if args.show_text:
        summary["raw_outputs"] = [r.get("raw_output") for r in recs]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "smoke", "check", "main", "fresh"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
        p.add_argument("--model-dir", required=True)
        if name in ("check", "main", "fresh"):
            p.add_argument("--out", required=True)
            p.add_argument("--expected-commit", required=True)
        if name == "smoke":
            p.add_argument("--n", type=int, default=3)
            p.add_argument("--show-text", action="store_true")
        if name == "fresh":
            p.add_argument("--index", type=int, required=True, choices=range(1, 6))
    args = parser.parse_args()
    commands = {
        "verify": cmd_verify,
        "smoke": cmd_smoke,
        "check": cmd_check,
        "main": cmd_main,
        "fresh": cmd_fresh,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
