"""Measure AXBENCH-TR training-token counts with the real Qwen3-1.7B tokenizer.

Each record is rendered exactly as the MLX export renders it (``axiom.prompt/1`` system and user
turns, compact gold JSON as the assistant turn) through the model's own ``chat_template`` with
``enable_thinking=False``, then tokenised with ``tokenizer.json``. Only the tokenizer files
(SRC-M01, no weights) are read. Output is JSON on stdout.

Optional dependencies, not part of the locked environment: ``tokenizers`` and ``jinja2``.

    PYTHONPATH=src python benchmarks/axbench_tr_token_stats.py [--version v0.2-draft]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from axbench_tr.build import OUT_ROOT
from axbench_tr.fetch import raw_dir
from axiom_router.prompt import build_router_input, chat_messages
from axiom_router.schema import project_columns


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "records": len(values),
        "total_tokens": sum(values),
        "mean": round(statistics.fmean(values), 1),
        "median": int(statistics.median(values)),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0.2-draft")
    args = parser.parse_args()
    try:
        import jinja2
        from tokenizers import Tokenizer
    except ImportError as exc:  # optional, measurement-only dependencies
        print(f"missing optional dependency: {exc.name} (pip install tokenizers jinja2)")
        return 2

    tok_dir = next((raw_dir() / "hf" / "Qwen" / "Qwen3-1.7B").glob("*/tokenizer.json")).parent
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    config = json.loads((tok_dir / "tokenizer_config.json").read_text("utf-8"))

    def _raise(message: str) -> None:
        raise ValueError(message)

    env = jinja2.Environment(loader=jinja2.BaseLoader(), keep_trailing_newline=True)
    env.globals["raise_exception"] = _raise
    template = env.from_string(config["chat_template"])

    report: dict[str, object] = {
        "tokenizer": "Qwen/Qwen3-1.7B tokenizer.json + chat_template, enable_thinking=False",
        "corpus": f"axbench-tr/{args.version.removeprefix('v')}",
    }
    for split in ("train", "validation"):
        path = OUT_ROOT / args.version / f"{split}.jsonl"
        counts: dict[str, list[int]] = {"all": [], "tr": [], "en": []}
        prompt_counts: list[int] = []
        for line in path.read_text("utf-8").splitlines():
            rec = json.loads(line)
            view = project_columns([(c["name"], c["type"]) for c in rec["schema"]["columns"]])
            msgs = chat_messages(build_router_input(rec["request"], view))
            target = json.dumps(rec["gold"], ensure_ascii=False, separators=(",", ":"))
            full = template.render(
                messages=[*msgs, {"role": "assistant", "content": target}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prompt = template.render(
                messages=msgs, add_generation_prompt=True, enable_thinking=False
            )
            n = len(tokenizer.encode(full, add_special_tokens=False).ids)
            counts["all"].append(n)
            counts[rec["language"]].append(n)
            prompt_counts.append(len(tokenizer.encode(prompt, add_special_tokens=False).ids))
        report[split] = {
            "per_example_tokens": _summary(counts["all"]),
            "turkish": _summary(counts["tr"]),
            "english": _summary(counts["en"]),
            "prompt_only_tokens": _summary(prompt_counts),
        }
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
