"""Measure AXBENCH-TR training-sequence lengths with the real Qwen3-1.7B tokenizer.

Rendering follows the EXP-010 convention: the prompt is the model's own ``chat_template`` applied
to the ``axiom.prompt/1`` system and user turns with ``enable_thinking=False`` and the generation
prompt; the sequence is that prompt, the compact gold JSON, and ``<|im_end|>``. Only the tokenizer
files (SRC-M01, no weights) are read. The report is JSON on stdout; ``--per-example`` writes one
line per record.

``--max-seq-length N`` is a gate, never a truncation: the exit status is 1 if any sequence is longer
than N, and the report lists how many exceed 1024 and N.

Optional dependencies, not in the locked environment: ``tokenizers`` and ``jinja2``.

    PYTHONPATH=src python benchmarks/axbench_tr_token_stats.py --max-seq-length 2048 \
        --per-example data/axbench-tr/v0.2-draft/token_lengths_per_example.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from axbench_tr.build import OUT_ROOT
from axbench_tr.fetch import raw_dir
from axiom_router.prompt import build_router_input, chat_messages
from axiom_router.schema import project_columns

PREVIOUS_LIMIT = 1024  # EXP-010 max_seq_length


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


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
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-example", help="write one JSON line per record to this path")
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

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    report: dict[str, object] = {
        "rendering": "chat_template(enable_thinking=False, generation prompt) + compact gold JSON "
        "+ <|im_end|>; Qwen3-1.7B tokenizer.json",
        "corpus": f"axbench-tr/{args.version.removeprefix('v')}",
        "previous_limit": PREVIOUS_LIMIT,
        "max_seq_length_gate": args.max_seq_length,
    }
    per_example: list[dict[str, object]] = []
    over_gate = 0
    for split in ("train", "validation"):
        path = OUT_ROOT / args.version / f"{split}.jsonl"
        counts: dict[str, list[int]] = {"all": [], "tr": [], "en": []}
        prompts: list[int] = []
        completions: list[int] = []
        for line in path.read_text("utf-8").splitlines():
            rec = json.loads(line)
            view = project_columns([(c["name"], c["type"]) for c in rec["schema"]["columns"]])
            msgs = chat_messages(build_router_input(rec["request"], view))
            target = json.dumps(rec["gold"], ensure_ascii=False, separators=(",", ":"))
            prompt = template.render(
                messages=msgs, add_generation_prompt=True, enable_thinking=False
            )
            n_prompt = count(prompt)
            n_completion = count(target) + 1  # + <|im_end|>
            n = n_prompt + n_completion
            counts["all"].append(n)
            counts[rec["language"]].append(n)
            prompts.append(n_prompt)
            completions.append(n_completion)
            over_gate += n > args.max_seq_length
            per_example.append(
                {
                    "id": rec["id"],
                    "split": split,
                    "language": rec["language"],
                    "columns": len(rec["schema"]["columns"]),
                    "prompt_tokens": n_prompt,
                    "completion_tokens": n_completion,
                    "sequence_tokens": n,
                }
            )
        report[split] = {
            "sequence_tokens": _summary(counts["all"]),
            "turkish": _summary(counts["tr"]),
            "english": _summary(counts["en"]),
            "prompt_tokens": _summary(prompts),
            "completion_tokens": _summary(completions),
            "over_1024": sum(c > PREVIOUS_LIMIT for c in counts["all"]),
            f"over_{args.max_seq_length}": sum(c > args.max_seq_length for c in counts["all"]),
        }
    longest = sorted(per_example, key=lambda r: -int(r["sequence_tokens"]))[:10]
    report["longest_ten"] = longest
    report["examples_over_gate"] = over_gate
    if args.per_example:
        lines = (json.dumps(r, sort_keys=True) for r in per_example)
        Path(args.per_example).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    print()
    return 1 if over_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
