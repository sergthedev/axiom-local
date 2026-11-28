"""Measure AXBENCH-TR training-sequence lengths with the real Qwen3-1.7B tokenizer.

Rendering follows the EXP-010 convention: the prompt is the model's own ``chat_template`` applied
to the ``axiom.prompt/1`` system and user turns with ``enable_thinking=False`` and the generation
prompt; the sequence is that prompt, the compact gold JSON, and ``<|im_end|>``. Only the tokenizer
files (SRC-M01, no weights) are read, and both are verified against the SHA-256 recorded in
``ACQUISITION.json`` before use (the run stops on a mismatch or a missing file).

Three lengths are reported per record, because a length measured on separately encoded pieces is
not the length the trainer sees:

* ``separate``: prompt and target encoded separately and added (the EXP-010 convention);
* ``joint``: the whole rendered sequence encoded once; the gate uses the larger of the two;
* ``template_full``: the conversation (system, user, assistant) rendered by the chat template in
  one pass, as a chat-format trainer would render it.

``boundary_straddles`` counts records where a joint-encoding token crosses the prompt/completion
boundary, i.e. where a loss mask built from the separate prompt length would be misplaced. This is
a tokenizer-level check; the full rendered sequences and masks must still be validated with the
actual MLX trainer on the target machine. The report is JSON on stdout; ``--per-example`` writes
one line per record.

``--max-seq-length N`` is a gate, never a truncation: the exit status is 1 if any sequence is longer
than N, and the report lists how many exceed 1024 and N.

Optional dependencies, not in the locked environment: ``tokenizers`` and ``jinja2``.

    PYTHONPATH=src python benchmarks/axbench_tr_token_stats.py --max-seq-length 2048 \
        --per-example data/axbench-tr/v0.2-rev1/token_lengths_per_example.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from axbench_tr.build import CURRENT_VERSION, OUT_ROOT
from axbench_tr.fetch import raw_dir, sha256_file
from axbench_tr.registry import ACQUISITION_PATH
from axiom_router.prompt import build_router_input, chat_messages
from axiom_router.schema import project_columns

PREVIOUS_LIMIT = 1024  # EXP-010 max_seq_length


def pinned_tokenizer_files() -> tuple[Path, Path, dict[str, str]]:
    """Tokenizer files of SRC-M01, verified against ``ACQUISITION.json`` (never a first glob)."""
    entry = json.loads(ACQUISITION_PATH.read_text("utf-8"))["sources"]["SRC-M01"]["files"]
    found: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for item in entry:
        name = item["path"].rsplit("/", 1)[-1]
        path = raw_dir() / item["path"]
        if not path.is_file():
            raise SystemExit(f"missing pinned tokenizer file: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"tokenizer file differs from ACQUISITION.json: {item['path']}")
        found[name] = path
        hashes[name] = item["sha256"]
    return found["tokenizer.json"], found["tokenizer_config.json"], hashes


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
    parser.add_argument("--version", default=CURRENT_VERSION)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-example", help="write one JSON line per record to this path")
    args = parser.parse_args()
    try:
        import jinja2
        from tokenizers import Tokenizer
    except ImportError as exc:  # optional, measurement-only dependencies
        print(f"missing optional dependency: {exc.name} (pip install tokenizers jinja2)")
        return 2

    tok_path, config_path, tok_hashes = pinned_tokenizer_files()
    tokenizer = Tokenizer.from_file(str(tok_path))
    config = json.loads(config_path.read_text("utf-8"))

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
        "tokenizer_sha256": tok_hashes,
        "previous_limit": PREVIOUS_LIMIT,
        "max_seq_length_gate": args.max_seq_length,
    }
    per_example: list[dict[str, object]] = []
    over_gate = 0
    straddles = 0
    separate_vs_joint = 0
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
            separate = n_prompt + n_completion
            joint_enc = tokenizer.encode(prompt + target + "<|im_end|>", add_special_tokens=False)
            joint = len(joint_enc.ids)
            boundary = len(prompt)
            straddle = any(a < boundary < b for a, b in joint_enc.offsets)
            full = template.render(
                messages=[*msgs, {"role": "assistant", "content": target}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            template_full = count(full)
            straddles += straddle
            separate_vs_joint += separate != joint
            n = max(separate, joint, template_full)
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
                    "separate_tokens": separate,
                    "joint_tokens": joint,
                    "template_full_tokens": template_full,
                    "boundary_straddle": straddle,
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
    report["records_where_separate_and_joint_length_differ"] = separate_vs_joint
    report["boundary_straddles"] = straddles
    report["template_full_longer_than_convention"] = sum(
        int(r["template_full_tokens"]) > int(r["separate_tokens"]) for r in per_example
    )
    if args.per_example:
        lines = (json.dumps(r, sort_keys=True) for r in per_example)
        Path(args.per_example).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    print()
    return 1 if over_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
