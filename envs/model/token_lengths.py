"""Token lengths of AXBENCH router prompts and targets under a pinned tokenizer (ADR 0012).

    uv run --locked python token_lengths.py --tokenizer <provisioned directory>

Renders every train and validation example exactly as inference would (system instruction, user JSON
from ``axiom.prompt/1``, chat template with ``enable_thinking=False``, generation prompt) and the
training form (the same messages plus the assistant target), and reports token counts. Also checks
that the training rendering starts with the inference prompt, so the completion boundary is
well defined. Offline only; no model weights are needed. Writes JSON to stdout, no texts.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _summary(values: list[int]) -> dict:
    values = sorted(values)
    q = statistics.quantiles(values, n=100)
    return {
        "n": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p95": q[94],
        "max": values[-1],
    }


def main() -> int:
    from transformers import AutoTokenizer

    from axbench.generate import generate_development_corpus
    from axiom_router.prompt import build_router_input, chat_messages
    from axiom_router.schema import project_columns

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    out: dict = {"tokenizer_class": type(tok).__name__, "splits": {}}
    for split, examples in generate_development_corpus().items():
        prompt_len, target_len, full_len, prefix_ok = [], [], [], 0
        for e in examples:
            r = e.to_record()
            view = project_columns([(c["name"], c["type"]) for c in r["schema"]["columns"]])
            msgs = chat_messages(build_router_input(r["request"], view))
            target = json.dumps(r["gold"], ensure_ascii=False, separators=(",", ":"))
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            full = tok.apply_chat_template(
                [*msgs, {"role": "assistant", "content": target}],
                tokenize=False,
                enable_thinking=False,
            )
            prefix_ok += full.startswith(prompt)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            prompt_len.append(len(p_ids))
            full_len.append(len(f_ids))
            target_len.append(len(f_ids) - len(p_ids))
        out["splits"][split] = {
            "prompt_tokens": _summary(prompt_len),
            "completion_tokens_incl_end": _summary(target_len),
            "full_tokens": _summary(full_len),
            "training_render_starts_with_inference_prompt": f"{prefix_ok}/{len(examples)}",
        }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
