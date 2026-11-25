# ADR 0012: Optional Apple Silicon model environment, pinned checkpoints and experiment protocols

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-26
- Issue / PR: Issue #4 (Phase 3A preparation; execution is Phase 3B); implements the DR-003 review
  ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7168013773))

## Context

AX-01 experiments need a local model stack on the target MacBook Air M5 (16 GB). The lead decisions require: MLX
not mandatory for Core or Windows CI; pinned checkpoint revisions, tokenizer, template, licence and artifact hashes;
explicit provisioning to local paths and offline inference; no weights in Git; resource limits and stop rules; and a
model-selection rule fixed before seeing validation predictions. Nothing may be trained in Phase 3A.

## Alternatives considered

- **Optional dependency group in the main `pyproject.toml`.** Rejected: it changes the Core `uv.lock`, whose digest is
  part of every analysis passport's environment record (ADR 0007), for packages Core never uses, and puts MLX
  resolution into the Windows CI lock.
- **PyTorch / Transformers training stack.** Rejected for now (DR-003): a second heavy stack without evidence that it
  is needed on Apple Silicon.
- **Newest releases** (mlx 0.32.5, published the day before). Not chosen: a two-week cut-off is applied to the model
  lock for supply-chain hygiene.

## Decision

**Environment.** `envs/model/` is a separate uv project (`package = false`) with its own `uv.lock`, restricted to
`sys_platform == 'darwin' and platform_machine == 'arm64'`, depending on `axiom-local` (editable path), `mlx==0.32.4`,
`mlx-lm==0.32.2` and `huggingface-hub`, locked with `exclude-newer = 2026-11-12` except for the Core pins (which are
reviewed in the main lock and are identical in both locks). It is never imported by `axiom`, `axiom_router` or
`axbench` and never installed in CI.

**Installed and checked on the development Mac (2026-11-26, observed once):** `uv sync --locked` in `envs/model`
installed mlx 0.32.4, mlx-metal 0.32.4, mlx-lm 0.32.2, transformers 5.22.2, tokenizers 0.23.3, huggingface-hub 2.4.0;
`mx.default_device()` reported `Device(gpu, 0)`, `mx.metal.is_available()` was true and a small array computation
returned the exact expected value; `mlx_lm.models.qwen3` imports. No checkpoint was loaded.

**Findings from the installed MLX-LM 0.32.2 source** (REF-035), which the protocols must handle explicitly:

1. The tokenizer wrapper passes `enable_thinking=True` to Qwen3's template unless the caller sets it, so every AXIOM
   inference call must pass `enable_thinking=False` explicitly.
2. The chat training dataset renders conversations without `enable_thinking`. With `--mask-prompt`, the prompt length
   is taken from the conversation without the assistant turn rendered *with* thinking enabled, so the empty think block
   that the template writes before the assistant content falls into the trained completion. The rendered training
   text equals the inference prompt followed by the target (from the template), only the loss boundary moves by those
   template tokens. EXP-010 must measure this with the pinned tokenizer and either accept it explicitly or render
   prompt and completion itself; it is not left implicit.
3. Trainer defaults differ from the DR-003 starting point (16 layers, batch 4, seed 0, maximum length 2048, LoRA on
   every linear layer when no `keys` are given, `scale` 20, Adam); every value is set explicitly in EXP-010.
4. `--report-to` (wandb, swanlab) is off by default and must stay unset (no remote trackers).

**Checkpoints** (REF-032, REF-033), both Apache-2.0, `Qwen3ForCausalLM`, 28 layers, BF16, vocabulary 151,936, the same
tokenizer configuration and chat template:

| Model | Revision | Weight files (bytes, SHA-256 prefix) |
|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | `model.safetensors` 1,503,300,328, `f47f7117` |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | two shards 3,441,185,608 + 622,329,984, `169ad53e` / `912becff` |

**Provisioning** (`envs/model/provision.py`): only after the owner approves the download; exact files at the pinned
revision; published SHA-256 verified (refuse on mismatch); other files hashed and recorded in `PROVENANCE.json`;
the target directory must be outside the repository and empty. `.gitignore` blocks `*.safetensors`, `*.gguf`, `*.npz`,
`models/` and `adapters/`.

**Inference configuration (for EXP-009 onwards).** `HF_HUB_OFFLINE=1`, local directory only, never download on
failure; prompt `axiom.prompt/1`; pinned chat template with `enable_thinking=False` (from the template, this renders
an empty think block after the generation prompt, and the model card states that the `/think` and `/no_think` soft
switches in user text are then not valid, so request text cannot turn on thinking); greedy decoding, at most 256 new
tokens, no constrained decoding, no output repair. The model card recommends sampling and warns against greedy
decoding in thinking mode; whether greedy decoding hurts this non-thinking task is measured, not assumed.

**Protocols** (proposed; committed before any model output exists):
[EXP-009](../evidence/experiments/EXP-009-untuned-baselines/PROTOCOL.md) untuned baselines, resources, repeatability
and the frozen base-selection rule; [EXP-010](../evidence/experiments/EXP-010-lora-pilot/PROTOCOL.md) LoRA smoke and
pilot with resource limits and stop rules.

## Tradeoffs and consequences

- Two lock files to maintain; the Core pins must be kept equal by hand (checked in review).
- The environment is Mac-only; a Windows inference path (for example a converted format) needs its own decision and
  an independent re-evaluation.
- Greedy decoding and strict parsing may lower untuned scores; this is intended, so that scores reflect what the
  Guard would receive.

## Unvalidated assumptions

- That MLX-LM 0.32.2 loads and trains these Qwen3 checkpoints as published (to be shown by EXP-009/EXP-010 smoke steps).
- That the 0.6B or 1.7B model fits the provisional 10 GB process peak during LoRA training on 16 GB; not measured.
- That the vendor's multilingual claim covers Turkish adequately for this task; not verified.

## References

REF-031, REF-032, REF-033, REF-034, REF-035
