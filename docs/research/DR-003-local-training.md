# DR-003: Local model training on Apple Silicon

- Status: **Synthesis of the technical lead's published DR-003 notes. Not independent evidence. No training has been
  performed.**
- Related: Issue #4, ADR 0012, [EXP-009 and EXP-010 protocols](../evidence/experiments/README.md).

## Provenance

- Technical lead research note
  ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7166468961)) and DR-003 review
  ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7168013773)), both REF-031. The raw report
  is private and is not quoted here.
- Re-opened by the implementer: the Hugging Face model pages, API metadata, `config.json`, `generation_config.json`,
  `tokenizer_config.json` and `LICENSE` of Qwen3-0.6B and Qwen3-1.7B at pinned revisions (REF-032, REF-033); PyPI
  metadata of `mlx`, `mlx-lm` and `mlx-metal` (REF-034); the installed MLX-LM source (REF-035). Other sources named by
  the lead (LoRA, QLoRA, MLX-LM LoRA guide, MLX-LM issue 1956, Qwen3.5-0.8B) were **not** re-opened and are not
  registered.

## What the lead decided (proposals, not results)

1. AX-01 is an optional bilingual semantic parser (PLAN/CLARIFY/REJECT); no arithmetic, code, data mutation, tools,
   paths, network, credentials or raw rows. A valid but unintended plan is a distinct failure class.
2. Candidates: Qwen/Qwen3-0.6B and Qwen/Qwen3-1.7B (Apache-2.0), compared untuned on validation with the same
   examples, information, decoding and prompt budget; choose the smallest that meets predeclared quality and resource
   gates. No model selection on the final test. No full fine-tuning, no training from scratch. If rules or the untuned
   model win, report it and do not ship a worse adapter.
3. A Mac-only optional environment with pinned MLX and MLX-LM, kept out of the portable Core and the Windows CI.
   Pinned revisions, tokenizer, template, licence and artifact hashes; explicit provisioning; offline local paths; no
   automatic download on failure; no remote trackers; weights and adapters never in Git.
4. Training only after the dataset and contract are approved: a 10-20 update smoke, then a pilot. Provisional
   starting point: LoRA rank 8 on query/value projections of the last 8 blocks, explicit scale, learning rate 1e-5,
   dropout 0, micro-batch 1, gradient accumulation 4, maximum sequence 512 only if measured lengths fit, seed 2026,
   completion-only loss, non-thinking template consistent between training and inference, no chain-of-thought
   targets, greedy decoding with a bounded output.
5. Resource policy on the 16 GB MacBook Air M5: about 10 GB process peak and 4-6 GB headroom as provisional targets;
   measure MLX peak, process memory and system memory pressure/swap separately; stop on sustained pressure, swap,
   slowdown, non-finite loss or allocation errors. No speed or memory claims without measurement.

## What the implementer verified (Issue #4)

- Pinned revisions, file sizes and LFS SHA-256 of both checkpoints, Apache-2.0 licence text, identical tokenizer
  configuration and chat template for both sizes, and the template's `enable_thinking=False` branch, which inserts an
  empty think block (REF-032, REF-033).
- The model card's statement that `/think` and `/no_think` in user text switch modes when thinking is enabled; the
  card says the soft switches are not valid when `enable_thinking=False`. AXIOM must always render with
  `enable_thinking=False` so request text cannot switch modes.
- The card recommends sampling and warns against greedy decoding in thinking mode; for non-thinking mode it suggests
  temperature 0.7. AXIOM uses greedy decoding for repeatability; whether that degrades this task is an open question
  for EXP-009, not an assumption.
- An optional environment `envs/model/` (separate uv project and lock) with `mlx` 0.32.4 and `mlx-lm` 0.32.2; what
  was installed and imported is recorded in ADR 0012.
