# EXP-010 protocol (proposed, not run): LoRA smoke and pilot for AX-01

- Status: **proposed**. Starts only after EXP-009 has selected a base under its frozen rule, the frozen corpus exists
  (revision 3: the owner-authorized AI-reviewed exploratory corpus `axbench/0.1-ai-reviewed`, not human-validated),
  the owner has approved the checkpoint download, and the technical lead has reviewed this protocol. Phase 3B; nothing
  has been run. Pilot results on this corpus are exploratory and are not presented as human-validated or
  population-representative.
- Issue: #4 (protocol). Decision record: ADR 0012. Lead starting point: DR-003 review (REF-031).
- Phase 3C (2026-11-26): EXP-009 selected Qwen3-1.7B and the selection was accepted (PR #10). The operational
  definitions, measured inputs (export hashes, token lengths, `max_seq_length` 1024, 458,752 trainable parameters),
  the installed MLX-LM findings and the pre-registered pilot criteria and gates are in the pre-run record
  [PRERUN.md](PRERUN.md), proposed for technical review. The protocol text below is unchanged (revision 3); nothing
  has been trained.

## Fixed inputs

- Base: the model chosen by EXP-009 at its pinned revision, BF16, provisioned and hash-verified.
- Data: `python -m axbench export-mlx --corpus <frozen corpus directory> --out <directory outside the repository>`
  from the same frozen corpus as EXP-009, `axbench/0.1-ai-reviewed` (owner-authorized AI review, exploratory, 0 human
  decisions; 400 train and 160 validation; revision 3), chat format: system, user JSON, assistant target = compact gold JSON. The
  test split is never exported. The corpus version, its manifest SHA-256 and the SHA-256 of both exported files are
  recorded. (Revision 2, PR #8 review response: the export reads the verified frozen files, not the templates.)
- Template: the same rendering as inference (`enable_thinking=False`); `token_lengths.py` must show that every
  training rendering starts with the inference prompt and report prompt/target/full token counts before training.
  Loss on the completion only; no chain-of-thought targets. The installed MLX-LM 0.32.2 chat dataset computes the
  `--mask-prompt` boundary with thinking enabled, so the template's empty think block would be trained as part of the
  completion (ADR 0012, finding 2). Before the smoke, render both boundaries with the pinned tokenizer, record the
  token difference, and either train with an explicit prompt/completion rendering that uses `enable_thinking=False`
  for both, or record that the empty think block is part of the target.
- Maximum sequence length: the smallest power of two at least the measured maximum full length of train and
  validation; if that is above 1024, stop and report. No example may be truncated; an over-length example is a
  stop condition, not silently dropped.

## Configuration (provisional starting point from DR-003; one run, seed 2026)

LoRA rank 8 on the query and value projections of the last 8 transformer blocks, explicit scale as set in the
installed MLX-LM configuration (the value is recorded; it is not assumed to equal an alpha), dropout 0, learning rate
1e-5 (Adam as in the installed trainer), micro-batch 1, gradient accumulation 4, gradient checkpointing only if the
smoke shows memory pressure. Before the smoke, the exact option and configuration names are verified against the
installed `mlx_lm` 0.32.2 source and recorded in the manifest; nothing is copied from example configurations unread.
Read so far (REF-035): `--num-layers`, `--batch-size`, `--grad-accumulation-steps`, `--learning-rate`, `--mask-prompt`,
`--max-seq-length`, `--grad-checkpoint`, `--seed`, `--adapter-path`, `--save-every`, `--steps-per-eval`,
`--val-batches`; configuration `lora_parameters` with `rank`, `scale`, `dropout` and optional `keys` (without `keys`
every linear layer of the last `num_layers` blocks is adapted, so `keys` must name the query and value projections).
Defaults that must be overridden: 16 layers, batch 4, seed 0, maximum length 2048. `--report-to` stays unset.

1. **Smoke:** 20 optimizer updates (80 micro-steps); record actual micro-steps and updates, loss finiteness, peak
   memory, steps per second; save the adapter, reload it in a fresh process, run real inference on 5 train examples.
   Not an accuracy result.
2. **Pilot:** 2 epochs over the 400 train examples (200 optimizer updates), validation loss every 25 updates on all
   160 validation examples, adapter saved every 50 updates. The adapter at the end of training is evaluated; no
   checkpoint picking on validation exact match within this experiment.
3. **Evaluation:** the EXP-009 harness on validation with the same prompt, template, decoding and parser, paired with
   the same base untuned (EXP-009) and with the rules; trainable parameter count, wall time and loss curves recorded.

## Resource limits and stop rules (MacBook Air M5, 16 GB, mains power, other applications closed)

Provisional targets: process peak at most 10 GB, keeping 4-6 GB for the system. Measured separately and never summed:
MLX peak memory, process RSS, system swap usage and memory pressure (sampled every 30 s by a separate shell loop),
steps per second over windows of 10 updates. Stop on: allocation error; non-finite loss; memory pressure "critical";
swap growth above 2 GB; throughput falling below half of the first window for three consecutive windows. A stopped
run is recorded as failed with its logs; any change (for example gradient checkpointing or the smaller base) is a new
run with its reason logged. At most two predeclared validation-only changes; no sweep.

## Not claimed

Loss is not accuracy. One run supports checkpoint-specific statements only. No training-determinism claim. If the
adapter does not beat its own untuned base by the provisional margin, that is the result.
