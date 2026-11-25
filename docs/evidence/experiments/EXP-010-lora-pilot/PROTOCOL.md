# EXP-010 protocol (proposed, not run): LoRA smoke and pilot for AX-01

- Status: **proposed**. Starts only after EXP-009 has selected a base under its frozen rule, the owner has approved
  the AXBENCH annotation review, and the technical lead has reviewed this protocol. Phase 3B; nothing has been run.
- Issue: #4 (protocol). Decision record: ADR 0012. Lead starting point: DR-003 review (REF-031).

## Fixed inputs

- Base: the model chosen by EXP-009 at its pinned revision, BF16, provisioned and hash-verified.
- Data: `python -m axbench export-mlx --out <directory outside the repository>` (400 train, 160 validation, chat
  format: system, user JSON, assistant target = compact gold JSON). The test split is never exported. The SHA-256 of
  both exported files is recorded.
- Template: the same rendering as inference (`enable_thinking=False`); `token_lengths.py` must show that every
  training rendering starts with the inference prompt and report prompt/target/full token counts before training.
  Loss on the completion only (`--mask-prompt` or the installed equivalent); no chain-of-thought targets.
- Maximum sequence length: the smallest power of two at least the measured maximum full length of train and
  validation; if that is above 1024, stop and report. No example may be truncated; an over-length example is a
  stop condition, not silently dropped.

## Configuration (provisional starting point from DR-003; one run, seed 2026)

LoRA rank 8 on the query and value projections of the last 8 transformer blocks, explicit scale as set in the
installed MLX-LM configuration (the value is recorded; it is not assumed to equal an alpha), dropout 0, learning rate
1e-5 (Adam as in the installed trainer), micro-batch 1, gradient accumulation 4, gradient checkpointing only if the
smoke shows memory pressure. Before the smoke, the exact option and configuration names are verified against the
installed `mlx_lm` 0.32.2 source and recorded in the manifest; nothing is copied from example configurations unread.

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
