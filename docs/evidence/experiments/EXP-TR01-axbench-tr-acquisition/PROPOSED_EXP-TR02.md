# Proposed EXP-TR02: AXBENCH-TR LoRA, feasibility smoke first (proposal only; not approved, not run)

Status: **proposed**. Nothing here starts without the owner's separate approval, and the technical lead's review
of this protocol and of PR #13 comes first. EXP-010 is untouched; this protocol reuses its measured numbers
read from its committed records (`PILOT_TRAINING.md`, `prerun/token_lengths.json`) and changes nothing there.

## Question
Part A (this smoke): can the M5 16 GB train Qwen3-1.7B LoRA on AXBENCH-TR sequences up to 1,219 tokens, with
no truncation, inside the EXP-010 resource limits, and how long would a full pass take? **No accuracy question is
asked and no accuracy is reported.**
Part B (later, separate approval): does adding AXBENCH-TR to v0.1 training improve Turkish routing without
hurting English, against the EXP-010 adapter? Not designed here beyond the arms (A v0.1 only, B v0.1 + TR,
C TR only) and the need for a reviewed validation set.

## Why the smoke is needed (measured facts)
- EXP-010 ran with `max_seq_length` 1024; its longest v0.1 sequence was 514 tokens, MLX peak 4.71 GB (load 3.44 GB),
  swap unchanged, pressure normal, one throughput window 0.86 of the first (EXP-010 PILOT_TRAINING.md).
- AXBENCH-TR train has 91 sequences over 1,024 (max 1,219) on two wide real tables (37 and 43 columns), and 0 over
  2,048 (`TOKEN_STATS.json`). The EXP-010 rule says: stop above 1,024, no truncation, no silent drop. So a
  declared change to 2,048 is needed, and activation memory for the longest rows is unmeasured.

## Fixed inputs
- Base: Qwen3-1.7B at `70d244cc86cc…`, BF16, re-hashed as in EXP-010. mlx 0.32.4, mlx-lm 0.32.2 (EXP-010 pins).
- Data: `python -m axbench_tr export-mlx --out <directory outside the repository>` from the committed
  corpus (`full/`, `smoke/` and `peak_probe/`, hashes in `EXPORT_MANIFEST.json`). No test file exists or is exported.
- Rendering: the same chat template with `enable_thinking=False`; the EXP-010 completion-mask self-check runs on every
  exported example and must pass.
- Configuration: EXP-010 D7 (rank 8, scale 20, dropout 0, q/v of blocks 20-27, learning rate 1e-5, micro-batch 1,
  gradient accumulation 4, seed 2026, checkpointing off) with one **declared change**: `max_seq_length` 2048.
  Gate before any step: `benchmarks/axbench_tr_token_stats.py --max-seq-length 2048` exits 0 (it does today).

## Procedure (resource-efficient: about 15 minutes of compute expected, to be measured)
1. **Preflight (no training):** mains power, other applications closed, memory pressure normal, swap baseline, the
   checkpoint re-hash, the length gate and the mask check.
2. **Peak probe:** 20 optimizer micro-steps on `peak_probe/` (the 20 longest train rows, all near the 1,219-token
   maximum, so each step is a worst case). Record MLX peak, RSS, per-step seconds and tokens per
   second. If MLX peak exceeds 10 GB or any stop rule fires, stop and report; the next run (gradient checkpointing,
   or a length-capped schema policy) is a new, logged, owner-approved run.
3. **Smoke run:** 50 optimizer updates (200 micro-steps) on `smoke/train.jsonl` (400 rows stratified over all 23
   frames), validation loss only at updates 0 and 50 on `smoke/valid.jsonl` (40 rows), adapter saved at the end.
4. **Reload:** load the adapter in a fresh process and run real inference on 5 smoke-train examples to prove it
   loads and produces text. Not scored.
5. **Report:** per-update seconds, tokens per second (tokens from `token_lengths_per_example.jsonl`), the first versus
   last window ratio, memory and swap, mask check, loss finiteness, then the **full-pass time estimate** =
   3,487,013 tokens / measured tokens per second.

## Resource limits and stop rules (EXP-010, unchanged)
Process peak at most 10 GB. Stop on: allocation error; non-finite loss; memory pressure "critical"; swap growth above
2 GB; throughput below half of the first window for three consecutive windows; battery power. A stop is a failed run
with its logs, not a retry. At most two declared validation-only changes; no sweep.

## What a full pass would cost (derived, not measured on this corpus)
From EXP-010's committed record: 2 epochs over 182,592 tokens = 365,184 training tokens; trainer compute 230.2 s
(about 1,586 tokens/s) and the whole 800-micro-step loop 480.4 s including nine validation evaluations (about
760 tokens/s).
- One pass over AXBENCH-TR train (3,487,013 tokens): about 37 minutes at the compute rate, about 76 minutes at the
  loop rate. Three passes: about 1.8 to 3.8 hours.
- Caveats: sequences are longer on average here only slightly (mean 502 vs 456) but up to 1,219; the later EXP-010
  windows were about 14% slower on a fanless laptop and longer runs may throttle further. The smoke replaces these
  extrapolations with a measurement.

## Not claimed
Accuracy, Turkish quality, English retention, deployment readiness. Teacher-forced loss, if recorded, is descriptive.
The AI-reviewed validation labels and the unreviewed Turkish data cannot support those claims.

## Gates before the smoke and before Part B
- Smoke: owner approval; PR #13 reviewed; this protocol reviewed.
- Part B: the 300-record Turkish review accepted under a rule the owner confirms; licence summary reviewed; the
  sealed-test overlap check run by the owner with counts reported; a pre-registered analysis; separate approval.
