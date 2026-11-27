# Proposed EXP-TR02: AXBENCH-TR LoRA feasibility smoke (proposal v2; not approved, not run)

Status: **proposed, revised after the technical review of PR #13.** No model has been loaded or run, and none
may be until this protocol and PR #13 are reviewed and the owner approves the run. EXP-010 (PR #11) is untouched;
this protocol reads its committed numbers and changes nothing there. The harness is new and independent
(`training/tr02/`).

## Question
Part A (this smoke): can the M5 16 GB train Qwen3-1.7B LoRA on AXBENCH-TR sequences of up to 1,343 tokens, with
no truncation, inside declared memory limits, and how long would a full pass take? **No accuracy question is asked
and no accuracy is reported.**
Part B (later, separate approval, not designed here beyond its preconditions): does adding AXBENCH-TR to v0.1
training improve Turkish routing without hurting English? It needs frozen comparators, a new independently authored
evaluation and separately approved criteria.

## What reading the stock trainer showed (mlx-lm 0.32.2 source, read, not run)
`python -m mlx_lm lora` is therefore not used. The harness reuses its `train`, `evaluate` and `iterate_batches`
code and changes only what is listed.
1. **It truncates silently.** `iterate_batches` cuts a sequence longer than `max_seq_length` and prints a warning.
   Harness: the dataset raises on any sequence over the limit and an exposure checker compares every drawn row's
   token hash and length with the declared schedule, so a truncated or wrong row stops the run.
2. **Its Tokens/sec is loss-token throughput.** The trainer sums the loss mask, i.e. the supervised answer
   (about 29 tokens per row; 26 on the smoke rows), not the roughly 512 tokens it computes on. Harness: the compute rate is the full tokens
   of the exposed rows (from the schedule and the pinned per-example lengths) divided by the trainer's measured
   training seconds.
3. **Its final in-loop validation runs before the last optimizer step.** With `iters` equal to the microsteps,
   the "last" validation sees 99 of 100 updates. Harness: no validation inside the loop; validation runs before the
   first and after the last update, timed on its own.
4. **Its default `val_batches` is 25**, which would skip 15 of 40 validation rows. Harness: all rows.
5. **Its chat dataset supervises the template's empty thinking block and a trailing newline** (5 extra tokens per row;
   measured on the 400 smoke rows). The EXP-010 rendering, and the inference prompt, contain neither as loss.
   Harness: the full conversation is rendered by the model's template, cut at `<|im_end|>`, and the prompt is
   rendered with `enable_thinking=False`; loss covers exactly the compact JSON and `<|im_end|>`. Checked for all 7,432
   rows: the length equals the measured one, and the supervised tokens equal the measured completion tokens
   (171,840 in train).
6. **Its row order depends on a global NumPy seed set elsewhere.** Harness: the order is `numpy.random.seed(2026)`
   then one permutation, set immediately before training and verified against the stock iterator (a test runs the
   real `mlx_lm` iterator when it is importable; it matched on both subsets).
7. Its default LoRA target is every linear layer of the chosen blocks. Harness: `self_attn.q_proj` and
   `self_attn.v_proj` of the last 8 of 28 blocks (blocks 20 to 27), as in EXP-010.

## Fixed, immutable inputs (all hashes in `training/tr02/config.json`)
- Base: `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16. `tr02.py verify-model` re-hashes
  every provisioned file against `PROVENANCE.json`, requires weights verified against a published hash, and checks the
  tokenizer files against the pinned SHA-256 (also in `ACQUISITION.json`).
- Packages: mlx 0.32.4, mlx-lm 0.32.2 (`envs/model/uv.lock`).
- Corpus `v0.2-rev1` (train and validation SHA-256 pinned), the per-example length file, and the exported chat files
  (`python -m axbench_tr export-mlx --out <directory outside the repository>`) with `ROW_IDS.json`. `tr02.py
  verify-inputs` stops on any changed, missing, unpinned or test-named file. No test split exists in the export.
- Hyperparameters: EXP-010 D7 (rank 8, scale 20, dropout 0, Adam, learning rate 1e-5, micro-batch 1, gradient
  accumulation 4, seed 2026, no checkpointing) with one declared change, `max_seq_length` 2048 (EXP-010 used 1024).
  The gate was measured with the pinned tokenizer and the trainer's own rendering: longest sequence 1,343, 70 train
  rows over 1,024, none over 2,048.

## Phases (each its own process and a fresh, empty output directory)
Phases P1 to P3 are started by `python training/tr02/tr02.py supervise --phase <probe|smoke> ...` (below) and
never retried; commands are in `training/tr02/README.md`.

**P0 Preflight (no weights):** `verify-inputs`, `check-data` (all rows through the harness dataset: rendering,
supervised span, measured length, no truncation), `verify-model`, mains power, other applications closed, memory
pressure normal, swap baseline recorded. Any failure ends the attempt.

**P1 Peak probe:** the 20 longest train rows (1,199 to 1,343 tokens, 24,260 tokens in all), each exposed exactly once
in the seeded order: **20 microsteps = 5 optimizer updates**, fresh adapter directory `probe/`. Purpose: the memory
and time of the worst case. The loss measured before and after on the *same* 20 rows is **diagnostic only** (a
sanity check that the mask and loss are finite); it is not validation and is never reported as such. The probe
adapter is not reused by any later phase.

**P2 Smoke:** a new process, fresh adapter directory `smoke/`, fresh optimizer, no state from the probe. The 400 rows of
`smoke/train.jsonl` (stratified over all frames, none of them in the probe, 206,656 tokens) are each exposed
**exactly once** in the seeded order: **400 microsteps = 100 optimizer updates** = one pass. Validation loss on the
40 rows of `smoke/valid.jsonl` before training (update 0) and after update 100 only. One report per update.
The adapter is saved once, at the end; there are no intermediate checkpoints, so there is nothing to select.

**P3 Reload:** a fresh process loads the saved adapter and re-evaluates the 40 validation rows. The loss must equal
the P2 post-training loss; the difference is reported. Nothing is scored.

## Memory limits and stop rules (supervised, separate, no retry)
`tr02.py` runs the driver as a child of a supervisor that applies these rules to the driver's per-update events and
to OS samples every 5 s; on the first hit it terminates the child and writes `STOP.json`. A stop is a failed run with
its logs. **No retry, no resume, no sweep.** A new attempt is a new logged, owner-approved run.

| Rule | Limit | Source |
| --- | --- | --- |
| MLX allocator peak (`mx.get_peak_memory`, decimal GB) | 10.0 GB | the trainer's report, every update |
| Process RSS, **separate** from the MLX peak (decimal GB) | 11.0 GB | `ps` sample every 5 s |
| Swap growth over the preflight baseline | 2.0 GB | `vm.swapusage` |
| Memory pressure | stop at critical | `kern.memorystatus_vm_pressure_level` |
| Power source | stop on battery | `pmset -g batt` |
| Loss | stop on non-finite | every update |
| Full-token throughput | below half of the first 10-update window for 3 consecutive windows | schedule tokens over measured training seconds |
| Any driver error or probe failure | stop | allocation error, a drawn row differing from the schedule (including a truncated one), an unreadable OS probe, a non-zero exit |

## What is measured and reported
Per update and per 10-update window: full-token compute rate (tokens the trainer computes on per second of training
time, **not** the trainer's loss-token rate, which is reported separately and labelled); loss; MLX peak; maximum
RSS; swap and pressure; model-load time; **validation overhead measured separately** (seconds of each of the two
evaluations); first-versus-last window ratio; adapter size. Then the **full-pass estimate = 3,066,063 train tokens
/ measured full-token compute rate + validation evaluations planned x measured validation seconds**.

Derived expectation, not a measurement on this corpus (from EXP-010's committed record: compute about 1,586
tokens/s and whole loop about 760 tokens/s including nine validation evaluations): P2 about 2 to 5 minutes, P1 about
15 to 30 seconds; a full pass about 32 minutes at the compute rate and about 67 minutes at the loop rate. The first
windows of EXP-010 were about 14% faster than the later ones on this fanless machine; longer rows here raise
activation memory in a way that is unmeasured above 514 tokens, which is exactly what P1 measures.

## Pass criteria (feasibility only)
All phases finish with no stop rule; every scheduled row was drawn once in order; no truncation; losses finite; reload
loss equal; the full-token rate and both memory figures recorded. A pass says the run is feasible and what a full pass
costs. It says nothing about accuracy, Turkish quality or English retention.

## Not claimed, not done
Accuracy, Turkish quality, English retention, deployment readiness. **No sealed-test access and no contamination
check** (deferred to an independent final-evaluation protocol that the owner runs). Teacher-forced loss is descriptive.
Unreviewed AI-authored data cannot support quality claims.

## Gates
- Before the smoke: owner approval of this protocol; PR #13 reviewed and its data gates understood. The smoke trains
  nothing that is kept and uses unreviewed data, which is acceptable only because it measures resources.
- Before any training that is used: native-speaker review of the stratified sample passing the predeclared gates
  (95% gold correct and 90% natural-or-corrected per origin; no serious unsafe or unsupported error after correction;
  second review with adjudication; AI review kept apart from human decisions), the independently authored Turkish
  held-out tasks (`HELDOUT_AUTHORING_KIT.md`), and resolution of the licence items in `DATA_SOURCES.md`.
- Part B: frozen comparators, the new independent evaluation, separately approved criteria, a pre-registered analysis.
