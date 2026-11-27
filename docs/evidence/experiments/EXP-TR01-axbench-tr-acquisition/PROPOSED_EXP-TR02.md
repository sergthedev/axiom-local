# Proposed EXP-TR02: AXBENCH-TR LoRA feasibility smoke (proposal v5; the v4 probe stopped on a name check; not approved to run again)

Status: **proposed, revised after the second technical review of PR #13 (harness revision) and again for the owner-authorized final licensing revision (probe and smoke restricted to schema sources with documented terms).** No model has been
loaded or run, and none may be until this protocol and PR #13 are reviewed, the M5 no-weight preflight below has been
run and its output reviewed, and the owner approves the run **for a named commit**. EXP-010 (PR #11) is untouched;
this protocol reads its committed numbers and changes nothing there. The harness is new and independent
(`training/tr02/`).

## Revision 5 (after the first authorized probe stopped): precise delta

The one probe authorized at code commit `d5f6cd56` (protocol v4) loaded the checkpoint (MLX peak 3.44 GB), attached 32 LoRA
tensors with 458,752 trainable parameters and stopped before any update on `lora_shapes_not_ok` (evidence:
`docs/evidence/experiments/EXP-TR02-feasibility-smoke/runs/p1-probe-stopped-at-d5f6cd5/RUN_RESULT.md`, claim C-TR-17). P2 and
P3 never ran. That authorization is spent. Revision 5 changes **only** the following:

1. **Name contract.** The pinned model-specific expectation is `model.layers.N.self_attn.{q,v}_proj.lora_{a,b}` (mlx-lm 0.32.2
   `qwen3.Model` nests the blocks in `self.model`), derived from the checkpoint config and the architecture, not from what the
   model reports; it refuses other model types; no wildcard or suffix matching; missing, extra, wrong-layer,
   wrong-projection, wrong-shape and unprefixed names fail. Counts and parameter totals are unchanged (32 and 458,752).
2. **Checks around it.** The `lora` event carries a bounded name and shape report; `run_reload` additionally compares the
   reloaded model's live LoRA tensors with the saved file (names, shapes, finiteness, values) before the reload loss; the
   conversion, freezing and adapter config now come from named helpers (`attach_lora`, `lora_adapter_config`) that write
   byte-identical adapter configs to revision 4.
3. **Tests.** Real tiny random Qwen3 regression tests (`tests/test_tr02_lora_tiny_qwen3.py`, skipped where MLX is absent) and
   a `check-lora` command that runs the same fixture in the M5 no-weight preflight.
4. **Evidence workflow.** A documented separate-evidence-checkout procedure and a `sanitize-evidence` command with a
   redaction manifest; `collect-evidence` is stricter (existing run directories, unique names, no weight copy, no
   overwrite) and its destination guard is unchanged.

**Unchanged:** model and checkpoint pins, corpus, export, per-example hashes and subset selection, hyperparameters, stop
limits, phases, optimizer counts, evaluation gates, the event audit's gates and `training/tr02/config.json` (byte-identical).
Execution needs a new, separately granted owner authorization for a named code commit after technical review.

## Question
Part A (this smoke): can the M5 16 GB train Qwen3-1.7B LoRA on AXBENCH-TR sequences of up to 1,342 tokens (1,343 with
the template's trailing newline), with no truncation, inside declared memory limits, and how long would a full pass
take? **No accuracy question is asked and no accuracy is reported.** The smoke is discard-only: nothing it trains is
kept, evaluated for quality, or used later.
Part B (later, separate approval, not designed here beyond its preconditions): does adding AXBENCH-TR to v0.1
training improve Turkish routing without hurting English? It needs frozen comparators, a new independently authored
evaluation and separately approved criteria.

## Data used: rights-cleared draft subset only
The probe and smoke rows are drawn **only from `synthetic_template` records** (this project's own text on real table
headers; no upstream text) **whose table schema comes from SRC-O01 (World Bank WDI), SRC-O02 (Eurostat) or SRC-O06
(İzmir open data)**, the three sources whose reuse terms are documented for the exact datasets used
(`data/axbench-tr/sources/LICENCE_EVIDENCE_ADDENDUM_1.json`; the original `LICENCE_EVIDENCE.json` is unchanged). Schemas
from SRC-O03 (ECB), SRC-O04 (OECD) and SRC-O05 (TCMB), whose terms are unverified, are excluded. Compared with the
previous selection, 22 smoke train rows and 1 smoke validation row (23 in all) were replaced by the next-ranked
eligible rows of the same frame; the other 378 train rows, 39 validation rows and all 20 probe rows are unchanged.
`external_adapted` records are not used: the upstream Gretel generation terms are
unresolved and the SRC-N03 licence conflict is open, so there is no final corpus clearance, and the 97 Turkish external
PLANs are `translation_unverified`. The data are unreviewed AI-authored text; that is acceptable only because the
smoke measures resources. The export records the subset (`rights_cleared_subsets` in `EXPORT_MANIFEST.json`) and the
tests check that no external row and no row on a schema source outside that list is in `smoke/` or `peak_probe/`, and that the addendum covers every table those rows use.

## What reading the stock trainer showed (mlx-lm 0.32.2 source; data path and loss mask run on a CPU build)
`python -m mlx_lm lora` is therefore not used. The harness reuses its `train`, `evaluate` and `iterate_batches`
code and changes only what is listed.
1. **It truncates silently.** `iterate_batches` cuts a sequence longer than `max_seq_length` and prints a warning.
   Harness: the dataset raises on any sequence over the limit and an exposure checker compares every drawn row's
   token hash and length with the declared schedule, so a truncated or wrong row stops the run.
2. **Its Tokens/sec is loss-token throughput.** The trainer sums the loss mask, i.e. the supervised answer
   (about 26 tokens per smoke row), not the roughly 500 tokens it computes on. Harness: two compute rates are
   reported, **full tokens** (the sequence tokens of the exposed rows; used by the stop rule) and **padded tokens**
   (`1 + 32 * ceil(n / 32)` capped at 2,048, minus the input shift; what the model computes on; 3.0% higher on the
   smoke rows). Both are the schedule's tokens over the trainer's measured training seconds.
3. **Its final in-loop validation runs before the last optimizer step.** With `iters` equal to the microsteps,
   the "last" validation sees 99 of 100 updates. Harness: no validation inside the loop; validation runs before the
   first and after the last update, timed on its own.
4. **Its default `val_batches` is 25**, which would skip 15 of 40 validation rows. Harness: all rows (checked: the
   validation event must report all of them).
5. **Its chat dataset supervises the template's empty thinking block and a trailing newline** (5 extra tokens per row).
   The EXP-010 rendering, and the inference prompt, contain neither as loss. Harness: the full conversation is
   rendered by the model's template, cut at `<|im_end|>`, and the prompt is rendered with `enable_thinking=False`.
   Checked for all 7,269 rows: the length equals the measured one and the supervised span equals the compact JSON and
   `<|im_end|>` (168,102 supervised tokens in train).
6. **Its `default_loss` scores one extra target per row.** It masks `offset <= step <= length`; the last step is the
   zero padding after `<|im_end|>`. Found by running the stock loss on the exposed rows (`check-iterator`): 37 mask
   tokens against a 36-token answer on the first probe row; 31 against 30 on the first smoke row. Harness:
   `make_exact_loss` (`step < length`) is used for training, every validation and the reload, and `check-iterator`
   asserts mask tokens equal supervised tokens.
7. **Its row order depends on a global NumPy seed set elsewhere.** Harness: the order is `numpy.random.seed(2026)`
   then one permutation, set immediately before training and verified against the real iterator (`check-iterator`:
   order, token lengths, padded widths; no mismatch on 20 probe and 400 smoke rows).
8. Its default LoRA target is every linear layer of the chosen blocks. Harness: `self_attn.q_proj` and
   `self_attn.v_proj` of the last 8 of 28 blocks (blocks 20 to 27), as in EXP-010: 32 tensors, 458,752 parameters,
   shapes derived from the checkpoint's own `config.json` and checked.

## Fixed, immutable inputs (all hashes in `training/tr02/config.json`)
- Base: `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16. **All ten accepted EXP-009
  checkpoint files are pinned by size and SHA-256** (including `config.json`, the weight index and both weight
  shards, from the committed EXP-009 record). `verify-model` and the preflight reject a missing, extra, resized or
  changed file, and require `PROVENANCE.json` with weights verified against a published hash.
- Packages: mlx 0.32.4, mlx-lm 0.32.2, Python 3.12 (`envs/model/uv.lock`), checked at run time. Offline mode
  (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) is required. The checkout must be **clean at the exact commit the
  owner approved** (`--expect-commit`).
- Corpus `v0.2-rev1` (train and validation SHA-256 pinned), the per-example length file (read only through its pinned
  hash), and the exported chat files (`uv run --locked python -m axbench_tr export-mlx --out <directory outside the
  repository>`) with `ROW_IDS.json`. Every check above runs in the supervisor **and again in the child, at execution
  time**, not only as an optional CLI step. No test split exists in the export.
- Hyperparameters: EXP-010 D7 (rank 8, scale 20, dropout 0, Adam, learning rate 1e-5, micro-batch 1, gradient
  accumulation 4, seed 2026, no checkpointing) with one declared change, `max_seq_length` 2048 (EXP-010 used 1024).
  Measured with the pinned tokenizer and the trainer's own rendering: longest sequence 1,342 (1,343 template-full),
  70 train rows over 1,024, none over 2,048.

## Preflight on the M5, before the owner is asked to approve (no weights are loaded)
From the repository root, with the locked model environment (exact commands in `training/tr02/README.md`):
`verify-inputs`, `verify-model`, `check-data`, **`check-iterator` for `probe` and `smoke`** (the real mlx-lm iterator
and loss mask on the M5's MLX build; the CPU-build check done in preparation does not validate the GPU path), and
`preflight --phase probe`. The outputs are committed and reviewed before approval. Any failure ends the attempt.

### Preflight results and the proposed authorization target
The no-weight preflight was run twice on the owner's M5, each time once and from a clean worktree at the exact commit: at
96cbdf7 (superseded subset) and, after the licensing revision, at **d5f6cd56c0d3a12ecb9e917570d028788809ac78** (all steps exit
0, no problems, export hashes equal the pins; `docs/evidence/experiments/EXP-TR02-feasibility-smoke/preflight/`). The feasibility
run's `--expect-commit` is checked against a clean tree at the approved commit, so the proposed target is that same code head,
`d5f6cd56c0d3a12ecb9e917570d028788809ac78`, which is the commit the preflight evidence covers; later commits on the branch
change only evidence and documents (and `.gitattributes`), not code, config or data. If the owner prefers a later head, the
no-weight preflight must be repeated at it. Nothing is authorized by this text.

## Phases (each its own supervised process, fresh output and supervisor directories, no retry)
Each phase is started by its own `supervise --phase ...` command, chained with `--after <previous output directory>`.
A phase refuses to start unless the previous phase wrote `COMPLETE.json` (written only after a finished, valid run
whose recorded file hashes still hold). A stopped or failed phase leaves `STOP.json` and its logs; no script runs the
next phase after it.

**P1 Peak probe:** the 20 longest rights-cleared train rows (1,199 to 1,342 sequence tokens, 24,260 in all, 24,448
padded), each exposed exactly once in the seeded order: **20 microsteps = 5 optimizer updates**, fresh adapter
directory. Purpose: the memory and time of the worst case. The loss before and after on the *same* 20 rows is
**diagnostic only** (a sanity check that mask and loss are finite); it is not validation and is never reported as such.
The probe adapter is not reused by any later phase.

**P2 Smoke:** a new process, fresh adapter directory, fresh optimizer, no state from the probe (requires a COMPLETE
probe). The 400 rows of `smoke/train.jsonl` (stratified over all frames, none of them in the probe, 206,526 tokens,
212,704 padded) are each exposed **exactly once** in the seeded order: **400 microsteps = 100 optimizer updates** =
one pass. Validation loss on the 40 rows of `smoke/valid.jsonl` before training (update 0) and after update 100 only.
One report per update. The adapter is saved once, at the end; there are no intermediate checkpoints, so there is
nothing to select.

**P3 Reload (supervised):** a fresh process loads the saved adapter (requires a COMPLETE smoke) and re-evaluates the 40
validation rows. The adapter and config hashes must equal the ones the smoke recorded, the adapter tensors must have
the expected shapes and be finite, and the loss must equal the P2 post-training loss within an **absolute tolerance of
1e-4**; a larger difference stops the run and the process exits non-zero. Nothing is generated, scored or selected.

## Event audit, memory limits and stop rules (supervised, separate, no retry)
`tr02.py supervise` runs the driver as a child. **Before the child starts**, the supervisor takes the baseline
(AC power, normal memory pressure, swap) and refuses to launch otherwise. It then applies these rules to *every*
driver event (not only the training reports) and to OS samples every 5 s; on the first hit it terminates the child and
writes `STOP.json`. A stop is a failed run with its logs. **No retry, no resume, no sweep.** A new attempt is a new
logged, owner-approved run.

| Rule | Limit | Source |
| --- | --- | --- |
| Event sequence | exactly: start, load, lora, validation before, one `train` per update, `train_total`, adapter, validation after, done (reload: start, load, adapter check, validation, reload, done); anything else, out of order or after done stops | driver events |
| Metrics | every loss, rate and duration finite and in range, including `load`, validation and `done` | every event |
| MLX allocator peak (`mx.get_peak_memory`, decimal GB) | 10.0 GB at load, every report, validation and the final `done`; never decreasing | driver events |
| Process RSS, **separate** from the MLX peak (decimal GB) | 11.0 GB | `ps` sample every 5 s |
| Swap growth over the pre-launch baseline | 2.0 GB | `vm.swapusage` |
| Memory pressure | refuse to launch unless normal; stop at critical | `kern.memorystatus_vm_pressure_level` |
| Power source | refuse to launch and stop on battery | `pmset -g batt` |
| Iteration indices and rate | iteration 4, 8, ... exactly; `iterations_per_second` finite and positive | train events |
| Optimizer | the step counter rises by one per report and ends at the planned number of updates (5 or 100); rows exposed equal the microsteps | `optimizer.step`, exposure checker |
| Adapter | 32 tensors with the shapes derived from the checkpoint config, 458,752 parameters, finite, changed from initialisation, non-zero `lora_b`; the saved files re-hashed by the supervisor equal the logged hashes | adapter event and files |
| Full-token throughput | below half of the first 10-update window for 3 consecutive windows | schedule tokens over measured training seconds |
| Completion | the whole sequence arrived and the child exited 0; otherwise the run is not a success | supervisor |
| Any driver error or probe failure | stop | allocation error, a drawn row differing from the schedule (including a truncated one), an unreadable OS probe, a malformed event, a non-zero exit |

A `ps` failure on a child that has already exited is a normal exit, not a probe failure. The child is terminated on any
stop, malformed event, probe failure, error or interrupt. Logs are never overwritten and partial logs are kept.

## What is measured and reported, and the evidence kept
Per update and per 10-update window: full-token and padded-token compute rates (the trainer's own loss-token rate is
logged separately and labelled); loss; MLX peak; maximum RSS; swap and pressure; model-load time; **validation overhead
measured separately** (seconds of each evaluation); first-versus-last window ratio; adapter size. Then the
**full-pass estimate = 3,018,924 train tokens / measured full-token compute rate + validation evaluations planned x
measured validation seconds**, with the padded-rate estimate beside it.

**Evidence is committed to the repository, not only summarised:** `collect-evidence` copies each phase's raw
`events.jsonl`, supervisor log, child log, `STOP.json` or `COMPLETE.json` and the adapter config into
`docs/evidence/experiments/EXP-TR02-feasibility-smoke/runs/` with `EVIDENCE.json` (SHA-256 of every file and of the
adapters, which are hashed and not copied). The report may cite only those files.

Derived expectation, not a measurement on this corpus (from EXP-010's committed record: compute about 1,586
tokens/s and whole loop about 760 tokens/s including nine validation evaluations): P2 about 2 to 5 minutes, P1 about
15 to 35 seconds; a full pass about 32 minutes at the compute rate and about 66 minutes at the loop rate. The first
windows of EXP-010 were about 14% faster than the later ones on this fanless machine; longer rows here raise
activation memory in a way that is unmeasured above 514 tokens, which is exactly what P1 measures.

## Pass criteria (feasibility only)
All phases finish with no stop rule and each wrote `COMPLETE.json`; every scheduled row was drawn once in order; no
truncation; losses finite; the adapter is the trained adapter (changed, non-zero, expected shape); reload loss within
tolerance; the full-token and padded rates and both memory figures recorded. A pass says the run is feasible and what a
full pass costs. It says nothing about accuracy, Turkish quality or English retention.

## Not claimed, not done
Accuracy, Turkish quality, English retention, deployment readiness. **No sealed-test access and no contamination
check** (deferred to an independent final-evaluation protocol that the owner runs). Teacher-forced loss is descriptive.
Unreviewed AI-authored data cannot support quality claims. The no-weight tests and the CPU-build iterator check do not
show that the GPU path, memory use or timing work.

## Gates
- Before the smoke: owner approval of this protocol **for a named commit**; PR #13 reviewed and its data gates
  understood; the M5 no-weight preflight run and its output reviewed. The smoke trains nothing that is kept and uses
  unreviewed rights-cleared template rows, which is acceptable only because it measures resources.
- Before any training that is used: native-speaker review of the stratified sample passing the predeclared gates
  (at least 95% gold correct and 90% natural-or-corrected per origin, reported by language and frame; zero serious
  errors remaining after correction; independent second review with adjudication; AI review kept apart from human
  decisions), reviewers and authors **designated by the owner** (an AI cannot stand in for them), the independently
  authored Turkish held-out tasks (`HELDOUT_AUTHORING_KIT.md`), quarantine or human review of the
  `translation_unverified` records, and resolution of the upstream Gretel terms and the SRC-N03 conflict (or exclusion
  of the affected records). Until then there is no final corpus or product clearance.
- Part B: frozen comparators, the new independent evaluation, separately approved criteria, a pre-registered analysis.
