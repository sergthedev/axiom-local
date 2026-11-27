# EXP-TR02 harness (revision 5; the revision-4 probe stopped on a name check; no training has run)

Independent of the EXP-010 harness (PR #11, untouched). **No model has been loaded or run by this code.** The model
path (`run_phase`, `run_reload`) is reviewed by reading only; everything else is unit-tested without weights in
`tests/test_tr02_harness.py` (pins and inventory checks, schedule, the full event audit, stop rules, probe parsers,
supervisor, evidence collection). The first run needs the owner's approval of
`docs/evidence/experiments/EXP-TR01-axbench-tr-acquisition/PROPOSED_EXP-TR02.md`.

**Revision 5 (this code).** The one authorized probe at revision 4 (code commit `d5f6cd56`) loaded the weights and stopped
before any update on `lora_shapes_not_ok`: the harness expected the trainable tensors to be named `layers.N...` and
mlx-lm 0.32.2 names them `model.layers.N...` (`qwen3.Model` keeps its blocks in `self.model`). Evidence:
`docs/evidence/experiments/EXP-TR02-feasibility-smoke/runs/p1-probe-stopped-at-d5f6cd5/RUN_RESULT.md`. Revision 5 changes
only the name contract and the checks around it (below) and the evidence workflow; model and checkpoint pins, corpus, export
and subsets, hyperparameters, stop limits, phases, optimizer counts and evaluation gates are unchanged, and `config.json` is
byte-identical to revision 4. The earlier execution authorization is spent; a new one is needed for any run.

- `config.json`: immutable pins (model revision and all ten checkpoint files with size and SHA-256, packages and
  Python, tokenizer hashes, corpus, export and per-example hashes), the hyperparameters, the limits, and the three
  phases (`probe`, `smoke` requires `probe`, `reload` requires `smoke`). Changing it means a new version and review.
- `tr02.py`: commands `pin`, `verify-inputs`, `schedule`, `check-data`, `check-iterator`, `verify-model`, `preflight`,
  `supervise`, `run` (the supervised child; not started by hand), `collect-evidence`.

Why not `python -m mlx_lm lora`: it truncates over-length sequences with a warning, reports loss-token (not full-token)
throughput, validates one update early, supervises a thinking block the EXP-010 rendering does not, skips validation
rows by default, **and its `default_loss` scores one extra target per row (the zero padding after `<|im_end|>`)**.
The last point was found by `check-iterator` against mlx-lm 0.32.2 (stock mask 37 tokens, supervised answer 36 on the
first probe row); the harness uses `make_exact_loss` for training, every validation and the reload. The rest of the
comparison is in the protocol; every statement was read from the mlx-lm 0.32.2 source.

## Data subsets

`probe` (the 20 longest eligible train rows) and `smoke` (400 train rows stratified by frame, 40 validation rows) use only
`synthetic_template` records on table schemas from SRC-O01, O02 and O06, the sources whose terms are documented for the
datasets used (`data/axbench-tr/sources/LICENCE_EVIDENCE_ADDENDUM_1.json`). Smoke: 206,526 full tokens (212,704 padded),
supervised answer tokens 10,530 (train) and 1,069 (validation); probe: 24,260 full tokens (24,448 padded).

## What is checked, and when

Checked at execution time, by the supervisor before it starts the child **and again by the child** (`preflight`):
export hashes (and no extra or test-named file), committed corpus and per-example hashes (the schedule reads the
per-example file only through its pinned hash), the model directory against the exact ten-file inventory (a missing,
extra, resized or changed file is a failure), `mlx`, `mlx-lm` and Python versions, offline mode
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), a clean git tree at the **exact commit the owner approved**
(`--expect-commit`, 40 characters), output directories outside the repository and new, and for `smoke` and `reload`
the previous phase's `COMPLETE.json` with every recorded file hash still true.

Checked on every driver event (`StopRules`): the exact event sequence, finite and in-range metrics for every event
including `load`, `val` and `done`, the MLX peak on all of them (and that it never decreases), the exact iteration
indices (4, 8, ...), a finite positive `iterations_per_second`, an optimizer step that rises by one per report and ends
at the planned number of updates, every scheduled row drawn once in order, the LoRA tensors (32, shapes from the
checkpoint's `config.json`, 458,752 parameters), and a saved adapter that is finite, changed, with non-zero `lora_b`
and whose file hashes equal the ones logged. The supervisor re-hashes the saved files itself after the child exits.
A phase finishes only when the whole sequence has arrived and the child exited 0; only then is `COMPLETE.json` written.

Before the supervisor starts the child it takes the baseline (AC power, normal memory pressure, swap). A `ps` failure
on a child that has already exited is a normal exit, not a probe failure. The child is terminated on any stop, malformed
event, probe failure, error or interrupt; logs are never overwritten and partial logs are kept.

## Commands (Apple Silicon Mac, from the repository root, after owner approval)

Every command is run with the locked model environment (`envs/model/uv.lock`) from the repository root. The export
runs in the locked Core environment.

```
RUN="uv run --locked --project envs/model python training/tr02/tr02.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COMMIT=<the 40-character commit the owner approved; the checkout must be clean at it>
WORK=<a directory outside the repository>   MODEL=<the pinned Qwen3-1.7B directory>

uv run --locked python -m axbench_tr export-mlx --out "$WORK/export"
$RUN verify-inputs   --export "$WORK/export"
$RUN verify-model    --model-dir "$MODEL"
$RUN check-data      --export "$WORK/export" --tokenizer-dir "$MODEL"
$RUN check-iterator  --export "$WORK/export" --tokenizer-dir "$MODEL" --phase probe   # no weights; real mlx-lm iterator and loss mask
$RUN check-iterator  --export "$WORK/export" --tokenizer-dir "$MODEL" --phase smoke
$RUN check-lora   # tiny random Qwen3 only: name, save and reload contract; no pretrained weights
$RUN preflight  --phase probe --model-dir "$MODEL" --export "$WORK/export" --out "$WORK/probe" --expect-commit "$COMMIT"

$RUN supervise --phase probe  --model-dir "$MODEL" --export "$WORK/export" --out "$WORK/probe"  --expect-commit "$COMMIT"
$RUN supervise --phase smoke  --model-dir "$MODEL" --export "$WORK/export" --out "$WORK/smoke"  --expect-commit "$COMMIT" --after "$WORK/probe"
$RUN supervise --phase reload --model-dir "$MODEL" --export "$WORK/export" --out "$WORK/reload" --expect-commit "$COMMIT" --after "$WORK/smoke"

$RUN collect-evidence --dest docs/evidence/experiments/EXP-TR02-feasibility-smoke/runs \
    --run probe="$WORK/probe" --run smoke="$WORK/smoke" --run reload="$WORK/reload"
```

Each `supervise` is a separate command: a later phase is never started by a script after a failed one, and it refuses
to start without the earlier phase's `COMPLETE.json`. A stop is a failed run: no retry, no resume. The three
`<out>.supervisor/` directories and the three outputs must be new. `collect-evidence` copies the raw `events.jsonl`,
supervisor log, child log, `STOP.json`/`COMPLETE.json` and the adapter config into the repository with a SHA-256
index; the adapters (weights) are hashed, not copied.

## LoRA name contract (revision 5)

`expected_lora_shapes` derives the exact names and shapes from the checkpoint's `config.json` and the Qwen3 architecture in
mlx-lm 0.32.2, never from what the model reports: `model.layers.N.self_attn.{q,v}_proj.lora_{a,b}` for the last
`lora_layers` blocks, `lora_a` `(hidden, rank)`, `lora_b` `(rank, out)`, with q out = heads * head_dim and v out = kv_heads *
head_dim. It refuses any `model_type` other than `qwen3`. There is no wildcard or suffix matching: a missing, extra,
wrong-layer, wrong-projection, wrong-shape or unprefixed name is a failure (`lora_problems`). For Qwen3-1.7B the contract is
32 tensors and 458,752 trainable parameters (tested). The same contract is applied at every call site: the initial attachment
(`lora` event), the saved adapter against the contract and against the live trainable tensors (`run_phase`), the saved adapter
against the contract in `run_reload`, and now also the **reloaded model's live LoRA tensors against the saved file** (names,
shapes, finiteness, values) before the reload loss is computed. The supervisor's `build_rules` uses only the contract's
tensor and parameter counts. The `lora` event now carries a bounded `report` (counts and the first six observed and expected
names with their shapes; no tensor contents, no paths) so that a failure shows the observed names.

`tests/test_tr02_lora_tiny_qwen3.py` builds a real, randomly initialized tiny mlx-lm Qwen3 model (no download, no pretrained
weights, no optimizer step), runs the production conversion, freezing and flattening, the exact name and shape contract, a
safetensors save with the trainer's own call and a reload through mlx-lm's `load_adapters`, and the negatives (missing, extra,
wrong layer, wrong projection, wrong shape, unprefixed and mixed namespace, tampered values, non-finite values, mismatched
adapter config). These tests are **skipped, not passed,** where MLX or mlx-lm is absent (the Core CI environment); they run
in `envs/model`. The M5 preflight runs the same fixture as `check-lora`.

## Collecting evidence after the run (revision 5)

The first collection failed on the command's own guard, not on training: `collect-evidence` writes only inside a repository,
and the execution checkout must stay clean and immutable. Collect from a **separate evidence checkout**, only after every
phase has ended or stopped, and commit only sanitized copies:

```
# execution has finished or stopped; the execution worktree stays untouched
git worktree add "$WORK/evidence-wt" <the PR branch>       # a different checkout of this repository
cd "$WORK/evidence-wt"
RUN="uv run --locked --project envs/model python training/tr02/tr02.py"
RAW=.evidence-raw/<run-id>                                  # ignored by git; holds private paths
$RUN collect-evidence --dest "$RAW" --run probe="$WORK/probe" --run smoke="$WORK/smoke" --run reload="$WORK/reload"
$RUN sanitize-evidence --src "$RAW" \
    --dest docs/evidence/experiments/EXP-TR02-feasibility-smoke/runs/<run-id>/sanitized \
    --term "<account name>=<user>" --term "<device name>=<device>"   # the home directory is replaced by default
```

Pass only the phases that exist. `collect-evidence` requires every run directory to exist and the destination to be new,
copies the events, supervisor log, child log, `STOP.json`/`COMPLETE.json` and the adapter config, never copies a weight file,
and records the SHA-256 and size of every file it did not copy in `EVIDENCE.json`. `sanitize-evidence` requires a new
directory inside the repository, writes sanitized copies (home prefix to `$HOME`, other local paths to `<local-path>`, named
terms to placeholders, credential-like tokens to `<redacted-credential>`), refuses weights and non-UTF-8 files, fails if a
private path or a named term survives, and writes `REDACTIONS.json` with the original and sanitized SHA-256, byte counts and
redaction counts of every file. The originals stay where they are. Verify every committed file against `REDACTIONS.json`
with `sha256sum` before the commit, force-add the `*.log` files (`.gitignore` ignores them), and keep logs recovered after the
fact apart from the logs verified earlier (name them in the run record).

## Compute rates

Two rates are reported for every 10-update window: **full tokens** (the schedule's token count for the exposed rows;
the stop rule uses this) and **padded tokens** (what the model actually computes on, `1 + 32 * ceil(n / 32)` capped at
2,048, minus the one input shift; about 3% higher on the smoke rows). The trainer's own `Tokens/sec` (loss tokens) is
logged separately and never used.

## Preparation checks already run on Linux, without weights

The pinned export reproduces its hashes; all rows render through the harness dataset to the measured lengths with
the supervised span equal to the answer; the seeded order, the padded widths and the loss mask equal the stock `mlx_lm`
iterator for both the probe and the smoke subsets (`check-iterator`, mlx 0.32.4 CPU build, tokenizer only); the
no-weight tests above. They do **not** show that the GPU path, memory use or timing work: that is the point of the
run, and `check-iterator` is to be run again on the M5 as a no-weight preflight before the owner is asked to clear it.
