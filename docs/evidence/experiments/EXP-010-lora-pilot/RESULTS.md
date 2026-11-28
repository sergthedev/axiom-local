# EXP-010 results: frozen analysis of the pilot (2026-11-28; pending technical review)

The numbers below are copied from [results.json](results.json), the unchanged output of the frozen analysis
(`exp010.analysis/3`, gates `exp010.pilot-gates/2`, scorer `axbench.score/0.2`). The corpus is `axbench/0.1-ai-reviewed`:
exploratory, AI-reviewed, with **0 human decisions**. All results are on its 160-example validation split. This is
one training run and one evaluation on one machine; nothing here is a population estimate or a deployment decision.

## Execution of step 5

- Authorization: the lead's
  [raw-preservation acceptance and scoring clearance](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7235124783)
  cleared exactly one frozen Core analysis under the owner's bounded delegation. The owner's instruction followed in the
  project thread at 06:13 UTC.
- Head: `d8d0782c5be7142d51b8fab700ac84b1b9e6949a`, the remote branch tip, with no untracked or modified file.
  - The analysis ran in the existing checkout of the frozen branch at that exact commit. It did not use a separate
    worktree: the implementation session is bound to this one folder and may not create worktrees.
  - `main` was not merged, and nothing was rebased or resolved.
- Checks before scoring (all passed):
  - **Ancestry:** `9902a02` and `edf49fa` are ancestors of `d8d0782`. All 29 paths changed since `9902a02` are under
    `docs/evidence/`.
  - **Byte check:** 158 tracked files were compared with their `HEAD` blobs (`git hash-object --no-filters`), with no
    mismatch. They cover the EXP-010 run records, the EXP-009 records, the frozen corpus, `src/`, `benchmarks/` and
    `envs/model/`.
  - **Pins:** the comparator pins (untuned `cace5920…`, rules `7645eea7…`), the EXP-009 `results.json` pin and the
    corpus manifest pin are all equal.
  - **Clean start:** no `results.json` existed, and no model process was running.
  - **Local checks** (Core environment, `uv run --locked`, Python 3.12.16, at `d8d0782`):
    - `ruff check`: all checks passed;
    - `ruff format --check`: 178 files already formatted;
    - `pytest`: **1,019 passed, 0 skipped, 0 failed** (110.5 s).
    - Outputs are in [analysis/](analysis/).
  - **No CI result exists for `d8d0782`.** GitHub runs no `pull_request` workflow while PR #11 conflicts with `main`, so
    none is claimed. The last CI is run 46006213431 on `edf49fa`.
- Analysis:
  - **Command:** PRERUN section 8 step 5, run once (06:16:03-06:16:04 UTC, exit 0, empty stderr) with
    `--approved-commit 9902a022f50ccff65861d0f23d713e765ed9b251`.
  - **Output handling:** stdout went to a new file outside the repository, which was then moved unchanged to
    `results.json` (SHA-256 `b9000152936e1a1ed5d72c49d133a1bba110fdbbde767e4c2b924bbf140f5182`, 481,396 bytes).
  - **Recorded state:** the result records `git.commit` `d8d0782…` with `dirty: false`, and 21 input SHA-256 values.

## Pre-registered pilot outcome (D13): `improvement_demonstrated`

- **Validity:** training completed, evaluation completed, and `integrity_ok`. Every problem list is empty: corpus,
  frozen code, raw evidence preserved, training, evaluation, adapter records and comparators.
- **Primary endpoint:** semantic exact match, adapter versus the untuned Qwen3-1.7B (EXP-009 comparator), paired by
  example ID over all 160 examples.

| | Count |
|---|---|
| Adapter exact | **31 / 160** (19.4 %) |
| Untuned 1.7B exact | 12 / 160 (7.5 %) |
| Both correct / only untuned / only adapter / both wrong | 6 / 6 / 25 / 123 |
| Difference | **+11.875 pp** |
| Paired cluster bootstrap (24 lineage clusters, 2,000 replicates, seed 2026, 95 % percentile) | **[+7.41, +17.48] pp** |
| Rule 3 (more than 5 pp and interval excluding 0) | met |
| McNemar exact p (IID, illustrative only) | 0.00088 |

- **Safety:**
  - 0 unauthorized dispatches for the adapter, the untuned model and the rules;
  - the Guard reviewed all 160 adapter rows (`complete_coverage: true`);
  - no confirmation was given.

## Deployment-quality gates (D14): descriptive reading, not all passed

`every_gate_assessed_and_passed: false`. Every gate was assessable, and 6 of the 12 passed.

| Gate | Rule | Adapter | Passed |
|---|---|---|---|
| G1 candidate-schema validity | ≥ 99 % | 116 / 160 (72.5 %) | no |
| G2 semantic exact match | ≥ 85 % | 31 / 160 (19.4 %) | no |
| G3 improvement over own untuned base | ≥ 5 pp | +19 examples (+11.9 pp) | yes |
| G4 Turkish exact match | ≥ 80 % | 11 / 64 (17.2 %) | no |
| G5 Turkish within 5 pp of English | EN − TR ≤ 5 pp | 4.69 pp (EN 14/64, TR 11/64) | yes |
| G6 REJECT recall | ≥ 95 % | 17 / 30 (56.7 %) | no |
| G7 false REJECT among gold PLAN | ≤ 5 % | 4 / 85 (4.7 %) | yes |
| G8 CLARIFY recall | ≥ 90 % | 11 / 45 (24.4 %) | no |
| G9 wrong executable proposals | < 2 % | 32 / 160 (20.0 %) | no |
| G10 unauthorized dispatches | = 0 | 0 | yes |
| G11 median warm latency | ≤ 2,000 ms | 1,026 ms (160 of 160 usable) | yes |
| G12 process peak RSS and MLX peak | each ≤ 10 × 10⁹ bytes | RSS 2.67 × 10⁹, MLX 3.88 × 10⁹ | yes |

G5 passes because both languages are low, not because Turkish is good. G9 means that 32 of the 160 candidates were
schema-valid executable plans that were wrong. All of them stopped at `awaiting_confirmation` because no confirmation is
ever given.

## Descriptive comparisons (D15; no decisions)

| Paired comparison (by example ID, n = 160) | Δ | Cluster bootstrap 95 % |
|---|---|---|
| Adapter vs untuned, candidate-schema validity (116 vs 53) | +39.4 pp | [+23.3, +53.6] |
| Adapter vs untuned, semantic validity (105 vs 41) | +40.0 pp | [+27.2, +52.4] |
| Adapter vs untuned, JSON validity (132 vs 132) | 0.0 pp | [−15.7, +14.8] |
| Adapter vs rules `axiom.rules/0.1`, exact match (31 vs 73) | **−26.25 pp** | [−46.5, −7.2] |

The adapter remains clearly below the deterministic rule baseline. Its Guard and action profile:
- **Guard statuses:** 50 awaiting confirmation (18 correct plans, 32 wrong), 30 reject, 25 clarify, 28 invalid JSON,
  16 invalid candidate, 11 semantically invalid.
- **Action:** accuracy 64/160 and macro-F1 0.471. Per-class recall: PLAN 36/85, CLARIFY 11/45, REJECT 17/30.

Exact match by slice (adapter / untuned / rules):

| Slice | n | Adapter | Untuned | Rules |
|---|---|---|---|---|
| English | 64 | 14 | 5 | 35 |
| Turkish | 64 | 11 | 2 | 22 |
| Mixed | 32 | 6 | 5 | 16 |
| A held-out family | 54 | 6 | 2 | 16 |
| B novel instance | 53 | 11 | 5 | 27 |
| C familiar instance | 53 | 14 | 5 | 30 |
| Gold PLAN | 85 | 18 | — | — |
| Gold CLARIFY | 45 | 2 | — | — |
| Gold REJECT | 30 | 11 | — | — |

**Generation and resources.**
- **Generation:** 160 of 160 requests have status `ok`. Generated tokens have a median of 30 (range 21-256), and 8
  requests hit the 256-token cap; they are scored as recorded and stay in the denominator.
- **Latency:** median 1,026 ms (quartiles 830 and 1,392 ms, maximum 7,728 ms).
- **Evaluation memory:** MLX peak 3.88 × 10⁹ bytes (load 3.44 × 10⁹), RSS 2.67 × 10⁹ bytes, no swap growth, pressure
  level 1.
- **Training:** see [PILOT_TRAINING.md](PILOT_TRAINING.md). The validation loss (teacher-forced, descriptive) fell from
  2.158 to 0.148, and no checkpoint was selected.

## Interpretation and limitations

- **The pilot met its pre-registered criterion, but not deployment quality.** The 200-update LoRA pilot improved
  semantic exact match over its own untuned base by +11.9 pp, and the interval excludes zero. It failed 6 of 12
  deployment gates and stays 26 pp below the rule baseline. This result is not deployment readiness and not evidence of
  AX-01 quality.
- **The low teacher-forced loss did not carry over.** The validation loss (0.148) did not turn into high exact match
  under greedy decoding. Most gains are in output format (schema validity +39 pp) rather than correct plans, and CLARIFY
  handling is weak (11/45 recall).
- **The scope is narrow.**
  - One seed, one run, one machine.
  - The 160 validation examples come from 24 lineage clusters (largest 21), so the interval is wide and conditional on
    this corpus.
  - The corpus is AI-reviewed with no human decisions.
  - The McNemar p-value assumes independence and is illustrative only.
- **Evidence state:** no CI ran on the analysis head `d8d0782`, and the local results above are the only test evidence
  at that head. The sealed final test was not accessed.

## Next step

The lead reviews these results. Nothing further is authorized: no additional training, generation, scoring rerun,
`main` merge or sealed-test access. Integrating the branch with `main` is left to a separate review, and it must not
modify this experimental record.
