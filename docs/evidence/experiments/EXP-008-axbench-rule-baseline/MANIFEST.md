# Experiment manifest: EXP-008 AXBENCH v0.1-draft audit and bilingual rule baseline on validation

- Issue / PR: #4 (Phase 3A), draft PR linked to Issue #4
- Date: 2026-11-26
- Status: run (one machine, one run)
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `409b46c` before the run

## Provenance
- Commit (full SHA): `857d4ee2c779ca4eec80154dab63558ccb11ae98` (pre-registration commit `409b46c` plus one commit that
  only adds `.gitattributes` rules keeping AXBENCH corpus bytes identical on Windows checkouts; recorded by the script
  as `git.commit`, tree clean)
- Hardware and OS: MacBook Air (Mac17,3), Apple M5, 16 GB; macOS 27.0.1, arm64
- Python 3.12.16; dependencies from `uv.lock` via `uv sync --locked` (uv 0.13.10); no model and no new dependency
- Dataset: AXBENCH `axbench/0.1-draft`, synthetic, MIT, [dataset card](../../../../data/axbench/DATASET_CARD.md);
  `train.jsonl` SHA-256 `fd885f1535d31983771493b6aa8e27ea3e9e0182ec58f385d96f1c686f9c2f3e` (400),
  `validation.jsonl` SHA-256 `87771e489e186e4b4753adcf35916f41541355ed159c8d9bfe321d9d100ed9e4` (160);
  final test split not generated, read or scored
- System: rule router `axiom.rules/0.1`, harness `axbench.evaluate/0.1`, score `axbench.score/0.1`; cluster bootstrap
  2000 replicates, seed 2026, over lineages

## Execution
- Command: `uv run python benchmarks/issue4_axbench_baseline.py --predictions docs/evidence/experiments/EXP-008-axbench-rule-baseline/rules_validation.jsonl > docs/evidence/experiments/EXP-008-axbench-rule-baseline/results.json`
- Exit code: 0 (first run; no failure, no rerun)
- Outputs: [results.json](results.json) (all metrics, per-example scores for validation, no test text),
  [rules_validation.jsonl](rules_validation.jsonl) (raw validation predictions)

## Results (measured only)

Criteria:

| ID | Result | Observed |
|----|--------|----------|
| A1 | met | regenerated, committed and manifest SHA-256 identical for both files; no `test.jsonl` under `data/axbench/` |
| A2 | met | audit passed; 100 connected components, 0 spanning splits; 0 cross-split pairs at or above Jaccard 0.8 (max 0.4833 train~validation, 0.4722 train~test, 0.3370 validation~test) |
| A3 | met | no contamination finding |
| R1 | met | 160 of 160 status `ok`, JSON-valid and `axiom.candidate/1`-valid (also 160 of 160 semantically valid against the schema) |
| R2 | met | 0 confirmations, 0 unauthorized dispatches; Guard statuses: 56 awaiting confirmation, 89 clarify, 15 reject |

Measurements, validation (n = 160, full denominator):

| Metric | Value |
|---|---|
| Primary semantic exact match | **72 / 160 = 45.0 %**; Wilson 95 % (IID, illustrative) 37.5-52.7 %; cluster bootstrap 95 % over 24 lineages 28.5-62.5 % (largest lineage 21 examples) |
| Action accuracy / macro F1 | 114 / 160 = 71.3 % / 0.701 |
| PLAN: precision, recall | 0.964, 0.635 (85 gold PLAN) |
| Operation accuracy among gold PLAN | 33 / 85 = 38.8 % |
| Slot accuracy | inspect columns 12/21, relate pair 9/22, relate method 9/22, compare group_by 0/21, compare measure 0/21, cluster features 9/21, cluster k 10/21 |
| Clarify recall / reason accuracy | 45 / 45 = 100 % / 37 / 45 = 82.2 % |
| Reject recall / reason accuracy | 15 / 30 = 50.0 % / 5 / 30 = 16.7 % |
| False rejection among supported requests | 0 / 130 |
| Validator-accepted plans | 56 / 160; 30 of them exact |
| Wrong executable proposals | 26 / 160 = 16.3 % (24 with gold PLAN, 2 with gold REJECT `hypothesis_test`); all held at `awaiting_confirmation`, none dispatched |
| By language (exact) | TR 21/64, EN 35/64, mixed 16/32 |
| By stratum (exact) | A held-out family 15/54, B novel instance 27/53, C familiar instance 30/53 |
| Latency | median 0.69 ms per request (pure Python, this Mac; not a model latency) |

Development reference, train (n = 400; the rules were tuned on it): 400 / 400 exact, 0 wrong executable proposals,
0 dispatches.

Largest error groups on validation (from `per_example`): all 21 gold `compare` requests were proposed as `inspect`
(no validation compare phrasing matched a training cue); 7 relate requests of lineage `L-reld-03` (Spearman
described as "rank-based correlation of a monotonic relationship") got `missing_correlation_method`, and 7 cluster
requests of lineage `L-clu-04` (count given as "N gruba") got `missing_cluster_count`; inspect, relate and cluster
requests (17 in all, including 5 "all columns" requests) got `missing_column` or `column_not_found`; of 30 REJECT requests, 15 were rejected (5 with the right reason,
10 as `out_of_scope`), 13 were asked for clarification and 2 (`t-test`) were proposed as plans.

Independent reference: the scoring functions have unit tests with hand-derived cases and the interval helpers are
checked against SciPy (`tests/test_axbench.py`); no external benchmark exists for this task.

## Limitations and failures
- One run on one machine; the rules are deterministic, so a rerun on the same commit should give the same outputs.
- The rule baseline is not blind: the same tool wrote the templates (including validation) and the rules. The
  contamination audit removes held-out-only vocabulary but cannot remove the author's knowledge.
- Validation is synthetic, unreviewed and small (24 lineages); the cluster bootstrap interval is wide, and examples
  are not independent, so the Wilson interval is illustrative only.
- The pre-registered prediction (validation clearly below train, lowest in stratum A) held; this is a check of the
  generalization gap of keyword rules, not evidence about any model.
- The 26 wrong executable proposals show why the Guard confirmation step is needed: a schema-valid plan can be the
  wrong plan. The experiment does not measure whether a person would notice the wrong interpretation.
