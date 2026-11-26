# Experiment manifest: EXP-013 rule baseline on the frozen AI-reviewed AXBENCH corpus

- Issue / PR: #4 (Phase 3A follow-up: annotation decisions and freeze)
- Date: 2026-11-26
- Status: run (one machine, one run)
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `b9f5d2e` before the run

## Provenance
- Commit (full SHA): `b9f5d2eb13030b49bdb3a05016493738f454c1b2` (recorded by the script as `git.commit`, tree clean)
- Hardware and OS: MacBook Air (Mac17,3), Apple M5, 16 GB; macOS 27.0.1, arm64; Python 3.12.16, `uv sync --locked`
- Corpus: `axbench/0.1-ai-reviewed`, manifest SHA-256 `b8b6d39f…16239c` (pinned and verified), train 400, validation
  160; owner-authorized AI review, **not human annotation** (exploratory corpus)
- System: `axiom.rules/0.1` (unchanged), harness `axbench.evaluate/0.2`, scorer `axbench.score/0.2`; cluster bootstrap
  2000 replicates, seed 2026, over 24 lineages; final test not generated, read or scored; no model

## Execution
- Command: `uv run python benchmarks/issue4_rules_frozen.py --predictions docs/evidence/experiments/EXP-013-rules-frozen-ai-reviewed/rules_validation.jsonl > docs/evidence/experiments/EXP-013-rules-frozen-ai-reviewed/results.json`
- Exit code: 0 (first run; no failure, no rerun)
- Outputs: [results.json](results.json), [rules_validation.jsonl](rules_validation.jsonl)

## Results (measured only)

| ID | Result | Observed |
|----|--------|----------|
| F1 | met | manifest hash equal to the pinned value; 400 / 160 records with verified file hashes |
| F2 | met | 160 of 160 status `ok`, schema-valid and semantically valid |
| F3 | met | 0 confirmations, 0 unauthorized dispatches; Guard statuses 56 awaiting confirmation (30 correct, 26 wrong), 88 clarify, 16 reject |
| F4 | met | one example differs from EXP-011: `validation-0037` (now explicitly a linear regression request) changed from CLARIFY `missing_operation` to the gold REJECT `regression`; `validation-0025` keeps the same wrong prediction (inspect of two of the three columns) |

| Metric (validation, n = 160) | Value |
|---|---|
| Primary semantic exact match | **73 / 160 = 45.6 %**; cluster bootstrap 95 % 29.1-63.3 %; Wilson 95 % (IID, illustrative) 38.1-53.4 % |
| Action accuracy / macro F1 | 115 / 160 / 0.713 |
| Operation accuracy among gold PLAN | 33 / 85 |
| Clarify recall / reason accuracy | 45 / 45 / 37 / 45 |
| Reject recall / reason accuracy | 16 / 30 / 6 / 30 |
| Gold Compare | 0 / 21 |
| Wrong executable proposals | 26 / 160 (24 gold PLAN, 2 gold REJECT); none dispatched |
| By language | TR 22 / 64, EN 35 / 64, mixed 16 / 32 |
| By stratum | A held-out family 16 / 54, B novel instance 27 / 53, C familiar instance 30 / 53 |

Train (development reference; the rules were tuned on the draft train split): 400 / 400, 0 dispatches.

The prediction (72 ± 2) held. **73 / 160 is the rule-baseline reference on `axbench/0.1-ai-reviewed`;** the historical
72 / 160 (EXP-008, EXP-011) belongs to the draft and is not replaced or edited.

## Limitations
- The corpus correction was made by an owner-authorized AI review; no human has reviewed any example.
- Not blind: the rules, templates and experiments come from the same implementing tool, and the draft outcomes of the
  changed examples were known.
- The 1-example difference is far inside the cluster-bootstrap uncertainty; it reflects the request correction, not a
  change in the rules.
