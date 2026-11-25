# Experiment manifest: EXP-011 EXP-008 rule-baseline predictions rescored with `axbench.score/0.2`

- Issue / PR: #4 (Phase 3A), PR #8 review response
- Date: 2026-11-26
- Status: run (one machine, one run)
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `5519973` before the run

## Provenance
- Commit (full SHA): `5519973529f86d8d7239ac2f870e3a9c5c72f75d` (recorded by the script as `git.commit`, tree clean)
- Hardware and OS: MacBook Air (Mac17,3), Apple M5, 16 GB; macOS 27.0.1, arm64
- Python 3.12.16; dependencies from `uv.lock` via `uv sync --locked`; no model and no new dependency
- Inputs: committed `validation.jsonl` and `train.jsonl` of `axbench/0.1-draft`, EXP-008 `rules_validation.jsonl` and
  `results.json`, all matching the pinned SHA-256 values; EXP-008 files were only read
- System: `axiom.rules/0.1` (unchanged), harness `axbench.evaluate/0.2`, score `axbench.score/0.2`, Guard policy
  `axiom.guard-policy/2`; cluster bootstrap 2000 replicates, seed 2026, over lineages

## Execution
- Command: `uv run python benchmarks/issue4_rescore_v02.py > docs/evidence/experiments/EXP-011-axbench-rescore-v02/results.json`
- Exit code: 0 (first run; no failure, no rerun)
- Output: [results.json](results.json) (side-by-side comparison, full 0.2 report with per-example scores for
  validation, train summary; no test text)

## Results (measured only)

| ID | Result | Observed |
|----|--------|----------|
| C1 | met | all four input hashes equal the pinned values |
| C2 | met | the rerun of `axiom.rules/0.1` reproduced `status` and `raw_output` for 160 of 160 committed predictions |
| C3 | met | no metric changed, so nothing needed accounting for; 0 validation examples changed exact-match outcome |
| C4 | met | 0 confirmations, 0 unauthorized dispatches; Guard statuses 56 awaiting confirmation (30 correct, 26 wrong), 89 clarify, 15 reject, as in EXP-008 |

Under scorer 0.2 every compared validation metric equals the EXP-008 value: exact match 72 / 160, action accuracy
114 / 160, macro F1 0.701 (all three classes included; none excluded), operation accuracy 33 / 85, clarify recall
45 / 45, reject recall 15 / 30, 56 validator-accepted plans, 26 wrong executable proposals. New field: raw action
recognition 114 / 160 (equal to action accuracy, because all 160 outputs are semantically valid). Train (development
reference, rules tuned on it): 400 / 400 under both versions.

The pre-registered prediction held: the 0.1 defects are not exercised by these predictions (all valid, every class
predicted, no accepted variants). The EXP-008 numbers therefore stand for this system under both scorer versions.
They will differ for systems that emit semantically invalid candidates, which is why EXP-009 onwards uses 0.2.

## Limitations
- A re-analysis of known predictions, not a blind measurement; it shows the correction does not move this
  baseline, nothing more.
- The corpus is still the AI-generated draft pending owner annotation review; no human label review is reflected.
- One machine; the comparison is deterministic given the pinned inputs.
