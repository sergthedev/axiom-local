# Experiment manifest: EXP-012 train-only diagnosis of the rule baseline's Compare cues

- Issue / PR: #4 (Phase 3A), PR #8 review response (rule baseline concern)
- Date: 2026-11-26
- Status: run (one machine, one run); **exploratory, not pre-registered**
- Why exploratory: the question was prompted by the validation result already recorded in EXP-008 (all 21 gold
  `compare` requests proposed as `inspect`). This analysis is therefore validation-informed in its motivation. It reads
  only the training split and changes nothing.

## Provenance
- Commit (full SHA): `a2c50e1fb0ef7419a6e69ee9f0bd4a4128ea305b` (recorded by the script as `git.commit`, tree clean)
- Hardware and OS: MacBook Air (Mac17,3), Apple M5, 16 GB; macOS 27.0.1, arm64; Python 3.12.16, `uv sync --locked`
- Input: committed `data/axbench/v0.1-draft/train.jsonl`, SHA-256 `fd885f15…2f3e` (verified by the script). No
  validation or test record, template or prediction was read by the script or consulted for this analysis.
- System: `axiom.rules/0.1`, unchanged

## Execution
- Command: `uv run python benchmarks/issue4_rules_compare_diagnosis.py > docs/evidence/experiments/EXP-012-rules-compare-train-diagnosis/results.json`
- Exit code: 0 (first run on a clean tree; an identical dry run preceded it on the same commit's working copy)
- Output: [results.json](results.json) (cue counts by lineage; no request text)

## Method
For the 86 train examples whose gold involves Compare (52 `compare` PLANs in lineages L-cmp-01/02/03 and 34 CLARIFY
examples with a `compare` operation hint), record which `compare` and `inspect` operation cues of the rules fire.
Leave-one-lineage-out: an example counts as covered when it contains a Compare cue that also fires in at least one
*other* train Compare lineage, which approximates whether the cue vocabulary would transfer to a lineage the rules
were not written against.

## Results (measured only)

| Quantity | Observed |
|---|---|
| Compare cues in the rule list / matched in train | 11 / 11 |
| Cues that occur in exactly one train lineage | 7 of 11 (`comparison`, `differ`, `for each`, `group`, `grubu için`, `gruplarına göre`, `karşılaştırma`) |
| Leave-one-lineage-out coverage, all 86 | 55 / 86 |
| Leave-one-lineage-out coverage, 52 `compare` PLANs | 24 / 52: L-cmp-01 24/24, L-cmp-02 **0/21**, L-cmp-03 **0/7** |
| `compare` PLANs that also contain an `inspect` cue | 21 / 52, all in L-cmp-02 (`summarize`, `özet…`); there, Compare wins only because `for each` / `group` / `grubu için` fire, and those occur in no other lineage |

## Diagnosis
On train alone, two of the three Compare lineages are recognised only through words that no other lineage uses. The
rules therefore reproduce the training phrasings of Compare instead of a transferable signal; when a new phrasing
lacks these words, the rule order falls through to `inspect` if any summary-type word is present. This mechanism is
consistent with the EXP-008 validation outcome, but this record does not confirm it on validation, and it must not be
used to tune the rules against validation.

## Decision and limitation
- EXP-008 and `axiom.rules/0.1` stay unchanged and remain the recorded baseline, with this limitation documented.
- No new rule version is introduced in this PR. Any later, more competent baseline (for example, one that uses the
  schema-type pattern of one text and one numeric mention as a Compare signal, developed with leave-one-lineage-out on
  train) would be a separate system, pre-registered before any evaluation, and its validation numbers would be labelled
  validation-informed, because the Compare weakness is known from validation. It is not done now and is not artificially
  weakened either.
