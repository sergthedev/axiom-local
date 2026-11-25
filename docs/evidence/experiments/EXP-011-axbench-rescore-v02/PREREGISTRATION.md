# EXP-011 pre-registration: EXP-008 rule-baseline predictions rescored with `axbench.score/0.2`

- Issue: #4 (Phase 3A), response to the technical review of PR #8
  ([comment](https://github.com/sergthedev/axiom-local/pull/8#issuecomment-7175525079)). Written and committed **before**
  the run, in the same commit as the script `benchmarks/issue4_rescore_v02.py`.
- Purpose: report the EXP-008 rule baseline under the corrected scorer (`axbench.score/0.2`) and harness
  (`axbench.evaluate/0.2`, committed in `cafce5d`) **using the same predictions**. No new predictions are scored for
  the record, no rule is changed, and nothing is tuned. EXP-008 (scorer 0.1, commit `857d4ee`) and its files stay
  unchanged; this experiment only reads them.
- Disclosure: the scorer changes were designed from the reviewer's findings and from synthetic cases, not from the
  validation predictions. EXP-008's validation results were already known when the changes were written, so this is a
  re-analysis of known predictions, not a blind measurement.
- The final test split is not generated, read or scored. No model is loaded.

Command (repository root, clean tree, after `uv sync --locked`):

```bash
uv run python benchmarks/issue4_rescore_v02.py > docs/evidence/experiments/EXP-011-axbench-rescore-v02/results.json
```

## Inputs pinned (SHA-256)

| File | SHA-256 |
|---|---|
| `data/axbench/v0.1-draft/validation.jsonl` | `87771e489e186e4b4753adcf35916f41541355ed159c8d9bfe321d9d100ed9e4` |
| `data/axbench/v0.1-draft/train.jsonl` | `fd885f1535d31983771493b6aa8e27ea3e9e0182ec58f385d96f1c686f9c2f3e` |
| EXP-008 `rules_validation.jsonl` | `40c9ba2208d016120a6683640e5dd9ce2daf4f79f8fa74afd3ab3c3655fb2e65` |
| EXP-008 `results.json` | `2105fb23959a6b6f8987ec38e926fa7378ef23235e07d507a22703319e7ffaf4` |

## Criteria

| ID | Criterion |
|----|-----------|
| C1 | All four input hashes equal the pinned values (otherwise the script aborts and nothing is scored). |
| C2 | The frozen `axiom.rules/0.1` router, run again on the committed validation file, produces the same `status` and `raw_output` for 160 of 160 examples as the committed EXP-008 predictions (timings excluded). |
| C3 | Every metric that differs between scorer 0.1 and 0.2 is accounted for by the listed per-example changes or by a documented definition change (macro F1 class handling, validity requirement). |
| C4 | After the Guard layer (with request contexts), zero confirmations and zero unauthorized dispatches. |

Measurements (no threshold): every metric of the EXP-008 table under scorer 0.2, the side-by-side comparison
produced by the script, and the new `raw_action_recognition` and `action_macro_f1_classes` fields. Train is scored
again with the regenerated rule outputs as the development reference.

Prediction (not a criterion): no metric changes. EXP-008 recorded 160 of 160 semantically valid validation outputs,
every action class had predicted positives, and no example has accepted variants, so none of the 0.1 defects is
exercised by these predictions. If this holds, the corrected scorer leaves the EXP-008 numbers unchanged for this
system; the defects would matter for systems that emit invalid candidates (such as untuned models).
