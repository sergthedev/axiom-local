# EXP-013 pre-registration: rule baseline on the frozen AI-reviewed AXBENCH corpus

- Issue: #4. Requested by the technical lead's annotation decision
  ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7185793594)): "rerun rules on the same
  frozen validation set rather than treating historical 45% as the new score". Written and committed **before** the run,
  in the same commit as `benchmarks/issue4_rules_frozen.py`.
- Corpus: `axbench/0.1-ai-reviewed` (committed in `773271c`), manifest SHA-256
  `b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c`, train `cf56e6da…01bd9a` (400), validation
  `80457d95…eed3b8` (160). Owner-authorized **AI review**, not human annotation; exploratory corpus.
- System: `axiom.rules/0.1` unchanged (EXP-008), harness `axbench.evaluate/0.2`, scorer `axbench.score/0.2`, Guard
  policy 2. One run. The rules are not changed after this run within this experiment.
- Disclosure: the validation outcome of the rules on the draft is known (EXP-008/EXP-011), including the two changed
  validation examples (`validation-0025` and `validation-0037`, both wrong before). This is a re-measurement on the
  corrected frozen file, not a blind one.
- The final test split is not generated, read or scored. No model is loaded.

Command (repository root, clean tree, after `uv sync --locked`):

```bash
uv run python benchmarks/issue4_rules_frozen.py --predictions docs/evidence/experiments/EXP-013-rules-frozen-ai-reviewed/rules_validation.jsonl > docs/evidence/experiments/EXP-013-rules-frozen-ai-reviewed/results.json
```

## Criteria

| ID | Criterion |
|----|-----------|
| F1 | The frozen corpus loads with the pinned manifest hash and verified file hashes (400 / 160). |
| F2 | All 160 validation outputs have status `ok` and are valid `axiom.candidate/1` objects. |
| F3 | Zero confirmations and zero unauthorized dispatches after the Guard layer. |
| F4 | Every validation example whose prediction or exact-match outcome differs from EXP-011 is one of the two examples whose request changed (`validation-0025`, `validation-0037`). |

Measurements (no threshold): the full AXBENCH metric set on the frozen validation set with the cluster bootstrap (2000
replicates, seed 2026), slices, Guard statuses; train as the development reference. This is the rule-baseline
reference for EXP-009 on this corpus version; the historical 72 / 160 belongs to the draft.

Prediction (not a criterion): exact match stays within 72 ± 2 of 160, because only two validation requests changed.
