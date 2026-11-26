# AI pre-review flags on the Turkish annotation batch (not a human review)

**Status: AI review only.** These flags come from an AI-assisted pass by the technical lead over the 54 rows of
[owner_review_batch_tr.csv](owner_review_batch_tr.csv) (46 Turkish, 8 mixed), posted in the PR #8 review
([comment](https://github.com/sergthedev/axiom-local/pull/8#issuecomment-7175525079), 2026-11-25). They are
**not** human annotation, **not** independent review and **not** decisions. No example, gold label or provenance field
has been changed because of them, and none may be: only the owner's own decisions, recorded in the review CSV and
applied with `python -m axbench freeze --human-review`, change the corpus (ADR 0011, revision 2). The implementing AI
tool has added no flags of its own here.

The pass reported the batch as mostly consistent with the proposed labelling conventions, and the numeric-grouping
rejection/clarification (row 5), missing `k` / method, instruction-override and hypothesis-test policies as
reasonable. Rows flagged for the owner's decision:

| Row | Example | Flag (summarised from the review) | Owner decision needed |
|---|---|---|---|
| 24 | `validation-0025` | Request asks explicitly for mean/median of text columns (Season, Destination). | Narrow the convention, or rewrite the request to ask for numeric statistics and text counts. |
| 42 | `validation-0037` | A generic "predictive model" request is labelled as regression without saying regression. | Rewrite the request to name the method, or define a broader unsupported-prediction mapping. |
| 21 | `train-0269` | Severe, unnatural typo ("kira igderi"). | Check whether it is still inferable; rewrite if not. |
| 46 | `validation-0083` | Severe, unnatural typo ("emsafe"). | Check whether it is still inferable; rewrite if not. |
| 9 | `train-0235` | "kıdem" versus the column Years Of Experience is context-dependent. | Confirm the mapping or change the label. |
| 18 | `validation-0051` | Multiple-operations precedence is applied although the tariff mention is also ambiguous. | Confirm the precedence convention. |

To act on these, the owner fills `decision` (`approve`, `correct`, `reject`), and for `correct` a `corrected_gold`
(an `axiom.candidate/1` JSON object) and/or a `corrected_request` column, in a copy of the CSV, then runs:

```bash
uv run python -m axbench freeze --corpus data/axbench/v0.1-draft --out data/axbench/<new version> --version <new version> --decisions <filled CSV> --reviewer owner --human-review
```

The frozen manifest then records the exact number of examples with a human decision, by split and decision; every
other example stays labelled `human_review: "none"`.

## Resolution (2026-11-26, added after the decision; the text above is unchanged)

The owner delegated the review of this batch to the technical lead AI. Its decision
([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7185793594)) is an owner-authorized **AI
review**, still not human annotation: 51 rows approved as-is, rows 9, 24 and 42 corrected (request only, gold
unchanged), 0 removed; rows 18, 21 and 46 retained as explained in the dataset card. The decisions are transcribed in
[AI_REVIEW_DECISIONS.csv](AI_REVIEW_DECISIONS.csv) and applied in the frozen version `axbench/0.1-ai-reviewed`, where
every example keeps `human_review: "none"`. The draft corpus files (`train.jsonl`, `validation.jsonl`, `MANIFEST.json`) are unchanged.

