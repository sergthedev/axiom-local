# Review protocol for the AXBENCH-TR Turkish sample

Status: **unreviewed**. This file explains how to review `REVIEW_SAMPLE.csv` (the same rows are in
`REVIEW_SAMPLE.jsonl`). The sample has 300 Turkish records chosen by `python -m axbench_tr review-sample`
(seed and rule in `REVIEW_SAMPLE_MANIFEST.json`); the manifest also holds the SHA-256 of the corpus files it
was drawn from and of both sample files. Nothing here has been filled in.

## What each row shows
`request` (the Turkish text), `schema_columns` (what the model sees: id, name, kind), `gold_readable` and
`gold_json` (the label), `origin` and `request_authoring` (how the text was made), `source_id`, `source_revision`,
`source_licence` and `upstream_row` (where it came from), and `frame` (what the record is meant to test).
The model never sees rows of data, only column names and kinds.

## What to decide (the last four columns)
1. `reviewer_request_natural`: `yes` if a Turkish speaker would plausibly write or say this; `fixable` if it is
   understandable but awkward (note the fix); `no` if it is wrong, unnatural or not Turkish.
2. `reviewer_gold_correct`: `yes` if the gold plan is the right reading of the request on that schema under these
   rules, else `no`:
   - the method is named or described (otherwise the right answer is CLARIFY `missing_correlation_method`);
   - "all columns" means every column;
   - compare is a per-group summary of one numeric measure by one text column;
   - cluster needs an explicit k from 2 to 64 (otherwise CLARIFY);
   - a request for forecasts, regression, tests, causal claims, code, data changes or unrelated topics is REJECT.
3. `reviewer_decision`: `keep`, `fix` (the request or label can be repaired; say how in the notes) or `drop`.
4. `reviewer_notes`: free text; do not paste anything private.

## Rules for the reviewer
- Review the row as shown. Do not open the sealed AXBENCH test split and do not compare against it.
- Machine-translated rows (`origin` = `external_adapted`) are expected to be stilted; mark `fixable` or `no`
  only if the meaning is lost or wrong.
- Record your reviewer name or code once in the commit that adds the filled file, not in the cells.

## After review
Save the filled CSV as `REVIEW_RESULTS.csv` in this folder and run
`python -m axbench_tr review-summary --file REVIEW_RESULTS.csv` to count decisions. A suggested acceptance rule,
to be confirmed by the owner before training: at least 95% `gold_correct = yes` over the 200 template rows, and no
more than 10% `drop` over the 100 external rows. Rejected and fixed rows feed back into templates and adapters;
the corpus is then rebuilt, and the sample is redrawn with a new seed so the same rows are not reused as proof.
