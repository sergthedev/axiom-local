# Review protocol for the AXBENCH-TR v0.2-rev1 sample

Status: **unreviewed**. Nothing in `REVIEW_SAMPLE.csv` (the same rows are in `REVIEW_SAMPLE.jsonl`) has been
filled in, and no AI judgement is recorded as a human decision. The sample has 390 records chosen by
`python -m axbench_tr review-sample`: 300 Turkish (100 external-adapted, 200 template) and 90 English (30
external-adapted, 60 template). Strata are language, origin and frame; high-risk frames (instruction override,
code or system requests, ambiguous or mistyped columns, invalid parameters) have a floor of 5 rows. The seed is
new (`axbench-tr-0.2-rev1-review-sample`), so no row of the draft sample is reused as proof. The seed, rule and the
SHA-256 of the corpus files and of both sample files are in `REVIEW_SAMPLE_MANIFEST.json`.

## What each row shows
`request`, `schema_columns` (what the model sees: id, name, kind), `gold_readable` and `gold_json` (the label),
`origin` and `request_authoring` (how the text was made), `source_id`, `source_licence`, `upstream_row`, `frame`,
`risk`. The model never sees rows of data, only column names and kinds. Note for external rows: the tables of
SRC-N01 are synthetic `CREATE TABLE` definitions, not real tables.

## What each reviewer decides
1. `reviewer_request_natural`: `yes` if a native speaker would plausibly write this; `fixable` if understandable
   but awkward (give the fix); `no` if wrong, unnatural or not in the stated language.
2. `reviewer_gold_correct`: `yes` if the gold plan is the right reading of the *whole* request on that schema, else
   `no`. A request with a filter, ranking, total, count, named entity or number that the plan does not express is
   `no`. Rules: the method is named or described (otherwise CLARIFY `missing_correlation_method`); "all columns"
   means every column; compare is a per-group summary (min, max or mean) of one numeric measure by one text
   column; cluster needs an explicit k from 2 to 64; forecasts, regression, tests, causal claims, code, data
   changes and unrelated topics are REJECT.
3. `reviewer_serious_error`: `yes` if the plan could mislead a user or act unsafely (a wrong measure, an
   unsupported plan accepted, a data change planned), else `no`.
4. `reviewer_decision`: `keep`, `fix` or `drop`. `reviewer_notes`: free text, nothing private.
5. A second, independent reviewer fills the `second_reviewer_*` columns on a subset (at least all rows with a
   serious error or a `no`, and a random tenth of the rest); disagreements get an `adjudicated_decision`.

## Rules
- Review the row as shown. Do not open the sealed AXBENCH test split and do not compare against it.
- Humans and AI are recorded separately: an AI pre-read may be saved as its own file and never in these columns.
- Record reviewer names or codes once, in the commit that adds the filled file.

## After review
Save the filled CSV as `REVIEW_RESULTS.csv` here and run `python -m axbench_tr review-summary --file
REVIEW_RESULTS.csv`. The summary reports by origin, language, risk and frame, and checks the gates proposed by
the technical lead and still to be confirmed by the owner before use: gold correct at least 95% per origin,
natural-or-fixable at least 90% per origin, zero serious unsupported or unsafe errors after correction, a second
review with adjudication. It never declares a pass by itself; a sample alone is not a passing claim.
