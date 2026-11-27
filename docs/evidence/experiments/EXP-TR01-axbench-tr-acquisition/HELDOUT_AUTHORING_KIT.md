# Kit for independently authored Turkish held-out tasks (for human authors; nothing here is authored)

An AI cannot supply the independence this set needs: the training data are AI-authored, so tasks written by the same
system share its habits. This kit only prepares the work. **Do not tune on the resulting tasks, and do not add them to
training.** Status: not started; needs named human authors.

## What to author
- Turkish requests written by native speakers, each for one licensed economic table schema chosen from the
  İzmir (CC BY 4.0) or World Bank (CC BY 4.0) tables in `data/axbench-tr/schemas/real_table_schemas.json`
  (column names and kinds only). Eurostat, ECB, OECD and TCMB layouts only if the owner confirms their terms.
- Held out by **source and intent family**: choose schemas whose table family and source layout appear in no training
  record (`python -m axbench_tr validate` reports the families used); pick intent phrasings the authors have not
  seen in the review sample. Record both in the file.
- A spread of PLAN (inspect, relate, compare, cluster), CLARIFY (nine reasons) and REJECT (nine reasons), with some
  requests that look like a supported plan but are not (a total, a filter, a ranking, a forecast).
- Each task: request text, schema id, gold JSON in `axiom.candidate/1`, author code, date, and one line on why the gold
  is correct. No private data; no request copied from a dataset or a model output; no peeking at the sealed test.

## Process
1. Authors write; a second person checks gold against the rules in `v0.2-rev1/review/REVIEW_PROTOCOL.md`.
2. Run `axbench.audit.gold_problems` on each task (consistency with the contract, not human judgement).
3. Freeze the file; record its SHA-256 and author roster in the PR before any training is approved.
4. Use it only for evaluation of a model that never saw it, under frozen comparators and criteria the owner approves.

Suggested size: enough that each origin family and action has at least 10 tasks (the owner decides the final number).
