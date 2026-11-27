# Dataset card: AXBENCH-TR v0.2-draft

Status: **draft**. Every record was authored or adapted by software (templates, AI-assisted template
design, machine translation upstream). **No record has been reviewed by a human**
(`provenance.human_review` is `"none"` on every record). The draft has not been used to train a model.
It is not a benchmark: it is a training and validation corpus for the AX-01 router (Qwen3-1.7B).

Everything below is read from `v0.2-draft/MANIFEST.json`, `AUDIT.json` and `TOKEN_STATS.json`, which
`python -m axbench_tr build` writes. Re-run the build after any change; the numbers here are for the
committed files.

## Purpose and task

Map one data-analysis request to one `axiom.candidate/1` object over an `axiom.schema-view/1`
projection (ids, names, numeric or text kind; no rows). Three actions: `PLAN` (inspect, relate,
compare, cluster), `CLARIFY` (nine reasons), `REJECT` (nine reasons). The record layout, labelling
conventions (method named or described; inspect-all means every column; compare is a per-group
summary; cluster needs an explicit k from 2 to 64) and the deterministic engine are unchanged from
AXBENCH v0.1. The engine and the model contracts were not modified.

## Contents

| | train | validation | total |
| --- | ---: | ---: | ---: |
| Records | 6,942 | 1,683 | 8,625 |
| Turkish / English | 5,606 / 1,336 | 1,354 / 329 | 6,960 / 1,665 (80.7% / 19.3%) |
| PLAN / CLARIFY / REJECT | 3,845 / 1,422 / 1,675 | 934 / 342 / 407 | |
| Synthetic template / external adapted | 5,400 / 1,542 | 1,300 / 383 | |
| Economic / non-economic tables | 3,393 / 3,549 | 983 / 700 | |
| Distinct request texts | 6,064 | 1,467 | 7,531 across the corpus |

About 13% of records repeat the request text of another record over a different table; the table and
therefore the gold ids differ. Repetition is reported, not hidden.

Schemas come from real published tables: 255 tables in the inventory (`schemas/real_table_schemas.json`,
names and kinds only), of which 200 are usable (one task slot at least). Sources: İzmir Metropolitan
Municipality open data (CC BY 4.0), World Bank indicators, Eurostat, ECB, OECD and TCMB column layouts,
and the Cognipeer text-to-SQL table definitions. No rows from the schema sources are inside the examples.
Tables were partitioned between splits by a hash of their group, so no table group appears in both
splits.

## Provenance classes

| Class | Where | Notes |
| --- | --- | --- |
| `synthetic_template` | 6,700 records | Requests rendered from 122 AI-authored templates on real headers. Turkish headers stay Turkish; English requests are 20% by construction. |
| `external_adapted` (machine-translated) | 1,925 records | Cognipeer prompts (SRC-N01, translated by the publisher) and finance negatives (SRC-N03). Turkish text is machine-made. |

Observations (the economic CSVs under `economic/`) are kept apart from the intent examples: no example
contains an observation, and the model never sees them. They exist so a later task can check plans
against real data. ECB, OECD and TCMB observations are not redistributed; only their headers are used.

## Splitting, validation and contamination

- Each request idea (`lineage_id`) is assigned to one split before any text is generated; the
  validator found 0 shared lineages, 0 shared request texts and 0 shared table groups across splits.
- Deterministic validation (`axbench.audit.gold_problems` plus the extra mutation-tested checks in
  `axbench_tr/validate.py`) found 0 problems in 8,625 records. A record that fails refuses the build.
- Overlap with AXBENCH v0.1 train and validation (560 requests): 0 exact normalised matches, 0
  near-duplicate pairs at 4-gram Jaccard 0.8. Closest template pair: similarity 0.6957 (0 pairs at or above 0.8).
- Near-duplicate audit (character n-gram TF-IDF cosine on normalised requests, a lexical proxy, not an
  embedding model): across splits 8 pairs reach 0.8, none reach 0.9; all 8 are short requests dominated by the
  same column names from different lineages. Within a split, 2,898 pairs of the same lineage reach 0.9 (the
  template varies only in a column name) and 47 pairs of different lineages reach 0.9. This shows how narrow the
  template phrasing is.
- **The sealed AXBENCH test split was never accessed.** The owner can measure overlap with the sealed
  requests without exposing them: `python -m axbench_tr contamination --file <requests.jsonl>` prints
  counts only. That check has not been run.
- No external benchmark test split (TAT-QA, FinQA, ConvFinQA, DataBench, StatQA) is in any record.

## Sequence lengths (Qwen3-1.7B tokenizer, chat template, `enable_thinking=False`)

Rendering follows EXP-010: the template prompt with the generation prompt, plus the compact gold JSON and
`<|im_end|>`. Per-example lengths are in `v0.2-draft/token_lengths_per_example.jsonl`.

| Split | Examples | Total tokens | Mean | Median | p95 | p99 | Max | Over 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6,942 | 3,487,013 | 502.3 | 469 | 804 | 1,076 | 1,219 | 91 |
| validation | 1,683 | 845,727 | 502.5 | 486 | 816 | 847 | 890 | 0 |

About 94% of tokens are the fixed instruction and schema prompt (train prompt tokens 3,290,807); the answer
averages 28.3 tokens. Turkish and English examples have about the same length.

**The previous 1,024-token configuration is too short for this corpus.** 91 train examples (1.3%) are longer,
all on two real wide tables (37 and 43 columns); the longest is 1,219. Truncating them would cut the schema and
corrupt the label, so none may be truncated or silently dropped. The corpus fits a 2,048-token limit with no
example over it. `benchmarks/axbench_tr_token_stats.py --max-seq-length 2048` is a gate that exits 1 if any
sequence is longer; the smoke protocol uses it. Reproduce with that script (needs `tokenizers` and `jinja2`,
not in the lockfile).

## Human-review sample

`v0.2-draft/review/` holds a reproducible 300-record Turkish sample for native-speaker review: 100
external-adapted and 200 template records, stratified by origin and 23 frames (floor 3 per stratum, one record per
lineage preferred), with provenance, upstream licence, gold plan in readable and JSON form, and empty reviewer
columns. See `REVIEW_PROTOCOL.md` there. `python -m axbench_tr review-sample` regenerates it byte for byte.

## Known limitations

1. No human review. Gold labels are deterministic functions of templates, but the naturalness and
   correctness of Turkish phrasing has not been checked by a native speaker.
2. Cognipeer Turkish text is machine-translated; some phrasing will be stilted.
3. Templates were written by an AI assistant from the AXBENCH v0.1 conventions, so the template
   family is narrow (122 templates). Models can learn the templates rather than the task.
4. The sealed test overlap check has not been run.
5. TÜİK, TCMB EVDS, ILOSTAT, IMF and İBB data could not be acquired under their access limits; the
   other Turkish ministries and agencies were probed and not found to offer open machine-readable
   data suited to this pass. Real Turkish official table schemas are therefore mostly municipal (İzmir),
   World Bank and Eurostat layouts, not national-statistics-office layouts. See `DATA_SOURCES.md`.
6. Only the four supported operations are covered; this is not a general table-QA corpus.

## Licences

The corpus files are AI-generated derivatives. Inputs: İzmir and World Bank (CC BY 4.0), Cognipeer
and the Gretel original (Apache-2.0), AlicanKiraz0 (MIT per card; the README badge says Apache-2.0),
Eurostat (reuse policy). The repository code is MIT. Attribution is in `THIRD_PARTY_NOTICES.md`; the
per-source terms, the derivative-rights summary and the evidence read (page, SHA-256, excerpt; 8 checks, all
found) are in `DATA_SOURCES.md` and `sources/LICENCE_EVIDENCE.json`. Dataset-card licence tags were read, not
independently verified, and the summary has had no legal review. The upstream Gretel data were model-generated and
the terms of that model were not checked.

## Reproduction

```
uv sync --locked
PYTHONPATH=src python -m axbench_tr acquire     # raw files -> data/axbench-tr/_raw (ignored)
PYTHONPATH=src python -m axbench_tr verify      # re-hash against ACQUISITION.json
PYTHONPATH=src python -m axbench_tr schemas
PYTHONPATH=src python -m axbench_tr economic
PYTHONPATH=src python -m axbench_tr build       # needs pyarrow==26.0.1 for the Cognipeer part
PYTHONPATH=src python -m axbench_tr validate
PYTHONPATH=src python -m axbench_tr licences    # re-fetch licence pages
PYTHONPATH=src python -m axbench_tr review-sample
PYTHONPATH=src python -m axbench_tr docs
```
Two builds from the same raw files give byte-identical corpus, manifest and audit (checked by hashing).

`pyarrow` is a build-time dependency only and is not in `pyproject.toml` or `uv.lock`.
