# Dataset card: AXBENCH-TR v0.2-rev1

Status: **draft revision 1, not cleared for training or for any product use**. Every record was authored or adapted by software
(templates written by an AI assistant, rule-based adaptation of upstream request texts, machine translation
upstream). **No record has been reviewed by a human** (`provenance.human_review` is `"none"` on every record).
It is a training and validation corpus for the AX-01 router (Qwen3-1.7B), not a benchmark.

`v0.2-draft/` (8,625 records) is the first corpus. It is kept unchanged as a record and is superseded: its adapted
plans were derived from SQL shape and included SUM, filters and rankings, and its split checks covered template
records only. This card describes `v0.2-rev1/`. Numbers are read from its `MANIFEST.json`, `AUDIT.json` and
`token_lengths_per_example.jsonl`; `python -m axbench_tr build` writes the first two.

## Purpose and task

Map one data-analysis request to one `axiom.candidate/1` object over an `axiom.schema-view/1` projection (ids,
names, numeric or text kind; no rows). Actions: `PLAN` (inspect, relate, compare, cluster), `CLARIFY` (nine
reasons), `REJECT` (nine reasons). Labelling conventions and the deterministic engine are unchanged from AXBENCH
v0.1; the engine and model contracts were not modified. The supported summaries are the Core descriptive kernels;
a request for a sum or a count is not a PLAN here.

## Contents

| | train | validation | total |
| --- | ---: | ---: | ---: |
| Records | 5,873 | 1,396 | 7,269 |
| Turkish / English | 4,649 / 1,224 | 1,108 / 288 | 5,757 / 1,512 (79.2% / 20.8%) |
| PLAN / CLARIFY / REJECT | 3,006 / 1,422 / 1,445 | 717 / 342 / 337 | |
| Synthetic template / external adapted | 5,400 / 473 | 1,300 / 96 | 6,700 / 569 |
| Tagged economic / non-economic | 3,239 / 2,634 | 1,022 / 374 | 4,261 / 3,008 |
| Distinct request texts | 4,979 | 1,177 | 6,156 across the corpus |

**About 92% of the records are template-derived**, and 122 AI-authored templates generate them. The corpus is wide
in tables (234 table families) and narrow in phrasing. 58.6% of records are tagged economic by their table. This is
a stated limitation, not something quota padding should hide: more genuine Turkish economic diversity needs real
authored tasks (see "What is still needed").

The 569 external records (SRC-N01 only) are: 375 data-modification REJECTs (300 Turkish, 75 English), 182 compare
PLANs (91 per language) and 12 inspect PLANs (6 per language). Turkish and English prompts of one upstream row share
a lineage and a split. The 97 Turkish compare and inspect PLANs carry the tag `translation_unverified` (see below).

## Sources actually used

7 of the 35 surveyed sources contribute to the corpus or data files: SRC-N01 (request texts), SRC-O01, O02, O03,
O04, O05, O06 (table names and column names; observations from O01 and O06 under `economic/`), plus the Qwen
tokenizer files (SRC-M01) for token counting only. The count of 35 is surveyed sources, not used ones; the rest are
rejected, blocked or unused, with reasons in `DATA_SOURCES.md`. **SRC-N03** (a Turkish finance question dataset) is
**quarantined** because its licence evidence conflicts (MIT on the Hub tag, Apache-2.0 on the README badge): it
contributes nothing to v0.2-rev1. TÜİK, TCMB EVDS, ILOSTAT, IMF and İBB could not be acquired under their access
limits.

Schemas: 255 tables in the inventory (`schemas/real_table_schemas.json`, names and kinds only), 200 usable. Real
tables are the İzmir Metropolitan Municipality open data (CC BY 4.0), World Bank indicators, and the Eurostat, ECB,
OECD and TCMB column layouts. **The SRC-N01 tables are not real**: they are synthetic `CREATE TABLE` definitions
from a model-generated text-to-SQL dataset. No schema source contributes rows to any example.

## Provenance classes

| Class | Records | Notes |
| --- | ---: | --- |
| `synthetic_template` | 6,700 | Requests rendered from AI-authored templates on real headers. Turkish headers stay Turkish. |
| `external_adapted` | 569 | Upstream request texts (LLM-written; the Turkish side machine-translated), gold derived by rule, a synthetic schema. |

### How adapted plans are kept honest

The router sees only a schema and a request, so a plan derived from an upstream SQL query is correct only if the
**whole request** says what the plan does. An upstream row becomes a PLAN only if its English request is a plain
"average / minimum / maximum of a measure [per group]" question whose measure and group words match the bound
columns (`axbench_tr/intent.py`). SUM and COUNT are never accepted. Requests with a number, a quote, a proper
noun, a filter or other clause, a ranking, or different measure words are excluded with a recorded reason
(`AUDIT.json` > `external_adapter_audit`: for example 537 compare and 443 inspect rows excluded for an extra clause,
81,017 SELECT rows for an unsupported shape). Revision 1 is stricter than the first pass: the English measure and
group words must be exactly the bound column's words (whole-word stems; no extra modifier, so "average total tax
revenue" no longer matches a column "tax revenue" quietly), and negation, "only", "except", "top", "each", "all" and
similar words are excluded. Data-modification rows are kept only if the prompt asks for the change and is not a
question. Nothing is relabelled or rewritten. Gold ids are checked again in the tests from the record alone.

**The Turkish side is not verified for meaning.** The Turkish gate checks the aggregate word, the group marker and
the absence of negation, filter and ranking words; it does **not** check that the translated measure and group are
the bound columns' (a mistranslated measure or group passes; a test documents this limitation). Those 97 records are
therefore tagged `translation_unverified` and are **quarantined from product training**: they may be used only after
a native-speaker review of the Turkish text, and the revision 1 smoke does not use any external record. Adversarial
mutation tests (wrong English measure or group, extra modifier, negation, lowercase filters, ranking words) fail the
English gate as intended; they do not prove semantic correctness of the gate.

## Splitting, validation and contamination

- A request idea (`lineage_id`) is assigned to one split before text is generated. For every origin the validator
  checks lineages, table families, normalised request texts, **schema fingerprints** (normalised column names and
  kinds) and upstream row ids across splits: 0 shared in each case. An upstream schema used by domains of both splits
  is used in one split only (5,223 rows excluded for it).
- Deterministic validation (`axbench.audit.gold_problems` plus mutation-tested checks in `axbench_tr/validate.py`)
  found 0 problems in 7,269 records; a failing record refuses the build.
- Overlap with AXBENCH v0.1 train and validation (560 requests): 0 exact and 0 near-duplicate (4-gram Jaccard 0.8).
  The v0.1 splits are comparator evidence and must stay out of v0.2 training.
- Near-duplicate audit (character n-gram TF-IDF cosine on normalised requests): this is a **lexical proxy for
  paraphrase, not semantic deduplication** and no embedding model was used. A pair of different text at cosine 0.9
  or more across the split boundary (3, all among upstream English requests) was resolved by removing the
  validation-side lineage (1 lineage). Now 5 pairs reach 0.8 across splits and none reach 0.9. Within a split, 2,499
  pairs of the same lineage reach 0.9 (a template varying only in a column name) and 37 pairs of different lineages.
- **The sealed AXBENCH test split was never accessed.** The owner can measure overlap without exposing it:
  `python -m axbench_tr contamination --file <requests.jsonl>` prints counts only. It has not been run.
- No external benchmark test split (TAT-QA, FinQA, ConvFinQA, DataBench, StatQA) is in any record.

## Sequence lengths (Qwen3-1.7B tokenizer, chat template, `enable_thinking=False`)

Measured with the tokenizer files pinned by SHA-256 in `sources/ACQUISITION.json` (the script stops on a mismatch).
Each sequence is measured three ways (prompt and target encoded separately, the whole sequence encoded once, and
the full chat conversation rendered by the template); the largest is reported. Joint and separate lengths were equal
for all 7,269 records, no token crosses the prompt/answer boundary, and the template-rendered conversation is one
token longer (a trailing newline). This is a tokenizer-level check; full sequences and loss masks must still be
validated with the real MLX trainer.

| Split | Examples | Total tokens | Mean | Median | p95 | p99 | Max | Over 1,024 | Over 2,048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5,873 | 3,024,797 | 515.0 | 478 | 815 | 1,068 | 1,343 | 70 | 0 |
| validation | 1,396 | 727,890 | 521.4 | 497 | 822 | 848 | 891 | 0 | 0 |

About 94% of tokens are the fixed instruction and schema prompt; the answer averages 28.6 tokens. The 70 train
sequences over 1,024 tokens (57 Turkish, 13 English) all come from two wide real tables (37 and 43 columns).
Truncating them would cut the schema and corrupt the label, so none may be truncated or dropped silently. A
2,048-token limit holds every example with margin 705. `benchmarks/axbench_tr_token_stats.py --max-seq-length 2048`
is a gate that exits 1 if any sequence is longer.

## Human-review sample

`v0.2-rev1/review/` holds a reproducible 390-record sample drawn with a new seed: 300 Turkish (100 external, 200
template) and 90 English (30 external, 60 template), stratified by language, origin, frame and risk; all 100 external
Turkish rows are `translation_unverified` candidates for the native-speaker check. Rows carry the request, the schema, the
gold plan (readable and JSON), provenance and licence, and empty reviewer columns for two independent reviewers.
Review gates are predeclared in `REVIEW_PROTOCOL.md` (correct gold at least 95% and natural-or-fixable at least
90% per origin, reported by language and frame; zero serious errors remaining after correction; an independent
second review with adjudication; AI review kept apart from human decisions). They are accepted in principle and need
the owner to designate the actual reviewers (an AI cannot stand in for them). No sample passes by itself, no human
decision exists yet, and the sample is **not** an independent benchmark.

## What is still needed before training

1. Native-speaker review of the sample, with a second independent review on a subset and adjudication.
2. Independently authored Turkish tasks on licensed economic schemas, held out by source and intent family and not
   used for tuning. An AI cannot supply that independence; an authoring kit is in the experiment folder.
3. More genuine Turkish economic diversity (the corpus is 92% template-derived).
4. Resolution of the SRC-N03 licence conflict and of the upstream Gretel generation terms. **Until the Gretel terms
   are resolved or the 569 external records are excluded there is no final corpus or product clearance.** The EXP-TR02
   smoke is limited to the rights-cleared draft subset (`synthetic_template` records only; no upstream text; table
   schemas from SRC-O01, O02 and O06 only, whose terms are documented in `sources/LICENCE_EVIDENCE_ADDENDUM_1.json`)
   and is discard-only.

## Known limitations

1. No human review; naturalness and correctness of Turkish have not been checked by a native speaker.
2. External Turkish requests are machine translations of LLM-written English prompts, checked only for the aggregate
   word and for added constraints, and tagged `translation_unverified` (quarantined from product training).
3. Template phrasing is narrow (122 templates); a model can learn the templates rather than the task.
4. Deduplication is lexical only.
5. The sealed test overlap check has not been run.
6. Real Turkish official schemas are mostly municipal (İzmir), World Bank and Eurostat layouts; TÜİK, EVDS, ILOSTAT,
   IMF and İBB were not acquired.
7. Only four operations are covered; this is not a general table-QA corpus.

## Licences

There is no single licence for the corpus. `synthetic_template` records are this project's text under the repository
licence (MIT). `external_adapted` records keep their upstream licence (Apache-2.0, in `provenance.upstream_license`;
text in `LICENSES/Apache-2.0.txt`) and the repository licence is not asserted over them. The İzmir and World Bank
files under `economic/` are CC BY 4.0 (text in `LICENSES/CC-BY-4.0.txt`; attribution in `THIRD_PARTY_NOTICES.md`).
ECB, OECD and TCMB observations are not redistributed. The privacy and per-resource licence audit of the committed
economic CSVs is `economic/PRIVACY_AUDIT.json` (one CSV with contact details was removed). The licence evidence read
(page, SHA-256, excerpt) is in `sources/LICENCE_EVIDENCE.json` and `DATA_SOURCES.md`. Dataset-card licence tags
were read, not independently verified; there has been no legal review. Raw data kept outside git is not proof of
rights. The Gretel original was model-generated and the terms of that model were not checked.

## Reproduction

Isolated, locked environment for the build (pyarrow, tokenizers, jinja2; `exclude-newer` pinned):

```
uv sync --locked --project envs/data
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr acquire   # raw files -> data/axbench-tr/_raw (ignored)
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr verify     # re-hash against ACQUISITION.json
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr build      # verifies the raw files first, fails on mismatch
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr validate
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr review-sample
PYTHONPATH=src envs/data/.venv/bin/python benchmarks/axbench_tr_token_stats.py --max-seq-length 2048 \
    --per-example data/axbench-tr/v0.2-rev1/token_lengths_per_example.jsonl
```

Two consecutive builds, one in the isolated environment and one in the main one, gave byte-identical
`train.jsonl`, `validation.jsonl`, `MANIFEST.json` and `AUDIT.json`. The raw files were fetched once on 2026-11-27 at
pinned revisions; the actual rebuild from them was re-run, but a fresh download from the upstream hosts was not.
