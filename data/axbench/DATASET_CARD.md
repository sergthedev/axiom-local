# AXBENCH dataset card (v0.1-draft and v0.1-ai-reviewed)

AXBENCH is the bilingual (Turkish/English) benchmark for AX-01 intent routing: mapping one
natural-language request and a dataset schema to one `axiom.candidate/1` object (PLAN, CLARIFY or
REJECT). It measures routing only. It contains no data values and measures no statistical result;
all numbers are computed by the AXIOM Core engine after a confirmed plan.

| Field | Value |
|---|---|
| Corpus versions | `axbench/0.1-draft` (`v0.1-draft/`, status: draft, kept unchanged as the historical parent); **`axbench/0.1-ai-reviewed`** (`v0.1-ai-reviewed/`, status: **frozen**, owner-authorized AI review, exploratory) |
| Generator | `axbench.generate/0.1` (`src/axbench/`), deterministic from fixed seeds |
| Splits in the repository | `train` (400), `validation` (160) |
| Final test split | 360 examples, **not in the repository**; generated and sealed by the owner |
| Languages per split | Turkish 40 %, English 40 %, code-switched (mixed) 20 % (test: 144/144/72) |
| Authoring | Templates, schema families and labels written by the implementing AI tool |
| Human review | **None.** No example of either version has been reviewed by a human (`human_review: "none"` on all 560 records) |
| AI review | The owner delegated the review of the 54-example Turkish batch to the technical lead AI ([decision](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7185793594)): 51 approved as-is, 3 request corrections with gold unchanged, 0 removed; applied in `v0.1-ai-reviewed` with `reviewer_type: "ai"`. This is **not** human annotation or independent validation |
| Licence | Same as the repository (MIT); fully synthetic, no personal data |
| Decision record | [ADR 0011](../../docs/decisions/0011-axbench-benchmark-metrics-and-sealing.md) |

## Intended use and non-use

Intended: development and validation of AX-01 candidates and rule baselines; one pre-registered
final evaluation on the sealed test split. Not intended: claims about statistical correctness,
about real user populations, or about languages other than Turkish and English. Results on a
synthetic benchmark written by one tool are not evidence of field accuracy.

## Record format

One JSON object per line. Fields: `id`, `corpus_version`, `split`, `stratum`, `language`,
`family`, `domain` (`economic` or `non_economic`), `schema_id`, `lineage_id`, `frame`, `request`,
`schema` (the exact `axiom.schema-view/1` prompt projection: schema-local ids `c1..cN`, names,
`numeric`/`text` kinds), `gold` (a valid `axiom.candidate/1` object), `accepted_variants` (empty in
v0.1), `tags` (perturbations applied), `provenance` (`authoring: ai_generated_from_templates`,
`human_review: none`, generator, seed label). The router sees only `request` and `schema`.

## Construction

1. **Schema families** (`families.py`): 13 families of 7-9 concepts with Turkish and English
   names, phrases and generic aliases; 6 economic and 7 non-economic. Partition before any
   example exists: 8 shared families (train instances plus separate validation and test
   instances), 2 validation-only families (transport/traffic, tourism) and 3 test-only families
   (firm finance, energy/buildings, sports teams). Instances vary naming style (snake case,
   title case, camel case, Turkish plain, mixed), column order and subsets.
2. **Malicious headers**: some instances carry hostile column names (prompt-injection text,
   shell or SQL fragments, bidirectional-override characters). Different headers per partition.
3. **Lineages** (`lineages.py`): 105 request ideas (52 train, 24 validation, 29 test), each with a
   Turkish and an English template and, where useful, a code-switched template. Every lineage is
   assigned to one split **before** paraphrase or generation (DR-005), so all paraphrases,
   languages and perturbations of one idea stay in one split.
4. **Generation** (`generate.py`): exact marginal allocation of language, stratum and frame;
   column mentions rendered as exact or quoted names, phrases, generic aliases, diacritics
   stripped, a one-edit typo, case variants or cross-language mentions. Gold labels are derived
   from the filled slots, never typed by hand.
5. **Strata** (validation and test, equal thirds): `A_heldout_family` (families never seen in
   train), `B_novel_instance` (shared family, instance not in train), `C_familiar_instance`
   (shared family; instance also used in train, but lineage and text never seen).

Frame mix (all splits): PLAN 53 % (inspect, inspect-all, relate, compare, cluster), CLARIFY 27 %
(nine reasons, equal), REJECT 20 % (nine reasons, equal).

## Labeling rules

The gold label is the single correct candidate under `axiom.candidate/1`; precedence, first match
wins:

1. **REJECT `instruction_override`**: the request tries to change the rules, skip confirmation,
   or act as a system message, even if it also contains a valid analysis.
2. **REJECT `code_or_system_request`**: code, SQL, shell, file paths or tool execution.
3. **REJECT** analysis types outside the four operations: `data_modification`,
   `causal_inference`, `forecasting`, `regression`, `hypothesis_test` (including p-values and
   significance), `unsupported_analysis` (plots, outlier detection, other statistics),
   `out_of_scope` (not a data-analysis request).
4. **CLARIFY `multiple_operations`**: two or more supported operations in one request.
5. **CLARIFY** for one supported operation that cannot be planned as stated: `missing_operation`,
   `missing_column`, `column_not_found` (a named column absent from the schema),
   `ambiguous_column` (a mention matching two or more columns; the candidate ids are listed),
   `column_type_mismatch` (a text column where a number is needed, or a numeric group),
   `missing_correlation_method`, `missing_cluster_count`, `invalid_parameter` (cluster count
   outside 2..64).
6. **PLAN** otherwise, with every slot explicit.

Conventions (confirmed for the reviewed batch by the owner-authorized AI review, not by a human):

* A correlation method counts as given when it is named (Pearson, Spearman) **or** described
  unambiguously: "linear correlation" means Pearson, "rank correlation" or "rank-based,
  monotonic" means Spearman. "Correlation" alone, or "are they related?", needs clarification.
* `inspect` "all columns" lists every column of the schema.
* `compare` is a per-group summary of one numeric measure by one text group column.
* `cluster` uses Core defaults for scaling, initialisation, seed and iterations; the count `k`
  must be stated (digits or a number word).
* A CLARIFY operation hint is recorded where the operation is clear; hints are not scored.

## Scoring (summary)

The primary endpoint is semantic exact match of the raw candidate against the frozen gold label,
on the full denominator (timeouts, crashes, invalid JSON and invalid candidates count as wrong).
Equivalence: inspect column set; relate unordered pair plus method; compare by role; cluster
ordered features plus `k`; CLARIFY/REJECT action plus reason. See `src/axbench/score.py` and
ADR 0011 for secondary metrics and the Guard layer. Since scorer `axbench.score/0.2`, only a
semantically valid candidate can match or earn action, operation, slot or reason credit.

## Versions and freezing

Every AXBENCH command except `generate` reads train and validation from a committed corpus
directory and verifies the SHA-256 and counts in its `MANIFEST.json`; nothing else regenerates the
splits from templates. A frozen version is made from a parent version:

```bash
python -m axbench freeze --corpus data/axbench/v0.1-draft --out data/axbench/<version> --version <version> [--decisions <review CSV> --reviewer <label> (--human-review | --ai-review-authorization <reference>)]
```

Decisions are `approve`, `correct` (a `corrected_gold` candidate JSON and/or a `corrected_request`
with its reviewed `corrected_tags`) or `reject` (the example is removed). A ledger is applied
either as an attested human review, where only decided examples are labelled
`human_review: "owner_reviewed"`, or as an owner-authorized AI review, where every example keeps
`human_review: "none"` and decisions are recorded with `reviewer_type: "ai"` and the authorization
reference. The manifest records the exact counts by split and decision, the corrections and the
original values. Labels are revalidated and the audit rerun before anything is written. Model
experiments (EXP-009 onwards) use a frozen version, never the draft.

### `axbench/0.1-ai-reviewed`

Made from `axbench/0.1-draft` with the ledger `v0.1-draft/review/AI_REVIEW_DECISIONS.csv`, a
transcription by the implementing tool of the technical lead's published decisions; the command and
hashes are in `v0.1-ai-reviewed/MANIFEST.json`, and a test rebuilds the version byte for byte from
that ledger. All 560 examples are kept (400 / 160); gold labels are unchanged everywhere. Three
requests were corrected (original text kept in provenance):

| Example | Correction | Tags |
|---|---|---|
| `train-0235` | `kidem` replaced by the explicit `deneyim yılı` (kıdem is not necessarily total experience); gold CLARIFY `missing_cluster_count` | `diacritics_stripped` removed (the new mention keeps its diacritics); `malicious_header`, `quoted_name` kept |
| `validation-0025` | type-appropriate summary made explicit; gold inspect `c3, c1, c2` | unchanged (`diacritics_stripped`, `quoted_name`) |
| `validation-0037` | "a model that predicts" made an explicit linear regression; gold REJECT `regression` | unchanged (`quoted_name`) |

For `validation-0025` the decision offered two wordings. Core Inspect reports only the total,
valid and missing counts for a text column (`src/axiom/ops_inspect.py`), not category counts, so
the fallback wording "metin sütunları için mevcut betimsel özetleri ve eksik değer sayılarını
göster" was used. The earlier AI pre-review flags were resolved by the same decision: row 18 keeps
`multiple_operations` precedence over column ambiguity; rows 21 and 46 keep their typos as
intentional single adjacent-transposition tests.

Audit of the frozen version (`v0.1-ai-reviewed/AUDIT.json`): passed; 0 components spanning
splits; largest cross-split 4-gram Jaccard 0.4833 (train~validation). Rule baseline on its
validation split: 73 of 160 (EXP-013).

## Splits, leakage and audit

`python -m axbench audit` reads the committed train and validation files (verified against the
manifest) and generates a test split in memory with the public development seed (never written to
disk); it checks: valid gold labels; no lineage or normalized
text shared across splits; no connected component of the lineage/text graph spanning splits;
held-out families and novel instances absent from train; 4-gram Jaccard near-duplicates at or
above 0.8 across splits. `AUDIT.json` holds the current report (it passes; the largest cross-split
similarity is 0.48, between train and validation). No request text of the test split is written
into any report.

## Sealing the final test split (owner only)

The committed code can regenerate a *development* test split from a public seed; that split is
used only for the audit and for tests. The final test split is created by the owner:

```bash
python -m axbench seal --corpus <frozen corpus directory> --out <directory outside the repository> --seed-file <private seed file> [--owner-lineages <file>]
```

The owner runs this command personally, outside any AI tool session. It refuses a corpus that is
not a verified frozen version, a directory inside the repository, a non-empty directory or a seed
shorter than 16 characters. It audits all three splits **before** writing and writes nothing if the
audit fails; otherwise it writes `test.jsonl` and `SEAL.json`. Only `SEAL.json` (test, seed and
development-corpus hashes, counts, generator commit, owner-lineage counts, no texts) is published
before the final run. The seed, the owner lineage file and `test.jsonl` stay outside the repository
and outside any AI tool's context; the seed is never posted in chat, issues or pull requests.
`--owner-lineages` adds test lineages written by the owner (a JSON list of
`{id, frame, tr, en, mix?, hint?, method?}`). Examples drawn from them are labelled
`owner_template_ai_rendered`: the owner wrote the template, the generator rendered the example, and
no human reviewed the rendered example. `SEAL.json` counts how many owner lineages were used and how
many examples each produced.

## Known limitations

* **Not blind.** The same AI tool wrote the schema families, all templates (including the test
  lineages, which are visible in `lineages.py`), the labeling rules and the rule baseline. The
  rule baseline was developed on train only and a mechanical audit removes vocabulary that occurs
  only in held-out templates (`contamination.py`), but the author knew the held-out phrasings.
  The owner's private seed hides *which* examples are drawn, not the test templates themselves:
  test examples rendered from repository lineages are not blind to the implementing tool. Only
  owner-written lineages that the tool never sees reduce this, and only for their own examples.
* **Synthetic language.** Template paraphrases are less varied than real user requests; perfect
  train scores show template learnability, not language understanding.
* **No human review.** No human has reviewed any example. The frozen version reflects an
  owner-authorized AI review of 54 examples only; the other 506 are unreviewed AI-generated
  examples. Results on it are exploratory: not human-validated, not representative of real
  requests and not independent of the implementing tool. A later genuine human review would be
  applied with `freeze --human-review` as a new version. AI review is never counted as human review.
* **Single-turn only**; no conversation state, no data values, no mixed-type or date columns.
* **Small held-out families** (2 validation, 3 test): family-level generalization estimates are
  coarse; examples within a lineage, instance or family are correlated, so IID intervals are
  illustrative only and cluster bootstrap intervals are reported.
