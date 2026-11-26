# ADR 0011: AXBENCH benchmark design, metrics and owner sealing of the final test

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-26
- Issue / PR: Issue #4 (Phase 3A); implements the DR-005 technical lead decisions
  ([DR-005 review](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7169124248)); uses ADRs 0009 and
  0010

## Context

AX-01 needs a benchmark that can show whether a router maps Turkish, English and mixed requests to the right
candidate, before any training. DR-005 fixes the provisional sizes (400 train, 160 validation, 360 final test; final
TR 144 / EN 144 / mixed 72), requires lineages to be split before paraphrase, three held-out strata, a full-denominator
semantic exact-match primary endpoint, separate raw-candidate, after-Guard and dispatch layers, IID statistics only
as illustrations, a competent rule baseline, truthful provenance and a final test kept away from the implementing
agent.

## Alternatives considered

- **Random split of generated examples.** Rejected: paraphrases and translations of one idea would cross splits.
- **Pilot sizes (200/80/180).** Not needed: the generator meets the full target exactly, so the pilot sizes are
  supported but unused. They were not chosen after seeing any score.
- **Commit the final test with a hash.** Rejected: a hash proves identity, not secrecy; a test in the repository is
  visible to the implementing tool and to any later training run.
- **Human-written corpus.** Preferred in principle but not available yet; the corpus is synthetic and labelled so,
  and the owner review and owner-authored test lineages are the path towards human input.

## Decision

Implemented in `src/axbench/` (dataset card: `data/axbench/DATASET_CARD.md`; tests: `tests/test_axbench.py`).

**Construction.** 13 schema families (6 economic, 7 non-economic) are partitioned before generation: 8 shared, 2
validation-only, 3 test-only. 105 request lineages (Turkish and English templates, optional code-switched template)
are assigned to exactly one split before generation. Examples are rendered with exact or quoted names, phrases,
aliases, stripped diacritics, one-edit typos, case variants, cross-language mentions, malicious headers and request
injections; gold labels are derived from filled slots. Validation and test are split in equal thirds into
`A_heldout_family`, `B_novel_instance` and `C_familiar_instance`. Language, stratum and frame counts are allocated
exactly. Provenance on every record: `ai_generated_from_templates`, `human_review: none`.

**Audit** (`python -m axbench audit`). Valid gold labels; no shared lineage or normalized text across splits; a
union-find over lineages and texts with zero components spanning splits; held-out families and novel instances absent
from train; cross-split 4-gram Jaccard near-duplicates (threshold 0.8) reported, never with test texts. A separate
contamination audit fails if any rule-baseline vocabulary occurs only in validation or test templates.

**Metrics** (`axbench.score`, `axbench.evaluate`). Primary: semantic exact match of the raw candidate against the
frozen gold, on the full denominator (technical failures and invalid outputs count as wrong). Equivalence follows the
engine: inspect column set; relate unordered pair plus method; compare by role; cluster ordered features plus `k`;
CLARIFY/REJECT action plus reason. Secondary: JSON validity, candidate-schema validity, semantic validity, error
codes, action accuracy and per-class F1, operation and slot accuracy, clarify and reject recall and reason accuracy,
false rejection among gold PLAN and among all supported requests, PLAN coverage (all N and gold PLAN), selective risk
among proposed and validator-accepted plans, wrong executable proposals (per all N and per accepted plans, by gold
action), latency as recorded, and slices by language, stratum, domain, family, action, operation and tag. The
**after-Guard** layer passes every raw output through a real `Guard` bound to a synthetic dataset of the example's
schema; the harness never confirms, and any executor call is counted as an unauthorized dispatch.

**Statistics** (`axbench.stats`). Wilson intervals and exact McNemar tests are reported only as IID-conditional
illustrations; the preferred interval is a percentile cluster bootstrap over lineages (2000 replicates, seed 2026)
with the number of clusters and the largest cluster. Paired comparisons report the 2x2 counts and the
percentage-point difference. No p-value is a deployment rule.

**Final test sealing** (`python -m axbench seal`, owner only). The owner chooses a private seed (at least 16
characters) and an empty directory outside the repository, optionally adds owner-written test lineages, and publishes
only `SEAL.json` (test and seed hashes, counts, generator commit, audit result, no texts). Repository commands refuse
the `test` split; the MLX export writes train and validation only. A development test split from a public seed exists
only in memory for audits and tests.

**Provisional gates** (from the DR-005 review; project choices, not measured standards, to be frozen after validation
and before unsealing): structural validity at least 99 %; all-case exact match at least 85 %; tuned at least 5
points above its own untuned base; Turkish at least 80 % and within 5 points of English; unsupported-request recall
at least 95 %; false rejection on gold PLAN at most 5 %; clarify recall at least 90 %; wrong executable proposals
below 2 % of all cases; zero observed unauthorized dispatches; severe systematic errors block regardless of
aggregates.

### Revision 2 (PR #8 review response)

The review of revision 1 found that a semantically invalid candidate could count as exact, that macro F1 dropped a
supported class without predicted positives, that duplicate prediction ids silently overwrote each other, that every
consumer regenerated the corpus from templates (so an owner correction could not reach it) and that the seal wrote
its files after a failed audit. Each was reproduced on `51a6a3a` before the fix (development log).

**Scorer `axbench.score/0.2` and harness `axbench.evaluate/0.2`.** A match requires a semantically valid candidate
(any accepted variant still counts); action, operation, slot and reason metrics credit only semantically valid
candidates, so an invalid prediction is wrong everywhere; `raw_action_recognition` separately reports the action of
any schema-valid candidate. Per-class precision and F1 use zero for an undefined ratio when the class has gold
support, so a supported class with no predicted positives enters the macro F1 as 0; a class without gold support is
listed under `action_macro_f1_classes.excluded_no_gold_support`. Prediction files must match the examples one to
one: duplicate, unknown or id-less predictions raise an error (missing ones still count as errors). The after-Guard
layer opens a Guard request context per example (ADR 0010 revision 2) and counts an awaiting-confirmation preview as
correct when it matches the gold label or an accepted variant, the same rule as the raw exact match. Tests:
`tests/test_score_v02.py`. EXP-008 stays as recorded under 0.1 at its commit; EXP-011 rescored its 160 committed
predictions under 0.2 and found no metric changed (all 160 outputs are semantically valid, every class was predicted
and no example has accepted variants).

**Frozen corpus versions** (`src/axbench/corpus.py`). Every command except `generate` (audit, review batch, rules,
score, MLX export, token lengths, seal) loads train and validation from a committed corpus directory whose
`MANIFEST.json` SHA-256 and counts are verified; nothing else regenerates them. `python -m axbench freeze` derives a
new frozen version from a parent and can apply a human reviewer's decision ledger (the filled review CSV: `approve`,
`correct` with a corrected gold label and/or request, or `reject`, which removes the example). The command refuses a
ledger unless the reviewer is named and the review attested as human (`--human-review`): an AI review is not a
decision. Only examples with a decision get `human_review: "owner_reviewed"` (with the original gold or request kept
in provenance); all others stay `"none"`. The manifest records the parent hashes, the ledger hash, the exact counts by
split and decision and the removed ids; reviewer notes are not copied. Labels are revalidated, the train/validation
audit is rerun, and nothing is written unless everything passes. Gold labels and hashes are therefore fixed in a
frozen version before any model prediction on it. Tests: `tests/test_corpus_freeze.py`, with synthetic ledgers only.

**Seal `axbench.seal/0.2`.** Requires a frozen corpus directory, verifies it and records its manifest and file
hashes in `SEAL.json`; checks unique owner lineage ids; runs the three-split audit **before** writing and writes
nothing (the directory is not even created) when the audit fails; labels examples drawn from owner lineages
`owner_template_ai_rendered` with `human_review: "none"` (the owner wrote the template, the generator rendered the
example) and reports owner lineages supplied, used, examples per lineage and examples from repository lineages. The
owner runs the seal outside any AI tool session; the seed, the owner lineage file, `test.jsonl` and the owner's
private requests and labels stay outside the repository and outside any AI tool's context, and the seed is never
posted in chat, issues or pull requests.

**Blindness.** A private seed hides which examples are drawn, not the content of the repository test lineages: their
templates are in the repository and were written by the implementing tool, so test examples rendered from them are
not blind to it. Only owner-written lineages that the tool never sees are blind in that sense, and their rendered
examples are still not human-reviewed.

### Revision 3: owner-authorized AI review and the first frozen version

The owner delegated the annotation review of the 54-example batch to the technical lead AI
([decision](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7185793594)), which asked that the delegated
review not pass through the human path. `freeze` (`axbench.freeze/0.2`) therefore has a second, mutually exclusive
mode, `--ai-review-authorization <reference>`: every example keeps `human_review: "none"`, each decided example records
`reviewer_type: "ai"`, the reviewer label, the decision and the authorization reference, corrected fields keep their
originals (`original_request`, `original_tags`, `request_source: "ai_review_corrected"`), and the manifest counts
`examples_with_a_decision` (54) separately from `examples_with_a_human_decision` (0). A corrected request must carry its
reviewed perturbation tags, validated against the corpus tag vocabulary.

The first frozen version, `axbench/0.1-ai-reviewed` (`data/axbench/v0.1-ai-reviewed/`), keeps all 560 examples and all
gold labels, corrects three requests, passes the audit and is rebuilt byte for byte from the committed ledger by a
test. It is an **exploratory** corpus: not human-validated, not population-representative and not independent of the
implementing tool. The draft stays unchanged as its parent; EXP-008/EXP-011 remain draft results, and EXP-013 is the
rule reference on the frozen version (73 / 160). EXP-009/EXP-010 (revision 3) use this version. A genuine human review
would produce a new frozen version.

## Tradeoffs and consequences

- The benchmark is synthetic and was written by the same AI tool that wrote the rule baseline and will run the
  experiments. It is not blind. Owner-authored test lineages and an owner-chosen seed reduce this but cannot remove
  the tool's knowledge of the repository templates (dataset card, "Known limitations").
- Template learnability: the rule baseline scores 400 of 400 on train and 72 of 160 on validation (EXP-008).
- With 24 validation lineages, cluster-level uncertainty is wide; small differences between systems will not be
  resolvable.
- Owner corrections now require a new frozen version and a rerun of every experiment that should use them; results
  on different corpus versions are not compared directly.
- The perturbation suite (schema order and id permutations, 40-60 pairs) and shortlist recall are not implemented
  (P1).

## Unvalidated assumptions

- That the labeling conventions (dataset card) match the owner's intent; the owner review batch is pending.
- That template-rendered Turkish is natural enough to predict behaviour on real Turkish requests.

## References

REF-031
