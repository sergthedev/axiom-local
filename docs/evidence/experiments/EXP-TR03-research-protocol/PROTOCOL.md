# Proposed EXP-TR03: prospective research protocol for AX-01 LoRA fine-tuning on AXBENCH-TR (draft)

Status: **proposed, draft, preparation only.** Nothing in this protocol has been run, approved or cleared. **No model has been
loaded, no training or generation has been performed, no system has been scored, and no sealed test has been touched.** This
document and the tooling behind it were prepared on the owner's explicit authorization of the technical lead's task
`EXP-TR03-PREP-50bea1f` (PR #13, comment 7235303471). Any execution needs a separate reviewed, immutable clearance and must meet
every gate in section 6. This revision answers the technical lead's gate-hardening and statistics review (PR #15 comment 7235630804); it
is still preparation only. The identifier EXP-TR03 was unused in the repository when this was written.

Frozen and untouched: PR #11 (EXP-010), PR #13 (this work is based on its evidence head `50bea1f968ec482153dd2d684e34b3f758f14813`
and adds files only), the EXP-TR02 harness and its evidence, PR #14.

## 1. Question and what this protocol is not

Question: does a bounded single LoRA run on rights-documented, **human-validated** synthetic Turkish-first routing data improve
AX-01's intent routing on **independently human-authored** Turkish tasks, compared with fixed comparators, without increasing
wrong executable proposals?

This is research on one adapter from one run. It is **not** product training, **not** a quality claim about AXBENCH-TR, and it
does not rebrand the draft corpus as a product corpus. A synthetic corpus alone supports no research-quality claim. The
comparisons are only meaningful on the new evaluation set (section 4); no score may be compared across different sets, in
particular not with the EXP-010 validation numbers.

## 2. Motivation (exploratory, relayed, not re-verified here)

The technical lead's review accepted PR #11's frozen EXP-010 result head `3452c9af…` as an exploratory pilot, not a deployment
result (comment 7235303471): D13 31/160 against 12/160 untuned (+11.875 pp, cluster bootstrap 95% [+7.41, +17.48] pp), 73/160 for
the rules, 6 of 12 D14 gates, 116 schema-valid candidates, Turkish 11/64, REJECT recall 17/30, CLARIFY recall 11/45, 32 wrong
executable proposals, 0 dispatches under Guard without confirmation. These figures are quoted from that comment; the PR #11
evidence is not part of this branch and was not re-read or re-computed here. They motivate hypotheses only. The pilot trained on
400 AXBENCH v0.1 rows, all AI-authored.

## 3. Hypotheses (prospective; direction fixed before any data exists)

All are about the single adapter from **EXP-TR03-T** (section 7) on **EXP-TR03-E** (section 4), against the fixed comparators C0
(untuned Qwen3-1.7B) and C1 (rules `axiom.rules/0.1`). C2 (section 5.1) is an optional secondary comparator and is never needed for a
decision.

| ID | Hypothesis | Endpoint (frozen in section 5) | Role | Would count against it |
|----|------------|--------------------------------|------|------------------------|
| H1 | The adapter emits more schema-valid `axiom.candidate/1` objects than C0 | schema-valid share, full denominator | secondary, descriptive | no increase over C0 |
| H2 | Turkish routing improves over **both** C0 and C1 | P1a, P1b: Turkish semantic exact match, S2 minus C0 and S2 minus C1 | **decision-bearing** (co-primary pair) | either contrast's point estimate below +10 pp or its one-sided bound not above 0 |
| H2r | REJECT and CLARIFY recall on Turkish items are not lower than C1's | R1 REJECT recall, R2 CLARIFY recall, S2 against C0 and C1 | secondary, descriptive, never part of the pass rule | a point estimate of S2 below the comparator's |
| H3 | Wrong executable proposals do not increase against **either** comparator (noninferiority) | N1a, N1b: wrong-executable-proposal rate, S2 minus C0 and S2 minus C1 | **decision-bearing** (co-required pair) | either contrast's one-sided upper bound above +2 pp |
| H4 | Fewer outputs hit the 256-token cap without ending | share with `finish_reason: length` | secondary, descriptive | share above C0's |
| H5 | The gain over C0 holds on layout-held-out and intent-held-out tasks | gain over C0 per stratum | secondary, descriptive | a stratum with a non-positive point estimate |

H1 to H5 are motivated by aggregate pilot findings (candidate validity, Turkish accuracy, REJECT and CLARIFY recall, wrong executable
proposals and "capped generation" named in the comment, for which no figure is available here). A negative, mixed or inconclusive
result is a result and is reported as such. The pass rule (section 5.4) uses only H2, H3 and the structural gates S1 and S3.

## 4. Evaluation set EXP-TR03-E (to be authored by humans; does not exist)

- **Authors.** Named native-Turkish humans designated by the owner (gate G5), none of whom holds a reviewer or adjudicator role, so
  that no author has seen the training templates or rows. An AI cannot author, check or adjudicate any task, and no AI-authored
  task may be labelled human. Kit: `docs/evidence/experiments/EXP-TR01-axbench-tr-acquisition/HELDOUT_AUTHORING_KIT.md` (authoring
  only; nothing authored).
- **Independence.** No task copied from a dataset, model output, the review worklists or any request in `data/axbench-tr/`;
  written without sight of the training templates; checked by a second person who is not its author; gold checked with
  `axbench.audit.gold_problems` for contract consistency (not human judgement). The sealed AXBENCH test is not read, compared or
  contamination-checked by this protocol.
- **Schemas and the economic track.** The licensed layouts of SRC-O01, O02 and O06 are heavily used by training (the top family is
  34.8% of train rows) and only 5 of the 55 licensed tables in no corpus record are economic, with on average 64% of their column
  names already seen in training (`HELDOUT_SCHEMA_CANDIDATES.csv`, C-TR-22). So economic diversity cannot come from the existing
  licensed layouts. **Proposed (owner decides; a prospective amendment to the earlier kit):** an independently authored synthetic
  economic-schema track: human authors, blind to training, invent novel fictional tables with novel column names and no imported
  observations or third-party schemas, a second person checks the gold, and these tasks form a separately reported stratum
  (`table_source: independent_fictional`); and/or genuinely licensed *new* economic schemas with the exact source rights recorded
  (`licensed_new`); unused layouts of the existing licensed sources (`licensed_unused_layout`) are the third allowed source. Tasks on
  a training layout are not held out and are not admitted. Unreviewed or padded data is never added to reach a count.
- **Strata.** Language (Turkish primary, English secondary), action, operation or reason, layout-held-out versus not, intent-held-out
  versus not, near-miss requests (look like a plan but contain a total, filter, ranking or forecast), `table_source`.
- **Planning target, not a power calculation (owner decides).** Turkish 300 (PLAN 120, CLARIFY 90, REJECT 90; at least 30 per
  operation, 10 per reason and 60 near-miss), English 100. These sizes were chosen from IID Wilson arithmetic: the worst-case
  half-width of a single rate is 0.056 at n = 300 and 0.096 at n = 100 (`SIZING.json`). That arithmetic treats tasks as independent
  and describes one rate; it is **not** the precision of a paired difference between systems and **not** cluster-adjusted. Nothing
  here proves the set is large enough to detect a +10 pp gain or to show a +2 pp noninferiority margin; section 5.5 says what to
  report instead.
- **Clusters and their predeclared minimum.** A cluster is an (author, table family) pair. `HELDOUT_FROZEN.json` is refused unless
  it has at least 4 distinct authors with none above 35% of the tasks, and at least 20 distinct table families with none above 10%
  (enforced in `tr03_gates.heldout_problems`). These are proposed values; a different value is a prospective amendment for human
  approval, decided before authoring.
- **Freeze.** The task file's SHA-256, the author and checker roster, and the bindings to the exact protocol and prep manifest are
  recorded before training is cleared (G5). The adapter never sees it; nothing is tuned on it.

## 5. Systems, metrics, inference and gates (frozen before any evaluation)

### 5.1 Systems

All run **once** on the same frozen evaluation IDs, same prompt `axiom.prompt/1`, template with `enable_thinking=False`, greedy, at
most 256 new tokens, no repair, no constrained decoding, Guard policy 2, scorer `axbench.score/0.2`, harness `axbench.evaluate/0.2`
(the EXP-009 conditions):

- **C0** untuned `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **C1** the frozen rule baseline `axiom.rules/0.1`.
- **S2** the TR03-T adapter on that base (final adapter only).
- **C2 (optional secondary comparator)** the frozen EXP-010 adapter, included **only** if its adapter hash is pinned and a
  compatible frozen inference path exists at a recorded commit; it is never retrained, never used for selection, never part of the
  pass rule, and reported separately. The preregistration (`PREREGISTRATION.json`) is refused if a C2 entry lacks the hash, the
  commit or the `retrained: false, used_for_selection: false` declarations.

No score is compared between this set and any other set, including AXBENCH v0.1 validation.

### 5.2 Endpoints

Full denominators; Turkish subset first, then all, then by stratum. **Semantic exact match**; **schema-valid share**; per-action
precision and recall; **wrong executable proposal** = a schema-valid, semantically valid PLAN that does not equal the gold or an
accepted variant (counted over all items, so a system that refuses everything has rate 0 and fails P1); share of outputs reaching the
256-token cap; Guard layer and dispatch counts; the **absolute** wrong-plan rate of every system with its interval.

### 5.3 Inference: paired cluster bootstrap, adjusted bounds

The earlier draft said "Holm over the two contrasts" while describing an unadjusted 95% percentile interval. That wording is
withdrawn: a 95% two-sided percentile interval is not Holm-adjusted. The method is now:

- **Resampling.** Paired cluster bootstrap over the Turkish subset (English reported separately): draw clusters (section 4) with
  replacement, keep all tasks of a drawn cluster for all systems at once, 2,000 replicates, seed 2026, the same draws for every
  contrast (so contrasts are resampled jointly, but each bound uses its own marginal quantile, not a max-t simultaneous band),
  percentile method (`axbench.stats` as in EXP-009).
- **Families and levels.** Two co-required families of two contrasts each: the efficacy family {P1a: S2 minus C0, P1b: S2 minus C1}
  and the safety family {N1a, N1b}. Within each family the one-sided error is split by Bonferroni: family one-sided level 0.05,
  **0.025 per contrast**, i.e. a one-sided **97.5%** bound (lower bound: the 2.5th percentile of the replicate differences; upper
  bound: the 97.5th). Every component of the pass rule must hold (a conjunction), so no further adjustment across families is made;
  Bonferroni inside a family is conservative for a conjunction and is kept because it is simple and was asked for.
- **Sensitivity.** The same bounds under resampling by table family alone; a claim must survive both the (author, table family)
  and the table-family-only resampling. Report cluster counts and the number of tasks per cluster.
- **Everything else** (H1, H2r, H4, H5, per-stratum, English, C2) gets a 95% two-sided percentile interval, labelled unadjusted and
  descriptive; no multiplicity claim is made.
- **Resolution.** With 2,000 replicates the 2.5% tail rests on about 50 draws. A bound within 0.5 pp of its threshold is reported
  **inconclusive**, not passed; nothing is rerun with another seed or more replicates after seeing a result.

### 5.4 Outcomes (proposed; not yet approved by the lead or the owner)

Each component is `met`, `not met` or `inconclusive` (a bound within 0.5 pp of its threshold, fewer clusters than the minimum, or a
failed sensitivity). The experiment's **supporting** outcome needs every component `met`:

- **P1a, P1b (H2).** For S2 against C0 and against C1 on Turkish semantic exact match: point estimate at least **+10 pp** (a practical
  target, not a statistical test) **and** the one-sided 97.5% lower bound above 0.
- **N1a, N1b (H3).** For S2 against C0 and against C1: the one-sided 97.5% upper bound of the difference in wrong-executable-proposal
  rate (Turkish subset, all items) is at most **+2 pp**, a prospective noninferiority margin for this research comparison only. It
  is **not** a deployment criterion or a safety claim. The margin may be unattainable at this n; then the component is `not met` or
  `inconclusive`, which is the intended direction.
- **S1.** Zero Guard dispatches for every system. Policy: the evaluation grants **no** confirmation, ever; a PLAN that Guard
  would run only after a human confirms is counted as `needs_confirmation` and is never dispatched, so any dispatch is a harness or
  Guard violation. The `needs_confirmation` counts and the wrong-plan rate among them are reported for each system. Zero observed is
  structural, not proof.
- **S3.** Over **all** high-risk REJECT items in the set (instruction override, code or system request, data modification; both
  languages; at least 10 per reason, so at least 30), the adapter proposes a PLAN on **zero**. The denominator is the whole set of
  such items, not a sample of them; a single failure makes S3 `not met`. Report the exact one-sided 95% upper bound for zero events,
  `1 - 0.05^(1/n)` (9.5% at n = 30; the 3/n rule of three gives 10%); zero observed does not show absence.
- A result that fails N1 or S3 is a failed experiment whatever P1 shows, and an `inconclusive` N1 or S3 blocks any positive reading.

**Descriptive context that is never a pass criterion.** The absolute wrong-plan rate of each system and the earlier D14 thresholds from
the EXP-010 report are listed next to the observed values (the thresholds are quoted from PR #11 and were not re-derived here). A
noninferiority pass against a poor comparator is not a safety or deployment result.

One run, one seed: the result concerns this adapter, not training in general, and is not a determinism claim.

### 5.5 Precision, clusters and what the numbers do not show

- The 300 + 100 sizes are a planning target (section 4). Before authoring, the number of authors, table families and tasks per
  cluster are predeclared (section 4); after the run, report the cluster counts, the cluster sizes and the width of every
  paired interval, and state that bootstrap bounds from a few dozen clusters tend to undercover.
- Wilson half-widths treat items as independent; **they are not cluster-adjusted.** The zero-failure sizes (59 for a 5% bound, 29 for
  10%) are one-sided binomial statements about the *sampled* stratum under independence. They are not cluster-adjusted and are **not**
  evidence that an observed 95% in a sample guarantees 95% in the population: an observed 56 of 59 can come from a population with
  a lower rate, and rows from one template are not independent (87 train templates; the `inspect_all` stratum comes from 3).
  The review sample is a screen; the primary control for template-generated defects is the review of **all 122 templates**.
- A paired difference between two systems on the same 300 items has its own variance; the Wilson numbers above do not describe it.

## 6. Human, rights and clearance gates (none is satisfied)

Existing project policy requires real humans before any training whose product is used. This protocol does not relax it. An AI
review, including mine, is not a human decision and may not be entered in a ledger. The checker (`python -m axbench_tr.tr03 gates`,
`training/tr03/tr03.py readiness`) validates files people fill in under `data/axbench-tr/tr03-prep/ledgers/` (the directory does not
exist in the repository; **no dummy ledger is committed**; the templates stay empty).

**State vocabulary.** Each gate is `absent`, `invalid` (a rule is contradicted; one bad cell fails the file; nothing is repaired or
inferred), `incomplete` (valid so far, items undecided), `criteria_not_met`, or `consistent_unverified`. **Nothing is "passed".**
The tool cannot authenticate a person, verify a rights opinion, judge an evaluation set or confirm a clearance, so a file's
presence or even internal consistency never opens a gate: `research_training_open` is always `false` and the exit status is 1
(2 if every artifact is consistent), never 0. G4 to G7 stay pending until reviewed substantive artifacts exist, each bound by
SHA-256 to the exact protocol, prep manifest, plan, config and tooling it was made for (`current_bindings`); a stale hash makes the
artifact `invalid`.

| Gate | What closes it (the checker verifies only structure and consistency) | Who | State |
|------|-----------------------------------------------------------------------|-----|-------|
| G1 | `ledgers/HUMAN_ROSTER.json` (schema `tr03-roster/2`): the owner's designation and attestation text accepted; unique valid private ids (no AI-like names); `type: human`; roles from reviewer, adjudicator, author, checker, owner; booleans `native_turkish`, `fluent_english`, `independent_of_corpus_authoring` never missing; an owner; for each language at least 2 competent reviewers, 1 competent adjudicator and 3 competent judges; authors never also reviewers | owner | absent |
| G2 | `TEMPLATE_REVIEW_RESULTS.csv`: all **122** frozen templates (87 train, 35 validation), exact header and identity cells, resolved decisions under the adjudication rule below; each frame class keeps at least 2 train and 1 validation template | reviewers | absent |
| G3 | `ROW_REVIEW_RESULTS.csv`: all 518 frozen rows resolved, and the gates below | reviewers | absent |
| G4 | `RIGHTS_DECISIONS.csv`: the 7 frozen source rows decided by an owner on the roster (`include` or `exclude`; `include` only for SRC-O01, O02, O06 and with notes on the caveat; O03, O04, O05 and external-adapted must be `exclude`). Options: `RIGHTS_DECISION_OPTIONS.md` | owner (counsel if wished) | absent |
| G5 | `HELDOUT_FROZEN.json`: section 4's structure (task list with authors and checkers on the roster, minimum clusters, attestations by the owner, bindings); the task texts themselves stay with the humans (hash only) | authors | absent |
| G6 | `PREREGISTRATION.json`: bindings equal the current protocol, config, plan, manifest and tooling; C0 revision equal to the pinned base; C1 `axiom.rules/0.1`; optional verified C2; the analysis constants of section 5.3 | lead and owner | absent |
| G7 | `CLEARANCE.json`: an immutable 40-hex commit, the same bindings, the SHA-256 of every ledger above, GitHub comment URLs of the lead's clearance and the owner's authorization, CI and no-weight M5 preflight references. Its authenticity is checked by people reading GitHub, not by code | lead and owner | absent |

### Adjudication rule (G2, G3), exactly as implemented

1. An item has a first review. **Required second review:** the pre-planned random tenth (`second_review_planned`) and every item whose
   first review is not a clean `keep`. The second reviewer is a different human with the reviewer role.
2. The two reviews **agree** only if every judgement field is identical (for rows: naturalness, gold, serious error, decision). Any
   difference is a **dispute**.
3. A dispute needs an **adjudication** by a third person with the adjudicator role who is neither reviewer, dated on or after both
   reviews. The **effective judgement** is the adjudicator's; otherwise it is the (identical) review. An undecided dispute is
   `incomplete`, never resolved by default; an adjudication without a dispute is `invalid`.
4. Every judge must be human on the roster with the role and competent in the item's language (native Turkish for Turkish items,
   fluent English for English items), with an ISO date between the designation and today.
5. Only an effective `keep` on a clean item can train; `fix` is excluded because no verified correction process exists. A `keep`
   with a wrong gold, a serious error, a `fixable` or `no` request is `invalid`. A serious error must be `drop`.
6. **Denominators are the frozen counts**: a blank is a failure and a missing, duplicated or foreign key invalidates the file; rows
   cannot be deleted or repeated to change n. Any edit to an identity cell (request, gold, schema, stratum, batch, authorship label)
   invalidates the file.

### Row gates: the earlier accepted gates are kept; the frame gates are an addition

Earlier accepted gates (the technical lead's comments and `REVIEW_PROTOCOL.md`): gold correct at least 95% and natural-or-fixable at
least 90% **per origin**, reported by language and frame, zero serious errors remaining after correction, a second independent
review with adjudication. The checker applies them as follows and does not replace them:

- **per origin** (all eligible `synthetic_template` rows, both languages, 518 frozen rows): gold at least 95% and natural-or-fixable at
  least 90% (`G3o_*`); **English** pooled (105 rows) the same two thresholds (`G3l_*`, English cannot disappear from the review);
  **Turkish** pooled (413 rows) the same (`G3t_*`); reported, not gated: by language, frame stratum and **schema source**
  (SRC-O01, O02, O06; each source must have at least 10 reviewed rows, `G3s`);
- **added, proposed for human approval** (`G3f`): each of the seven Turkish frame strata has 59 frozen rows and gold at least 95%
  (at most 2 wrong of 59), chosen so that zero failures would bound a stratum's failure rate below 5% under independence (section
  5.5 on why that is not a guarantee);
- zero serious errors kept (`G3d`); second review and adjudication complete (every item `resolved`).

**Frozen sample and dropped templates.** The 122 templates and the 518-row sample are a *provisional burden*, fixed by the SHA-256
in `MANIFEST.json` before any decision (this revision only added columns and workload batches; the selected rows are unchanged). The
sample is never redrawn or topped up after decisions, so selection cannot depend on review results. A dropped or fixed template
removes all its rows and is **not replaced or regenerated** in this experiment; its sampled rows were still reviewed and still count
in the frozen denominators (conservative: the retained set may be cleaner than the measurement). A template with a reviewed row whose
gold is wrong or that has a serious error is removed whole, and a row marked `fix` or `drop` is removed alone. If a frame class
falls below 2 kept train or 1 kept validation templates, G2 is `criteria_not_met` and training cannot proceed without a prospective
amendment. The exact filtered train and validation lists, and the exposure plan for them, are derived by `research_manifest` and
`training/tr03/tr03.py research-plan` only from complete, valid ledgers and a rights ledger (rows of excluded sources are removed).

Rights caveats recorded, not resolved: WDI indicators compiled from IMF, ILO, UN, OECD or Eurostat were not checked against those
providers' terms; Eurostat's commercial-reuse exception list names countries other than EU, EFTA and official acceding and candidate
countries (Türkiye is a candidate; C-TR-15); İzmir free-text package notes were not read; nothing is legal advice. SRC-O03, O04,
O05 (unverified) and all external records (Gretel terms, SRC-N03 conflict, unverified translations) are excluded by rule. Attribution
(source, publisher, licence id, evidence ids) is in `INVENTORY.json` > `source_attribution`. Per-source evidence and options for the
owner: `RIGHTS_DECISION_OPTIONS.md` (no decision is made there).

Human work is therefore the critical path. A short guide with worked PLAN, CLARIFY and REJECT examples is `REVIEW_GUIDE.md`; the
worklists carry frozen workload batches so the owner can allocate people (`RW-tr-01..08`, `RW-en-01..02`, `TW-01..06`). The prepared
worklists contain only empty decision columns: 122 templates, 518 rows, 7 source rows, 103 held-out layout candidates.

## 7. Bounded single-run training plan

Two arms are distinguished on purpose.

**EXP-TR03-T (research training).** Eligible only after G1 to G7. One run:

| Item | Value |
|------|-------|
| Base | fresh pinned Qwen3-1.7B, revision `70d244cc…`, all ten checkpoint files hash-pinned (copied from the TR02 config; tested equal) |
| Initialization | none: no probe, smoke, EXP-010 or resource-arm adapter, no optimizer state |
| LoRA | rank 8, scale 20, dropout 0, `self_attn.q_proj` and `v_proj`, last 8 blocks (32 tensors, 458,752 parameters, name contract of TR02 revision 5) |
| Sequence cap | 2,048 tokens, no truncation (the longest eligible train row has 1,342) |
| Loss | exact answer mask of TR02 (`make_exact_loss`); unchanged |
| Optimizer | Adam, learning rate 1e-5 constant, micro-batch 1, gradient accumulation 4 (effective batch 4), seed 2026 |
| Rows | exactly the ids derived by `research_manifest` from complete, valid ledgers (effective-keep templates, rights-included sources, minus reviewed rows and whole templates removed by the row findings); sorted by id; at most 5,091 |
| Epochs | **1** |
| Partial batch | `drop_remainder`: the last `n mod 4` rows of the permutation are not exposed and are listed in the evidence; at 5,091 rows, 3 rows |
| Updates | `floor(n / 4)` per epoch; 1,272 at 5,091 rows (5,088 microsteps) |
| Diagnostics | teacher-forced loss on 200 fixed validation rows before training and after updates `floor(k·U/4)`, k = 1..4, and once on all 1,274 eligible validation rows at the end; **no selection and no stopping decision uses them** |
| Checkpoints | adapter saved at the four evaluation points for post-mortem (hash recorded); never evaluated, selected or resumed from; the adapter after the last update is the only candidate |
| Environment | AC power, normal memory pressure, offline flags, locked model environment (`envs/model/uv.lock`), clean checkout at the cleared commit, no concurrent model work |
| Supervision | fail closed; stop rules below; no retry, no resume, no sweep; evidence-only output into a new directory outside the repository, collected and sanitized as in TR02 revision 5; no weights committed |

Why one epoch, not chosen to fill time: no measurement supports a second epoch; the corpus is 87 templates (mean 58.5 rows per
template per epoch), so repetition is already high; checkpoint selection is forbidden, so there is no stopping criterion that
could protect a longer run from overfitting the templates; a different schedule is a different experiment. The durations of two and
three epochs are recorded in `TRAINING_PLAN.json` to show they were not used (nominal 62 and 89 minutes, against 35).

Budget (`TRAINING_PLAN.json`; extrapolated from the TR02 smoke, **not measured at this scale**) at the upper bound of 5,091 rows,
2,643,854 full tokens: training 27.1 min at the slowest observed 1,625 full tokens/s; validation 7.8 min (five 200-row passes plus
one 1,274-row pass at 0.206 s per row), shown separately and equal to 29% of training time; total nominal 34.9 min; **conservative
46.6 min** with a 25% thermal allowance on both rates; hard wall-time cap 60 min (1.25 times the conservative total, rounded up to 15
minutes). MLX peak 6.2 GB was measured in TR02 against the 10 GB limit. A smaller human-approved row set scales these linearly and
the cap is recomputed from `tr03.py plan` at clearance.

Stop rules (`training/tr03/config.json` > `monitor`, tested in `tests/test_tr03_plan.py`): MLX peak above 10 GB (also invalid or decreasing),
RSS above 11 GB (separate), swap growth above 2 GB, critical memory pressure, loss of AC power, non-finite loss or rate, wall time
above the cap, a missing or repeated 25-update window, **sustained throughput loss** (six consecutive windows below 70% of the median of
the first four, 3 consecutive below 50%), **sustained memory-pressure warning** (12 consecutive 5-second samples at warning level
after the first window; warning at model load was seen in 3 of 12 TR02 probe samples), and **any thermal or performance
statement** other than the three "no ... recorded" lines observed in the TR02 run (fail closed on unparseable text).

**EXP-TR03-R (discard-only resource run): withdrawn.** The earlier draft proposed an extended discard-only run to measure sustained
throughput and thermal behaviour. The technical lead's instruction (comment 7235630804) is that no extended resource arm is wanted for
now: the bottleneck is human review, not compute; the 35 to 47 minute estimate is enough for planning; and a short smoke run would
not be a thermal soak anyway. Nothing builds, clears or runs R; `training/tr03/config.json` keeps the arm's entry marked withdrawn only
so that the earlier plan numbers stay interpretable. No driver, model load, training, generation or scoring is authorized.

## 8. Tooling prepared and tested (no weights)

| Piece | File | Purpose |
|-------|------|---------|
| Data tooling | `src/axbench_tr/tr03.py` (`python -m axbench_tr.tr03 all`, `check`) | eligibility, inventory, audits, frozen worklists with workload batches, sizing |
| Gate checker | `src/axbench_tr/tr03_gates.py` (`python -m axbench_tr.tr03 gates`) | strict fail-closed validation of roster, review, rights, held-out, preregistration and clearance files; adjudication rule; derivation of the filtered train and validation lists; never opens execution |
| Generated records | `data/axbench-tr/tr03-prep/` | `INVENTORY.json`, `AUDIT.json`, `SIZING.json`, four worklists, empty roster template, `TRAINING_PLAN.json`, `MANIFEST.json` |
| Human guides | `REVIEW_GUIDE.md`, `RIGHTS_DECISION_OPTIONS.md` (this folder) | how to judge PLAN, CLARIFY and REJECT with real examples outside the sample; per-source evidence and options for the owner |
| Plan tooling | `training/tr03/tr03.py`, `config.json`, `README.md` | config checks against TR02, exposure schedule with partial-batch rule, budget, long-run monitor, thermal parser, `readiness`, `research-plan` (exact exposure plan for the ledger-derived rows) |
| Tests | `tests/test_tr03_data.py` (26), `tests/test_tr03_gates.py` (94), `tests/test_tr03_plan.py` (62) | negative and positive tests on synthetic ledgers (duplicates, missing and foreign keys, spoofed identity cells, blanks, invalid enums and dates, same reviewer, wrong roles, conflicting or incomplete decisions, serious errors kept, empty or arbitrary G5 to G7 files, stale hashes, excluded rights); event streams; committed files equal a fresh deterministic rebuild |

New logic is the schedule, budget, monitor and gate checker, all pure and tested with synthetic fixtures (a synthetic ledger tests the checker's logic; it is not a review, and none is committed). The loss mask, rendering,
LoRA name contract and event audit are TR02's and unchanged (the TR02 code and tests are untouched). `check-therm` parses
`pmset -g therm` text; it was tested against the 53 samples recorded in the committed TR02 evidence but has not been run on a
Mac in this task.

### What the audits measured (`INVENTORY.json`, `AUDIT.json`; one machine, deterministic)

- Eligible (project-authored template text, SRC-O01/O02/O06 layouts): **5,091 train and 1,274 validation** of 5,873 and 1,396;
  excluded: 473 + 96 external records and 309 train and 26 validation rows on SRC-O03/O04/O05 layouts. 79.8% Turkish in train;
  56.7% economic; actions PLAN 53%, CLARIFY 26%, REJECT 21%.
- 87 train and 35 validation templates, all AI-written; 0 records human-reviewed. Top table family (`worldbank/train`) is 34.8% of train rows and the
  five largest families are 45.9%; 840 train records repeat an earlier request text (on another layout).
- Token statistics: train 2,645,703 full tokens (mean 519.7, p99 1,083, max 1,342, 70 rows above 1,024, none above 2,048), supervised
  answer tokens 147,460 (5.6% of full tokens); validation 667,446 (max 890).
- Split independence on the eligible rows: 0 shared lineages, template texts, table families, table ids, schema fingerprints or
  normalized requests; 0 lexical near-duplicate pairs at char 4-gram Jaccard 0.8 (proxy only, not semantic deduplication); 0 exact or
  near matches with the 560 AXBENCH v0.1 train and validation requests (the sealed test was not read).
- Gold labels: 0 contract problems in 6,365 records; no request on the same layout has conflicting labels by column name.
- Descriptive shortcut statistics (shares in the data, not results): the first numeric column is a gold column 14.1% of the time in
  train and 5.1% in validation; the first text column is the group 54.0% and 67.9%; 83.1% of train PLAN requests name every gold column
  verbatim (77.0% in validation), so alias and typo coverage is what makes them non-trivial.

## 9. Launch recommendation (for technical review)

- **EXP-TR03-T: do not launch.** It is blocked by G1 to G7, none of which holds (every gate is `absent`), and it is not eligible to run
  as "research" or as anything used. The earliest sensible order of work: the owner designates reviewers (G1) and rights decisions
  (G4) in parallel; reviewers do the 122 templates (G2) before the 518 rows (G3), because a dropped template removes its rows;
  authors, who must not see the training data, write the evaluation set (G5); the lead approves hashes of this protocol, config,
  plan and tooling (G6).
- **EXP-TR03-R: withdrawn** (section 7). No driver, no model load, no training, generation or scoring is authorized or built.
- Proposed inputs for any later clearance (this PR's content, not a clearance): the bindings printed by `gates`
  (`bindings_current`: protocol, config, plan, prep manifest, tooling); the clearing commit must be a later, frozen head.

Decisions requested of the technical lead and owner: (1) accept or amend sections 3 to 5 (the +10 pp practical target and +2 pp
noninferiority margin are retained as the lead specified; the inference method, families, sensitivity and the 0.5 pp resolution rule
are new in this revision); (2) the roster requirements of G1 and the minimum clusters of section 4, both proposed values; (3) the
added Turkish frame-stratum gate `G3f` as an addition to, not a replacement of, the earlier per-origin gates; (4) whether to
authorize the independently authored economic-schema track; (5) the owner's rights decisions (`RIGHTS_DECISION_OPTIONS.md`);
(6) whether C2 should be pinned (optional).

## 10. Limitations and not claimed

AI-authored templates and AI-prepared worklists; no human has judged anything; shares and counts are of a draft corpus; the lexical
audit is not semantic; durations are extrapolated; the thermal parser is unverified on a Mac in this task; one machine, one
seed, one run if ever executed; eligible data are rights-documented for resource use, not legally cleared; no result of any system
exists, so no hypothesis is supported or refuted; the licensed sources limit economic diversity; the 5,091 rows come from 87
templates, so they are far fewer independent examples than rows. The gate checker validates consistency of files that people write;
it cannot tell a person from a string, so every human, rights and clearance claim rests on the owner's and lead's attestations
and on their reading of GitHub, and the tool never opens execution. The checker, tests and guide were written by the same AI that
wrote the templates; no human has run the checker on a real ledger. The review sample is a screen with IID arithmetic; the
IID and cluster limits of section 5.5 apply. The +2 pp noninferiority margin may be unattainable at the planned size.

## 11. Reproduce

```
uv sync --locked
uv run python -m axbench_tr.tr03 check            # committed prep files equal a fresh deterministic rebuild
uv run python training/tr03/tr03.py verify-config
uv run python training/tr03/tr03.py check-plan
uv run python -m axbench_tr.tr03 gates; echo $?   # 1: every gate is absent; the tool never exits 0
uv run python training/tr03/tr03.py research-plan; echo $?   # 1: no ledgers, nothing derived
uv run pytest tests/test_tr03_data.py tests/test_tr03_gates.py tests/test_tr03_plan.py
```
