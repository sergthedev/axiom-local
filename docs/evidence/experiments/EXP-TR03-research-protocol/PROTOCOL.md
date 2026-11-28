# Proposed EXP-TR03: prospective research protocol for AX-01 LoRA fine-tuning on AXBENCH-TR (draft)

Status: **proposed, draft, preparation only.** Nothing in this protocol has been run, approved or cleared. **No model has been
loaded, no training or generation has been performed, no system has been scored, and no sealed test has been touched.** This
document and the tooling behind it were prepared on the owner's explicit authorization of the technical lead's task
`EXP-TR03-PREP-50bea1f` (PR #13, comment 7235303471). Any execution needs a separate reviewed, immutable clearance and must meet
every gate in section 6. The identifier EXP-TR03 was unused in the repository when this was written.

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
(untuned Qwen3-1.7B) and C1 (rules `axiom.rules/0.1`).

| ID | Hypothesis | Primary metric (frozen in section 5) | Would count against it |
|----|------------|--------------------------------------|------------------------|
| H1 | Candidate schema compliance: the adapter emits more schema-valid `axiom.candidate/1` objects than C0 | schema-valid share, full denominator | no increase over C0, or any decrease |
| H2 | Turkish routing: higher semantic exact match on the Turkish subset than C0 **and** than C1; REJECT and CLARIFY recall above C1's | Turkish semantic exact match; recall per action | lower bound of either paired difference not above 0 |
| H3 | Wrong executable proposals do not increase: rate not above C0's or C1's | wrong-executable-proposal rate | upper bound of the paired difference above +2 pp (margin proposed) |
| H4 | Capped generation: fewer outputs that hit the 256-token cap without ending | share of outputs with `finish_reason: length` | share above C0's |
| H5 | Held-out families: the gain over C0 holds on tasks whose table layout and intent phrasing are absent from training, not only on familiar ones | gain over C0 in the layout-held-out and intent-held-out strata | a stratum with a non-positive point estimate |

H1 to H5 are motivated by aggregate pilot findings (candidate validity, Turkish accuracy, REJECT and CLARIFY recall, wrong
executable proposals, and "capped generation" named in the comment, for which no figure is available here). A negative or mixed
result is a result and is reported as such. Only H2 and H3 are decision-bearing; H1, H4 and H5 are secondary and descriptive.

## 4. Evaluation set EXP-TR03-E (to be authored by humans; does not exist)

- **Authors.** Named native-Turkish humans designated by the owner (gate G5). An AI cannot author, check or adjudicate any task.
  Kit: `docs/evidence/experiments/EXP-TR01-axbench-tr-acquisition/HELDOUT_AUTHORING_KIT.md` (authoring only; nothing authored).
- **Independence.** No task copied from a dataset, model output, the review worklists or any request in `data/axbench-tr/`;
  written without sight of the training templates; checked by a second person who is not its author; gold checked with
  `axbench.audit.gold_problems` for contract consistency (not human judgement). The sealed AXBENCH test is not read, compared or
  contamination-checked by this protocol.
- **Schemas.** Licensed layouts only (SRC-O01, O02, O06). Authors should prefer layouts absent from every training row. The
  audit found 55 licensed tables in no corpus record (`HELDOUT_SCHEMA_CANDIDATES.csv`), of which only 5 are economic; their
  column vocabulary is not new (mean 64% of column names also appear in training schemas, 7 tables share none). Genuine
  Turkish economic diversity beyond the existing layouts is **limited by the licensed sources**, which is a stated limit of the
  design, not a problem to pad over.
- **Strata.** Language (Turkish primary, English secondary), action, operation or reason, layout-held-out versus not, intent-held-out
  versus not, near-miss requests (look like a plan but contain a total, filter, ranking or forecast).
- **Proposed size (owner decides).** Turkish 300 (PLAN 120, CLARIFY 90, REJECT 90; at least 30 per operation and 10 per reason, 60
  near-miss), English 100. At n = 300 the worst-case Wilson half-width is 0.056; at n = 100, 0.096 (`SIZING.json`).
- **Clusters for intervals.** (author, table family), because tasks by one author on one layout are not independent.
- **Freeze.** The file, its SHA-256 and the author and checker roster are recorded before training is cleared (G5). The adapter
  never sees it; nothing is tuned on it.

## 5. Systems, metrics and gates (frozen before any evaluation)

Systems, all run **once** on the same frozen evaluation IDs, same prompt `axiom.prompt/1`, template with
`enable_thinking=False`, greedy, at most 256 new tokens, no repair, no constrained decoding, Guard policy 2, scorer
`axbench.score/0.2`, harness `axbench.evaluate/0.2` (the EXP-009 conditions):

- **C0** untuned `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **C1** the frozen rule baseline `axiom.rules/0.1`.
- **S2** the TR03-T adapter on that base (final adapter only).
- **C2 (optional, lead's decision)** the frozen EXP-010 adapter, if its weights hash is pinned; it answers a different question
  (does more data beat the pilot) and is reported separately.

No score is compared between this set and any other set, including AXBENCH v0.1 validation.

Frozen metrics (full denominator, Turkish subset first, then all, then by stratum): semantic exact match; candidate-schema-valid
share; per-action precision and recall; **wrong executable proposal** = a schema-valid, semantically valid PLAN that does not equal
the gold or an accepted variant; share of outputs reaching the 256-token cap; Guard layer and dispatch counts. Intervals:
cluster bootstrap over the clusters above, 2,000 replicates, seed 2026, paired where two systems are compared (the same method as
EXP-009, `axbench.stats`).

Proposed gates, **not yet approved by the lead or the owner**:

- **P1 (H2).** For S2 versus C0 and S2 versus C1 on Turkish semantic exact match, the paired difference's 95% lower bound is above 0
  (Holm over the two contrasts) and the point estimate is at least +10 pp.
- **S1.** Zero unauthorized dispatches for every system (structural; zero observed is not proof).
- **S2.** Wrong-executable-proposal rate: the upper 95% bound of the paired difference S2 minus C1, and S2 minus C0, is at most +2 pp.
- **S3.** Zero outputs from S2 that propose a PLAN where the gold is a REJECT for instruction override, code or system request or
  data modification.
- A result that fails S2 or S3 is a failed experiment whatever P1 shows.

One primary question (P1). Everything else is descriptive. One run, one seed: the result concerns this adapter, not training in
general, and is not a determinism claim.

## 6. Human, rights and clearance gates (none is satisfied)

Existing project policy requires real humans before any training whose product is used. This protocol does not relax it. An AI
review, including mine, is not a human decision and is refused by the tooling. `python -m axbench_tr.tr03 gates` and
`training/tr03/tr03.py readiness` report these; today they report all closed and exit 1.

| Gate | What closes it | Who | State |
|------|----------------|-----|-------|
| G1 | A roster of named or coded human reviewers and authors, at least two reviewers, at least one native Turkish speaker, designated by the owner (`ledgers/HUMAN_ROSTER.json`; template in `HUMAN_ROSTER.template.json`) | owner | open |
| G2 | A human decision on each of the 122 templates (`TEMPLATE_REVIEW_WORKLIST.csv`): keep, fix or drop; a template is used only if kept without a serious error; nothing is relabelled automatically | reviewers | open |
| G3 | Row review of the 518-row stratified worklist (`ROW_REVIEW_WORKLIST.csv`): Turkish gold correct at least 95% and natural-or-fixable at least 90% overall and gold at least 95% in each of the seven Turkish strata, each with at least 59 reviewed rows; zero serious errors kept; second independent review of every "no" or serious error and a random tenth; all disagreements adjudicated (the gates the technical lead accepted in principle) | reviewers | open |
| G4 | Owner decision for each source in `RIGHTS_WORKLIST.csv`, including the caveats below | owner (counsel if wished) | open |
| G5 | EXP-TR03-E authored, checked and frozen with SHA-256 and roster; section 4 | authors | open |
| G6 | This protocol and the comparators (C0, C1, optional C2) pinned and approved at a named hash | lead and owner | open |
| G7 | Technical clearance at an immutable code commit with green macOS and Windows CI, a no-weight M5 preflight (including `check-therm`, section 8), and a separate owner authorization | lead and owner | open |

Rights caveats recorded, not resolved: WDI indicators compiled from IMF, ILO, UN, OECD or Eurostat were not checked against those
providers' terms; Eurostat's commercial-reuse exception list names EU candidate countries (Turkey is one) and only column names
reach examples (C-TR-15); İzmir free-text package notes were not read; nothing is legal advice. SRC-O03, O04, O05 (unverified)
and all external records (Gretel terms, SRC-N03 conflict, unverified translations) are excluded by rule. Attribution
(source, publisher, licence id, evidence ids) is in `INVENTORY.json` > `source_attribution`.

Human work is therefore the critical path. The prepared worklists contain only empty decision columns: 122 templates, 518 rows,
7 source rows, 103 held-out layout candidates.

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
| Rows | eligible train rows of human-kept templates minus rows a human marked fix, drop or serious error; sorted by id; at most 5,091 |
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

**EXP-TR03-R (discard-only resource run).** The only arm that no human gate blocks, because its adapter is discarded unseen: one
supervised fresh run of the same configuration on all 5,091 eligible rows, to measure what EXP-TR02 could not: sustained throughput and
thermal behaviour over about half an hour instead of 2.5 minutes, validation overhead at scale, and the real duration of one epoch.
Justification: those numbers set T's wall cap and show whether a 35-minute run is safe on a fanless laptop; they carry no quality
information and will not be reported as any. The adapter is hashed, never evaluated, copied or used to initialize T, and it
is not a usable AX-01 adapter. R is not required for T; it is optional and needs its own clearance (G7 only) and the driver of
section 9.

## 8. Tooling prepared and tested (no weights)

| Piece | File | Purpose |
|-------|------|---------|
| Data tooling | `src/axbench_tr/tr03.py` (`python -m axbench_tr.tr03 all`, `check`, `gates`) | eligibility, inventory, audits, worklists, sizing, human-gate checker |
| Generated records | `data/axbench-tr/tr03-prep/` | `INVENTORY.json`, `AUDIT.json`, `SIZING.json`, four worklists, roster template, `TRAINING_PLAN.json`, `MANIFEST.json` |
| Plan tooling | `training/tr03/tr03.py`, `config.json`, `README.md` | config checks against TR02, exposure schedule with partial-batch rule, budget, long-run monitor, thermal parser, readiness |
| Tests | `tests/test_tr03_data.py` (28), `tests/test_tr03_plan.py` (62) | synthetic ledgers and event streams; committed files equal a fresh deterministic rebuild |

New logic is the schedule, budget, monitor and gate checker, all pure and tested with synthetic fixtures. The loss mask, rendering,
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

- **EXP-TR03-T: do not launch.** It is blocked by G1 to G7, none of which holds, and it is not eligible to run as "research" or
  as anything used. The earliest sensible order of work: the owner designates reviewers (G1) and rights decisions (G4) in
  parallel; reviewers do the 122 templates (G2) before the 518 rows (G3), because a dropped template removes its rows; authors
  write the evaluation set (G5) without seeing the training templates; the lead approves a hash of this protocol (G6).
- **EXP-TR03-R: not launchable today either, and optional.** There is no driver: running it needs a thin extension of
  `training/tr02/tr02.py` `run_phase` (arbitrary row set, 1,272 updates, the monitor above), reviewed and tested on the tiny-model
  fixtures first, then a no-weight M5 preflight including `check-therm`. If the lead wants measured sustained numbers before the
  human gates close, that driver is the next engineering task; I recommend waiting for the lead's decision rather than building it
  unasked, because human review, not compute, is the bottleneck.
- Proposed inputs for any later clearance (this PR's content, not a clearance): corpus files pinned in `config.json`; plan and
  data hashes in `data/axbench-tr/tr03-prep/MANIFEST.json`; the clearing commit must be a later, frozen head.

Decisions requested of the technical lead: (1) accept or amend the hypotheses and gates in sections 3 and 5 (the +10 pp and +2 pp
margins are mine); (2) whether C2 is wanted; (3) whether R is worth building; (4) the proposed review and evaluation sizes.

## 10. Limitations and not claimed

AI-authored templates and AI-prepared worklists; no human has judged anything; shares and counts are of a draft corpus; the lexical
audit is not semantic; durations are extrapolated; the thermal parser is unverified on a Mac in this task; one machine, one
seed, one run if ever executed; eligible data are rights-documented for resource use, not legally cleared; no result of any system
exists, so no hypothesis is supported or refuted; the licensed sources limit economic diversity; the 5,091 rows come from 87
templates, so they are far fewer independent examples than rows.

## 11. Reproduce

```
uv sync --locked
uv run python -m axbench_tr.tr03 check            # committed prep files equal a fresh deterministic rebuild
uv run python training/tr03/tr03.py verify-config
uv run python training/tr03/tr03.py check-plan
uv run python -m axbench_tr.tr03 gates; echo $?   # 1: every human gate is closed
uv run pytest tests/test_tr03_data.py tests/test_tr03_plan.py
```
