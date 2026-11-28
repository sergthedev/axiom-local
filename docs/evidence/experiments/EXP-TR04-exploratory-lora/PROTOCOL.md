# Proposed EXP-TR04: automated, exploratory one-epoch LoRA run on audited synthetic data (preparation only)

Status: **proposed, prepared, not executed, not approved.** No model has been loaded, nothing has been trained, generated or scored,
and the sealed AXBENCH test was not accessed. This is an **exploratory** experiment. Its data are machine-audited and
machine-screened, **not human-reviewed**; whatever it produces is **not human-validated and not deployment-ready**. It is separate
from EXP-TR03, whose human-review gates (G1 to G7) are frozen and unchanged: nothing here weakens, replaces or satisfies them.

Basis: the technical lead accepted the EXP-TR03 gate-hardening preparation at `150992f` and asked for this separate automated
experiment. This change is based on that head and adds files only (plus a narrow `E501` lint exception, section 9). The EXP-TR02
harness, EXP-TR03 files, EXP-010 and every existing module are unedited.

## 1. Question and what this is not

Question (exploratory, descriptive): when Qwen3-1.7B is LoRA-tuned for one epoch, with the frozen TR02 configuration, on
AXBENCH-TR rows that passed an automated rights screen and an automated label audit, how does the adapter behave on a separate,
synthetic held-out benchmark, compared with the untuned model and the rule baseline?

It is **not**: a test of any human-validated hypothesis, a quality claim about the corpus, product training, a replacement for
EXP-TR03, or evidence about real Turkish user requests. The training rows, the validation rows and the held-out rows are all
AI-authored. A pass of every automated check shows consistency with the generator's own rules, not correctness.

## 2. Four statuses are kept apart

| Status | In this experiment |
|--------|--------------------|
| proposed | the training run itself and the held-out evaluation (nothing is cleared) |
| implemented | rights screen, label audit, template contract, held-out generator, training driver, stop rules |
| measured | counts, hashes, audit findings, detection of injected faults, no-weight checks, CI results (all cited from files) |
| independently verified | **nothing**: no human reviewed any row, label, template or rights decision |

## 3. Data: rights screen and label audit (`src/axbench_tr/tr04*.py`, outputs in `data/axbench-tr/tr04/`)

Input: corpus `v0.2-rev1`, unchanged (5,873 train, 1,396 validation; hashes in `training/tr04/config.json`).

**Rights rule (a record is kept only if all hold; anything uncertain is excluded).** Origin is `synthetic_template` (project-authored
request text; no external or adapted text); no `translation_unverified` tag; the table layout comes from a source whose terms were
actually read for the datasets used, and for İzmir (SRC-O06) the CKAN `package_show` record of the **exact package** (168 packages,
fetched 2026-11-28 through `resource_show`, raw SHA-256 recorded, contact fields dropped) has an allowed licence id and no hit in a
text screen for permission, copyright, commercial, restriction, personal-data, third-party or non-open-licence wording.
Excluded without further review: SRC-O01 (World Bank WDI: upstream IMF/ILO/UN/OECD compilers' terms not read), SRC-O02 (Eurostat:
dataset-level notes not found), SRC-O03, O04, O05 (terms unverified), all 569 external records. The text screen is a pattern match,
not a legal opinion; it excluded one İzmir package; **no hit does not mean no restriction.**

**Label audit.** (a) Template contract: all 122 lineages, Turkish and English, checked against reviewable cue lists
(`tr04_cues.py`: own operation family present, no other family or reject cue, slot and method consistency, language parity); 0
contradictions after the lists were refined (15 initial hits were false alarms of the lexicon, all fixed by narrowing cues, none by
relabelling). (b) Row re-derivation (`tr04.row_problems`, fail closed): each request is rebuilt from the lineage template and the
recorded mentions, every mention must be a generator rendering of exactly its gold column, gold passes the candidate contract,
frame, split and language label agree, and request texts are unique per layout. Findings: 25 duplicate rows and 12 rows whose
ambiguous mention is also an exact column name were excluded (37 label exclusions); 6 rows were trimmed so every split is a
multiple of the 4-row batch.

**Result.** 3,164 train rows (87 lineages, 102 table families, 79.9% Turkish, 31.1% on İzmir tables of the economic domain) and 376 validation
rows (35 lineages, 23 families, 81.9% Turkish); 0 shared lineages, families, schema fingerprints or normalized requests between
them. All kept rows are İzmir layouts; no World Bank or Eurostat layout remains. 2,709 train and 1,020 validation records are not
used (exclusion list: `EXCLUSIONS.csv`, one reason per record).

**Detection power, not accuracy.** `MUTATION_REPORT.json`: deliberately wrong copies of kept rows (wrong method, changed `k`, wrong
action, dropped gold column, a mention naming another column, swapped language label, changed frame, a request from another row, and
template-level faults) are flagged 100% of the time (rows applied per kind in the file). These are injected faults on synthetic
copies; the audit restates the generator's rules and the faults break exactly those rules. This does **not** measure how many real
label errors remain.

## 4. Held-out benchmark (`src/axbench_tr/tr04_heldout.py`, `data/axbench-tr/tr04/heldout/`)

Synthetic and AI-authored: 346 rows, 55 lineages (110 texts, Turkish and English, `H4-*`, different wording from the 122 `T2-*`
templates), 28 fictional layouts invented for this benchmark (14 Turkish and 14 English headers, 4 fictional "economies"), all 23
frames, 182 PLAN, 100 CLARIFY, 64 REJECT. Labels come from the same generator logic as the corpus and pass the same row audit and
template contract. The audit proves disjointness from the **whole** v0.2-rev1 corpus (including excluded rows): 0 shared layout
fingerprints, families, normalized requests or template texts, 0 lexical near duplicates at 0.8, 0 overlap with AXBENCH v0.1
requests, and no layout whose column-name set has Jaccard >= 0.5 with a corpus layout (maximum 0.1111). Limits: same author as the
training templates (correlated style and blind spots), cleaner than real headers, few clusters, no real observation, **no human
wrote or checked any row**. The sealed AXBENCH test is not referenced anywhere. The held-out rows never enter the export, the
training run or the validation passes (a test and the preflight check both enforce this).

## 5. Training run (frozen configuration, not started)

`training/tr04/config.json` (status "not approved, no clearance, exploratory") copies the TR02 pins unchanged: model
`Qwen/Qwen3-1.7B` at revision `70d244cc…5e`, ten pinned files, mlx 0.32.4, mlx-lm 0.32.2, Python 3.12, LoRA rank 8, scale 20, last 8
layers, `q_proj` and `v_proj` (32 tensors, 458,752 parameters), Adam, constant learning rate 1e-5, seed 2026, micro-batch 1 with 4
accumulation steps, maximum sequence 2,048 (over-length is an error, never truncated), loss only on the answer and `<|im_end|>`
(`make_exact_loss`). One **epoch** over every kept train row exactly once: 3,164 microsteps, 791 optimizer updates, 1,661,578 full
tokens (no row dropped; 3,164 is a multiple of 4). Fresh base: no earlier adapter or optimizer state. The exposure order is
verified step by step against the declared schedule (`ExposureChecker`). Validation (every one of the 376 rows, teacher-forced loss)
runs before training, after the last update and once more in the reload process; no intermediate evaluation, no checkpoint
selection, no resume, no retry. The adapter after the last update is the only candidate.

**Phases** (separate supervised processes, each with new output directories): `train`, then `reload` (needs a complete `train`;
fresh process, loads the saved adapter, verifies hashes, tensor names and values, re-evaluates all validation rows; the loss must
equal the post-training loss within 1e-4). There is no probe or smoke phase: TR02 already ran both and its rates are the basis of the
budget.

**Safety limits (fail closed; the first stop ends the run as a failed run).** From TR02 (unchanged): MLX peak 10 GB, RSS 11 GB, swap
growth 2 GB, memory pressure critical stops, AC power required, exact event sequence, finite metrics, 32 LoRA tensors and shapes,
changed and finite adapter, saved-adapter hashes equal the logged ones. Added from the TR03 tooling: any thermal or performance
statement other than the three known "no ... recorded" lines, or unreadable output (`pmset -g therm` is sampled with every OS
sample and before launch); a steady-state memory-pressure warning (12 consecutive samples after training starts); throughput below
0.7 of the reference (median of the first 4 windows of 25 updates) for 6 windows or below 0.5 for 3 windows; a hard wall-clock cap
of 45 minutes for `train` and 15 minutes for `reload`, measured from the pre-launch baseline. The preflight (also re-run by the
child) pins the exact commit, a clean tree, every hash (corpus, per-example, export, audit outputs, held-out files, model files),
offline mode, packages, and output directories outside the repository.

**Budget (extrapolated from the 2.5 minute TR02 smoke, not measured at this scale).** Training 17.1 min nominal, three full
validation passes 3.9 min, total 20.9 min nominal and 27.9 min with a 25% thermal allowance; caps 45 + 15 min.

## 6. Evaluation criteria (frozen with this commit; nothing is evaluated until separately cleared)

**A. Run health (automated, in the stop rules).** The run is valid only if the supervisor writes `COMPLETE.json` for `train` and for
`reload`: every stop rule held, the schedule was drawn exactly, the adapter changed and is finite, the reload loss equals the
post-training loss within 1e-4. Anything else is a failed run: no retry, no resume, no result is reported as a model result.
The validation losses before and after training are **diagnostics** (teacher-forced, same synthetic distribution as training,
shared templates). A fall in loss is expected and is not evidence of quality. If the post-training loss is not lower than the
pre-training loss, the run is reported as "no improvement" and no held-out evaluation is requested.

**B. Held-out evaluation (a separate clearance is required; the generation runner is not part of this change).** Systems on the same
346 held-out rows: S0 untuned Qwen3-1.7B; S1 rule baseline `axiom.rules/0.1`; S2 the TR04 adapter (final adapter only). Decoding,
prompt (`axiom.prompt/1`), chat template (`enable_thinking=False`), greedy, at most 256 new tokens and no output repair as in
EXP-009 (`envs/model/exp009_run.py`), asserted by the runner. Scoring only through `axbench.score/0.2` by
`axbench_tr.tr04_heldout.score_predictions`: full denominators (a timeout, invalid JSON or invalid candidate is a miss), semantic
validity required for credit, Guard outcome, 0 unauthorized dispatches required. Primary descriptive endpoint: semantic exact match
on the 346 rows with a cluster bootstrap over lineages (95% interval). Secondary, all descriptive: JSON, candidate-schema and
semantic validity, action accuracy and macro F1, PLAN operation and slot accuracy, CLARIFY and REJECT recall and reason accuracy,
false reject among gold PLAN, wrong executable proposals, and the slices language, domain, family, frame, action and tag.
Pairwise S2 minus S0 and S2 minus S1 with `axbench.stats.paired` (paired cluster bootstrap). **No hypothesis test, no pass/fail
threshold and no multiplicity adjustment is claimed; all numbers are exploratory.** Wrong executable proposals are reported next to
exact match and never traded for it.

**C. How a result may be described.** As "an exploratory observation on a synthetic AI-authored held-out set from one run on one
machine, with correlated authorship". It may not be described as human-validated, as an improvement in general, as evidence about
real Turkish requests or economic tables, or as deployment-ready. A result that is worse than S0 or S1 is reported with equal
prominence.

## 7. What technical clearance would have to verify

(1) The rights rule and its consequence (every World Bank and Eurostat layout is excluded; the economic-domain rows that remain are İzmir
tables); (2) the contract, audit and mutation reports read as limited (section 3); (3) the held-out benchmark's construction and
limits (section 4); (4) the config, hashes and stop rules (section 5) and the green no-weight preflight on the Mac; (5) the
commit to be run (40 characters), authorization for the Mac run, and that no human-review claim is made. Clearance is given on
GitHub by the technical lead; this repository cannot give it.

## 8. Commands (not to be run before clearance; from the repository root)

No-weight (allowed for the preflight): `tr04 verify-config`, `plan`, `verify-inputs`, `schedule`, `verify-model`, `check-data`,
`check-iterator`, `check-lora`, `preflight`. Execution (`supervise --phase train`, then `--phase reload --after <train out>`) is
**not part of this change** and starts only on clearance. See `training/tr04/README.md`.

## 9. Limitations and open points (stated once)

- Everything is AI-authored and machine-audited; the audit shares its author with the templates it audits. No human review exists.
- The rights rule leaves only İzmir layouts: no World Bank or Eurostat table, no multi-country economic layout. The kept validation
  set is small (35 lineages, 23 families); validation loss is a weak diagnostic. A human rights decision for SRC-O01 and O02 (see
  `RIGHTS_DECISION_OPTIONS.md` in EXP-TR03) would widen the data; it is not assumed.
- The text screen reads package notes, extras and resource descriptions only; it is a pattern match.
- Durations are extrapolated; thermal behaviour over about 20 to 30 minutes is unmeasured; one seed, one run.
- The held-out generation runner for S0 to S2 does not exist yet; building it is a separate task after clearance.
- `pyproject.toml` gains a narrow `E501` exception for `src/axbench_tr/tr04*.py` because the audit modules write long statements
  verbatim into hashed reports; no other rule changes.
- This experiment does not satisfy, replace or weaken EXP-TR03; research claims remain blocked on its gates.
