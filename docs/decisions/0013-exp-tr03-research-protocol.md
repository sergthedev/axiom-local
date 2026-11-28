# ADR 0013: EXP-TR03 research protocol: eligibility, human gates and a bounded single-run LoRA plan

- Status: Proposed (draft; awaiting technical review and owner decision; nothing approved or run)
- Date: 2026-11-28
- Issue / PR: #12; draft PR based on the PR #13 evidence head (task: PR #13 comment 7235303471)

## Context

EXP-TR02 showed that a Qwen3-1.7B LoRA run is feasible on the M5 (C-TR-19) for 400 discard-only rows. The technical lead asked for a
prospective research-only protocol, deterministic tooling and a bounded single-run plan, **prepared, not launched**. The corpus is
AI-authored with no human review; project policy requires real humans before any training whose product is used; upstream rights
are unresolved for most external data; and PR #11's pilot figures are exploratory.

## Alternatives considered

- **Train on the full 7,269-record corpus.** Rejected: external records have unresolved terms and unverified translations.
- **Treat the TR02 rights-cleared subset as the research training set.** Rejected: rights documentation is not human validation, and
  a corpus of 87 AI-written templates is not independent evidence. It is acceptable only for discard-only resource work.
- **Evaluate on a split of the AI-authored corpus.** Rejected: same author, same habits; the lead requires independently authored
  human tasks.
- **More epochs, or checkpoint selection by validation loss or accuracy.** Rejected: no measurement supports it and selection on the
  data that decides the result invites leakage.
- **A research-arm and a separate discard-only resource arm (chosen)** instead of one run that pretends to be both.
- **Build the training driver now.** Deferred: it loads weights and needs its own review; human review is the bottleneck.

## Decision

1. Eligibility is the TR02 rights rule, unchanged: project-authored template text on SRC-O01, O02 or O06 layouts. Eligible means
   rights-documented for resource use, not cleared and not reviewed.
2. Research training (T) needs gates G1 to G7 (named human roster, human template and row review with the predeclared thresholds,
   owner rights decisions, a frozen independent human-authored evaluation set, a pinned protocol and comparators, immutable
   clearance). The tooling counts only decisions by roster-listed humans and refuses AI rows.
3. One run, one epoch, TR02 hyperparameters and pins unchanged so that TR02's measured rates apply, fresh base, `drop_remainder`
   partial-batch rule, fixed diagnostic evaluation and snapshot points, final adapter only, no selection, no retry, fail-closed monitor
   with sustained-throughput, wall-time, memory, swap, AC and thermal rules.
4. A discard-only resource arm (R) is described and justified separately; its adapter is never evaluated or used.
5. New logic lives in new files (`src/axbench_tr/tr03.py`, `training/tr03/`); no frozen experiment or harness file is edited.

## Tradeoffs and consequences

The plan is conservative: it will not run until humans exist, so progress depends on the owner's designation of reviewers. It
reuses TR02's measured configuration (no tuning), which makes the budget credible but forbids exploring other settings. The
research data remain 87 templates, so even a pass supports only a narrow claim.

## Unvalidated assumptions

Durations are extrapolated from a 2.5 minute smoke; a 25% thermal allowance is a guess; the +10 pp and +2 pp margins, 59 rows per
stratum and the 12-sample pressure-warning rule are proposals; the thermal parser is checked only against recorded text; a
fixed seed and one run say nothing about run-to-run variation.

## References

REF-TR-01 to REF-TR-07 (existing, unchanged); no new external source was consulted for this ADR.
