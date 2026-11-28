# ADR 0014: EXP-TR04: a separate, automated, exploratory one-epoch run on audited synthetic data

- Status: Proposed (draft; awaiting technical clearance; nothing approved or run)
- Date: 2026-11-28
- Issue / PR: draft PR based on the EXP-TR03 head `150992f` (PR #15); technical instruction relayed by the project owner after the
  lead accepted the gate-hardening preparation

## Context

EXP-TR03 keeps research training blocked behind human review, rights decisions and an independent human-authored evaluation set,
and its gates are accepted and frozen. The technical lead and owner asked for a separate automated exploratory experiment that
proceeds without waiting for humans: audit the corpus automatically, exclude uncertain rights and incorrect labels, build a
separate synthetic held-out benchmark, implement an executable driver on the proven TR02 infrastructure, and freeze a one-epoch
configuration, then return for clearance. It must say everywhere that it is exploratory, not human-validated and not
deployment-ready.

## Alternatives considered

- **Relax or reuse the TR03 gates for this run.** Rejected: the instruction is to preserve them; this run is a different, weaker
  kind of evidence and must not be able to satisfy them.
- **Keep every rights-documented source (WDI, Eurostat) as TR02/TR03 did.** Rejected for this experiment: the instruction is to
  exclude uncertain rights. The WDI indicators are compiled by IMF, ILO, UN and OECD whose terms were not read, Eurostat's
  dataset-level notes were not found, and SRC-O03 to O05 are unverified. The cost is stated: no World Bank or Eurostat layout
  remains.
- **Trust the 93-package licence addendum for İzmir.** Rejected: it covers the probe and smoke subsets only; all 168 packages used by
  the corpus were fetched and text-screened (a pattern screen, not a legal opinion; one package excluded).
- **Fix suspect labels by hand or by rule.** Rejected: nothing is corrected silently; a record that fails the audit is excluded and
  listed with its reason.
- **Evaluate on a split of the corpus.** Rejected: same templates and author. A separate held-out benchmark with new layouts and
  new phrasings, audited for disjointness from the whole corpus, is used; it is still AI-authored and says so.
- **A new driver from scratch.** Rejected: the TR02 harness is proven on the M5 (C-TR-19); the new driver imports it unchanged and
  adds a config, phase names, thermal and wall-clock rules, a generalized reload and a preflight.
- **Intermediate evaluations and checkpoint selection (as TR03 planned).** Dropped for simplicity and to avoid selection on the data
  that decides the result: validation before, after and in the reload process only.

## Decision

1. Add `src/axbench_tr/tr04*.py` (rights screen, template contract, row audit, mutation report, held-out benchmark),
   `training/tr04/` (config and driver) and tests; edit no frozen file. A narrow `E501` lint exception covers the audit modules.
2. Keep a record only if its text is project-authored template text on a layout whose exact package text was screened clean and whose
   source terms were read; otherwise exclude it with a reason. Exclude records that fail the contract or the row re-derivation.
   Every exclusion is listed.
3. Freeze one run: one epoch over every kept train row exactly once, the TR02 pins and hyperparameters, validation before, after and
   in a reload process, no selection, no retry, with the TR02 stop rules plus thermal, sustained-pressure, throughput-reference and
   wall-clock rules.
4. Freeze the evaluation criteria as descriptive and exploratory (PROTOCOL section 6) and require a separate clearance for any
   training and for any evaluation.

## Tradeoffs and consequences

The data are narrower than TR02's: only İzmir layouts, 87 train lineages, and a small validation set. The automated audit shares an
author with the templates, so it can only find contradictions of the generator's own rules; the injected-fault test measures
detection power, not correctness. A run could look successful (the loss falls) while meaning little, which is why the criteria
separate run health from any held-out observation and forbid quality language. The run could be repeated on a wider rights basis
only after a human rights decision.

## Unvalidated assumptions

Durations are extrapolated from a 2.5 minute smoke; a 25% thermal allowance is a guess; the sustained-throughput ratios are
borrowed from the TR03 proposal; the package text screen is a pattern list written by the same tool; pass-through of the thermal
parser on a Mac was only checked against text recorded in TR02; the held-out generator's layouts are invented by the same tool.

## References

REF-TR-06a (existing licence addendum, reused for the WDI compilers' list), REF-TR-08 (new: İzmir CKAN API records), REF-TR-07
(mlx-lm source, unchanged).
