# December 2 readiness report: AXBENCH-TR for the first product demo

Written 2026-11-27, five days before the demo date. Everything marked *measured* was measured on the
files in this branch; everything marked *assumption* was not measured here, because no M5 was available.

## What exists now (measured)
- A reproducible acquisition pipeline: 35 sources surveyed, 21 downloaded, SHA-256 for every file, licence evidence per source.
- A draft corpus of 8,625 records (80.7% Turkish), 0 validation problems, 0 overlap with v0.1 public splits, split isolation proved by the validator.
- Real token counts: 3,487,013 train and 845,727 validation tokens, per-example mean 502, p99 1,076, max 1,219.
- 26 new tests pass; ruff passes.

## Hardware feasibility on MacBook Air M5 16 GB
| Item | Value | Basis |
| --- | --- | --- |
| Base model weights, BF16 | 4,063,515,592 bytes (about 4.1 GB) | measured: REF-033 shard sizes |
| Untuned inference peak, 1.7B | 3.80 GB MLX peak (EXP-009, C-053) | measured, other thread's data |
| Longest training example | 1,219 tokens: over the old 1,024 setting (91 rows), under 2,048 | measured |
| Training memory | EXP-010 pilot: MLX peak 4.71 GB on sequences up to 514 tokens; unmeasured for 1,219 | EXP-010 record plus an open measurement (the peak probe in the proposed smoke) |
| Training time per pass | about 37 to 76 minutes | derived from the EXP-010 record (1,586 tok/s compute, 760 tok/s whole loop) applied to 3,487,013 tokens; extrapolation, not measured here |
| Storage | corpus 15 MB, raw acquisition 217 MB, weights 4.1 GB, one adapter well under 0.5 GB | corpus and raw measured; adapter size an assumption |

A fanless laptop may throttle on a multi-hour run, which EXP-009 already showed as a risk for long jobs
(the 1.7B run was interrupted once by an external cause). Plan for pauses and resumable checkpoints.

## Can the corpus be used by December 2?
Technically yes: the files export to the MLX chat format and fit the context length. Whether it should be
used is a different question, and the answer today is **not for a demo claim**:
1. No human has reviewed any record. Turkish phrasing quality is unknown.
2. Template diversity is narrow (122 templates); a model may memorise phrasing.
3. Sealed-test overlap has not been checked; the owner can run it in minutes without exposing the test.
4. A training run is not approved. The smoke (`PROPOSED_EXP-TR02.md`) is the first step; a full pass is derived at about 37 to 76 minutes from EXP-010 numbers, but that is an extrapolation until the smoke measures it.
5. The old 1,024-token setting would truncate 91 rows; the run needs a declared 2,048 setting, with memory unmeasured above 514 tokens.

## Risks, ranked
1. **Quality risk (high):** AI-authored, unreviewed data. Mitigation: a 300-record native-speaker review (about a day).
2. **Schedule risk (high):** training plus evaluation of three arms is tight in five days. Mitigation: run only arms A and B, or only B.
3. **Coverage risk (medium):** TÜİK, EVDS, ILOSTAT and IMF layouts are missing, so "official Turkish statistics" is municipal, World Bank and Eurostat today. Mitigation: owner downloads TÜİK/EVDS exports by hand; the pipeline already ingests CSV headers.
4. **Licence risk (medium):** card metadata was not verified; one source has conflicting badges. Mitigation: review `DATA_SOURCES.md` before any release.
5. **Contamination risk (low to medium):** v0.1 public overlap is 0; sealed overlap unknown.
6. **Thermal and memory risk (medium):** assumption; mitigated by the EXP-010 stop rules.

## Decisions needed from the owner
1. Name the reviewers for the 300-record sample in `data/axbench-tr/v0.2-draft/review/`.
2. Decide when to run the sealed-overlap command (not yet run, per the owner's instruction).
3. Approve or decline EXP-TR02 (see `PROPOSED_EXP-TR02.md`).
4. Provide TÜİK/EVDS/ILOSTAT/IMF exports, if wanted.

## Update after the technical lead's pre-training list (2026-11-27)
- Review sample, licence evidence, near-duplicate audit, length gate and an export for the smoke now exist (PR #13).
- Replay failures: root cause is the order of two OpenBLAS pools with identical prefixes in the environment
  fingerprint; a one-line sort-key fix (`src/axiom/environment.py`) made all 74 replay and regression tests pass in 3
  runs here. It is **not applied** because it touches `src/axiom`; it needs the owner's approval as a separate change.
- CI on the first PR commit: macOS passed; Windows failed one test because Git rewrote line endings in the hashed
  corpus files. `.gitattributes` now marks `data/axbench-tr` data files `-text`, as for `data/axbench`.
