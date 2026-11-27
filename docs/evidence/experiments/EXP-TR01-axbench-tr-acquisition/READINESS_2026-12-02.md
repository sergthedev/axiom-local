# December 2 readiness report: AXBENCH-TR for the first product demo

Written 2026-11-27, five days before the demo date. Everything marked *measured* was measured on the
files in this branch; everything marked *assumption* was not measured here, because no M5 was available.

## What exists now (measured)
- A reproducible acquisition pipeline: 35 sources surveyed, 21 downloaded, SHA-256 for every file, licence evidence per source.
- A draft corpus of 8,625 records (80.7% Turkish), 0 validation problems, 0 overlap with v0.1 public splits, split isolation proved by the validator.
- Real token counts: 3,493,955 train and 847,410 validation tokens, per-example mean 503, p99 1,077, max 1,220.
- 26 new tests pass; ruff passes.

## Hardware feasibility on MacBook Air M5 16 GB
| Item | Value | Basis |
| --- | --- | --- |
| Base model weights, BF16 | 4,063,515,592 bytes (about 4.1 GB) | measured: REF-033 shard sizes |
| Untuned inference peak, 1.7B | 3.80 GB MLX peak (EXP-009, C-053) | measured, other thread's data |
| Longest training example | 1,220 tokens, under the default 2,048 max length | measured |
| Training memory (LoRA, batch 4) | not measured | assumption: weights plus activations should fit in 16 GB at batch 4 and 1,220 tokens, but this needs the EXP-010 measurement |
| Training time per pass | 3.2 h / 1.6 h / 0.8 h at 300 / 600 / 1,200 tok/s | arithmetic on measured tokens; the throughputs are placeholders |
| Storage | corpus 15 MB, raw acquisition 217 MB, weights 4.1 GB, one adapter well under 0.5 GB | corpus and raw measured; adapter size an assumption |

A fanless laptop may throttle on a multi-hour run, which EXP-009 already showed as a risk for long jobs
(the 1.7B run was interrupted once by an external cause). Plan for pauses and resumable checkpoints.

## Can the corpus be used by December 2?
Technically yes: the files export to the MLX chat format and fit the context length. Whether it should be
used is a different question, and the answer today is **not for a demo claim**:
1. No human has reviewed any record. Turkish phrasing quality is unknown.
2. Template diversity is narrow (122 templates); a model may memorise phrasing.
3. Sealed-test overlap has not been checked; the owner can run it in minutes without exposing the test.
4. A training run is not approved. If approved on Nov 28, one pass at an assumed 600 tok/s is about 1.6 h and three arms about 5 to 15 h, leaving little room before Dec 2.

## Risks, ranked
1. **Quality risk (high):** AI-authored, unreviewed data. Mitigation: a 300-record native-speaker review (about a day).
2. **Schedule risk (high):** training plus evaluation of three arms is tight in five days. Mitigation: run only arms A and B, or only B.
3. **Coverage risk (medium):** TÜİK, EVDS, ILOSTAT and IMF layouts are missing, so "official Turkish statistics" is municipal, World Bank and Eurostat today. Mitigation: owner downloads TÜİK/EVDS exports by hand; the pipeline already ingests CSV headers.
4. **Licence risk (medium):** card metadata was not verified; one source has conflicting badges. Mitigation: review `DATA_SOURCES.md` before any release.
5. **Contamination risk (low to medium):** v0.1 public overlap is 0; sealed overlap unknown.
6. **Thermal and memory risk (medium):** assumption; mitigated by the EXP-010 stop rules.

## Decisions needed from the owner
1. Approve the human-review sample (and who reviews).
2. Run the sealed-overlap command and report the counts.
3. Approve or decline EXP-TR02 (see `PROPOSED_EXP-TR02.md`).
4. Provide TÜİK/EVDS/ILOSTAT/IMF exports, if wanted.
