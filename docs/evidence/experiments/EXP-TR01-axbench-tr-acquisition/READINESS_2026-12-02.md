# December 2 readiness report: AXBENCH-TR for the first product demo (revision 1)

Written 2026-11-27 after the technical review of PR #13. *Measured* means measured on the files in this branch;
*derived* means calculated from measured figures; *assumption* means not measured, because no M5 was available.

## Where things stand
- **Acquisition evidence:** accepted by the technical review (35 sources surveyed, 21 downloaded, 7 used, SHA-256 for
  every file, licence evidence per source). SRC-N03 is quarantined.
- **Corpus v0.2-rev1:** 7,432 records, 78.5% Turkish, built so that adapted plans say what the whole request says (no
  SUM, filters or rankings), with split isolation proved on lineages, families, texts, schema fingerprints and upstream
  rows for every origin. Byte-identical rebuilds in two environments. 90% of records are template-derived; this
  limits what a model can learn about real Turkish phrasing.
- **Not cleared for training.** The gates are: native-speaker review of the stratified sample with a second reviewer
  and the predeclared thresholds; independently authored Turkish held-out tasks; licence items resolved.
- **EXP-TR02 smoke:** protocol v2 and an independent harness are prepared and tested without any model. No run is
  approved.

## Hardware feasibility on MacBook Air M5 16 GB
| Item | Value | Basis |
| --- | --- | --- |
| Base model weights, BF16 | about 4.1 GB | measured: REF-033 shard sizes |
| Untuned inference peak, 1.7B | 3.80 GB MLX peak (EXP-009, C-053) | measured, other thread's data |
| Longest training example | 1,343 tokens: over the old 1,024 setting (70 train rows), under 2,048 | measured, pinned tokenizer |
| Training memory | EXP-010: MLX peak 4.71 GB at up to 514 tokens; **unmeasured at 1,343** | the peak probe in the smoke measures it |
| Time per full train pass | about 32 to 67 minutes | derived: 3,066,063 tokens at EXP-010's 1,586 tok/s (compute) and 760 tok/s (loop); an extrapolation until the smoke measures it |
| Storage | corpus about 12 MB, raw acquisition 217 MB, weights 4.1 GB | measured; adapter size an assumption |

A fanless laptop may throttle on a long run; EXP-010 saw later windows about 14% slower.

## Can the corpus be used by December 2?
The files export to the MLX chat format and fit a 2,048-token limit with margin. Whether it should be used is a
separate question, and today the answer is **not for a quality claim**:
1. No human has reviewed any record; Turkish phrasing quality is unknown.
2. 90% template-derived; a model may learn the templates. More genuine Turkish economic diversity needs real authored
   tasks, not padding.
3. Independence: no independently authored held-out set exists yet (kit prepared).
4. A training run is not approved. The smoke measures feasibility only and has no accuracy gate.
5. The sealed-test overlap check has not been run (owner instruction); it belongs to an independent final protocol.

## Risks, ranked
1. **Quality (high):** AI-authored, unreviewed data. Mitigation: the 390-record review (300 Turkish) and the second review.
2. **Schedule (high):** review, held-out authoring and training inside five days is tight. Prioritise correctness over
   example count; a credible demo may rest on the existing v0.1 model plus a small reviewed Turkish set.
3. **Coverage (medium):** TÜİK, EVDS, ILOSTAT and IMF layouts are missing; Turkish official data is municipal, World
   Bank and Eurostat today. Owner-downloaded exports can be ingested by header.
4. **Licence (medium):** card metadata was read, not verified; SRC-N03 conflict and Gretel generation terms open.
5. **Thermal and memory (medium):** assumption; mitigated by supervised stop rules.

## Decisions needed from the owner
1. Name the reviewers (and a second reviewer) for the sample in `data/axbench-tr/v0.2-rev1/review/`, and name human authors for the held-out tasks.
2. Confirm the predeclared review gates.
3. Approve or decline the separate minimal PR that fixes the environment-fingerprint order (replay failures; it touches `src/axiom`, so it is not in PR #13).
4. Approve or decline running EXP-TR02 (`PROPOSED_EXP-TR02.md`) on the M5 after review.
5. Provide TÜİK/EVDS/ILOSTAT/IMF exports, if wanted.

## Replay failures and CI
Root cause: the environment fingerprint sorts `detected_pools` by API and prefix only, and NumPy and SciPy bundle two
OpenBLAS builds with the same prefix, so the order varies between fresh processes. A one-line sort-key fix made all 74
replay and regression tests pass in 3 runs here; it is **not applied** in PR #13 or PR #11. macOS and Windows CI results
are on the PR.
