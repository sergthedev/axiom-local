# Experiment manifest: EXP-TR01 AXBENCH-TR v0.2 acquisition and revised corpus (v0.2-rev1)

- Issue / PR: [Issue #12](https://github.com/sergthedev/axiom-local/issues/12), draft [PR #13](https://github.com/sergthedev/axiom-local/pull/13)
- Date: 2026-11-27 (acquisition and first draft), revised twice the same day after the technical reviews of PR #13 (second revision: stricter intent gates, translation quarantine, rights-cleared smoke subset, harness revision)
- Status: run (one machine; acquisition, corpus build and audits only; nothing was trained, no model was loaded)

## Provenance
- Base commit: `fbd4adc3bab3d50e6912935a6bf982fbec62f10b` (`origin/main`); the work commit is the head of branch `feat/axbench-tr-v02-dataset-acquisition`
- Hardware: cloud container, x86_64, Linux (not the MacBook Air M5 target). Python 3.12 from `uv sync --locked`
- Build environment: isolated locked project `envs/data` (pyarrow 26.0.1, tokenizers 0.23.3, jinja2 3.1.6, `exclude-newer` 2026-11-12), so the main `uv.lock` is unchanged. The build verifies every raw file against `ACQUISITION.json` first and stops on a mismatch
- Sources, licences, revisions: [DATA_SOURCES.md](../../../../data/axbench-tr/DATA_SOURCES.md), `data/axbench-tr/sources/registry.json`. 35 surveyed; 21 downloaded; **7 supply content to the corpus or data files** (SRC-N01, O01 to O06); SRC-N03 is quarantined
- SHA-256 of each download: `data/axbench-tr/sources/ACQUISITION.json`; of each corpus file: `data/axbench-tr/v0.2-rev1/MANIFEST.json` (it also records the registry and acquisition-manifest hashes)
- Seed: `axbench-tr-0.2-rev1` (corpus), `axbench-tr-0.2-rev1-review-sample` (review sample). The first corpus, `v0.2-draft` (seed `axbench-tr-0.2-draft`, 8,625 records), is kept unchanged as a superseded record

## Execution
```
uv sync --locked --project envs/data
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr acquire && ... verify
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr schemas && ... economic && ... privacy-audit
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr build && ... validate
PYTHONPATH=src envs/data/.venv/bin/python -m axbench_tr licences && ... review-sample && ... docs
PYTHONPATH=src envs/data/.venv/bin/python benchmarks/axbench_tr_token_stats.py --max-seq-length 2048 \
    --per-example data/axbench-tr/v0.2-rev1/token_lengths_per_example.jsonl
python -m axbench_tr export-mlx --out <directory outside the repository>
pytest tests/test_axbench_tr.py tests/test_tr02_harness.py
# on the M5, before clearance (no weights): see training/tr02/README.md, `check-iterator`
```
- Exit codes: 0 on the final run of each command. No logs kept beyond the files named (no absolute paths in them).
- Rebuild check: `train.jsonl`, `validation.jsonl`, `MANIFEST.json` and `AUDIT.json` hashed after a build in the isolated environment and again after a build in the main environment: identical. A fresh download from the upstream hosts was not repeated.

## Results (measured only)
- Corpus: 7,269 records (train 5,873, validation 1,396); Turkish 5,757 (79.2%), English 1,512 (20.8%); 6,700 template and 569 external-adapted (97 Turkish external PLANs tagged `translation_unverified`, quarantined from product training); 4,261 tagged economic; 0 validation problems. The first revision had 7,432 records; the second revision's stricter English gate and removal of one near-duplicate validation lineage account for the difference (163 fewer external rows).
- Split isolation (all origins): 0 shared lineages, table families, request texts, schema fingerprints or upstream rows across splits.
- Overlap with AXBENCH v0.1 train and validation (560 requests): 0 exact, 0 near-duplicate pairs at 4-gram Jaccard 0.8.
- Near-duplicates (char n-gram TF-IDF cosine, lexical proxy, not semantic): 0 cross-split pairs at 0.9, 5 at 0.8, after removing one validation lineage.
- Adapted plans: only plain average, minimum or maximum questions whose measure and group words equal the bound columns' (no extra modifier, no negation or filter word) qualify; exclusion reasons and counts in `AUDIT.json`. The Turkish measure and group are **not** verified; adversarial mutation tests cover the English gate and document the Turkish limitation.
- Sequence lengths (pinned tokenizer, joint encoding): train 3,024,797 tokens (70 rows over 1,024, max 1,343; 3,018,924 in the harness convention without the template's trailing newline); validation 727,890 (max 891); none over 2,048.
- Review sample: 390 records (300 Turkish) with predeclared gates (`v0.2-rev1/review/`); no review decision exists.
- Licence evidence: 8 pages or cards fetched, hashed, excerpt found each; licence texts for Apache-2.0 and CC BY 4.0; economic CSV privacy audit (`economic/PRIVACY_AUDIT.json`).
- EXP-TR02 harness checks (no model; Linux, mlx 0.32.4 CPU build, tokenizer only): export, corpus, per-example and checkpoint pins; all rows render to the measured lengths; seeded order, padded widths and loss mask checked against the real mlx-lm iterator for the probe and smoke subsets. **Finding:** the stock `default_loss` scores one extra target per row (the zero padding after `<|im_end|>`; 37 mask tokens against a 36-token answer on the first probe row); the harness uses an exact-mask loss. The no-weight tests (62 pass with the MLX CPU build, 57 with 5 skipped without it) do not show that the GPU path, memory or timing work.

## Limitations and failures
- No human review of any record, gold plan, Turkish text or licence reading.
- 92% of records come from 122 AI-authored templates; deduplication is lexical only; external Turkish is machine-translated and checked only for the aggregate word and added constraints (hence `translation_unverified`).
- Upstream Gretel generation terms and the SRC-N03 licence conflict are unresolved: no final corpus or product clearance; the smoke and probe use only rights-cleared template rows on schemas from SRC-O01, O02 and O06 (terms documented per dataset in `sources/LICENCE_EVIDENCE_ADDENDUM_1.json`; ECB, OECD and TCMB schemas excluded) and are discard-only.
- The sealed AXBENCH test split was never accessed; the owner-run overlap check has not been run.
- TÜİK, EVDS, ILOSTAT, IMF and İBB were not acquired; no restriction was bypassed.
- Mistakes found and fixed are in the development log entries of 2026-11-27, including the draft's SUM, filter and ranking labels (superseded) and a rebuild comparison that passed only because the build had crashed (caught and redone).
