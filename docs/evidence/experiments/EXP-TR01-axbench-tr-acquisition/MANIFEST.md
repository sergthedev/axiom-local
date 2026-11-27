# Experiment manifest: EXP-TR01 AXBENCH-TR v0.2 acquisition and revised corpus (v0.2-rev1)

- Issue / PR: [Issue #12](https://github.com/sergthedev/axiom-local/issues/12), draft [PR #13](https://github.com/sergthedev/axiom-local/pull/13)
- Date: 2026-11-27 (acquisition and first draft), revised the same day after the technical review of PR #13
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
```
- Exit codes: 0 on the final run of each command. No logs kept beyond the files named (no absolute paths in them).
- Rebuild check: `train.jsonl`, `validation.jsonl`, `MANIFEST.json` and `AUDIT.json` hashed after a build in the isolated environment and again after a build in the main environment: identical. A fresh download from the upstream hosts was not repeated.

## Results (measured only)
- Corpus: 7,432 records (train 5,998, validation 1,434); Turkish 5,837 (78.5%), English 1,595 (21.5%); 6,700 template and 732 external-adapted; 4,281 tagged economic; 0 validation problems.
- Split isolation (all origins): 0 shared lineages, table families, request texts, schema fingerprints or upstream rows across splits.
- Overlap with AXBENCH v0.1 train and validation (560 requests): 0 exact, 0 near-duplicate pairs at 4-gram Jaccard 0.8.
- Near-duplicates (char n-gram TF-IDF cosine, lexical proxy, not semantic): 0 cross-split pairs at 0.9, 7 at 0.8, after removing one validation lineage.
- Adapted plans: only plain average, minimum or maximum questions qualify; exclusion reasons and counts in `AUDIT.json`.
- Sequence lengths (pinned tokenizer, joint encoding): train 3,072,061 tokens (70 rows over 1,024, max 1,343); validation 742,194 (max 891); none over 2,048.
- Review sample: 390 records (300 Turkish) with predeclared gates (`v0.2-rev1/review/`); no review decision exists.
- Licence evidence: 8 pages or cards fetched, hashed, excerpt found each; licence texts for Apache-2.0 and CC BY 4.0; economic CSV privacy audit (`economic/PRIVACY_AUDIT.json`).
- EXP-TR02 harness checks (no model): export hashes pinned; all rows render to the measured lengths; seeded order equals the stock mlx-lm iterator.

## Limitations and failures
- No human review of any record, gold plan, Turkish text or licence reading.
- 90% of records come from 122 AI-authored templates; deduplication is lexical only; external Turkish is machine-translated and checked only for the aggregate word and added constraints.
- The sealed AXBENCH test split was never accessed; the owner-run overlap check has not been run.
- TÜİK, EVDS, ILOSTAT, IMF and İBB were not acquired; no restriction was bypassed.
- Mistakes found and fixed are in the development log entries of 2026-11-27, including the draft's SUM, filter and ranking labels (superseded) and a rebuild comparison that passed only because the build had crashed (caught and redone).
