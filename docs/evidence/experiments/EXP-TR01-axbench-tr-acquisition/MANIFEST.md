# Experiment manifest: EXP-TR01 AXBENCH-TR v0.2 acquisition and draft corpus

- Issue / PR: AXBENCH-TR v0.2 thread (issue and draft PR numbers are filled in on the PR; GitHub write access returned 403 when this was written)
- Date: 2026-11-27
- Status: run (one machine; acquisition and corpus build only; nothing was trained)

## Provenance
- Base commit: `fbd4adc3bab3d50e6912935a6bf982fbec62f10b` (`origin/main`); the work commit is the head of branch `feat/axbench-tr-v02-dataset-acquisition`
- Hardware: cloud container, 4 vCPU, x86_64, Linux (not the MacBook Air M5 target). Python 3.12.3 from `uv sync --locked`
- Deviations from the lockfile: `pyarrow==26.0.1` (build-time, Cognipeer parquet only), `tokenizers==0.23.3` and `jinja2==3.1.6` (token statistics only); none are in `pyproject.toml` or `uv.lock`
- Sources, licences, revisions: [DATA_SOURCES.md](../../../../data/axbench-tr/DATA_SOURCES.md), `data/axbench-tr/sources/registry.json`
- SHA-256 of each download: `data/axbench-tr/sources/ACQUISITION.json`; of each corpus file: `data/axbench-tr/v0.2-draft/MANIFEST.json` (it also records the SHA-256 of the registry and acquisition manifest)
- Seed: `axbench-tr-0.2-draft`; split rule: request ideas assigned to one split before generation, table groups partitioned by hash

## Execution
```
PYTHONPATH=src python -m axbench_tr acquire && python -m axbench_tr verify
PYTHONPATH=src python -m axbench_tr schemas && python -m axbench_tr economic
PYTHONPATH=src python -m axbench_tr build && python -m axbench_tr validate
PYTHONPATH=src python -m axbench_tr docs
PYTHONPATH=src python benchmarks/axbench_tr_token_stats.py > data/axbench-tr/v0.2-draft/TOKEN_STATS.json
pytest tests/test_axbench_tr.py
PYTHONPATH=src python -m axbench_tr licences && python -m axbench_tr review-sample
PYTHONPATH=src python -m axbench_tr export-mlx --out <directory outside the repository>
```
- Exit codes: 0 for each command on the final run. Logs: none kept beyond the files above (no absolute paths in them).

## Results (measured only)
- Corpus: 8,625 records (train 6,942, validation 1,683); Turkish 6,960 (80.7%), English 1,665 (19.3%); 0 validation problems; 0 shared lineages, request texts or table groups between splits (`AUDIT.json`).
- Overlap with AXBENCH v0.1 train and validation (560 requests): 0 exact, 0 near-duplicate pairs at 4-gram Jaccard 0.8.
- Sequence lengths, Qwen3-1.7B tokenizer, EXP-010 rendering: train 3,487,013 tokens (91 rows over 1,024, max 1,219); validation 845,727 (max 890); none over 2,048 (`TOKEN_STATS.json`, `token_lengths_per_example.jsonl`).
- Near-duplicates (char n-gram TF-IDF cosine, lexical proxy): 0 cross-split pairs at 0.9; 8 at 0.8 (`AUDIT.json`).
- Review sample: 300 Turkish records, 100 external + 200 template, 23 frames (`v0.2-draft/review/`).
- Licence evidence: 8 pages or cards fetched, hashed, excerpt found for each (`sources/LICENCE_EVIDENCE.json`).
- Acquisition: 35 sources surveyed, 21 with downloads, about 217 MB of raw files outside Git (`du` of the raw directory); committed data about 15 MB.
- Independent reference: the validator re-derives every gold label from the record's own schema through `axbench.audit.gold_problems`; this is the same contract code, so it checks consistency with the contract, not human judgement. Mutation tests show the validator rejects broken records.

## Limitations and failures
- No human review of any record; Turkish phrasing and licence readings are unreviewed.
- Cognipeer Turkish text is machine-translated; 122 AI-written templates; about 13% repeated request texts.
- The sealed AXBENCH test split was never accessed; the owner-run overlap check has not been run.
- TÜİK, EVDS, ILOSTAT, IMF and İBB were not acquired; no restriction was bypassed.
- Failures found and fixed are listed in the development log entry of 2026-11-27.
