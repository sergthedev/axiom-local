# Experiment manifest: EXP-TR04 exploratory automated LoRA preparation (no training)

- Issue / PR: draft PR based on the EXP-TR03 head `150992f` (PR #15); technical instruction relayed by the project owner.
- Date: 2026-11-28
- Status: **proposed (preparation only), exploratory, not human-validated, not deployment-ready.** Nothing was trained, loaded,
  generated or scored. This record documents prepared tooling and deterministic, automated audits of a committed draft corpus.
  It is not an experiment result about any model. EXP-TR03 and its human-review gates are frozen and unchanged.

## Provenance
- Base: `150992f` (EXP-TR03 gate-hardening head). No file of EXP-TR02, EXP-TR03, EXP-010, `src/axiom*` or any existing module is
  edited; `pyproject.toml` gains a narrow `E501` lint exception for `src/axbench_tr/tr04*.py`; `.gitattributes`, indexes and
  evidence logs get appended lines.
- Hardware and OS: a Linux x86-64 cloud container (no Apple Silicon, no GPU, no MLX model path). Nothing here measures speed or memory.
- Python and packages: the repository's locked Core environment (`uv sync --locked`), Python 3.12; the model environment was not
  used for the audits. Network: only the İzmir CKAN API (REF-TR-08) on 2026-11-28 for the package records.
- Dataset: `data/axbench-tr/v0.2-rev1/` (unchanged; hashes below). No test split exists in this corpus; the sealed AXBENCH test was not accessed.

## Frozen hashes (SHA-256, copied from `training/tr04/config.json`)
- MLX export (written outside the repository):
  - `ROW_IDS.json`: `8b5aac32b70e0876fda682815521d5c0bdbf459e6ea19d61ad884e3fce0904f1`
  - `tr04/train.jsonl`: `594b9879375f9c0b384b6b5aff6838b6d7f571259428bf50b6cc201e4b0ae860`
  - `tr04/valid.jsonl`: `5181935fe71f4a61b5958270e652f7a8c78541195904beaa515621915b676d04`
  - `EXPORT_MANIFEST.json`: `bb946467661f66171bd1f0b5f2a3a1eff3012b9ce98ad8360c6ee97cc460eb34`
- Audit outputs (`data/axbench-tr/tr04/`):
  - `AUDIT.json`: `6764607af8967fc0ccd533b83a07957c89d21126bd5503d49b09289f80628237`
  - `EXCLUSIONS.csv`: `3c146b5ef1c2d80755b0e7b045cbf46ea97276d254014cc346718368a4d3ec4b`
  - `KEEP_TRAIN.txt`: `9785351c03d7b2253a8e673875a9c9f977f33e2332dee865442bfda6ff1c4848`
  - `KEEP_VALIDATION.txt`: `15d43f60e3c17f8224dd649c1d8daa200632b21fa02e7e8e48cc2cc68bb55748`
  - `MANIFEST.json`: `c13b400aed529828d78c22b38c140b55c06ae43d13d9667e23ef1509aec7e7f2`
  - `MUTATION_REPORT.json`: `d31f26a644954143e4c40cd121772a701b57fcd1812e1dfd8dd487220f61e1bb`
  - `TEMPLATE_CONTRACT.json`: `64e87d8ba56fce4df6fefbf986eee0744c904344ae2e498294e1c25c6c2188db`
  - `rights/IZMIR_PACKAGE_NOTES.json`: `972671a4eb181a69264c58d81ba8ef2a9f7d369348af4e81507c9965a7a6a989`
- Held-out (`data/axbench-tr/tr04/heldout/`):
  - `HELDOUT.jsonl`: `b0a1dd11c5c5994cd2c979e281f8c980b84fbef2a2a8c3ec7879df5f6bb22cea`
  - `HELDOUT_AUDIT.json`: `1e6d6ceadbb2bb5e1893adddfcf44e39dbb2a21889296cf890f0f9b3d6b20cfa`
  - `HELDOUT_MANIFEST.json`: `5856a15aa3cfb3eea8dce0d7351c3e3c052b198b06622f77335864b8602414f6`
- Corpus inputs (unchanged): `train.jsonl` `d5466e80c4e9cf06753d8fbc5c5d28476c35ea309b9f78e34a1a6ee94cb11372`, `validation.jsonl` `d8e302d02b05579f50b035a2840445c19603ba555100ac6636a87decb0eb6301`, per-example `8dde1cfb9b20c2e9ce095cd0161e860a406c6b8ffad1f24e75d1bbfba175c95d`.
- Config `training/tr04/config.json`: `d5100d12f0c0382e2ace23ed7c63c345e50c8a91d02876b4038ff00bb3786f96`; it pins the TR02 config hash `7d31d916621523281d2226de9802671ddc63b05aad219388064a275abaf5301c`.

## Execution (all local, deterministic, no model)
- `uv run python -m axbench_tr.tr04 all` then `check`; `uv run python -m axbench_tr.tr04_heldout all` then `check`;
  `uv run python -m axbench_tr.tr04 rights-check` (offline); `... tr04 export-mlx --out <dir outside the repo>`;
  `uv run python training/tr04/tr04.py pin --export <dir>`, `verify-config`, `plan`.
- `uv run pytest` (full suite), `uv run ruff check .`, `uv run ruff format --check .`: results in the PR description.
- No run produced a private log. Outputs contain no absolute path and no contact data (a test scans for them).

## Results (measured only)
- Rights and label audit: 3,164 train and 376 validation records kept of 5,873 and 1,396; exclusion counts per stage and reason in
  `AUDIT.json` (rights 2,672 + 1,014, label 34 + 3, trim 3 + 3). Claims C-TR-25 and C-TR-26.
- Held-out benchmark: 346 rows, disjointness audit passed. Claim C-TR-27.
- Driver and config: 90 no-weight tests pass; budget extrapolated. Claim C-TR-28.
- No model result exists: no hypothesis is supported or refuted (C-TR-29 is proposed).
- Independent reference: none. The label audit restates the generator's rules, the cue lists and the package screen were written by
  the tool that wrote the templates, and a rebuild equality shows reproducibility, not correctness. The injected-fault report
  measures detection of faults that break exactly those rules.

## Limitations and failures
- All rows, templates and held-out items are AI-authored and unreviewed; no human decision exists. Not human-validated.
- The strict rights rule keeps İzmir layouts only; validation is small (35 lineages, 23 families).
- The package screen is a pattern list, not a legal opinion.
- Durations and thermal behaviour are extrapolated, not measured at this scale.
- No generation runner exists for the held-out evaluation; it needs its own review and clearance.
- Faults found and fixed in this work before freezing: 15 lexicon false alarms in the template contract; a rights-evidence gap (93
  of 168 packages screened); `KeyError`, `StopIteration` and a self-replacing mention in the fault injection; two held-out phrasings
  that failed their own contract; stale counts in a docstring and a limitation string. No check was weakened.
- No M5 result is recorded in this manifest unless a separate preflight record is added to this folder.
