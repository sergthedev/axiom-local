# Experiment manifest: EXP-TR03 research protocol preparation (no training)

- Issue / PR: #12 (stays open); draft PR based on the PR #13 evidence head. Technical task: PR #13 comment 7235303471
  (`EXP-TR03-PREP-50bea1f`), owner-authorized preparation only.
- Date: 2026-11-28
- Status: **proposed (preparation only).** Nothing was trained, loaded, generated or scored. This record documents prepared
  tooling and deterministic audits of a committed draft corpus. It is not an experiment result about any model.

## Provenance
- Base (immutable PR #13 evidence head): `50bea1f968ec482153dd2d684e34b3f758f14813`; code head of the frozen TR02 harness: `ed4e75b…`
  (unchanged; no file of EXP-TR02, EXP-010, `src/axiom*` or any existing module is edited by this change).
- Hardware and OS: a Linux x86-64 cloud container (no Apple Silicon, no GPU, no MLX). Nothing here measures speed or memory.
- Python and packages: the repository's locked Core environment (`uv sync --locked`), Python 3.12 per `pyproject.toml`; no model
  environment was used and no model package was imported by the new code (a test checks the plan module's source).
- Dataset: `data/axbench-tr/v0.2-rev1/` (unchanged): `train.jsonl` `d5466e80…1372`, `validation.jsonl` `d8e302d0…0301`,
  per-example tokens `8dde1cfb…5d`. Eligible subset: `synthetic_template` rows on SRC-O01, O02 and O06 layouts, 5,091 train and 1,274
  validation. No test split exists in this corpus; the sealed AXBENCH test was not accessed.
- Plan and config: `training/tr03/config.json` (inherits `training/tr02/config.json` sha256 recorded inside; seed 2026).

## Execution (all local, deterministic, no model)
- `uv run python -m axbench_tr.tr03 all` writes `data/axbench-tr/tr03-prep/*` (about 10 s).
- `uv run python -m axbench_tr.tr03 check`: the committed files equal a fresh rebuild.
- `uv run python training/tr03/tr03.py verify-config | plan --write | check-plan`.
- `uv run python -m axbench_tr.tr03 gates` and `training/tr03/tr03.py readiness`: exit status 1, every human gate closed (expected).
- `uv run pytest` (full suite), `uv run ruff check .`, `uv run ruff format --check .`: results in the PR description.
- Sanitized log path: none needed; no run produced a private log. Outputs contain no absolute path.

## Results (measured only)
- Counts, token statistics, split-independence checks, gold-contract checks, descriptive shortcut statistics and held-out layout
  availability: `data/axbench-tr/tr03-prep/INVENTORY.json` and `AUDIT.json`, summarised in `PROTOCOL.md` section 8. Claims C-TR-20 to
  C-TR-23 in `docs/evidence/CLAIMS.md`.
- Sizing arithmetic (exact; Wilson from `axbench.stats`): `SIZING.json`; known answers 59 and 29 verified in tests.
- Time budget for one epoch: extrapolated from the TR02 smoke, not measured at this scale (`TRAINING_PLAN.json`).
- No model result exists: no hypothesis in `PROTOCOL.md` is supported or refuted.
- Independent reference: the gold audit uses the router contract (`axbench.audit.gold_problems`), the budget arithmetic is checked
  by hand in a test, and the thermal parser is checked against the 53 samples recorded in the committed TR02 evidence. A hash or a
  rebuild equality shows reproducibility, not correctness.

## Limitations and failures
- All templates and rows are AI-authored; no human decision exists; the worklists are empty by design and the gate checker refuses
  AI judgements. Research training is blocked by gates G1 to G7 (`PROTOCOL.md` section 6).
- Lexical near-duplicate counts are a proxy, not semantic deduplication; the 87 train templates limit independence (rows are not
  independent examples).
- The thermal parser and wall-time budget are unverified on a Mac in this task; no no-weight M5 preflight was run (none requested).
- No driver exists to run T or R; it is a separate, reviewed task if the lead wants it.
- A false alarm was found and fixed in the audit itself before commit: a "label conflict" count that turned out to be column-id
  numbering and unordered column lists, now compared by column name with unordered sets (0 true conflicts).
