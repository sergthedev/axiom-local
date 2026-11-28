# EXP-TR03 preparation records (generated; contains no human decision)

Written by `uv run python -m axbench_tr.tr03 all` from the committed corpus `v0.2-rev1`; `uv run python -m axbench_tr.tr03 check`
verifies byte equality with a fresh rebuild. Protocol: `docs/evidence/experiments/EXP-TR03-research-protocol/PROTOCOL.md`.

| File | Content |
|------|---------|
| `INVENTORY.json` | rights rule, eligible and excluded counts, distributions, token statistics, the 122 templates, source attribution |
| `AUDIT.json` | split independence, v0.1 overlap, gold consistency, descriptive shortcut statistics, held-out layout availability |
| `SIZING.json` | exact sample-size arithmetic and the proposed review and evaluation sizes |
| `TEMPLATE_REVIEW_WORKLIST.csv`, `ROW_REVIEW_WORKLIST.csv`, `RIGHTS_WORKLIST.csv`, `HELDOUT_SCHEMA_CANDIDATES.csv` | empty worklists for humans |
| `HUMAN_ROSTER.template.json` | empty template of the owner's roster, with the attestation text (an AI cannot be listed as human) |
| `TRAINING_PLAN.json` | schedule and budget for all eligible train rows (from `training/tr03/tr03.py plan --write`) |
| `MANIFEST.json` | SHA-256 of every generated file |

Filled-in results belong in `ledgers/` (`HUMAN_ROSTER.json`, `TEMPLATE_REVIEW_RESULTS.csv`, `ROW_REVIEW_RESULTS.csv`,
`RIGHTS_DECISIONS.csv`, `HELDOUT_FROZEN.json`, `PREREGISTRATION.json`, `CLEARANCE.json`); that directory does not exist, and no dummy
ledger is committed. `python -m axbench_tr.tr03 gates` reports each gate as absent, invalid, incomplete, criteria not met or
consistent-but-unverified, never as passed, and never opens execution. Guides: `REVIEW_GUIDE.md` and `RIGHTS_DECISION_OPTIONS.md` in the
protocol folder. Record reviewer codes once, in the commit that adds a filled file.
