# EXP-TR04 data records (generated; automated, exploratory, not human-reviewed)

Written by `uv run python -m axbench_tr.tr04 all` (rights screen, template contract, row audit, mutation report, kept-id lists) and
`uv run python -m axbench_tr.tr04_heldout all` (held-out benchmark). `... check` verifies byte equality with a fresh rebuild.
`uv run python -m axbench_tr.tr04 rights-check` verifies the committed package evidence offline. Protocol:
`docs/evidence/experiments/EXP-TR04-exploratory-lora/PROTOCOL.md`. Nothing here is human-reviewed or human-validated, and no model
was involved in producing it.

| File | Content |
|------|---------|
| `KEEP_TRAIN.txt`, `KEEP_VALIDATION.txt` | the 3,164 and 376 record ids kept for the EXP-TR04 run (one id per line, file order = export order) |
| `EXCLUSIONS.csv` | every other record of v0.2-rev1 with its stage (`rights`, `label`, `trim`) and one reason |
| `AUDIT.json` | counts, distributions, rights rule and per-source decisions, independence of the kept splits, limitations |
| `TEMPLATE_CONTRACT.json` | the cue patterns and the verdict for each of the 122 lineages |
| `MUTATION_REPORT.json` | detection of deliberately injected faults on synthetic copies (power, not accuracy) |
| `rights/IZMIR_PACKAGE_NOTES.json` | kept text fields (no contact data) and raw SHA-256 of the CKAN `package_show` record of each of the 168 İzmir packages, screen flags, retrieval date |
| `heldout/HELDOUT.jsonl` | 346 synthetic held-out rows (AI-authored, fictional layouts, never trained or validated on) |
| `heldout/HELDOUT_AUDIT.json`, `heldout/HELDOUT_MANIFEST.json` | disjointness audit against the corpus and AXBENCH v0.1, SHA-256 of the held-out files |
| `MANIFEST.json` | SHA-256 of every generated file above and of the corpus inputs |

The MLX chat export used by the run is written outside the repository (`python -m axbench_tr.tr04 export-mlx --out <dir>`); its
hashes are pinned in `training/tr04/config.json`. There is no test split; the sealed AXBENCH test is not referenced.
