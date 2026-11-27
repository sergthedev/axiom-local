# EXP-TR02 second no-weight preflight on the MacBook Air M5, at head d5f6cd5 (evidence record)

Status: **run once, all steps passed.** This is the single fresh no-weight preflight the owner-authorized review of PR #13
asked for after the probe and smoke subsets changed. Immutable head `d5f6cd56c0d3a12ecb9e917570d028788809ac78`; CI was green on
macOS and Windows at that head before the run. **No model weights were loaded, `supervise` and `run` were not executed, nothing
was trained or generated, the sealed test was not touched, and no code was changed.** The feasibility run itself is **not**
cleared: it needs technical review of this evidence and a separate owner authorization at a named head.

Source of the numbers: the sanitized raw logs in `at-d5f6cd5-sanitized-logs/` (byte-exact copies, hashes in
`LOG_VERIFICATION.md`). Everything is measured on one machine on 2026-11-27. The first preflight at 96cbdf7 is kept as
`PREFLIGHT_NOWEIGHT.md`; its subset is superseded.

## Device and safety checks (saved in `00_environment.txt`, before any step)
| Item | Value |
| --- | --- |
| Power, pressure, other work | AC power (95%); memory pressure level 1 (and 1 before and after steps 5 and 6); no MLX or Python process running |
| Machine | Apple M5, `hw.model` Mac17,3, 17,179,869,184 bytes, macOS 27.0.1 (26A434); uv 0.13.10 |
| Worktree | new isolated worktree, `HEAD` exactly `d5f6cd56c0d3a12ecb9e917570d028788809ac78`, `git status --porcelain --untracked-files=all` empty (also after step 1) |
| Checkpoint | provisioned Qwen/Qwen3-1.7B, `PROVENANCE.json` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; hashed only, never loaded |
| Environment | `uv sync --locked --project envs/model` (45 packages incl. mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.2, tokenizers 0.23.3); `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` for every step |
| MLX | import check only: `Device(gpu, 0)`, mlx 0.32.4, mlx-lm 0.32.2 |
| Output directories | the export and probe directories did not exist beforehand; the probe directory was not created by the preflight |

## Export hashes
Compared with the values pinned in `training/tr02/config.json`: all equal.

| File | SHA-256 |
| --- | --- |
| `EXPORT_MANIFEST.json` | `f99d236d0681852e3ba0c57959f6ace89a41fabc4c96bd4fd7bf5ee9ba033b92` |
| `ROW_IDS.json` | `2646499c1583414e50f39fd59f25a1e6d7a519eb39a826892b51edcf26412197` |
| `full/train.jsonl` | `68edeb0674cd0b445bbedc2a515f1bef10ee2cd9cbdad4f8eba7a615dbbfe4ed` |
| `full/valid.jsonl` | `edcd5bcf492dc8a5ecc617d371a67486656759aba472a338105f48505ddd0066` |
| `smoke/train.jsonl` | `455099332e82f8803c9d15ef0ac5d1ae397faeb44b73265a40e49f34adf2ce2c` |
| `smoke/valid.jsonl` | `6b56ff796a83ea712969db29d2444fb5945a16e666662518ede747ef6a8c543a` |
| `peak_probe/train.jsonl` and `valid.jsonl` | `787296439900464802bf4600daaead77194d1c1a4897b9c32f418101e79d4f76` (identical to the first run) |

## Steps (each exit status 0, each run once; `RUN="uv run --locked --project envs/model python training/tr02/tr02.py"`)
| # | Command | Wall | Max RSS | Result |
| --- | --- | ---: | ---: | --- |
| 0 | `uv sync --locked --project envs/model` | 1.16 s | 51,953,664 B | 45 packages installed |
| 1 | `uv run --locked python -m axbench_tr export-mlx --out "$WORK/export"` | 15.76 s | 360,464,384 B | rows: full/train 5,873, full/valid 1,396, smoke/train 400, smoke/valid 40, peak_probe 20 and 20 |
| 2 | `$RUN verify-inputs` | 0.06 s | 33,210,368 B | `{"problems": []}` |
| 3 | `$RUN verify-model` (hashing the ten pinned files only) | 1.61 s | 33,226,752 B | `{"problems": []}` |
| 4 | `$RUN check-data` | 7.52 s | 320,028,672 B | every file `"problems": []` |
| 5 | `$RUN check-iterator ... --phase probe` | 2.12 s | 278,773,760 B | rows 20, `problem_count` 0 |
| 6 | `$RUN check-iterator ... --phase smoke` | 1.51 s | 282,525,696 B | rows 400, `problem_count` 0 (stdout kept in the log) |
| 7 | `$RUN preflight --phase probe ... --expect-commit d5f6cd56c0d3a12ecb9e917570d028788809ac78` | 1.53 s | 33,472,512 B | `"problems": []` |

`check-data` supervised tokens: full/train 5,873 rows 168,102 (21 to 184); full/valid 1,396 rows 39,940 (21 to 104);
peak_probe/train and /valid 20 rows 798 (22 to 184); **smoke/train 400 rows 10,530 (21 to 76)** (10,534 in the first run);
smoke/valid 40 rows 1,069 (21 to 38).

`check-iterator` mask samples (microstep: exact mask / supervised answer / stock `default_loss` mask), first eight microsteps:
- probe (unchanged from the first run): 36/36/37; 33/33/34; 38/38/39; 29/29/30; 36/36/37; 33/33/34; 33/33/34; 23/23/24
- smoke (new rows): 37/37/38; 28/28/29; 30/30/31; 27/27/28; 22/22/23; 29/29/30; 40/40/41; 27/27/28

The stock mask is one token higher than the answer in every sample, as before. `preflight` record: commit
`d5f6cd56c0d3a12ecb9e917570d028788809ac78`, phase probe, python 3.12, mlx 0.32.4, mlx-lm 0.32.2, offline true,
`export_manifest_sha256` `f99d236d...033b92`, `per_example_sha256`
`8dde1cfb9b20c2e9ce095cd0161e860a406c6b8ffad1f24e75d1bbfba175c95d`, model revision as above.

## What this does and does not show
It shows that the new pinned export, the ten-file checkpoint inventory, the harness dataset rendering, the real mlx-lm iterator
order, the padded widths and the exact loss mask pass on the M5's MLX build for the 20 probe rows and the 400 regenerated smoke
rows, and that the preflight accepts the immutable commit. It does **not** show GPU memory use, LoRA and optimizer behaviour,
throughput or any training outcome: no model was loaded.

## Open items
- The first run's `PREFLIGHT_NOWEIGHT.md` and `SUBSET_SOURCES.json` describe the superseded subset (23 rows on schemas from
  sources with unverified terms); `SUBSET_SOURCES_d5f6cd5.json` records the current subset and the delta.
- Unreviewed AI-authored rows; resource measurement only.
