# EXP-TR02 no-weight preflight on the MacBook Air M5 (evidence record)

> **Superseded for the subset.** This is the first preflight (head 96cbdf7). Its probe and smoke subset was replaced after
> the licensing review (23 smoke rows on ECB, OECD and TCMB schemas); see `PREFLIGHT_NOWEIGHT_d5f6cd5.md`. The sanitized raw logs of
> this run are now committed in `at-96cbdf7-sanitized-logs/` and the numbers below were checked against them (`LOG_VERIFICATION.md`):
> no numeric discrepancy was found.

Status: **run once, all seven steps passed.** This is the technical lead's cleared no-weight preflight for PR #13 at the
immutable head `96cbdf7edf30e26bf535a7eb0a84bb6467302ffd`. **No model weights were loaded, `supervise` and `run` were not
executed, nothing was trained or generated, the sealed test was not touched, and no code was changed.** The full smoke is
**not** cleared: it needs acceptance of this preflight and a separate owner authorization at a named head.

Provenance of this record: the commands were run on the owner's M5 by a Remote Control session started from this thread;
the numbers below are transcribed from that session's report (it kept one log per command with the command, output, the
full `time -l` block and the exit status on the device; absolute local paths are left out of this public text). The raw logs
are not in the repository. Everything below is measured on that one machine on 2026-11-27.

## Device and safety checks (before any step)
| Item | Value |
| --- | --- |
| Power, pressure, other work | AC power (95%); `kern.memorystatus_vm_pressure_level` 1; no MLX, EXP-010 or other Python process running |
| Machine | Apple M5, `hw.model` Mac17,3, 17,179,869,184 bytes memory, macOS 27.0.1 (26A434) |
| Tools | uv 0.13.10; Python 3.12 from the locked model environment; mlx 0.32.4; mlx-lm 0.32.2 |
| MLX default device | `Device(gpu, 0)` (only `mlx.core` was imported for this check) |
| Worktree | new isolated worktree outside the existing checkout; `HEAD` exactly `96cbdf7edf30e26bf535a7eb0a84bb6467302ffd`; `git status --porcelain --untracked-files=all` empty before the environment sync, after it, and at the end (`.venv` directories are ignored) |
| Checkpoint | the already provisioned Qwen/Qwen3-1.7B directory, `PROVENANCE.json` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; no download |
| Environment | `uv sync --locked --project envs/model` succeeded; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` for every step |
| Memory pressure | 1 before and after both `check-iterator` steps, and throughout |

## Steps (exact order; each exit status 0; `RUN="uv run --locked --project envs/model python training/tr02/tr02.py"`)
| # | Command | Wall | Max RSS | Result |
| --- | --- | ---: | ---: | --- |
| 1 | `uv run --locked python -m axbench_tr export-mlx --out "$WORK/export"` | 16.14 s | 361,824,256 B | rows: full/train 5,873, full/valid 1,396, smoke/train 400, smoke/valid 40, peak_probe/train 20, peak_probe/valid 20 |
| 2 | `$RUN verify-inputs --export "$WORK/export"` | 0.13 s | 33,062,912 B | `{"problems": []}` |
| 3 | `$RUN verify-model --model-dir "$MODEL"` (hashing the ten pinned files only) | 1.62 s | 33,210,368 B | `{"problems": []}` |
| 4 | `$RUN check-data --export "$WORK/export" --tokenizer-dir "$MODEL"` | 7.50 s | 320,405,504 B | every file `"problems": []` (below) |
| 5 | `$RUN check-iterator ... --phase probe` | 2.62 s | 278,773,760 B | rows 20, `problem_count` 0 |
| 6 | `$RUN check-iterator ... --phase smoke` | 1.53 s | 282,738,688 B | rows 400, `problem_count` 0 |
| 7 | `$RUN preflight --phase probe --model-dir "$MODEL" --export "$WORK/export" --out "$WORK/probe" --expect-commit 96cbdf7edf30e26bf535a7eb0a84bb6467302ffd` | 1.61 s | 35,618,816 B | `"problems": []`; the output directory was not created |

`check-data` (supervised tokens = the compact answer plus `<|im_end|>`): full/train 5,873 rows, 168,102 tokens (21 to 184);
full/valid 1,396 rows, 39,940 (21 to 104); peak_probe/train and /valid 20 rows, 798 (22 to 184); smoke/train 400 rows, 10,534
(21 to 76); smoke/valid 40 rows, 1,069 (21 to 38). A "PyTorch was not found" notice from `transformers` is expected: only the
tokenizer is used.

`check-iterator` loss-mask samples (microstep: exact mask tokens / supervised answer tokens / stock `default_loss` mask tokens),
the first eight microsteps of each phase, on the M5's MLX build:

- probe: 1: 36/36/37; 2: 33/33/34; 3: 38/38/39; 4: 29/29/30; 5: 36/36/37; 6: 33/33/34; 7: 33/33/34; 8: 23/23/24
- smoke: 1: 30/30/31; 2: 28/28/29; 3: 27/27/28; 4: 36/36/37; 5: 22/22/23; 6: 22/22/23; 7: 40/40/41; 8: 27/27/28

The iterator order, token lengths and padded widths of all 20 probe and 400 smoke microsteps matched the declared schedule.
The stock mask is one token higher than the answer in every sample, as found on the CPU build, so that result holds on the GPU
build too. This is a property of the stock `default_loss` in mlx-lm 0.32.2 and of the exact-mask loss this harness uses; it is not
an analysis of EXP-010's own loss, which is unchanged and was not examined here.

`preflight` record: commit `96cbdf7edf30e26bf535a7eb0a84bb6467302ffd`, phase probe, python 3.12, mlx 0.32.4, mlx-lm 0.32.2,
offline true, `export_manifest_sha256` `63fa9306a89ac113afe93024f7ad7e02fa1964629204078b5753b444621532c1`, `per_example_sha256`
`8dde1cfb9b20c2e9ce095cd0161e860a406c6b8ffad1f24e75d1bbfba175c95d`, model revision as above.

## Hashes
- `EXPORT_MANIFEST.json` `63fa9306a89ac113afe93024f7ad7e02fa1964629204078b5753b444621532c1`
- `ROW_IDS.json` `1f8538b687c28002244aaac4c18f9efe2d32e1ad586769b068a11c869728747e`

Both equal the hashes of an export built on the Linux container and pinned in `training/tr02/config.json`, so the export is
byte-identical across the two machines.

## Source rights of the probe and smoke rows (`SUBSET_SOURCES.json`)
Requested by the technical review: no unresolved upstream-derived schema in the smoke subset, with the subset IDs and the
source and licence evidence recorded.
- Every selected row (20 probe, 400 smoke train, 40 smoke validation) is `synthetic_template` with `human_review: none`. **No
  row uses a schema from SRC-N01** (the Cognipeer/Gretel synthetic `CREATE TABLE` definitions), and no row carries upstream
  request text.
- Schemas are real-table header layouts (names and kinds only; no observations) from SRC-O01 (World Bank, CC BY 4.0 by the
  catalog's default), SRC-O02 (Eurostat reuse policy), SRC-O06 (Izmir portal, CC BY 4.0). Counts: probe O06 20; smoke train O06 241,
  O01 127, O02 10, O05 10, O03 8, O04 4; smoke validation O01 27, O06 11, O02 1, O04 1.
- **Open point for the reviewer:** SRC-O03 (ECB), SRC-O04 (OECD) and SRC-O05 (TCMB) are registered `accepted_schema_only` with
  terms **not verified** in this environment. The smoke subset includes 22 train rows and 1 validation row on those header
  layouts (row IDs in `SUBSET_SOURCES.json`). Column names of public statistical series are low-risk and no rows or values
  are used, but the licence reading is unverified. If that is not acceptable, the smoke subset can be restricted to the
  verified sources (O01, O02, O06) by a code change that needs a new head and a new preflight; nothing was changed here.

## Not done
No weights were loaded and no GPU memory or timing for training was measured; `supervise`, `run`, the probe, the smoke and
the reload were not executed. The M5's GPU path for loading, LoRA and the optimizer is therefore still unmeasured.
