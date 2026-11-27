# EXP-TR02 revision-5 no-weight preflight on the MacBook Air M5, at head ed4e75b (evidence record)

Status: **run once, all ten steps exit 0.** Immutable code head `ed4e75b7668c5541c0912a066b57cce88c938713` (CI green on macOS and
Windows, run 46065804096). **No model weights were loaded** (the checkpoint was hashed and its tokenizer read, never loaded as a
model), `supervise`, `run`, `smoke` and `reload` were not executed, no `probe` output directory was created, nothing was trained
or generated, the sealed test was not touched and no code was changed. This evidence does **not** clear a feasibility run: that
needs technical review and a separate owner authorization at a named head.

Source of every number: the sanitized raw logs in `at-ed4e75b-sanitized-logs/` (byte-exact copies of the files sanitized on the
device; the originals stay on the device). Verification: `REDACTIONS.json` (SHA-256
`7b245b1f15a8286dd7a4d2868b75c9ee97c7d1399bfb36539227a8268b8a4630`, 8184 bytes) lists the original and sanitized SHA-256 and
byte count of each file; every file here was decoded from the base64 transport and matched against it (all 18 matched).

## Device and safety checks (`00_environment.txt`, before any step; `11_post_checks.txt` after the last)
| Item | Value |
| --- | --- |
| Power, pressure, other work | AC power (95%); memory pressure level 1 before and after; swap used 2.44 MB of 1024 MB, unchanged after; no MLX, Python or training process before or after |
| Machine | Apple M5, `hw.model` Mac17,3, 17,179,869,184 bytes, macOS 27.0.1 (26A434); uv 0.13.10 |
| Worktree | `HEAD` exactly `ed4e75b7668c5541c0912a066b57cce88c938713`, `git status --porcelain --untracked-files=all` empty before and after |
| `config.json` | SHA-256 `7d31d916621523281d2226de9802671ddc63b05aad219388064a275abaf5301c` at revision 5, at the revision-4 commit d5f6cd5 and as expected: byte-identical |
| Checkpoint | `PROVENANCE.json` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; hashed, never loaded |
| Environment | `uv sync --locked --project envs/model`; python 3.12.16, mlx 0.32.4 (`Device(gpu, 0)`), mlx-lm 0.32.2, import check only; offline flags set |
| Output directories | the probe, smoke and reload directories and their three supervisor directories did not exist before and do not exist after (`11_post_checks.txt`); the export was written outside the repository |

## Steps (each run once, each `EXIT_STATUS: 0`; `RUN="uv run --locked --project envs/model python training/tr02/tr02.py"`)
| # | Command | Wall | Max RSS | Result |
| --- | --- | ---: | ---: | --- |
| 1 | `uv sync --locked --project envs/model` | 0.96 s | 52,674,560 B | 45 packages installed |
| 2 | `uv run --locked python -m axbench_tr export-mlx --out <work>/export` | 20.23 s | 360,923,136 B | rows: full/train 5,873, full/valid 1,396, smoke/train 400, smoke/valid 40, peak_probe 20 and 20 |
| 3 | `$RUN verify-inputs` | 0.07 s | 34,242,560 B | `"problems": []` |
| 4 | `$RUN verify-model` (hashes the ten pinned files only) | 1.61 s | 34,144,256 B | `"problems": []` |
| 5 | `$RUN check-data` | 7.40 s | 320,487,424 B | every file `"problems": []` |
| 6 | `$RUN check-iterator ... --phase probe` | 1.96 s | 280,100,864 B | rows 20, `problem_count` 0 |
| 7 | `$RUN check-iterator ... --phase smoke` | 1.51 s | 283,639,808 B | rows 400, `problem_count` 0 |
| 8 | `$RUN check-lora` | 1.05 s | 117,800,960 B | `"problems": []`, 8 tensors, 1,792 trainable parameters, names `model.layers.2.self_attn.{q,v}_proj.lora_{a,b}` (tiny random fixture, MLX 0.32.4; the log prints the first four names) |
| 9 | `$RUN preflight --phase probe ... --expect-commit ed4e75b...` | 1.51 s | 34,242,560 B | `"problems": []`, commit equals the expected one, export manifest `f99d236d...033b92`, per-example `8dde1cfb...75c95d` |
| 10 | `uv run --locked --project envs/model --with pytest==9.2.0 python -m pytest tests/test_tr02_lora_tiny_qwen3.py -q -p no:cacheprovider` | 1.77 s | 125,550,592 B | **13 passed in 1.12 s** (real tiny random Qwen3, no pretrained weights) |

Export hashes and the `check-data` token counts are unchanged from the previous preflight at d5f6cd5 (the export and the
subsets did not change in revision 5): supervised tokens full/train 168,102, full/valid 39,940, peak_probe 798 and 798,
smoke/train 10,530, smoke/valid 1,069. The `check-iterator` mask samples are the same eight microsteps as before
(probe 36/33/38/29/36/33/33/23, smoke 37/28/30/27/22/29/40/27 supervised tokens; the stock mask is one higher in every sample).

## What this shows and does not show
It shows that the revision-5 harness at the immutable commit passes the pinned-input, checkpoint-inventory, data, iterator-mask
and preflight checks on the M5's MLX build, that `check-lora` and the 13 tiny-model regression tests pass **on the M5 with MLX 0.32.4**
(not only in the Linux CPU environment), and that the pinned configuration is unchanged. It does **not** show the observed LoRA names
on the real 1.7B checkpoint, memory peak, throughput, validation, optimizer behaviour or reload on that checkpoint: no model was
loaded. The only test of the real model's names is the next authorized probe, which is **not** authorized yet.

## Sanitizing and transport notes
- Sanitized on the device with `sanitize-evidence` (home directory prefix to `$HOME`, other absolute local paths to
  `<local-path>`, device and account names to `<device>`, `<owner>`, `<user>`, credential-like tokens to
  `<redacted-credential>`); redaction counts per file are in `REDACTIONS.json`. The home prefix was replaced 15 times and other
  absolute local paths 2 times (two paths in the uv logs); no named term and no credential-like token was found (all counts 0). An
  extra `--term` for the owner's first name never fired because the account-name term already matches case-insensitively; this is
  not a gap in the result, a leak scan of the committed files for the owner's name, e-mail address and device name found none.
- The bytes reached this repository as base64 text in messages. One file (`02_export.log`) first arrived with one word missing; the
  hash check caught it and the one-word fix is verified (the committed copy equals the device hash). `REDACTIONS.json` arrived in
  three parts after a duplicated fragment was caught in a first attempt. The committed files equal the device's sanitized files.
- `06_pressure.txt` to `10_pressure.txt` are byte-identical to `05_pressure.txt` (same values: pressure 1 before and after, swap
  unchanged); the device reported their hashes in `REDACTIONS.json` and sent one copy, which is reproduced for each name here. They
  are not independent time series.
- Step 10 used `--with pytest==9.2.0`, a tool added to the otherwise locked environment for this one step (5 packages, from
  the uv cache); the locked environment itself does not contain pytest.
- `.gitattributes` marks this directory `-text` (the committed bytes are the hashed bytes on every platform).

## Files and SHA-256 (sanitized; byte counts match `REDACTIONS.json`)
| File | SHA-256 | Bytes |
| --- | --- | ---: |
| `00_environment.txt` | `ed613a6d4ab661754ed5aa4c2525bdb21834cc2272083d34cb937617aeb9f321` | 1401 |
| `01_uv_sync.log` | `d52dc7dcb11a86e5dd7d5424448c4ed5ddca63d00ac6b547df7d3e8fb24b5ae2` | 2113 |
| `02_export.log` | `750578b3acd471a3ef904492d15ca3dc9976a0507949d10210bfefd554becb17` | 1207 |
| `03_verify_inputs.log` | `29b10d389890a2a3214b6473010bb79b0fd579ea55390f3ec61e4cdeae73653b` | 970 |
| `04_verify_model.log` | `fa8006b3ff65ebd7a443784193cf36973116537d61d203803cc37675904b6f30` | 974 |
| `05_check_data.log` | `6c0f74abeff90737b8b0e84e59dc725bccab011d2a4cb1f8a4bb74f76cee1c5f` | 1967 |
| `05_pressure.txt` | `32f351b1c00faba737579385a0fe975a7fb165498030f84d300dbe88e39cfcd9` | 184 |
| `06_check_iterator_probe.log` | `e27520f6d2bff93af8d972c57410e2b53c2fe6f01e94ec80a89fba33c5312ff4` | 1946 |
| `06_pressure.txt` | `32f351b1c00faba737579385a0fe975a7fb165498030f84d300dbe88e39cfcd9` | 184 |
| `07_check_iterator_smoke.log` | `b890a3ba7606ff5b686929fe915033db390697f9d941c7159656623e45a53877` | 1947 |
| `07_pressure.txt` | `32f351b1c00faba737579385a0fe975a7fb165498030f84d300dbe88e39cfcd9` | 184 |
| `08_check_lora.log` | `33f54dc53111f708b31c7b899e4ba8d7b279bebd42405a4cabbf6edc3f001b39` | 1181 |
| `08_pressure.txt` | `32f351b1c00faba737579385a0fe975a7fb165498030f84d300dbe88e39cfcd9` | 184 |
| `09_preflight_probe.log` | `c695b56fe106bb4f6a217b7062d00f097a40cf72cce23ce494b7e586e5c4194e` | 1590 |
| `09_pressure.txt` | `32f351b1c00faba737579385a0fe975a7fb165498030f84d300dbe88e39cfcd9` | 184 |
| `10_pressure.txt` | `32f351b1c00faba737579385a0fe975a7fb165498030f84d300dbe88e39cfcd9` | 184 |
| `10_pytest_tiny_qwen3.log` | `e9ee76d3f40757270a9fb2bd012a4fafdfb60953d3e98778adb937a736dcda53` | 1094 |
| `11_post_checks.txt` | `0683fa5a2f52daedd4dab4809fa669f693d05edf33ebe3386c11a466499b8560` | 457 |
| `REDACTIONS.json` | `7b245b1f15a8286dd7a4d2868b75c9ee97c7d1399bfb36539227a8268b8a4630` | 8184 |
