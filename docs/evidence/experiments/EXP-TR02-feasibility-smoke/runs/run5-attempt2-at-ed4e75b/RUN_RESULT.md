# EXP-TR02 sequence at ed4e75b (protocol revision 5): P1, P2 and P3 all completed

Status: **measured result, evidence only; discard-only feasibility run.** Execution commit (code):
**`ed4e75b7668c5541c0912a066b57cce88c938713`** (clean detached checkout, status empty before and after every phase).
Evidence checkout: the PR branch head at collection time (`be7dbfc`); the evidence head is not the execution commit.
This is the second attempt; the first (at `d5f6cd5`) stopped at the harness LoRA-name audit
(`../p1-probe-stopped-at-d5f6cd5/RUN_RESULT.md`) and its authorization is spent. This is not a resume of it.

## Authorization and conditions
- Owner decision "Yes, run it once" on the decision card at 2026-11-27T21:13:16Z for exactly one probe, smoke and reload sequence at the
  commit above; technical lead conditions in the PR comment 7228236693 (as relayed to the device session, which cannot read the thread).
  Recorded in `AUTHORIZATION.md` before any model load (21:14:39Z).
- First launch attempt (21:16:32Z): the local permission control denied `supervise --phase probe`. **Nothing ran** (no probe
  directory, no process, no weight or COMPLETE/STOP file; confirmed again before the resume and logged in `00b_recheck.txt`).
  After the owner's later resume message (2026-11-28T00:25:07Z) the prerequisites were rechecked (logs 51 to 57), and the identical
  commands were issued once each; the owner approved interactively and none was denied. No retry, resume, sweep, checkpoint
  selection, generation, accuracy scoring, sealed-test access, contamination command, code or configuration change.
- Prelaunch (all exit 0, `problems: []`): sync, export (six hashes equal the pins), verify-inputs, verify-model, check-data,
  check-iterator probe and smoke, check-lora, preflight; AC power, memory pressure 1, swap 2.44 MB used, no other model process,
  `config.json` sha256 `7d31d916621523281d2226de9802671ddc63b05aad219388064a275abaf5301c` (equal to revision 4).

## Outcomes (every number is read from the committed logs)
| Phase | Start (UTC) | Wall | Exit | COMPLETE | Peak MLX | `time -l` max RSS | Swap delta |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| P1 probe | 2026-11-28 00:26:44 | 41.93 s | 0 | yes | 6.1958 GB | 1,501,544,448 B | 0 |
| P2 smoke | 00:28:05 | 147.72 s | 0 | yes | 6.0269 GB | 2,116,042,752 B | 0 |
| P3 reload | 00:31:07 | 16.66 s | 0 | yes | 4.1878 GB | 3,021,832,192 B | 0 |

Stop rules (frozen, unchanged): none fired. MLX peak limit 10 GB, RSS limit 11 GB, swap growth limit 2 GB, critical memory pressure,
AC loss, throughput floor and the event/optimizer audit all held.

- **P1 probe** (20 longest eligible rows, 20 microsteps, 5 optimizer updates, diagnostic loss only): LoRA audit passed with the
  `model.layers.N...` names (32 tensors, 458,752 parameters); training 14.93 s; per-report iterations/s 1.296, 1.375, 1.291, 1.391,
  1.366; validation before 9.08 s (loss 1.7688), after 9.03 s (loss 1.2109); adapter sha256 `34f77bddf94eb013a3bfa007d22338ebd9f2b67abd6b4531633310a5d3d570a5`.
  The probe contains the longest training row of the whole corpus (1,342 full tokens), so its peak covers the longest sequence at micro-batch 1.
- **P2 smoke** (fresh adapter and optimizer, 400 rows once, 400 microsteps, 100 optimizer updates, 40 validation rows before and after):
  training 124.145 s; throughput per 10-update window, full tokens/s: 1678.6, 1668.4, 1679.1, 1692.3, 1679.7, 1669.4, 1669.7,
  1651.4, 1643.3, 1642.1 (mean 1667.4); padded tokens/s: 1734.7, 1717.9, 1733.7, 1734.4, 1725.6, 1725.9, 1722.6, 1693.3, 1688.0, 1696.4.
  Validation: before 7.64 s (loss 2.5601), after 8.24 s (loss 0.1904), 15.88 s together = 12.8% of the training time.
  Training loss 2.452 at the first report and 0.134 at the last. Adapter: 32 tensors, finite, all changed, `lora_b` non-zero,
  1,838,449 bytes, sha256 `e3777eede1a715a2bf336cf087f268ab999fbb01e48747341c93acb58affed7d`, config sha256 `f3aaa07b59a559189cbcd2af64f90ade46560fc6787eb21a63c9e54282ce3541`.
- **P3 reload** (fresh process, adapter and config hashes verified equal to the smoke's, 40 validation rows): reload loss 0.19035693
  against the smoke's 0.19035695, absolute difference 3e-8, tolerance 1e-4: **pass**. Validation 7.78 s; model load 0.62 s.

## Thermal and independent OS samples
A separate 5-second sampler (not the harness) logged memory pressure, swap, AC state, python/mlx RSS and the `pmset -g therm` lines
(`run-logs/os_samples_*.log`). Probe: 12 samples, pressure 1 in 9 and 2 (warning) in 3 samples at 00:27:00 to 00:27:10Z, never 4;
smoke: 34 samples, all pressure 1; reload: 7 samples, all pressure 1. AC power and swap 2.44 MB in every sample; no thermal,
performance or CPU power warning was ever recorded. Throughput degradation over the 10 smoke windows: last vs first window -2.2%
(mean of the last three vs the first three -1.8%). A 2.5 minute run is too short to show a thermal soak.

## Estimates for training on the whole corpus (extrapolation, not measured)
Corpus: 5,873 train rows (3,018,924 full tokens), 1,396 validation rows (726,494 tokens). At the measured 1,642 to 1,692 full tokens/s
(probe on long rows: 1,625), one epoch is about 1,790 to 1,860 s (30 to 31 min). One full validation pass at about 0.19 to 0.21 s per
row is about 4.5 to 4.9 min. One epoch with one validation pass is therefore about 35 min, three epochs with a pass each about 1.8 h.
Peak MLX memory was 6.2 GB with a 10 GB limit at micro-batch 1.

## Limits
One machine (MacBook Air M5, 16 GB), one run, one 400-row subset of project-authored synthetic-template rows on documented schemas,
LoRA rank 8, learning rate 1e-5, gradient accumulation 4. Sustained-run thermal behaviour and any quality of the adapter are not
measured; the validation loss values are teacher-forced diagnostics on 40 (or 20) rows and are no quality or routing claim.
Adapters are experimental, discard-only and never copied (only hashed); they are not product candidates or an initialization.
Broader upstream rights, human-review and independent held-out gates remain unresolved for product training.

## Evidence
`sanitized/` holds 50 files: the collected raw records of the three phases (events, supervisor and child logs, COMPLETE.json, adapter
configuration, `EVIDENCE.json` with the hashes of the adapters that were not copied) and `run-logs/` (prelaunch and recheck logs,
the three supervise logs, OS samples, `AUTHORIZATION.md`, `phase_marks.txt`). They were sanitized on the device with
`sanitize-evidence` (home prefix to `$HOME`, other local paths to `<local-path>`, named terms to placeholders); `REDACTIONS.json`
lists the original and sanitized SHA-256 and sizes of every file (sha256 of the manifest itself:
`7c26e9132eaa2a4ffbd079ec518491a92ad3e40b0d1482e6495491ae68937d80`). Scans of the sanitized set for the home path, account, owner and
device names, e-mail-like strings and credential-like tokens found 0. The raw originals stay on the device. No weight file exists in this set.
