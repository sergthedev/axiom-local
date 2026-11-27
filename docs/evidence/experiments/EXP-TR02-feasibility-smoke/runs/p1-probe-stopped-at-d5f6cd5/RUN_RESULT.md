# EXP-TR02 sequence at d5f6cd5: P1 probe stopped by the harness, P2 and P3 not run

Status: **measured failure, evidence only.** No optimizer update, no validation, no adapter and no `COMPLETE.json` exist.
Execution commit (code): **`d5f6cd56c0d3a12ecb9e917570d028788809ac78`**; the evidence head is not the execution commit.

## What was authorized and what ran
- Authorization: the owner chose "Yes, run it once" on a decision card at 2026-11-27T19:08:24Z for exactly one probe, smoke and
  reload sequence at the code commit above (recorded in `AUTHORIZATION.md` before any model load; the device session could not
  read the thread, so it recorded the decision as relayed).
- First launch attempt (19:09 to 19:25): the machine's local permission control denied the first `supervise --phase probe`
  command. Nothing ran. It was not bypassed. The owner then asked for interactive approval of the original command.
- P1 ran **once**, unchanged, after that approval: `supervise --phase probe` with `--expect-commit d5f6cd56...`, offline flags set,
  from a clean detached worktree at that commit, AC power attached, memory pressure 1, swap used 2.44 MB (baseline), no other MLX,
  Python or EXP-010 process. Start 2026-11-27 19:26:02 UTC, end 19:26:10 UTC.
- **P2 and P3 were not started** (no valid P1 `COMPLETE.json`). No retry, resume, sweep, checkpoint selection, generation or
  accuracy scoring; no code or configuration change; no sealed-test access.

## Outcome (observed in the committed logs)
- Supervisor result: `stopped: true`, `reason: lora_shapes_not_ok`, `child_exit_code: 1`, `events_accepted: 2` (`STOP.json`).
- Events (`probe__events.jsonl`): `start` (preflight record equal to the earlier preflight: export manifest `f99d236d...`, model
  revision `70d244cc...`, mlx 0.32.4, mlx-lm 0.32.2, offline true), `load` (MLX peak 3.44114996 GB, 0.682 s: the weights **were**
  loaded), `lora` (32 tensors, 458,752 trainable parameters, `shapes_ok: false`), `error`
  (`RuntimeError: LoRA tensors differ from the expected shapes: missing tensor: layers.20.self_attn.q_proj.lora_a; ...`).
- No iteration, validation, optimizer update, adapter or `COMPLETE.json`. No memory, RSS, swap, pressure or AC stop rule fired;
  the stop was the harness's own LoRA-name audit.
- Resources of the whole process (`/usr/bin/time -l` in `10_supervise_probe.log`): 6.65 s wall, max resident set 2,063,335,424 bytes,
  peak memory footprint 10,027,536 bytes, no swaps; swap used and memory pressure unchanged after the run (from the device report).
- **Throughput, tokens per second, validation loss, validation overhead, reload difference and adapter hashes: not measured**
  (nothing past model load and LoRA attachment ran).

## Cause (analysis, not part of the device evidence)
The audit compares the names from `tree_flatten(model.trainable_parameters())` with names built as `layers.{i}.{key}.lora_a/b`
(`expected_lora_shapes`, `training/tr02/tr02.py`). In mlx-lm 0.32.2 the Qwen3 `Model` holds its blocks under `self.model`, so the
flattened names carry a `model.` prefix (`model.layers.N.self_attn.q_proj.lora_a`). A tiny randomly initialised Qwen3 model built on
CPU with the same `linear_to_lora_layers` call and keys reproduces the prefixed names. Every expected name therefore looks
missing. The count (32 tensors, 458,752 parameters) equals the expectation. The Linux unit tests did not exercise the real
mlx-lm model object, and the event log keeps only the first five problems and not the observed names, so the observed names on the
M5 were not logged; the explanation rests on the source and on the CPU reproduction. The saved-adapter checks later in the code
use the same comparison and probably need the same review. **A fix needs a code change, a new commit, a new review and a new
authorization; none was made.**

## Evidence in `sanitized-logs/`
Sanitized on the device (home directory prefix to `$HOME`, other absolute local paths to `<local-path>`, the owner's first name
to `<owner>`; counts are in `REDACTIONS.json`, SHA-256 `18f12881dbfd305cd9ae71331f8818f5b55c24700fccc4c5ca033dd7075e8ae5`).
Each file was received here as base64 text, decoded, and matched by SHA-256 and byte count against `REDACTIONS.json`; all matched.
`.gitattributes` marks this directory `-text`.

| File | Sanitized SHA-256 | Bytes |
| --- | --- | ---: |
| `probe.supervisor__STOP.json` | `9a115f54f8451ebdd86b2545032ac5a51a9fa04d594c06eb70acff49ade5a3c4` | 100 |
| `probe.supervisor__supervisor.jsonl` | `dd59cbbd3fb12bbd8788241395c0ee9966d7c8955fc199b44eb6631ce94de896` | 1365 |
| `probe__events.jsonl` | `86256cf5a92a185c50d3a351d141485d3d19400fbc736b3849aa78dc74edab8e` | 1219 |
| `probe.supervisor__child.log` | `34aa85454b6ead7b49d7269b9099488070b33ce656462291712376bab5f3492e` | 549 |
| `10_supervise_probe.log` | `a431f25ec97a67a86ee603319f987bf12d9eb5286145b9876daac06eb73ab098` | 1190 |
| `10_supervise_probe.stdout` | `6700a1bd3676fb33e70dcd38941b82b82dd4f37f476afed65d9df0d35d788294` | 892 |
| `20_collect_evidence.log` | `ec641a287ce634290007c57472c12ad713c8eabab3a73fd52f2cfa3675aacde6` | 1043 |
| `AUTHORIZATION.md` | `d8a6fece29ea33dcbbe687561d18d1f295980aeaa2c3417d660a33d8885241c6` | 919 |
| `phase_marks.txt` | `322d0c5f175c3732c8eb17a86802c684536dff61309add81a6800d4da755c72c` | 80 |
| `REDACTIONS.json` | `18f12881dbfd305cd9ae71331f8818f5b55c24700fccc4c5ca033dd7075e8ae5` | 10588 |

Not committed, and why:
- The pre-launch logs of this run (`00_environment.txt`, `00b_uv_sync.log`, `01` to `07`): all steps exited 0 with no problems and
  the export hashes reproduced the pins, according to the device session's report. Their bytes were not verified here (the
  transport of this group was stopped), so they are not committed; their original and sanitized hashes are in `REDACTIONS.json`
  and the originals stay on the device. `phase_marks.txt` marks the P1 launch at 19:26:02 and its end at 19:26:10 with rc=1.
- `collect-evidence` failed once ("the evidence directory must be inside the repository", exit 1; `20_collect_evidence.log`),
  because the README command only accepts a destination inside the repository and the execution worktree had to stay clean.
  The raw outputs were instead copied byte-identically on the device (verified with `cmp`) and sanitized there. The harness
  `collect-evidence` index (SHA-256 index of the raw run files) was therefore **not** produced; the original `probe/` and
  `probe.supervisor/` directories remain on the device untouched.
- Raw originals (with private paths) stay on the device; no weights exist or were copied.
- One protocol deviation: the first chained pre-launch call stopped after step 5 because of a shell error in the device
  session's wrapper; steps 6 and 7 ran once each in a second call. No step ran twice.

## Limits
One machine, one run. This is a failure of the harness's own audit before training, not a measurement of the model or of memory
use under training. It does not show that the GPU training path, memory or throughput work or do not work.
