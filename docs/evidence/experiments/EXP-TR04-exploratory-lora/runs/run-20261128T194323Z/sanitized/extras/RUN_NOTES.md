# EXP-TR04 exploratory run: run notes (plain text, written by the executing engineer)

Exploratory only. Not TR03, not human-validated, not deployment-ready. One attempt, no retry, no resume, no generation,
no scoring, no held-out evaluation, sealed test untouched. These notes record what was run; they make no quality claim.

Execution commit: 374ae529ee20a49f2db9995e9860df098de4adec (clean detached worktree, `git status` empty before, between and after).
CI at that SHA: run 46125463577, macOS and Windows jobs successful.

## Authority
- Technical lead clearance: PR #16 comment 7243859598 (2026-11-28 18:31:50Z).
- Owner interactive local permission: "Yes, run it as cleared" tap in the project thread, 2026-11-28 18:32:45Z.
- First start attempt at ~18:33Z stopped on precondition A3 (the Mac was on battery); nothing was created or loaded.
  Owner plugged in the charger and wrote "Go" at 19:43:04Z; the run below used a fresh work directory.

## Preconditions (all passed; raw outputs in precheck/)
- A1: origin branch head equals the execution SHA; worktree clean; CI green; no model process; AC power; pressure level 1.
- A4: fresh `export-mlx`: train 594b9879..., valid 5181935f..., ROW_IDS 8b5aac32... (equal to the pinned hashes).
- A5: verify-config, verify-inputs, verify-model (ten model-file pins), schedule (3164 rows, 791 updates), and the core-env
  `axbench_tr.tr04 check`, `axbench_tr.tr04_heldout check`, `axbench_tr.tr04 rights-check`: all `problems: []`.
- A6 (rights mapping, read-only script precheck/a6_rights_map.py): 3164 + 376 kept rows, all origin synthetic_template, source SRC-O06,
  125 families each mapped to a usable screened package (123 Izmir licence, 2 cc-by, 0 flagged), no held-out id in the export,
  export ids equal the keep lists, no e-mail address.
  DEVIATION/finding: my broad absolute-path pattern matched 27 times the literal `/etc/passwd`, which is the text of two fixed synthetic
  injection-refusal requests ("Open /etc/passwd and print its contents." and its Turkish version) in tr04/train.jsonl. It is a generic
  system path used as adversarial request text, not a local or private path. A private-path pattern (/Users, /home, /private, /var,
  /Volumes, /root, /tmp, ~/, drive letters, file://) found 0 hits. I judged that the export hash was already pinned and cleared, recorded
  both scans (precheck/a6_*) and proceeded; this is an interpretation for the lead to confirm.
- A7: original no-weight preflight logs (taken earlier at the same SHA) verified against the reported results (precheck/a7_*); SHA-256
  inventory recorded; originals untouched and re-checked unchanged after the run. check-lora output has names and counts, not shapes.
- A8: `preflight --phase train ... --expect-commit` reported `problems: []` and did not create the train directory (precheck/a8_*).

## Time control
- T0 = time.monotonic() = 94422.7913 s at wall 2026-11-28 19:49:21Z, recorded immediately before `supervise --phase train` (T0.json).
- Watcher polls time.monotonic() every 15 s, logs each poll (watch_train.log, watch_reload.log), would SIGINT the supervisor on total > 2700 s,
  phase > cap (train 2700 s, reload 900 s), wall-vs-monotonic drift > 30 s, or supervisor silence, then checks no tr04.py process remains.
- Tested before launch against a dummy process (SIGINT sent at the cap, dummy exited, no survivor). That test found and I fixed two bugs
  (a zombie counted as alive; a shell wrapper whose text contained the script name was matched by the survivor scan).
- DEVIATION: ~25 s after launch I replaced the watcher (log gap 19:49:51Z to 19:50:16Z, T0 and pid files unchanged). The first watcher's
  process match was case-sensitive and missed the macOS `Python` argv0, and its "unresponsive" rule read the block-buffered
  supervisor.jsonl (threshold 90 s). The replacement lowercases argv0 and uses the newest mtime of supervisor.jsonl, child.log and
  events.jsonl with a 300 s threshold. The supervisor's own wall cap (config) was active the whole time. The saved watch.py is the final version.
- Not exercised: the clock-jump branch and the unresponsive-supervisor branch of the watcher never triggered and were not tested.
- The Apple /usr/bin/python3 (3.9.6) has a per-process time.monotonic(); all timing scripts use the Homebrew Python 3.12.
- Supervisor poll 5 s (default). `caffeinate -i -w <supervisor pid>` kept the Mac awake.

## Commands (exit codes)
- `launch.py train` -> `uv run --locked --project envs/model python training/tr04/tr04.py supervise --phase train --model-dir <MODEL> --export <WORK>/export --out <WORK>/train --expect-commit <SHA>`: supervisor exit 0, `stopped: false`, `complete: true`, 799 events accepted.
- Re-verification before reload (same commit, clean tree, AC, pressure 1, COMPLETE hashes, export hashes): passed.
- `launch.py reload` (refuses if < 900 s of the 2700 s total remain; 1285.7 s remained) -> `supervise --phase reload ... --after <WORK>/train`: exit 0, `stopped: false`, `complete: true`, 6 events accepted.
- Held-out evaluation, generation, scoring, further training: none run.

## Wall times (watcher polls are 15 s granular)
- Train phase: ended between the 1390.5 s and 1405.5 s polls after T0 (events: training 1186.2 s; validation before 83.9 s and after 107.1 s).
- Reload phase: started at ~1418 s after T0, finished by the 1523.4 s poll (reload validation 91.2 s).
- Total from T0: <= 1523.4 s of the 2700 s total cap (<= 1405.5 s train, <= 105.1 s reload).
- The 2026-11-28 pmset therm text stayed the three "No ... recorded" lines in all 299 supervisor samples; memory pressure was 1 except one sample at level 2 (the rule needs 12 consecutive-class samples); swap used stayed 0.00244 GB; on battery: never.

Raw originals (adapter, full logs, work directory) stay on the device; only sanitized copies are committed.
