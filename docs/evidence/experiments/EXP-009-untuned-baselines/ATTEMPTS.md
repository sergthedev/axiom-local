# EXP-009 attempts

All model processes run at the reviewed commit `995ae3f092aab9ac1de5b063f2d05bf1742e9212` (PRERUN revision 1). No attempt
was chosen by its scores. Each attempt keeps its own directory, unchanged.

| Attempt | Directory | Model | Status |
|---|---|---|---|
| 1 | `runs/qwen3-0.6b/` | Qwen3-0.6B | **complete**: main (5 warm-ups, 160 validation, 600 repeats) and fresh processes 1-5 all `completed`, 0 request errors, no stop rule |
| 1 | `runs/qwen3-1.7b/` | Qwen3-1.7B | **incomplete (externally interrupted)**: kept unchanged as evidence and **not used for selection** |
| 2 | `runs_attempt2_qwen3-1.7b/qwen3-1.7b/` | Qwen3-1.7B | **complete**: one full rerun from scratch under D19, approved in [PR #10 comment 7189459615](https://github.com/sergthedev/axiom-local/pull/10#issuecomment-7189459615). It ran 10:32:10-11:22:03 UTC, detached, with exit status 0. Main (5 warm-ups, 160 validation, 600 repeats) and fresh processes 1-5 all `completed`, no stop rule. This is the accepted 1.7B attempt. |

**Accepted attempts for the analysis:** Qwen3-0.6B attempt 1 and Qwen3-1.7B attempt 2. They were fixed by completeness
before any score was computed. Attempt-1 files re-checked against `logs/attempt1_files.sha256` after attempt 2: all
unchanged.

## Interruption of attempt 1 (Qwen3-1.7B), recorded before attempt 2 started

- Sequence log: [logs/attempt1_sequence.log](logs/attempt1_sequence.log). The 0.6B processes and the 1.7B main started at
  the logged times. The log ends after `qwen3-1.7b main` with no exit line from that process.
- Hashes of every attempt-1 file as left: [logs/attempt1_files.sha256](logs/attempt1_files.sha256). They were taken
  before attempt 2 and the files have not been edited since.
- State of the 1.7B main process files: 5 warm-up records, 160 validation records and 598 of the 600 expected repeat
  records (the last is repeat 20, `validation-0117`). Every record has status `ok`. The measurement file was last
  written at the end of the validation phase (09:15:54 UTC) and still says `status: running`. No stop reason and no
  failure are recorded, the slowdown check had not fired, and the last repeat record was written at 09:50:42 UTC.
- No experiment process existed when this was checked at 10:28 UTC. The Mac had last booted at 08:37 UTC, so it did
  not restart during the run. The swap and pressure readings in the measurement file stay at 0 bytes and level 1.
- Most likely cause: the background shell that ran the sequence belonged to a Claude Code session that ended. When
  the session ended, the job was torn down; the tool reported that it "didn't finish before the previous session
  ended". This is not independently established. No experiment stop condition fired.
- Attempt 2 therefore runs detached from any Claude Code session. A new session (`setsid`) ignores hang-ups
  (`nohup`), and `caffeinate -i` prevents idle sleep, which changes no experimental setting. The script
  [logs/attempt2_sequence.sh](logs/attempt2_sequence.sh) writes `logs/attempt2_sequence.log` and
  `logs/attempt2_exit_status.txt`. If attempt 2 fails it is kept and returned for review, never retried
  automatically.
