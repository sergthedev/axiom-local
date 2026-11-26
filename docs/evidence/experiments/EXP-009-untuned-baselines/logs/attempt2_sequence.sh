#!/bin/zsh
# EXP-009 Qwen3-1.7B attempt 2 (D19, approved in PR #10 comment 7189459615): one complete rerun from
# scratch at the reviewed commit, in its own directory; stops at the first non-zero exit and never
# retries. Started detached (new session, SIGHUP ignored, idle sleep prevented) so that it survives
# the end of the Claude Code session. Writes the sequence log and the final exit status here.
HERE=${0:A:h}
REPO=${HERE:h:h:h:h:h}
cd "$REPO/envs/model" || exit 1
C=995ae3f092aab9ac1de5b063f2d05bf1742e9212
OUT=../../docs/evidence/experiments/EXP-009-untuned-baselines/runs_attempt2_qwen3-1.7b
m=qwen3-1.7b
rc=0
echo "== $(date -u +%FT%TZ) start pid $$ head $(git rev-parse HEAD)"
echo "== $(date -u +%FT%TZ) $m main"
uv run --locked python exp009_run.py main --model $m --model-dir ~/axiom-models/$m --out $OUT --expected-commit $C || rc=$?
if [[ $rc -eq 0 ]]; then
  for i in 1 2 3 4 5; do
    echo "== $(date -u +%FT%TZ) $m fresh $i"
    uv run --locked python exp009_run.py fresh --model $m --model-dir ~/axiom-models/$m --out $OUT --expected-commit $C --index $i || { rc=$?; echo "STOPPED at fresh $i rc=$rc"; break; }
  done
else
  echo "STOPPED at main rc=$rc"
fi
echo "== $(date -u +%FT%TZ) end rc=$rc"
echo "$rc $(date -u +%FT%TZ)" > "$HERE/attempt2_exit_status.txt"
exit $rc
