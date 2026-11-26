# Experiment manifest: EXP-009 untuned Qwen3 baselines on AXBENCH validation and base selection

- Issue / PR: #4 (Phase 3B), PR #10
- Date: 2026-11-26
- Status: **run** (one machine). Qwen3-0.6B ran once (attempt 1). Qwen3-1.7B attempt 1 was externally interrupted and
  is archived; attempt 2 is a complete rerun approved under D19 ([ATTEMPTS.md](ATTEMPTS.md)).
- Protocol: [PROTOCOL.md](PROTOCOL.md) revision 3. Pre-run record: [PRERUN.md](PRERUN.md) and its revision 1
  (operational definitions D1-D19). The record was committed in `e405efa` and its revision in `995ae3f`, both before
  any validation prediction existed.
- **Corpus status for every claim below:** synthetic, AI-generated, owner-authorized AI review, 0 human decisions.
  The results are exploratory. They are not human-validated, not representative of real requests, and not
  independent of the implementing tool.

## Provenance
- Model processes: commit `995ae3f092aab9ac1de5b063f2d05bf1742e9212` with a clean tree in all 12 accepted processes
  (6 per model). Runner `exp009.runner/2`.
- Analysis: commit `b2586dafb50eb7bbe395b126c6cc8145c3e4cadc` with a clean tree. Raw outputs were committed unscored
  in `6d4da72` before the analysis ran.
- Hardware and OS: MacBook Air (Mac17,3), Apple M5, 16 GB, macOS 27.0.1, arm64. Mains power in every process (the
  runner refuses battery power). The Mac was restarted by the owner before the runs, and the owner reports that
  unrelated applications were closed.
- Packages (`envs/model` lock): mlx 0.32.4, mlx-metal 0.32.4, mlx-lm 0.32.2, transformers 5.22.2,
  tokenizers 0.23.3, Python 3.12.16. Every process recorded these versions in its measurement file.
- Checkpoints: Qwen/Qwen3-0.6B @ `c1899de2…` and Qwen/Qwen3-1.7B @ `70d244cc…`, BF16 as published. They were
  provisioned and verified once ([provenance/](provenance/)) and re-hashed before every process (D6').
- Corpus: `axbench/0.1-ai-reviewed`, manifest `b8b6d39f…16239c`, validation 160 (`80457d95…`), loaded through the
  verified manifest. Train was used only for the five warm-ups. The final test was not generated, read or scored.
- Inference: offline, `axiom.prompt/1`, `enable_thinking=False`, greedy, ≤256 new tokens, stop set
  {151643, 151645} (asserted), batch 1, no constrained decoding, no repair.
- Scoring: harness `axbench.evaluate/0.2`, scorer `axbench.score/0.2`, Guard policy 2. Selection rule
  `exp009.base-selection/rev3`. Bootstrap: 2000 replicates, seed 2026, over 24 lineages.

## Execution
- Attempt 1 (sequence log [logs/attempt1_sequence.log](logs/attempt1_sequence.log)): `exp009_run.py main` and
  `fresh --index 1..5` for qwen3-0.6b from 08:49:38 to 09:07:30 UTC, all with exit code 0. Then qwen3-1.7b `main` from
  09:07:30 UTC, killed externally after 598 of 600 repeat records (last record at 09:50:42 UTC; no stop rule; no exit
  line). It is archived.
- Attempt 2 for qwen3-1.7b ([logs/attempt2_sequence.log](logs/attempt2_sequence.log),
  [logs/attempt2_sequence.sh](logs/attempt2_sequence.sh)): detached from the Claude Code session, 10:32:10 to
  11:22:03 UTC, `main` and `fresh 1..5` all with exit code 0 (`logs/attempt2_exit_status.txt`: `0`).
- Analysis:
  `uv run python benchmarks/issue4_exp009_analyze.py --runs runs --rules-predictions rules_validation.jsonl --attempt qwen3-1.7b=runs_attempt2_qwen3-1.7b/qwen3-1.7b --archived qwen3-1.7b=runs/qwen3-1.7b > results.json`
  (paths relative to this directory). Exit code 0.
- Outputs: [results.json](results.json), [rules_validation.jsonl](rules_validation.jsonl), raw per-request outputs and
  measurements in `runs/qwen3-0.6b/` and `runs_attempt2_qwen3-1.7b/qwen3-1.7b/`, and the archived attempt in
  `runs/qwen3-1.7b/`.

## Results (measured only; validation, n = 160, full denominator)

**Integrity (D17):** no issue. The rules rerun was identical to EXP-013 on all 160 raw outputs. Every output file
held exactly its expected records. Both models had 160 of 160 requests with status `ok`.

| Metric | rules `axiom.rules/0.1` | Qwen3-0.6B untuned | Qwen3-1.7B untuned |
|---|---|---|---|
| **Semantic exact match** | **73 (45.6 %)**, cluster 95 % 29.1-63.3 % | **0 (0.0 %)** | **12 (7.5 %)**, cluster 95 % 2.6-12.6 % |
| JSON valid | 160 | 2 | 132 |
| **Candidate-schema valid** | 160 | **2 (1.25 %)** | **53 (33.1 %)** |
| Semantically valid | 160 | 2 | 41 |
| Action accuracy / macro F1 | 115 / 0.713 | 2 / 0.042 | 28 / 0.297 |
| Operation accuracy among gold PLAN (85) | 33 | 0 | 14 |
| Clarify recall (45) / reject recall (30) | 45 / 16 | 0 / 2 | 2 / 11 |
| Wrong executable proposals | 26 | 0 | 14 |
| Guard: awaiting confirmation (correct / wrong) | 56 (30 / 26) | 0 | 22 (8 / 14) |
| **Unauthorized dispatches** | **0** | **0** | **0** |
| Exact match by language: TR (64) / EN (64) / mixed (32) | 22 / 35 / 16 | 0 / 0 / 0 | 2 / 5 / 5 |
| By stratum: A held-out family (54) / B novel (53) / C familiar (53) | 16 / 27 / 30 | 0 / 0 / 0 | 2 / 5 / 5 |

Main failure modes, from the scorer's error codes:
- **Qwen3-0.6B:** 133 invalid JSON and 25 nested too deeply. 158 of its 160 raw outputs begin with a Markdown code
  fence, which strict parsing rejects. No 1.7B output does.
- **Qwen3-1.7B:** 79 schema-invalid candidates, 26 nested too deeply, 6 out-of-range `k`, 6 wrong column kinds, and
  2 other JSON errors.

**Paired comparisons** (2×2 counts: both correct / only first / only second / neither; paired cluster bootstrap 95 %
of the difference):

| Comparison | Metric | 2×2 | Δ (pp) | 95 % interval |
|---|---|---|---|---|
| 0.6B vs rules | exact match | 0 / 73 / 0 / 87 | −45.6 | −63.3 to −29.1 |
| 1.7B vs rules | exact match | 5 / 68 / 7 / 80 | −38.1 | −57.7 to −19.8 |
| 1.7B vs 0.6B | exact match | 0 / 0 / 12 / 148 | +7.5 | +2.6 to +12.6 |
| 1.7B vs 0.6B | candidate-schema valid | 2 / 0 / 51 / 107 | +31.9 | +20.7 to +44.9 |

McNemar p-values (an IID illustration only) are in `results.json`. Both untuned models are **below the rule
baseline**.

**Resources** (MacBook Air M5 16 GB; one observation per process; measured separately and never summed):

| | Qwen3-0.6B | Qwen3-1.7B (attempt 2) |
|---|---|---|
| Warm latency, 160 validation requests: median (IQR), max | 854 ms (718-1165), 3024 ms | 1717 ms (1496-2162), 7779 ms |
| Generated tokens per request: median (IQR), requests at the 256 cap | 70 (58-97), 0 | 56 (48-72), 9 |
| Decode throughput baseline (median of the first 30 requests) | 88.9 tokens/s | 36.2 tokens/s |
| Cold load (main process; files cached by the re-hash, D6') | 0.34 s | 0.78 s |
| Process peak RSS: main / maximum over all 6 processes | 1.68 GB / 1.68 GB | 1.29 GB / 3.93 GB |
| MLX peak: load / validation run | 1.19 GB / 1.65 GB | 3.44 GB / 3.80 GB |
| Swap growth since the pre-load baseline (all processes) | 0 bytes | 0 bytes |
| Maximum memory-pressure level (all samples, all processes) | 1 (normal) | 1 (normal) |
| `memory_pressure` free %: preflight / after warm-up / after validation / end (main) | 72 / 64 / 66 / 70 | 76 / 52 / 55 / 54 |
| Stop rules triggered | none | none |

**Repeatability (D12, D13, D18).** Thirty stratified requests were compared with each request's output in the
160-request run:

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Observations: in-process (20×) / fresh processes (5×) | 600 of 600 / 150 of 150, 0 failed | 600 of 600 / 150 of 150, 0 failed |
| Exact text identical: in-process / fresh | 30 of 30 / 30 of 30 | 30 of 30 / 30 of 30 |
| Canonical candidate identical | not assessable for all 30 (no reference is a valid candidate) | 9 of 9 assessable, 21 not assessable |
| Semantic correctness identical | 30 of 30 / 30 of 30 | 30 of 30 / 30 of 30 |

The interrupted 1.7B attempt 1 (archived) produced validation raw outputs identical to attempt 2 on 160 of 160
requests, comparing whole texts byte for byte. This is a reproducibility observation across two processes on the
same machine and was not used to choose an attempt.

## Base selection (rule frozen in protocol revision 3)

1. **Safety stop:** not triggered. Unauthorized dispatches were 0 for the rules, 0.6B and 1.7B.
2. **Eligibility (D10'):** both models are eligible.
   - Every required process completed without a stop.
   - Maximum process peak RSS was 1.68 GB and 3.93 GB, both under 10 × 10⁹ bytes.
   - Median warm latency was 854 ms and 1717 ms, both under 2000 ms.
3. **Primary (rule 4):** not applied. Neither model reached the 20 % exact-match floor (0 % and 7.5 %; 32 of 160 is
   needed).
4. **Secondary (rule 5):** Qwen3-1.7B reached the 20 % candidate-schema-validity floor (33.1 %); Qwen3-0.6B did not
   (1.25 %). The 1.7B is better by rule 3: +31.9 pp, more than 5 pp, with a 95 % interval of +20.7 to +44.9 that
   excludes zero.
5. **Result: Qwen3-1.7B is selected as the base for the LoRA pilot (EXP-010)**, under the secondary comparison and on
   the strength of a rule-3 difference.

This decides only which base EXP-010 would start from. EXP-010 still needs its own review and the owner's approval.
No untuned model is proposed for use in the application.

## Limitations and failures
- **Exploratory corpus.** It is synthetic and AI-generated, reviewed only by an owner-authorized AI review, with no
  human review. It is not blind: one tool wrote the corpus, rules, harness, runner and analysis.
- **Weak evidence.** Untuned zero-shot scores say little about which base will train best. The selection rests on
  schema validity, a structural proxy, because neither model reached the exact-match floor. The paired 1.7B-vs-0.6B
  exact-match difference (+7.5 pp, interval +2.6 to +12.6) is reported but was not used by the rule.
- **Strict scoring.** No repair or constrained decoding was used, as intended. The 0.6B's results are dominated by
  formatting (code fences), not by routing decisions.
- **RSS does not capture GPU memory.** Process RSS from `getrusage` does not track MLX's Metal allocations. The
  1.7B's main process RSS (1.29 GB) is below its MLX peak (3.80 GB), and its fresh processes reached 3.93 GB RSS. The
  10 GB eligibility check uses RSS as the protocol specifies. Both MLX peaks are also far below 10 GB.
- **Discrete sampling.** Swap and pressure were sampled after each request (D8'), so a peak between two samples
  could be missed.
- **Cached cold load.** "Cold load" was measured after the pre-load re-hash, so the files came from the operating
  system cache (D6').
- **Interrupted attempt.** Qwen3-1.7B attempt 1 was killed externally, most likely by the end of the Claude Code
  session that owned the background job. It is archived and was not spliced. The rerun was approved in advance
  (D19) and chosen by completeness, before scoring.
- **Single run.** One machine, one accepted run per model, with no independent reproduction. Latency and throughput
  are single observations on a fanless laptop and can vary with temperature and background load.
