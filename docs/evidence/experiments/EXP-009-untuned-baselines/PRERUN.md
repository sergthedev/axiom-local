# EXP-009 pre-run record (written before the first validation prediction)

- Issue: #4, Phase 3B. Protocol: [PROTOCOL.md](PROTOCOL.md) revision 3 (unchanged by this record). Decision record:
  ADR 0012.
- Date: 2026-11-26. This file, the runner and the analysis code are committed **before** any validation prediction
  exists. The selection rule, its thresholds and denominators are those of the protocol and are not changed here.
- Authorization: the owner explicitly authorized downloading both pinned checkpoints and executing EXP-009 in the
  Phase 3B execution instruction of 2026-11-26 (project conversation, restated in the pull request). EXP-009 only; no
  LoRA training.

## Inputs restated

| Item | Value |
|---|---|
| Accepted main at branch point | `6d887f94bec25450d7482481ab4dcf19dfac9d88` (PR #9 merge) |
| Corpus | `axbench/0.1-ai-reviewed`, manifest SHA-256 `b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c` |
| Train | 400 records, SHA-256 `cf56e6da454331a6278109c89c0d687efb5852d3cba7cbd7f26daabe0f01bd9a` (warm-ups only) |
| Validation | 160 records, SHA-256 `80457d95a63ef2da96771add7275c4378a9fe0a58de80bc7c1301688f0eed3b8` |
| Corpus status | synthetic, AI-generated, owner-authorized AI review (54 decisions), **0 human decisions**; exploratory |
| Final test | sealed privately by the owner ([seal record](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7186347486), sealed test SHA-256 `f2c313a7…470028`); not generated, read, scored or reconstructed here; the seed is not known to the implementing agent |
| Rule reference | EXP-013: `axiom.rules/0.1`, 73 / 160; rerun on the same file by the analysis script |

Hashes recomputed locally on 2026-11-26 before any inference: manifest, train and validation match the values above.

## Provisioning (done, verified)

`envs/model/provision.py` at the pinned revisions into a directory outside the repository; both runs exited 0.
Every file whose SHA-256 the Hub publishes matched (refuse-on-mismatch was armed); the others were hashed and
recorded on first use. Full records: [provenance/qwen3-0.6b.json](provenance/qwen3-0.6b.json),
[provenance/qwen3-1.7b.json](provenance/qwen3-1.7b.json). An independent re-hash of the weight files with
`exp009_run.py verify` matched the pinned values for every safetensors shard of both checkpoints (the 1.7B shard index has no published hash; its first-use hash is in the provenance record).

| Model | Revision | Weight files verified | Download wall time (observed once) |
|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | `model.safetensors` `f47f7117…` | 1 min 16 s |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | shards `169ad53e…`, `912becff…` | 3 min 6 s |

The model environment `envs/model` was resynced with `uv sync --locked` (only the editable `axiom-local` package was
reinstalled after main moved); `uv sync --locked --check` then reported no changes.

## Harness (code committed with this record)

- `envs/model/exp009_run.py` (`exp009.runner/1`, model environment only): `verify`, `smoke` (train examples only,
  prints, writes nothing), `main` and `fresh` phases as described below. It refuses a dirty tree, a corpus whose
  manifest hash differs, a model directory whose provenance or sizes differ from the pins, and battery power (the
  `--allow-battery` switch exists only for harness checks and is not used for protocol runs).
- `benchmarks/issue4_exp009_analyze.py` (Core environment): scores raw outputs with harness `axbench.evaluate/0.2`,
  scorer `axbench.score/0.2` and Guard policy 2; reruns the rules; paired comparisons; resources; repeatability;
  applies the rule.
- `src/axbench/selection.py` (`exp009.base-selection/rev3`): the protocol's selection rule as code;
  `src/axbench/runwatch.py`: stop-rule bookkeeping and the fixed subsets. Tests: `tests/test_exp009_selection.py`.

Harness checks before this record (not results): train-only smoke runs of three train examples per model, on battery
power with `--allow-battery`, confirmed that loading works offline, the chat template renders with
`enable_thinking=False`, generation stops on the end-of-turn token and raw text is captured unmodified. No validation
example was given to any model.

## Operational definitions (not stated in the protocol; fixed here before any validation prediction)

The protocol leaves these details open. They are fixed here and submitted for review; none of them is a threshold of
the selection rule.

| ID | Protocol text | Operational definition |
|---|---|---|
| D1 | "stop at the end-of-turn token" | Generation stops on the EOS set MLX-LM 0.32.2 loads from the pinned checkpoint: `151645` (`<\|im_end\|>`, the end-of-turn token) and `151643` (`<\|endoftext\|>`), the same pair the checkpoint's `generation_config.json` lists. The stop token is not part of the scored text. |
| D2 | "greedy decoding (temperature 0)" | MLX-LM's default sampler (argmax); no sampler arguments; batch size 1; requests one after another. |
| D3 | "at most 256 new tokens" | `max_tokens=256` in `stream_generate`; a request that reaches the cap ends with `finish_reason: length` and its raw text is scored as is. |
| D4 | per-request latency | Wall time of the router call: chat-template rendering, tokenization, prefill and decode until the last token has been evaluated (each token is synchronized by MLX-LM), as `axbench.evaluate.run_router` measures it. "Median warm latency" (eligibility) is the median over the 160 validation requests after the five warm-ups. |
| D5 | "five warm-up requests (train examples)" | The first five train records in file order (`train-0001` to `train-0005`); their outputs are stored and never scored. |
| D6 | "cold load (one observation per fresh process)" | Wall time of `mlx_lm.load` (including evaluation of all parameters) in a new process. The operating system's file cache is not flushed, so files may already be cached from earlier processes; reported as observed. |
| D7 | MLX peak memory "reset before the run" | `mx.reset_peak_memory()` after the warm-ups, immediately before the 160 validation requests; the load-phase and repeatability-phase peaks are reported separately. |
| D8 | "swap growth above 2 GB" | `sysctl vm.swapusage` used bytes, sampled after every request, minus the value sampled immediately before the run; limit 2 × 10⁹ bytes. |
| D9 | memory pressure level "critical" | `sysctl kern.memorystatus_vm_pressure_level` = 4 (1 normal, 2 warning, 4 critical), sampled after every request; the `memory_pressure` free percentage is recorded before and after each phase. |
| D10 | "its run completed without a stop rule"; "peak process RSS" | The run that decides eligibility is the 160-request validation run. Peak RSS is `getrusage` `ru_maxrss` sampled right after that run (the process-lifetime peak, including repeatability, is reported as well). A stop during a repeatability run is reported as a failed repeatability run. |
| D11 | "sustained slowdown of more than 50 % against the first 30 measured requests" | Throughput per request = MLX-LM decode tokens per second. Baseline = median of the first 30 validation requests. After every further block of 10 requests, the block median below 50 % of the baseline counts as a slow window; three consecutive slow windows stop the run. This mirrors the EXP-010 throughput rule. Each repeatability run has its own baseline. |
| D12 | "30 stratified validation requests" | 10 per stratum (`A_heldout_family`, `B_novel_instance`, `C_familiar_instance`), drawn with `random.Random(2026).sample` from each stratum's sorted ids (`axbench.runwatch.repeat_subset`). |
| D13 | "repeated 20 times in one process and once in each of five fresh processes" | The 20 repeats run in the main process right after the 160 requests (repeat-major order). Each fresh process loads the model and runs the 30 requests once, without warm-up. The reference output for each request is its output in the 160-request run. |
| D14 | "Stop a run on any allocation error" | An exception whose message matches `malloc`, `alloc`, `out of memory` or `insufficient memory` stops the run. Any other exception becomes an `error` status for that request, as in the harness. |
| D15 | run order | 0.6B main, 0.6B fresh 1-5, 1.7B main, 1.7B fresh 1-5, each in a new process with nothing else from this experiment running; then the analysis. A stopped run is kept as failed; any rerun gets a new run directory and a logged reason. |
| D16 | "mains power, other applications closed" | The runner refuses to start on battery power. Closing other applications is done and attested by the owner before the runs; it cannot be checked by the runner. |

## Commands

```bash
# envs/model, clean tree at the commit containing this record, mains power, other applications closed
uv run --locked python exp009_run.py main  --model qwen3-0.6b --model-dir <dir> --out ../../docs/evidence/experiments/EXP-009-untuned-baselines/runs
uv run --locked python exp009_run.py fresh --model qwen3-0.6b --model-dir <dir> --out ../../docs/evidence/experiments/EXP-009-untuned-baselines/runs --index 1   # ... 5
# the same for qwen3-1.7b, then from the repository root:
uv run python benchmarks/issue4_exp009_analyze.py --runs docs/evidence/experiments/EXP-009-untuned-baselines/runs --rules-predictions docs/evidence/experiments/EXP-009-untuned-baselines/rules_validation.jsonl > docs/evidence/experiments/EXP-009-untuned-baselines/results.json
```

## Disclosed state of the target machine at preparation time

On 2026-11-26 at 08:18 UTC, the MacBook Air (Mac17,3, Apple M5, 16 GB, macOS 27.0.1) was on battery power with
2.9 GB of its 4 GB swap in use and a memory-pressure level of 2 (warning). The runs need mains power, and the starting
values are recorded in each run's measurements; stop rule D8 measures growth from that starting point.
