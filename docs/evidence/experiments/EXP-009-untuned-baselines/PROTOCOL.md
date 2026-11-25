# EXP-009 protocol (proposed, not run): untuned Qwen3 baselines on AXBENCH validation and base selection

- Status: **proposed**. Requires technical review, the owner's annotation approval of AXBENCH and the owner's approval
  to download the checkpoints (about 1.5 GB and 4.1 GB). Phase 3B; nothing here has been executed.
- Issue: #4 (protocol), Phase 3B (execution). Decision record: ADR 0012. This protocol is committed before any model
  prediction exists; the selection rule below may only be changed by a new, reviewed protocol version before EXP-009
  runs.

## Systems

| System | Identity |
|---|---|
| `rules` | `axiom.rules/0.1`, frozen (EXP-008), rerun for the paired comparison |
| `qwen3-0.6b-untuned` | Qwen/Qwen3-0.6B @ `c1899de289a04d12100db370d81485cdf75e47ca`, BF16 weights as published |
| `qwen3-1.7b-untuned` | Qwen/Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16 weights as published |

Same for both models: `envs/model` lock (mlx 0.32.4, mlx-lm 0.32.2), files provisioned with `provision.py` and
verified by SHA-256, offline (`HF_HUB_OFFLINE=1`), prompt `axiom.prompt/1` (system instruction + user JSON of the
request and schema projection), the pinned chat template with `enable_thinking=False`, zero-shot, greedy decoding
(temperature 0), at most 256 new tokens, stop at the end-of-turn token, no constrained decoding, no output repair (the
raw text after the generation prompt is scored as is; a code fence or trailing text is invalid). Same 160 validation
examples, same harness (`axbench.evaluate`) and scorer as EXP-008. No quantization; a quantized deployment is a
separate later comparison.

## Measurements

1. Full AXBENCH metric set (ADR 0011) per system on validation, with per-example raw outputs committed.
2. Paired comparisons (each model vs rules, 1.7B vs 0.6B): 2x2 counts, percentage-point delta, paired cluster
   bootstrap over lineages (2000 replicates, seed 2026); McNemar only as an IID illustration.
3. Resources on the MacBook Air M5 16 GB, mains power, other applications closed, recorded per model:
   cold load (one observation per fresh process), five warm-up requests (train examples), then all 160 validation
   requests timed with synchronized MLX evaluation; median and IQR of per-request latency; generated tokens per
   request; process peak RSS (`resource.getrusage`), MLX peak memory (`mx.get_peak_memory`, reset before the run)
   and system swap usage (`sysctl vm.swapusage`) and memory pressure (`memory_pressure`) before and after, reported
   separately and never summed.
4. Repeatability: 30 stratified validation requests repeated 20 times in one process and once in each of five fresh
   processes; exact text, canonical candidate and semantic correctness reported separately.

## Stop rules

Stop a run on any allocation error, swap growth above 2 GB, memory pressure level "critical", or a sustained
slowdown of more than 50 % against the first 30 measured requests; record the partial run as failed.

## Base selection rule for the LoRA pilot (frozen here, before any prediction)

A model is **eligible** if its run completed without a stop rule, peak process RSS was at most 10 GB and median warm
latency was at most 2.0 s per request. Among eligible models, choose **Qwen3-0.6B** unless Qwen3-1.7B is higher by
more than 5 percentage points in validation semantic exact match; if both models are below 10 % exact match, compare
candidate-schema validity with the same 5-point margin instead. Ties go to the smaller model. If no model is eligible,
EXP-010 does not start and the result is reported. The final test is not used for this choice.

## Not claimed

Untuned scores are not the AX-01 result and are not compared with any external leaderboard. A model below the rule
baseline is reported as such.
