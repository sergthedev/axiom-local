# EXP-009 protocol (proposed, not run): untuned Qwen3 baselines on AXBENCH validation and base selection

- Status: **proposed**. Requires technical review, a frozen AXBENCH corpus (see revision 3) and the owner's approval
  to download the checkpoints (about 1.5 GB and 4.1 GB). Phase 3B; nothing here has been executed.
- Issue: #4 (protocol), Phase 3B (execution). Decision record: ADR 0012. This protocol is committed before any model
  prediction exists; the selection rule below may only be changed by a new, reviewed protocol version before EXP-009
  runs.
- **Revision 2** (2026-11-26, PR #8 review response, still before any model prediction exists): scorer
  `axbench.score/0.2`, harness `axbench.evaluate/0.2` and Guard policy 2; evaluation on a frozen corpus version whose
  manifest hash is recorded before the first prediction; the selection rule gains a safety stop, an informativeness
  floor, a statistical condition on the practical margin, and a pre-declared fallback. Revision 1 is in the Git
  history of this file.
- **Revision 3** (2026-11-26, after the owner delegated the annotation review to the technical lead AI; still before
  any model prediction exists): the pre-run gate accepts the owner-authorized **AI-reviewed exploratory** corpus
  `axbench/0.1-ai-reviewed` (manifest SHA-256 `b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c`,
  400 train / 160 validation, 0 examples reviewed by a human). No example was removed, so every denominator (160) and
  threshold below is unchanged. The rule reference on this corpus is EXP-013 (73 / 160), not the draft's 72 / 160.

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
raw text after the generation prompt is scored as is; a code fence or trailing text is invalid). Validation examples:
the `validation.jsonl` of the frozen corpus `axbench/0.1-ai-reviewed` (`data/axbench/v0.1-ai-reviewed/`, manifest
SHA-256 above), loaded through its verified manifest; the EXP-009 pre-run record restates this hash before the first
prediction. The draft `axbench/0.1-draft` is not used for EXP-009. **Corpus status for every EXP-009 claim:** synthetic,
AI-generated, reviewed only by an owner-authorized AI review (`reviewer_type: ai`, 54 decisions, 3 request
corrections, 0 human decisions). Results on it are exploratory: they are not human-validated, not representative of
any population of real requests and not independent of the implementing tool. A later genuine human review would
create a new frozen version, and stronger claims would need a new run on it. Harness
`axbench.evaluate/0.2`, scorer `axbench.score/0.2` (EXP-011), Guard policy `axiom.guard-policy/2`; the rule system is
rerun on the same frozen file for the paired comparison. No quantization; a quantized deployment is a separate later
comparison.

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

Untuned zero-shot scores are weak evidence about which base will train best, so the rule is conservative and decides
in this order. All rates use the full denominator of 160 and scorer 0.2.

1. **Safety stop.** If the harness observes any unauthorized dispatch for any system, no model is selected and
   EXP-010 does not start until the cause is found and fixed under a new reviewed protocol.
2. **Eligibility.** A model is eligible if its run completed without a stop rule, peak process RSS was at most 10 GB
   and median warm latency was at most 2.0 s per request. If no model is eligible, EXP-010 does not start and the
   result is reported.
3. **Practical margin.** One model is *better* than the other on a metric only if it is higher by **more than 5
   percentage points** (at least 9 of 160 examples) **and** the 95 % paired cluster-bootstrap interval of the
   difference (lineages, 2000 replicates, seed 2026) excludes zero.
4. **Primary comparison, with an informativeness floor.** If at least one eligible model reaches **20 %** semantic
   exact match (32 of 160), choose Qwen3-1.7B only if it is better than Qwen3-0.6B on exact match by rule 3;
   otherwise choose Qwen3-0.6B.
5. **Secondary comparison.** If neither model reaches the exact-match floor but at least one reaches **20 %**
   candidate-schema validity, apply rule 4 to candidate-schema validity instead.
6. **Fallback.** If neither floor is reached, untuned scores are treated as uninformative: choose **Qwen3-0.6B** by
   default (lower memory and latency on the 16 GB target) and record that the choice was made by default, not by
   evidence. If only one model is eligible, it is chosen whatever its scores, and that is recorded the same way.
7. Ties and every unresolved case go to the smaller model. The final test is not used for this choice.

These floors only decide which base the LoRA pilot starts from. No untuned model is proposed for use in the
application: the application's fallback when no router meets a reviewed quality bar is the rule router and the manual
plan path behind the same Guard. A quality bar for using a trained model is not part of this protocol; it requires its
own pre-registered criterion before any evaluation of a trained adapter.

## Not claimed

Untuned scores are not the AX-01 result and are not compared with any external leaderboard. A model below the rule
baseline is reported as such.
