# Model card — AX-01 (no model exists)

**Status: no model has been trained or evaluated.** Two candidate base checkpoints are identified and pinned, an
optional environment is prepared, and experiment protocols are proposed (ADR 0012). Every result field stays empty
until backed by an experiment in `docs/evidence/`.

| Field | Value |
|-------|-------|
| Intended role | Propose one `axiom.candidate/1` object (PLAN, CLARIFY or REJECT) for a Turkish, English or mixed request and a schema projection; never computes numbers, never executes; every PLAN needs explicit user confirmation through the AXIOM Guard (ADRs 0009, 0010) |
| Candidate base models | Qwen/Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`; Qwen/Qwen3-1.7B at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; both Apache-2.0 (REF-032, REF-033). Not downloaded in the repository; weights are never committed |
| Selection | Qwen3-1.7B, chosen by the frozen EXP-009 rule (secondary, schema validity) as the base for the exploratory LoRA pilot only, not for deployment (C-052) |
| Training data | AXBENCH `axbench/0.1-ai-reviewed` `train` split only (400 synthetic, AI-generated examples; owner-authorized AI review, 0 human decisions; [dataset card](data/axbench/DATASET_CARD.md)) — not yet used |
| Training procedure | LoRA pilot proposed (EXP-010 protocol and [pre-run record](docs/evidence/experiments/EXP-010-lora-pilot/PRERUN.md): rank 8, q/v, last 8 blocks, 458,752 trainable parameters, completion-only loss); harness prepared, not run |
| Evaluation data | AXBENCH validation for development; owner-sealed final test for one frozen evaluation (ADR 0011) |
| Measured results | None for any model. The rule baseline (not a model) scored 72 of 160 on validation (EXP-008) |
| Inference configuration (planned) | Chat template from the pinned tokenizer with `enable_thinking=False`; greedy decoding; bounded output; no repair of output; offline local files only |
| Limitations and failure modes | Unknown until measured. Known risks: wrong but valid plans (hence confirmation), Turkish morphology, synthetic training data written by the same tool as the benchmark |
| Out-of-scope use | Calculation, forecasting, causal claims, code or SQL generation, data modification, general chat |
