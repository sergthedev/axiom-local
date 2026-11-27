# Proposed EXP-TR02: LoRA on AXBENCH-TR (proposal only, not approved, not run)

Status: **proposed**. Training needs separate owner approval. EXP-010 is untouched and this proposal
does not depend on its outcome except for the throughput number it will measure.

## Question
Does adding AXBENCH-TR to the AXBENCH v0.1 training data improve Qwen3-1.7B on Turkish requests
without losing English accuracy, compared with the EXP-010 adapter trained on v0.1 alone?

## Design (to be pre-registered before any run)
- Base: Qwen3-1.7B, MLX LoRA, the same hyper-parameters, seed and `enable_thinking=False` as EXP-010.
- Arms: (A) v0.1 train only (the EXP-010 adapter, reused); (B) v0.1 train + AXBENCH-TR train; (C) AXBENCH-TR train only, to see what it teaches alone.
- Validation: the v0.1 validation split for English and continuity; AXBENCH-TR validation for Turkish; strict parsing, no repair, `axbench.score` rules.
- Sealed test: **not used.** Any use needs the owner's separate approval, after the owner runs the overlap command in the dataset card.
- Before any training: a native-speaker review of a stratified sample (suggested 300 records, 100 per provenance class and frame group), with record-level decisions kept in the repository; rejects feed back into the templates and the corpus is rebuilt.
- Report: per-language and per-frame exact match, schema-valid rate, 0 unauthorized dispatches, paired cluster bootstrap against arm A.

## Cost (arithmetic, not measurement)
- One pass over AXBENCH-TR train is 3,493,955 tokens (measured). With v0.1 train added the pass is larger by 400 short records.
- Hours per pass = 3,493,955 / (tokens per second) / 3600: 3.2 h at 300 tok/s, 1.6 h at 600, 0.8 h at 1,200. These throughputs are placeholders; the real value comes from the EXP-010 run on the M5.
- Three passes at the 600 tok/s placeholder: about 4.9 h per arm.

## Stop rules
Stop on swap growth, thermal throttling, memory pressure above normal, or non-finite loss, using the same protocol as EXP-010. Do not retry after a failure without a decision from the owner.

## Approvals required
Owner approval to start, to use any sealed data, and to merge anything.
