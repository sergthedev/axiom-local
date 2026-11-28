# Training

**No model training has been performed.** This directory holds no training code or weights.

- Protocols (proposed, not run): EXP-009 untuned baselines and model selection, EXP-010 LoRA smoke and pilot, in
  [docs/evidence/experiments/](../docs/evidence/experiments/README.md); decision record
  [ADR 0012](../docs/decisions/0012-optional-model-environment.md).
- Environment: the optional Apple Silicon project in [envs/model/](../envs/model/README.md).
- Data: `python -m axbench export-mlx --out <directory outside the repository>` writes chat-format `train.jsonl`
  and `valid.jsonl` from the AXBENCH development splits. It never writes the test split.
- Training, validation and held-out test data stay separate. Model weights, adapters and caches are never committed.
- EXP-TR03 (proposed, draft, nothing run): research protocol and plan tooling in [tr03/](tr03/README.md) and
  [docs/evidence/experiments/EXP-TR03-research-protocol/](../docs/evidence/experiments/EXP-TR03-research-protocol/PROTOCOL.md);
  blocked by human, rights and clearance gates.
