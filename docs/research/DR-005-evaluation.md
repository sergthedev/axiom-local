# DR-005: Evaluation protocol for AX-01

- Status: **Synthesis of the technical lead's published DR-005 review. Not independent evidence.**
- Related: Issue #4, ADR 0011, `data/axbench/DATASET_CARD.md`, EXP-008.

## Provenance

- Technical lead DR-005 review ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7169124248)),
  REF-031. The raw report is private. The lead independently recalculated two hypothetical figures from the report
  (Wilson 300/360: 79.14-86.83 %; McNemar 30 vs 12 discordant: p = 0.0079159); they are arithmetic checks, not AXIOM
  data. The implementer reproduced both with `axbench.stats` and SciPy while developing the helpers.
- Sources named by the lead (Spider, CFQ, Deng et al. 2024, TurkishMMLU, TurBLiMP, NIST McNemar page, MLX lazy
  evaluation, scikit-learn group splitting) were **not** re-opened in this issue and are not registered.

## Decisions taken over (proposals, not results)

1. Sizes 400/160/360 (final TR 144, EN 144, mixed 72); a smaller pilot only if declared before any evaluation.
   Implemented at full size; the pilot sizes were not used.
2. Lineages split globally before paraphrase; connected-overlap audit; three final strata (held-out family, novel
   instance, familiar instance with new lineage). Implemented and audited (EXP-008 A2).
3. Provenance tracked truthfully; human review counted exactly; the final test kept outside the implementing agent's
   normal context in an owner-controlled directory, with only a hash and metadata published before the final run; a
   hash does not prevent access. Implemented as `axbench seal`; owner review and owner lineages pending.
4. Systems: competent deterministic bilingual rule router, both untuned Qwen3 candidates, then one LoRA model; the
   size/resource selection rule frozen before seeing validation predictions (EXP-009 protocol).
5. Primary endpoint: full-denominator semantic exact match before human correction; the secondary metrics and the
   separate raw/after-Guard/dispatch layers listed in ADR 0011. Implemented in `axbench.score` and `axbench.evaluate`.
6. Statistics: Wilson and McNemar as IID illustrations only; paired cluster bootstrap preferred; no p-value as a
   deployment rule; no equivalence claims from non-significance. Implemented in `axbench.stats`.
7. Provisional gates (ADR 0011) to be frozen after validation and before unsealing.
8. Performance and repeatability: synchronized timing, cold and warm runs, five warm-ups and at least 30 measured
   warm requests, median and IQR; process memory, MLX peak and memory pressure/swap recorded separately; repeatability
   of 30 stratified cases. Protocol only (EXP-009); nothing measured yet.
