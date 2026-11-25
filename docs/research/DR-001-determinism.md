# DR-001: Reproducibility, determinism and replay

- Status: **Synthesis of the technical lead's published DR-001 note and the DR-001 + DR-002 decision. Not independent
  evidence.**
- Related: Issue #3, ADRs 0007 and 0008, EXP-004, EXP-005.

## Provenance

- The technical lead published a DR-001 research note
  ([comment](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7166468157)) and a later decision that
  supersedes its provisional tolerances
  ([decision](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7167524679)); both are REF-021. The lead also
  reviewed six private AI-assisted Deep Research reports (DR-001 to DR-006). **The raw reports are private and are not
  published or quoted here**; statements below come from the published comments only.
- Primary sources named by the lead were re-opened by the implementer before registration where they could be retrieved
  (REF-022 to REF-030). The National Academies report (NBK547535, DOI 10.17226/25303) could not be retrieved (the request
  was answered with a browser check) and is therefore **not** registered; its distinction between computational
  reproducibility and independent replication is taken here only as the lead's statement.

## What the lead required (proposals and decisions, not results)

1. Four identities (raw bytes, ordered canonical typed dataset, trusted bound plan, full-precision canonical result);
   versioned serialization; timing, path, host and PID outside the scientific payload; no rounding to manufacture agreement;
   no RFC 8785 claim without compliance.
2. Captured environment: commit, lock and installed versions, OS/architecture, BLAS/OpenMP/threadpool/CPU dispatch where
   available; single native thread in reference mode where enforceable, configured before imports in the subprocess;
   unknown policy labelled unverified.
3. Separate accuracy gates from cross-platform comparison; per-operation thresholds fixed before evaluation; exact discrete
   comparison; byte identity in the same environment; tolerances only across platforms; exact partition sets.
4. Fresh-process replay without result cache, with a distinct workspace and run nonce; 20 actual executions per
   representative plan in development (100 per golden scenario is a later release target); mutation with predictable
   effect (old binding fails; rebinding gives the independently expected change); hash-sensitivity fixtures.
5. Statuses exact/numeric/semantic and pass/fail/not-applicable/not-run; not-run never green; independent oracle
   comparison distinguished from repeatability and internal postconditions.
6. Windows x64 numerical comparison distinct from package smoke tests and release packaging; Windows ARM64, ReproBLAS,
   GPU and exhaustive benchmarks deferred.

## What was implemented and where it differs

| Item | Status |
|------|--------|
| Four identities | **Reused unchanged** from Issue #2 (ADR 0005). |
| Passport and environment manifest | **Implemented** (ADR 0007). Data bytes are not embedded; the passport digest is not a signature. |
| Thread policy | **Implemented**: thread variables set before the worker interpreter starts; effective pools recorded. Apple Accelerate (NumPy's BLAS on the Mac) is invisible to threadpoolctl and is reported as unverified, not as single-threaded. |
| Replay and classification | **Implemented** (ADR 0008). |
| Universal tolerance atol 1e-12 / rtol 1e-9 (DR-001 note) | **Not used**: superseded by the decision; per-field criteria derived from ADRs 0004 and 0006 instead. |
| 20 fresh executions per plan | **Measured** in EXP-004 (ten plans). |
| macOS arm64 vs Windows x64 | **Measured** in EXP-005 on GitHub-hosted runners, report-only CI step. |
| Offline (network-disabled) rehearsal | **Not performed** in this issue; the engine has no networking code, but a network-disabled run was not executed. |
| PROV / RO-Crate | **Deferred** by the decision. |

## Limitations

- Everything was produced by the same AI implementer that wrote the engine; no independent human replication exists.
- Repeatability on one machine and agreement on CI runners do not show correctness; correctness evidence remains the
  independent oracles and NIST comparisons of Issue #2.
