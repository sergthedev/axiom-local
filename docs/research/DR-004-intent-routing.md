# DR-004: Intent routing, candidate protocol and authorization

- Status: **Synthesis of the technical lead's published DR-004 review. Not independent evidence.**
- Related: Issue #4, ADRs 0009 and 0010, `tests/test_router_contract.py`, `tests/test_guard.py`.

## Provenance

- Technical lead DR-004 review ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7168311964))
  and the earlier research note, REF-031. The raw report is private.
- Pydantic strict mode was re-opened earlier (REF-014). The other sources named in the review (Python 3.12 `json`,
  RFC 8259, Outlines 1.2.10 MLX-LM page, semantic parsing tutorial, PICARD) were **not** re-opened in this issue and
  are not registered; they would matter for constrained decoding, which is deferred.

## Decisions taken over (proposals, not results) and where they are implemented

| Lead requirement | Implementation |
|---|---|
| Closed, versioned PLAN/CLARIFY/REJECT union distinct from the Core plan; no execution parameters, paths, prose, code | `axiom.candidate/1` (ADR 0009) |
| Strict single-object JSON, duplicate keys rejected at every level, no NaN/Infinity/overflow, bounded size, depth, collections and digits, no repair | `parse_strict_json`, `CandidateLimits` |
| Pydantic strict, extra forbidden; bool-as-int, coercion, nested extras, unknown enums, repeated features, unpaired surrogates tested | `parse_candidate`; contract tests |
| Trusted semantic checks against the real schema | `validate_candidate` |
| Bind the dataset and complete plan **before** confirmation; display resolved plan, defaults, missing policy; confirmation tied to pending record, plan digest, dataset fingerprint, policy; revalidate and fail closed; one-shot consumption | `Guard` (ADR 0010) |
| Router technical failures are statuses, not REJECT; no hidden fallback | `Guard.router_failure`, harness statuses |
| Schema projection with schema-local ids, untrusted names escaped, collisions detected, no silent alias resolution | `axiom.schema-view/1` |
| Missing correlation method and cluster count clarify; unsupported analyses reject; multiple operations clarify | labeling rules (dataset card) |
| Instrumented dispatcher; zero calls for invalid, ambiguous, rejected, unconfirmed, cancelled, stale or edited plans; exactly-once positive test; same-schema dataset switch; digest mutation; reused approval | `tests/test_guard.py` |
| Manual path independent of the router | `Guard.propose_manual`; Core API unchanged |

Not implemented (deferred, stated in ADRs): a bounded schema shortlist for wide schemas (wider than 64 columns are
refused for the router), constrained decoding, the counterfactual permutation suite, persistent clarification state.
Zero observed dispatches in tests is not proof of absence; the Guard is not a sandbox against local code.
