# ADR 0005: Typed plans, plan binding and result serialization v1

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-25
- Issue / PR: Issue #2 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230), item 1); compatibility with [Issue #3](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7167524679)

## Context

A model (later) or a person writes a request. The engine must treat it as untrusted data, bind it to exactly one
dataset snapshot, and emit a result whose identity does not depend on time, path or host.

## Decision

Implemented in `src/axiom/plan.py`, `src/axiom/canonical.py`, `src/axiom/engine.py`.

- **Candidate plans** are strict, frozen Pydantic models (`extra="forbid"`, strict types so `"2"` and `2.0` are not
  coerced to `2`; REF-014), versioned `axiom.plan/1`, one model per operation, discriminated by `operation`.
  JSON input additionally rejects duplicate keys and `NaN`/`Infinity`; infinite floats are rejected by field.
  Columns are explicit, ordered and unique. Cluster options (`scaling`, `init`, `algorithm`, `seed`, `n_init`,
  `max_iter`, `tol`) have documented defaults that are written out in the plan that is hashed.
- **No model-supplied hashes**: a candidate plan with any hash field is rejected as an unknown field. `bind_plan`
  validates the plan against the dataset schema (known columns, correct kinds, caps) and *the engine* computes the
  plan hash and records the dataset hashes, producing an immutable `BoundPlan`. `execute` refuses a bound plan whose
  dataset hashes differ from the snapshot (`plan_not_bound_to_dataset`) or whose plan hash no longer matches
  (`plan_hash_mismatch`). Plans are data; nothing is evaluated (no `eval`/`exec`).
- **Identities** in every result: raw bytes SHA-256, logical dataset SHA-256, plan SHA-256 (with the policy version
  `axiom.policy/1`), and the SHA-256 of the canonical result core.
- **Canonical JSON v1** (`axiom.canonical-json/1`): UTF-8, no whitespace, keys sorted by code point, finite numbers
  only, `-0.0` written as `0.0`, floats via Python `repr` (shortest round trip, never rounded). This is **not** RFC 8785:
  number formatting differs, and no conformance is claimed.
- **Separation**: the result *core* has no timestamps, paths, host names, PIDs or timings. `environment_info()`
  returns library versions and platform separately; Issue #3 owns the run passport and replay.
- Errors are structured (`AxiomError` subclasses with `code`, `message`, `details`); undefined statistics are `null` with a reason.

## Alternatives considered

- Accepting model-provided hashes (rejected by the lead decision).
- Rounding floats before hashing (rejected: would manufacture agreement).
- A JCS-conformant serializer (deferred; would need an ECMAScript-style number formatter).

## Tradeoffs and consequences

- Hashes establish identity and integrity only, never correctness or anonymity.
- Result hashes were identical across 20 fresh processes on one machine (EXP-001); cross-platform agreement is not claimed.

## Unvalidated assumptions

- Python `repr` of floats is stable across the supported platforms (true for CPython 3.12 by specification of
  shortest-round-trip repr, not independently measured here).

## References

REF-014

## Amendment 1 (2026-11-25): response to the technical review of PR #6, blocker 3

The original text above is preserved. Changes:

- **Revalidation at both trusted boundaries.** Frozen Pydantic models still allow `model_copy(update=...)` and
  `model_construct` to create instances that skipped validation. `bind_plan` and `execute` now re-run strict validation from
  the field values (`revalidate_plan`), reject subclasses and foreign types, and return/use the freshly validated plan.
- **Limits at execution.** `execute(dataset, bound, limits)` re-checks the plan against the dataset schema and the limits
  supplied *at execution*, so stricter limits cannot be bypassed by an earlier binding (`validate_plan_for_dataset`).
- **What a bound plan is.** `plan_sha256` identifies the canonical computation (plan plus policy version); the pair of
  dataset fingerprints (raw bytes and typed dataset) says which snapshot it was bound to. It is **not** a signature and not an
  authorization: any code can build a `BoundPlan`, and a forged plan that is valid and has a recomputed digest executes (a test
  documents this). The engine makes no claim to resist arbitrary local Python code. Human authorization of plans is deferred
  to the router work (Issue #4). `Dataset` fingerprints are not recomputed at execution (cost O(n)); a forged `Dataset` is out of
  scope in the same way.
- **Bounded plan input.** `parse_plan_json` enforces caps (provisional): 65,536 bytes, bracket nesting 8, 4,096 JSON values
  (`Limits.max_plan_bytes/depth/nodes`); the nesting check is a non-recursive scan that ignores brackets inside strings; pathological
  numbers (for example integers beyond Python's digit limit), lone surrogates, `RecursionError`, circular objects and non-text input
  produce structured `PlanError`s instead of uncaught exceptions. `parse_plan_object` has the same behaviour. No new framework.
- **Result identity.** `AnalysisResult` stores the canonical JSON text at construction; `core` returns an independent parsed copy,
  so mutating it cannot change `canonical_json` or `result_sha256`. Immutable-result claims remain limited to these bytes; Issue #3
  will test snapshot semantics for replay.
