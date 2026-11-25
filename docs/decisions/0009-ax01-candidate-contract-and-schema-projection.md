# ADR 0009: AX-01 candidate contract, schema projection and semantic validation

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-26
- Issue / PR: Issue #4 (Phase 3A); implements the DR-003/DR-004 technical lead decisions
  ([DR-003](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7168013773),
  [DR-004](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7168311964)); builds on ADR 0005

## Context

AX-01 is an optional bilingual (Turkish/English) router that turns one natural-language request into a proposal for
one of the four accepted Core operations. Its output is untrusted text from a small language model. The lead
decisions require a closed, versioned output protocol distinct from the trusted Core plan, strict parsing with no
repair, semantic validation against the real dataset schema, and no data values, code, paths or tool use anywhere in
the model's input or output. The Core plan contract (`axiom.plan/1`, ADR 0005) and the engine digest (ADR 0007) must
not change.

## Alternatives considered

- **Let the model emit `axiom.plan/1` directly.** Rejected: Core plans carry execution parameters (cluster seed,
  `n_init`, scaling) and column names; a router must not choose them, and names are untrusted text.
- **Lenient JSON parsing with repair** (strip code fences, fix quotes, take the first object). Rejected by DR-004: a
  repaired output is not what the model said and inflates validity.
- **Put the router inside the `axiom` package.** Rejected: the engine digest is computed over every `.py` file of
  `axiom` (ADR 0007), so any router change would change every analysis passport. A separate package keeps Core
  identities stable and keeps the router out of the Core import graph.
- **Constrained decoding as the only safety layer.** Deferred (P1 in DR-004); a constraint helps validity but is
  not semantic correctness, and strict validation is required regardless.

## Decision

Implemented in the separate package `src/axiom_router/` (`contract.py`, `schema.py`, `semantic.py`, `prompt.py`).
Importing `axiom` never imports `axiom_router`; `axiom_router` imports no ML library (both tested).

**Candidate protocol `axiom.candidate/1`.** One JSON object, a tagged union on `action`:

| Action | Fields |
|---|---|
| `PLAN` + `inspect` | `columns`: 1-256 distinct column ids |
| `PLAN` + `relate` | `x`, `y` (distinct ids), `method`: `pearson` or `spearman` |
| `PLAN` + `compare` | `group_by`, `measure` (distinct ids) |
| `PLAN` + `cluster` | `features`: 1-64 distinct ids (ordered; Core decides the minimum), `k`: integer 2-999 (structural; Core limit applied later) |
| `CLARIFY` | `reason` (9 codes), optional `operation` hint, optional `columns` (1-8 ids, for ambiguity) |
| `REJECT` | `reason` (9 codes) |

Clarify reasons: `missing_operation`, `multiple_operations`, `missing_column`, `column_not_found`,
`ambiguous_column`, `column_type_mismatch`, `missing_correlation_method`, `missing_cluster_count`,
`invalid_parameter`. Reject reasons: `instruction_override`, `code_or_system_request`, `data_modification`,
`causal_inference`, `forecasting`, `regression`, `hypothesis_test`, `unsupported_analysis`, `out_of_scope`. Every
object carries `"version": "axiom.candidate/1"`. There is no free text, confidence, authorization flag, path, hash or
execution parameter in the protocol. Router technical failures (timeout, unavailable, crash) are system statuses of
the caller, never a REJECT.

**Two-stage strict parsing, no repair.** Stage `json`: bytes and UTF-8 checks, a 4096-byte cap, byte-order mark,
nesting depth 3, node count, string length 64, list length 256, integer digits 4, then a parse that rejects duplicate
keys at every level, floats, `NaN`/`Infinity`, unpaired surrogates and any trailing text; the top level must be an
object. Stage `schema`: Pydantic strict models with `extra="forbid"` and a callable discriminator, so unknown fields,
unknown enum values, booleans as integers, string numbers and repeated ids are rejected. Errors carry a stage and a
stable code (`CandidateError`).

**Schema projection `axiom.schema-view/1`.** The model sees only `{"version", "columns": [{"id", "name", "type"}]}`:
schema-local ids `c1..cN` in dataset order, the exact header text (untrusted data) and the Core kind
(`numeric`/`text`). No rows, sample values, file names or descriptions. More than 64 columns: the router input is
refused (`schema_too_wide`) and the user takes the manual path; no shortlist is implemented yet, so nothing is
silently truncated. A schema digest binds the projection to the dataset. Matching helpers (Turkish-aware lowering,
de-asciification, identifier splitting) only detect collisions; they never resolve a mention. Headers that collide
under these normalizations are recorded and any candidate that references a colliding column is refused
(`column_collision`). `display_name` renders control, format, bidirectional and private-use characters as visible
`\u{XXXX}` escapes for people; confusable letters are not detected, so the id is always shown next to the name.

**Semantic validation** (`validate_candidate`). For PLAN candidates: every id exists in the projection, kinds match
the operation (numeric `x`, `y`, `measure` and `features`; text `group_by`), no colliding column, `k` and feature
count within the Core `Limits`, and the Core plan model accepts the result. The output is a resolved pair (candidate,
Core plan) built with the Core defaults for clustering (standard scaling, `k-means++`, Lloyd, seed 42, `n_init` 10).
Validation proves schema compatibility only: a valid candidate can still be the wrong interpretation (ADR 0010).

**Router input.** `build_router_input(request, schema_view)`: the request (1-2000 characters, valid Unicode) and the
projection, serialized as JSON in the user turn after a fixed system instruction (`axiom.prompt/1`). The model never
receives data values.

## Tradeoffs and consequences

- The protocol is narrower than natural language: no multi-operation plans, no column subsets for Compare, no
  options; such requests are clarified or rejected.
- Strictness costs validity points for an untuned model; DR-005 counts those failures on the full denominator.
- Header collisions block plans on those columns even when the user meant one of them; the manual path remains.
- The structural cap `k <= 999` exists to bound parsing; the effective cap is the Core `max_k` (64).

## Unvalidated assumptions

- That a small model can produce this protocol reliably without constrained decoding (to be measured, EXP-009).
- That 64 columns and 2000 characters fit the prompt budget of the candidate models (token counts to be measured).
- That collision detection over the implemented normalizations catches the confusions that matter in Turkish and
  English headers; confusable scripts are out of scope.

## References

REF-014, REF-031
