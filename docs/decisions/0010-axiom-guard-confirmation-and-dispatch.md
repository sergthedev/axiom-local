# ADR 0010: AXIOM Guard, explicit confirmation and one-shot dispatch

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-26
- Issue / PR: Issue #4 (Phase 3A); implements the DR-004 correction on binding before confirmation
  ([DR-004 review](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7168311964)); builds on ADRs
  0005 and 0009

## Context

A router candidate that parses and validates is still only a guess about what the person wants. DR-004 requires
four questions to stay separate (structural validity, semantic validity, user intent correctness, execution
authorization), the dataset snapshot and the complete Core plan to be bound **before** the confirmation is shown,
confirmation to be tied to a trusted pending record, and every stale, edited, cancelled, malformed or unconfirmed
path to dispatch nothing. The Core API and the manual path must keep working without any router.

## Alternatives considered

- **Execute valid candidates directly** (confirmation optional). Rejected: a valid plan can be the wrong plan;
  EXP-008 measured 26 such proposals out of 160 for the rule baseline.
- **Confirmation flag inside the candidate or the request.** Rejected: authorization must never come from model or
  request text.
- **Bind at confirmation time.** Rejected by DR-004: the person must see the resolved columns, defaults and dataset
  they approve; binding later could silently attach the approval to different data.
- **Persistent multi-turn clarification state.** Deferred: v0.1 returns a clarification to a trusted editable form.

## Decision

Implemented in `src/axiom_router/guard.py` (`axiom.guard-policy/1`); tests in `tests/test_guard.py`.

- A `Guard` instance is one session bound to one Core `Dataset`. Inputs: router text
  (`review_candidate_text`), a manual Core plan (`propose_manual`, no router involved), an edit of a pending preview
  (`edit`, which creates a new preview and invalidates the old one), or a router failure (`router_failure`, a status
  only). Every review returns a `Review` with a status: `router_unavailable`, `router_timeout`, `router_error`,
  `invalid_json`, `invalid_candidate`, `semantic_invalid`, `binding_rejected`, `clarify`, `reject` or
  `awaiting_confirmation`.
- Only `awaiting_confirmation` carries a `ConfirmationView`: a random token (`secrets`), the confirmation digest, the
  source (router or manual), the operation, every column with role, id, kind and escaped display name, all
  parameters including the Core cluster defaults, the missing-value policy text, the canonical plan and its digest,
  the dataset raw and typed fingerprints, the schema digest and the policy versions. The plan has already been bound
  to the dataset (`bind_plan`) when the view is created.
- The pending record holds the token, the digest of everything displayed, the creation time and a state. A new
  proposal supersedes all earlier pending previews; `cancel`, `set_dataset` (any dataset change, even with the same
  schema), a policy change and expiry (default 900 seconds) invalidate them. At most 256 closed records are kept.
- `confirm(token, confirmation_digest)` checks, under a lock: well-formed inputs, a known token, state `pending`,
  constant-time digest comparison (`hmac.compare_digest`), expiry, the same dataset and policy, then revalidates and
  re-binds the plan against the current dataset and requires the result to equal what was displayed. Any failure
  raises `GuardError` with a stable code and dispatches nothing; a mismatch requires a fresh preview. On success the
  record is consumed **before** the executor is called, so a preview runs at most once, even under concurrent
  confirmation and even if the executor fails. The executor (default `axiom.engine.execute`) is injectable for tests
  and runs outside the lock.

What the Guard is not: it does not prove that a plan matches the person's intent (confirmation authorizes execution,
it does not validate intent); it is not a sandbox against arbitrary local Python code, which can call the Core
directly; and it does not replace replay verification of an earlier approved analysis (ADR 0008).

## Tradeoffs and consequences

- Every routed analysis needs one explicit confirmation step; the application must render the view faithfully.
- In-memory records only: a restart forgets pending previews, which fails closed.
- The policy version must change whenever the meaning of a confirmation changes, so old previews are invalidated.

## Unvalidated assumptions

- That people read the confirmation view carefully enough to catch a wrong interpretation; not measured. The view
  design and its user testing belong to the Studio phase, which has not started.
- That the time-to-live of 900 seconds is appropriate; a provisional choice.

## References

REF-031
