"""AXIOM Guard: the deterministic boundary between router candidates and Core execution (ADR 0010).

Four questions are kept apart:

1. **structural validity**: is the router output a well-formed candidate (``contract``)?
2. **semantic validity**: can it be expressed as a Core plan for this schema (``semantic``) and does
   the Core accept the binding (``axiom.plan.bind_plan``)?
3. **user intent correctness**: is it the analysis the person meant? Only the person can say; the
   Guard shows the exact interpreted plan and the dataset it is bound to and asks.
4. **execution authorization**: has the person confirmed *this* preview, for *this* dataset
   snapshot, and has nothing changed since? Only then is the executor called, exactly once.

A valid candidate is never authorized by itself. Confirmation is tied to a trusted pending record
held by this ``Guard`` instance (one session): a random token, the digest of what was displayed
(plan digest, dataset fingerprints, schema digest, policy versions), a creation time and a state.
Any later proposal supersedes earlier previews; edits, cancellation, a dataset change, a policy
change or expiry invalidate a preview; a preview can be consumed once. On confirmation the plan is
bound again against the current dataset and compared with what was displayed; any difference
fails closed and requires a fresh preview. There is no silent rebinding.

The manual path does not depend on any router: :meth:`Guard.propose_manual` accepts a validated
Core plan, and the Core API (``axiom.run`` / ``bind_plan`` / ``execute``) is unchanged.

This is an in-process control for the application flow. It is not a defense against arbitrary
local Python code (which can call the Core directly) and does not prove that a plan is what the
person intended: confirmation authorizes execution, it does not validate intent.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from axiom.canonical import canonical_sha256
from axiom.engine import AnalysisResult, execute
from axiom.errors import AxiomError, PlanError
from axiom.ingest import Dataset
from axiom.limits import DEFAULT_LIMITS, Limits
from axiom.plan import (
    POLICY_VERSION,
    BoundPlan,
    CandidatePlan,
    bind_plan,
    plan_to_dict,
    revalidate_plan,
)

from .contract import (
    CANDIDATE_VERSION,
    DEFAULT_CANDIDATE_LIMITS,
    CandidateError,
    CandidateLimits,
    ClarifyCandidate,
    RejectCandidate,
    parse_candidate,
)
from .schema import SchemaView, display_name, project_schema
from .semantic import SemanticError, validate_candidate

GUARD_POLICY_VERSION = "axiom.guard-policy/1"
DEFAULT_TTL_SECONDS = 900.0
MAX_CLOSED_RECORDS = 256  # closed previews remembered so a reused token reports why it is refused

Executor = Callable[[Dataset, BoundPlan, Limits], AnalysisResult]


class ReviewStatus(StrEnum):
    # Technical router failures: system statuses, never a semantic REJECT.
    ROUTER_UNAVAILABLE = "router_unavailable"
    ROUTER_TIMEOUT = "router_timeout"
    ROUTER_ERROR = "router_error"
    # Structural stage.
    INVALID_JSON = "invalid_json"
    INVALID_CANDIDATE = "invalid_candidate"
    # Semantic stage (including Core binding).
    SEMANTIC_INVALID = "semantic_invalid"
    BINDING_REJECTED = "binding_rejected"
    # Valid candidates.
    CLARIFY = "clarify"
    REJECT = "reject"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


ROUTER_FAILURES = frozenset(
    {ReviewStatus.ROUTER_UNAVAILABLE, ReviewStatus.ROUTER_TIMEOUT, ReviewStatus.ROUTER_ERROR}
)


class GuardError(AxiomError):
    """Authorization refused. ``code`` is stable; no dispatch happened."""


@dataclass(frozen=True, slots=True)
class ConfirmationView:
    """Everything the person must see before confirming, built from trusted values only."""

    token: str
    confirmation_digest: str
    source: str
    operation: str
    columns: tuple[dict[str, str], ...]
    parameters: dict[str, Any]
    missing_values: str
    plan: dict[str, Any]
    plan_sha256: str
    dataset_raw_sha256: str
    dataset_logical_sha256: str
    schema_sha256: str
    policy_version: str
    guard_policy_version: str


@dataclass(frozen=True, slots=True)
class Review:
    status: ReviewStatus
    code: str | None = None
    reason: str | None = None
    confirmation: ConfirmationView | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def structurally_valid(self) -> bool:
        return self.status not in ROUTER_FAILURES and self.status not in (
            ReviewStatus.INVALID_JSON,
            ReviewStatus.INVALID_CANDIDATE,
        )

    @property
    def semantically_valid(self) -> bool:
        return self.status in (
            ReviewStatus.CLARIFY,
            ReviewStatus.REJECT,
            ReviewStatus.AWAITING_CONFIRMATION,
        )


_MISSING = {
    "inspect": "Each column is summarized on its non-missing values; missing counts are reported.",
    "relate": "Rows with a missing value in either column are excluded and counted.",
    "compare": "Rows with a missing group key are excluded and counted; group statistics use "
    "non-missing measure values.",
    "cluster": "Only rows with a value in every feature are clustered; others are excluded.",
}


class _State(StrEnum):
    PENDING = "pending"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    STALE = "stale"


@dataclass(slots=True)
class _Pending:
    bound: BoundPlan
    digest: str
    schema_sha256: str
    created: float
    state: _State


def _digest(token: str, bound: BoundPlan, schema_sha256: str) -> str:
    return canonical_sha256(
        {
            "guard_policy_version": GUARD_POLICY_VERSION,
            "token": token,
            "plan_sha256": bound.plan_sha256,
            "policy_version": bound.policy_version,
            "dataset_raw_sha256": bound.dataset_raw_sha256,
            "dataset_logical_sha256": bound.dataset_logical_sha256,
            "schema_sha256": schema_sha256,
        }
    )


class Guard:
    """One analysis session's authorization boundary for one current dataset snapshot."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        limits: Limits = DEFAULT_LIMITS,
        candidate_limits: CandidateLimits = DEFAULT_CANDIDATE_LIMITS,
        executor: Executor = execute,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("Guard requires a Dataset")
        self._lock = threading.Lock()
        self._dataset = dataset
        self._view = project_schema(dataset)
        self._limits = limits
        self._candidate_limits = candidate_limits
        self._executor = executor
        self._clock = clock
        self._ttl = float(ttl_seconds)
        self._pending: dict[str, _Pending] = {}

    # ------------------------------------------------------------------ session state

    @property
    def schema_view(self) -> SchemaView:
        return self._view

    def set_dataset(self, dataset: Dataset) -> None:
        """Switch the session to another snapshot; every open preview becomes stale."""
        if not isinstance(dataset, Dataset):
            raise TypeError("Guard requires a Dataset")
        with self._lock:
            self._invalidate_all(_State.STALE)
            self._dataset = dataset
            self._view = project_schema(dataset)

    def cancel(self, token: str) -> None:
        with self._lock:
            record = self._pending.get(token)
            if record is not None and record.state is _State.PENDING:
                record.state = _State.CANCELLED

    # ------------------------------------------------------------------ proposals

    @staticmethod
    def router_failure(status: ReviewStatus, code: str | None = None) -> Review:
        """Record a technical router failure (unavailable, timeout, error): never a REJECT."""
        if status not in ROUTER_FAILURES:
            raise ValueError("not a router failure status")
        return Review(status, code=code)

    def review_candidate_text(self, text: Any, *, source: str = "model") -> Review:
        """Validate router output and, for a valid PLAN, open a confirmation preview."""
        try:
            candidate = parse_candidate(text, self._candidate_limits)
        except CandidateError as exc:
            status = (
                ReviewStatus.INVALID_JSON if exc.stage == "json" else ReviewStatus.INVALID_CANDIDATE
            )
            return Review(status, code=exc.code)
        with self._lock:
            try:
                resolved = validate_candidate(candidate, self._view, self._limits)
            except SemanticError as exc:
                return Review(ReviewStatus.SEMANTIC_INVALID, code=exc.code)
            if isinstance(candidate, ClarifyCandidate):
                return Review(ReviewStatus.CLARIFY, reason=candidate.reason)
            if isinstance(candidate, RejectCandidate):
                return Review(ReviewStatus.REJECT, reason=candidate.reason)
            assert resolved.plan is not None
            return self._open_preview(resolved.plan, source)

    def propose_manual(self, plan: CandidatePlan) -> Review:
        """Manual path: a Core plan written or edited by a person gets the same preview."""
        try:
            plan = revalidate_plan(plan)
        except PlanError as exc:
            return Review(ReviewStatus.INVALID_CANDIDATE, code=exc.code)
        with self._lock:
            return self._open_preview(plan, "manual")

    def edit(self, token: str, plan: CandidatePlan) -> Review:
        """Replace a preview by an edited plan. The old preview can never be confirmed again."""
        with self._lock:
            record = self._pending.get(token)
            if record is not None and record.state is _State.PENDING:
                record.state = _State.SUPERSEDED
        return self.propose_manual(plan)

    def _open_preview(self, plan: CandidatePlan, source: str) -> Review:
        # Caller holds the lock.
        try:
            bound = bind_plan(self._dataset, plan, self._limits)
        except AxiomError as exc:
            return Review(ReviewStatus.BINDING_REJECTED, code=exc.code)
        self._invalidate_all(_State.SUPERSEDED)
        token = secrets.token_urlsafe(24)
        digest = _digest(token, bound, self._view.schema_sha256)
        self._pending[token] = _Pending(
            bound, digest, self._view.schema_sha256, self._clock(), _State.PENDING
        )
        return Review(
            ReviewStatus.AWAITING_CONFIRMATION,
            confirmation=self._view_of(token, digest, bound, source),
        )

    def _view_of(self, token: str, digest: str, bound: BoundPlan, source: str) -> ConfirmationView:
        plan = plan_to_dict(bound.plan)
        ids = {c.name: c for c in self._view.columns}
        roles: list[tuple[str, str]] = []
        params: dict[str, Any] = {}
        op = plan["operation"]
        if op == "inspect":
            roles = [("column", n) for n in plan["columns"]]
        elif op == "relate":
            roles = [("x", plan["x"]), ("y", plan["y"])]
            params = {"method": plan["method"]}
        elif op == "compare":
            roles = [("group_by", plan["group_by"]), ("measure", plan["measure"])]
        elif op == "cluster":
            roles = [("feature", n) for n in plan["features"]]
            params = {
                k: plan[k] for k in plan if k not in ("schema_version", "operation", "features")
            }
        columns = tuple(
            {
                "role": role,
                "id": ids[name].id,
                "name": display_name(name),
                "kind": ids[name].kind,
            }
            for role, name in roles
        )
        return ConfirmationView(
            token=token,
            confirmation_digest=digest,
            source=source,
            operation=op,
            columns=columns,
            parameters=params,
            missing_values=_MISSING[op],
            plan=plan,
            plan_sha256=bound.plan_sha256,
            dataset_raw_sha256=bound.dataset_raw_sha256,
            dataset_logical_sha256=bound.dataset_logical_sha256,
            schema_sha256=self._view.schema_sha256,
            policy_version=POLICY_VERSION,
            guard_policy_version=GUARD_POLICY_VERSION,
        )

    def _invalidate_all(self, state: _State) -> None:
        for record in self._pending.values():
            if record.state is _State.PENDING:
                record.state = state
        closed = [t for t, r in self._pending.items() if r.state is not _State.PENDING]
        for old in closed[: max(0, len(closed) - MAX_CLOSED_RECORDS)]:
            del self._pending[old]  # an evicted token is then reported as unknown: still refused

    # ------------------------------------------------------------------ authorization

    def confirm(self, token: Any, confirmation_digest: Any) -> AnalysisResult:
        """Execute a previewed plan once, after every check passes; otherwise raise GuardError."""
        if not isinstance(token, str) or not isinstance(confirmation_digest, str):
            raise GuardError("malformed_confirmation", "token and digest must be text")
        with self._lock:
            record = self._pending.get(token)
            if record is None:
                raise GuardError("unknown_token", "no preview with this token in this session")
            if record.state is not _State.PENDING:
                raise GuardError(f"preview_{record.state.value}", "preview can no longer be used")
            if not hmac.compare_digest(record.digest, confirmation_digest):
                raise GuardError("digest_mismatch", "confirmation does not match the preview")
            if self._clock() - record.created > self._ttl:
                record.state = _State.STALE
                raise GuardError("preview_expired", "preview expired; review the plan again")
            dataset = self._dataset
            if (
                record.bound.dataset_raw_sha256 != dataset.raw_sha256
                or record.bound.dataset_logical_sha256 != dataset.logical_sha256
                or record.schema_sha256 != self._view.schema_sha256
                or record.bound.policy_version != POLICY_VERSION
            ):
                record.state = _State.STALE
                raise GuardError("preview_stale", "dataset or policy changed since the preview")
            try:
                fresh = bind_plan(dataset, record.bound.plan, self._limits)
            except AxiomError as exc:
                record.state = _State.STALE
                raise GuardError(
                    "revalidation_failed", "plan no longer binds", cause=exc.code
                ) from exc
            if (
                fresh.plan_sha256 != record.bound.plan_sha256
                or _digest(token, fresh, self._view.schema_sha256) != record.digest
            ):
                record.state = _State.STALE
                raise GuardError("plan_changed", "plan differs from the confirmed preview")
            record.state = _State.CONSUMED  # one-shot: consumed before dispatch
            executor, limits = self._executor, self._limits
        return executor(dataset, fresh, limits)


__all__ = [
    "CANDIDATE_VERSION",
    "GUARD_POLICY_VERSION",
    "ConfirmationView",
    "Guard",
    "GuardError",
    "Review",
    "ReviewStatus",
]
