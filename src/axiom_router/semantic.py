"""Semantic validation of a candidate against the actual schema (ADR 0009).

Structural validity (``contract``) says the text is a well-formed candidate. This module says
whether a PLAN candidate can be expressed as a Core plan for *this* schema: identifiers exist,
kinds match the operation, arity and caps hold, and no referenced column is in a name collision.
It translates identifiers to the exact column names through the trusted projection and builds an
``axiom.plan/1`` plan with the Core models. Core binding (``axiom.plan.bind_plan``) remains the
final check (for example the cluster workload cap, which depends on the row count).

Semantic validity is not intent correctness: a valid plan can still be the wrong analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from axiom.limits import DEFAULT_LIMITS, Limits
from axiom.plan import CandidatePlan, ClusterPlan, ComparePlan, InspectPlan, RelatePlan

from .contract import (
    Candidate,
    ClarifyCandidate,
    ClusterCandidate,
    CompareCandidate,
    InspectCandidate,
    RejectCandidate,
    RelateCandidate,
)
from .schema import SchemaView


class SemanticError(Exception):
    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True, slots=True)
class Resolved:
    """A semantically valid candidate. ``plan`` is set for PLAN candidates only."""

    candidate: Candidate
    plan: CandidatePlan | None


def _kind(view: SchemaView, column_id: str) -> str:
    col = view.by_id(column_id)
    if col is None:
        raise SemanticError("unknown_column_id", "candidate references an unknown column id")
    return col.kind


def _name(view: SchemaView, column_id: str) -> str:
    col = view.by_id(column_id)
    assert col is not None  # checked by _kind first
    return col.name


def _require(view: SchemaView, ids: tuple[str, ...], kind: str | None) -> None:
    collided = {i for group in view.collisions for i in group}
    for column_id in ids:
        actual = _kind(view, column_id)
        if kind is not None and actual != kind:
            raise SemanticError(
                "wrong_column_kind", f"column must be {kind}", column=column_id, actual=actual
            )
        if column_id in collided:
            raise SemanticError(
                "column_collision",
                "column name collides with another after normalization; ask the user",
                column=column_id,
            )


def validate_candidate(
    candidate: Candidate, view: SchemaView, limits: Limits = DEFAULT_LIMITS
) -> Resolved:
    """Return the resolved candidate or raise :class:`SemanticError`."""
    match candidate:
        case InspectCandidate():
            _require(view, candidate.columns, None)
            if len(candidate.columns) > limits.max_columns:
                raise SemanticError("too_many_columns", "column cap exceeded")
            return Resolved(
                candidate, _core(InspectPlan, "inspect", columns=_names(view, candidate.columns))
            )
        case RelateCandidate():
            _require(view, (candidate.x, candidate.y), "numeric")
            plan = _core(
                RelatePlan,
                "relate",
                x=_name(view, candidate.x),
                y=_name(view, candidate.y),
                method=candidate.method,
            )
            return Resolved(candidate, plan)
        case CompareCandidate():
            _require(view, (candidate.group_by,), "text")
            _require(view, (candidate.measure,), "numeric")
            plan = _core(
                ComparePlan,
                "compare",
                group_by=_name(view, candidate.group_by),
                measure=_name(view, candidate.measure),
            )
            return Resolved(candidate, plan)
        case ClusterCandidate():
            _require(view, candidate.features, "numeric")
            if candidate.k > limits.max_k:
                raise SemanticError("k_out_of_range", "k exceeds the trusted cap", cap=limits.max_k)
            if len(candidate.features) > limits.max_cluster_features:
                raise SemanticError("too_many_features", "feature cap exceeded")
            plan = _core(
                ClusterPlan, "cluster", features=_names(view, candidate.features), k=candidate.k
            )
            return Resolved(candidate, plan)
        case ClarifyCandidate():
            for column_id in candidate.columns or ():
                _kind(view, column_id)
            return Resolved(candidate, None)
        case RejectCandidate():
            return Resolved(candidate, None)
    raise SemanticError("invalid_candidate", "not a candidate")


def _names(view: SchemaView, ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_name(view, i) for i in ids)


def _core(model: type, operation: str, **fields: object) -> CandidatePlan:
    """Build a Core plan with the Core models; defaults are the Core defaults (ADR 0005/0006)."""
    try:
        return model.model_validate(
            {"schema_version": "axiom.plan/1", "operation": operation, **fields}
        )
    except ValidationError as exc:
        raise SemanticError("core_plan_invalid", "Core plan validation failed") from exc
