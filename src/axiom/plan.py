"""Versioned, strict analysis plans and the binding of a candidate plan to a dataset (ADR 0005).

A *candidate* plan is untrusted input (manual or, later, model-produced). It never carries hashes.
The engine computes the dataset hash and the plan hash itself and produces a :class:`BoundPlan`.
A plan is data: nothing in it is ever evaluated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from .canonical import canonical_sha256
from .errors import PlanError
from .ingest import Dataset
from .limits import DEFAULT_LIMITS, Limits

PLAN_SCHEMA_VERSION = "axiom.plan/1"
POLICY_VERSION = "axiom.policy/1"

_Name = Annotated[str, Field(min_length=1)]


class _PlanBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["axiom.plan/1"]


def _unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


class InspectPlan(_PlanBase):
    operation: Literal["inspect"]
    columns: tuple[_Name, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> InspectPlan:
        _unique(self.columns, "columns")
        return self


class RelatePlan(_PlanBase):
    operation: Literal["relate"]
    x: _Name
    y: _Name
    method: Literal["pearson", "spearman"]

    @model_validator(mode="after")
    def _check(self) -> RelatePlan:
        if self.x == self.y:
            raise ValueError("x and y must be different columns")
        return self


class ComparePlan(_PlanBase):
    operation: Literal["compare"]
    group_by: _Name
    measure: _Name

    @model_validator(mode="after")
    def _check(self) -> ComparePlan:
        if self.group_by == self.measure:
            raise ValueError("group_by and measure must be different columns")
        return self


class ClusterPlan(_PlanBase):
    operation: Literal["cluster"]
    features: tuple[_Name, ...] = Field(min_length=1)
    k: int = Field(ge=2)
    scaling: Literal["standard_ddof0"] = "standard_ddof0"
    init: Literal["k-means++"] = "k-means++"
    algorithm: Literal["lloyd"] = "lloyd"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    n_init: int = Field(default=10, ge=1, le=100)
    max_iter: int = Field(default=300, ge=1, le=10_000)
    tol: float = Field(default=1e-4, gt=0.0, le=1e-1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check(self) -> ClusterPlan:
        _unique(self.features, "features")
        return self


CandidatePlan = InspectPlan | RelatePlan | ComparePlan | ClusterPlan
_ADAPTER: TypeAdapter[CandidatePlan] = TypeAdapter(
    Annotated[CandidatePlan, Field(discriminator="operation")]
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise PlanError("duplicate_json_key", "duplicate key in plan JSON")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise PlanError("nonfinite_json_constant", f"{name} is not allowed in plan JSON")


def parse_plan_json(text: str) -> CandidatePlan:
    """Parse and validate plan JSON text (rejects duplicate keys, NaN/Infinity, unknown fields)."""
    try:
        json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise PlanError("invalid_json", "plan is not valid JSON", pos=exc.pos) from exc
    try:
        return _ADAPTER.validate_json(text, strict=True)
    except ValidationError as exc:
        raise PlanError(
            "invalid_plan",
            "plan failed schema validation",
            errors=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc


def parse_plan_object(obj: Any) -> CandidatePlan:
    """Validate a plain Python object (dict/list/str/number) by routing it through strict JSON."""
    try:
        text = json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PlanError("invalid_plan", "plan object is not JSON-compatible") from exc
    return parse_plan_json(text)


def plan_to_dict(plan: CandidatePlan) -> dict[str, Any]:
    """Fully explicit JSON-compatible form of a plan (defaults included)."""
    return plan.model_dump(mode="json")


def plan_sha256(plan: CandidatePlan) -> str:
    return canonical_sha256({"policy_version": POLICY_VERSION, "plan": plan_to_dict(plan)})


@dataclass(frozen=True, slots=True)
class BoundPlan:
    """A validated plan bound to one dataset snapshot. Created only by :func:`bind_plan`."""

    plan: CandidatePlan
    plan_sha256: str
    dataset_raw_sha256: str
    dataset_logical_sha256: str
    policy_version: str


def _columns_of(plan: CandidatePlan) -> list[tuple[str, str]]:
    """(column, required kind) pairs referenced by a plan."""
    match plan:
        case InspectPlan():
            return [(c, "any") for c in plan.columns]
        case RelatePlan():
            return [(plan.x, "numeric"), (plan.y, "numeric")]
        case ComparePlan():
            return [(plan.group_by, "text"), (plan.measure, "numeric")]
        case ClusterPlan():
            return [(c, "numeric") for c in plan.features]
    raise AssertionError("unreachable")


def bind_plan(dataset: Dataset, plan: CandidatePlan, limits: Limits = DEFAULT_LIMITS) -> BoundPlan:
    """Semantic validation against the dataset schema; computes both hashes."""
    if not isinstance(plan, (InspectPlan, RelatePlan, ComparePlan, ClusterPlan)):
        raise PlanError("invalid_plan", "plan must be a validated plan model")
    for column, required in _columns_of(plan):
        if column not in dataset.columns:
            raise PlanError(
                "unknown_column", "plan references a column not in the dataset", column=column
            )
        actual = dataset.kind(column)
        if required != "any" and actual != required:
            raise PlanError(
                "wrong_column_kind",
                f"column must be {required}",
                column=column,
                actual=actual,
                required=required,
            )
    if isinstance(plan, InspectPlan) and len(plan.columns) > limits.max_columns:
        raise PlanError("too_many_columns", "column cap exceeded", cap=limits.max_columns)
    if isinstance(plan, ClusterPlan):
        if plan.k > limits.max_k:
            raise PlanError("k_too_large", "k exceeds cap", cap=limits.max_k)
        if len(plan.features) > limits.max_cluster_features:
            raise PlanError(
                "too_many_features", "feature cap exceeded", cap=limits.max_cluster_features
            )
    return BoundPlan(
        plan=plan,
        plan_sha256=plan_sha256(plan),
        dataset_raw_sha256=dataset.raw_sha256,
        dataset_logical_sha256=dataset.logical_sha256,
        policy_version=POLICY_VERSION,
    )
