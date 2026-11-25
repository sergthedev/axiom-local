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
    # v1 solver policy (provisional, ADR 0006): bounded ranges around the defaults.
    n_init: int = Field(default=10, ge=1, le=20)
    max_iter: int = Field(default=300, ge=1, le=1_000)
    tol: float = Field(default=1e-4, ge=1e-8, le=1e-2, allow_inf_nan=False)

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


def _scan_depth(text: str, max_depth: int) -> None:
    """Reject excessive bracket nesting before any recursive parser sees the text."""
    depth = 0
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > max_depth:
                raise PlanError("plan_too_deep", "plan JSON nesting too deep", cap=max_depth)
        elif ch in "]}":
            depth -= 1


def _count_nodes(obj: Any, cap: int) -> None:
    stack = [obj]
    seen = 0
    while stack:
        item = stack.pop()
        seen += 1
        if seen > cap:
            raise PlanError("plan_too_large", "plan JSON has too many values", cap=cap)
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def parse_plan_json(text: str, limits: Limits = DEFAULT_LIMITS) -> CandidatePlan:
    """Parse and validate plan JSON text.

    Bounded input (byte, nesting and value-count caps from ``limits``); rejects duplicate keys,
    NaN/Infinity, unknown fields and non-strict types. Pathological input yields a structured
    :class:`PlanError`, never an uncaught ``RecursionError`` or ``ValueError``.
    """
    if not isinstance(text, str):
        raise PlanError("invalid_json", "plan JSON must be text")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise PlanError("invalid_json", "plan JSON is not valid UTF-8 text") from exc
    if size > limits.max_plan_bytes:
        raise PlanError("plan_too_large", "plan JSON exceeds byte cap", cap=limits.max_plan_bytes)
    _scan_depth(text, limits.max_plan_depth)
    try:
        parsed = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise PlanError("invalid_json", "plan is not valid JSON", pos=exc.pos) from exc
    except (RecursionError, ValueError, OverflowError, MemoryError) as exc:
        raise PlanError(
            "plan_input_rejected", f"plan JSON could not be parsed ({type(exc).__name__})"
        ) from exc
    _count_nodes(parsed, limits.max_plan_nodes)
    try:
        return _ADAPTER.validate_json(text, strict=True)
    except ValidationError as exc:
        raise PlanError(
            "invalid_plan",
            "plan failed schema validation",
            errors=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc


def parse_plan_object(obj: Any, limits: Limits = DEFAULT_LIMITS) -> CandidatePlan:
    """Validate a plain Python object (dict/list/str/number) by routing it through strict JSON."""
    try:
        text = json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise PlanError("invalid_plan", "plan object is not JSON-compatible") from exc
    return parse_plan_json(text, limits)


_PLAN_TYPES = (InspectPlan, RelatePlan, ComparePlan, ClusterPlan)


def revalidate_plan(plan: Any) -> CandidatePlan:
    """Re-run full strict validation on a plan object and return a freshly validated copy.

    Pydantic's frozen models still allow ``model_copy(update=...)`` and ``model_construct`` to
    create instances that skipped validation. Every trusted boundary (binding and execution)
    therefore validates again from the field values; subclasses and foreign types are rejected.
    """
    if type(plan) not in _PLAN_TYPES:
        raise PlanError("invalid_plan", "plan must be one of the validated plan models")
    try:
        return type(plan).model_validate(plan.model_dump(), strict=True)
    except ValidationError as exc:
        raise PlanError(
            "invalid_plan",
            "plan failed revalidation",
            errors=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise PlanError(
            "invalid_plan", f"plan could not be revalidated ({type(exc).__name__})"
        ) from exc


def plan_to_dict(plan: CandidatePlan) -> dict[str, Any]:
    """Fully explicit JSON-compatible form of a plan (defaults included)."""
    return plan.model_dump(mode="json")


def plan_sha256(plan: CandidatePlan) -> str:
    return canonical_sha256({"policy_version": POLICY_VERSION, "plan": plan_to_dict(plan)})


@dataclass(frozen=True, slots=True)
class BoundPlan:
    """A validated plan paired with the fingerprints of one dataset snapshot.

    Produced by :func:`bind_plan`. ``plan_sha256`` identifies the canonical computation (plan plus
    policy version); the pair of dataset fingerprints (raw bytes, typed dataset) says which
    snapshot it was bound to. None of this is a signature or an authorization: any code can
    construct a ``BoundPlan``. ``execute`` therefore revalidates the plan and re-checks it against
    the dataset and the limits supplied at execution. It does not defend against arbitrary local
    Python code, and human authorization of plans is deferred to the router work (Issue #4).
    """

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


def cluster_work(plan: ClusterPlan, n_rows: int) -> int:
    """Upper bound on distance-component evaluations: rows * k * features * n_init * max_iter."""
    return n_rows * plan.k * len(plan.features) * plan.n_init * plan.max_iter


def validate_plan_for_dataset(dataset: Dataset, plan: CandidatePlan, limits: Limits) -> None:
    """Semantic validation of a (re)validated plan against the dataset schema and the limits."""
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
        work = cluster_work(plan, dataset.n_rows)
        if work > limits.max_cluster_work:
            raise PlanError(
                "cluster_workload_too_large",
                "rows * k * features * n_init * max_iter exceeds the workload cap",
                work=work,
                cap=limits.max_cluster_work,
            )


def bind_plan(dataset: Dataset, plan: CandidatePlan, limits: Limits = DEFAULT_LIMITS) -> BoundPlan:
    """Revalidate a plan, validate it against the dataset and limits, and compute both hashes."""
    plan = revalidate_plan(plan)
    validate_plan_for_dataset(dataset, plan, limits)
    return BoundPlan(
        plan=plan,
        plan_sha256=plan_sha256(plan),
        dataset_raw_sha256=dataset.raw_sha256,
        dataset_logical_sha256=dataset.logical_sha256,
        policy_version=POLICY_VERSION,
    )
