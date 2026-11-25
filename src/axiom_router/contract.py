"""AX-01 candidate contract ``axiom.candidate/1`` (ADR 0009).

A *candidate* is the advisory output of an intent router (the AX-01 model, the rule baseline or a
test). It is untrusted text. It is distinct from the trusted Core plan (``axiom.plan/1``): it names
columns only by schema-local identifiers (``c1``, ``c2``, ...), carries no execution parameters
beyond the operation's user-facing slots, no hashes, paths, code, numbers computed from data, free
text or authorization flags. Three outcomes exist:

* ``PLAN``: exactly one supported operation with explicit slots;
* ``CLARIFY``: a supported operation seems intended but the request cannot be planned as stated;
* ``REJECT``: the request asks for something AXIOM does not do or must not do.

Parsing is two separate stages so evaluation can tell them apart:

1. *strict JSON* (:func:`parse_strict_json`): bounded bytes, nesting, value count, string length and
   integer digits; exactly one JSON object; no duplicate keys at any level; no floats, ``NaN`` or
   ``Infinity``; no unpaired surrogates; nothing before or after the object except JSON whitespace;
2. *candidate schema* (:func:`parse_candidate`): strict Pydantic models with ``extra="forbid"`` and
   closed vocabularies, so unknown fields, unknown enum values and coerced types are rejected.

Neither stage knows the dataset; semantic validation against the real schema is
:mod:`axiom_router.semantic`. Nothing is repaired, truncated or evaluated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    Tag,
    TypeAdapter,
    ValidationError,
    model_validator,
)

CANDIDATE_VERSION = "axiom.candidate/1"

Operation = Literal["inspect", "relate", "compare", "cluster"]
OPERATIONS: tuple[str, ...] = ("inspect", "relate", "compare", "cluster")

# CLARIFY: a supported operation is (or may be) intended; a slot is missing, ambiguous or invalid.
ClarifyReason = Literal[
    "missing_operation",
    "multiple_operations",
    "missing_column",
    "column_not_found",
    "ambiguous_column",
    "column_type_mismatch",
    "missing_correlation_method",
    "missing_cluster_count",
    "invalid_parameter",
]
CLARIFY_REASONS: tuple[str, ...] = ClarifyReason.__args__  # type: ignore[attr-defined]

# REJECT: no supported operation can satisfy the request, or the request must not be followed.
RejectReason = Literal[
    "instruction_override",
    "code_or_system_request",
    "data_modification",
    "causal_inference",
    "forecasting",
    "regression",
    "hypothesis_test",
    "unsupported_analysis",
    "out_of_scope",
]
REJECT_REASONS: tuple[str, ...] = RejectReason.__args__  # type: ignore[attr-defined]

# Schema-local column identifier: c1 .. c999 (the projection caps the real count; see schema.py).
ColumnId = Annotated[str, StringConstraints(pattern=r"^c[1-9][0-9]{0,2}$")]
MAX_K_STRUCTURAL = 999  # structural bound only; the trusted cap is Limits.max_k (semantic stage)


@dataclass(frozen=True, slots=True)
class CandidateLimits:
    """Bounds applied *before* any expensive processing of router output (provisional)."""

    max_bytes: int = 4_096
    max_depth: int = 3
    max_nodes: int = 300
    max_string_chars: int = 64
    max_int_digits: int = 4
    max_list_items: int = 256


DEFAULT_CANDIDATE_LIMITS = CandidateLimits()


class CandidateError(Exception):
    """Structured rejection. ``stage`` is ``"json"`` or ``"schema"``; ``code`` is stable."""

    def __init__(self, stage: str, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{stage}/{code}: {message}")
        self.stage = stage
        self.code = code
        self.message = message
        self.details = details


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["axiom.candidate/1"]


def _unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


class InspectCandidate(_Base):
    action: Literal["PLAN"]
    operation: Literal["inspect"]
    columns: tuple[ColumnId, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _check(self) -> InspectCandidate:
        _unique(self.columns, "columns")
        return self


class RelateCandidate(_Base):
    action: Literal["PLAN"]
    operation: Literal["relate"]
    x: ColumnId
    y: ColumnId
    method: Literal["pearson", "spearman"]

    @model_validator(mode="after")
    def _check(self) -> RelateCandidate:
        if self.x == self.y:
            raise ValueError("x and y must be different columns")
        return self


class CompareCandidate(_Base):
    action: Literal["PLAN"]
    operation: Literal["compare"]
    group_by: ColumnId
    measure: ColumnId

    @model_validator(mode="after")
    def _check(self) -> CompareCandidate:
        if self.group_by == self.measure:
            raise ValueError("group_by and measure must be different columns")
        return self


class ClusterCandidate(_Base):
    """Only the user-facing slots. Solver settings come from the Core defaults (ADR 0006)."""

    action: Literal["PLAN"]
    operation: Literal["cluster"]
    features: tuple[ColumnId, ...] = Field(min_length=1, max_length=64)
    k: int = Field(ge=2, le=MAX_K_STRUCTURAL)

    @model_validator(mode="after")
    def _check(self) -> ClusterCandidate:
        _unique(self.features, "features")
        return self


class ClarifyCandidate(_Base):
    """``operation`` and ``columns`` are optional hints for the trusted clarification form."""

    action: Literal["CLARIFY"]
    reason: ClarifyReason
    operation: Operation | None = None
    columns: tuple[ColumnId, ...] | None = Field(default=None, min_length=1, max_length=8)

    @model_validator(mode="after")
    def _check(self) -> ClarifyCandidate:
        if self.columns is not None:
            _unique(self.columns, "columns")
        return self


class RejectCandidate(_Base):
    action: Literal["REJECT"]
    reason: RejectReason


PlanCandidate = InspectCandidate | RelateCandidate | CompareCandidate | ClusterCandidate
Candidate = PlanCandidate | ClarifyCandidate | RejectCandidate
_CANDIDATE_TYPES = (
    InspectCandidate,
    RelateCandidate,
    CompareCandidate,
    ClusterCandidate,
    ClarifyCandidate,
    RejectCandidate,
)


def _tag(value: Any) -> str | None:
    if isinstance(value, dict):
        action = value.get("action")
        if action == "PLAN":
            op = value.get("operation")
            return op if isinstance(op, str) else None
        return action if isinstance(action, str) else None
    return None


_ADAPTER: TypeAdapter[Candidate] = TypeAdapter(
    Annotated[
        Annotated[InspectCandidate, Tag("inspect")]
        | Annotated[RelateCandidate, Tag("relate")]
        | Annotated[CompareCandidate, Tag("compare")]
        | Annotated[ClusterCandidate, Tag("cluster")]
        | Annotated[ClarifyCandidate, Tag("CLARIFY")]
        | Annotated[RejectCandidate, Tag("REJECT")],
        Discriminator(_tag),
    ]
)


# --------------------------------------------------------------------------- strict JSON stage


def _dup_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise CandidateError("json", "duplicate_key", "duplicate key in candidate JSON")
    return dict(pairs)


def _no_float(token: str) -> Any:
    raise CandidateError("json", "float_not_allowed", "the candidate contract has no real numbers")


def _no_constant(token: str) -> Any:
    raise CandidateError("json", "nonfinite_constant", f"{token} is not allowed")


def _make_int_hook(max_digits: int):
    def _int(token: str) -> int:
        if len(token.lstrip("-")) > max_digits:
            raise CandidateError("json", "integer_too_long", "integer has too many digits")
        return int(token)

    return _int


def _scan_depth(text: str, max_depth: int) -> None:
    """Non-recursive bracket-depth scan that ignores brackets inside strings."""
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
                raise CandidateError("json", "too_deep", "nesting too deep", cap=max_depth)
        elif ch in "]}":
            depth -= 1


def _check_values(obj: Any, limits: CandidateLimits) -> None:
    stack = [obj]
    seen = 0
    while stack:
        item = stack.pop()
        seen += 1
        if seen > limits.max_nodes:
            raise CandidateError("json", "too_many_values", "too many JSON values")
        if isinstance(item, str):
            _check_string(item, limits)
        elif isinstance(item, dict):
            for key in item:
                _check_string(key, limits)
            stack.extend(item.values())
        elif isinstance(item, list):
            if len(item) > limits.max_list_items:
                raise CandidateError("json", "list_too_long", "array has too many items")
            stack.extend(item)


def _check_string(value: str, limits: CandidateLimits) -> None:
    if len(value) > limits.max_string_chars:
        raise CandidateError("json", "string_too_long", "string exceeds the length cap")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CandidateError("json", "invalid_unicode", "unpaired surrogate in string") from exc


def parse_strict_json(text: Any, limits: CandidateLimits = DEFAULT_CANDIDATE_LIMITS) -> dict:
    """Stage 1: return the single JSON object in ``text`` or raise a ``json``-stage error."""
    if not isinstance(text, str):
        raise CandidateError("json", "not_text", "router output must be text")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise CandidateError("json", "invalid_unicode", "output is not valid Unicode text") from exc
    if size > limits.max_bytes:
        raise CandidateError(
            "json", "too_large", "output exceeds the byte cap", cap=limits.max_bytes
        )
    if text.startswith("﻿"):
        raise CandidateError("json", "bom", "byte order mark is not allowed")
    _scan_depth(text, limits.max_depth)
    try:
        obj = json.loads(
            text,
            object_pairs_hook=_dup_hook,
            parse_float=_no_float,
            parse_int=_make_int_hook(limits.max_int_digits),
            parse_constant=_no_constant,
        )
    except CandidateError:
        raise
    except json.JSONDecodeError as exc:
        raise CandidateError(
            "json", "invalid_json", "not a single valid JSON value", pos=exc.pos
        ) from exc
    except (RecursionError, ValueError, OverflowError, MemoryError) as exc:
        raise CandidateError("json", "invalid_json", f"rejected ({type(exc).__name__})") from exc
    if not isinstance(obj, dict):
        raise CandidateError("json", "not_an_object", "top-level value must be a JSON object")
    _check_values(obj, limits)
    return obj


def parse_candidate(text: Any, limits: CandidateLimits = DEFAULT_CANDIDATE_LIMITS) -> Candidate:
    """Stage 1 and stage 2: strict JSON, then the strict candidate schema."""
    parse_strict_json(text, limits)
    try:
        return _ADAPTER.validate_json(text, strict=True)
    except ValidationError as exc:
        raise CandidateError(
            "schema",
            "invalid_candidate",
            "candidate failed schema validation",
            errors=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc


def revalidate_candidate(candidate: Any) -> Candidate:
    """Re-run strict validation on an object that claims to be a candidate."""
    if type(candidate) not in _CANDIDATE_TYPES:
        raise CandidateError("schema", "invalid_candidate", "not a candidate model instance")
    try:
        return type(candidate).model_validate_json(candidate.model_dump_json(), strict=True)
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        raise CandidateError(
            "schema", "invalid_candidate", "candidate failed revalidation"
        ) from exc


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    """JSON-compatible form; optional CLARIFY hints that are unset are omitted."""
    return candidate.model_dump(mode="json", exclude_none=True)


def candidate_to_json(candidate: Candidate) -> str:
    """Compact JSON text with a fixed key order (the order of the model fields)."""
    return json.dumps(candidate_to_dict(candidate), ensure_ascii=False, separators=(",", ":"))
