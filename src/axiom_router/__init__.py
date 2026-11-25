"""AX-01 router boundary: candidate contract, schema projection, semantic validation and Guard.

Nothing in this package computes statistics or imports a model runtime. Numbers come only from the
deterministic Core (``axiom``), and only after the Guard has recorded an explicit confirmation.
"""

from .contract import (
    CANDIDATE_VERSION,
    CLARIFY_REASONS,
    OPERATIONS,
    REJECT_REASONS,
    Candidate,
    CandidateError,
    CandidateLimits,
    candidate_to_dict,
    candidate_to_json,
    parse_candidate,
    parse_strict_json,
)
from .guard import GUARD_POLICY_VERSION, ConfirmationView, Guard, GuardError, Review, ReviewStatus
from .prompt import PROMPT_VERSION, RouterInput, RouterInputError, build_router_input
from .schema import SchemaView, display_name, project_columns, project_schema
from .semantic import Resolved, SemanticError, validate_candidate

__all__ = [
    "CANDIDATE_VERSION",
    "CLARIFY_REASONS",
    "GUARD_POLICY_VERSION",
    "OPERATIONS",
    "PROMPT_VERSION",
    "REJECT_REASONS",
    "Candidate",
    "CandidateError",
    "CandidateLimits",
    "ConfirmationView",
    "Guard",
    "GuardError",
    "Resolved",
    "Review",
    "ReviewStatus",
    "RouterInput",
    "RouterInputError",
    "SchemaView",
    "SemanticError",
    "build_router_input",
    "candidate_to_dict",
    "candidate_to_json",
    "display_name",
    "parse_candidate",
    "parse_strict_json",
    "project_columns",
    "project_schema",
    "validate_candidate",
]
