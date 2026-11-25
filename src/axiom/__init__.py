"""AXIOM: local-first, deterministic data intelligence."""

from .engine import AnalysisResult, environment_info, execute, run
from .errors import AxiomError, ComputationError, InputError, PlanError
from .ingest import ColumnSpec, CsvProfile, Dataset, load_csv, load_csv_bytes, read_header
from .limits import DEFAULT_LIMITS, Limits
from .plan import (
    BoundPlan,
    CandidatePlan,
    ClusterPlan,
    ComparePlan,
    InspectPlan,
    RelatePlan,
    bind_plan,
    parse_plan_json,
    parse_plan_object,
)

__version__ = "0.0.0"

__all__ = [
    "DEFAULT_LIMITS",
    "AnalysisResult",
    "AxiomError",
    "BoundPlan",
    "CandidatePlan",
    "ClusterPlan",
    "ColumnSpec",
    "ComparePlan",
    "ComputationError",
    "CsvProfile",
    "Dataset",
    "InputError",
    "InspectPlan",
    "Limits",
    "PlanError",
    "RelatePlan",
    "__version__",
    "bind_plan",
    "environment_info",
    "execute",
    "load_csv",
    "load_csv_bytes",
    "parse_plan_json",
    "parse_plan_object",
    "read_header",
    "run",
]
