"""Execution of bound plans (ADR 0004-0006).

All numbers come from this deterministic engine. The result *core* contains no timestamps, paths,
host names or timings; those belong to a separate environment manifest (:func:`environment_info`).
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any

from .canonical import SERIALIZER_VERSION, canonical_dumps, sha256_hex
from .errors import PlanError
from .ingest import Dataset
from .limits import DEFAULT_LIMITS, Limits
from .ops_inspect import run_inspect
from .plan import (
    POLICY_VERSION,
    BoundPlan,
    CandidatePlan,
    InspectPlan,
    bind_plan,
    plan_sha256,
    plan_to_dict,
)

RESULT_SCHEMA_VERSION = "axiom.result/1"

_EXECUTORS = {
    InspectPlan: run_inspect,
}


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Deterministic result core plus its canonical serialization and hash."""

    core: dict[str, Any]

    @property
    def canonical_json(self) -> str:
        return canonical_dumps(self.core)

    @property
    def result_sha256(self) -> str:
        return sha256_hex(self.canonical_json.encode("utf-8"))


def execute(dataset: Dataset, bound: BoundPlan, limits: Limits = DEFAULT_LIMITS) -> AnalysisResult:
    """Run a bound plan. Fails if the plan is not bound to exactly this dataset snapshot."""
    if (
        bound.dataset_raw_sha256 != dataset.raw_sha256
        or bound.dataset_logical_sha256 != dataset.logical_sha256
    ):
        raise PlanError(
            "plan_not_bound_to_dataset", "bound plan does not match this dataset snapshot"
        )
    if bound.plan_sha256 != plan_sha256(bound.plan) or bound.policy_version != POLICY_VERSION:
        raise PlanError("plan_hash_mismatch", "bound plan was modified after binding")
    result, warns = _EXECUTORS[type(bound.plan)](dataset, bound.plan, limits)
    core = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "serializer": SERIALIZER_VERSION,
        "policy_version": POLICY_VERSION,
        "operation": bound.plan.operation,
        "inputs": {
            "raw_sha256": dataset.raw_sha256,
            "dataset_sha256": dataset.logical_sha256,
            "plan_sha256": bound.plan_sha256,
            "n_rows": dataset.n_rows,
            "columns": [
                {"name": n, "kind": k}
                for n, k in zip(dataset.column_names, dataset.column_kinds, strict=True)
            ],
        },
        "plan": plan_to_dict(bound.plan),
        "result": result,
        "warnings": warns,
    }
    return AnalysisResult(core=core)


def run(dataset: Dataset, plan: CandidatePlan, limits: Limits = DEFAULT_LIMITS) -> AnalysisResult:
    """Bind ``plan`` to ``dataset`` and execute it."""
    return execute(dataset, bind_plan(dataset, plan, limits), limits)


def environment_info() -> dict[str, Any]:
    """Environment manifest. Deliberately separate from the deterministic result core."""
    import numpy
    import pydantic
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pydantic": pydantic.__version__,
    }
