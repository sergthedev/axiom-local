"""Analysis Passport v1: a versioned, machine-readable record of one analysis (ADR 0007).

A passport carries three things that are kept apart:

* the **scientific payload**: the four identities (raw bytes, typed dataset, bound plan, canonical
  result), the explicit inputs needed to recompute it (CSV profile, plan, limits) and the full
  precision canonical result core. Nothing here depends on time, path, host or process;
* the **environment manifest** (:mod:`axiom.environment`): where it was computed;
* the **execution record**: run nonce, mode, wall-clock timestamps and timings.

The data file itself is never embedded (privacy); only its fingerprints and size are. The
``passport_sha256`` field protects the file against accidental or naive edits; it is not a
signature, since anyone can recompute it. A passport shows identity and provenance, not correctness.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

from pydantic import ValidationError

from .canonical import SERIALIZER_VERSION, canonical_dumps, canonical_sha256
from .engine import RESULT_SCHEMA_VERSION, AnalysisResult
from .errors import AxiomError, PlanError
from .ingest import DATASET_FORMAT_VERSION, CsvProfile
from .limits import Limits
from .plan import PLAN_SCHEMA_VERSION, POLICY_VERSION, parse_plan_object, plan_sha256, plan_to_dict

PASSPORT_VERSION = "axiom.passport/1"

# Verification vocabulary (Issue #3 decision): kinds and statuses. ``not_run`` is never a pass.
CHECK_KINDS = ("exact", "numeric", "semantic")
CHECK_STATUSES = ("pass", "fail", "not_applicable", "not_run")
# What a passing check is evidence of. Only ``independent_reference`` compares against an
# independent method; the others are integrity, repeatability or internal consistency.
EVIDENCE_TYPES = (
    "integrity",
    "environment",
    "repeatability",
    "cross_environment",
    "internal_consistency",
    "independent_reference",
)

_TOP_LEVEL = {
    "passport_version",
    "identity",
    "inputs",
    "result",
    "environment",
    "execution",
    "verification",
    "passport_sha256",
}


class PassportError(AxiomError):
    """A passport file is malformed or fails an integrity check."""


def check(
    check_id: str, kind: str, evidence: str, status: str, detail: str = "", **extra: Any
) -> dict[str, Any]:
    """One verification entry in the shared vocabulary."""
    if kind not in CHECK_KINDS or status not in CHECK_STATUSES or evidence not in EVIDENCE_TYPES:
        raise ValueError(f"invalid check vocabulary: {kind!r}, {evidence!r}, {status!r}")
    entry = {"id": check_id, "kind": kind, "evidence": evidence, "status": status, "detail": detail}
    entry.update(extra)
    return entry


def limits_to_dict(limits: Limits) -> dict[str, int]:
    return dataclasses.asdict(limits)


def limits_from_dict(data: Any) -> Limits:
    names = {f.name for f in dataclasses.fields(Limits)}
    if not isinstance(data, dict) or set(data) != names:
        raise PassportError("invalid_passport", "limits must list exactly the Limits fields")
    if not all(type(v) is int and v > 0 for v in data.values()):
        raise PassportError("invalid_passport", "limits must be positive integers")
    return Limits(**data)


def profile_to_dict(profile: CsvProfile) -> dict[str, Any]:
    return profile.model_dump(mode="json")


def profile_from_dict(data: Any) -> CsvProfile:
    try:
        return CsvProfile.model_validate_json(json.dumps(data), strict=True)
    except (ValidationError, TypeError, ValueError) as exc:
        raise PassportError("invalid_passport", "CSV profile in passport is invalid") from exc


def creation_checks(result_core: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks recorded when a passport is created. Replay and cross-environment are ``not_run``."""
    checks = [
        check(
            "plan_digest_recomputed",
            "exact",
            "integrity",
            "pass",
            "engine recomputed the plan digest at binding and execution",
        ),
        check(
            "result_digest_recomputed",
            "exact",
            "integrity",
            "pass",
            "result_sha256 is the SHA-256 of the canonical result bytes",
        ),
    ]
    if result_core.get("operation") == "cluster":
        checks.append(
            check(
                "cluster_postconditions",
                "numeric",
                "internal_consistency",
                "pass",
                "independent recomputation agreed with the solver (ADR 0006); a local solution, "
                "not a global optimum",
            )
        )
    checks += [
        check("fresh_process_replay", "exact", "repeatability", "not_run", "use axiom replay"),
        check(
            "cross_environment_agreement",
            "numeric",
            "cross_environment",
            "not_run",
            "compare with a passport from another environment (axiom compare)",
        ),
        check(
            "independent_reference",
            "numeric",
            "independent_reference",
            "not_run",
            "a passport does not compare its result with an independent oracle; see the test "
            "suite and EXP-001/EXP-002 for that evidence",
        ),
    ]
    return checks


def build_passport(
    *,
    result: AnalysisResult,
    profile: CsvProfile,
    limits: Limits,
    n_bytes: int,
    environment: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a passport from an executed result. Identities are taken from the result core."""
    core = result.core
    inputs = core["inputs"]
    passport: dict[str, Any] = {
        "passport_version": PASSPORT_VERSION,
        "identity": {
            "raw_sha256": inputs["raw_sha256"],
            "dataset_sha256": inputs["dataset_sha256"],
            "plan_sha256": inputs["plan_sha256"],
            "result_sha256": result.result_sha256,
            "dataset_format": DATASET_FORMAT_VERSION,
            "plan_schema": PLAN_SCHEMA_VERSION,
            "policy_version": core["policy_version"],
            "result_schema": core["schema_version"],
            "serializer": core["serializer"],
        },
        "inputs": {
            "csv_profile": profile_to_dict(profile),
            "n_bytes": n_bytes,
            "n_rows": inputs["n_rows"],
            "plan": result.core["plan"],  # an independent copy, never an alias of the core
            "limits": limits_to_dict(limits),
        },
        "result": core,
        "environment": copy.deepcopy(environment),
        "execution": copy.deepcopy(execution),
        "verification": creation_checks(core),
    }
    passport["passport_sha256"] = passport_digest(passport)
    return passport


def passport_digest(passport: dict[str, Any]) -> str:
    """SHA-256 of the canonical passport without its own ``passport_sha256`` field."""
    body = {k: v for k, v in passport.items() if k != "passport_sha256"}
    return canonical_sha256(body)


def dumps_passport(passport: dict[str, Any]) -> str:
    """Human-readable JSON text (sorted keys, two-space indent). Floats keep full precision."""
    return (
        json.dumps(passport, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise PassportError("invalid_passport", "duplicate key in passport JSON")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise PassportError("invalid_passport", f"{name} is not allowed in passport JSON")


def loads_passport(text: str) -> dict[str, Any]:
    """Parse passport JSON strictly (no duplicate keys, no NaN/Infinity) and check its shape."""
    try:
        data = json.loads(
            text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise PassportError("invalid_passport", "passport is not valid JSON") from exc
    except (RecursionError, ValueError) as exc:
        raise PassportError(
            "invalid_passport", f"passport rejected ({type(exc).__name__})"
        ) from exc
    if not isinstance(data, dict) or set(data) != _TOP_LEVEL:
        raise PassportError("invalid_passport", "passport has missing or unknown top-level fields")
    if data["passport_version"] != PASSPORT_VERSION:
        raise PassportError(
            "unsupported_passport_version",
            "passport version is not supported",
            found=data["passport_version"],
        )
    for key in ("identity", "inputs", "result", "environment", "execution"):
        if not isinstance(data[key], dict):
            raise PassportError("invalid_passport", f"passport field {key!r} must be an object")
    return data


def integrity_checks(passport: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute every digest a passport states, without the data file.

    Returns one check per item; any ``fail`` means the passport was altered or is inconsistent,
    which is an integrity mismatch, not a numerical discrepancy.
    """
    out: list[dict[str, Any]] = []

    def add(check_id: str, ok: bool, detail: str) -> None:
        status = "pass" if ok else "fail"
        out.append(check(check_id, "exact", "integrity", status, detail))

    add(
        "passport_digest",
        passport.get("passport_sha256") == passport_digest(passport),
        "passport_sha256 equals the digest of the canonical passport body",
    )
    identity, inputs, core = passport["identity"], passport["inputs"], passport["result"]
    try:
        result_ok = canonical_sha256(core) == identity.get("result_sha256")
    except (TypeError, ValueError):
        result_ok = False
    add("result_digest", result_ok, "embedded result core hashes to identity.result_sha256")
    try:
        plan = parse_plan_object(inputs.get("plan"))
        plan_ok = (
            plan_sha256(plan) == identity.get("plan_sha256")
            and plan_to_dict(plan) == inputs.get("plan")
            and identity.get("policy_version") == POLICY_VERSION
        )
    except PlanError:
        plan_ok = False
    add("plan_digest", plan_ok, "embedded plan is valid, explicit and hashes to plan_sha256")
    core_inputs = core.get("inputs", {}) if isinstance(core.get("inputs"), dict) else {}
    consistent = (
        core_inputs.get("raw_sha256") == identity.get("raw_sha256")
        and core_inputs.get("dataset_sha256") == identity.get("dataset_sha256")
        and core_inputs.get("plan_sha256") == identity.get("plan_sha256")
        and core.get("plan") == inputs.get("plan")
        and core.get("schema_version") == RESULT_SCHEMA_VERSION
        and core.get("serializer") == SERIALIZER_VERSION
    )
    add("identity_consistency", consistent, "identities agree with those inside the result core")
    try:
        profile_from_dict(inputs.get("csv_profile"))
        limits_from_dict(inputs.get("limits"))
        inputs_ok = True
    except PassportError:
        inputs_ok = False
    add("replay_inputs_valid", inputs_ok, "CSV profile and limits can be reconstructed")
    return out


def result_from_passport(passport: dict[str, Any]) -> AnalysisResult:
    return AnalysisResult(canonical_dumps(passport["result"]))
