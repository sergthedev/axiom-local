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
import math
from typing import Any

from pydantic import ValidationError

from .canonical import SERIALIZER_VERSION, canonical_dumps, canonical_sha256
from .engine import RESULT_SCHEMA_VERSION, AnalysisResult
from .errors import PlanError
from .ingest import DATASET_FORMAT_VERSION, CsvProfile
from .limits import Limits
from .passport_schema import (
    CHECK_KINDS,
    CHECK_STATUSES,
    EVIDENCE_TYPES,
    PassportError,
    passport_spec,
    validate_document,
)
from .plan import PLAN_SCHEMA_VERSION, POLICY_VERSION, parse_plan_object, plan_sha256, plan_to_dict

PASSPORT_VERSION = "axiom.passport/1"

# Upper bound on passport text read or parsed (review B1). A passport embeds the full result core;
# the largest one the default limits allow (K-means on 1,000,000 rows: two per-row integer lists,
# pretty-printed) is about 30 MB, so 64 MiB leaves a margin. Larger text is rejected unparsed.
MAX_PASSPORT_BYTES = 64 * 1024 * 1024


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
    return validate_passport(passport)  # never issue a passport that would be rejected on reading


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


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e999, which float() turns into infinity
        raise PassportError("invalid_passport", "number out of binary64 range in passport JSON")
    return value


def validate_passport(passport: Any) -> dict[str, Any]:
    """Check the complete structure of a passport object (strict; see :mod:`.passport_schema`).

    Raises :class:`PassportError` with a stable code: ``unsupported_passport_version`` or
    ``invalid_passport`` (naming the first offending field). Shape only: cross-field consistency
    and digests are :func:`integrity_checks`. Passing this is not authentication.
    """
    if isinstance(passport, dict) and "passport_version" in passport:
        version = passport["passport_version"]
        if version != PASSPORT_VERSION:
            raise PassportError(
                "unsupported_passport_version",
                "passport version is not supported",
                found=version[:64] if isinstance(version, str) else None,
            )
    validate_document(passport, passport_spec(PASSPORT_VERSION), "passport")
    return passport


def loads_passport(text: str | bytes, *, max_bytes: int = MAX_PASSPORT_BYTES) -> dict[str, Any]:
    """Parse passport JSON strictly and validate its complete structure.

    Text longer than ``max_bytes`` (UTF-8) is rejected before parsing. Duplicate keys, ``NaN``,
    ``Infinity`` and numbers outside the binary64 range are rejected.
    """
    if isinstance(text, str):
        try:
            raw = text.encode("utf-8")
        except UnicodeEncodeError as exc:  # unpaired surrogate in the text itself
            raise PassportError("invalid_passport", "passport text is not valid Unicode") from exc
    elif isinstance(text, (bytes, bytearray)):
        raw = bytes(text)
    else:
        raise PassportError("invalid_passport", "passport text must be str or bytes")
    if len(raw) > max_bytes:
        raise PassportError(
            "passport_too_large", "passport exceeds the size limit", limit=max_bytes
        )
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except PassportError:
        raise
    except json.JSONDecodeError as exc:
        raise PassportError("invalid_passport", "passport is not valid JSON") from exc
    except (RecursionError, ValueError) as exc:  # includes UnicodeDecodeError, huge integers
        raise PassportError(
            "invalid_passport", f"passport rejected ({type(exc).__name__})"
        ) from exc
    return validate_passport(data)


def integrity_checks(passport: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute every digest a passport states and check its internal consistency, without data.

    The structure is validated first (:func:`validate_passport`; raises :class:`PassportError`).
    Returns one check per item; any ``fail`` means the passport was altered or is inconsistent,
    which is an integrity mismatch, not a numerical discrepancy. Passing every check is not
    authentication: a consistently re-sealed forgery passes and is exposed only by recomputation.
    """
    validate_passport(passport)
    out: list[dict[str, Any]] = []

    def add(check_id: str, ok: bool, detail: str) -> None:
        status = "pass" if ok else "fail"
        out.append(check(check_id, "exact", "integrity", status, detail))

    add(
        "passport_digest",
        passport["passport_sha256"] == passport_digest(passport),
        "passport_sha256 equals the digest of the canonical passport body",
    )
    identity, inputs, core = passport["identity"], passport["inputs"], passport["result"]
    add(
        "result_digest",
        canonical_sha256(core) == identity["result_sha256"],
        "embedded result core hashes to identity.result_sha256",
    )
    try:
        plan = parse_plan_object(inputs["plan"])
        plan_ok = (
            plan_sha256(plan) == identity["plan_sha256"]
            and plan_to_dict(plan) == inputs["plan"]
            and plan.operation == core["operation"]
        )
    except PlanError:
        plan_ok = False
    validated_plan = inputs["plan"] if plan_ok else None
    add(
        "plan_digest",
        plan_ok,
        "embedded plan is valid, explicit, hashes to plan_sha256 and names the result's operation",
    )
    core_inputs = core["inputs"]
    consistent = (
        core_inputs["raw_sha256"] == identity["raw_sha256"]
        and core_inputs["dataset_sha256"] == identity["dataset_sha256"]
        and core_inputs["plan_sha256"] == identity["plan_sha256"]
        and core_inputs["n_rows"] == inputs["n_rows"]
        and core["plan"] == inputs["plan"]
        and core["policy_version"] == identity["policy_version"]
        and core["schema_version"] == identity["result_schema"] == RESULT_SCHEMA_VERSION
        and core["serializer"] == identity["serializer"] == SERIALIZER_VERSION
        and identity["dataset_format"] == DATASET_FORMAT_VERSION
        and identity["plan_schema"] == PLAN_SCHEMA_VERSION
        and identity["policy_version"] == POLICY_VERSION
    )
    add(
        "identity_consistency",
        consistent,
        "identities, versions and row count agree with those inside the result core",
    )
    try:
        profile_from_dict(inputs["csv_profile"])
        limits = limits_from_dict(inputs["limits"])
    except PassportError:
        limits = None
    add("replay_inputs_valid", limits is not None, "CSV profile and limits can be reconstructed")
    detail = (
        "counts, row ordinals, plan fields and cluster sizes in the result agree with each other "
        "and with the validated plan and recorded limits (internal consistency of the record, not "
        "correctness of the numbers)"
    )
    if validated_plan is None or limits is None:
        # Nothing in the result is checked against an unvalidated plan or limits (final audit):
        # the passport already fails plan_digest or replay_inputs_valid.
        missing = [name for name, v in (("plan", validated_plan), ("limits", limits)) if v is None]
        out.append(
            check(
                "result_consistency",
                "exact",
                "integrity",
                "not_run",
                detail,
                problems=[f"not checked: invalid {' and '.join(missing)}"],
            )
        )
    else:
        problems = result_problems(core, limits, validated_plan)
        out.append(
            check(
                "result_consistency",
                "exact",
                "integrity",
                "fail" if problems else "pass",
                detail,
                problems=problems[:10],
            )
        )
    add(
        "verification_record",
        passport["verification"] == creation_checks(core),
        "recorded verification is exactly what axiom.passport/1 records at creation; it is a "
        "historical record and was not performed again here",
    )
    return out


def _label(name: str) -> str:
    """A column or group name for a problem message, shortened so reports stay bounded."""
    text = repr(name)
    return text if len(text) <= 40 else text[:37] + "..."


def _stats_consistent(entry: dict[str, Any]) -> bool:
    """A statistic is either a number or has an undefined reason, never both or neither."""
    stats, undefined = entry["stats"], entry["undefined"]
    return set(undefined) == {k for k, v in stats.items() if v is None}


def _ordinals_consistent(ordinals: list[int], truncated: bool, n: int, cap: int) -> bool:
    return (
        len(ordinals) == min(n, cap)
        and truncated == (n > cap)
        and ordinals == sorted(set(ordinals))
        and all(o >= 1 for o in ordinals)
    )


def result_problems(
    core: dict[str, Any], limits: Limits, plan: dict[str, Any] | None = None
) -> list[str]:
    """Internal-consistency problems of a structurally valid result core (empty if none).

    ``plan`` is the validated plan (default: the core's own plan); ``limits`` are the recorded
    limits. Every loop and allocation here is linear in the size of the document: no count stated
    in the record (such as the cluster count ``k``) is used to size or drive any work (review B1).
    """
    op, res = core["operation"], core["result"]
    plan = core["plan"] if plan is None else plan
    n_rows = core["inputs"]["n_rows"]
    cap = limits.max_reported_ordinals
    kinds = {c["name"]: c["kind"] for c in core["inputs"]["columns"]}
    p: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            p.append(what)

    if op == "inspect":
        need(res["n_rows"] == n_rows, "n_rows")
        need([c["name"] for c in res["columns"]] == plan.get("columns"), "columns != plan")
        for c in res["columns"]:
            name = _label(c["name"])
            need(c["n_total"] == n_rows, f"column {name} n_total")
            need(c["n_valid"] + c["n_missing"] == c["n_total"], f"column {name} counts")
            need(kinds.get(c["name"]) == c["kind"], f"column {name} kind")
            if c["kind"] == "numeric":
                need(_stats_consistent(c), f"column {name} stats/undefined")
    elif op == "relate":
        for key in ("x", "y", "method"):
            need(res[key] == plan.get(key), f"{key} != plan")
        need(res["n_rows_total"] == n_rows, "n_rows_total")
        need(res["n_used"] + res["n_excluded"] == n_rows, "n_used + n_excluded")
        need(
            res["excluded_missing_x_only"]
            + res["excluded_missing_y_only"]
            + res["excluded_missing_both"]
            == res["n_excluded"],
            "excluded counts",
        )
        ex = res["excluded"]
        need(
            _ordinals_consistent(
                ex["row_ordinals"], ex["row_ordinals_truncated"], res["n_excluded"], cap
            ),
            "excluded row ordinals",
        )
        need((res["coefficient"] is None) != (res["undefined"] is None), "coefficient/undefined")
        need(res["coefficient"] is None or abs(res["coefficient"]) <= 1.0, "coefficient range")
    elif op == "compare":
        need(res["group_by"] == plan.get("group_by"), "group_by != plan")
        need(res["measure"] == plan.get("measure"), "measure != plan")
        need(res["n_rows_total"] == n_rows, "n_rows_total")
        need(res["n_groups"] == len(res["groups"]), "n_groups")
        keys = [g["key"] for g in res["groups"]]
        need(keys == sorted(set(keys)), "group keys not unique and ordered")
        need(
            sum(g["n_total"] for g in res["groups"]) + res["n_missing_group_key"] == n_rows,
            "group totals",
        )
        for g in res["groups"]:
            need(g["n_valid"] + g["n_missing"] == g["n_total"], f"group {_label(g['key'])} counts")
            need(_stats_consistent(g), f"group {_label(g['key'])} stats/undefined")
        mk = res["missing_group_key"]
        need(
            _ordinals_consistent(
                mk["row_ordinals"], mk["row_ordinals_truncated"], res["n_missing_group_key"], cap
            ),
            "missing group key ordinals",
        )
    elif op == "cluster":
        features, k, clusters = res["features"], res["k"], res["clusters"]
        need(features == plan.get("features"), "features != plan")
        need(k == plan.get("k"), "k != plan")
        need(k <= limits.max_k, "k exceeds the recorded max_k")
        need(len(clusters) == k, "number of clusters != k")
        need(res["scaler"]["kind"] == plan.get("scaling"), "scaler kind != plan")
        solver = res["solver"]
        for key in ("init", "algorithm", "seed", "n_init", "max_iter", "tol"):
            need(solver[key] == plan.get(key), f"solver {key} != plan")
        need(res["n_rows_total"] == n_rows, "n_rows_total")
        need(res["n_eligible"] + res["n_excluded"] == n_rows, "n_eligible + n_excluded")
        need(
            _ordinals_consistent(
                res["excluded_row_ordinals"],
                res["excluded_row_ordinals_truncated"],
                res["n_excluded"],
                cap,
            ),
            "excluded row ordinals",
        )
        eligible, labels = res["eligible_row_ordinals"], res["assignments"]
        need(len(eligible) == len(labels) == res["n_eligible"], "assignment count")
        need(eligible == sorted(set(eligible)) and all(o >= 1 for o in eligible), "eligible rows")
        # Ids, labels and sizes are compared over the entries that exist, never over range(k).
        n_clusters = len(clusters)
        need(all(c["id"] == i for i, c in enumerate(clusters)), "cluster ids")
        need(all(label < n_clusters for label in labels), "assignment labels")  # labels are NAT
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        need(all(c["size"] == counts.get(c["id"], 0) for c in clusters), "cluster sizes")
        # The engine never issues an empty cluster or a failed postcondition (ADR 0006).
        need(all(c["size"] >= 1 for c in clusters), "empty cluster")
        need(all(res["postconditions"].values()), "recorded postcondition false")
        width, feature_set = len(features), set(features)
        need(len(res["scaler"]["mean"]) == len(res["scaler"]["scale"]) == width, "scaler width")
        for i, c in enumerate(clusters):
            need(len(c["centroid_standardized"]) == width, f"cluster {i} centroid width")
            need(c["original_units"].keys() == feature_set, f"cluster {i} features")
    return p


def result_from_passport(passport: dict[str, Any]) -> AnalysisResult:
    return AnalysisResult(canonical_dumps(passport["result"]))
