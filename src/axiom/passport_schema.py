"""Strict structural validation of ``axiom.passport/1`` documents (ADR 0007, review B2).

A passport is untrusted input: it may come from another machine or have been edited. Before any
digest is recomputed or any value is compared, :func:`validate_passport` checks the complete shape:
exact keys, JSON types, finite numbers, bounded nesting, the verification vocabulary, the format
versions this code supports, digest syntax, and the full result block of each operation.

This is a shape check, not authentication. A structurally valid passport can still be forged
consistently (every digest recomputed); only recomputation from the data exposes that. Cross-field
consistency (row counts, plan agreement, recorded verification) is checked separately by
:func:`axiom.passport.integrity_checks` and reported as an integrity failure.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from .canonical import SERIALIZER_VERSION
from .engine import RESULT_SCHEMA_VERSION
from .environment import ENVIRONMENT_SCHEMA_VERSION
from .errors import AxiomError
from .ingest import DATASET_FORMAT_VERSION
from .kernels import DESCRIPTIVE_KEYS
from .plan import PLAN_SCHEMA_VERSION, POLICY_VERSION

# Deepest nesting a valid passport reaches is 8 (cluster original-unit statistics); 32 leaves room
# without allowing pathological documents.
MAX_DEPTH = 32
# Every integer in a passport is a count, a size, a seed, an index or a limit; all fit in a signed
# 64-bit integer (the NumPy index type). Larger integers are rejected before any arithmetic,
# hashing or text conversion sees them: Python refuses to convert integers of more than 4300
# digits to text, so hashing one would raise ``ValueError`` (final audit).
INT_MIN, INT_MAX = -(2**63), 2**63 - 1
# Unpaired UTF-16 surrogates are valid in a Python ``str`` (and JSON ``\ud800`` escapes decode to
# them) but have no UTF-8 encoding, so canonical hashing would raise ``UnicodeEncodeError``.
_SURROGATE = re.compile("[\ud800-\udfff]")
# Values and path components quoted in error messages are shortened to this many characters, so an
# error report does not echo (a multiple of) the rejected document.
_QUOTE_CHARS = 64

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX32 = re.compile(r"[0-9a-f]{32}")


class PassportError(AxiomError):
    """A passport file is malformed or fails an integrity check."""


class _Invalid(Exception):
    def __init__(self, path: tuple[Any, ...], message: str) -> None:
        super().__init__(message)
        self.path = path
        self.message = message


Spec = Any  # a type, a dict of exact keys, or one of the validator callables below
Check = Callable[[Any, tuple[Any, ...]], None]


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _quote(value: Any) -> str:
    """Bounded ``repr`` for error messages (strings and scalars only; containers by type)."""
    if isinstance(value, (dict, list)):
        return f"<{_type_name(value)}>"
    if isinstance(value, int) and not isinstance(value, bool) and not INT_MIN <= value <= INT_MAX:
        return "<integer out of range>"
    text = repr(value)
    return text if len(text) <= _QUOTE_CHARS else text[: _QUOTE_CHARS - 3] + "..."


def _validate(value: Any, spec: Spec, path: tuple[Any, ...]) -> None:
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            raise _Invalid(path, f"expected an object, found {_type_name(value)}")
        if set(value) != set(spec):
            missing = sorted(set(spec) - set(value))
            unknown = sorted(set(value) - set(spec))
            raise _Invalid(path, f"keys differ (missing {missing}, unknown {unknown})")
        for key, sub in spec.items():
            _validate(value[key], sub, (*path, key))
        return
    if callable(spec) and not isinstance(spec, type):
        spec(value, path)
        return
    raise TypeError(f"bad spec {spec!r}")  # programming error, not input error


def _scalar(kind: str) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        ok = {
            "str": isinstance(value, str),
            "bool": isinstance(value, bool),
            "int": type(value) is int,
            "nat": type(value) is int and value >= 0,
            "float": type(value) is float and math.isfinite(value),
        }[kind]
        if not ok:
            raise _Invalid(path, f"expected {kind}, found {_type_name(value)}")

    return check


STR, BOOL, INT, NAT, FLOAT = (_scalar(k) for k in ("str", "bool", "int", "nat", "float"))


def const(expected: Any) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if type(value) is not type(expected) or value != expected:
            raise _Invalid(path, f"unsupported value {_quote(value)} (expected {expected!r})")

    return check


def enum(*allowed: str) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if not isinstance(value, str) or value not in allowed:
            raise _Invalid(path, f"value {_quote(value)} is not one of {list(allowed)}")

    return check


def pattern(regex: re.Pattern[str], what: str) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if not isinstance(value, str) or not regex.fullmatch(value):
            raise _Invalid(path, f"expected {what}")

    return check


HEX64 = pattern(_HEX64, "a lowercase hexadecimal SHA-256 digest")
HEX32 = pattern(_HEX32, "a 32-digit lowercase hexadecimal run id")


def optional(spec: Spec) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if value is not None:
            _validate(value, spec, path)

    return check


def list_of(spec: Spec) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if not isinstance(value, list):
            raise _Invalid(path, f"expected a list, found {_type_name(value)}")
        for i, item in enumerate(value):
            _validate(item, spec, (*path, i))

    return check


def map_of(key: Check, spec: Spec) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if not isinstance(value, dict):
            raise _Invalid(path, f"expected an object, found {_type_name(value)}")
        for k, v in value.items():
            key(k, (*path, k))
            _validate(v, spec, (*path, k))

    return check


def one_of(selector: str, variants: dict[str, Spec]) -> Check:
    """Object whose exact shape depends on the value of its ``selector`` field."""

    def check(value: Any, path: tuple[Any, ...]) -> None:
        tag = value.get(selector) if isinstance(value, dict) else None
        if not isinstance(tag, str) or tag not in variants:
            raise _Invalid((*path, selector), f"unknown or missing {selector!r}")
        _validate(value, variants[tag], path)

    return check


def json_value(value: Any, path: tuple[Any, ...]) -> None:
    """Any JSON value (finiteness and depth are checked over the whole document)."""


def unavailable_or(spec: Spec) -> Check:
    def check(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict) and value.get("status") == "unavailable":
            _validate(value, UNAVAILABLE, path)
        else:
            _validate(value, spec, path)

    return check


UNAVAILABLE = {"status": const("unavailable"), "reason": STR}

# --- result block of each operation (engine output, ADR 0004-0006) --------------------------------

STATS = {key: optional(FLOAT) for key in DESCRIPTIVE_KEYS}
UNDEFINED_STATS = map_of(enum(*DESCRIPTIVE_KEYS), STR)
ORDINALS = {"row_ordinals": list_of(NAT), "row_ordinals_truncated": BOOL}
COUNTS = {"n_total": NAT, "n_valid": NAT, "n_missing": NAT}

INSPECT_RESULT = {
    "n_rows": NAT,
    "n_blank_lines": NAT,
    "n_all_empty_records": NAT,
    "columns": list_of(
        one_of(
            "kind",
            {
                "numeric": {
                    "name": STR,
                    "kind": const("numeric"),
                    **COUNTS,
                    "stats": STATS,
                    "undefined": UNDEFINED_STATS,
                },
                "text": {"name": STR, "kind": const("text"), **COUNTS},
            },
        )
    ),
}
RELATE_RESULT = {
    "x": STR,
    "y": STR,
    "method": enum("pearson", "spearman"),
    "n_rows_total": NAT,
    "n_used": NAT,
    "n_excluded": NAT,
    "excluded_missing_x_only": NAT,
    "excluded_missing_y_only": NAT,
    "excluded_missing_both": NAT,
    "excluded": ORDINALS,
    "coefficient": optional(FLOAT),
    "undefined": optional({"code": STR, "columns": list_of(enum("x", "y"))}),
    "minimum_n": NAT,
}
COMPARE_RESULT = {
    "group_by": STR,
    "measure": STR,
    "n_rows_total": NAT,
    "n_missing_group_key": NAT,
    "missing_group_key": ORDINALS,
    "n_groups": NAT,
    "groups": list_of(
        {"key": STR, **COUNTS, "stats": STATS, "undefined": UNDEFINED_STATS},
    ),
}
CLUSTER_RESULT = {
    "features": list_of(STR),
    "k": NAT,
    "n_rows_total": NAT,
    "n_eligible": NAT,
    "n_excluded": NAT,
    "excluded_row_ordinals": list_of(NAT),
    "excluded_row_ordinals_truncated": BOOL,
    "scaler": {"kind": STR, "mean": list_of(FLOAT), "scale": list_of(FLOAT)},
    "solver": {
        "init": STR,
        "algorithm": STR,
        "seed": NAT,
        "n_init": NAT,
        "max_iter": NAT,
        "tol": FLOAT,
        "copy_x": BOOL,
        "threadpool_limit": NAT,
        "n_iter_diagnostic": NAT,
    },
    "inertia_standardized": FLOAT,
    "clusters": list_of(
        {
            "id": NAT,
            "size": NAT,
            "centroid_standardized": list_of(FLOAT),
            "original_units": map_of(STR, {"mean": FLOAT, "min": FLOAT, "max": FLOAT}),
        }
    ),
    "eligible_row_ordinals": list_of(NAT),
    "assignments": list_of(NAT),
    "postconditions": {
        "nearest_center": BOOL,
        "nonempty_clusters": BOOL,
        "centroid_consistency_approx": BOOL,
        "inertia_consistent": BOOL,
        "scaler_consistent": BOOL,
    },
    "diagnostics": {"lloyd_label_fixed_point": BOOL},
}
OPERATION_RESULTS = {
    "inspect": INSPECT_RESULT,
    "relate": RELATE_RESULT,
    "compare": COMPARE_RESULT,
    "cluster": CLUSTER_RESULT,
}


def _core_variant(operation: str) -> dict[str, Spec]:
    return {
        "schema_version": const(RESULT_SCHEMA_VERSION),
        "serializer": const(SERIALIZER_VERSION),
        "policy_version": const(POLICY_VERSION),
        "operation": const(operation),
        "inputs": {
            "raw_sha256": HEX64,
            "dataset_sha256": HEX64,
            "plan_sha256": HEX64,
            "n_rows": NAT,
            "columns": list_of({"name": STR, "kind": enum("numeric", "text")}),
        },
        "plan": map_of(STR, json_value),  # validated by the plan parser in integrity_checks
        "result": OPERATION_RESULTS[operation],
        "warnings": list_of({"code": STR, "detail": STR}),
    }


RESULT_CORE = one_of("operation", {op: _core_variant(op) for op in OPERATION_RESULTS})

# --- recorded metadata ----------------------------------------------------------------------------

CHECK_KINDS = ("exact", "numeric", "semantic")
CHECK_STATUSES = ("pass", "fail", "not_applicable", "not_run")
EVIDENCE_TYPES = (
    "integrity",
    "environment",
    "repeatability",
    "cross_environment",
    "internal_consistency",
    "independent_reference",
)
VERIFICATION_ENTRY = {
    "id": STR,
    "kind": enum(*CHECK_KINDS),
    "evidence": enum(*EVIDENCE_TYPES),
    "status": enum(*CHECK_STATUSES),
    "detail": STR,
}

IDENTITY = {
    "raw_sha256": HEX64,
    "dataset_sha256": HEX64,
    "plan_sha256": HEX64,
    "result_sha256": HEX64,
    "dataset_format": const(DATASET_FORMAT_VERSION),
    "plan_schema": const(PLAN_SCHEMA_VERSION),
    "policy_version": const(POLICY_VERSION),
    "result_schema": const(RESULT_SCHEMA_VERSION),
    "serializer": const(SERIALIZER_VERSION),
}
INPUTS = {
    "csv_profile": map_of(STR, json_value),  # validated strictly by profile_from_dict
    "n_bytes": NAT,
    "n_rows": NAT,
    "plan": map_of(STR, json_value),
    "limits": map_of(STR, json_value),  # validated by limits_from_dict
}
_TEXT = unavailable_or(STR)
ENVIRONMENT = {
    "schema_version": const(ENVIRONMENT_SCHEMA_VERSION),
    "python": {
        "version": _TEXT,
        "implementation": _TEXT,
        "compiler": _TEXT,
        "build": list_of(STR),
        "bits": NAT,
        "float_repr_style": STR,
    },
    "os": {
        "system": _TEXT,
        "release": _TEXT,
        "version": _TEXT,
        "platform": _TEXT,
        "machine": _TEXT,
        "processor": _TEXT,
        "logical_cpus": unavailable_or(NAT),
        "byteorder": STR,
    },
    "packages": map_of(STR, _TEXT),
    "code": {
        "package_files_sha256": HEX64,
        "commit": unavailable_or(pattern(re.compile(r"[0-9a-f]{40,64}"), "a Git commit id")),
        "dirty": unavailable_or(BOOL),
        "lock_sha256": unavailable_or(HEX64),
    },
    "numerical_backend": unavailable_or(map_of(STR, json_value)),
    "threads": {
        "environment_variables": map_of(STR, optional(STR)),
        "detected_pools": unavailable_or(list_of(map_of(STR, json_value))),
        "fit_limit_probe": enum("verified_single_thread", "unverified"),
        "blas_thread_control": STR,
    },
}
EXECUTION = {
    "run_id": HEX32,
    "mode": const("fresh_process"),
    "workspace": STR,
    "result_cache": STR,
    "thread_environment": map_of(STR, STR),
    "started_utc": optional(STR),
    "finished_utc": optional(STR),
    "timing_seconds": map_of(STR, optional(FLOAT)),
}


def _scan(value: Any, path: tuple[Any, ...], depth: int) -> None:
    """Whole-document bounds: nesting depth, string keys, finite numbers, 64-bit integers and
    UTF-8-encodable text only. Linear in the size of the document."""
    if depth > MAX_DEPTH:
        raise _Invalid(path, f"nesting deeper than {MAX_DEPTH}")
    if isinstance(value, float) and not math.isfinite(value):
        raise _Invalid(path, "non-finite number")
    if isinstance(value, int) and not isinstance(value, bool) and not INT_MIN <= value <= INT_MAX:
        raise _Invalid(path, "integer outside the signed 64-bit range")
    if isinstance(value, str) and _SURROGATE.search(value):
        raise _Invalid(path, "string contains an unpaired surrogate (not encodable as UTF-8)")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise _Invalid(path, "object keys must be strings")
            if _SURROGATE.search(k):
                raise _Invalid(path, "key contains an unpaired surrogate (not encodable as UTF-8)")
            _scan(v, (*path, k), depth + 1)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _scan(v, (*path, i), depth + 1)
    elif not (value is None or isinstance(value, (str, int, float))):
        raise _Invalid(path, f"not a JSON value ({type(value).__name__})")


def passport_spec(passport_version: str) -> dict[str, Spec]:
    return {
        "passport_version": const(passport_version),
        "identity": IDENTITY,
        "inputs": INPUTS,
        "result": RESULT_CORE,
        "environment": ENVIRONMENT,
        "execution": EXECUTION,
        "verification": list_of(VERIFICATION_ENTRY),
        "passport_sha256": HEX64,
    }


def _short(part: Any) -> str:
    text = part if isinstance(part, str) else str(part)  # path parts are keys or list indices
    return text if len(text) <= _QUOTE_CHARS else text[: _QUOTE_CHARS - 3] + "..."


def validate_document(document: Any, spec: dict[str, Spec], what: str) -> None:
    """Raise :class:`PassportError` (``invalid_passport``) naming the first offending field."""
    try:
        _scan(document, (), 0)
        _validate(document, spec, ())
    except _Invalid as exc:
        where = "/".join(_short(p) for p in exc.path) or "(root)"
        raise PassportError(
            "invalid_passport", f"{what} is malformed at {where}: {exc.message}", field=where
        ) from None
    except RecursionError:
        raise PassportError("invalid_passport", f"{what} is nested too deeply") from None
