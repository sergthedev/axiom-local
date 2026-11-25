"""Canonical JSON serializer v1 and SHA-256 helpers.

Format ``axiom.canonical-json/1``: UTF-8, no insignificant whitespace, object keys sorted by Unicode
code point, ``ensure_ascii=False``, only finite numbers, negative zero written as ``0.0``, floats
written with Python ``repr`` (shortest round-trip, full precision, never rounded). This is *not*
RFC 8785 (JCS): number formatting differs (for example ``1e+16``), and no conformance is claimed.
A hash of this output establishes identity and integrity only, never correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

SERIALIZER_VERSION = "axiom.canonical-json/1"


def _prepare(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite float cannot be serialized")
        return obj + 0.0  # -0.0 + 0.0 == 0.0 (canonical signed zero)
    if isinstance(obj, (list, tuple)):
        return [_prepare(v) for v in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError("object keys must be strings")
            out[key] = _prepare(value)
        return out
    raise TypeError(f"type {type(obj).__name__} is not serializable")


def canonical_dumps(obj: Any) -> str:
    """Serialize ``obj`` to canonical JSON text (see module docstring)."""
    return json.dumps(
        _prepare(obj),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def canonical_bytes(obj: Any) -> bytes:
    return canonical_dumps(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))
