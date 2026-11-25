"""Helpers shared by the operation modules."""

from __future__ import annotations

from typing import Any

from .kernels import describe
from .limits import Limits


def warn(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def valid_values(values: tuple) -> list:
    return [v for v in values if v is not None]


def capped_ordinals(ordinals: list[int], limits: Limits) -> dict[str, Any]:
    cap = limits.max_reported_ordinals
    return {"row_ordinals": ordinals[:cap], "row_ordinals_truncated": len(ordinals) > cap}


def stats_block(values: list[float]) -> dict[str, Any]:
    """Shared descriptive kernel output used by Inspect and Compare."""
    stats, undefined = describe(values)
    return {"stats": stats, "undefined": undefined}
