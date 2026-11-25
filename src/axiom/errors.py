"""Typed, structured errors. Every failure mode of the engine is an :class:`AxiomError`."""

from __future__ import annotations

from typing import Any


class AxiomError(Exception):
    """Base class. ``code`` is a stable machine-readable identifier."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class InputError(AxiomError):
    """The CSV bytes, the CSV profile or the schema were rejected."""


class PlanError(AxiomError):
    """An analysis plan is malformed, semantically invalid, or not bound to the dataset."""


class ComputationError(AxiomError):
    """A computation could not be completed or failed a postcondition (fail-closed)."""
