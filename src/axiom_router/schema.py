"""Schema projection: what a router may see about a dataset (ADR 0009).

A router receives only the ordered column names, their trusted kinds and schema-local identifiers
``c1..cN``. Never rows, sample values, file names, row counts or descriptions. Column names are
untrusted text: they may contain instructions, markup, control characters or bidirectional
overrides. They are passed to a model as JSON string data and are shown to a person only through
:func:`display_name`, which makes invisible and control characters visible.

Matching helpers (:func:`match_key`, :func:`loose_key`) are used to *suggest* and to detect
collisions. They are never used to silently resolve a column: the exact identifier is what binds.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from axiom.canonical import canonical_sha256
from axiom.ingest import Dataset

SCHEMA_VIEW_VERSION = "axiom.schema-view/1"
MAX_PROMPT_COLUMNS = (
    64  # provisional prompt budget; wider schemas go to the manual path (no silent cut)
)

ColumnKind = Literal["numeric", "text"]

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_DEASCII = str.maketrans(
    {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c", "â": "a", "î": "i", "û": "u"}
)
_WORD = re.compile(r"[^\W_]+")
_CAMEL = re.compile(r"(?<=[a-zçğıöşü0-9])(?=[A-ZÇĞİÖŞÜ])")


@dataclass(frozen=True, slots=True)
class ProjectedColumn:
    id: str
    name: str
    kind: ColumnKind


@dataclass(frozen=True, slots=True)
class SchemaView:
    """The model-facing projection of one dataset schema and its digest.

    ``collisions`` lists groups of column ids whose names are equal under :func:`loose_key`
    (for example ``Gelir`` and ``gelir`` or ``ücret`` and ``ucret``). A plan that uses any of them
    needs explicit disambiguation (semantic validation reports ``column_collision``).
    """

    columns: tuple[ProjectedColumn, ...]
    schema_sha256: str
    collisions: tuple[tuple[str, ...], ...]

    def by_id(self, column_id: str) -> ProjectedColumn | None:
        for col in self.columns:
            if col.id == column_id:
                return col
        return None

    def to_prompt_payload(self) -> dict[str, Any]:
        """Exactly what a router may receive about the dataset."""
        return {
            "version": SCHEMA_VIEW_VERSION,
            "columns": [{"id": c.id, "name": c.name, "type": c.kind} for c in self.columns],
        }


def turkish_lower(text: str) -> str:
    """NFC, then Turkish-aware lower case (``I`` -> ``ı``, ``İ`` -> ``i``) for matching only."""
    return unicodedata.normalize("NFC", text).translate(_TR_LOWER).lower()


def deasciify(text: str) -> str:
    """Map Turkish letters with diacritics to ASCII look-alikes (``ş`` -> ``s``; matching only)."""
    return turkish_lower(text).translate(_DEASCII)


def split_identifier(name: str) -> str:
    """Split ``snake_case``, ``camelCase`` and punctuation into space-separated words."""
    return _CAMEL.sub(" ", unicodedata.normalize("NFC", name))


def match_words(text: str) -> list[str]:
    """Words of ``text`` in Turkish-aware lower case, diacritics kept (identifiers are split)."""
    return _WORD.findall(turkish_lower(split_identifier(text)))


def match_key(text: str) -> str:
    """Exact-match key: ``Hane_Geliri`` and ``hane geliri`` give ``hane geliri``."""
    return " ".join(match_words(text))


def loose_key(text: str) -> str:
    """Case, diacritics, separators removed: ``Hane_Geliri`` == ``haneGeliri`` == ``hane geliri``.

    Used for collision detection and suggestions only.
    """
    return "".join(_WORD.findall(deasciify(split_identifier(text))))


def display_name(name: str) -> str:
    """Render an untrusted name for a person: controls, format and bidi characters become visible.

    Characters of the Unicode categories Cc, Cf, Cs, Co, Cn, Zl and Zp (which includes the
    bidirectional overrides and isolates and zero-width characters) are written as ``\\u{XXXX}``.
    Confusable letters are not detected; the column id is always shown next to the name.
    """
    out = []
    for ch in name:
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}:
            out.append(f"\\u{{{ord(ch):04X}}}")
        else:
            out.append(ch)
    return "".join(out)


def _collisions(columns: Sequence[ProjectedColumn]) -> tuple[tuple[str, ...], ...]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        groups.setdefault(loose_key(col.name), []).append(col.id)
    return tuple(tuple(ids) for key, ids in groups.items() if len(ids) > 1 and key)


def project_columns(columns: Sequence[tuple[str, ColumnKind]]) -> SchemaView:
    """Project ordered ``(name, kind)`` pairs. Names are NFC-normalized (as the CSV guard does)."""
    projected = []
    for i, (name, kind) in enumerate(columns, start=1):
        if not isinstance(name, str) or not name:
            raise ValueError("column names must be non-empty text")
        if kind not in ("numeric", "text"):
            raise ValueError("column kind must be 'numeric' or 'text'")
        projected.append(ProjectedColumn(f"c{i}", unicodedata.normalize("NFC", name), kind))
    if len({c.name for c in projected}) != len(projected):
        raise ValueError("column names must be unique")
    payload = {
        "version": SCHEMA_VIEW_VERSION,
        "columns": [[c.id, c.name, c.kind] for c in projected],
    }
    return SchemaView(tuple(projected), canonical_sha256(payload), _collisions(projected))


def project_schema(dataset: Dataset) -> SchemaView:
    """Projection of a loaded dataset: names and kinds only, in header order."""
    return project_columns(list(zip(dataset.column_names, dataset.column_kinds, strict=True)))
