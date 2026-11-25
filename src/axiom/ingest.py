"""CSV input and guard: strict, explicit, fail-closed (ADR 0003).

Nothing is guessed: encoding, delimiter, decimal mark and the numeric/text schema are supplied by
the caller. An empty field is the only missing value. There are no implicit NA tokens, no thousands
separators, no locale guessing, no coercion of bad values to missing, and no deduplication.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import struct
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import InputError
from .limits import DEFAULT_LIMITS, Limits

DATASET_FORMAT_VERSION = "axiom.dataset/1"
SAFE_INTEGER_MAX = 2**53  # conservative envelope for integer tokens in numeric columns
_SAFE_INTEGER_DIGITS = len(str(SAFE_INTEGER_MAX))

ColumnKind = Literal["numeric", "text"]

_NONFINITE_TOKENS = frozenset({"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    kind: ColumnKind


class CsvProfile(BaseModel):
    """Explicit parsing profile. ``columns`` must list every header column, in header order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8"
    delimiter: Literal[",", ";"] = ","
    decimal: Literal[".", ","] = "."
    columns: tuple[ColumnSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> CsvProfile:
        if self.decimal == "," and self.delimiter != ";":
            raise ValueError("decimal comma requires the semicolon delimiter")
        names = [unicodedata.normalize("NFC", c.name) for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("schema column names collide after NFC normalization")
        return self


@dataclass(frozen=True, slots=True)
class Dataset:
    """Immutable typed snapshot of exactly the bytes that were parsed.

    Numeric columns hold ``float | None``; text columns hold ``str | None``; ``None`` is missing.
    Row order is source order. ``row_ordinal`` values are 1-based indexes of data records (header
    excluded, blank lines excluded), not physical line numbers.
    """

    column_names: tuple[str, ...]
    column_kinds: tuple[ColumnKind, ...]
    n_rows: int
    columns: Mapping[str, tuple[float | None, ...] | tuple[str | None, ...]]
    raw_sha256: str
    logical_sha256: str
    n_blank_lines: int
    n_all_empty_records: int
    profile: CsvProfile

    def kind(self, name: str) -> ColumnKind:
        return self.column_kinds[self.column_names.index(name)]


def _fail(code: str, message: str, **details: object) -> InputError:
    return InputError(code, message, **details)


def _decode(data: bytes, profile: CsvProfile) -> str:
    try:
        text = data.decode(profile.encoding)  # strict error handling
    except UnicodeDecodeError as exc:
        raise _fail(
            "invalid_encoding", f"bytes are not valid {profile.encoding}", pos=exc.start
        ) from exc
    if text.startswith("﻿"):
        raise _fail("unexpected_bom", "byte order mark present; declare encoding 'utf-8-sig'")
    if "\x00" in text:
        raise _fail("nul_character", "NUL character in input")
    return text


def _numeric_regex(decimal: str) -> re.Pattern[str]:
    d = re.escape(decimal)
    return re.compile(rf"[+-]?(?:[0-9]+(?:{d}[0-9]*)?|{d}[0-9]+)(?:[eE][+-]?[0-9]+)?")


_INTEGER_TOKEN = re.compile(r"[+-]?[0-9]+")


def _parse_numeric(token: str, pattern: re.Pattern[str], decimal: str, where: dict) -> float:
    if token.lower() in _NONFINITE_TOKENS:
        raise _fail("nonfinite_numeric", "NaN/Infinity tokens are not accepted", **where)
    if pattern.fullmatch(token) is None:
        raise _fail("invalid_numeric", "token is not a plain decimal number", **where)
    if _INTEGER_TOKEN.fullmatch(token):
        digits = token.lstrip("+-").lstrip("0")
        if len(digits) > _SAFE_INTEGER_DIGITS or (
            len(digits) == _SAFE_INTEGER_DIGITS and int(digits) > SAFE_INTEGER_MAX
        ):
            raise _fail(
                "unsafe_integer",
                "integer token exceeds +/-2^53; declare the column as text if it is an identifier",
                **where,
            )
    value = float(token.replace(decimal, ".") if decimal != "." else token)
    if not math.isfinite(value):
        raise _fail("numeric_overflow", "value overflows binary64", **where)
    if value == 0.0:
        mantissa = re.split(r"[eE]", token)[0]
        if any(ch in "123456789" for ch in mantissa):
            raise _fail("numeric_underflow", "nonzero value underflows to zero", **where)
    return value + 0.0  # canonical signed zero


def _hash_dataset(
    names: tuple[str, ...],
    kinds: tuple[ColumnKind, ...],
    columns: Mapping[str, tuple],
    n_rows: int,
) -> str:
    h = hashlib.sha256()
    h.update(DATASET_FORMAT_VERSION.encode("ascii") + b"\x00")
    h.update(struct.pack(">QQ", len(names), n_rows))
    for name, kind in zip(names, kinds, strict=True):
        raw = name.encode("utf-8")
        h.update(struct.pack(">I", len(raw)) + raw + (b"N" if kind == "numeric" else b"T"))
    for name, kind in zip(names, kinds, strict=True):
        if kind == "numeric":
            for v in columns[name]:
                h.update(b"\x00" if v is None else b"\x01" + struct.pack(">d", v))
        else:
            for v in columns[name]:
                if v is None:
                    h.update(b"\x00")
                else:
                    raw = v.encode("utf-8")
                    h.update(b"\x01" + struct.pack(">Q", len(raw)) + raw)
    return h.hexdigest()


def read_header(
    data: bytes,
    *,
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8",
    delimiter: Literal[",", ";"] = ",",
) -> tuple[str, ...]:
    """Return the NFC-normalized header names, to help a caller write an explicit schema."""
    probe = CsvProfile(
        encoding=encoding, delimiter=delimiter, columns=(ColumnSpec(name="_", kind="text"),)
    )
    text = _decode(data, probe)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        for record in reader:
            if record:
                return _normalize_header(record, DEFAULT_LIMITS)
    except csv.Error as exc:
        raise _fail("malformed_csv", str(exc)) from exc
    raise _fail("empty_input", "no header record")


def _normalize_header(record: list[str], limits: Limits) -> tuple[str, ...]:
    if len(record) > limits.max_columns:
        raise _fail("too_many_columns", "column cap exceeded", cap=limits.max_columns)
    names: list[str] = []
    for raw in record:
        if raw == "":
            raise _fail("empty_header_name", "header contains an empty column name")
        if len(raw) > limits.max_header_chars:
            raise _fail(
                "header_name_too_long", "header name exceeds cap", cap=limits.max_header_chars
            )
        names.append(unicodedata.normalize("NFC", raw))
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise _fail(
            "duplicate_header", "duplicate header names after NFC normalization", names=dupes
        )
    return tuple(names)


def load_csv_bytes(data: bytes, profile: CsvProfile, limits: Limits = DEFAULT_LIMITS) -> Dataset:
    """Parse ``data`` into an immutable :class:`Dataset`. The raw hash covers exactly ``data``."""
    if not isinstance(profile, CsvProfile):
        raise _fail("invalid_profile", "profile must be a CsvProfile")
    if len(data) > limits.max_bytes:
        raise _fail("input_too_large", "input exceeds byte cap", cap=limits.max_bytes)
    raw_sha256 = hashlib.sha256(data).hexdigest()
    text = _decode(data, profile)

    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=profile.delimiter,
        quotechar='"',
        doublequote=True,
        skipinitialspace=False,
        strict=True,
    )
    numeric_pattern = _numeric_regex(profile.decimal)
    names: tuple[str, ...] | None = None
    kinds: tuple[ColumnKind, ...] = ()
    cols: list[list] = []
    n_rows = n_blank = n_all_empty = 0
    try:
        for record in reader:
            if not record:  # a physically blank line (zero fields)
                n_blank += 1
                continue
            if names is None:
                names = _normalize_header(record, limits)
                spec_names = tuple(unicodedata.normalize("NFC", c.name) for c in profile.columns)
                if spec_names != names:
                    raise _fail(
                        "schema_mismatch",
                        "schema columns must equal the header columns, in header order",
                        header=list(names),
                        schema=list(spec_names),
                    )
                kinds = tuple(c.kind for c in profile.columns)
                cols = [[] for _ in names]
                continue
            ordinal = n_rows + 1
            if len(record) != len(names):
                raise _fail(
                    "record_width",
                    "record has a different number of fields than the header",
                    record_ordinal=ordinal,
                    expected=len(names),
                    found=len(record),
                )
            if n_rows >= limits.max_rows:
                raise _fail("too_many_rows", "row cap exceeded", cap=limits.max_rows)
            if all(f == "" for f in record):
                n_all_empty += 1
            for j, field in enumerate(record):
                if len(field) > limits.max_field_chars:
                    raise _fail(
                        "field_too_long",
                        "field exceeds cap",
                        record_ordinal=ordinal,
                        column=names[j],
                        cap=limits.max_field_chars,
                    )
                if field == "":
                    cols[j].append(None)
                elif kinds[j] == "text":
                    cols[j].append(field)
                else:
                    where = {"record_ordinal": ordinal, "column": names[j]}
                    cols[j].append(_parse_numeric(field, numeric_pattern, profile.decimal, where))
            n_rows += 1
    except csv.Error as exc:
        raise _fail("malformed_csv", str(exc), line=reader.line_num) from exc
    if names is None:
        raise _fail("empty_input", "no header record")

    frozen = MappingProxyType({n: tuple(c) for n, c in zip(names, cols, strict=True)})
    return Dataset(
        column_names=names,
        column_kinds=kinds,
        n_rows=n_rows,
        columns=frozen,
        raw_sha256=raw_sha256,
        logical_sha256=_hash_dataset(names, kinds, frozen, n_rows),
        n_blank_lines=n_blank,
        n_all_empty_records=n_all_empty,
        profile=profile,
    )


def load_csv(path: str | Path, profile: CsvProfile, limits: Limits = DEFAULT_LIMITS) -> Dataset:
    """Read at most ``max_bytes`` from ``path`` once and parse exactly those bytes."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(limits.max_bytes + 1)
    except OSError as exc:
        raise _fail("unreadable_file", f"cannot read input file: {type(exc).__name__}") from exc
    return load_csv_bytes(data, profile, limits)


__all__ = [
    "ColumnSpec",
    "CsvProfile",
    "Dataset",
    "load_csv",
    "load_csv_bytes",
    "read_header",
]
