"""Shared test helpers."""

from __future__ import annotations

from axiom import ColumnSpec, CsvProfile, Dataset, load_csv_bytes


def make_profile(spec: dict[str, str], **kwargs: object) -> CsvProfile:
    return CsvProfile(
        columns=tuple(ColumnSpec(name=n, kind=k) for n, k in spec.items()),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def make_dataset(csv_text: str, spec: dict[str, str], **kwargs: object) -> Dataset:
    return load_csv_bytes(csv_text.encode("utf-8"), make_profile(spec, **kwargs))
