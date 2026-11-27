"""Machine-readable source registry and its validation.

The registry file is ``data/axbench-tr/sources/registry.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "axbench-tr.source-registry/1"
REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "axbench-tr" / "sources" / "registry.json"
)
ACQUISITION_PATH = REGISTRY_PATH.with_name("ACQUISITION.json")

STATUSES = frozenset(
    {
        "accepted_adapted",  # content adapted into training examples
        "accepted_schema_only",  # only table schemas (column headers) are used
        "accepted_reference",  # observations kept apart from training data, for demos and checks
        "reviewed_not_used",
        "probed_not_acquired",
        "blocked_access",
        "rejected_duplicate",
        "rejected_license_conflict",
        "rejected_license_noncommercial",
        "rejected_license_unverified",
        "rejected_not_found",
        "rejected_provenance",
        "rejected_scope_and_license",
        "rejected_task_mismatch",
        "rejected_undocumented",
    }
)
REQUIRED = (
    "id",
    "name",
    "category",
    "publisher",
    "url",
    "license",
    "language",
    "domain",
    "status",
    "justification",
    "suitability",
    "access_date",
    "acquire",
)
CATEGORIES = frozenset(
    {"nlp", "official_economic", "municipal_open_data", "international_economic"}
)
LICENSE_KEYS = ("id", "evidence", "derivatives_allowed", "redistribution")


class RegistryError(Exception):
    pass


def load_registry(path: Path | None = None) -> dict[str, Any]:
    data = json.loads((path or REGISTRY_PATH).read_text(encoding="utf-8"))
    validate_registry(data)
    return data


def validate_registry(data: dict[str, Any]) -> None:
    if data.get("schema") != SCHEMA:
        raise RegistryError(f"unexpected registry schema: {data.get('schema')!r}")
    seen: set[str] = set()
    for src in data["sources"]:
        sid = src.get("id", "<missing id>")
        if sid in seen:
            raise RegistryError(f"duplicate source id {sid}")
        seen.add(sid)
        for key in REQUIRED:
            if key not in src:
                raise RegistryError(f"{sid}: missing field {key!r}")
        if src["status"] not in STATUSES:
            raise RegistryError(f"{sid}: unknown status {src['status']!r}")
        if src["category"] not in CATEGORIES:
            raise RegistryError(f"{sid}: unknown category {src['category']!r}")
        if not src["url"].startswith("https://"):
            raise RegistryError(f"{sid}: url must be https")
        for key in LICENSE_KEYS:
            if key not in src["license"]:
                raise RegistryError(f"{sid}: license lacks {key!r}")
        if (
            src["status"].startswith("accepted")
            and src["license"]["derivatives_allowed"] is not True
        ):
            # Accepted sources must have a verified licence that permits derivatives, unless only
            # schemas (column headers, which are facts) are used.
            if src["status"] != "accepted_schema_only":
                raise RegistryError(f"{sid}: accepted without verified derivative rights")
        if src["status"].startswith("rejected") and src["license"]["derivatives_allowed"] is True:
            if src["status"] not in {
                "rejected_duplicate",
                "rejected_task_mismatch",
                "rejected_provenance",
                "rejected_undocumented",
                "rejected_not_found",
            }:
                raise RegistryError(
                    f"{sid}: licence-rejected source marked as allowing derivatives"
                )
