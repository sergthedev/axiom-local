"""Registry-driven acquisition: downloads eligible files, records SHA-256 for every byte.

Run ``python -m axbench_tr acquire [--source SRC-N01 ...]``. Raw files go to the raw
directory (outside Git, see :func:`axbench_tr.fetch.raw_dir`); the checksum manifest
``ACQUISITION.json`` is committed.
Network failures are recorded, never hidden, and never replaced by made-up values.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .fetch import FetchError, FileRecord, download, raw_dir
from .registry import ACQUISITION_PATH, load_registry

Method = Callable[[dict[str, Any]], list[FileRecord]]


def _hf_files(spec: dict[str, Any]) -> list[FileRecord]:
    repo, rev = spec["repo"], spec["revision"]
    out = []
    for f in spec["files"]:
        prefix = "datasets/" if spec.get("repo_type", "dataset") == "dataset" else ""
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{rev}/{f}"
        out.append(download(url, f"hf/{repo}/{rev[:12]}/{f}"))
    return out


def _github_files(spec: dict[str, Any]) -> list[FileRecord]:
    repo, commit = spec["repo"], spec["commit"]
    out = []
    for f in spec["files"]:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{f}"
        out.append(download(url, f"github/{repo}/{commit[:12]}/{f}"))
    return out


def _http_files(spec: dict[str, Any]) -> list[FileRecord]:
    """Plain HTTPS files. With ``"tolerate_missing": true`` a failing file (for example a bank
    holiday without a bulletin) is skipped, and the manifest lists only the files that exist."""
    out = []
    for item in spec["files"]:
        try:
            out.append(download(item["url"], item["path"], headers=item.get("headers")))
        except FetchError:
            if not spec.get("tolerate_missing"):
                raise
    return out


METHODS: dict[str, Method] = {
    "hf_files": _hf_files,
    "github_files": _github_files,
    "http_files": _http_files,
}


def register_method(name: str, fn: Method) -> None:
    METHODS[name] = fn


def acquire(source_ids: list[str] | None = None) -> dict[str, Any]:
    from . import official  # noqa: F401  (registers the official-source methods)

    registry = load_registry()
    previous = json.loads(ACQUISITION_PATH.read_text("utf-8")) if ACQUISITION_PATH.exists() else {}
    results: dict[str, Any] = dict(previous.get("sources", {}))
    for src in registry["sources"]:
        if source_ids and src["id"] not in source_ids:
            continue
        spec = src["acquire"]
        if spec["method"] == "none":
            continue
        entry: dict[str, Any] = {"method": spec["method"], "files": [], "errors": []}
        try:
            entry["files"] = [r.as_dict() for r in METHODS[spec["method"]](spec)]
        except FetchError as exc:
            entry["errors"].append(str(exc))
        results[src["id"]] = entry
        print(src["id"], len(entry["files"]), "files", len(entry["errors"]), "errors")
    manifest = {
        "schema": "axbench-tr.acquisition/1",
        "raw_dir_env": "AXBENCH_TR_RAW",
        "sources": results,
    }
    ACQUISITION_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    return manifest


def verify(source_ids: list[str] | None = None) -> list[str]:
    """Re-hash local raw files against the committed manifest; return mismatch descriptions."""
    from .fetch import sha256_file

    manifest = json.loads(ACQUISITION_PATH.read_text("utf-8"))
    problems = []
    for sid, entry in manifest["sources"].items():
        if source_ids and sid not in source_ids:
            continue
        for rec in entry["files"]:
            path = raw_dir() / rec["path"]
            if not path.exists():
                problems.append(f"{sid}: missing {rec['path']}")
            elif sha256_file(path) != rec["sha256"]:
                problems.append(f"{sid}: checksum differs {rec['path']}")
    return problems
