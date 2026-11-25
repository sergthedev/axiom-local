"""Owner-controlled sealing of the final AXBENCH test split (ADR 0011).

The final test split is never written into the repository. The owner generates it with a seed only
the owner knows, into a directory outside the repository, and may add owner-authored test lineages
from a file that the implementing tool never sees. The command writes:

* ``test.jsonl`` (the sealed examples) and ``SEAL.json`` into the owner's directory;
* the audit of all three splits with test texts omitted.

Only ``SEAL.json`` (hashes, counts, generator commit, no texts) is meant to be published before
the final evaluation. The hash proves later that the evaluated file is the sealed one; it does not
prevent anyone with the file or the seed from reading it, and the repository templates of the
test lineages are visible to the implementing tool (see the dataset card). The test file is opened
for one frozen, pre-registered final evaluation only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .audit import audit
from .generate import (
    CORPUS_VERSION,
    GENERATOR_VERSION,
    TARGET_SIZES,
    generate_development_corpus,
    generate_split,
    normalize_request,
)
from .lineages import LINEAGES, Lineage

SEAL_VERSION = "axbench.seal/0.1"


class SealError(Exception):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_owner_lineages(path: str | os.PathLike) -> list[Lineage]:
    """Owner-authored test lineages: a JSON list of {id, frame, tr, en, mix?, hint?, method?}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {lin.frame for lin in LINEAGES}
    out = []
    for item in data:
        if item.get("frame") not in known:
            raise SealError(f"unknown frame in owner lineage {item.get('id')!r}")
        fixed = {"method": item["method"]} if item.get("method") else {}
        out.append(
            Lineage(
                f"O-{item['id']}",
                item["frame"],
                "test",
                item["tr"],
                item["en"],
                item.get("mix"),
                fixed,
                item.get("hint"),
            )
        )
    return out


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seal(
    out_dir: str | os.PathLike,
    seed: str,
    *,
    owner_lineages: list[Lineage] | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    if _inside(out, _repo_root()):
        raise SealError("the sealed test must be written outside the repository")
    if len(seed) < 16:
        raise SealError("use a seed of at least 16 characters that only the owner knows")
    if out.exists() and any(out.iterdir()):
        raise SealError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    dev = generate_development_corpus()
    seen = {normalize_request(e.request) for exs in dev.values() for e in exs}
    test = generate_split(
        "test",
        seed=seed,
        size=size or TARGET_SIZES["test"],
        seen=seen,
        extra_lineages=tuple(owner_lineages or ()),
    )
    records = [e.to_record() for e in test]
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    data = text.encode("utf-8")
    (out / "test.jsonl").write_bytes(data)  # bytes: no newline translation, so the hash holds
    report = audit(
        {
            "train": [e.to_record() for e in dev["train"]],
            "validation": [e.to_record() for e in dev["validation"]],
            "test": records,
        }
    )
    manifest = {
        "seal_version": SEAL_VERSION,
        "corpus_version": CORPUS_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_commit": _commit(),
        "test_sha256": hashlib.sha256(data).hexdigest(),
        "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "n_test": len(records),
        "owner_lineages": len(owner_lineages or []),
        "audit_passed": report["checks"]["passed"],
        "audit_failures": report["checks"]["failures"],
        "test_distribution": report["splits"]["test"],
        "note": "Publish this file only; keep test.jsonl and the seed private until the final run.",
    }
    (out / "SEAL.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


def verify_sealed(test_path: str | os.PathLike, seal_path: str | os.PathLike) -> list[dict]:
    """Load a sealed test only if its bytes match the published seal hash."""
    data = Path(test_path).read_bytes()
    manifest = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    if hashlib.sha256(data).hexdigest() != manifest["test_sha256"]:
        raise SealError("test file does not match the seal")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
