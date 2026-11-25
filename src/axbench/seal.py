"""Owner-controlled sealing of the final AXBENCH test split (ADR 0011).

The final test split is never written into the repository. The owner generates it with a seed only
the owner knows, into a directory outside the repository, and may add owner-authored test lineages
from a file that the implementing tool never sees. The command writes:

* ``test.jsonl`` (the sealed examples) and ``SEAL.json`` into the owner's directory;
* the audit of all three splits with test texts omitted.

Only ``SEAL.json`` (hashes, counts, generator commit, no texts) is meant to be published before
the final evaluation. The hash proves later that the evaluated file is the sealed one; it does not
prevent anyone with the file or the seed from reading it. A private seed hides which examples are
drawn, **not** the repository test lineages: their templates are visible to the implementing tool,
so test examples rendered from them are not blind (see the dataset card). The test file is opened
for one frozen, pre-registered final evaluation only.

Revision 2 (Phase 3A review): the seal reads the development splits from a **frozen** corpus
directory and verifies their hashes (no regeneration from templates); the audit runs before
anything is written, and a failed audit writes nothing; examples drawn from owner lineages are
labelled ``owner_template_ai_rendered`` (the owner wrote the template, the generator rendered the
example; nobody reviewed the rendered example) and are counted per lineage.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .audit import audit
from .corpus import CorpusError, load_corpus, manifest_sha256
from .generate import (
    GENERATOR_VERSION,
    TARGET_SIZES,
    generate_split,
    normalize_request,
)
from .lineages import LINEAGES, Lineage

SEAL_VERSION = "axbench.seal/0.2"
OWNER_PREFIX = "O-"


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
                f"{OWNER_PREFIX}{item['id']}",
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
    corpus_dir: str | os.PathLike,
    owner_lineages: list[Lineage] | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    """Generate, audit and write the sealed test against a frozen development corpus.

    Raises :class:`SealError` and writes nothing when the corpus is not a verified frozen version,
    the output is inside the repository or not empty, or the audit fails.
    """
    out = Path(out_dir)
    if _inside(out, _repo_root()):
        raise SealError("the sealed test must be written outside the repository")
    if len(seed) < 16:
        raise SealError("use a seed of at least 16 characters that only the owner knows")
    if out.exists() and any(out.iterdir()):
        raise SealError("output directory must be empty")
    try:
        dev_manifest, dev = load_corpus(corpus_dir)
    except CorpusError as exc:
        raise SealError(f"development corpus not usable: {exc}") from exc
    if dev_manifest.get("status") != "frozen":
        raise SealError("seal only against a frozen development corpus (python -m axbench freeze)")
    owner = list(owner_lineages or ())
    if len({lin.id for lin in owner}) != len(owner):
        raise SealError("owner lineage ids must be unique")
    seen = {normalize_request(r["request"]) for records in dev.values() for r in records}
    test = generate_split(
        "test",
        seed=seed,
        size=size or TARGET_SIZES["test"],
        seen=seen,
        extra_lineages=tuple(owner),
    )
    records = [e.to_record() for e in test]
    per_owner_lineage = {lin.id: 0 for lin in owner}
    for r in records:
        r["corpus_version"] = dev_manifest["corpus_version"]
        if r["lineage_id"] in per_owner_lineage:
            per_owner_lineage[r["lineage_id"]] += 1
            r["provenance"]["authoring"] = "owner_template_ai_rendered"
    report = audit({"train": dev["train"], "validation": dev["validation"], "test": records})
    if not report["checks"]["passed"]:
        raise SealError(f"audit failed; nothing written: {report['checks']['failures']}")
    data = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ).encode("utf-8")
    manifest = {
        "seal_version": SEAL_VERSION,
        "corpus_version": dev_manifest["corpus_version"],
        "development_corpus": {
            "manifest_sha256": manifest_sha256(corpus_dir),
            "files": dev_manifest["files"],
        },
        "generator_version": GENERATOR_VERSION,
        "generator_commit": _commit(),
        "test_sha256": hashlib.sha256(data).hexdigest(),
        "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "n_test": len(records),
        "owner_lineages": {
            "supplied": len(owner),
            "used": sum(1 for n in per_owner_lineage.values() if n),
            "examples": sum(per_owner_lineage.values()),
            "examples_per_lineage": dict(sorted(per_owner_lineage.items())),
            "note": "owner-written templates rendered by the generator; the rendered examples "
            "are not human-reviewed",
        },
        "examples_from_repository_lineages": len(records) - sum(per_owner_lineage.values()),
        "audit_passed": True,
        "test_distribution": report["splits"]["test"],
        "note": "Publish this file only; keep test.jsonl, the seed and the owner lineage file "
        "private and outside the repository and any AI tool's context until the final run.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "test.jsonl").write_bytes(data)  # bytes: no newline translation, so the hash holds
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
