"""Diagnostic for the EXP-005 Windows finding: are checked-out bytes the committed bytes? (EXP-006)

For every file whose bytes the environment manifest hashes (the ``axiom`` package's ``.py`` files
and ``uv.lock``) and for the text fixtures read by the experiments, compares the Git blob at
``HEAD`` (the committed bytes) with the working-tree file (the checked-out bytes): SHA-256 of
each, counts of CRLF and lone LF line endings, and the line-ending attributes Git applies. It also
recomputes the two manifest digests from the committed bytes, so the checkout digests can be
compared with digests of the repository content.

Read-only. Prints JSON with repository-relative paths only. Run from the repository root:

    uv run python benchmarks/checkout_line_endings.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "axiom"


def _git(*args: str, binary: bool = False) -> bytes | str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    out = proc.stdout if proc.returncode == 0 else b""
    return out if binary else out.decode("utf-8", "replace").strip()


def _endings(data: bytes) -> dict[str, int]:
    crlf = data.count(b"\r\n")
    return {"crlf": crlf, "lf_only": data.count(b"\n") - crlf}


def _entry(rel: str) -> dict:
    blob = _git("cat-file", "blob", f"HEAD:{rel}", binary=True)
    work = (ROOT / rel).read_bytes()
    attrs = _git("check-attr", "text", "eol", "--", rel).splitlines()
    return {
        "path": rel,
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "checkout_sha256": hashlib.sha256(work).hexdigest(),
        "identical": blob == work,
        # True if the only difference is LF -> CRLF conversion of the committed bytes
        "identical_after_crlf_to_lf": blob == work.replace(b"\r\n", b"\n"),
        "blob_endings": _endings(blob),
        "checkout_endings": _endings(work),
        "attributes": [a.split(": ", 1)[-1] for a in attrs],
    }


def _package_digest(read) -> str:
    """Same construction as ``axiom.environment.package_files_sha256`` with a pluggable reader."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix().encode("utf-8")
        data = read(path)
        h.update(len(rel).to_bytes(4, "big") + rel + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


def main() -> int:
    package_files = sorted(p.relative_to(ROOT).as_posix() for p in PACKAGE.rglob("*.py"))
    fixtures = sorted(
        p.relative_to(ROOT).as_posix()
        for pattern in ("examples/*.csv", "examples/**/*.json", "tests/data/nist/*.dat")
        for p in ROOT.glob(pattern)
    )
    entries = [_entry(rel) for rel in [*package_files, "uv.lock", *fixtures]]

    def blob_reader(path: Path) -> bytes:
        return _git("cat-file", "blob", f"HEAD:{path.relative_to(ROOT).as_posix()}", binary=True)

    from axiom.environment import code_revision  # the digests the manifest actually records

    recorded = code_revision()
    report = {
        "diagnostic": "checkout line endings (EXP-006 D1/D2)",
        "platform": sys.platform,
        "head": _git("rev-parse", "HEAD"),
        "git_config": {
            key: _git("config", "--get", key) or None
            for key in ("core.autocrlf", "core.eol", "core.safecrlf")
        },
        "files_differing": [e["path"] for e in entries if not e["identical"]],
        "n_files": len(entries),
        "digests": {
            "package_files_sha256_recorded": recorded["package_files_sha256"],
            "package_files_sha256_of_checkout": _package_digest(Path.read_bytes),
            "package_files_sha256_of_committed_blobs": _package_digest(blob_reader),
            "lock_sha256_recorded": recorded["lock_sha256"],
            "lock_sha256_of_committed_blob": hashlib.sha256(
                _git("cat-file", "blob", "HEAD:uv.lock", binary=True)
            ).hexdigest(),
        },
        "files": entries,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
