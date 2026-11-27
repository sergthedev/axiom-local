"""Hash-recording downloader (standard library only).

Every download is streamed to a temporary file, size-capped, hashed with SHA-256 and moved into
place atomically. Nothing is executed or unpacked. Hosts are reached through whatever proxy and CA
bundle the environment configures (``HTTPS_PROXY``, ``SSL_CERT_FILE``); certificate verification
is never disabled.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

USER_AGENT = "axiom-local-axbench-tr/0.2 (open-source research; offline-first data acquisition)"
DEFAULT_MAX_BYTES = 400 * 1024 * 1024


class FetchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str  # relative to the raw directory, POSIX separators
    url: str
    bytes: int
    sha256: str
    retrieved: str  # UTC date, YYYY-MM-DD

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def raw_dir() -> Path:
    """Raw (never committed) data directory; override with ``AXBENCH_TR_RAW``."""
    override = os.environ.get("AXBENCH_TR_RAW")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "axbench-tr" / "_raw"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(
    url: str,
    rel_path: str,
    *,
    root: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> FileRecord:
    """Download ``url`` to ``root/rel_path`` and return its record. Raises :class:`FetchError`."""
    if not url.startswith("https://"):
        raise FetchError(f"refusing non-HTTPS url: {url}")
    if rel_path.startswith("/") or ".." in Path(rel_path).parts:
        raise FetchError(f"unsafe relative path: {rel_path}")
    root = root or raw_dir()
    dest = root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    digest = hashlib.sha256()
    total = 0
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=timeout) as resp:
            if resp.status != 200:
                raise FetchError(f"HTTP {resp.status} for {url}")
            while chunk := resp.read(1 << 20):
                total += len(chunk)
                if total > max_bytes:
                    raise FetchError(f"{url} exceeds the {max_bytes}-byte cap")
                digest.update(chunk)
                out.write(chunk)
        os.replace(tmp_name, dest)
    except FetchError:
        raise
    except Exception as exc:  # network, TLS, HTTP errors
        raise FetchError(f"{url}: {type(exc).__name__}: {exc}") from exc
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    today = _dt.datetime.now(_dt.UTC).date().isoformat()
    return FileRecord(Path(rel_path).as_posix(), url, total, digest.hexdigest(), today)
