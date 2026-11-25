"""Environment capture for the Analysis Passport (ADR 0007).

Records what can actually be observed about the runtime: Python build, operating system and
architecture, installed package versions, code revision, NumPy's numerical backend and the effective
native thread pools. Anything that cannot be observed is recorded as ``unavailable`` with a reason,
never guessed. No absolute paths, host names, user names or process ids are recorded.

This information describes *where* a result was computed. It is not part of the deterministic
result core and it does not prove correctness or reproducibility.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ENVIRONMENT_SCHEMA_VERSION = "axiom.environment/1"

# Distributions whose versions can change numerical results or validation behaviour.
PACKAGES = (
    "axiom-local",
    "numpy",
    "scipy",
    "scikit-learn",
    "threadpoolctl",
    "joblib",
    "pydantic",
    "pydantic-core",
)

# Thread-control environment variables documented by scikit-learn for OpenMP and BLAS (REF-025);
# VECLIB_MAXIMUM_THREADS is Apple Accelerate's variable and is recorded for completeness only.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Fields whose equality defines "the same environment" for exact replay (ADR 0008).
COMPARABLE_FIELDS = (
    ("python", "version"),
    ("python", "implementation"),
    ("python", "compiler"),
    ("os", "system"),
    ("os", "release"),
    ("os", "machine"),
    ("packages",),
    ("numerical_backend",),
    ("code", "commit"),
    ("code", "dirty"),
    ("code", "lock_sha256"),
    ("code", "package_files_sha256"),
)


def unavailable(reason: str) -> dict[str, str]:
    """Explicit marker for information that could not be observed."""
    return {"status": "unavailable", "reason": reason}


def is_unavailable(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == "unavailable"


def _source_root() -> Path | None:
    """Repository root if this package runs from a source checkout (editable install)."""
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "pyproject.toml").is_file() else None


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", *args],  # read-only: never refresh the index
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def package_files_sha256() -> str:
    """SHA-256 over the relative names and bytes of the ``axiom`` package's ``.py`` files.

    Identifies the engine code that actually ran, also outside a Git checkout. Relative names only.
    """
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix().encode("utf-8")
        data = path.read_bytes()
        h.update(len(rel).to_bytes(4, "big") + rel + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


def code_revision() -> dict[str, Any]:
    """Engine file digest, plus commit, dirty flag and ``uv.lock`` digest when observable."""
    out: dict[str, Any] = {"package_files_sha256": package_files_sha256()}
    root = _source_root()
    if root is None:
        reason = "package is not running from a source checkout"
        out.update(commit=unavailable(reason), dirty=unavailable(reason))
        out["lock_sha256"] = unavailable(reason)
        return out
    if (root / ".git").exists():
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        out["commit"] = commit if commit else unavailable("git rev-parse failed or git is missing")
        out["dirty"] = (
            bool(status)
            if status is not None
            else unavailable("git status failed or git is missing")
        )
    else:
        out["commit"] = unavailable("source checkout has no .git directory")
        out["dirty"] = unavailable("source checkout has no .git directory")
    lock = root / "uv.lock"
    out["lock_sha256"] = (
        hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.is_file()
        else unavailable("uv.lock not found in the source checkout")
    )
    return out


def package_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = unavailable("distribution not installed")
    return out


def numerical_backend() -> dict[str, Any]:
    """NumPy's build-time BLAS/LAPACK and CPU dispatch information (``show_config``)."""
    import numpy

    try:
        config = numpy.show_config(mode="dicts")
    except Exception as exc:  # noqa: BLE001 - any failure means "not observable"
        return unavailable(f"numpy.show_config failed ({type(exc).__name__})")

    def lib(key: str) -> dict[str, Any]:
        entry = config.get("Build Dependencies", {}).get(key)
        if not isinstance(entry, dict):
            return unavailable(f"numpy build information has no {key} entry")
        return {"name": entry.get("name"), "version": entry.get("version")}

    simd = config.get("SIMD Extensions")
    return {
        "numpy_blas": lib("blas"),
        "numpy_lapack": lib("lapack"),
        "simd": (
            {
                "baseline": list(simd.get("baseline", [])),
                "found": list(simd.get("found", [])),
                "not_found": list(simd.get("not found", [])),
            }
            if isinstance(simd, dict)
            else unavailable("numpy build information has no SIMD entry")
        ),
    }


def _pools() -> list[dict[str, Any]]:
    """Native thread pools seen by threadpoolctl, without file paths."""
    from threadpoolctl import threadpool_info

    keep = ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer")
    pools = [{k: p.get(k) for k in keep if k in p} for p in threadpool_info()]
    return sorted(pools, key=lambda p: (str(p.get("user_api")), str(p.get("prefix"))))


def thread_policy() -> dict[str, Any]:
    """Observed thread configuration and a probe of the limit used for K-means fitting.

    ``detected_pools`` lists the pools threadpoolctl can see in this process (after the engine's
    numerical modules are imported). ``fit_limit_probe`` re-reads them inside
    ``threadpool_limits(limits=1)``, the context the Cluster operation fits in. A BLAS library that
    threadpoolctl cannot introspect (for example Apple Accelerate) is reported as unverified.
    """
    from threadpoolctl import threadpool_limits

    import axiom.ops_cluster  # noqa: F401  - load the solver's native libraries before probing

    env = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    try:
        pools = _pools()
        with threadpool_limits(limits=1):
            limited = _pools()
    except Exception as exc:  # noqa: BLE001
        return {
            "environment_variables": env,
            "detected_pools": unavailable(f"threadpoolctl failed ({type(exc).__name__})"),
            "fit_limit_probe": "unverified",
            "blas_thread_control": "unverified",
        }
    probe_ok = bool(limited) and all(p.get("num_threads") == 1 for p in limited)
    backend = numerical_backend()
    blas_name = None
    if isinstance(backend, dict) and isinstance(backend.get("numpy_blas"), dict):
        blas_name = backend["numpy_blas"].get("name")
    blas_seen = any(p.get("user_api") == "blas" for p in pools)
    return {
        "environment_variables": env,
        "detected_pools": pools,
        "fit_limit_probe": "verified_single_thread" if probe_ok else "unverified",
        "blas_thread_control": (
            "introspectable"
            if blas_seen
            else f"unverified (NumPy BLAS {blas_name!r} is not visible to threadpoolctl)"
        ),
    }


def capture_environment() -> dict[str, Any]:
    """Environment manifest (``axiom.environment/1``). Excludes paths and host names."""
    cpu_count: Any = os.cpu_count() or unavailable("os.cpu_count returned None")
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "bits": 64 if sys.maxsize > 2**32 else 32,
            "float_repr_style": sys.float_repr_style,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine() or unavailable("platform.machine returned ''"),
            "processor": platform.processor() or unavailable("platform.processor returned ''"),
            "logical_cpus": cpu_count,
            "byteorder": sys.byteorder,
        },
        "packages": package_versions(),
        "code": code_revision(),
        "numerical_backend": numerical_backend(),
        "threads": thread_policy(),
    }


# Fields of the thread record compared as well (review B3): the variables set, the pools actually
# observed (with their thread counts), the probe inside the fit's thread limit, and whether NumPy's
# BLAS is visible at all.
THREAD_FIELDS = (
    ("threads", "environment_variables"),
    ("threads", "detected_pools"),
    ("threads", "fit_limit_probe"),
    ("threads", "blas_thread_control"),
)

EQUAL, DIFFERENT, UNKNOWN = "equal", "different", "unknown"


def _field(env: Any, path: tuple[str, ...]) -> Any:
    cur: Any = env
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return unavailable("field missing")
        cur = cur[key]
    return cur


def _is_unknown_leaf(path: tuple[str, ...], value: Any) -> bool:
    """Observations that state that something is *not* known.

    An unavailable marker anywhere; ``None`` or ``"unknown"`` reported for a version or other
    value (NumPy reports Accelerate's version as ``"unknown"``; threadpoolctl reports libomp's as
    ``None``); a thread status starting with ``unverified``. In the thread-control variables
    ``None`` is an observation ("not set"), not an unknown.
    """
    if is_unavailable(value):
        return True
    if path[:2] == ("threads", "environment_variables"):
        return False
    if value is None or value == "unknown":
        return True
    return path[:1] == ("threads",) and isinstance(value, str) and value.startswith("unverified")


def _compare_values(path: tuple[str, ...], a: Any, b: Any) -> str:
    """``equal``, ``different`` (both sides observed and unequal somewhere) or ``unknown``."""
    if _is_unknown_leaf(path, a) or _is_unknown_leaf(path, b):
        return UNKNOWN
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return DIFFERENT
        states = [_compare_values((*path, k), a[k], b[k]) for k in sorted(a)]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return DIFFERENT
        states = [_compare_values(path, x, y) for x, y in zip(a, b, strict=True)]
    else:
        return EQUAL if type(a) is type(b) and a == b else DIFFERENT
    if DIFFERENT in states:
        return DIFFERENT
    return UNKNOWN if UNKNOWN in states else EQUAL


def compare_environments(reference: Any, other: Any) -> dict[str, Any]:
    """Classify two environment manifests for replay and comparison (ADR 0008, review B3).

    ``status`` is:

    * ``verified_same``: every compared field was observed on both sides and is equal;
    * ``different``: at least one compared field was observed on both sides and differs;
    * ``unverified``: nothing observed differs, but some field is unknown on at least one side
      (for example a BLAS whose threads threadpoolctl cannot see). Unknown is never equal.

    Only ``different`` permits the cross-environment numeric criteria; in the other two cases a
    replay must still be byte-identical, so unknown metadata cannot weaken exact replay.
    """
    differences: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for path in (*COMPARABLE_FIELDS, *THREAD_FIELDS):
        a, b = _field(reference, path), _field(other, path)
        state = _compare_values(path, a, b)
        entry = {"field": ".".join(path), "reference": a, "observed": b}
        if state == DIFFERENT:
            differences.append(entry)
        elif state == UNKNOWN:
            unverified.append(entry)
    status = "different" if differences else ("unverified" if unverified else "verified_same")
    return {
        "status": status,
        "fields_compared": [".".join(p) for p in (*COMPARABLE_FIELDS, *THREAD_FIELDS)],
        "differences": differences,
        "unverified_fields": unverified,
    }


def environment_differences(reference: dict[str, Any], other: dict[str, Any]) -> list[dict]:
    """Compared fields that differ or cannot be verified (``comparable=False``).

    Empty only when :func:`compare_environments` reports ``verified_same``.
    """
    result = compare_environments(reference, other)
    return [dict(d, comparable=True) for d in result["differences"]] + [
        dict(d, comparable=False) for d in result["unverified_fields"]
    ]


def native_thread_status(environment: Any) -> dict[str, Any]:
    """Whether single-threaded native execution was verified for *every* relevant library.

    ``verified_single_thread`` only if threadpoolctl saw at least one pool, every pool reported one
    thread inside the fit's limit, and NumPy's BLAS was among the introspectable pools. Apple
    Accelerate is invisible to threadpoolctl, so on such a build the status is ``unverified``
    with the reason, even when the OpenMP pool is verified (review B3).
    """
    threads = _field(environment, ("threads",))
    if not isinstance(threads, dict):
        return {"status": "unverified", "reason": "no thread record"}
    pools = threads.get("detected_pools")
    if not isinstance(pools, list) or not pools:
        return {"status": "unverified", "reason": "no native pools observed"}
    if any(isinstance(p, dict) and p.get("num_threads") != 1 for p in pools):
        return {"status": "multi_threaded", "reason": "a detected pool reported more than 1 thread"}
    if threads.get("fit_limit_probe") != "verified_single_thread":
        return {"status": "unverified", "reason": "thread-limit probe not verified"}
    if threads.get("blas_thread_control") != "introspectable":
        return {"status": "unverified", "reason": str(threads.get("blas_thread_control"))}
    return {"status": "verified_single_thread", "reason": "all detected pools at 1 thread"}
