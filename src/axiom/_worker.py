"""Fresh-process worker for analysis and replay (ADR 0008). Not a public interface.

Started by :mod:`axiom.replay` as ``python -I -m axiom._worker`` with thread-control environment
variables already set (so they take effect before NumPy and the solver libraries load), in a fresh
temporary working directory. It reads one JSON request on stdin and writes one JSON response on
stdout. The request never contains a reference result, so the worker cannot copy one: it re-reads
the data file, re-parses, re-binds or re-checks the binding, and recomputes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from typing import Any

WORKER_PROTOCOL = "axiom.worker/1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def handle(request: dict[str, Any]) -> dict[str, Any]:
    started = _now()
    t0 = time.perf_counter()
    from .engine import execute
    from .environment import capture_environment
    from .errors import AxiomError
    from .ingest import load_csv_bytes
    from .passport import limits_from_dict, profile_from_dict
    from .plan import BoundPlan, bind_plan, parse_plan_object

    import_seconds = time.perf_counter() - t0
    response: dict[str, Any] = {
        "protocol": WORKER_PROTOCOL,
        "run_id": request.get("run_id"),
        "started_utc": started,
        "status": "error",
        "error": None,
        "fingerprints": None,
        "result_canonical_json": None,
        "timing": {"import_seconds": import_seconds},
    }
    try:
        limits = limits_from_dict(request["limits"])
        profile = profile_from_dict(request["profile"])
        t1 = time.perf_counter()
        with open(request["data_path"], "rb") as handle_:
            data = handle_.read(limits.max_bytes + 1)
        response["fingerprints"] = {
            "raw_sha256": hashlib.sha256(data).hexdigest(),
            "n_bytes": len(data),
        }
        dataset = load_csv_bytes(data, profile, limits)
        response["fingerprints"].update(
            dataset_sha256=dataset.logical_sha256, n_rows=dataset.n_rows
        )
        t2 = time.perf_counter()
        plan = parse_plan_object(request["plan"], limits)
        binding = request.get("binding")
        if binding is None:  # analysis: the engine binds and hashes itself
            bound = bind_plan(dataset, plan, limits)
        else:  # replay: the old binding is presented as-is; execute() re-checks it
            bound = BoundPlan(
                plan=plan,
                plan_sha256=binding["plan_sha256"],
                dataset_raw_sha256=binding["raw_sha256"],
                dataset_logical_sha256=binding["dataset_sha256"],
                policy_version=binding["policy_version"],
            )
        result = execute(dataset, bound, limits)
        t3 = time.perf_counter()
        response.update(status="ok", result_canonical_json=result.canonical_json)
        response["timing"].update(load_seconds=t2 - t1, compute_seconds=t3 - t2)
    except AxiomError as exc:
        response["error"] = exc.to_dict()
    except OSError as exc:
        response["error"] = {
            "type": "OSError",
            "code": "unreadable_file",
            "message": f"cannot read data file ({type(exc).__name__})",
            "details": {},
        }
    except Exception as exc:  # noqa: BLE001 - reported as an execution failure, never hidden
        response["error"] = {
            "type": type(exc).__name__,
            "code": "unexpected_exception",
            "message": "worker raised an unexpected exception",
            "details": {},
        }
    response["environment"] = capture_environment()
    response["finished_utc"] = _now()
    return response


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if request.get("protocol") != WORKER_PROTOCOL:
            raise ValueError("unsupported worker protocol")
    except (ValueError, AttributeError) as exc:
        json.dump(
            {"protocol": WORKER_PROTOCOL, "status": "bad_request", "error": str(exc)}, sys.stdout
        )
        return 2
    json.dump(handle(request), sys.stdout, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
