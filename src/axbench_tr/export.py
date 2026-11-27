"""Export AXBENCH-TR in the MLX-LM chat format (``python -m axbench_tr export-mlx --out DIR``).

Same record layout as ``python -m axbench export-mlx``: a system turn (the ``axiom.prompt/1``
instruction), a user turn (the router input JSON) and an assistant turn (compact gold JSON). Three
sets are written under the output directory, which must be outside the repository:

* ``full/``: all train and validation records;
* ``smoke/``: a deterministic 400-record train subset (stratified by frame) plus 40 validation
  records, for a feasibility smoke run;
* ``peak_probe/``: the 20 longest train records, to measure worst-case memory first.

Nothing is truncated or dropped here; length is checked by ``benchmarks/axbench_tr_token_stats.py``.
No test file exists in this corpus and none is written.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from axiom_router.prompt import build_router_input, chat_messages
from axiom_router.schema import project_columns

from .build import CURRENT_VERSION, OUT_ROOT
from .fetch import sha256_bytes

SMOKE_TRAIN = 400
SMOKE_VALID = 40
PROBE = 20
SEED = "axbench-tr-0.2-export"
CLEARED_ORIGIN = "synthetic_template"  # the only origin with no unresolved upstream terms


def chat_record(rec: dict[str, Any]) -> dict[str, Any]:
    view = project_columns([(c["name"], c["type"]) for c in rec["schema"]["columns"]])
    messages = chat_messages(build_router_input(rec["request"], view))
    target = json.dumps(rec["gold"], ensure_ascii=False, separators=(",", ":"))
    return {"messages": [*messages, {"role": "assistant", "content": target}]}


def _rank(key: str, rec_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{key}|{rec_id}".encode()).hexdigest()


def stratified_subset(rows: list[dict[str, Any]], n: int, key: str) -> list[dict[str, Any]]:
    """Deterministic subset of about ``n`` records spread evenly over frames (round robin)."""
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: _rank(key, r["id"])):
        by_frame[r["frame"]].append(r)
    out: list[dict[str, Any]] = []
    queues = [by_frame[f] for f in sorted(by_frame)]
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return sorted(out, key=lambda r: r["id"])


def _write(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    data = (
        "\n".join(json.dumps(chat_record(r), ensure_ascii=False) for r in records) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"examples": len(records), "bytes": len(data), "sha256": sha256_bytes(data)}


def export(out: Path, version: str = CURRENT_VERSION) -> dict[str, Any]:
    out = out.resolve()
    if OUT_ROOT.parents[1] in (out, *out.parents):
        raise SystemExit("refusing to export inside the repository; choose another directory")
    corpus = {
        s: [
            json.loads(line)
            for line in (OUT_ROOT / version / f"{s}.jsonl").read_text("utf-8").splitlines()
        ]
        for s in ("train", "validation")
    }
    lengths = {
        r["id"]: r["sequence_tokens"]
        for r in map(
            json.loads,
            (OUT_ROOT / version / "token_lengths_per_example.jsonl")
            .read_text("utf-8")
            .splitlines(),
        )
    }
    train, valid = corpus["train"], corpus["validation"]
    # The smoke and the probe use only rights-cleared records: this project's own template text.
    # External adapted records (upstream Gretel generation terms unresolved) and unverified
    # machine translations stay out of them.
    cleared_train = [r for r in train if r["provenance"]["origin"] == CLEARED_ORIGIN]
    cleared_valid = [r for r in valid if r["provenance"]["origin"] == CLEARED_ORIGIN]
    longest = sorted(cleared_train, key=lambda r: (-lengths[r["id"]], r["id"]))[:PROBE]
    probe_ids = {r["id"] for r in longest}
    smoke_train = stratified_subset(
        [r for r in cleared_train if r["id"] not in probe_ids], SMOKE_TRAIN, "t"
    )
    smoke_valid = stratified_subset(cleared_valid, SMOKE_VALID, "v")
    sets = {
        "full": (train, valid),
        "smoke": (smoke_train, smoke_valid),
        "peak_probe": (longest, longest[:SMOKE_VALID]),
    }
    files: dict[str, Any] = {}
    row_ids: dict[str, list[str]] = {}
    for name, (tr, va) in sets.items():
        files[f"{name}/train.jsonl"] = _write(out / name / "train.jsonl", tr)
        files[f"{name}/valid.jsonl"] = _write(out / name / "valid.jsonl", va)
        row_ids[f"{name}/train.jsonl"] = [r["id"] for r in tr]  # file order, line by line
        row_ids[f"{name}/valid.jsonl"] = [r["id"] for r in va]
    ids_bytes = (json.dumps(row_ids, indent=0, sort_keys=True) + "\n").encode("utf-8")
    (out / "ROW_IDS.json").write_bytes(ids_bytes)
    files["ROW_IDS.json"] = {"bytes": len(ids_bytes), "sha256": sha256_bytes(ids_bytes)}
    manifest = {
        "schema": "axbench-tr.mlx-export/1",
        "corpus_version": f"axbench-tr/{version.removeprefix('v')}",
        "format": "mlx-lm chat: system, user (axiom.prompt/1 JSON), assistant (compact gold JSON)",
        "files": files,
        "peak_probe_lengths": sorted((lengths[r["id"]] for r in longest), reverse=True),
        "test_file": "none: this corpus has no test split and none is exported",
        "rights_cleared_subsets": {
            "smoke": f"origin {CLEARED_ORIGIN} only",
            "peak_probe": f"origin {CLEARED_ORIGIN} only",
            "full": "all origins; external_adapted rights unresolved: not cleared for training",
        },
    }
    (out / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
