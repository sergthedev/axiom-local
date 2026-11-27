"""EXP-010 pilot analysis (PRERUN D18), frozen before the pilot and before any adapter output.

Pure logic: every input (records, measurements, comparator bytes, Git facts) is passed in, so the
checks can be tested on synthetic fixtures; ``benchmarks/issue4_exp010_analyze.py`` reads the files
and the Git state and calls :func:`analyze`. Nothing here is tuned on predictions.

Order of work:

1. **Integrity before scoring.** The raw records must be committed and unchanged (preserved before
   any score exists); the code that trained, evaluated and analyses must be the approved commit
   (later commits may only add evidence under ``docs/evidence/``); the EXP-009 comparators and the
   accepted EXP-009 results must match their pinned SHA-256; the training record must show the
   approved pilot (200 optimizer updates, 800 micro-steps, every planned loss and save record, no
   stop, the final adapter equal to the update-200 save, the pinned base files, the sequence's
   common baseline, no gradient checkpointing); the evaluation record must name that adapter and
   its training commit and contain exactly the 5 warm-ups (never scored) and the 160 validation
   records, unique and in file order; the corpus must hold only the train and validation splits
   (the sealed final test is never read).
2. **Safety, independently of validity.** Every recoverable adapter row with a known validation ID
   (duplicates included) is reviewed by the Guard (``axbench.evaluate/0.2``), even when the run is
   invalid or other rows are malformed; unreviewable rows are counted, and the coverage is stated.
3. **Scoring** with ``axbench.score/0.2`` and ``axbench.evaluate/0.2`` (Guard policy 2): every
   record, errors and empty outputs included, stays in every denominator. The comparators are
   rescored with the same code and must reproduce the accepted EXP-009 per-example results.
4. **Decisions** exactly as pre-registered: :func:`axbench.pilotgates.pilot_outcome` (D13) with
   vectors paired by example ID and the frozen lineage clusters, and
   :func:`axbench.pilotgates.deployment_gates` (D14) as a descriptive reading.
5. **Descriptive reports** (D15): paired comparisons, slices, losses, resources, latency and
   uncertainty. No population inference; McNemar only as an IID illustration.

Revision 1 of the freeze (lead's D18 review): malformed or mistyped inputs give a structured
``invalid_run`` and never an exception; a measurement is usable only if it is a finite,
non-negative, non-boolean number, and an unusable one is reported as ``null`` with its status;
the result contains no NaN or infinity (strict JSON); provenance is checked against the pinned
checkpoint files, the common baseline, the monitor records and the adapter hash records.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .evaluate import HARNESS_VERSION, evaluate, guard_layer
from .pilotgates import GATES_VERSION, VALIDATION_N, PilotRun, deployment_gates, pilot_outcome
from .score import SCORE_VERSION, STATUSES
from .selection import better
from .stats import paired
from .trainwatch import (
    LEARNING_RATE,
    LORA_PARAMETERS,
    NUM_LAYERS,
    SEED,
    check_training_records,
    schedule,
)

ANALYSIS_VERSION = "exp010.analysis/3"  # /1 6b10882, /2 ac444cf; revised before any use
SYSTEM = "qwen3-1.7b-lora-exp010"
UNTUNED = "qwen3-1.7b-untuned"
RULES = "rules"
HARNESS = "exp010.harness/3"  # the harness version the pilot and its evaluation must record
EXPECTED_LINEAGES = 24
WARMUP_COUNT = 5
CORPUS_SPLITS = ("train", "validation")  # the sealed final test is not in the repository
MANIFEST_SHA256 = "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"
EXPORT_SHA256 = {
    "train.jsonl": "437c2ef0ce44bdedcb865f7f115ef2b7eda461063e127b5e4cb77da20ec99b31",
    "valid.jsonl": "40f83a3ca85a1802d6877c4d09d6d620dcb416b4ee9169b1417239d1e85d4361",
}
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# The accepted checkpoint record (EXP-009 provenance/qwen3-1.7b.json): (bytes, SHA-256) per file.
BASE_FILES = {
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "config.json": (726, "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"),
    "generation_config.json": (
        239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model-00001-of-00002.safetensors": (
        3441185608, "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
    "model-00002-of-00002.safetensors": (
        622329984, "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
    "model.safetensors.index.json": (
        25605, "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d"),
    "tokenizer.json": (
        11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (
        9732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}  # fmt: skip
PINNED_TRAINABLE_PARAMETERS = 458_752
PROMPT_VERSION = "axiom.prompt/1"
DECODING = {
    "sampler": "greedy (argmax)",
    "max_new_tokens": 256,
    "enable_thinking": False,
    "batch_size": 1,
    "stop_token_ids": [151643, 151645],
}
EXP009 = "docs/evidence/experiments/EXP-009-untuned-baselines"
COMPARATORS = {
    UNTUNED: (f"{EXP009}/runs_attempt2_qwen3-1.7b/qwen3-1.7b/main_validation.jsonl",
              "cace592012340cb45807250e879ed22c68fef83cd78769d252574bc23b85550d"),
    RULES: (f"{EXP009}/rules_validation.jsonl",
            "7645eea783caeb1b80ba216d715bd06dae5154aaa425ef88a32906c0cfb11b9a"),
}  # fmt: skip
EXP009_RESULTS = (f"{EXP009}/results.json",
                  "fe10efea2448c4c9b2d514e4ae68c2dc359c73b4de1b7288748639f7d439c4a2")  # fmt: skip
CHECKPOINT_RECORD = f"{EXP009}/provenance/qwen3-1.7b.json"
# After the approved commit, only evidence may change (raw records, write-ups, logs).
EVIDENCE_ONLY_PREFIXES = ("docs/evidence/",)
ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors",
                 *(f"update_{u:04d}.safetensors"
                   for u in schedule("pilot").save_updates))  # fmt: skip
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 2026
LIST_LIMIT = 5  # problems name at most this many examples of one kind


# ------------------------------------------------------------------ raw input parsing


@dataclass(frozen=True)
class Malformed:
    """A raw JSONL line that is not JSON (kept in place, never dropped or repaired)."""

    line: int
    error: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_jsonl(data: bytes) -> list[Any]:
    """Strict parse of a pinned file: a malformed line raises ``ValueError``."""
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def parse_jsonl_lines(data: bytes) -> list[Any]:
    """Every physical line (split on ``\\n``), in order: its JSON value, or :class:`Malformed`
    for a line that is blank, not UTF-8 or not JSON. Never raises. The newline ending the last line
    is a terminator, not an extra row; each line is decoded on its own, so one undecodable line
    leaves every other line readable (freeze revision 2)."""
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()  # the normal terminal newline
    out: list[Any] = []
    for n, raw in enumerate(lines, start=1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            out.append(Malformed(n, "not UTF-8"))
            continue
        if not line.strip():
            out.append(Malformed(n, "blank line"))
            continue
        try:
            out.append(json.loads(line))
        except (ValueError, RecursionError) as exc:
            out.append(Malformed(n, type(exc).__name__))
    return out


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _same_json(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except (TypeError, ValueError):
        return False


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))  # fmt: skip


def measurement_status(value: Any) -> str:
    """``ok`` only for a finite, non-negative, non-boolean number."""
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "not_a_number"
    try:
        number = float(value)
    except OverflowError:  # an integer beyond the float range (freeze revision 2)
        return "out_of_range"
    if not math.isfinite(number):
        return "not_finite"
    return "negative" if number < 0 else "ok"


def measurement(value: Any) -> Any:
    """The value if it is a usable measurement, otherwise ``None`` (reported as null)."""
    return value if measurement_status(value) == "ok" else None


def _first(items: Sequence[Any]) -> list[Any]:
    return list(items[:LIST_LIMIT])


# ------------------------------------------------------------------ record checks


def _row_problems(rows: list[Any]) -> list[str]:
    malformed = [r.line for r in rows if isinstance(r, Malformed)]
    not_object = [i for i, r in enumerate(rows, 1)
                  if not isinstance(r, dict | Malformed)]  # fmt: skip
    bad_id = [i for i, r in enumerate(rows, 1)
              if isinstance(r, dict) and not isinstance(r.get("example_id"), str)]  # fmt: skip
    problems = []
    if malformed:
        problems.append(f"{len(malformed)} malformed JSON lines (lines {_first(malformed)})")
    if not_object:
        problems.append(f"{len(not_object)} rows are not objects (rows {_first(not_object)})")
    if bad_id:
        problems.append(f"{len(bad_id)} rows without a string example_id (rows {_first(bad_id)})")
    return problems


def check_records(records: Any, expected_ids: Sequence[str], system: str) -> list[str]:
    """Exactly the expected IDs, unique, in the expected (file) order, one system, known statuses.

    Errors, timeouts and empty outputs are valid records (they are scored as wrong and stay in
    every denominator); a missing, duplicated, unexpected, reordered, malformed or mistyped record
    is not. Never raises on malformed input.
    """
    if not isinstance(records, list):
        return ["records are missing or not a list"]
    problems = _row_problems(records)
    rows = [r for r in records if isinstance(r, dict)]
    ids = [r["example_id"] for r in rows if isinstance(r.get("example_id"), str)]
    counts = Counter(ids)
    expected = set(expected_ids)
    missing = [i for i in expected_ids if i not in counts]
    duplicated = sorted(i for i, c in counts.items() if c > 1)
    unexpected = sorted(i for i in counts if i not in expected)
    if missing:
        problems.append(f"{len(missing)} missing records (first {_first(missing)})")
    if duplicated:
        problems.append(f"duplicated IDs {_first(duplicated)}")
    if unexpected:
        problems.append(f"unexpected IDs {_first(unexpected)}")
    if not problems and ids != list(expected_ids):
        problems.append("records are not in file order")
    if any(r.get("system") != system for r in rows):
        problems.append(f"records not labelled {system!r}")
    statuses = {r.get("status") for r in rows if isinstance(r.get("status"), str)}
    if any(not isinstance(r.get("status"), str) for r in rows) or statuses - set(STATUSES):
        problems.append(f"unknown statuses {_first(sorted(statuses - set(STATUSES)))}")
    if any(r.get("status") == "ok" and not isinstance(r.get("raw_output"), str) for r in rows):
        problems.append("an ok record without a raw output string")
    return problems


def check_comparator(name: str, data: bytes | None, expected_ids: Sequence[str]
                     ) -> tuple[list[dict[str, Any]] | None, list[str]]:  # fmt: skip
    """Pinned hash first; records are parsed only from the pinned bytes."""
    path, pinned = COMPARATORS[name]
    if data is None:
        return None, [f"comparator {name}: {path} is missing"]
    if sha256(data) != pinned:
        return None, [f"comparator {name}: SHA-256 differs from the pin (tampered or replaced)"]
    records = parse_jsonl_lines(data)
    problems = [f"comparator {name}: {p}" for p in check_records(records, expected_ids, name)]
    return (None if problems else records), problems


def check_corpus(splits: Sequence[str], manifest_sha256: str, validation: Sequence[dict]
                 ) -> list[str]:  # fmt: skip
    problems = []
    if tuple(sorted(splits)) != CORPUS_SPLITS:
        problems.append(f"corpus splits {sorted(splits)}; only train and validation may be read")
    if manifest_sha256 != MANIFEST_SHA256:
        problems.append("corpus manifest differs from the pinned hash")
    ids = [e["id"] for e in validation]
    if len(ids) != VALIDATION_N or len(set(ids)) != VALIDATION_N:
        problems.append(f"validation has {len(ids)} IDs ({len(set(ids))} unique), not 160")
    if any(e.get("split") != "validation" for e in validation):
        problems.append("a validation example is not from the validation split")
    lineages = {e.get("lineage_id") for e in validation}
    if len(lineages) != EXPECTED_LINEAGES or None in lineages:
        problems.append(f"{len(lineages)} lineage clusters, the frozen design has 24")
    return problems


def check_frozen_code(changed_since_approved: Sequence[str] | None, label: str) -> list[str]:
    """Code may not change after the approved commit; only evidence may be added or edited.

    ``changed_since_approved`` is ``git diff --name-only <approved> <commit>``; ``None`` means the
    approved commit is not an ancestor (or is unknown), which is itself a failure."""
    if changed_since_approved is None:
        return [f"{label}: the approved commit is not an ancestor"]
    code = sorted(p for p in changed_since_approved if not p.startswith(EVIDENCE_ONLY_PREFIXES))
    return [f"{label}: files outside docs/evidence/ changed since the approved commit: {code[:5]}"
            ] if code else []  # fmt: skip


def check_preserved(inputs: Mapping[str, tuple[str | None, str | None]]) -> list[str]:
    """Raw evidence is preserved before scoring: each input ``path -> (working-tree SHA-256,
    committed SHA-256 at HEAD)`` must exist, be committed and be unchanged."""
    problems = []
    for path, (work, head) in sorted(inputs.items()):
        if work is None:
            problems.append(f"{path}: missing")
        elif head is None:
            problems.append(f"{path}: not committed before scoring")
        elif work != head:
            problems.append(f"{path}: differs from the committed version")
    return problems


# ------------------------------------------------------------------ provenance checks


@dataclass(frozen=True)
class SequenceRecord:
    """What the analysis reads from one supervised sequence directory."""

    exit_status: str | None
    monitor_stop: bool
    done: bool
    baseline: Any = None  # baseline.json
    monitor: Any = None  # monitor.jsonl rows (parse_jsonl_lines)
    monitor_ready: Any = None  # monitor_ready.json


def check_monitor(label: str, rows: Any) -> list[str]:
    """The external monitor's records agree with a clean exit: valid samples, none with a stop,
    and one final record at the end."""
    if not isinstance(rows, list) or not rows:
        return [f"{label} monitor: no records"]
    problems = [f"{label} monitor: {p}" for p in _row_problems(rows) if "example_id" not in p]
    samples = [r for r in rows if isinstance(r, dict) and not r.get("final")]
    finals = [i for i, r in enumerate(rows) if isinstance(r, dict) and r.get("final") is True]
    if finals != [len(rows) - 1]:
        problems.append(f"{label} monitor: not exactly one final record at the end")
    if not samples:
        problems.append(f"{label} monitor: no sample")
    for r in samples:
        if r.get("stop") is not None:
            problems.append(f"{label} monitor: a sample recorded a stop")
            break
    for key in ("swap_used", "pressure_level"):  # the stop-rule fields (supervise.RULE_KEYS)
        if any(measurement_status(r.get(key)) != "ok" for r in samples):
            problems.append(f"{label} monitor: a sample has no usable {key}")
    return problems


def check_sequence(label: str, seq: SequenceRecord) -> list[str]:
    """The supervised sequence ended with exit 0, its done marker, no monitor stop, a ready
    monitor and monitor records consistent with that."""
    problems = []
    status = seq.exit_status if isinstance(seq.exit_status, str) else None
    if status is None or status.split()[:1] != ["0"]:
        problems.append(f"{label} sequence: exit status {seq.exit_status!r}, not 0")
    if seq.monitor_stop:
        problems.append(f"{label} sequence: the external monitor wrote a stop record")
    if not seq.done:
        problems.append(f"{label} sequence: no done marker")
    if not isinstance(seq.monitor_ready, dict):
        problems.append(f"{label} sequence: no monitor-ready record")
    if not isinstance(seq.baseline, dict) or not seq.baseline:
        problems.append(f"{label} sequence: no common baseline record")
    return problems + check_monitor(label, seq.monitor)


def _not_stopped(meas: Mapping[str, Any], label: str) -> list[str]:
    problems = []
    if meas.get("status") != "completed":
        problems.append(f"{label}: status {meas.get('status')!r}")
    for key in ("stop_reason", "failure", "finalization_errors", "external_monitor_stop"):
        if meas.get(key):
            problems.append(f"{label}: {key} recorded")
    return problems


def _environment(meas: Mapping[str, Any], label: str) -> list[str]:
    env = _obj(meas.get("environment"))
    problems = []
    if _obj(env.get("git")).get("dirty") is not False:
        problems.append(f"{label}: tree not recorded as clean")
    if env.get("runner_version") != HARNESS:
        problems.append(f"{label}: harness {env.get('runner_version')!r}, expected {HARNESS!r}")
    return problems


def _base_files(model: Any, label: str) -> list[str]:
    """Revision and every one of the ten checkpoint files against the accepted record."""
    m = _obj(model)
    files = _obj(m.get("files"))
    got = {k: (_obj(v).get("bytes"), _obj(v).get("sha256")) for k, v in files.items()}
    problems = []
    if m.get("revision") != BASE_REVISION:
        problems.append(f"{label}: base revision differs from the pin")
    if got != BASE_FILES:
        bad = sorted(k for k in set(got) | set(BASE_FILES) if got.get(k) != BASE_FILES.get(k))
        problems.append(f"{label}: base files differ from the pinned checkpoint record {bad[:5]}")
    return problems


def _common_baseline(meas: Mapping[str, Any], baseline: Any, label: str) -> list[str]:
    rec = _obj(meas.get("stop_rule_baseline"))
    if rec.get("source") != "sequence baseline.json":
        return [f"{label}: not run under the supervised sequence baseline"]
    if (
        not isinstance(baseline, dict)
        or not baseline
        or any(not _same_json(rec.get(k), v) for k, v in baseline.items())
    ):
        return [f"{label}: stop-rule baseline differs from the sequence's baseline.json"]
    return []


def _resource_summaries(meas: Mapping[str, Any], label: str, keys: Sequence[str]) -> list[str]:
    problems = []
    for key in keys:
        summary = meas.get(key)
        if not isinstance(summary, dict) or summary.get("stopped") is not None:
            problems.append(f"{label}: {key} summary missing or records a stop")
    mon = _obj(meas.get("resource_monitor"))
    for key in ("max_cumulative_swap_growth_bytes", "max_pressure_level"):
        if measurement_status(mon.get(key)) != "ok":
            problems.append(f"{label}: resource monitor has no usable {key}")
    return problems


def _adapter_file_records(files: Mapping[str, Any], label: str) -> list[str]:
    bad = sorted(k for k, v in files.items()
                 if not _is_sha(_obj(v).get("sha256"))
                 or measurement_status(_obj(v).get("bytes")) != "ok"
                 or not _obj(v).get("bytes"))  # fmt: skip
    return [f"{label}: adapter file records without a SHA-256 string or size {bad[:5]}"
            ] if bad else []  # fmt: skip


def check_training(meas: Any, log: Any, approved: str, baseline: Any = None) -> list[str]:
    """The approved pilot, completed as planned (D4-D8): re-derived from the records.

    Never raises: a missing, malformed or mistyped record is a problem, not an exception."""
    sched = schedule("pilot")
    p: list[str] = []
    if not isinstance(meas, dict):
        p.append("training: measurements missing or not an object")
        meas = {}
    if not isinstance(log, list):
        p.append("training log: missing or not a list")
        log = []
    p += [f"training log: {x}" for x in _row_problems(log) if "example_id" not in x]
    rows = [r for r in log if isinstance(r, dict)]
    p += [*_not_stopped(meas, "training"), *_environment(meas, "training")]
    p += _common_baseline(meas, baseline, "training")
    if meas.get("experiment") != "EXP-010" or meas.get("phase") != "pilot":
        p.append("training: not the EXP-010 pilot phase")
    if meas.get("system") != SYSTEM:
        p.append("training: system name differs")
    if _obj(_obj(meas.get("environment")).get("git")).get("commit") != approved:
        p.append("training: commit differs from the approved commit")
    if not _same_json(meas.get("schedule"), sched.as_dict()):
        p.append("training: schedule differs from the pilot plan")
    if meas.get("optimizer_steps") != sched.updates:
        p.append(f"training: optimizer steps {meas.get('optimizer_steps')!r}, planned 200")
    if meas.get("updates_recorded") != sched.updates:
        p.append(f"training: {meas.get('updates_recorded')!r} updates recorded, planned 200")
    cfg = _obj(meas.get("configuration"))
    planned = {"base": "qwen3-1.7b", "lora_parameters": LORA_PARAMETERS,
               "num_layers": NUM_LAYERS, "learning_rate": LEARNING_RATE, "seed": SEED}  # fmt: skip
    if any(not _same_json(cfg.get(k), v) for k, v in planned.items()):
        p.append("training: configuration differs from the plan")
    if cfg.get("grad_checkpoint") is not False or cfg.get("declared_change") is not None:
        p.append("training: a predeclared change (C1/C2) needs its own prospective approval and "
                 "a new analysis version; the initial pilot runs without one")  # fmt: skip
    if _obj(meas.get("lora")).get("trainable_parameters") != PINNED_TRAINABLE_PARAMETERS:
        p.append("training: trainable parameter count differs from the pin")
    p += _base_files(meas.get("model"), "training")
    data = _obj(meas.get("data"))
    if (
        data.get("corpus_manifest_sha256") != MANIFEST_SHA256
        or {k: _obj(v).get("sha256") for k, v in _obj(data.get("files")).items()} != EXPORT_SHA256
    ):
        p.append("training: data hashes differ from the pinned export")
    eval_set = _obj(meas.get("eval_set"))
    if eval_set.get("name") != "validation" or eval_set.get("examples") != VALIDATION_N:
        p.append("training: the loss curve was not evaluated on the 160 validation examples")
    if not _obj(meas.get("completion_checks")).get("passed"):
        p.append("training: completion checks did not pass")
    if meas.get("final_adapter_equals_last_update_save") is not True:
        p.append("training: final adapter is not the update-200 save")
    p += _resource_summaries(meas, "training", ("resource_monitor", "throughput"))
    files = _obj(meas.get("adapter_files"))
    if sorted(files) != sorted(ADAPTER_FILES):
        p.append(f"training: adapter files {sorted(files)} differ from the plan")
    p += _adapter_file_records(files, "training")
    p += [f"training log: {x}" for x in check_training_records(rows, sched)]
    evals = [r for r in rows if r.get("kind") == "eval"]
    if any(r.get("set") != "validation" for r in evals):
        p.append("training log: an evaluation record is not labelled 'validation'")
    save_rows = [r for r in rows if r.get("kind") == "save"]
    bad_updates = [
        r.get("update")
        for r in save_rows
        if not isinstance(r.get("update"), int) or isinstance(r.get("update"), bool)
    ]
    if bad_updates:  # freeze revision 2: never hash or format an unvalidated value
        p.append(f"training log: {len(bad_updates)} save records without an integer update")
    saves = {
        r["update"]: r.get("sha256")
        for r in save_rows
        if isinstance(r.get("update"), int) and not isinstance(r.get("update"), bool)
    }
    final = _obj(files.get("adapters.safetensors")).get("sha256")
    if not _is_sha(final) or final != saves.get(sched.updates):
        p.append("training: final adapter hash differs from the update-200 save record")
    for u, sha in saves.items():
        if _obj(files.get(f"update_{u:04d}.safetensors")).get("sha256") != sha:
            p.append(f"training: update-{u} save hash differs from its file record")
    updates = [r for r in rows if r.get("kind") == "update"]
    if not updates or updates[-1].get("micro_steps") != sched.micro_steps:
        p.append("training: the last update is not at micro-step 800")
    return p


def check_warmups(records: Any, warmup_ids: Sequence[str]) -> list[str]:
    """Exactly the 5 warm-ups in order, as well-formed records of this system with a known
    status. They are never scored, so a failed warm-up is not itself a problem."""
    if not isinstance(records, list):
        return ["warm-up records missing or not a list"]
    problems = [f"warm-up {x}" for x in check_records(records, warmup_ids, SYSTEM)]
    return problems


def check_evaluation(
    meas: Any,
    train_meas: Any,
    approved: str,
    warmup_records: Any,
    warmup_ids: Sequence[str],
    validation_ids: Sequence[str],
    baseline: Any = None,
) -> list[str]:
    """The fresh-process evaluation of exactly the trained final adapter (D12). Never raises."""
    p: list[str] = []
    if not isinstance(meas, dict):
        p.append("evaluation: measurements missing or not an object")
        meas = {}
    train_meas = _obj(train_meas)
    p += [*_not_stopped(meas, "evaluation"), *_environment(meas, "evaluation")]
    p += _common_baseline(meas, baseline, "evaluation")
    if meas.get("experiment") != "EXP-010" or meas.get("phase") != "main":
        p.append("evaluation: not the EXP-010 evaluation")
    if meas.get("system") != SYSTEM:
        p.append("evaluation: system name differs")
    if meas.get("validation_completed") is not True:
        p.append("evaluation: validation not completed")
    if meas.get("counts") != {"validation": VALIDATION_N, "warmup": WARMUP_COUNT}:
        p.append(f"evaluation: record counts {meas.get('counts')!r}")
    if meas.get("corpus_manifest_sha256") != MANIFEST_SHA256:
        p.append("evaluation: corpus manifest differs from the pin")
    if meas.get("prompt_version") != PROMPT_VERSION or not _same_json(meas.get("decoding"),
                                                                       DECODING):  # fmt: skip
        p.append("evaluation: prompt or decoding differs from EXP-009")
    p += _base_files(meas.get("model"), "evaluation")
    if not _same_json(meas.get("model"), train_meas.get("model")):
        p.append("evaluation: base checkpoint differs from the one trained")
    p += _resource_summaries(meas, "evaluation", ("resource_monitor", "validation_slowdown"))
    ad = _obj(meas.get("adapter"))
    files = _obj(train_meas.get("adapter_files"))
    final = _obj(files.get("adapters.safetensors"))
    config = _obj(files.get("adapter_config.json"))
    if (
        ad.get("phase") != "pilot"
        or ad.get("file") != "adapters.safetensors"
        or not _is_sha(ad.get("sha256"))
        or ad.get("sha256") != final.get("sha256")
        or ad.get("bytes") != final.get("bytes")
    ):
        p.append("evaluation: adapter is not the trained final pilot adapter")
    if not _is_sha(ad.get("config_sha256")) or ad.get("config_sha256") != config.get("sha256"):
        p.append("evaluation: adapter configuration differs from the trained one")
    if ad.get("training_commit") != approved:
        p.append("evaluation: adapter training commit is not the approved commit")
    if meas.get("warmup_ids") != list(warmup_ids) or len(warmup_ids) != WARMUP_COUNT:
        p.append("evaluation: warm-up IDs are not the 5 pinned train warm-ups")
    p += [f"evaluation: {x}" for x in check_warmups(warmup_records, warmup_ids)]
    if set(warmup_ids) & set(validation_ids):
        p.append("evaluation: a warm-up ID is a validation ID")
    return p


# ------------------------------------------------------------------ descriptive helpers


def _quartiles(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    q = statistics.quantiles(values, n=4) if len(values) >= 2 else [values[0]] * 3
    return {"n": len(values), "median": statistics.median(values), "q1": q[0], "q3": q[2],
            "min": min(values), "max": max(values)}  # fmt: skip


def latency(records: Sequence[Any]) -> dict[str, Any]:
    """Warm latency over the validation requests. The median is a G11 measurement only when every
    one of the 160 requests has a usable latency (finite, non-negative, not a boolean; the EXP-009
    rule); otherwise it is ``None`` and G11 is not assessable."""
    rows = [r for r in records if isinstance(r, dict)]
    status = Counter(measurement_status(r.get("elapsed_ms")) for r in rows)
    vals = [float(r["elapsed_ms"]) for r in rows if measurement_status(r.get("elapsed_ms")) == "ok"]
    complete = len(vals) == len(records) == VALIDATION_N
    return {"ms": _quartiles(vals), "requests_with_latency": len(vals),
            "latency_status": dict(sorted(status.items())),
            "median_for_gate": statistics.median(vals) if complete else None}  # fmt: skip


def generation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gen = [float(r["generation_tokens"]) for r in records
           if measurement_status(r.get("generation_tokens")) == "ok"]  # fmt: skip
    return {
        "generated_tokens": _quartiles(gen),
        "requests_hitting_256_token_cap": sum(r.get("finish_reason") == "length" for r in records),
        "finish_reasons": dict(sorted(Counter(str(r.get("finish_reason")) for r in records)
                                      .items())),
        "status": dict(sorted(Counter(str(r.get("status")) for r in records).items())),
    }  # fmt: skip


def training_report(meas: Mapping[str, Any], log: Sequence[dict[str, Any]]) -> dict[str, Any]:
    updates = [r for r in log if r.get("kind") == "update"]
    evals = [r for r in log if r.get("kind") == "eval"]
    secs = [r["update_seconds"] for r in updates if measurement_status(r.get("update_seconds"))
            == "ok"]  # fmt: skip
    return {
        "train_loss_per_update": [[r["update"], r.get("train_loss")] for r in updates],
        "validation_loss": [[r["update"], r.get("loss"), r.get("seconds")] for r in evals],
        "loss_note": "validation loss is token-weighted over completion tokens; it is not an "
        "endpoint and no checkpoint is selected by it",
        "trainable_parameters": _obj(meas.get("lora")).get("trainable_parameters"),
        "wall_seconds": measurement(meas.get("wall_seconds")),
        "train_loop_seconds": measurement(meas.get("train_loop_seconds")),
        "train_loop_note": "train_loop_seconds starts after the update-0 evaluation; it includes "
        "the later evaluations, the saves and the callbacks",
        "sum_update_seconds": sum(secs) if secs else None,
        "update_seconds": _quartiles(secs),
        "throughput": meas.get("throughput"),
        "trained_completion_tokens": updates[-1].get("trained_tokens") if updates else None,
        "mlx_peak_bytes": measurement(meas.get("mlx_peak_bytes")),
        "process_peak_rss_bytes": measurement(meas.get("rss_peak_process_bytes")),
        "resource_monitor": meas.get("resource_monitor"),
        "adapter_files": meas.get("adapter_files"),
    }


EVAL_MEASUREMENTS = ("cold_load_seconds", "load_mlx_peak_bytes", "mlx_peak_bytes_load_and_warmup",
                     "mlx_peak_bytes_load_warmup_validation", "rss_peak_after_load_bytes",
                     "rss_peak_after_validation_bytes", "rss_peak_process_bytes")  # fmt: skip


def evaluation_resources(meas: Mapping[str, Any]) -> dict[str, Any]:
    """Usable measurements as numbers; anything else as ``null`` with its status."""
    out: dict[str, Any] = {k: measurement(meas.get(k)) for k in EVAL_MEASUREMENTS}
    out["measurement_status"] = {k: measurement_status(meas.get(k)) for k in EVAL_MEASUREMENTS}
    out["validation_slowdown"] = meas.get("validation_slowdown")
    out["resource_monitor"] = meas.get("resource_monitor")
    out["note"] = "RSS, MLX peak, swap and pressure are separate measurements, never summed"
    return out


def safety_scan(by_id: Mapping[str, dict[str, Any]], records: Any) -> dict[str, Any]:
    """Guard review of every recoverable adapter row whose ID is a validation ID, duplicates
    included, whatever the validity of the run: the unauthorized-dispatch count never depends on
    it. Rows that cannot be reviewed are counted by reason and the coverage is stated."""
    rows = records if isinstance(records, list) else []
    reasons: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    reviewed_ids: set[str] = set()
    reviewed = dispatches = 0
    for r in rows:
        if isinstance(r, Malformed):
            reasons["malformed_line"] += 1
            continue
        if not isinstance(r, dict):
            reasons["not_an_object"] += 1
            continue
        i = r.get("example_id")
        if not isinstance(i, str):
            reasons["example_id_not_a_string"] += 1
            continue
        if i not in by_id:
            reasons["unknown_id"] += 1
            continue
        try:
            review = guard_layer([by_id[i]], [r])
        except Exception:  # noqa: BLE001 - an unreviewable row is counted, never dropped
            reasons["review_error"] += 1
            continue
        reviewed += 1
        reviewed_ids.add(i)
        dispatches += review["unauthorized_dispatch_count"]
        statuses.update(review["guard_status"])
    return {
        "records_present": isinstance(records, list),
        "rows": len(rows),
        "records_reviewed": reviewed,
        "records_not_reviewable": sum(reasons.values()),
        "not_reviewable_by_reason": dict(sorted(reasons.items())),
        "validation_ids_reviewed": len(reviewed_ids),
        "complete_coverage": isinstance(records, list) and not reasons
        and reviewed_ids == set(by_id),
        "unauthorized_dispatches": dispatches,
        "guard_status": dict(sorted(statuses.items())),
        "confirmations_given": 0,
        "note": "a zero count covers only the reviewed rows; see complete_coverage",
    }  # fmt: skip


def by_id(report: Mapping[str, Any], field: str) -> dict[str, bool]:
    return {r["id"]: r[field] for r in report["per_example"]}


def paired_by_id(a: Mapping[str, bool], b: Mapping[str, bool], clusters: Mapping[str, str],
                 ids: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """Paired comparison (b - a) formed by example ID, never by position."""
    return paired([a[i] for i in ids], [b[i] for i in ids], [clusters[i] for i in ids],
                  reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)  # fmt: skip


def d14_summary(adapter: Mapping[str, Any], untuned_exact: int, dispatches: int,
                lat: Mapping[str, Any], eval_meas: Mapping[str, Any]
                ) -> dict[str, Any]:  # fmt: skip
    """The D14 count summary. G12 uses the evaluation process: its lifetime peak RSS and the MLX
    peak since loading (load, warm-up and validation; lead decision 9). An unusable resource
    value is passed as ``None`` (not assessable)."""
    raw = adapter["raw_candidate"]
    lang = adapter["slices"]["language"]

    def cn(d: Mapping[str, Any]) -> tuple[int, int]:
        return d["count"], d["n"]

    return {
        "exact": cn(raw["primary_semantic_exact_match"]),
        "schema_valid": cn(raw["candidate_schema_valid"]),
        "exact_tr": cn(lang.get("tr", {"count": 0, "n": 0})),
        "exact_en": cn(lang.get("en", {"count": 0, "n": 0})),
        "reject_recall": cn(raw["reject_recall"]),
        "false_reject_gold_plan": cn(raw["false_reject_among_gold_plan"]),
        "clarify_recall": cn(raw["clarify_recall"]),
        "wrong_executable": cn(raw["wrong_executable_proposals"]["per_all"]),
        "unauthorized_dispatches": dispatches,
        "untuned_exact_count": untuned_exact,
        "median_warm_latency_ms": lat["median_for_gate"],
        "peak_rss_bytes": measurement(eval_meas.get("rss_peak_process_bytes")),
        "mlx_peak_bytes": measurement(eval_meas.get("mlx_peak_bytes_load_warmup_validation")),
    }


def strict_json(value: Any, path: str = "$", found: list[str] | None = None) -> Any:
    """A copy with every NaN or infinity replaced by ``None``; the paths are appended to
    ``found``. Raw evidence is never changed; only the reported values are normalized."""
    if found is None:
        found = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(path)
        return None
    if isinstance(value, dict):
        return {k: strict_json(v, f"{path}.{k}", found) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [strict_json(v, f"{path}[{i}]", found) for i, v in enumerate(value)]
    return value


# ------------------------------------------------------------------ the analysis


@dataclass
class Inputs:
    """Everything the analysis reads, already loaded; nothing else is consulted."""

    approved_commit: str
    corpus_splits: Sequence[str]
    corpus_manifest_sha256: str
    validation: Sequence[dict[str, Any]]
    warmup_ids: Sequence[str]  # the first five train IDs (runwatch.warmup_ids)
    train_meas: Any
    train_log: Any
    train_sequence: SequenceRecord
    eval_meas: Any
    eval_sequence: SequenceRecord
    warmup_records: Any
    adapter_records: Any
    comparator_bytes: Mapping[str, bytes | None]
    exp009_results_bytes: bytes | None
    rules_rerun: Sequence[dict[str, Any]] | None  # axiom.rules/0.1 rerun on the frozen file
    changed_since_approved: Mapping[str, Sequence[str] | None] = field(default_factory=dict)
    preserved: Mapping[str, tuple[str | None, str | None]] = field(default_factory=dict)
    input_hashes: Mapping[str, str | None] = field(default_factory=dict)
    # Problems found while reading the files (unreadable or malformed JSON), by section.
    input_problems: Mapping[str, Sequence[str]] = field(default_factory=dict)


def _exp009_reproduced(data: bytes | None, reports: Mapping[str, Any]) -> list[str]:
    """The rescored comparators must equal the accepted EXP-009 per-example results."""
    if data is None:
        return ["EXP-009 results.json is missing"]
    if sha256(data) != EXP009_RESULTS[1]:
        return ["EXP-009 results.json differs from the pinned hash"]
    accepted = json.loads(data.decode("utf-8"))["systems"]
    problems = []
    for name in (UNTUNED, RULES):
        if name not in reports:
            continue
        old = {r["id"]: (r["exact_match"], r["schema_valid"])
               for r in accepted[name]["per_example"]}  # fmt: skip
        new = {r["id"]: (r["exact_match"], r["schema_valid"])
               for r in reports[name]["per_example"]}  # fmt: skip
        if old != new:
            problems.append(f"rescored {name} differs from the accepted EXP-009 results")
    return problems


SECTIONS = ("corpus", "frozen_code", "raw_evidence_preserved", "training", "evaluation",
            "adapter_records", "comparators")  # fmt: skip


def analyze(x: Inputs) -> dict[str, Any]:
    """The frozen analysis. Never raises on malformed run inputs; the result is strict JSON.

    The safety review runs first, on its own; if any later step raises, the result is a
    structured ``invalid_run`` that keeps the safety counts and coverage (freeze revision 2)."""
    examples = {e["id"]: e for e in x.validation}
    safety = safety_scan(examples, x.adapter_records)
    try:
        return _analyze(x, safety)
    except Exception as exc:  # noqa: BLE001 - reported as an invalid run, never raised
        return failure_report(x.approved_commit, x.input_hashes, safety,
                              f"{type(exc).__name__}: {exc}")  # fmt: skip


def failure_report(approved: str, input_hashes: Mapping[str, Any] | None,
                   safety: Mapping[str, Any] | None, error: str) -> dict[str, Any]:  # fmt: skip
    """A structured ``invalid_run`` when the analysis could not complete; already computed
    safety counts and coverage are kept, never discarded."""
    adapter = (
        dict(safety)
        if safety
        else {"complete_coverage": False, "note": "not computed: the analysis did not reach it"}
    )
    dispatches = adapter.get("unauthorized_dispatches")
    return _finish({
        "experiment": "EXP-010",
        "analysis_version": ANALYSIS_VERSION,
        "approved_commit": approved,
        "input_sha256": dict(sorted(input_hashes.items())) if input_hashes else None,
        "pilot_outcome": {
            "outcome": "invalid_run",
            "problems": {"analysis": [error]},
            "unauthorized_dispatches": dispatches,
            "safety_flag": bool(dispatches),
            "safety_review_complete": bool(adapter.get("complete_coverage")),
        },
        "safety": {"adapter": adapter},
        "deployment_gates": {"computed": False, "reason": "invalid run: no result"},
        "descriptive": {"computed": False, "reason": "invalid run: no result"},
    })  # fmt: skip


def _scoring_copy(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """The adapter records as passed to the scorer: an unusable ``elapsed_ms`` (for example an
    integer beyond the float range, which the scorer cannot convert) is passed as ``None``. Only
    this reported latency field is normalized; accuracy fields are untouched and raw evidence is
    never changed (freeze revision 2)."""
    out, ids = [], []
    for r in records:
        if r.get("elapsed_ms") is not None and measurement_status(r.get("elapsed_ms")) != "ok":
            out.append({**r, "elapsed_ms": None})
            ids.append(r["example_id"])
        else:
            out.append(r)
    return out, ids


def _analyze(x: Inputs, safety: dict[str, Any]) -> dict[str, Any]:
    val = list(x.validation)
    ids = [e["id"] for e in val]
    examples = {e["id"]: e for e in val}
    clusters = {e["id"]: e["lineage_id"] for e in val}
    train_rows = x.train_log if isinstance(x.train_log, list) else None
    problems: dict[str, list[str]] = {
        "corpus": check_corpus(x.corpus_splits, x.corpus_manifest_sha256, val),
        "frozen_code": [q for label, changed in sorted(x.changed_since_approved.items())
                        for q in check_frozen_code(changed, label)],
        "raw_evidence_preserved": check_preserved(x.preserved),
        "training": [*check_sequence("training", x.train_sequence),
                     *check_training(x.train_meas, x.train_log, x.approved_commit,
                                     x.train_sequence.baseline)],
        "evaluation": [*check_sequence("evaluation", x.eval_sequence),
                       *check_evaluation(x.eval_meas, x.train_meas, x.approved_commit,
                                         x.warmup_records, x.warmup_ids, ids,
                                         x.eval_sequence.baseline)],
        "adapter_records": check_records(x.adapter_records, ids, SYSTEM),
        "comparators": [],
    }  # fmt: skip
    for section, found in x.input_problems.items():
        problems[section if section in SECTIONS else "raw_evidence_preserved"] += list(found)
    if not x.changed_since_approved:
        problems["frozen_code"].append("no frozen-code check was supplied")
    if not x.preserved:
        problems["raw_evidence_preserved"].append("no preservation check was supplied")
    comparators: dict[str, list[dict[str, Any]]] = {}
    for name in COMPARATORS:
        recs, cp = check_comparator(name, x.comparator_bytes.get(name), ids)
        problems["comparators"] += cp
        if recs is not None:
            comparators[name] = recs
    if x.rules_rerun is not None and RULES in comparators:
        rerun = {r["example_id"]: (r.get("status"), r.get("raw_output")) for r in x.rules_rerun}
        pinned = {r["example_id"]: (r.get("status"), r.get("raw_output"))
                  for r in comparators[RULES]}  # fmt: skip
        if rerun != pinned:
            problems["comparators"].append("rules rerun differs from the pinned rules outputs")
    elif x.rules_rerun is None:
        problems["comparators"].append("no rules rerun was supplied")

    reports: dict[str, Any] = {}
    for name, recs in comparators.items():
        reports[name] = evaluate(val, recs, system=name, split="validation")
    problems["comparators"] += _exp009_reproduced(x.exp009_results_bytes, reports)
    if not problems["adapter_records"]:
        scoring, normalized = _scoring_copy(x.adapter_records)
        reports[SYSTEM] = evaluate(val, scoring, system=SYSTEM, split="validation")
        reports[SYSTEM]["elapsed_ms_passed_to_scorer_as_null"] = normalized
        if (
            reports[SYSTEM]["after_guard"]["unauthorized_dispatch_count"]
            != safety["unauthorized_dispatches"]
        ):
            problems["adapter_records"].append("Guard reviews disagree on dispatches")
    if not safety["complete_coverage"] and not problems["adapter_records"]:
        problems["adapter_records"].append("the safety review did not cover every record")
    integrity_ok = not any(problems[k] for k in ("corpus", "frozen_code", "raw_evidence_preserved",
                                                 "adapter_records", "comparators"))  # fmt: skip
    run = PilotRun(
        training_completed=not problems["training"],
        evaluation_completed=not problems["evaluation"],
        integrity_ok=integrity_ok,
    )
    adapter_exact = by_id(reports[SYSTEM], "exact_match") if SYSTEM in reports else {}
    untuned_exact = by_id(reports[UNTUNED], "exact_match") if UNTUNED in reports else {}
    outcome = pilot_outcome(run, ids, adapter_exact, untuned_exact, clusters,
                            safety["unauthorized_dispatches"])  # fmt: skip
    outcome["problems"] = problems
    outcome["safety_review_complete"] = safety["complete_coverage"]

    result: dict[str, Any] = {
        "experiment": "EXP-010",
        "analysis_version": ANALYSIS_VERSION,
        "gates_version": GATES_VERSION,
        "harness_version": HARNESS_VERSION,
        "score_version": SCORE_VERSION,
        "expected_training_harness": HARNESS,
        "approved_commit": x.approved_commit,
        "input_sha256": dict(sorted(x.input_hashes.items())),
        "corpus_status": "axbench/0.1-ai-reviewed: owner-authorized AI review, 0 human decisions; "
        "exploratory",
        "denominator": VALIDATION_N,
        "lineage_clusters": len(set(clusters.values())),
        "safety": {"adapter": safety} | {
            name: {"unauthorized_dispatches": r["after_guard"]["unauthorized_dispatch_count"]}
            for name, r in reports.items() if name != SYSTEM
        },
        "pilot_outcome": outcome,
    }  # fmt: skip
    if outcome["outcome"] == "invalid_run":
        result["deployment_gates"] = {"computed": False, "reason": "invalid run: no result"}
        result["descriptive"] = {"computed": False, "reason": "invalid run: no result"}
        result["systems"] = {k: v for k, v in reports.items() if k != SYSTEM}
        return _finish(result)

    adapter = reports[SYSTEM]
    lat = latency(x.adapter_records)
    summary = d14_summary(adapter, sum(untuned_exact.values()), safety["unauthorized_dispatches"],
                          lat, x.eval_meas)  # fmt: skip
    result["deployment_gates"] = deployment_gates(summary) | {"summary": summary}
    rules_exact = by_id(reports[RULES], "exact_match")
    _, vs_rules = better([rules_exact[i] for i in ids], [adapter_exact[i] for i in ids],
                         [clusters[i] for i in ids])  # fmt: skip
    vs_rules["rule3_label_descriptive_only"] = vs_rules.pop("rule3")
    comps = {"adapter vs rules: exact_match": vs_rules}
    for field_ in ("json_valid", "schema_valid", "semantic_valid"):
        comps[f"adapter vs untuned: {field_}"] = paired_by_id(
            by_id(reports[UNTUNED], field_), by_id(adapter, field_), clusters, ids
        )
    slice_pairs: dict[str, Any] = {}
    for key in ("language", "stratum"):
        for value in sorted({str(e[key]) for e in val}):
            sub = [i for i in ids if str(examples[i][key]) == value]
            slice_pairs[f"{key}={value}"] = {
                "n": len(sub),
                "adapter_exact": sum(adapter_exact[i] for i in sub),
                "untuned_exact": sum(untuned_exact[i] for i in sub),
                "rules_exact": sum(rules_exact[i] for i in sub),
            }
    result["descriptive"] = {
        "note": "descriptive only (D15); no population inference; McNemar is an IID illustration",
        "paired_comparisons": comps,
        "slices_exact_match": slice_pairs,
        "adapter_latency_ms": lat,
        "adapter_generation": generation(x.adapter_records),
        "evaluation_resources": evaluation_resources(x.eval_meas),
        "training": training_report(x.train_meas, train_rows or []),
    }
    result["systems"] = reports
    return _finish(result)


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    found: list[str] = []
    clean = strict_json(result, found=found)
    clean["non_finite_values_reported_as_null"] = found
    json.dumps(clean, allow_nan=False)  # strict JSON or an error, never NaN/Infinity tokens
    return clean
