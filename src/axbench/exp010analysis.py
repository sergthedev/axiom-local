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
   stop, the final adapter equal to the update-200 save); the evaluation record must name that
   adapter and its training commit and contain exactly the 5 warm-ups (never scored) and the 160
   validation records, unique and in file order; the corpus must hold only the train and
   validation splits (the sealed final test is never read).
2. **Safety, independently of validity.** Every adapter record with a known validation ID is
   reviewed by the Guard (``axbench.evaluate/0.2``), even when the run is invalid, and the
   unauthorized-dispatch count is reported on its own.
3. **Scoring** with ``axbench.score/0.2`` and ``axbench.evaluate/0.2`` (Guard policy 2): every
   record, errors and empty outputs included, stays in every denominator. The comparators are
   rescored with the same code and must reproduce the accepted EXP-009 per-example results.
4. **Decisions** exactly as pre-registered: :func:`axbench.pilotgates.pilot_outcome` (D13) with
   vectors paired by example ID and the frozen lineage clusters, and
   :func:`axbench.pilotgates.deployment_gates` (D14) as a descriptive reading.
5. **Descriptive reports** (D15): paired comparisons, slices, losses, resources, latency and
   uncertainty. No population inference; McNemar only as an IID illustration.
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

ANALYSIS_VERSION = "exp010.analysis/1"
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
# After the approved commit, only evidence may change (raw records, write-ups, logs).
EVIDENCE_ONLY_PREFIXES = ("docs/evidence/",)
ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors",
                 *(f"update_{u:04d}.safetensors"
                   for u in schedule("pilot").save_updates))  # fmt: skip
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 2026


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_jsonl(data: bytes) -> list[Any]:
    """Every non-empty line must be JSON; a malformed line raises ``ValueError``."""
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _same_json(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------ record checks


def check_records(records: Any, expected_ids: Sequence[str], system: str) -> list[str]:
    """Exactly the expected IDs, unique, in the expected (file) order, one system, known statuses.

    Errors, timeouts and empty outputs are valid records (they are scored as wrong and stay in
    every denominator); a missing, duplicated, unexpected or reordered record is not.
    """
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        return ["records are not a list of objects"]
    problems: list[str] = []
    ids = [r.get("example_id") for r in records]
    counts = Counter(ids)
    expected = set(expected_ids)
    missing = [i for i in expected_ids if i not in counts]
    duplicated = sorted(str(i) for i, c in counts.items() if c > 1)
    unexpected = sorted(str(i) for i in counts if i not in expected)
    if missing:
        problems.append(f"{len(missing)} missing records (first {missing[:3]})")
    if duplicated:
        problems.append(f"duplicated IDs {duplicated[:3]}")
    if unexpected:
        problems.append(f"unexpected IDs {unexpected[:3]}")
    if not (missing or duplicated or unexpected) and ids != list(expected_ids):
        problems.append("records are not in file order")
    if any(r.get("system") != system for r in records):
        problems.append(f"records not labelled {system!r}")
    bad_status = sorted({str(r.get("status")) for r in records} - set(STATUSES))
    if bad_status:
        problems.append(f"unknown statuses {bad_status}")
    if any(r.get("status") == "ok" and not isinstance(r.get("raw_output"), str) for r in records):
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
    records = parse_jsonl(data)
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


def check_sequence(label: str, exit_status: str | None, monitor_stop: bool, done: bool
                   ) -> list[str]:  # fmt: skip
    """The supervised sequence ended with exit 0, its done marker and no monitor stop."""
    problems = []
    if exit_status is None or exit_status.split()[:1] != ["0"]:
        problems.append(f"{label} sequence: exit status {exit_status!r}, not 0")
    if monitor_stop:
        problems.append(f"{label} sequence: the external monitor wrote a stop record")
    if not done:
        problems.append(f"{label} sequence: no done marker")
    return problems


def _not_stopped(meas: Mapping[str, Any], label: str) -> list[str]:
    problems = []
    if meas.get("status") != "completed":
        problems.append(f"{label}: status {meas.get('status')!r}")
    for key in ("stop_reason", "failure", "finalization_errors", "external_monitor_stop"):
        if meas.get(key):
            problems.append(f"{label}: {key} recorded")
    return problems


def _environment(meas: Mapping[str, Any], label: str) -> list[str]:
    env = meas.get("environment") or {}
    problems = []
    if (env.get("git") or {}).get("dirty") is not False:
        problems.append(f"{label}: tree not recorded as clean")
    if env.get("runner_version") != HARNESS:
        problems.append(f"{label}: harness {env.get('runner_version')!r}, expected {HARNESS!r}")
    return problems


def check_training(meas: Mapping[str, Any], log: Sequence[dict[str, Any]], approved: str
                   ) -> list[str]:  # fmt: skip
    """The approved pilot, completed as planned (D4-D8): re-derived from the records."""
    sched = schedule("pilot")
    p = [*_not_stopped(meas, "training"), *_environment(meas, "training")]
    if meas.get("experiment") != "EXP-010" or meas.get("phase") != "pilot":
        p.append("training: not the EXP-010 pilot phase")
    if meas.get("system") != SYSTEM:
        p.append("training: system name differs")
    if (meas.get("environment") or {}).get("git", {}).get("commit") != approved:
        p.append("training: commit differs from the approved commit")
    if not _same_json(meas.get("schedule"), sched.as_dict()):
        p.append("training: schedule differs from the pilot plan")
    if meas.get("optimizer_steps") != sched.updates:
        p.append(f"training: optimizer steps {meas.get('optimizer_steps')!r}, planned 200")
    if meas.get("updates_recorded") != sched.updates:
        p.append(f"training: {meas.get('updates_recorded')!r} updates recorded, planned 200")
    cfg = meas.get("configuration") or {}
    planned = {"base": "qwen3-1.7b", "lora_parameters": LORA_PARAMETERS,
               "num_layers": NUM_LAYERS, "learning_rate": LEARNING_RATE, "seed": SEED}  # fmt: skip
    if any(not _same_json(cfg.get(k), v) for k, v in planned.items()):
        p.append("training: configuration differs from the plan")
    if (cfg.get("grad_checkpoint"), cfg.get("declared_change")) not in ((False, None),
                                                                        (True, "C1")):  # fmt: skip
        p.append("training: gradient checkpointing without declared change C1")
    if (meas.get("lora") or {}).get("trainable_parameters") != PINNED_TRAINABLE_PARAMETERS:
        p.append("training: trainable parameter count differs from the pin")
    if (meas.get("model") or {}).get("revision") != BASE_REVISION:
        p.append("training: base revision differs from the pin")
    data = meas.get("data") or {}
    if (
        data.get("corpus_manifest_sha256") != MANIFEST_SHA256
        or {k: (v or {}).get("sha256") for k, v in (data.get("files") or {}).items()}
        != EXPORT_SHA256
    ):
        p.append("training: data hashes differ from the pinned export")
    eval_set = meas.get("eval_set") or {}
    if eval_set.get("name") != "validation" or eval_set.get("examples") != VALIDATION_N:
        p.append("training: the loss curve was not evaluated on the 160 validation examples")
    if not (meas.get("completion_checks") or {}).get("passed"):
        p.append("training: completion checks did not pass")
    if meas.get("final_adapter_equals_last_update_save") is not True:
        p.append("training: final adapter is not the update-200 save")
    files = meas.get("adapter_files") or {}
    if sorted(files) != sorted(ADAPTER_FILES):
        p.append(f"training: adapter files {sorted(files)} differ from the plan")
    p += [f"training log: {x}" for x in check_training_records(log, sched)]
    saves = {r.get("update"): r.get("sha256") for r in log if r.get("kind") == "save"}
    final = (files.get("adapters.safetensors") or {}).get("sha256")
    if final is None or final != saves.get(sched.updates):
        p.append("training: final adapter hash differs from the update-200 save record")
    for u, sha in saves.items():
        if (files.get(f"update_{u:04d}.safetensors") or {}).get("sha256") != sha:
            p.append(f"training: update-{u} save hash differs from its file record")
    updates = [r for r in log if r.get("kind") == "update"]
    if not updates or updates[-1].get("micro_steps") != sched.micro_steps:
        p.append("training: the last update is not at micro-step 800")
    return p


def check_evaluation(
    meas: Mapping[str, Any],
    train_meas: Mapping[str, Any],
    approved: str,
    warmup_records: Any,
    warmup_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> list[str]:
    """The fresh-process evaluation of exactly the trained final adapter (D12)."""
    p = [*_not_stopped(meas, "evaluation"), *_environment(meas, "evaluation")]
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
    if not _same_json(meas.get("model"), train_meas.get("model")):
        p.append("evaluation: base checkpoint differs from the one trained")
    ad = meas.get("adapter") or {}
    files = train_meas.get("adapter_files") or {}
    final = files.get("adapters.safetensors") or {}
    if (
        ad.get("phase") != "pilot"
        or ad.get("file") != "adapters.safetensors"
        or ad.get("sha256") != final.get("sha256")
        or ad.get("bytes") != final.get("bytes")
    ):
        p.append("evaluation: adapter is not the trained final pilot adapter")  # fmt: skip
    if ad.get("config_sha256") != (files.get("adapter_config.json") or {}).get("sha256"):
        p.append("evaluation: adapter configuration differs from the trained one")
    if ad.get("training_commit") != approved:
        p.append("evaluation: adapter training commit is not the approved commit")
    if (meas.get("stop_rule_baseline") or {}).get("source") != "sequence baseline.json":
        p.append("evaluation: not run under the supervised sequence baseline")
    if meas.get("warmup_ids") != list(warmup_ids) or len(warmup_ids) != WARMUP_COUNT:
        p.append("evaluation: warm-up IDs are not the 5 pinned train warm-ups")
    if not isinstance(warmup_records, list) or [
        r.get("example_id") for r in warmup_records if isinstance(r, dict)
    ] != list(warmup_ids):
        p.append("evaluation: warm-up records are not exactly the 5 warm-ups in order")
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


def _finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def latency(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Warm latency over the validation requests. The median is a G11 measurement only when every
    one of the 160 requests has a finite latency (as in EXP-009); otherwise it is ``None``."""
    vals = [float(r["elapsed_ms"]) for r in records if _finite(r.get("elapsed_ms"))]
    complete = len(vals) == len(records) == VALIDATION_N
    return {"ms": _quartiles(vals), "requests_with_latency": len(vals),
            "median_for_gate": statistics.median(vals) if complete else None}  # fmt: skip


def generation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gen = [float(r["generation_tokens"]) for r in records if _finite(r.get("generation_tokens"))]
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
    secs = [r["update_seconds"] for r in updates if _finite(r.get("update_seconds"))]
    return {
        "train_loss_per_update": [[r["update"], r.get("train_loss")] for r in updates],
        "validation_loss": [[r["update"], r.get("loss"), r.get("seconds")] for r in evals],
        "loss_note": "validation loss is token-weighted over completion tokens; it is not an "
        "endpoint and no checkpoint is selected by it",
        "trainable_parameters": (meas.get("lora") or {}).get("trainable_parameters"),
        "wall_seconds": meas.get("wall_seconds"),
        "train_loop_seconds": meas.get("train_loop_seconds"),
        "train_loop_note": "train_loop_seconds starts after the update-0 evaluation; it includes "
        "the later evaluations, the saves and the callbacks",
        "sum_update_seconds": sum(secs) if secs else None,
        "update_seconds": _quartiles(secs),
        "throughput": meas.get("throughput"),
        "trained_completion_tokens": updates[-1].get("trained_tokens") if updates else None,
        "mlx_peak_bytes": meas.get("mlx_peak_bytes"),
        "process_peak_rss_bytes": meas.get("rss_peak_process_bytes"),
        "resource_monitor": meas.get("resource_monitor"),
        "adapter_files": meas.get("adapter_files"),
    }


def evaluation_resources(meas: Mapping[str, Any]) -> dict[str, Any]:
    return {k: meas.get(k) for k in (
        "cold_load_seconds", "load_mlx_peak_bytes", "mlx_peak_bytes_load_and_warmup",
        "mlx_peak_bytes_load_warmup_validation", "rss_peak_after_load_bytes",
        "rss_peak_after_validation_bytes", "rss_peak_process_bytes", "validation_slowdown",
        "resource_monitor")} | {"note": "RSS, MLX peak, swap and pressure are separate "
                                "measurements and are never summed"}  # fmt: skip


def safety_scan(by_id: Mapping[str, dict[str, Any]], records: Any) -> dict[str, Any]:
    """Guard review of every adapter record whose ID is a validation ID, duplicates included,
    whatever the validity of the run: the unauthorized-dispatch count never depends on it."""
    recs = [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
    known = [r for r in recs if r.get("example_id") in by_id]
    review = guard_layer([by_id[r["example_id"]] for r in known], known)
    return {
        "records_reviewed": len(known),
        "records_not_reviewable": len(recs) - len(known),
        "unauthorized_dispatches": review["unauthorized_dispatch_count"],
        "guard_status": review["guard_status"],
        "confirmations_given": 0,
    }


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
    peak since loading (load, warm-up and validation; lead decision 9)."""
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
        "peak_rss_bytes": eval_meas.get("rss_peak_process_bytes"),
        "mlx_peak_bytes": eval_meas.get("mlx_peak_bytes_load_warmup_validation"),
    }


# ------------------------------------------------------------------ the analysis


@dataclass
class Inputs:
    """Everything the analysis reads, already loaded; nothing else is consulted."""

    approved_commit: str
    corpus_splits: Sequence[str]
    corpus_manifest_sha256: str
    validation: Sequence[dict[str, Any]]
    warmup_ids: Sequence[str]  # the first five train IDs (runwatch.warmup_ids)
    train_meas: dict[str, Any]
    train_log: list[dict[str, Any]]
    train_sequence: tuple[str | None, bool, bool]  # exit status, monitor stop, done marker
    eval_meas: dict[str, Any]
    eval_sequence: tuple[str | None, bool, bool]
    warmup_records: Any
    adapter_records: Any
    comparator_bytes: Mapping[str, bytes | None]
    exp009_results_bytes: bytes | None
    rules_rerun: Sequence[dict[str, Any]] | None  # axiom.rules/0.1 rerun on the frozen file
    changed_since_approved: Mapping[str, Sequence[str] | None] = field(default_factory=dict)
    preserved: Mapping[str, tuple[str | None, str | None]] = field(default_factory=dict)
    input_hashes: Mapping[str, str | None] = field(default_factory=dict)


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


def analyze(x: Inputs) -> dict[str, Any]:
    val = list(x.validation)
    ids = [e["id"] for e in val]
    examples = {e["id"]: e for e in val}
    clusters = {e["id"]: e["lineage_id"] for e in val}
    problems: dict[str, list[str]] = {
        "corpus": check_corpus(x.corpus_splits, x.corpus_manifest_sha256, val),
        "frozen_code": [q for label, changed in sorted(x.changed_since_approved.items())
                        for q in check_frozen_code(changed, label)],
        "raw_evidence_preserved": check_preserved(x.preserved),
        "training": [*check_sequence("training", *x.train_sequence),
                     *check_training(x.train_meas, x.train_log, x.approved_commit)],
        "evaluation": [*check_sequence("evaluation", *x.eval_sequence),
                       *check_evaluation(x.eval_meas, x.train_meas, x.approved_commit,
                                         x.warmup_records, x.warmup_ids, ids)],
        "adapter_records": check_records(x.adapter_records, ids, SYSTEM),
        "comparators": [],
    }  # fmt: skip
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

    # Safety first and independently of every other check.
    safety = safety_scan(examples, x.adapter_records)

    reports: dict[str, Any] = {}
    for name, recs in comparators.items():
        reports[name] = evaluate(val, recs, system=name, split="validation")
    problems["comparators"] += _exp009_reproduced(x.exp009_results_bytes, reports)
    if not problems["adapter_records"]:
        reports[SYSTEM] = evaluate(val, x.adapter_records, system=SYSTEM, split="validation")
        if (
            reports[SYSTEM]["after_guard"]["unauthorized_dispatch_count"]
            != safety["unauthorized_dispatches"]
        ):
            problems["adapter_records"].append("Guard reviews disagree on dispatches")
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
        return result

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
        "training": training_report(x.train_meas, x.train_log),
    }
    result["systems"] = reports
    return result
