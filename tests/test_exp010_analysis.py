"""EXP-010 frozen analysis (PRERUN D18): integrity, safety, scoring, pairing and decisions.

Only synthetic fixtures and the historical, pinned EXP-009 comparator records are used: the
"adapter" records are the untuned EXP-009 outputs relabelled, optionally with the gold label
rendered as the output for chosen IDs (an oracle edit, not a model prediction). No smoke output,
no new validation prediction and no final-test data is read.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

import axbench.evaluate as evaluate_mod
from axbench import exp010analysis as ax
from axbench.corpus import load_corpus
from axbench.evaluate import guard_layer, rule_router, run_router
from axbench.pilotgates import VALIDATION_N
from axbench.runwatch import warmup_ids
from axbench.score import score_one
from axbench.trainwatch import expected_record_sequence, schedule

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "axbench" / "v0.1-ai-reviewed"
APPROVED = "a" * 40
MANIFEST, SPLITS = load_corpus(CORPUS)
VAL = SPLITS["validation"]
IDS = [e["id"] for e in VAL]
WARM = warmup_ids(SPLITS["train"])
COMP = {name: (ROOT / path).read_bytes() for name, (path, _) in ax.COMPARATORS.items()}
RESULTS = (ROOT / ax.EXP009_RESULTS[0]).read_bytes()
UNTUNED = ax.parse_jsonl(COMP[ax.UNTUNED])
RULES = ax.parse_jsonl(COMP[ax.RULES])


def _sha(u: int) -> str:
    return f"{u:064x}"


def _training() -> tuple[dict, list[dict]]:
    sched = schedule("pilot")
    log = []
    for kind, u in expected_record_sequence(sched):
        if kind == "eval":
            log.append({"kind": "eval", "update": u, "loss": 2.0 - u / 200, "seconds": 30.0,
                        "set": "validation"})  # fmt: skip
        elif kind == "update":
            log.append({"kind": "update", "update": u, "micro_steps": 4 * u, "train_loss": 1.0,
                        "update_seconds": 1.0, "trained_tokens": 100 * u})  # fmt: skip
        else:
            log.append({"kind": "save", "update": u, "file": f"update_{u:04d}.safetensors",
                        "sha256": _sha(u), "bytes": 1838449})  # fmt: skip
    files = {f"update_{u:04d}.safetensors": {"bytes": 1838449, "sha256": _sha(u)}
             for u in sched.save_updates}  # fmt: skip
    files["adapters.safetensors"] = {"bytes": 1838449, "sha256": _sha(200)}
    files["adapter_config.json"] = {"bytes": 2650, "sha256": "c" * 64}
    meas = {
        "experiment": "EXP-010", "phase": "pilot", "system": ax.SYSTEM, "status": "completed",
        "environment": {"git": {"commit": APPROVED, "dirty": False},
                        "runner_version": ax.HARNESS},
        "schedule": sched.as_dict(), "optimizer_steps": 200, "updates_recorded": 200,
        "configuration": {"base": "qwen3-1.7b", "lora_parameters": ax.LORA_PARAMETERS,
                          "num_layers": ax.NUM_LAYERS, "learning_rate": ax.LEARNING_RATE,
                          "seed": ax.SEED, "grad_checkpoint": False, "declared_change": None},
        "lora": {"trainable_parameters": ax.PINNED_TRAINABLE_PARAMETERS},
        "model": {"revision": ax.BASE_REVISION, "files": {"config.json": {"sha256": "d" * 64}}},
        "data": {"corpus_manifest_sha256": ax.MANIFEST_SHA256,
                 "files": {k: {"sha256": v} for k, v in ax.EXPORT_SHA256.items()}},
        "eval_set": {"name": "validation", "examples": 160},
        "completion_checks": {"passed": True}, "final_adapter_equals_last_update_save": True,
        "adapter_files": files, "wall_seconds": 400.0, "train_loop_seconds": 380.0,
    }  # fmt: skip
    return meas, log


def _evaluation(train_meas: dict) -> dict:
    files = train_meas["adapter_files"]
    return {
        "experiment": "EXP-010", "phase": "main", "system": ax.SYSTEM, "status": "completed",
        "environment": {"git": {"commit": APPROVED, "dirty": False},
                        "runner_version": ax.HARNESS},
        "validation_completed": True, "counts": {"validation": 160, "warmup": 5},
        "corpus_manifest_sha256": ax.MANIFEST_SHA256, "prompt_version": ax.PROMPT_VERSION,
        "decoding": copy.deepcopy(ax.DECODING), "model": copy.deepcopy(train_meas["model"]),
        "adapter": {"phase": "pilot", "file": "adapters.safetensors",
                    **files["adapters.safetensors"],
                    "config_sha256": files["adapter_config.json"]["sha256"],
                    "training_commit": APPROVED},
        "stop_rule_baseline": {"source": "sequence baseline.json"}, "warmup_ids": list(WARM),
        "rss_peak_process_bytes": 3_000_000_000,
        "mlx_peak_bytes_load_warmup_validation": 4_000_000_000,
    }  # fmt: skip


def _adapter_records(oracle: set[str] = frozenset()) -> list[dict]:
    """Historical untuned outputs relabelled; the gold label rendered for the ``oracle`` IDs."""
    gold = {e["id"]: json.dumps(e["gold"]) for e in VAL}
    out = []
    for r in UNTUNED:
        r = {**r, "system": ax.SYSTEM}
        if r["example_id"] in oracle:
            r.update(status="ok", raw_output=gold[r["example_id"]])
        out.append(r)
    return out


def _inputs(**over) -> ax.Inputs:
    train, log = _training()
    base = dict(
        approved_commit=APPROVED, corpus_splits=sorted(SPLITS),
        corpus_manifest_sha256=ax.MANIFEST_SHA256, validation=VAL, warmup_ids=WARM,
        train_meas=train, train_log=log, train_sequence=("0 2026-11-27T00:00:00Z", False, True),
        eval_meas=_evaluation(train), eval_sequence=("0 2026-11-27T01:00:00Z", False, True),
        warmup_records=[{"example_id": i, "status": "ok"} for i in WARM],
        adapter_records=_adapter_records(), comparator_bytes=dict(COMP),
        exp009_results_bytes=RESULTS, rules_rerun=RULES,
        changed_since_approved={"training commit": [], "evaluation commit": [],
                                "analysis HEAD": ["docs/evidence/x.md"]},
        preserved={"runs/pilot_eval/eval/main_validation.jsonl": ("e" * 64, "e" * 64)},
    )  # fmt: skip
    return ax.Inputs(**{**base, **over})


def _all_problems(result: dict) -> list[str]:
    return [p for ps in result["pilot_outcome"]["problems"].values() for p in ps]


def _improvable(k: int) -> set[str]:
    """``k`` IDs where the untuned output is wrong, spread over different lineages."""
    seen, out = set(), []
    for e, r in zip(VAL, UNTUNED, strict=True):
        if not score_one(e, r).exact_match and e["lineage_id"] not in seen:
            seen.add(e["lineage_id"])
            out.append(e["id"])
    assert len(out) >= k
    return set(out[:k])


# ------------------------------------------------------------------ the valid path


def test_valid_synthetic_run_keeps_the_full_denominator_and_reproduces_exp009():
    r = ax.analyze(_inputs())
    assert _all_problems(r) == []
    o = r["pilot_outcome"]
    assert o["outcome"] == "improvement_not_demonstrated"
    assert o["validity"] == {"training_completed": True, "evaluation_completed": True,
                             "integrity_ok": True}  # fmt: skip
    assert o["primary"]["n"] == VALIDATION_N and o["primary"]["delta_pp"] == 0.0
    assert o["primary"]["paired_cluster_bootstrap_delta"]["n_clusters"] == 24
    assert r["lineage_clusters"] == 24 and r["denominator"] == 160
    assert r["systems"][ax.UNTUNED]["raw_candidate"]["primary_semantic_exact_match"]["count"] == 12
    assert r["systems"][ax.RULES]["raw_candidate"]["primary_semantic_exact_match"]["count"] == 73
    assert r["safety"]["adapter"]["unauthorized_dispatches"] == 0
    assert r["safety"]["adapter"]["records_reviewed"] == 160
    gates = r["deployment_gates"]
    assert gates["summary"]["exact"] == (12, 160) and gates["summary"]["untuned_exact_count"] == 12
    assert gates["summary"]["exact_tr"][1] == 64 and gates["summary"]["exact_en"][1] == 64
    assert r["analysis_version"] == ax.ANALYSIS_VERSION and r["gates_version"].endswith("/2")
    d = r["descriptive"]
    assert d["slices_exact_match"]["language=mixed"]["n"] == 32
    assert d["paired_comparisons"]["adapter vs untuned: schema_valid"]["delta_pp"] == 0.0
    vs_rules = d["paired_comparisons"]["adapter vs rules: exact_match"]
    assert "rule3_label_descriptive_only" in vs_rules and "rule3" not in vs_rules
    assert len(d["training"]["validation_loss"]) == 9  # updates 0, 25, ..., 200
    assert d["training"]["sum_update_seconds"] == 200.0


@pytest.mark.parametrize(("k", "outcome", "delta"), [
    (8, "improvement_not_demonstrated", 5.0),  # exactly 5 pp: not "more than 5"
    (9, "improvement_demonstrated", 5.625),
])  # fmt: skip
def test_primary_endpoint_margin_boundary_through_the_real_scorer(k, outcome, delta):
    r = ax.analyze(_inputs(adapter_records=_adapter_records(_improvable(k))))
    o = r["pilot_outcome"]
    assert o["outcome"] == outcome and o["primary"]["delta_pp"] == delta
    assert o["primary"]["rule3"]["margin_more_than_5pp"] is (k == 9)
    g3 = next(g for g in r["deployment_gates"]["gates"] if g["gate"].startswith("G3"))
    assert g3["count"] == k and g3["passed"] is True  # >= 5 pp


def test_failed_and_empty_requests_stay_in_every_denominator():
    recs = _adapter_records(_improvable(9))
    failed = IDS[:6]
    for r in recs:
        if r["example_id"] in failed[:3]:
            r.update(status="error", raw_output=None, elapsed_ms=None, error="RuntimeError")
        elif r["example_id"] in failed[3:]:
            r.update(status="ok", raw_output="")
    res = ax.analyze(_inputs(adapter_records=recs))
    assert _all_problems(res) == []
    raw = res["systems"][ax.SYSTEM]["raw_candidate"]
    assert raw["n"] == 160 and raw["status"]["error"] == 3
    per = {p["id"]: p for p in res["systems"][ax.SYSTEM]["per_example"]}
    assert not any(per[i]["exact_match"] or per[i]["schema_valid"] for i in failed)
    assert res["descriptive"]["adapter_latency_ms"]["requests_with_latency"] == 157
    g11 = next(g for g in res["deployment_gates"]["gates"] if g["gate"].startswith("G11"))
    assert g11["passed"] is None  # a latency is missing: not assessable, never a pass


# ------------------------------------------------------------------ adapter records


def _drop(recs, i):
    return [r for r in recs if r["example_id"] != IDS[i]]


@pytest.mark.parametrize("mutate", [
    lambda r: _drop(r, 5),                                          # missing
    lambda r: [*r, dict(r[7])],                                     # duplicate
    lambda r: [*_drop(r, 0), {**r[0], "example_id": WARM[0]}],      # a warm-up scored
    lambda r: [{**x, "example_id": "test-0001"} if x["example_id"] == IDS[3] else x for x in r],
    lambda r: [r[1], r[0], *r[2:]],                                 # file order
    lambda r: [{**x, "system": "other"} if x["example_id"] == IDS[2] else x for x in r],
    lambda r: [{**x, "status": "skipped"} if x["example_id"] == IDS[2] else x for x in r],
    lambda r: [{**x, "raw_output": None} if x["example_id"] == IDS[2] else x for x in r],
    lambda r: {"records": r},
])  # fmt: skip
def test_defective_adapter_record_sets_are_an_invalid_run(mutate):
    r = ax.analyze(_inputs(adapter_records=mutate(_adapter_records(_improvable(9)))))
    o = r["pilot_outcome"]
    assert o["outcome"] == "invalid_run" and o["problems"]["adapter_records"]
    assert "primary" not in o and r["deployment_gates"]["computed"] is False
    assert r["safety"]["adapter"]["unauthorized_dispatches"] == 0  # reported anyway


# ------------------------------------------------------------------ comparators and corpus


def _tampered(name: str) -> dict:
    data = bytearray(COMP[name])
    data[data.index(b'"status": "ok"')] ^= 1
    return {**COMP, name: bytes(data)}


@pytest.mark.parametrize("comparators", [
    _tampered(ax.UNTUNED), _tampered(ax.RULES), {**COMP, ax.UNTUNED: None},
    {**COMP, ax.RULES: COMP[ax.UNTUNED]},
])  # fmt: skip
def test_comparator_tampering_or_replacement_is_an_invalid_run(comparators):
    r = ax.analyze(_inputs(comparator_bytes=comparators))
    assert r["pilot_outcome"]["outcome"] == "invalid_run"
    assert r["pilot_outcome"]["problems"]["comparators"]


def test_exp009_results_must_be_the_pinned_file_and_be_reproduced():
    r = ax.analyze(_inputs(exp009_results_bytes=RESULTS + b"\n"))
    assert "EXP-009 results.json differs from the pinned hash" in _all_problems(r)
    assert ax._exp009_reproduced(RESULTS, {ax.UNTUNED: {"per_example": [
        {"id": i, "exact_match": True, "schema_valid": True} for i in IDS]}}) == [
        f"rescored {ax.UNTUNED} differs from the accepted EXP-009 results"]  # fmt: skip


def test_rules_rerun_is_identical_to_the_pinned_outputs_and_a_difference_is_flagged():
    rerun = run_router(rule_router(), VAL, system="rules")
    assert ax.analyze(_inputs(rules_rerun=rerun))["pilot_outcome"]["problems"]["comparators"] == []
    changed = [dict(x) for x in rerun]
    changed[4]["raw_output"] = "{}"
    r = ax.analyze(_inputs(rules_rerun=changed))
    assert r["pilot_outcome"]["outcome"] == "invalid_run"
    r = ax.analyze(_inputs(rules_rerun=None))
    assert "no rules rerun was supplied" in _all_problems(r)


def test_sealed_test_split_or_another_corpus_is_refused():
    r = ax.analyze(_inputs(corpus_splits=["test", "train", "validation"]))
    assert r["pilot_outcome"]["outcome"] == "invalid_run"
    r = ax.analyze(_inputs(corpus_manifest_sha256="0" * 64))
    assert r["pilot_outcome"]["outcome"] == "invalid_run"


# ------------------------------------------------------------------ training provenance


def _train_variant(edit):
    meas, log = _training()
    edit(meas, log)
    return ax.check_training(meas, log, APPROVED)


def _set(path, value):
    def edit(meas, log):
        d = meas
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value

    return edit


TRAINING_DEFECTS = {
    "wrong commit": _set(("environment", "git", "commit"), "b" * 40),
    "dirty tree": _set(("environment", "git", "dirty"), True),
    "old harness": _set(("environment", "runner_version"), "exp010.harness/2"),
    "smoke phase": _set(("phase",), "smoke"),
    "stopped": _set(("status",), "stopped"),
    "stop reason": _set(("stop_reason",), "pressure"),
    "monitor stop": _set(("external_monitor_stop",), {"reason": "swap"}),
    "finalization": _set(("finalization_errors",), ["after: OSError"]),
    "199 optimizer steps": _set(("optimizer_steps",), 199),
    "199 updates": _set(("updates_recorded",), 199),
    "learning rate": _set(("configuration", "learning_rate"), 2e-5),
    "undeclared checkpointing": _set(("configuration", "grad_checkpoint"), True),
    "trainable count": _set(("lora", "trainable_parameters"), 1),
    "base revision": _set(("model", "revision"), "0" * 40),
    "export hash": _set(("data", "files", "train.jsonl", "sha256"), "0" * 64),
    "eval set": _set(("eval_set", "name"), "train_first_16"),
    "completion checks": _set(("completion_checks", "passed"), False),
    "final not last save": _set(("final_adapter_equals_last_update_save",), False),
    "final adapter hash": _set(("adapter_files", "adapters.safetensors", "sha256"), "9" * 64),
    "save file hash": _set(("adapter_files", "update_0100.safetensors", "sha256"), "9" * 64),
    "extra adapter file": _set(("adapter_files", "update_0175.safetensors"), {"sha256": "1"}),
    "missing loss record": lambda m, log: log.pop(next(i for i, r in enumerate(log)
                                                       if r["kind"] == "eval" and r["update"])),
    "missing save record": lambda m, log: log.pop(next(i for i, r in enumerate(log)
                                                       if r["kind"] == "save")),
    "non-finite loss": lambda m, log: next(r for r in log if r["kind"] == "update").update(
        train_loss=math.nan),
    "truncated at 199": lambda m, log: log.__delitem__(slice(-4, None)),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(TRAINING_DEFECTS))
def test_training_provenance_defects_are_detected(name):
    assert _train_variant(TRAINING_DEFECTS[name]), name


def test_declared_change_c1_is_accepted_and_valid_training_has_no_problem():
    assert _train_variant(lambda m, log: None) == []
    both = _set(("configuration", "grad_checkpoint"), True)
    assert _train_variant(lambda m, log: (both(m, log), m["configuration"].update(
        declared_change="C1"))) == []  # fmt: skip


def test_training_defect_makes_the_run_invalid_end_to_end():
    meas, log = _training()
    meas["optimizer_steps"] = 199
    r = ax.analyze(_inputs(train_meas=meas, train_log=log))
    assert r["pilot_outcome"]["outcome"] == "invalid_run"
    assert r["pilot_outcome"]["validity"]["training_completed"] is False


def test_missing_measurement_files_are_an_invalid_run_not_a_crash():
    r = ax.analyze(_inputs(train_meas={}, train_log=[], eval_meas={}, warmup_records=None,
                           adapter_records=None))  # fmt: skip
    o = r["pilot_outcome"]
    assert o["outcome"] == "invalid_run"
    assert o["validity"] == {"training_completed": False, "evaluation_completed": False,
                             "integrity_ok": False}  # fmt: skip
    assert r["safety"]["adapter"]["records_reviewed"] == 0


@pytest.mark.parametrize(("seq", "ok"), [
    (("0 2026-11-27T00:00:00Z", False, True), True),
    (("6 2026-11-27T00:00:00Z", False, True), False),
    (("0 2026-11-27T00:00:00Z", True, True), False),
    (("0 2026-11-27T00:00:00Z", False, False), False),
    ((None, False, True), False),
])  # fmt: skip
def test_supervised_sequence_must_end_cleanly(seq, ok):
    assert (ax.check_sequence("training", *seq) == []) is ok
    r = ax.analyze(_inputs(train_sequence=seq))
    assert (r["pilot_outcome"]["outcome"] != "invalid_run") is ok


# ------------------------------------------------------------------ evaluation provenance


EVAL_DEFECTS = {
    "wrong adapter": _set(("adapter", "sha256"), "9" * 64),
    "update checkpoint": _set(("adapter", "file"), "update_0150.safetensors"),
    "adapter config": _set(("adapter", "config_sha256"), "9" * 64),
    "smoke adapter": _set(("adapter", "phase"), "smoke"),
    "training commit": _set(("adapter", "training_commit"), "b" * 40),
    "stopped": _set(("status",), "stopped"),
    "monitor stop": _set(("external_monitor_stop",), {"reason": "pressure"}),
    "incomplete": _set(("validation_completed",), False),
    "counts": _set(("counts",), {"validation": 159, "warmup": 5}),
    "decoding": _set(("decoding", "max_new_tokens"), 512),
    "prompt": _set(("prompt_version",), "axiom.prompt/2"),
    "base": _set(("model", "revision"), "0" * 40),
    "unsupervised": _set(("stop_rule_baseline", "source"), "this process, before loading"),
    "warm-up ids": _set(("warmup_ids",), list(reversed(WARM))),
    "dirty": _set(("environment", "git", "dirty"), True),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(EVAL_DEFECTS))
def test_evaluation_provenance_defects_are_an_invalid_run(name):
    train, _ = _training()
    meas = _evaluation(train)
    EVAL_DEFECTS[name](meas, None)
    r = ax.analyze(_inputs(eval_meas=meas))
    assert r["pilot_outcome"]["outcome"] == "invalid_run", name
    assert r["pilot_outcome"]["problems"]["evaluation"]


@pytest.mark.parametrize("warm", [
    None, [], [{"example_id": i} for i in WARM[:4]], [{"example_id": i} for i in reversed(WARM)],
])  # fmt: skip
def test_warmup_records_must_be_exactly_the_five_warmups(warm):
    r = ax.analyze(_inputs(warmup_records=warm))
    assert r["pilot_outcome"]["validity"]["evaluation_completed"] is False


# ------------------------------------------------------------------ frozen code and evidence


def test_frozen_code_and_preserved_evidence_checks():
    assert ax.check_frozen_code(["docs/evidence/experiments/EXP-010-lora-pilot/x.json"], "a") == []
    assert ax.check_frozen_code(["src/axbench/score.py"], "a")
    assert ax.check_frozen_code(None, "a")
    assert ax.check_preserved({"p": ("1", "1")}) == []
    assert ax.check_preserved({"p": ("1", None)}) == ["p: not committed before scoring"]
    assert ax.check_preserved({"p": ("1", "2")}) == ["p: differs from the committed version"]
    assert ax.check_preserved({"p": (None, "2")}) == ["p: missing"]
    for over in ({"changed_since_approved": {"analysis HEAD": ["benchmarks/x.py"]}},
                 {"changed_since_approved": {"evaluation commit": None}},
                 {"changed_since_approved": {}},
                 {"preserved": {"x.jsonl": ("1", None)}}, {"preserved": {}}):  # fmt: skip
        r = ax.analyze(_inputs(**over))
        assert r["pilot_outcome"]["outcome"] == "invalid_run", over
        assert r["pilot_outcome"]["validity"]["integrity_ok"] is False


# ------------------------------------------------------------------ safety


def _dispatching_guard(examples, predictions):
    """The real Guard review plus one simulated unauthorized dispatch for adapter records."""
    out = guard_layer(examples, predictions)
    if any(p.get("system") == ax.SYSTEM for p in predictions):
        out["unauthorized_dispatch_count"] += 1
    return out


def test_unauthorized_dispatch_is_a_safety_stop_whatever_the_accuracy(monkeypatch):
    monkeypatch.setattr(evaluate_mod, "guard_layer", _dispatching_guard)
    monkeypatch.setattr(ax, "guard_layer", _dispatching_guard)
    r = ax.analyze(_inputs(adapter_records=_adapter_records(_improvable(20))))
    assert r["pilot_outcome"]["outcome"] == "safety_stop"
    assert r["pilot_outcome"]["safety_flag"] is True
    assert r["safety"]["adapter"]["unauthorized_dispatches"] == 1
    assert r["safety"][ax.UNTUNED]["unauthorized_dispatches"] == 0
    g10 = next(g for g in r["deployment_gates"]["gates"] if g["gate"].startswith("G10"))
    assert g10["passed"] is False


def test_safety_count_is_reported_for_an_invalid_run(monkeypatch):
    monkeypatch.setattr(ax, "guard_layer", _dispatching_guard)
    recs = _adapter_records()
    r = ax.analyze(_inputs(adapter_records=[*recs, dict(recs[0])]))  # duplicate -> invalid
    assert r["pilot_outcome"]["outcome"] == "invalid_run"
    assert r["pilot_outcome"]["unauthorized_dispatches"] == 1
    assert r["pilot_outcome"]["safety_flag"] is True
    assert r["safety"]["adapter"]["records_reviewed"] == 161


# ------------------------------------------------------------------ resources and pairing


@pytest.mark.parametrize(("key", "value"), [
    ("rss_peak_process_bytes", math.nan), ("rss_peak_process_bytes", None),
    ("mlx_peak_bytes_load_warmup_validation", math.inf),
    ("mlx_peak_bytes_load_warmup_validation", -1), ("rss_peak_process_bytes", True),
])  # fmt: skip
def test_missing_or_non_finite_resources_are_not_assessable(key, value):
    train, _ = _training()
    meas = _evaluation(train)
    meas[key] = value
    r = ax.analyze(_inputs(eval_meas=meas))
    g12 = next(g for g in r["deployment_gates"]["gates"] if g["gate"].startswith("G12"))
    assert g12["passed"] is None and "G12 process peak RSS and MLX peak" in r[
        "deployment_gates"]["not_assessable"]  # fmt: skip
    assert r["deployment_gates"]["every_gate_assessed_and_passed"] is False


def test_resource_limit_boundary():
    train, _ = _training()
    for value, passed in ((10 * 10**9, True), (10 * 10**9 + 1, False)):
        meas = _evaluation(train)
        meas["mlx_peak_bytes_load_warmup_validation"] = value
        g12 = next(g for g in ax.analyze(_inputs(eval_meas=meas))["deployment_gates"]["gates"]
                   if g["gate"].startswith("G12"))  # fmt: skip
        assert g12["passed"] is passed


def test_pairs_are_formed_by_example_id_not_position():
    a = {i: k % 3 == 0 for k, i in enumerate(IDS)}
    b = {i: k % 2 == 0 for k, i in enumerate(IDS)}
    clusters = {e["id"]: e["lineage_id"] for e in VAL}
    shuffled_b = dict(reversed(list(b.items())))
    assert ax.paired_by_id(a, b, clusters, IDS) == ax.paired_by_id(a, shuffled_b, clusters, IDS)
    by_position = ax.paired(list(a.values()), list(shuffled_b.values()),
                            [clusters[i] for i in IDS])  # fmt: skip
    assert by_position != ax.paired_by_id(a, b, clusters, IDS)


def test_lineage_and_validation_design_is_frozen():
    assert ax.check_corpus(sorted(SPLITS), ax.MANIFEST_SHA256, VAL) == []
    one_lineage = [{**e, "lineage_id": "L"} for e in VAL]
    assert ax.check_corpus(sorted(SPLITS), ax.MANIFEST_SHA256, one_lineage)
    assert ax.check_corpus(sorted(SPLITS), ax.MANIFEST_SHA256, VAL[:-1])


# ------------------------------------------------------------------ the script's Git helpers


def _script():
    spec = importlib.util.spec_from_file_location(
        "issue4_exp010_analyze", ROOT / "benchmarks" / "issue4_exp010_analyze.py")  # fmt: skip
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_reads_committed_bytes_and_ancestry_from_git():
    s = _script()
    for path, pinned in ax.COMPARATORS.values():
        assert s.committed_sha(ROOT / path) == pinned
    assert s.committed_sha(ROOT / "docs" / "never-committed-by-this-test.json") is None
    head = s.git_state()["commit"]
    assert s.changed_since(head, head) == []
    assert s.changed_since(head, None) is None
    assert s.changed_since("0" * 40, head) is None


def test_prerun_documents_the_frozen_analysis():
    text = (ROOT / "docs" / "evidence" / "experiments" / "EXP-010-lora-pilot" /
            "PRERUN.md").read_text(encoding="utf-8")  # fmt: skip
    for needle in (ax.ANALYSIS_VERSION, ax.HARNESS, ax.EXP009_RESULTS[1],
                   "issue4_exp010_analyze.py", "--phase evaluate"):  # fmt: skip
        assert needle in text, needle
