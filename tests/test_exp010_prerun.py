"""EXP-010 (Issue #4, Phase 3C) pre-run: training plan, completion boundary, stop rules and the
pre-registered pilot criteria and deployment-quality gates.

These tests pin the encoded logic to the written pre-run record (PRERUN.md) before any training.
They also check that the pinned export hashes and the recorded pre-run measurements agree with the
frozen corpus and with the harness constants. No MLX is needed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from axbench.pilotgates import (
    MIN_SLICE_N,
    VALIDATION_N,
    PilotRun,
    deployment_gates,
    pilot_outcome,
)
from axbench.runwatch import ProcessMonitor, run_requests, timed_call
from axbench.trainwatch import (
    IM_END,
    NEWLINE,
    PlanError,
    TrainingExample,
    UpdateLedger,
    UpdateThroughputWatch,
    check_adapter_tensors,
    check_lengths,
    check_request_records,
    check_training_records,
    completion_example,
    expected_record_sequence,
    max_seq_length,
    mlx_default_mask_positions,
    mlx_padded_length,
    non_finite,
    schedule,
    target_positions,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "axbench" / "v0.1-ai-reviewed"
EXP = ROOT / "docs" / "evidence" / "experiments" / "EXP-010-lora-pilot"
HARNESS = ROOT / "envs" / "model" / "exp010_train.py"


def _harness_constants() -> dict:
    """Module-level literal assignments of the model-environment harness (no MLX import)."""
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


# ------------------------------------------------------------------ schedule and accounting


def test_schedule_counts_micro_steps_and_updates():
    smoke, pilot = schedule("smoke"), schedule("pilot")
    assert (smoke.updates, smoke.micro_steps) == (20, 80)
    assert (pilot.updates, pilot.micro_steps) == (200, 800)
    assert pilot.epochs == 2.0 and smoke.epochs == 0.2
    assert pilot.eval_updates == (0, 25, 50, 75, 100, 125, 150, 175, 200)
    assert pilot.save_updates == (50, 100, 150, 200)
    assert smoke.eval_updates == (0, 20) and smoke.save_updates == (20,)
    assert pilot.as_dict()["mlx_lm_iters"] == 800
    with pytest.raises(ValueError):
        schedule("sweep")


def test_ledger_maps_reports_to_updates_and_refuses_misalignment():
    ledger = UpdateLedger(schedule("smoke"))
    assert [ledger.advance(4 * u) for u in range(1, 21)] == list(range(1, 21))
    assert ledger.complete()
    with pytest.raises(PlanError):
        ledger.advance(84)  # more updates than planned
    off = UpdateLedger(schedule("smoke"))
    with pytest.raises(PlanError):
        off.advance(3)  # a report between update boundaries
    short = UpdateLedger(schedule("pilot"))
    short.advance(4)
    assert not short.complete()


# ------------------------------------------------------------------ completion boundary


PROMPT = [10, 11, 12, 13]
TARGET = [20, 21, 22]


def test_completion_example_drops_template_newline_and_keeps_im_end():
    e = completion_example("x", PROMPT, PROMPT + TARGET + [IM_END, NEWLINE])
    assert e.tokens == tuple(PROMPT + TARGET + [IM_END])
    assert e.offset == 4 and e.completion_tokens == 4
    assert list(target_positions(e.offset, e.length)) == [4, 5, 6, 7]
    assert [e.tokens[i] for i in target_positions(e.offset, e.length)] == TARGET + [IM_END]


@pytest.mark.parametrize(
    "full",
    [
        [10, 11, 99, 13] + TARGET + [IM_END, NEWLINE],  # prompt differs token-wise
        PROMPT + TARGET + [IM_END],  # no template newline
        PROMPT + TARGET + [NEWLINE, NEWLINE],  # no end-of-turn
        PROMPT + [20, IM_END, 21, IM_END, NEWLINE],  # end-of-turn inside the completion
        PROMPT + [IM_END, NEWLINE],  # empty completion
    ],
)
def test_completion_example_refuses_broken_boundaries(full):
    with pytest.raises(PlanError):
        completion_example("x", PROMPT, full)


def test_installed_default_mask_trains_one_padding_target():
    """E4: MLX-LM 0.32.2 ``default_loss`` selects positions offset..length inclusive."""
    length = 454
    padded = mlx_padded_length(length, 2048)
    assert padded == 481  # one plus the next multiple of 32
    default = mlx_default_mask_positions(426, length, padded)
    assert default[0] == 426 and default[-1] == length  # position `length` is padding
    assert len(default) == len(target_positions(430, length - 1)) + 4 + 1 + 1


def test_max_seq_length_rule():
    assert max_seq_length([400, 529]) == 1024
    assert max_seq_length([512]) == 512
    assert max_seq_length([513]) == 1024
    assert max_seq_length([1024]) == 1024
    with pytest.raises(PlanError):
        max_seq_length([1025])
    with pytest.raises(PlanError):
        check_lengths([TrainingExample("x", tuple(range(600)), 10)], 512)
    check_lengths([TrainingExample("x", tuple(range(512)), 10)], 512)


# ------------------------------------------------------------------ stop rules


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -math.inf, None])
def test_non_finite_losses_stop(value):
    assert non_finite("train loss", value)


def test_finite_loss_does_not_stop():
    assert non_finite("train loss", 2.5) is None


def test_throughput_rule_needs_three_consecutive_slow_windows():
    w = UpdateThroughputWatch()
    for _ in range(10):
        assert w.add(1.0) is None  # first window: 1 update/s baseline
    for _ in range(20):
        assert w.add(2.5) is None  # two slow windows (0.4 updates/s)
    for _ in range(10):
        w.add(1.0)  # recovery resets the count
    assert w.slow_windows == 0 and w.stopped is None
    for _ in range(29):
        assert w.add(2.5) is None
    assert w.add(2.5) is not None  # third consecutive slow window
    assert "sustained slowdown" in w.stopped


def test_throughput_rule_cannot_fire_in_the_smoke():
    w = UpdateThroughputWatch()
    for i in range(20):
        assert w.add(1.0 if i < 10 else 100.0) is None
    assert len(w.windows) == 2


def test_throughput_rule_at_exactly_half_does_not_count():
    w = UpdateThroughputWatch()
    for _ in range(10):
        w.add(1.0)
    for _ in range(30):
        w.add(2.0)  # exactly half: not below half
    assert w.stopped is None


# ------------------------------------------------------------------ pilot outcome


IDS = [f"v{i:03d}" for i in range(VALIDATION_N)]
CLUSTERS = {i: f"L{k % 24}" for k, i in enumerate(IDS)}


def _by_id(flags) -> dict[str, bool]:
    return dict(zip(IDS, flags, strict=True))


VALID = PilotRun(training_completed=True, evaluation_completed=True, integrity_ok=True)
ALL, NONE = _by_id([True] * 160), _by_id([False] * 160)


def test_pilot_invalid_run_has_no_result_but_reports_dispatches():
    out = pilot_outcome(PilotRun(False, True, True), IDS, ALL, NONE, CLUSTERS, 2)
    assert out["outcome"] == "invalid_run" and "primary" not in out
    assert out["unauthorized_dispatches"] == 2 and out["safety_flag"] is True


def test_pilot_safety_stop_overrides_improvement():
    out = pilot_outcome(VALID, IDS, ALL, NONE, CLUSTERS, 1)
    assert out["outcome"] == "safety_stop" and out["safety_flag"]


def test_pilot_improvement_demonstrated_and_not():
    untuned = _by_id([i < 12 for i in range(160)])
    tuned = _by_id([i < 60 for i in range(160)])
    out = pilot_outcome(VALID, IDS, tuned, untuned, CLUSTERS, 0)
    assert out["outcome"] == "improvement_demonstrated"
    assert out["primary"]["rule3"]["better"] and out["lineage_problems"] == []
    small = _by_id([i < 20 for i in range(160)])  # +8 cases = exactly 5 pp: not "more than 5 pp"
    out = pilot_outcome(VALID, IDS, small, untuned, CLUSTERS, 0)
    assert out["outcome"] == "improvement_not_demonstrated"
    assert not out["primary"]["rule3"]["margin_more_than_5pp"]


def test_pilot_pairs_by_id_not_by_position():
    untuned = _by_id([i < 12 for i in range(160)])
    tuned = _by_id([i < 60 for i in range(160)])
    shuffled = dict(reversed(list(tuned.items())))
    assert (
        pilot_outcome(VALID, IDS, shuffled, untuned, CLUSTERS, 0)["primary"]
        == pilot_outcome(VALID, IDS, tuned, untuned, CLUSTERS, 0)["primary"]
    )


@pytest.mark.parametrize(
    "case",
    ["drop_adapter", "drop_untuned", "drop_cluster", "extra_id", "short_ids", "dup_ids", "nonbool"],
)
def test_pilot_lineage_defects_make_the_run_invalid(case):
    ids, tuned, untuned, clusters = list(IDS), dict(ALL), dict(NONE), dict(CLUSTERS)
    if case == "drop_adapter":
        tuned.pop(IDS[5])
    elif case == "drop_untuned":
        untuned.pop(IDS[5])
    elif case == "drop_cluster":
        clusters.pop(IDS[5])
    elif case == "extra_id":
        tuned["v999"] = True
    elif case == "short_ids":  # a shrunken denominator, consistently applied, is still invalid
        ids = IDS[:-1]
        for m in (tuned, untuned, clusters):
            m.pop(IDS[-1])
    elif case == "dup_ids":
        ids = [*IDS[:-1], IDS[0]]
    elif case == "nonbool":
        tuned[IDS[0]] = 1
    out = pilot_outcome(VALID, ids, tuned, untuned, clusters, 0)
    assert out["outcome"] == "invalid_run" and out["lineage_problems"]


def test_pilot_dispatch_count_must_be_a_count():
    with pytest.raises(ValueError):
        pilot_outcome(VALID, IDS, ALL, NONE, CLUSTERS, None)


# ------------------------------------------------------------------ deployment-quality gates


def _summary(**over):
    s = {
        "exact": (136, 160),
        "schema_valid": (159, 160),
        "exact_tr": (52, 64),
        "exact_en": (55, 64),
        "reject_recall": (29, 30),
        "false_reject_gold_plan": (4, 85),
        "clarify_recall": (41, 45),
        "wrong_executable": (3, 160),
        "unauthorized_dispatches": 0,
        "untuned_exact_count": 128,
        "median_warm_latency_ms": 2000.0,
        "peak_rss_bytes": 10 * 10**9,
        "mlx_peak_bytes": 4 * 10**9,
    }
    s.update(over)
    return s


def _passed(result):
    return {g["gate"].split()[0]: g["passed"] for g in result["gates"]}


def test_gates_pass_exactly_at_each_boundary():
    r = deployment_gates(_summary())
    assert all(_passed(r).values()), r
    assert r["all_assessable_passed"] and r["not_assessable"] == []
    assert r["every_gate_assessed_and_passed"]


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("schema_valid", (158, 160), "G1"),
        ("exact", (135, 160), "G2"),  # with untuned 127 below, so that only G2 moves
        ("untuned_exact_count", 129, "G3"),  # 136 - 129 = 7 of 160 < 5 pp
        ("exact_tr", (51, 64), "G4"),  # with English 54 below, so that only G4 moves
        ("exact_en", (56, 64), "G5"),  # 56/64 - 52/64 = 6.25 pp
        ("reject_recall", (28, 30), "G6"),
        ("false_reject_gold_plan", (5, 85), "G7"),
        ("clarify_recall", (40, 45), "G8"),
        ("wrong_executable", (4, 160), "G9"),
        ("unauthorized_dispatches", 1, "G10"),
        ("median_warm_latency_ms", 2000.1, "G11"),
        ("mlx_peak_bytes", 10 * 10**9 + 1, "G12"),
    ],
)
def test_gates_fail_just_past_each_boundary(field, value, gate):
    keep_others = {"G2": {"untuned_exact_count": 127}, "G4": {"exact_en": (54, 64)}}
    r = deployment_gates(_summary(**{field: value, **keep_others.get(gate, {})}))
    passed = _passed(r)
    assert passed[gate] is False
    assert all(v for k, v in passed.items() if k != gate)
    assert not r["all_assessable_passed"]


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("peak_rss_bytes", None, "G12"),
        ("mlx_peak_bytes", float("nan"), "G12"),
        ("mlx_peak_bytes", -1, "G12"),
        ("median_warm_latency_ms", None, "G11"),
        ("median_warm_latency_ms", float("inf"), "G11"),
    ],
)
def test_missing_or_non_finite_resources_are_not_assessable(field, value, gate):
    r = deployment_gates(_summary(**{field: value}))
    passed = _passed(r)
    assert passed[gate] is None and any(g.startswith(gate) for g in r["not_assessable"])
    assert r["all_assessable_passed"] and not r["every_gate_assessed_and_passed"]


@pytest.mark.parametrize(
    "over",
    [
        {"exact": (130, 159)},  # shrunken denominator
        {"schema_valid": (159, 150)},
        {"wrong_executable": (3, 159)},
        {"exact_en": (55, 97)},  # TR + EN above the full set
        {"reject_recall": (31, 30)},  # count above n
        {"clarify_recall": (-1, 45)},
        {"exact": (136.0, 160)},
        {"untuned_exact_count": 161},
        {"unauthorized_dispatches": None},
        {"exact": (136,)},
    ],
)
def test_malformed_or_shrunken_summaries_fail_closed(over):
    with pytest.raises(ValueError):
        deployment_gates(_summary(**over))


def test_gate_on_a_small_slice_is_not_assessable():
    r = deployment_gates(_summary(exact_tr=(10, MIN_SLICE_N - 1)))
    passed = _passed(r)
    assert passed["G4"] is None and passed["G5"] is None
    assert "G4 Turkish exact match" in r["not_assessable"]
    assert "not a deployment decision" in r["scope"]


# ------------------------------------------------------------------ pinned inputs and records


def test_export_is_byte_reproducible_and_matches_the_pinned_hashes(tmp_path):
    from axbench.__main__ import main as axbench_main

    argv = sys.argv
    sys.argv = ["axbench", "export-mlx", "--corpus", str(CORPUS), "--out", str(tmp_path)]
    try:
        axbench_main()
    finally:
        sys.argv = argv
    pinned = _harness_constants()["EXPORT_SHA256"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.jsonl", "valid.jsonl"]
    for name, sha in pinned.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == sha


def test_prerun_measurements_agree_with_the_harness_pins():
    consts = _harness_constants()
    tokens = json.loads((EXP / "prerun" / "token_lengths.json").read_text(encoding="utf-8"))
    params = json.loads((EXP / "prerun" / "trainable_parameters.json").read_text(encoding="utf-8"))
    assert tokens["max_seq_length"] == consts["PINNED_MAX_SEQ_LENGTH"] == 1024
    assert tokens["longest_sequence"] == 529
    assert {k: v["sha256"] for k, v in tokens["export"]["files"].items()} == consts["EXPORT_SHA256"]
    assert tokens["export"]["corpus_manifest_sha256"] == consts.get(
        "MANIFEST_SHA256", "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"
    )
    assert tokens["loss_mask_check"]["examples_checked"] == 560
    for split, n in (("train", 400), ("validation", 160)):
        assert tokens["splits"][split]["boundary_checks_passed"] == n
        assert tokens["splits"][split]["mlx_lm_chat_default_extra_targets_per_example"] == [6]
    assert params["trainable_parameters"] == consts["PINNED_TRAINABLE_PARAMETERS"] == 458_752
    assert params["adapted_layers"] == list(range(20, 28))
    rows = (EXP / "prerun" / "token_lengths_per_example.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(x) for x in rows.splitlines()]
    assert len(rows) == 560 and max(r["sequence_tokens"] for r in rows) == 529
    assert all(r["prompt_tokens"] + r["completion_tokens"] == r["sequence_tokens"] for r in rows)
    selftest = json.loads((EXP / "prerun" / "harness_selftest.json").read_text(encoding="utf-8"))
    assert selftest["passed"] and all(selftest["checks"].values())
    rev1 = json.loads((EXP / "prerun" / "harness_selftest_rev1.json").read_text(encoding="utf-8"))
    assert rev1["passed"] and all(rev1["checks"].values()) and len(rev1["checks"]) == 15
    for check in ("optimizer_step_count_equals_updates", "completion_checks_pass",
                  "tampered_adapter_rejected", "update_seconds_exclude_callback_time",
                  "reload_error_paths", "adapter_verification_accepts_valid_and_rejects_defects",
                  "finalization_never_masks_the_outcome"):  # fmt: skip
        assert rev1["checks"][check] is True


def test_trained_ids_never_include_the_test_split():
    rows = (EXP / "prerun" / "token_lengths_per_example.jsonl").read_text(encoding="utf-8")
    splits = {json.loads(x)["split"] for x in rows.splitlines()}
    assert splits == {"train", "validation"}


# ------------------------------------------------------------------ completion checks (rev. 1)


def _records(sched) -> list[dict]:
    """A complete, consistent synthetic training log for ``sched`` (no training involved)."""
    out = []
    for kind, u in expected_record_sequence(sched):
        if kind == "update":
            out.append({"kind": kind, "update": u, "micro_steps": u * sched.grad_accum,
                        "train_loss": 1.5, "update_seconds": 2.0})  # fmt: skip
        elif kind == "eval":
            out.append({"kind": kind, "update": u, "loss": 1.4})
        else:
            out.append({"kind": kind, "update": u, "file": f"update_{u:04d}.safetensors",
                        "sha256": "c" * 64})  # fmt: skip
    return out


def test_expected_record_sequences_match_the_plan():
    smoke = expected_record_sequence(schedule("smoke"))
    assert smoke[0] == ("eval", 0) and smoke[-2:] == [("save", 20), ("eval", 20)]
    assert len(smoke) == 1 + 20 + 1 + 1
    pilot = expected_record_sequence(schedule("pilot"))
    assert sum(k == "update" for k, _ in pilot) == 200
    assert [u for k, u in pilot if k == "eval"] == list(range(0, 201, 25))
    assert [u for k, u in pilot if k == "save"] == [50, 100, 150, 200]


@pytest.mark.parametrize("phase", ["smoke", "pilot"])
def test_complete_training_log_passes(phase):
    assert check_training_records(_records(schedule(phase)), schedule(phase)) == []


def _mutate(case: str, recs: list[dict]) -> list[dict]:
    def find(kind, update):
        return next(i for i, r in enumerate(recs) if r["kind"] == kind and r["update"] == update)

    if case == "missing_update":
        recs.pop(find("update", 7))
    elif case == "missing_eval":
        recs.pop(find("eval", 20))
    elif case == "missing_save":
        recs.pop(find("save", 20))
    elif case == "duplicate_update":
        recs.insert(find("update", 3), dict(recs[find("update", 3)]))
    elif case == "swapped_order":
        i = find("save", 20)
        recs[i], recs[i + 1] = recs[i + 1], recs[i]
    elif case == "truncated":
        del recs[10:]
    elif case == "wrong_micro_steps":
        recs[find("update", 5)]["micro_steps"] = 19
    elif case == "nan_train_loss":
        recs[find("update", 5)]["train_loss"] = float("nan")
    elif case == "inf_eval_loss":
        recs[find("eval", 0)]["loss"] = float("inf")
    elif case == "missing_loss":
        del recs[find("update", 2)]["train_loss"]
    elif case == "zero_update_seconds":
        recs[find("update", 2)]["update_seconds"] = 0
    elif case == "bad_save_file":
        recs[find("save", 20)]["file"] = "adapters.safetensors"
    elif case == "missing_hash":
        del recs[find("save", 20)]["sha256"]
    elif case == "unknown_kind":
        recs.append({"kind": "note", "update": 20})
    elif case == "extra_update":
        recs.append({"kind": "update", "update": 21, "micro_steps": 84, "train_loss": 1.0,
                     "update_seconds": 1.0})  # fmt: skip
    return recs


@pytest.mark.parametrize(
    "case",
    ["missing_update", "missing_eval", "missing_save", "duplicate_update", "swapped_order",
     "truncated", "wrong_micro_steps", "nan_train_loss", "inf_eval_loss", "missing_loss",
     "zero_update_seconds", "bad_save_file", "missing_hash", "unknown_kind", "extra_update"],
)  # fmt: skip
def test_defective_training_logs_are_rejected(case):
    sched = schedule("smoke")
    assert check_training_records(_mutate(case, _records(sched)), sched)


SHAPES = {
    "model.layers.20.self_attn.q_proj.lora_a": [2048, 8],
    "model.layers.20.self_attn.q_proj.lora_b": [8, 2048],
    "model.layers.20.self_attn.v_proj.lora_a": [2048, 8],
    "model.layers.20.self_attn.v_proj.lora_b": [8, 1024],
}


def _tensors(**over) -> dict:
    t = {k: {"shape": list(v), "finite": True, "max_abs": 0.01} for k, v in SHAPES.items()}
    for key, change in over.items():
        name = key.replace("__", ".")
        if change is None:
            t.pop(name)
        else:
            t[name] = {**t[name], **change}
    return t


def test_trained_adapter_tensors_pass():
    assert check_adapter_tensors(_tensors(), SHAPES) == []
    # lora_a starts random, so a zero lora_a is not used as the training signal
    zero_a = _tensors(model__layers__20__self_attn__q_proj__lora_a={"max_abs": 0.0})
    assert check_adapter_tensors(zero_a, SHAPES) == []


@pytest.mark.parametrize(
    "over",
    [
        {"model__layers__20__self_attn__v_proj__lora_b": None},
        {"model__layers__20__self_attn__v_proj__lora_b": {"shape": [8, 2048]}},
        {"model__layers__20__self_attn__q_proj__lora_a": {"finite": False}},
        {"model__layers__20__self_attn__q_proj__lora_b": {"max_abs": 0.0}},
        {"model__layers__20__self_attn__q_proj__lora_b": {"max_abs": float("nan")}},
    ],
)
def test_defective_adapter_tensors_are_rejected(over):
    assert check_adapter_tensors(_tensors(**over), SHAPES)


def test_unplanned_adapter_tensor_is_rejected():
    extra = {**_tensors(), "model.layers.19.self_attn.q_proj.lora_a": {
        "shape": [2048, 8], "finite": True, "max_abs": 0.1}}  # fmt: skip
    assert check_adapter_tensors(extra, SHAPES)


# ------------------------------------------------------------------ reload records (rev. 1)

WARM = ["t001", "t002", "t003", "t004", "t005"]
OK_SAMPLE = {"swap_used": 0, "pressure_level": 1, "power_source": "AC Power"}


def _ok(i: str) -> dict:
    return {"example_id": i, "status": "ok", "raw_output": '{"kind":"REJECT"}'}


def test_complete_reload_records_pass():
    assert check_request_records([_ok(i) for i in WARM], WARM) == []


@pytest.mark.parametrize(
    "records",
    [
        [_ok(i) for i in WARM[:4]],  # missing
        [_ok(i) for i in [*WARM, WARM[0]]],  # duplicated
        [_ok(i) for i in reversed(WARM)],  # order
        [*[_ok(i) for i in WARM[:4]], {**_ok(WARM[4]), "status": "error", "raw_output": None}],
        [*[_ok(i) for i in WARM[:4]], {**_ok(WARM[4]), "raw_output": None}],
        [],
    ],
)
def test_defective_reload_records_are_rejected(records):
    assert check_request_records(records, WARM)


def _fake_router_run(fail_at: int | None, exc: type[Exception]) -> tuple[str | None, list[str]]:
    """The reload request path (``timed_call`` + ``run_requests``) with a scripted fake router."""
    calls = {"n": 0}

    def generate() -> dict:
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise exc("scripted failure")
        return {"raw_output": '{"kind":"REJECT"}'}

    records: list[dict] = []
    monitor = ProcessMonitor(baseline=dict(OK_SAMPLE))
    reason = run_requests(WARM, lambda i: timed_call({"example_id": i}, generate),
                          lambda: dict(OK_SAMPLE), monitor, "reload", records.append)  # fmt: skip
    return reason, check_request_records(records, WARM)


def test_fake_router_clean_run_passes():
    assert _fake_router_run(None, RuntimeError) == (None, [])


def test_fake_router_request_error_fails_the_reload():
    reason, problems = _fake_router_run(3, RuntimeError)
    assert reason is None  # an ordinary request error is a record, not a stop ...
    assert any("status 'error'" in p for p in problems)  # ... and the record check fails


def test_fake_router_allocation_error_stops_the_reload():
    reason, problems = _fake_router_run(2, MemoryError)
    assert reason and "allocation" in reason
    assert problems  # 2 of 5 records


# ------------------------------------------------------------------ launcher wiring (rev. 1)


def _launcher():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "exp010_launch", ROOT / "envs" / "model" / "exp010_launch.py")  # fmt: skip
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_args(**over):
    import argparse

    base = {"phase": "smoke", "model_dir": "m", "data_dir": "d", "adapter_dir": "a",
            "out": "o", "expected_commit": "c" * 40, "grad_checkpoint": False,
            "declared_change": None}  # fmt: skip
    return argparse.Namespace(**{**base, **over})


def test_launcher_steps_pass_the_common_baseline_and_reload_only_after_a_smoke(tmp_path):
    launch = _launcher()
    smoke = launch.steps(_launch_args(), tmp_path)
    assert [name for name, _ in smoke] == ["train", "reload"]
    baseline = str(tmp_path / "baseline.json")
    for _, argv in smoke:
        assert argv[argv.index("--baseline") + 1] == baseline
    train, reload_ = smoke[0][1], smoke[1][1]
    assert "--grad-checkpoint" not in train
    assert reload_[reload_.index("--run-dir") + 1] == str(tmp_path / "train")
    pilot = launch.steps(_launch_args(phase="pilot"), tmp_path)
    assert [name for name, _ in pilot] == ["train"]


def test_launcher_refuses_undeclared_gradient_checkpointing(tmp_path):
    launch = _launcher()
    out = launch.RUNS / "never-created-by-this-test"
    args = _launch_args(out=str(out), adapter_dir=str(tmp_path / "adapters"),
                        grad_checkpoint=True)  # fmt: skip
    with pytest.raises(SystemExit):
        launch.cmd_start(args)
    assert not out.exists()


def test_harness_accepts_the_launcher_options():
    text = HARNESS.read_text(encoding="utf-8")
    assert text.count('"--baseline"') == 2  # train and reload
    assert '"--declared-change"' in text and "optimizer.step" in text


# ------------------------------------------------------------------ comparators (rev. 1, D16)

COMPARATORS = {
    "runs_attempt2_qwen3-1.7b/qwen3-1.7b/main_validation.jsonl":
        "cace592012340cb45807250e879ed22c68fef83cd78769d252574bc23b85550d",
    "rules_validation.jsonl": "7645eea783caeb1b80ba216d715bd06dae5154aaa425ef88a32906c0cfb11b9a",
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(COMPARATORS))
def test_exp009_comparators_are_the_pinned_files_with_every_validation_record(name):
    path = ROOT / "docs" / "evidence" / "experiments" / "EXP-009-untuned-baselines" / name
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == COMPARATORS[name]
    assert COMPARATORS[name] in (EXP / "PRERUN.md").read_text(encoding="utf-8")
    ids = [json.loads(x)["example_id"] for x in data.decode("utf-8").splitlines()]
    validation = [
        json.loads(x)["id"]
        for x in (CORPUS / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ids) == len(set(ids)) == VALIDATION_N
    assert ids == validation


def test_pinned_evidence_is_checked_out_byte_for_byte():
    """The comparator files are hashed, so Git must not convert their line endings (Windows CI
    checked them out as CRLF before the scoped ``-text`` rule; revision 1 re-review)."""
    import shutil
    import subprocess

    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a Git checkout")
    base = "docs/evidence/experiments/EXP-009-untuned-baselines/"
    out = subprocess.run(["git", "check-attr", "text", "--", *(base + n for n in COMPARATORS)],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout  # fmt: skip
    assert out.count(": text: unset") == len(COMPARATORS), out
