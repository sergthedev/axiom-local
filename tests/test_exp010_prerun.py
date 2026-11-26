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
    PilotRun,
    deployment_gates,
    pilot_outcome,
)
from axbench.trainwatch import (
    IM_END,
    NEWLINE,
    PlanError,
    TrainingExample,
    UpdateLedger,
    UpdateThroughputWatch,
    check_lengths,
    completion_example,
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


def _clusters(n: int = 160, groups: int = 24) -> list[str]:
    return [f"L{i % groups}" for i in range(n)]


VALID = PilotRun(training_completed=True, evaluation_completed=True, integrity_ok=True)


def test_pilot_invalid_run_has_no_result():
    out = pilot_outcome(PilotRun(False, True, True), [True] * 160, [False] * 160, _clusters(), 0)
    assert out["outcome"] == "invalid_run" and "primary" not in out


def test_pilot_safety_stop_overrides_improvement():
    out = pilot_outcome(VALID, [True] * 160, [False] * 160, _clusters(), 1)
    assert out["outcome"] == "safety_stop"


def test_pilot_improvement_demonstrated_and_not():
    untuned = [i < 12 for i in range(160)]
    tuned = [i < 60 for i in range(160)]
    out = pilot_outcome(VALID, tuned, untuned, _clusters(), 0)
    assert out["outcome"] == "improvement_demonstrated"
    assert out["primary"]["rule3"]["better"]
    small = [i < 20 for i in range(160)]  # +8 cases = exactly 5 pp: not "more than 5 pp"
    out = pilot_outcome(VALID, small, untuned, _clusters(), 0)
    assert out["outcome"] == "improvement_not_demonstrated"
    assert not out["primary"]["rule3"]["margin_more_than_5pp"]
    with pytest.raises(ValueError):
        pilot_outcome(VALID, tuned[:-1], untuned, _clusters(), 0)


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
        ("peak_rss_bytes", None, "G12"),
    ],
)
def test_gates_fail_just_past_each_boundary(field, value, gate):
    keep_others = {"G2": {"untuned_exact_count": 127}, "G4": {"exact_en": (54, 64)}}
    r = deployment_gates(_summary(**{field: value, **keep_others.get(gate, {})}))
    passed = _passed(r)
    assert passed[gate] is False
    assert all(v for k, v in passed.items() if k != gate)
    assert not r["all_assessable_passed"]


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


def test_trained_ids_never_include_the_test_split():
    rows = (EXP / "prerun" / "token_lengths_per_example.jsonl").read_text(encoding="utf-8")
    splits = {json.loads(x)["split"] for x in rows.splitlines()}
    assert splits == {"train", "validation"}
