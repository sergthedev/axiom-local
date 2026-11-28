"""EXP-TR03 plan tooling without weights: config, schedule, budget, monitor, thermal parser.

Only synthetic ids, lengths and event streams, plus the committed TR02 record text, are used. No
model is loaded and no optimizer step is taken.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plan = _load("tr03_plan", ROOT / "training" / "tr03" / "tr03.py")
tr02 = plan.tr02
CFG = plan.load_config()


# ----------------------------------------------------------------------------------- config


def test_config_is_consistent_with_tr02():
    assert plan.config_problems(CFG) == []


def test_config_copies_tr02_pins_byte_for_byte():
    tr02_cfg = json.loads((ROOT / "training/tr02/config.json").read_text("utf-8"))
    assert CFG["model"] == tr02_cfg["model"]
    assert CFG["packages"] == tr02_cfg["packages"]
    for key, value in tr02_cfg["hyperparameters"].items():
        assert CFG["hyperparameters"][key] == value
    assert CFG["inherits"]["tr02_config_sha256"] == plan.sha256_file(plan.TR02_CONFIG_PATH)
    assert len(CFG["model"]["files"]) == 10  # the ten pinned checkpoint files


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda c: c["model"].update(revision="0" * 40), "model_pin_differs_from_tr02"),
        (lambda c: c["packages"].update(mlx="0.0.1"), "packages_differ_from_tr02"),
        (
            lambda c: c["hyperparameters"].update(learning_rate=1e-4),
            "hyperparameter_learning_rate_differs_from_tr02",
        ),
        (
            lambda c: c["hyperparameters"].update(lora_rank=16),
            "hyperparameter_lora_rank_differs_from_tr02",
        ),
        (lambda c: c["hyperparameters"].update(init_adapter="probe"), "init_adapter_must_be_null"),
        (lambda c: c["hyperparameters"].update(partial_batch="pad"), "partial_batch_rule_changed"),
        (
            lambda c: c["hyperparameters"].update(grad_accumulation=8),
            "effective_batch_is_not_1_by_4",
        ),
        (
            lambda c: c["data"].update(per_example_sha256="0" * 64),
            "data_per_example_sha256_differs_from_tr02",
        ),
        (lambda c: c["monitor"].update(mlx_peak_gb=12.0), "monitor_mlx_peak_gb_differs_from_tr02"),
        (lambda c: c.update(status="approved"), "status_must_say_not_approved_and_no_clearance"),
        (
            lambda c: c["data"]["eligibility"].update(schema_sources=["SRC-O01", "SRC-O05"]),
            "eligibility_differs_from_the_data_tooling",
        ),
        (
            lambda c: c["monitor"].update(floor_ratio=0.9),
            "monitor_ratios_must_satisfy_0<floor<slow<1",
        ),
        (
            lambda c: c["arms"]["research"].update(needs_human_gates=False),
            "research_arm_must_need_human_gates",
        ),
    ],
)
def test_config_mutations_are_caught(mutate, problem):
    bad = copy.deepcopy(CFG)
    mutate(bad)
    assert problem in plan.config_problems(bad)


# ---------------------------------------------------------------------------------- schedule


def test_one_epoch_order_equals_the_frozen_tr02_order():
    for n in (5, 400, 5091):
        assert plan.exposure_order(n, 2026, 1)[0] == tr02.planned_order(n, 2026)


def test_second_epoch_is_a_new_permutation_from_the_same_seeding():
    first, second = plan.exposure_order(50, 7, 2)
    assert sorted(first) == sorted(second) == list(range(50))
    assert first != second
    assert plan.exposure_order(50, 7, 2) == [first, second]  # deterministic


def _synthetic(n: int, length: int = 500):
    ids = [f"r{i:03d}" for i in range(n)]
    return ids, {i: length + k for k, i in enumerate(ids)}


def test_schedule_drops_the_partial_batch_and_lists_the_rows():
    ids, lengths = _synthetic(10)
    s = plan.exposure_schedule(
        ids, lengths, seed=1, grad_accumulation=4, epochs=1, max_seq_length=2048, full=True
    )
    assert s["optimizer_updates"] == 2 and s["microsteps"] == 8
    assert s["unexposed_rows_per_epoch"] == [2] and len(s["unexposed_row_ids"][0]) == 2
    exposed = [st["id"] for st in s["steps"]]
    assert len(set(exposed)) == 8 and set(exposed) | set(s["unexposed_row_ids"][0]) == set(ids)
    assert s["each_row_exposed_at_most_once_per_epoch"]
    assert s["full_tokens_exposed"] == sum(lengths[i] for i in exposed)
    exact = plan.exposure_schedule(
        *_synthetic(12), seed=1, grad_accumulation=4, epochs=1, max_seq_length=2048
    )
    assert exact["unexposed_rows_per_epoch"] == [0] and exact["optimizer_updates"] == 3


def test_schedule_with_two_epochs_counts_updates_and_remainders_per_epoch():
    ids, lengths = _synthetic(10)
    s = plan.exposure_schedule(
        ids, lengths, seed=3, grad_accumulation=4, epochs=2, max_seq_length=2048
    )
    assert s["optimizer_updates"] == 4 and s["microsteps"] == 16
    assert s["unexposed_rows_per_epoch"] == [2, 2]


def test_schedule_is_a_function_of_the_row_set_not_of_its_list_order():
    ids, lengths = _synthetic(40)
    a = plan.exposure_schedule(
        ids, lengths, seed=5, grad_accumulation=4, epochs=1, max_seq_length=2048
    )
    assert (
        a["row_set_sha256"]
        == plan.exposure_schedule(
            list(reversed(ids)), lengths, seed=5, grad_accumulation=4, epochs=1, max_seq_length=2048
        )["row_set_sha256"]
    )
    b = plan.exposure_schedule(
        ids, lengths, seed=6, grad_accumulation=4, epochs=1, max_seq_length=2048
    )
    assert a["exposure_order_sha256"] != b["exposure_order_sha256"]


def test_schedule_refuses_truncation_duplicates_and_tiny_sets():
    ids, lengths = _synthetic(8)
    lengths[ids[0]] = 2049
    with pytest.raises(ValueError, match="none is truncated"):
        plan.exposure_schedule(
            ids, lengths, seed=1, grad_accumulation=4, epochs=1, max_seq_length=2048
        )
    ids, lengths = _synthetic(8)
    with pytest.raises(ValueError, match="duplicate"):
        plan.exposure_schedule(
            [*ids, ids[0]], lengths, seed=1, grad_accumulation=4, epochs=1, max_seq_length=2048
        )
    with pytest.raises(ValueError, match="fewer rows"):
        plan.exposure_schedule(
            ids[:3], lengths, seed=1, grad_accumulation=4, epochs=1, max_seq_length=2048
        )
    with pytest.raises(ValueError):
        plan.exposure_schedule(
            ids, lengths, seed=1, grad_accumulation=4, epochs=0, max_seq_length=2048
        )


def test_eval_points():
    assert plan.eval_points(1272) == [318, 636, 954, 1272]
    assert plan.eval_points(7) == [1, 3, 5, 7]
    assert plan.eval_points(4) == [1, 2, 3, 4]
    for too_few in (1, 2, 3):
        with pytest.raises(ValueError):
            plan.eval_points(too_few)


def test_diagnostic_validation_ids_are_fixed_and_spread_over_frames():
    from axbench_tr import tr03

    _, valid, _, _ = tr03.load()
    elig, _ = tr03.split_eligible(valid)
    a = plan.diagnostic_validation_ids(elig, 200, "tr03-diagnostic-validation")
    assert a == plan.diagnostic_validation_ids(elig, 200, "tr03-diagnostic-validation")
    assert len(a) == len(set(a)) == 200
    frames = {r["frame"] for r in elig if r["id"] in set(a)}
    assert len(frames) == len({r["frame"] for r in elig})  # every frame appears


# ------------------------------------------------------------------------------------ budget


def test_budget_arithmetic_by_hand():
    rates = {
        "full_tokens_per_s_slowest_observed": 1000.0,
        "validation_seconds_per_row_slowest_observed": 0.5,
        "model_load_seconds": 2.0,
        "thermal_allowance_fraction": 0.25,
        "wall_cap_factor": 1.25,
        "wall_cap_round_to_seconds": 900,
    }
    b = plan.budget(
        {"full_tokens_exposed": 600_000},
        diagnostic_rows=100,
        full_validation_rows=500,
        rates=rates,
        eval_count=4,
    )
    # training 600 s; validation 0.5 * (5 * 100 + 500) = 500 s; load 2 s
    assert b["seconds"]["training_nominal"] == 600
    assert b["seconds"]["validation_nominal"] == 500
    assert b["seconds"]["total_nominal"] == 1102
    # conservative: (600 + 500) / 0.75 + 2 = 1468.67 s; cap: ceil(1468.67 * 1.25 / 900) * 900
    assert b["seconds"]["total_conservative"] == 1469
    assert b["seconds"]["hard_wall_cap"] == 2700
    assert b["validation_overhead_fraction_of_training_nominal"] == pytest.approx(
        500 / 600, abs=1e-4
    )


def test_committed_training_plan_equals_a_fresh_rebuild_and_has_the_expected_shape():
    fresh = plan.plan_bytes(plan.build_plan())
    assert plan.PLAN_PATH.read_bytes() == fresh
    p = json.loads(fresh)
    assert p["rows"] == {"train_eligible": 5091, "validation_eligible": 1274}
    assert p["schedule"]["optimizer_updates"] == 1272 and p["schedule"]["microsteps"] == 5088
    assert p["schedule"]["unexposed_rows_per_epoch"] == [3]
    assert p["eval_points_after_updates"] == [0, 318, 636, 954, 1272]
    assert p["monitor_windows"]["windows"] == 51
    secs = p["budget"]["seconds"]
    assert secs["total_nominal"] < secs["total_conservative"] < secs["hard_wall_cap"] == 3600
    chosen = [a for a in p["epoch_alternatives_not_selected"] if a["selected"]]
    assert [a["epochs"] for a in chosen] == [1]
    assert "not measured at this scale" in p["budget"]["status"]


def test_plan_cli_check_and_readiness_exit_codes():
    def run(*args):
        return subprocess.run(
            [sys.executable, str(ROOT / "training/tr03/tr03.py"), *args],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

    assert run("verify-config").returncode == 0
    assert run("check-plan").returncode == 0
    ready = run("readiness")
    assert ready.returncode == 1  # every human gate is closed
    assert json.loads(ready.stdout)["research_training_open"] is False


# ------------------------------------------------------------------------- thermal parser

OBSERVED = (
    "No thermal warning level has been recorded;No performance warning level has been recorded;"
    "No CPU power status has been recorded;"
)


def test_thermal_parser_accepts_the_text_recorded_in_the_tr02_run():
    assert plan.parse_therm(OBSERVED) == []
    raw = "\n".join(
        f"Note: {s}" for s in OBSERVED.rstrip(";").split(";")
    )  # the raw pmset line form
    assert plan.parse_therm(raw) == []
    logs = (
        ROOT
        / "docs/evidence/experiments/EXP-TR02-feasibility-smoke/runs/run5-attempt2-at-ed4e75b"
        / "sanitized/run-logs"
    )
    texts = []
    for name in ("os_samples_probe.log", "os_samples_smoke.log", "os_samples_reload.log"):
        texts += re.findall(r'therm="([^"]*)"', (logs / name).read_text("utf-8"))
    assert len(texts) >= 50  # the samples recorded in the committed TR02 evidence
    assert all(plan.parse_therm(t) == [] for t in texts)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n",
        "CPU_Speed_Limit = 80",
        OBSERVED + "CPU_Scheduler_Limit = 100;",
        "Thermal warning level 1 has been recorded;No performance warning level has been recorded;",
        "No thermal warning level has been recorded;",  # performance statement missing
        "garbled",
    ],
)
def test_thermal_parser_fails_closed(text):
    assert plan.parse_therm(text)


# ------------------------------------------------------------------------------------ monitor

LIM = plan.MonitorLimits.from_config(CFG, max_wall_s=3600)
OS_OK = {
    "kind": "os",
    "rss_gb": 3.0,
    "swap_used_gb": 0.01,
    "pressure_level": 1,
    "on_battery": False,
    "therm": OBSERVED,
    "elapsed_s": 10.0,
}


def win(i, rate=1650.0, peak=6.2, loss=0.3, t=None):
    return {
        "kind": "window",
        "index": i,
        "full_tokens_per_s": rate,
        "mlx_peak_gb": peak,
        "train_loss": loss,
        "elapsed_s": float(t if t is not None else 30 * i),
    }


def os_sample(**kw):
    return {**OS_OK, **kw}


def feed_all(events):
    m = plan.LongRunMonitor(LIM)
    for ev in events:
        m.feed(ev)
    return m


def reason(events) -> str:
    with pytest.raises(plan.Stop) as info:
        feed_all(events)
    return info.value.reason


def test_a_clean_run_passes_and_completes():
    events = []
    for i in range(1, 52):
        events.append(win(i, rate=1650 - i, peak=6.0 + 0.001 * i))
        events.append(os_sample(elapsed_s=30.0 * i))
    events.insert(0, {"kind": "val", "name": "before", "loss": 2.5, "elapsed_s": 1.0})
    m = feed_all(events)
    assert m.complete(51, ["before"]) == []
    assert m.complete(50, ["before"]) == ["windows_51_expected_50"]
    assert "validation_events_differ_from_the_plan" in m.complete(51, ["before", "after"])


def test_reference_rate_is_the_median_of_the_first_windows():
    m = plan.LongRunMonitor(LIM)
    assert m.reference_rate is None
    for i, r in enumerate((1600, 1700, 1650, 1620), 1):
        m.feed(win(i, rate=r))
    assert m.reference_rate == pytest.approx((1620 + 1650) / 2)


def test_sustained_slowdown_stops_only_after_the_configured_windows():
    base = [win(i) for i in range(1, 5)]
    slow = [win(5 + k, rate=0.65 * 1650) for k in range(5)]
    feed_all(base + slow)  # five slow windows are tolerated
    assert reason(base + [win(5 + k, rate=0.65 * 1650) for k in range(6)]) == (
        "sustained_throughput_loss"
    )
    recover = base + slow + [win(10)] + [win(11 + k, rate=0.65 * 1650) for k in range(5)]
    feed_all(recover)  # a recovered window resets the run


def test_throughput_floor_and_slowdown_do_not_apply_inside_the_reference_windows():
    feed_all([win(i, rate=100.0) for i in range(1, 5)])  # the reference itself can be anything
    base = [win(i) for i in range(1, 5)]
    assert reason(base + [win(5 + k, rate=0.4 * 1650) for k in range(3)]) == "throughput_floor"


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([win(1, peak=10.5)], "mlx_peak_limit"),
        ([win(1, peak=6.0), win(2, peak=5.9)], "mlx_peak_invalid_or_decreasing"),
        ([win(1, loss=float("nan"))], "nonfinite_training_loss"),
        ([win(1, rate=0.0)], "nonfinite_or_nonpositive_rate"),
        ([win(1, rate=float("inf"))], "nonfinite_or_nonpositive_rate"),
        ([win(1), win(3)], "event_sequence"),
        ([win(1, t=4000.0)], "max_wall_time"),
        (
            [{"kind": "val", "name": "x", "loss": float("nan"), "elapsed_s": 1.0}],
            "nonfinite_validation_loss",
        ),
        ([{"kind": "mystery", "elapsed_s": 1.0}], "unknown_event"),
        ([{"kind": "os"}], "malformed_event"),
        ([os_sample(on_battery=True)], "not_on_ac_power"),
        ([os_sample(on_battery=None)], "not_on_ac_power"),
        (
            [
                os_sample(
                    therm="No thermal warning level has been recorded;Thermal warning level 2;"
                )
            ],
            "thermal_or_performance_warning",
        ),
        ([os_sample(therm="")], "thermal_or_performance_warning"),
        ([os_sample(rss_gb=11.5)], "rss_limit"),
        ([os_sample(pressure_level=4)], "memory_pressure_critical"),
        ([os_sample(swap_used_gb=0.0), os_sample(swap_used_gb=2.5)], "swap_growth"),
        ([os_sample(swap_used_gb=float("nan"))], "unreadable_os_sample"),
        ([os_sample(pressure_level="1")], "unreadable_os_sample"),
    ],
)
def test_each_stop_rule_fires(events, expected):
    assert reason(events) == expected


def test_swap_growth_is_measured_from_the_first_sample():
    feed_all([os_sample(swap_used_gb=1.0), os_sample(swap_used_gb=2.9)])  # +1.9 GB
    assert reason([os_sample(swap_used_gb=1.0), os_sample(swap_used_gb=3.2)]) == "swap_growth"


def test_pressure_warning_is_tolerated_at_load_but_not_as_a_steady_state():
    warm_up = [os_sample(pressure_level=2) for _ in range(30)]
    feed_all(warm_up)  # before the first window: model load, as TR02 saw in 3 of 12 probe samples
    after = [win(1)] + [os_sample(pressure_level=2) for _ in range(11)]
    feed_all(after)
    assert reason([win(1)] + [os_sample(pressure_level=2) for _ in range(12)]) == (
        "sustained_memory_pressure_warning"
    )
    interrupted = [win(1)] + [os_sample(pressure_level=2) for _ in range(11)]
    interrupted += [os_sample(pressure_level=1)] + [os_sample(pressure_level=2) for _ in range(11)]
    feed_all(interrupted)


def test_the_wall_cap_comes_from_the_budget_and_is_enforced():
    p = json.loads(plan.PLAN_PATH.read_text("utf-8"))
    cap = p["budget"]["seconds"]["hard_wall_cap"]
    lim = plan.MonitorLimits.from_config(CFG, max_wall_s=cap)
    m = plan.LongRunMonitor(lim)
    m.feed(win(1, t=cap))
    with pytest.raises(plan.Stop, match="max_wall_time"):
        m.feed(win(2, t=cap + 1))


def test_window_count():
    assert plan.window_count(1272, 25) == 51
    assert plan.window_count(1250, 25) == 50
    assert plan.window_count(1, 25) == 1


def test_plan_module_has_no_model_path():
    source = (ROOT / "training/tr03/tr03.py").read_text("utf-8")
    for forbidden in ("import mlx", "from mlx", "load_adapters", "safetensors", "optimizers."):
        assert forbidden not in source
