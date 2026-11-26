"""EXP-009 (Issue #4, Phase 3B): the frozen base-selection rule, stop rules and analysis pipeline.

The selection rule and stop rules were fixed in the protocol (revision 3) and the pre-run record
before any validation prediction; these tests pin the encoded logic to the written rule.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from axbench.corpus import load_corpus
from axbench.runwatch import (
    SWAP_GROWTH_LIMIT_BYTES,
    StopWatch,
    parse_free_percentage,
    parse_swapusage,
    repeat_subset,
    warmup_ids,
)
from axbench.selection import LARGE, SMALL, ModelRun, better, select_base

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "axbench" / "v0.1-ai-reviewed"
N = 160
CLUSTERS = [f"L{i % 24:02d}" for i in range(N)]


def _vec(k: int) -> list[bool]:
    return [True] * k + [False] * (N - k)


def _run(name, exact=0, schema=0, *, completed=True, rss=3 * 10**9, lat=900.0, disp=0):
    return ModelRun(name, completed, rss, lat, _vec(exact), _vec(schema), disp)


# ------------------------------------------------------------------ selection rule


def test_safety_stop_on_any_dispatch_including_rules():
    d = select_base(_run(SMALL, 100, 160), _run(LARGE, 150, 160), CLUSTERS, rules_dispatches=1)
    assert d.selected is None and d.basis == "safety_stop"
    d = select_base(_run(SMALL, 100), _run(LARGE, 150, disp=1), CLUSTERS, 0)
    assert d.selected is None and d.basis == "safety_stop"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed": False},
        {"rss": 10 * 10**9 + 1},
        {"lat": 2000.1},
        {"rss": None},
        {"lat": None},
    ],
)
def test_eligibility_boundaries(kwargs):
    d = select_base(_run(SMALL, 100, 160, **kwargs), _run(LARGE, 10, 10), CLUSTERS, 0)
    assert d.selected == LARGE and d.basis == "single_eligible" and not d.by_evidence


def test_eligibility_limits_are_inclusive():
    d = select_base(_run(SMALL, 0, 0, rss=10 * 10**9, lat=2000.0), _run(LARGE, 0, 0), CLUSTERS, 0)
    assert d.eligibility[SMALL] == {
        "completed_without_stop_rule": True,
        "peak_rss_at_most_10GB": True,
        "median_warm_latency_at_most_2s": True,
    }


def test_no_eligible_model():
    d = select_base(_run(SMALL, lat=5000), _run(LARGE, lat=5000), CLUSTERS, 0)
    assert d.selected is None and d.basis == "no_eligible"


def test_primary_larger_better_with_margin_and_interval():
    d = select_base(_run(SMALL, 40, 160), _run(LARGE, 80, 160), CLUSTERS, 0)
    assert d.selected == LARGE and d.basis == "primary" and d.by_evidence


def test_primary_margin_of_exactly_5pp_is_not_better():
    # 8 of 160 = 5.0 pp exactly: not "more than 5 percentage points".
    d = select_base(_run(SMALL, 40), _run(LARGE, 48), CLUSTERS, 0)
    assert d.selected == SMALL and d.basis == "primary" and not d.by_evidence


def test_primary_smaller_wins_ties_and_when_smaller_is_higher():
    assert select_base(_run(SMALL, 60), _run(LARGE, 60), CLUSTERS, 0).selected == SMALL
    assert select_base(_run(SMALL, 90), _run(LARGE, 40), CLUSTERS, 0).selected == SMALL


def test_floor_is_inclusive_at_32_of_160():
    d = select_base(_run(SMALL, 32, 0), _run(LARGE, 0, 160), CLUSTERS, 0)
    assert d.basis == "primary"
    d = select_base(_run(SMALL, 31, 0), _run(LARGE, 31, 160), CLUSTERS, 0)
    assert d.basis == "secondary" and d.selected == LARGE


def test_fallback_when_no_floor_reached():
    d = select_base(_run(SMALL, 31, 31), _run(LARGE, 31, 31), CLUSTERS, 0)
    assert d.selected == SMALL and d.basis == "fallback" and not d.by_evidence


def test_better_needs_interval_excluding_zero():
    # +10 pp concentrated in one lineage of a two-lineage set: margin met, interval spans zero.
    clusters = ["A"] * 16 + ["B"] * 144
    a = [False] * 160
    b = [True] * 16 + [False] * 144
    is_better, comp = better(a, b, clusters)
    assert comp["rule3"]["margin_more_than_5pp"]
    assert not comp["rule3"]["interval_excludes_zero"] and not is_better


def test_rule_requires_full_denominator_and_fixed_names():
    with pytest.raises(ValueError):
        select_base(_run(SMALL), _run(LARGE), CLUSTERS[:100], 0)
    with pytest.raises(ValueError):
        select_base(_run(LARGE), _run(SMALL), CLUSTERS, 0)


# ------------------------------------------------------------------ stop rules and sampling


def test_parse_swapusage_and_free_percentage():
    s = parse_swapusage("total = 4096.00M  used = 2947.88M  free = 1148.12M  (encrypted)")
    assert s["total_bytes"] == 4096 * 2**20 and s["used_bytes"] == round(2947.88 * 2**20)
    assert parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M")["used_bytes"] == 0
    assert parse_free_percentage("x\nSystem-wide memory free percentage: 33%\n") == 33
    assert parse_free_percentage("nothing") is None
    with pytest.raises(ValueError):
        parse_swapusage("garbage")


def test_stopwatch_swap_and_pressure():
    w = StopWatch(swap_used_before=1000)
    assert (
        w.check(throughput=50, swap_used=1000 + SWAP_GROWTH_LIMIT_BYTES, pressure_level=2) is None
    )
    assert "swap" in w.check(
        throughput=50, swap_used=1001 + SWAP_GROWTH_LIMIT_BYTES, pressure_level=1
    )
    w = StopWatch(swap_used_before=0)
    assert "critical" in w.check(throughput=50, swap_used=0, pressure_level=4)


def test_stopwatch_sustained_slowdown_needs_three_consecutive_windows():
    w = StopWatch(swap_used_before=0)
    for _ in range(30):
        assert w.check(throughput=80.0, swap_used=0, pressure_level=1) is None
    # Two slow windows, one recovered window, then two slow windows: no stop.
    for tps in [30.0] * 20 + [80.0] * 10 + [30.0] * 20:
        assert w.check(throughput=tps, swap_used=0, pressure_level=1) is None
    # Exactly half is not "below half".
    for _ in range(10):
        assert w.check(throughput=40.0, swap_used=0, pressure_level=1) is None
    w2 = StopWatch(swap_used_before=0)
    for tps in [80.0] * 30 + [39.9] * 29:
        assert w2.check(throughput=tps, swap_used=0, pressure_level=1) is None
    assert "sustained slowdown" in w2.check(throughput=39.9, swap_used=0, pressure_level=1)


def test_fixed_subsets_on_the_frozen_corpus():
    _, corpus = load_corpus(CORPUS)
    ids = repeat_subset(corpus["validation"])
    assert ids == repeat_subset(list(reversed(corpus["validation"])))  # order-independent
    by_id = {e["id"]: e for e in corpus["validation"]}
    strata = [by_id[i]["stratum"] for i in ids]
    assert len(ids) == len(set(ids)) == 30
    assert {s: strata.count(s) for s in set(strata)} == {
        "A_heldout_family": 10,
        "B_novel_instance": 10,
        "C_familiar_instance": 10,
    }
    assert warmup_ids(corpus["train"]) == [f"train-{i:04d}" for i in range(1, 6)]


# ------------------------------------------------------------------ analysis end to end (fake runs)


def _load_analyzer():
    path = ROOT / "benchmarks" / "issue4_exp009_analyze.py"
    spec = importlib.util.spec_from_file_location("issue4_exp009_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_runs(tmp_path: Path) -> Path:
    _, corpus = load_corpus(CORPUS)
    val = corpus["validation"]
    rep = repeat_subset(val)
    by_id = {e["id"]: e for e in val}
    runs = tmp_path / "runs"
    for folder, system, good in (
        ("qwen3-0.6b", SMALL, True),
        ("qwen3-1.7b", LARGE, False),
    ):
        d = runs / folder
        d.mkdir(parents=True)

        def rec(e, system=system, good=good, **extra):
            raw = json.dumps(e["gold"]) if good else "```json\n{}\n```"
            return {
                "example_id": e["id"],
                "system": system,
                "status": "ok",
                "raw_output": raw,
                "elapsed_ms": 800.0,
                "generation_tokens": 40,
                "finish_reason": "stop",
                "prompt_tokens": 430,
                **extra,
            }

        lines = [rec(e) for e in val]
        (d / "main_validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        reps = [rec(by_id[i], repeat=r) for r in (1, 2) for i in rep]
        (d / "main_repeat.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reps))
        snap = {"power_source": "AC Power"}
        meas = {
            "status": "completed",
            "before": snap,
            "rss_peak_process_bytes": 2 * 10**9,
            "rss_peak_after_validation_bytes": 2 * 10**9,
            "validation_completed": True,
            "environment": {},
            "model": {},
            "cold_load_seconds": 1.0,
        }
        (d / "main_measurements.json").write_text(json.dumps(meas))
        fresh = [rec(by_id[i], fresh_process=1) for i in rep]
        (d / "fresh_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fresh))
        (d / "fresh_1_measurements.json").write_text(json.dumps({"status": "completed"}))
    return runs


def test_analysis_pipeline_on_fake_runs(tmp_path, monkeypatch):
    mod = _load_analyzer()
    runs = _fake_runs(tmp_path)
    rules_out = tmp_path / "rules.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["x", "--runs", str(runs), "--rules-predictions", str(rules_out)]
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert mod.main() == 0
    out = json.loads(buf.getvalue())
    em = {
        k: v["raw_candidate"]["primary_semantic_exact_match"]["count"]
        for k, v in out["systems"].items()
    }
    assert em[SMALL] == 160 and em[LARGE] == 0 and em["rules"] == 73
    assert out["selection"]["selected"] == SMALL and out["selection"]["basis"] == "primary"
    rep = out["models"][SMALL]["repeatability"]
    assert rep["requests_identical"]["in_process_text_identical"] == 30
    assert rep["requests_identical"]["fresh_candidate_identical"] == 30
    large = out["models"][LARGE]["repeatability"]["per_request"]
    assert all(v["distinct_texts_all"] == 1 for v in large.values())
    assert len(rules_out.read_text().splitlines()) == 160
    assert out["rules_rerun_consistency"]["raw_output_differs"] == []
    assert out["selection"]["eligibility"][SMALL]["peak_rss_at_most_10GB"]
