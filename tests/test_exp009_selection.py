"""EXP-009 (Issue #4, Phase 3B): the frozen base-selection rule, stop rules and analysis pipeline.

The selection rule and stop rules were fixed in the protocol (revision 3) and the pre-run record
before any validation prediction; these tests pin the encoded logic to the written rule. The
regression tests for the lead's pre-run review (PRERUN revision 1) cover the pre-load resource
baseline, load and warm-up checks, MemoryError, partial repeatability, failed fresh processes,
eligibility over all required processes and the integrity checks of the analysis.
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
    AllocationStop,
    ProcessMonitor,
    SlowdownWatch,
    is_allocation_error,
    parse_free_percentage,
    parse_swapusage,
    preflight,
    repeat_subset,
    run_requests,
    timed_call,
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


# ------------------------------------------------------------------ preflight and resources

AC = "AC Power"


def _s(swap=0, level=1, power=AC):
    return {"swap_used": swap, "pressure_level": level, "power_source": power}


def test_parse_swapusage_and_free_percentage():
    s = parse_swapusage("total = 4096.00M  used = 2947.88M  free = 1148.12M  (encrypted)")
    assert s["total_bytes"] == 4096 * 2**20 and s["used_bytes"] == round(2947.88 * 2**20)
    assert parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M")["used_bytes"] == 0
    assert parse_free_percentage("x\nSystem-wide memory free percentage: 33%\n") == 33
    assert parse_free_percentage("nothing") is None
    with pytest.raises(ValueError):
        parse_swapusage("garbage")


def test_preflight_requires_mains_power_and_normal_pressure():
    assert preflight(_s()) is None
    assert "mains" in preflight(_s(power="Battery Power"))
    assert "not normal" in preflight(_s(level=2))
    assert "not normal" in preflight(_s(level=4))


def test_swap_growth_is_cumulative_from_the_pre_load_baseline():
    # Growth during load and warm-up counts; it is not hidden by a later re-baseline.
    m = ProcessMonitor(baseline=_s(swap=1000))
    assert m.check("load", _s(swap=1000 + 1_500_000_000)) is None
    assert m.check("warmup", _s(swap=1000 + 1_900_000_000)) is None
    assert m.check("validation", _s(swap=1000 + SWAP_GROWTH_LIMIT_BYTES)) is None
    reason = m.check("validation", _s(swap=1001 + SWAP_GROWTH_LIMIT_BYTES))
    assert reason.startswith("validation: swap growth")
    assert m.check("repeat", _s()) == reason  # stays stopped
    summary = m.summary()
    assert summary["max_cumulative_swap_growth_bytes"] == 1 + SWAP_GROWTH_LIMIT_BYTES
    assert set(summary["phases"]) == {"load", "warmup", "validation"}
    assert summary["phases"]["load"]["first_swap_used"] == 1000 + 1_500_000_000


def test_load_and_warmup_checks_stop_on_pressure_and_power():
    m = ProcessMonitor(baseline=_s())
    assert m.check("load", _s(level=4)) == "load: memory pressure level 4 (critical)"
    m = ProcessMonitor(baseline=_s())
    assert m.check("load", _s(level=2)) is None  # warning is recorded, not a stop
    assert "mains power interrupted" in m.check("warmup", _s(power="Battery Power"))
    assert m.summary()["max_pressure_level"] == 2


def test_slowdown_needs_three_consecutive_windows_below_half():
    w = SlowdownWatch()
    for _ in range(30):
        assert w.check(80.0) is None
    for tps in [30.0] * 20 + [80.0] * 10 + [30.0] * 20:
        assert w.check(tps) is None
    for _ in range(10):
        assert w.check(40.0) is None  # exactly half is not below half
    w2 = SlowdownWatch()
    for tps in [80.0] * 30 + [39.9] * 29:
        assert w2.check(tps) is None
    assert "sustained slowdown" in w2.check(39.9)


def test_memory_error_stops_regardless_of_message():
    assert is_allocation_error(MemoryError())
    assert is_allocation_error(RuntimeError("[metal::malloc] Attempting to allocate 9 GB"))
    assert not is_allocation_error(ValueError("bad token"))

    def boom():
        raise MemoryError

    with pytest.raises(AllocationStop) as info:
        timed_call({"example_id": "v1"}, boom)
    assert info.value.record["status"] == "error"
    assert info.value.record["error"] == "allocation:MemoryError"


def test_other_exceptions_are_request_errors_in_the_denominator():
    def bad():
        raise ValueError("decode")

    rec = timed_call({"example_id": "v1"}, bad)
    assert rec["status"] == "error" and rec["error"] == "ValueError" and rec["raw_output"] is None
    rec = timed_call({"example_id": "v2"}, lambda: {"raw_output": "x", "generation_tps": 5.0})
    assert rec["status"] == "ok" and rec["raw_output"] == "x"


def test_run_requests_emits_before_checking_and_keeps_the_failed_record():
    emitted, samples = [], []

    def request(i):
        if i == 3:
            return timed_call({"example_id": i}, lambda: (_ for _ in ()).throw(MemoryError()))
        return timed_call({"example_id": i}, lambda: {"raw_output": "{}", "generation_tps": 9.0})

    def sample():
        samples.append(1)
        return _s()

    m = ProcessMonitor(baseline=_s())
    reason = run_requests(range(6), request, sample, m, "validation", emitted.append)
    assert reason.startswith("validation: allocation error: MemoryError")
    assert [r["example_id"] for r in emitted] == [0, 1, 2, 3]
    assert emitted[-1]["error"] == "allocation:MemoryError"
    assert len(samples) == 3  # one resource sample after every completed request

    emitted.clear()
    m = ProcessMonitor(baseline=_s())
    values = iter([_s(), _s(level=4), _s()])
    reason = run_requests(range(5), lambda i: {"example_id": i}, lambda: next(values), m,
                          "fresh", emitted.append)  # fmt: skip
    assert "critical" in reason and len(emitted) == 2


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


# ------------------------------------------------------------------ analysis (fake runs)

COMMIT = "a" * 40
REVISIONS = {
    "qwen3-0.6b": "c1899de289a04d12100db370d81485cdf75e47ca",
    "qwen3-1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
MANIFEST = "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"


def _load_analyzer():
    path = ROOT / "benchmarks" / "issue4_exp009_analyze.py"
    spec = importlib.util.spec_from_file_location("issue4_exp009_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _meas(folder, system, phase, **extra):
    return {
        "experiment": "EXP-009",
        "phase": phase,
        "system": system,
        "status": "completed",
        "corpus_manifest_sha256": MANIFEST,
        "model": {"revision": REVISIONS[folder]},
        "environment": {
            "runner_version": "exp009.runner/2",
            "git": {"commit": COMMIT, "dirty": False},
        },  # fmt: skip
        "preflight": {"power_source": AC},
        "rss_peak_process_bytes": 2 * 10**9,
        "cold_load_seconds": 1.0,
        **extra,
    }


def _fake_runs(tmp_path: Path, good=(True, False)) -> Path:
    _, corpus = load_corpus(CORPUS)
    val = corpus["validation"]
    rep = repeat_subset(val)
    by_id = {e["id"]: e for e in val}
    runs = tmp_path / "runs"
    for (folder, system), ok in zip(
        (("qwen3-0.6b", SMALL), ("qwen3-1.7b", LARGE)), good, strict=True
    ):
        d = runs / folder
        d.mkdir(parents=True)

        def rec(e, system=system, ok=ok, **extra):
            raw = json.dumps(e["gold"]) if ok else "```json\n{}\n```"
            return {"example_id": e["id"], "system": system, "status": "ok", "raw_output": raw,
                    "elapsed_ms": 800.0, "generation_tokens": 40, "finish_reason": "stop",
                    "prompt_tokens": 430, **extra}  # fmt: skip

        _write_jsonl(d / "main_validation.jsonl", [rec(e) for e in val])
        _write_jsonl(
            d / "main_repeat.jsonl", [rec(by_id[i], repeat=r) for r in range(1, 21) for i in rep]
        )
        main = _meas(folder, system, "main", validation_completed=True,
                     rss_peak_after_validation_bytes=2 * 10**9)  # fmt: skip
        (d / "main_measurements.json").write_text(json.dumps(main))
        for f in range(1, 6):
            _write_jsonl(
                d / f"fresh_{f}_repeat.jsonl", [rec(by_id[i], fresh_process=f) for i in rep]
            )
            (d / f"fresh_{f}_measurements.json").write_text(
                json.dumps(_meas(folder, system, f"fresh_{f}"))
            )
    return runs


def _analyze(tmp_path, runs, monkeypatch) -> dict:
    mod = _load_analyzer()
    rules_out = tmp_path / "rules.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["x", "--runs", str(runs), "--rules-predictions", str(rules_out)]
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert mod.main() == 0
    assert len(rules_out.read_text().splitlines()) == 160
    return json.loads(buf.getvalue())


def _edit_jsonl(path: Path, fn) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    _write_jsonl(path, fn(records))


def _edit_json(path: Path, **changes) -> None:
    obj = json.loads(path.read_text())
    obj.update(changes)
    path.write_text(json.dumps(obj))


def test_analysis_complete_runs(tmp_path, monkeypatch):
    out = _analyze(tmp_path, _fake_runs(tmp_path), monkeypatch)
    em = {
        k: v["raw_candidate"]["primary_semantic_exact_match"]["count"]
        for k, v in out["systems"].items()
    }
    assert em[SMALL] == 160 and em[LARGE] == 0 and em["rules"] == 73
    assert out["rules_rerun_consistency"]["raw_output_differs"] == []
    sel = out["selection"]
    assert sel["selected"] == SMALL and sel["basis"] == "primary" and sel["issues"] == []
    assert all(sel["eligibility"][m]["completed_without_stop_rule"] for m in (SMALL, LARGE))
    rep = out["models"][SMALL]["repeatability"]["summary"]
    assert rep["in_process_text"] == {"identical": 30, "differs": 0, "not_assessable": 0,
                                      "n_requests": 30}  # fmt: skip
    assert rep["fresh_candidate"]["identical"] == 30
    assert rep["in_process_observations"] == {"observed": 600, "expected": 600}
    # The 1.7B outputs are never schema-valid: candidate stability is not claimed.
    large = out["models"][LARGE]["repeatability"]["summary"]
    assert large["in_process_candidate"]["not_assessable"] == 30
    assert large["in_process_text"]["identical"] == 30


def test_partial_repeat_is_not_assessable_and_makes_the_model_ineligible(tmp_path, monkeypatch):
    runs = _fake_runs(tmp_path)
    _edit_jsonl(runs / "qwen3-0.6b" / "main_repeat.jsonl", lambda r: r[:-35])
    out = _analyze(tmp_path, runs, monkeypatch)
    rep = out["models"][SMALL]["repeatability"]["summary"]
    assert rep["in_process_observations"] == {"observed": 565, "expected": 600}
    assert rep["in_process_text"]["not_assessable"] == 30  # last repeat incomplete everywhere
    assert out["models"][SMALL]["record_checks"]["repeat"]["missing"] == 35
    assert not out["models"][SMALL]["all_required_processes_completed"]
    assert out["models"][SMALL]["validation_only"]["completed"]
    assert out["selection"]["selected"] == LARGE
    assert out["selection"]["basis"] == "single_eligible"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "stopped", "stop_reason": "fresh_3: memory pressure level 4 (critical)"},
        {"status": "failed", "failure": "RuntimeError: x"},
        {"status": "running"},
        {"rss_peak_process_bytes": 10 * 10**9 + 1},
    ],
)
def test_a_failed_or_oversized_fresh_process_makes_the_model_ineligible(
    tmp_path, monkeypatch, change
):
    runs = _fake_runs(tmp_path)
    _edit_json(runs / "qwen3-0.6b" / "fresh_3_measurements.json", **change)
    out = _analyze(tmp_path, runs, monkeypatch)
    elig = out["selection"]["eligibility"][SMALL]
    assert not (elig["completed_without_stop_rule"] and elig["peak_rss_at_most_10GB"])
    assert out["selection"]["selected"] == LARGE
    fresh3 = out["models"][SMALL]["fresh_processes"]["3"]
    assert fresh3["status"] == change.get("status", "completed")


def test_missing_fresh_output_is_incomplete_not_assumed(tmp_path, monkeypatch):
    runs = _fake_runs(tmp_path)
    (runs / "qwen3-0.6b" / "fresh_5_repeat.jsonl").unlink()
    out = _analyze(tmp_path, runs, monkeypatch)
    assert out["models"][SMALL]["record_checks"]["fresh_5"]["present"] is False
    assert out["models"][SMALL]["repeatability"]["summary"]["fresh_text"]["not_assessable"] == 30
    assert not out["selection"]["eligibility"][SMALL]["completed_without_stop_rule"]


def test_failed_repeat_requests_are_counted_separately(tmp_path, monkeypatch):
    runs = _fake_runs(tmp_path)

    def fail_one(records):
        records[0] = {**records[0], "status": "error", "raw_output": None}
        return records

    _edit_jsonl(runs / "qwen3-0.6b" / "fresh_2_repeat.jsonl", fail_one)
    out = _analyze(tmp_path, runs, monkeypatch)
    rep = out["models"][SMALL]["repeatability"]["summary"]
    assert rep["fresh_failed_requests"] == 1
    assert rep["fresh_text"]["not_assessable"] == 1 and rep["fresh_text"]["identical"] == 29


@pytest.mark.parametrize("defect", ["duplicate", "commit", "revision", "dirty", "system"])
def test_integrity_issues_block_the_selection(tmp_path, monkeypatch, defect):
    runs = _fake_runs(tmp_path)
    d = runs / "qwen3-1.7b"
    if defect == "duplicate":
        _edit_jsonl(d / "main_validation.jsonl", lambda r: [*r, r[0]])
    elif defect == "commit":
        env = {"runner_version": "exp009.runner/2", "git": {"commit": "b" * 40, "dirty": False}}
        _edit_json(d / "fresh_1_measurements.json", environment=env)
    elif defect == "revision":
        _edit_json(d / "main_measurements.json", model={"revision": "0" * 40})
    elif defect == "dirty":
        env = {"runner_version": "exp009.runner/2", "git": {"commit": COMMIT, "dirty": True}}
        _edit_json(d / "main_measurements.json", environment=env)
    else:
        _edit_jsonl(d / "main_repeat.jsonl", lambda r: [{**r[0], "system": "other"}, *r[1:]])
    if defect == "duplicate":
        with pytest.raises(ValueError):  # the harness refuses duplicate prediction ids
            _analyze(tmp_path, runs, monkeypatch)
        return
    out = _analyze(tmp_path, runs, monkeypatch)
    assert out["selection"]["selected"] is None
    assert out["selection"]["basis"] == "integrity_failure" and out["selection"]["issues"]
