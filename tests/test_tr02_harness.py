"""EXP-TR02 harness: pinned inputs, schedule, stop rules, probes and supervisor (no model runs)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from axbench_tr import export, fetch
from axbench_tr.build import CURRENT_VERSION, OUT_ROOT
from axbench_tr.registry import ACQUISITION_PATH

ROOT = Path(__file__).resolve().parents[1]
CORPUS = OUT_ROOT / CURRENT_VERSION


def _load():
    path = ROOT / "training" / "tr02" / "tr02.py"
    spec = importlib.util.spec_from_file_location("tr02", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tr02"] = module
    spec.loader.exec_module(module)
    return module


tr02 = _load()
CFG = tr02.load_config()
PER_EXAMPLE = ROOT / CFG["inputs"]["per_example_path"]


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("export")
    export.export(out)
    return out


# ----------------------------------------------------------------------------- pinned inputs


def test_config_pins_match_the_committed_corpus_and_tokenizer_files() -> None:
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text("utf-8"))
    pinned = CFG["inputs"]["corpus_files_sha256"]
    assert pinned == {n: m["sha256"] for n, m in manifest["files"].items()}
    assert fetch.sha256_file(PER_EXAMPLE) == CFG["inputs"]["per_example_sha256"]
    acquired = json.loads(ACQUISITION_PATH.read_text("utf-8"))["sources"]["SRC-M01"]["files"]
    by_name = {f["path"].rsplit("/", 1)[-1]: f for f in acquired}
    assert CFG["model"]["revision"] in by_name["tokenizer.json"]["url"]
    assert {n: f["sha256"] for n, f in by_name.items()} == CFG["model"]["tokenizer_sha256"]


def test_a_fresh_export_matches_every_pinned_hash(export_dir: Path) -> None:
    assert tr02.verify_inputs(export_dir, CFG) == []


def test_verify_inputs_catches_a_changed_missing_or_extra_file(
    export_dir: Path, tmp_path: Path
) -> None:
    import shutil

    copy = tmp_path / "e"
    shutil.copytree(export_dir, copy)
    (copy / "smoke" / "train.jsonl").write_bytes(b"{}\n")
    (copy / "full" / "valid.jsonl").unlink()
    (copy / "smoke" / "test.jsonl").write_bytes(b"{}\n")
    problems = tr02.verify_inputs(copy, CFG)
    assert "hash differs: smoke/train.jsonl" in problems
    assert "missing: full/valid.jsonl" in problems
    assert "unpinned file: smoke/test.jsonl" in problems
    assert "test file present: smoke/test.jsonl" in problems


# ------------------------------------------------------------------------------------ schedule


def _phase(name: str) -> dict:
    return next(p for p in CFG["phases"] if p["name"] == name)


def test_smoke_exposes_each_of_400_rows_exactly_once_in_100_updates(export_dir: Path) -> None:
    plan = tr02.schedule(export_dir, PER_EXAMPLE, _phase("smoke"), CFG["hyperparameters"]["seed"])
    assert plan["rows"] == plan["microsteps"] == 400
    assert plan["optimizer_updates"] == 100
    assert plan["each_row_exposed_exactly_once"]
    assert len({s["id"] for s in plan["steps"]}) == 400
    assert plan["tokens_total"] == sum(s["tokens"] for s in plan["steps"])
    again = tr02.schedule(export_dir, PER_EXAMPLE, _phase("smoke"), CFG["hyperparameters"]["seed"])
    assert again == plan  # deterministic


def test_probe_runs_the_20_longest_rows_and_shares_none_with_smoke(export_dir: Path) -> None:
    probe = tr02.schedule(export_dir, PER_EXAMPLE, _phase("probe"), CFG["hyperparameters"]["seed"])
    assert probe["optimizer_updates"] == 5 and min(s["tokens"] for s in probe["steps"]) >= 1199
    ids = json.loads((export_dir / "ROW_IDS.json").read_text("utf-8"))
    assert not set(ids["peak_probe/train.jsonl"]) & set(ids["smoke/train.jsonl"])
    assert not set(ids["smoke/valid.jsonl"]) & set(ids["smoke/train.jsonl"])


def test_a_schedule_that_would_repeat_or_skip_rows_is_refused(export_dir: Path) -> None:
    bad = {**_phase("smoke"), "microsteps": 200}
    with pytest.raises(ValueError):
        tr02.schedule(export_dir, PER_EXAMPLE, bad, 2026)


def test_exposure_checker_fails_on_a_wrong_row_a_short_row_or_extra_rows() -> None:
    planned = [{"id": "a", "tokens": 10}, {"id": "b", "tokens": 20}]
    hashes = {"ha": "a", "hb": "b"}
    ok = tr02.ExposureChecker(planned, hashes)
    ok.observe("ha", 10)
    ok.observe("hb", 20)
    assert ok.complete()
    with pytest.raises(RuntimeError):  # more rows than scheduled
        ok.observe("ha", 10)
    swapped = tr02.ExposureChecker(planned, hashes)
    with pytest.raises(RuntimeError):
        swapped.observe("hb", 20)
    truncated = tr02.ExposureChecker(planned, hashes)
    with pytest.raises(RuntimeError, match="expected 10"):
        truncated.observe("ha", 9)  # a truncated sequence is a stop, not a warning


def test_the_run_and_probe_use_the_declared_batching() -> None:
    for p in CFG["phases"]:
        assert p["microsteps"] == p["optimizer_updates"] * p["grad_accumulation"]
        assert p["microsteps_per_report"] == p["grad_accumulation"]  # one report per update
    assert CFG["hyperparameters"]["max_seq_length"] == 2048
    assert CFG["hyperparameters"]["micro_batch"] == 1


# ------------------------------------------------------------------------------------ probes


def test_system_probes_parse_known_macos_output() -> None:
    swap = "total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)"
    assert tr02.parse_swapusage(swap) == pytest.approx(1.02450)
    assert tr02.parse_swapusage("total = 4.00G used = 0.50G free = 3.50G") == pytest.approx(0.5)
    assert tr02.parse_pressure_level("4\n") == 4
    assert tr02.parse_on_battery("Now drawing from 'Battery Power'\n -InternalBattery-0 100%")
    assert not tr02.parse_on_battery("Now drawing from 'AC Power'\n -InternalBattery-0 100%")
    assert tr02.parse_rss_kb(" 2048000\n") == pytest.approx(2.097152)
    for bad in (tr02.parse_swapusage, tr02.parse_on_battery):
        with pytest.raises(ValueError):
            bad("unexpected output")


# ---------------------------------------------------------------------------------- stop rules


def _rules(tokens: list[int] | None = None) -> object:
    return tr02.StopRules(tr02.Limits.from_config(CFG), 4, tokens or [500] * 400)


def _train(it: int, rate: float = 1.0, loss: float = 1.0, peak: float = 5.0) -> dict:
    return {
        "kind": "train",
        "iteration": it,
        "train_loss": loss,
        "peak_memory_gb": peak,
        "iterations_per_second": rate,
        "loss_tokens_per_second": 99999.0,  # never used: loss-token throughput is not the rate
    }


def _os(**kw: object) -> dict:
    base = {
        "kind": "os",
        "rss_gb": 5.0,
        "swap_used_gb": 1.0,
        "pressure_level": 1,
        "on_battery": False,
    }
    return base | kw


def test_each_stop_rule_fires_and_a_healthy_run_does_not() -> None:
    assert _rules().feed(_train(4)) is None and _rules().feed(_os()) is None
    assert _rules().feed(_train(4, loss=float("nan"))) == "non_finite_loss"
    assert _rules().feed(_train(4, loss=float("inf"))) == "non_finite_loss"
    assert _rules().feed(_train(4, peak=10.5)).startswith("mlx_peak_above_limit")
    assert _rules().feed(_os(rss_gb=11.5)).startswith("rss_above_limit")
    assert _rules().feed(_os(pressure_level=4)) == "memory_pressure_critical"
    assert _rules().feed(_os(on_battery=True)) == "battery_power"
    assert _rules().feed({"kind": "error", "message": "MemoryError"}).startswith("error:")
    r = _rules()
    assert r.feed(_os(swap_used_gb=1.0)) is None  # baseline
    assert r.feed(_os(swap_used_gb=2.9)) is None
    assert r.feed(_os(swap_used_gb=3.1)) == "swap_growth_above_limit"


def test_mlx_peak_and_rss_are_separate_rules_with_separate_limits() -> None:
    limits = tr02.Limits.from_config(CFG)
    assert limits.mlx_peak_gb == 10.0 and limits.rss_gb == 11.0
    r = _rules()
    assert r.feed(_train(4, peak=9.9)) is None  # high RSS alone is not an MLX-peak breach
    assert r.feed(_os(rss_gb=10.9)) is None
    assert r.feed(_os(rss_gb=11.1)) is not None


def _windows(rules: object, rates: list[float]) -> str | None:
    it = 0
    for rate in rates:
        for _ in range(10):  # ten reports (one per update) make a window
            it += 4
            if reason := rules.feed(_train(it, rate=rate)):
                return reason
    return None


def test_throughput_rule_uses_full_tokens_over_windows_and_needs_three_in_a_row() -> None:
    # a 500-token row at 1.0 it/s per microstep step: first window sets the reference
    assert _windows(_rules(), [1.0, 0.4, 0.4]) is None  # only two bad windows
    assert _windows(_rules(), [1.0, 0.4, 0.4, 0.4]) == (
        "throughput_below_half_of_first_window_for_3_windows"
    )
    assert _windows(_rules(), [1.0, 0.4, 0.4, 0.9, 0.4, 0.4]) is None  # a good window resets
    assert _windows(_rules(), [1.0, 0.6, 0.6, 0.6, 0.6]) is None  # slower but above half


def test_window_rate_is_full_tokens_per_compute_second_not_loss_tokens() -> None:
    r = _rules([100, 300] * 200)
    _windows(r, [2.0])
    # 40 microsteps: 20 x 100 + 20 x 300 = 8,000 tokens in 40 / 2.0 = 20 s of compute
    assert r.window_rates[0] == pytest.approx(400.0)


# ------------------------------------------------------------------------------- supervisor


def _side(out: Path) -> Path:
    return out.with_name(out.name + ".supervisor")


CHILD_EVENTS = """
import json, sys, time
out = sys.argv[1]
with open(out + "/events.jsonl", "a", encoding="utf-8") as fh:
    for ev in json.loads(sys.argv[2]):
        fh.write(json.dumps(ev) + "\\n"); fh.flush()
    time.sleep(float(sys.argv[3]))
sys.exit(int(sys.argv[4]))
"""


def _run_supervised(tmp_path: Path, events: list[dict], sleep: float, code: int, os_sample: dict):
    cmd = [
        sys.executable,
        "-c",
        CHILD_EVENTS,
        str(tmp_path),
        json.dumps(events),
        str(sleep),
        str(code),
    ]
    return tr02.supervise(
        cmd,
        tmp_path,
        _rules(),
        lambda pid: {**os_sample, "t": 0.0},
        poll_s=0.05,
        kill_after_s=5,
    )


def test_supervisor_lets_a_healthy_child_finish(tmp_path: Path) -> None:
    result = _run_supervised(tmp_path, [_train(4)], 0.2, 0, _os())
    assert result == {"stopped": False, "reason": None, "child_exit_code": 0}
    assert not (_side(tmp_path) / "STOP.json").exists()


def test_supervisor_stops_the_child_on_a_rule_and_does_not_retry(tmp_path: Path) -> None:
    result = _run_supervised(tmp_path, [_train(4, peak=11.0)], 30.0, 0, _os())
    assert result["stopped"] and result["reason"].startswith("mlx_peak_above_limit")
    assert result["child_exit_code"] != 0  # terminated well before its 30 s sleep
    assert json.loads((_side(tmp_path) / "STOP.json").read_text("utf-8"))["stopped"] is True


def test_supervisor_stops_on_an_os_sample_and_on_a_failing_child(tmp_path: Path) -> None:
    assert _run_supervised(tmp_path, [], 30.0, 0, _os(on_battery=True))["reason"] == "battery_power"
    other = tmp_path / "b"
    other.mkdir()
    assert _run_supervised(other, [], 0.1, 3, _os())["reason"] == "child_exit_code_3"


def test_an_unreadable_probe_is_a_stop_not_a_skip(tmp_path: Path) -> None:
    def broken(pid: int) -> dict:
        raise OSError("sysctl missing")

    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    result = tr02.supervise(cmd, tmp_path, _rules(), broken, poll_s=0.05, kill_after_s=5)
    assert result["reason"] == "probe_failed:OSError"


# --------------------------------------------------------- data path (optional dependencies)


def _tokenizer_dir() -> Path | None:
    hits = sorted((fetch.raw_dir() / "hf" / "Qwen" / "Qwen3-1.7B").glob("*/tokenizer.json"))
    return hits[0].parent if hits else None


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    where = _tokenizer_dir()
    if where is None:
        pytest.skip("pinned tokenizer files not downloaded")
    for name, digest in CFG["model"]["tokenizer_sha256"].items():
        assert fetch.sha256_file(where / name) == digest
    return transformers.AutoTokenizer.from_pretrained(str(where))


def test_harness_dataset_matches_the_measured_exp010_lengths(export_dir: Path, tokenizer) -> None:
    lengths = tr02.row_lengths(export_dir, PER_EXAMPLE)["smoke/train.jsonl"]
    rows = tr02.read_jsonl(export_dir / "smoke" / "train.jsonl")[:25]
    report = tr02.check_rows(rows, list(lengths)[:25], lengths, tokenizer, 2048)
    assert report["problems"] == []
    assert report["supervised_min"] > 0


def test_harness_dataset_refuses_to_truncate(export_dir: Path, tokenizer) -> None:
    rows = tr02.read_jsonl(export_dir / "peak_probe" / "train.jsonl")[:1]
    ds = tr02.TR02ChatDataset(rows, tokenizer, 1024)
    with pytest.raises(ValueError, match="exceeds 1024"):
        ds.process(rows[0])


def test_harness_planned_order_equals_the_stock_mlx_lm_iterator(
    export_dir: Path, tokenizer
) -> None:
    pytest.importorskip("mlx_lm")
    import numpy as np
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import iterate_batches

    rows = tr02.read_jsonl(export_dir / "peak_probe" / "train.jsonl")
    ds = CacheDataset(tr02.TR02ChatDataset(rows, tokenizer, 2048))
    want = [tr02.tokens_hash(ds[i][0]) for i in tr02.planned_order(len(rows), 2026)]
    np.random.seed(2026)
    stream = iterate_batches(dataset=ds, batch_size=1, max_seq_length=2048, loop=True)
    got = []
    for _, (batch, lengths) in zip(range(len(rows)), stream, strict=False):
        got.append(tr02.tokens_hash(batch[0, : int(lengths[0, 1])].tolist()))
    assert got == want


def test_supervisor_works_when_the_driver_creates_its_own_output_directory(tmp_path: Path) -> None:
    out = tmp_path / "phase"  # does not exist yet: the driver requires a new, empty directory
    code = (
        "import os, sys, time\n"
        "os.makedirs(sys.argv[1]); time.sleep(0.3)\n"
        "open(sys.argv[1] + '/events.jsonl', 'w').write(sys.argv[2] + '\\n'); time.sleep(30)\n"
    )
    event = json.dumps(_train(4, loss=float("nan")))
    result = tr02.supervise(
        [sys.executable, "-c", code, str(out), event],
        out,
        _rules(),
        lambda pid: _os(),
        poll_s=0.05,
        kill_after_s=5,
    )
    assert result["reason"] == "non_finite_loss"
    assert (out.with_name("phase.supervisor") / "STOP.json").exists()
