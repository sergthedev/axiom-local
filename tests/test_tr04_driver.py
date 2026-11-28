"""EXP-TR04 training driver without weights: config, pins, export, schedule, budget, stop rules.

Only synthetic ids and event streams, the committed audit outputs and a fresh MLX export of the
kept rows are used. No model is loaded, no optimizer step is taken and no weight file exists.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from axbench_tr import tr04 as tr04data

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drv = _load("tr04_driver", ROOT / "training" / "tr04" / "tr04.py")
tr02 = drv.tr02
CFG = drv.load_config()
COMMIT = "a" * 40
THERM_OK = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tr04export")
    tr04data.export_mlx(out)
    return out


# ----------------------------------------------------------------------------------- config


def test_config_is_internally_and_externally_consistent() -> None:
    assert drv.consistency_problems(CFG) == []


def test_config_says_exploratory_not_approved_and_not_validated() -> None:
    status = CFG["status"]
    for phrase in ("not approved", "no clearance", "exploratory", "not human-validated"):
        assert phrase in status
    assert CFG["evaluation"]["human_validated"] is False
    assert CFG["evaluation"]["deployment_ready"] is False
    assert CFG["hyperparameters"]["init_adapter"] is None
    assert CFG["hyperparameters"]["epochs"] == 1


def test_config_copies_tr02_pins_unchanged() -> None:
    tr02_cfg = json.loads((ROOT / "training/tr02/config.json").read_text("utf-8"))
    assert CFG["model"] == tr02_cfg["model"] and len(CFG["model"]["files"]) == 10
    assert CFG["packages"] == tr02_cfg["packages"] and CFG["python"] == tr02_cfg["python"]
    for key, value in tr02_cfg["hyperparameters"].items():
        assert CFG["hyperparameters"][key] == value
    for key in ("mlx_peak_gb", "rss_gb", "swap_growth_gb", "pressure_critical"):
        assert CFG["limits"][key] == tr02_cfg["limits"][key]
    assert (CFG["limits"]["mlx_peak_gb"], CFG["limits"]["rss_gb"]) == (10.0, 11.0)
    assert CFG["inherits"]["tr02_config_sha256"] == drv.sha256_file(drv.TR02_CONFIG_PATH)


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda c: c["model"].update(revision="0" * 40), "model_pin_differs_from_tr02"),
        (lambda c: c["packages"].update(mlx="0.0.1"), "packages_differ_from_tr02"),
        (
            lambda c: c["hyperparameters"].update(learning_rate=1e-4),
            "hyperparameter_learning_rate_differs_from_tr02",
        ),
        (lambda c: c["hyperparameters"].update(epochs=2), "epochs_must_be_one"),
        (lambda c: c["hyperparameters"].update(init_adapter="x"), "init_adapter_must_be_null"),
        (
            lambda c: c["hyperparameters"].update(grad_accumulation=8),
            "effective_batch_is_not_1_by_4",
        ),
        (lambda c: c["limits"].update(mlx_peak_gb=12.0), "limit_mlx_peak_gb_differs_from_tr02"),
        (lambda c: c["limits"].update(rss_gb=14.0), "limit_rss_gb_differs_from_tr02"),
        (
            lambda c: c["limits"].update(reports_per_window=10),
            "reports_per_window_differs_from_monitor_updates_per_window",
        ),
        (
            lambda c: c["monitor"].update(slow_ratio=1.2),
            "monitor_ratios_must_satisfy_0<floor<slow<1",
        ),
        (
            lambda c: c.update(status="approved; cleared"),
            "status_must_say:not approved",
        ),
        (
            lambda c: c["phases"][0].update(microsteps=3000),
            "train_microsteps_differ_from_the_kept_train_rows",
        ),
        (
            lambda c: c["phases"][0].update(optimizer_updates=1),
            "train_optimizer_updates_inconsistent",
        ),
        (
            lambda c: c["phases"][0].update(microsteps_per_report=2),
            "one_report_per_optimizer_update_required",
        ),
        (lambda c: c["phases"][1].update(requires="smoke"), "reload_must_require_train"),
        (
            lambda c: c["phases"][0].update(requires="probe"),
            "train_phase_must_not_require_a_prior_phase",
        ),
        (lambda c: c["phases"].pop(), "phases_must_be_train_then_reload"),
        (lambda c: c["data"].update(never_used=["x"]), "never_used_must_name:sealed"),
        (
            lambda c: c["evaluation"].update(deployment_ready=True),
            "evaluation_must_say_not_human_validated_not_deployment_ready",
        ),
        (lambda c: c["inherits"].update(tr02_config_sha256="0" * 64), "tr02_config_hash_differs"),
        (lambda c: c.update(schema="other"), "schema_is_not_exp_tr04_config_1"),
    ],
)
def test_config_mutations_are_caught(mutate, problem: str) -> None:
    cfg = copy.deepcopy(CFG)
    mutate(cfg)
    assert problem in drv.config_problems(cfg)


def test_pinned_audit_and_heldout_hashes_match_the_committed_files(tmp_path: Path) -> None:
    assert drv.data_problems(CFG) == []
    assert CFG["inputs"]["audit_files_sha256"] == drv.data_hashes()["audit_files_sha256"]
    assert CFG["inputs"]["heldout_files_sha256"] == drv.data_hashes()["heldout_files_sha256"]
    # a changed, a missing and an extra pin are all problems
    root = tmp_path / "tr04"
    shutil.copytree(drv.DATA_DIR, root)
    assert drv.data_problems(CFG, root) == []
    (root / "KEEP_TRAIN.txt").write_text("x\n", "utf-8")
    (root / "heldout" / "HELDOUT.jsonl").unlink()
    problems = drv.data_problems(CFG, root)
    assert "audit_files_sha256: hash differs KEEP_TRAIN.txt" in problems
    assert "heldout_files_sha256: missing HELDOUT.jsonl" in problems


def test_frozen_budget_equals_a_fresh_computation_and_caps_follow_it() -> None:
    fresh = drv.budget(CFG)
    assert CFG["budget"] == fresh
    assert CFG["phases"][0]["max_wall_s"] == fresh["seconds"]["hard_wall_cap_train_phase"]
    assert CFG["phases"][1]["max_wall_s"] == fresh["seconds"]["hard_wall_cap_reload_phase"]
    assert "not measured at this scale" in fresh["status"]
    broken = copy.deepcopy(CFG)
    broken["budget"]["seconds"]["hard_wall_cap_total"] += 900
    assert "frozen_budget_differs_from_a_fresh_computation" in drv.consistency_problems(broken)
    broken = copy.deepcopy(CFG)
    broken["phases"][0]["max_wall_s"] += 900
    assert "train_wall_cap_differs_from_budget" in drv.consistency_problems(broken)


def test_budget_arithmetic_by_hand() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["rates"] = {
        "full_tokens_per_s_slowest_observed": 1000.0,
        "validation_seconds_per_row_slowest_observed": 0.5,
        "model_load_seconds": 10.0,
        "thermal_allowance_fraction": 0.25,
        "wall_cap_factor": 1.25,
        "wall_cap_round_to_seconds": 900,
    }
    lengths = {"train": [1000] * 40, "validation": [100] * 20}
    b = drv.budget(cfg, lengths)
    assert b["full_tokens_exposed"] == 40_000
    # training 40 s; validation 0.5 s x 20 rows x 3 passes = 30 s; two loads = 20 s
    assert b["seconds"]["training_nominal"] == 40
    assert b["seconds"]["validation_nominal_three_passes"] == 30
    assert b["seconds"]["total_nominal"] == 90
    assert b["seconds"]["total_conservative"] == round(20 + (40 + 30) / 0.75)
    assert b["seconds"]["hard_wall_cap_train_phase"] == 900  # (10 + (40+20)/0.75) * 1.25 -> 900
    assert b["seconds"]["hard_wall_cap_total"] == 900


# ----------------------------------------------------------------------- export and schedule


def test_a_fresh_export_matches_every_pinned_hash(export_dir: Path) -> None:
    assert tr02.verify_inputs(export_dir, CFG) == []
    assert drv.export_keep_problems(export_dir) == []
    assert tr02.verify_corpus(CFG) == []
    names = sorted(
        p.relative_to(export_dir).as_posix() for p in export_dir.rglob("*") if p.is_file()
    )
    assert names == sorted(
        ["EXPORT_MANIFEST.json", "ROW_IDS.json", "tr04/train.jsonl", "tr04/valid.jsonl"]
    )
    assert not any("test" in n or "heldout" in n.lower() for n in names)


def test_export_rows_are_exactly_the_kept_rows_and_no_heldout_row(export_dir: Path) -> None:
    keep = tr04data.kept_ids()
    train = tr02.read_jsonl(export_dir / "tr04" / "train.jsonl")
    valid = tr02.read_jsonl(export_dir / "tr04" / "valid.jsonl")
    assert (len(train), len(valid)) == (len(keep["train"]), len(keep["validation"]))
    assert len(train) % 4 == 0
    ids = json.loads((export_dir / "ROW_IDS.json").read_text("utf-8"))
    assert ids == {"tr04/train.jsonl": keep["train"], "tr04/valid.jsonl": keep["validation"]}
    assert not set(keep["train"]) & set(keep["validation"])


def test_export_keep_problems_catch_a_swapped_extra_or_heldout_row(
    tmp_path: Path, export_dir: Path
) -> None:
    work = tmp_path / "export"
    shutil.copytree(export_dir, work)
    ids = json.loads((work / "ROW_IDS.json").read_text("utf-8"))
    swapped = copy.deepcopy(ids)
    swapped["tr04/train.jsonl"][0], swapped["tr04/train.jsonl"][1] = (
        swapped["tr04/train.jsonl"][1],
        swapped["tr04/train.jsonl"][0],
    )
    (work / "ROW_IDS.json").write_text(json.dumps(swapped), "utf-8")
    assert "export rows differ from the keep list: tr04/train.jsonl" in drv.export_keep_problems(
        work
    )
    extra = copy.deepcopy(ids) | {"tr04/test.jsonl": ["x"]}
    (work / "ROW_IDS.json").write_text(json.dumps(extra), "utf-8")
    assert "unexpected export part: tr04/test.jsonl" in drv.export_keep_problems(work)
    held = copy.deepcopy(ids)
    held["tr04/valid.jsonl"][0] = "h4-00001"
    (work / "ROW_IDS.json").write_text(json.dumps(held), "utf-8")
    assert "a held-out row id is in the training export" in drv.export_keep_problems(work)


def test_train_phase_exposes_every_kept_row_exactly_once_in_791_style_updates(
    export_dir: Path,
) -> None:
    phase = tr02.phase_config(CFG, "train")
    plan = tr02.schedule(export_dir, CFG, phase)
    n = len(tr04data.kept_ids()["train"])
    assert plan["rows"] == plan["microsteps"] == n
    assert plan["optimizer_updates"] == n // 4 == phase["optimizer_updates"]
    assert plan["each_row_exposed_exactly_once"] is True
    assert plan["tokens_total"] == CFG["budget"]["full_tokens_exposed"]
    assert plan["padded_tokens_total"] == CFG["budget"]["padded_tokens_exposed"]
    keep = tr04data.kept_ids()["train"]
    assert sorted(s["id"] for s in plan["steps"]) == sorted(keep)
    assert max(s["tokens"] for s in plan["steps"]) <= CFG["hyperparameters"]["max_seq_length"]


def test_validation_set_is_disjoint_from_train_and_from_the_heldout_benchmark() -> None:
    keep = tr04data.kept_ids()
    held = {r["id"] for r in drv.tr04_heldout.load_rows()}
    assert not set(keep["train"]) & set(keep["validation"])
    assert not (set(keep["train"]) | set(keep["validation"])) & held


# ------------------------------------------------------------------------------ preflight


def _fake_model(tmp_path: Path) -> tuple[Path, dict]:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    files = {
        "config.json": b'{"hidden_size": 8}',
        "tokenizer.json": b"tok",
        "tokenizer_config.json": b"tokcfg",
        "model.safetensors": b"weights",
    }
    for name, data in files.items():
        (model / name).write_bytes(data)
    (model / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "revision": CFG["model"]["revision"],
                "files": {"model.safetensors": {"verified_against": "published hash"}},
            }
        ),
        "utf-8",
    )
    cfg = copy.deepcopy(CFG)
    cfg["model"]["files"] = {
        n: {"bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in files.items()
    }
    cfg["model"]["tokenizer_sha256"] = {
        n: cfg["model"]["files"][n]["sha256"] for n in ("tokenizer.json", "tokenizer_config.json")
    }
    return model, cfg


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _repo(tmp_path: Path) -> tuple[Path, str]:
    """A throw-away git repository with the corpus files and the audit outputs the config pins."""
    repo = tmp_path / "repo"
    rel = Path("data/axbench-tr") / CFG["inputs"]["corpus_version"]
    (repo / rel).mkdir(parents=True)
    per = Path(CFG["inputs"]["per_example_path"]).name
    for name in (*CFG["inputs"]["corpus_files_sha256"], per):
        shutil.copy2(ROOT / rel / name, repo / rel / name)
    shutil.copytree(drv.DATA_DIR, repo / "data" / "axbench-tr" / "tr04")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x", "--no-gpg-sign")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def _preflight(tmp_path: Path, export_dir: Path, **kw):
    model, cfg = _fake_model(tmp_path)
    repo, head = _repo(tmp_path)
    args = {
        "config": cfg,
        "phase_name": "train",
        "export_dir": export_dir,
        "model_dir": model,
        "out": tmp_path / "out" / "train",
        "after": None,
        "expect_commit": head,
        "repo": repo,
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "versions": {"python": "3.12", "mlx": "0.32.4", "mlx-lm": "0.32.2"},
    }
    return drv.preflight(**(args | kw)), repo


# a fake model cannot carry the real model pins, so exactly this one problem is expected
FAKE_MODEL_ONLY = ["config: model_pin_differs_from_tr02"]


def test_preflight_passes_everything_except_the_deliberately_fake_model_pin(
    tmp_path: Path, export_dir: Path
) -> None:
    (record, problems), _ = _preflight(tmp_path, export_dir)
    assert problems == FAKE_MODEL_ONLY
    assert record["experiment"] == "EXP-TR04" and record["phase"] == "train"
    assert record["audit_files_sha256"] == CFG["inputs"]["audit_files_sha256"]
    assert record["heldout_files_sha256"] == CFG["inputs"]["heldout_files_sha256"]


def test_preflight_names_a_changed_audit_file_a_dirty_tree_and_a_wrong_export(
    tmp_path: Path, export_dir: Path
) -> None:
    work = tmp_path / "export"
    shutil.copytree(export_dir, work)
    (work / "tr04" / "valid.jsonl").write_bytes(b"")
    (_, problems), repo = _preflight(tmp_path, work)
    assert "hash differs: tr04/valid.jsonl" in problems
    (repo / "data/axbench-tr/tr04/KEEP_TRAIN.txt").write_text("tampered\n", "utf-8")
    (_, problems), _ = _preflight(tmp_path / "two", export_dir, repo=repo, expect_commit="b" * 40)
    text = "; ".join(problems)
    assert "data: audit_files_sha256: hash differs KEEP_TRAIN.txt" in text
    assert "not clean" in text and "owner-approved" in text


def test_preflight_requires_offline_mode_and_the_train_phase_needs_no_prior(
    tmp_path: Path, export_dir: Path
) -> None:
    (_, problems), _ = _preflight(tmp_path, export_dir, env={})
    assert sum("must be 1 (offline mode)" in p for p in problems) == 2
    (_, problems), _ = _preflight(tmp_path / "two", export_dir, phase_name="reload", after=None)
    assert any("needs --after" in p for p in problems)
    (_, problems), _ = _preflight(tmp_path / "three", export_dir)
    assert not any("needs --after" in p for p in problems)


def test_preflight_refuses_an_output_directory_inside_the_repository(
    tmp_path: Path, export_dir: Path
) -> None:
    repo = tmp_path / "repo"  # _preflight builds its repository here
    (_, problems), _ = _preflight(tmp_path, export_dir, out=repo / "out" / "train")
    assert any("inside the repository" in p for p in problems)


def test_the_child_patch_points_tr02_at_this_config_and_preflight(monkeypatch) -> None:
    assert tr02.preflight is drv._tr02_preflight  # importing the driver patches nothing
    monkeypatch.setattr(tr02, "load_config", tr02.load_config)
    monkeypatch.setattr(tr02, "preflight", tr02.preflight)
    drv._patch_tr02_for_child()
    assert tr02.load_config()["experiment"] == "EXP-TR04"
    assert tr02.preflight is drv.preflight


def test_the_driver_loads_no_model_library_and_verify_config_runs_without_weights() -> None:
    code = (
        "import sys, runpy; sys.argv = ['tr04.py', 'verify-config'];"
        "\ntry:\n runpy.run_path('training/tr04/tr04.py', run_name='__main__')"
        "\nexcept SystemExit as e:\n assert e.code == 0, e.code"
        "\nassert 'mlx' not in sys.modules and 'mlx_lm' not in sys.modules"
        "\nassert 'transformers' not in sys.modules"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-500:]


# -------------------------------------------------------------------- thermal text and rules


def test_therm_text_joins_pmset_output_into_the_form_the_parser_accepts() -> None:
    joined = drv.therm_text(THERM_OK)
    assert joined.count(";") == 2
    assert drv.tr03_tools.parse_therm(joined) == []
    assert drv.tr03_tools.parse_therm(drv.therm_text("")) == ["empty_thermal_text"]
    assert drv.tr03_tools.parse_therm(drv.therm_text(THERM_OK + "CPU_Scheduler_Limit = 80\n"))


MON = dict(CFG["monitor"], reference_windows=2, slow_windows=3, floor_windows=2)
SMALL = dataclasses.replace(tr02.Limits.from_config(CFG), reports_per_window=2)
LORA_PARAMS = 458_752
A_SHA, C_SHA = hashlib.sha256(b"adapter").hexdigest(), hashlib.sha256(b"{}").hexdigest()


def _rules(microsteps: int = 80, mon=MON, max_wall_s: float = 3600.0, **kw):
    base = {
        "kind": "train",
        "phase": "train",
        "expect_commit": COMMIT,
        "microsteps": microsteps,
        "microsteps_per_report": 4,
        "optimizer_updates": microsteps // 4,
        "full_tokens": [500] * microsteps,
        "padded_tokens": [tr02.padded_tokens(500, 2048)] * microsteps,
        "valid_rows": 5,
        "lora_tensors": 32,
        "lora_params": LORA_PARAMS,
        "mon": mon,
        "max_wall_s": max_wall_s,
    }
    return drv.TR04Rules(SMALL, **(base | kw))


def _os(t: float = 1000.0, **kw) -> dict:
    base = {
        "kind": "os",
        "t": t,
        "rss_gb": 5.0,
        "swap_used_gb": 1.0,
        "pressure_level": 1,
        "on_battery": False,
        "therm": drv.therm_text(THERM_OK),
    }
    return base | kw


def _head() -> list[dict]:
    return [
        {"kind": "start", "phase": "train", "commit": COMMIT, "preflight": {}},
        {"kind": "load", "seconds": 2.0, "peak_memory_gb": 3.0},
        {
            "kind": "lora",
            "tensors": 32,
            "trainable_params": LORA_PARAMS,
            "shapes_ok": True,
            "problems": [],
        },
        {
            "kind": "val",
            "tag": "before_training",
            "updates_done": 0,
            "rows": 5,
            "val_loss": 1.5,
            "seconds": 1.0,
            "peak_memory_gb": 4.0,
        },
    ]


def _train(it: int, rate: float = 1.0) -> dict:
    return {
        "kind": "train",
        "iteration": it,
        "optimizer_step": it // 4,
        "train_loss": 1.0,
        "peak_memory_gb": 5.0,
        "iterations_per_second": rate,
        "loss_tokens_per_second": 1.0,
    }


def _windows(rules, rates: list[float]) -> str | None:
    events, it = _head(), 0
    for rate in rates:
        for _ in range(2):  # two reports (one per update) make a window in this test
            it += 4
            events.append(_train(it, rate=rate))
    for ev in events:
        if reason := rules.feed(ev):
            return reason
    return None


def test_a_healthy_sample_passes_and_the_tr02_rules_still_fire_first() -> None:
    assert _rules().prelaunch(_os()) is None
    r = _rules()
    r.prelaunch(_os())
    assert r.feed(_os(t=1010.0)) is None
    assert r.feed(_os(t=1020.0, on_battery=True)) == "battery_power"
    assert r.feed(_os(t=1030.0, pressure_level=4)) == "memory_pressure_critical"
    assert r.feed(_os(t=1040.0, rss_gb=11.5)).startswith("rss_above_limit")


def test_any_thermal_or_performance_statement_stops_the_run_and_unreadable_text_fails_closed() -> (
    None
):
    bad = drv.therm_text(THERM_OK + "CPU_Scheduler_Limit = 80\n")
    r = _rules()
    r.prelaunch(_os())
    assert r.feed(_os(therm=bad)).startswith("thermal_or_performance_warning")
    assert _rules().feed(_os(therm="")).startswith("thermal_or_performance_warning")
    assert (
        _rules()
        .feed({k: v for k, v in _os().items() if k != "therm"})
        .startswith("thermal_or_performance_warning")
    )
    assert _rules().prelaunch(_os(therm=bad)).startswith("thermal_or_performance_warning")


def test_the_wall_clock_cap_counts_from_the_prelaunch_baseline() -> None:
    r = _rules(max_wall_s=100.0)
    assert r.prelaunch(_os(t=1000.0)) is None
    assert r.feed(_os(t=1099.0)) is None
    assert r.feed(_os(t=1101.0)).startswith("max_wall_time:101s>100s")
    assert _rules().feed(_os(t="x")) == "malformed_timestamp"
    assert _rules().feed(_os(t=float("nan"))) == "malformed_timestamp"
    assert _rules(max_wall_s=0.0).feed(_os(t=9e9)) is None  # a zero cap means "not set" only here


def test_sustained_memory_pressure_warning_stops_only_after_training_started() -> None:
    r = _rules()
    for k in range(20):  # warning level while loading is tolerated
        assert r.feed(_os(t=1000.0 + k, pressure_level=2)) is None
    r = _rules()
    for ev in _head():
        assert r.feed(ev) is None
    got = [r.feed(_os(t=1000.0 + k, pressure_level=2)) for k in range(12)]
    assert got[:-1] == [None] * 11
    assert got[-1].startswith("sustained_memory_pressure_warning")
    r = _rules()
    for ev in _head():
        r.feed(ev)
    assert [r.feed(_os(pressure_level=2 if k % 5 else 1)) for k in range(30)] == [None] * 30


def test_sustained_slowdown_stops_after_the_configured_number_of_windows() -> None:
    assert _windows(_rules(), [1.0, 1.0, 0.6, 0.6]) is None
    assert _windows(_rules(), [1.0, 1.0, 0.6, 0.6, 0.6]).startswith(
        "sustained_throughput_loss:3_windows_below_0.7"
    )
    assert _windows(_rules(), [1.0, 1.0, 0.6, 0.6, 0.9, 0.6, 0.6]) is None  # a good window resets
    assert _windows(_rules(), [1.0, 1.0, 0.75, 0.75, 0.75, 0.75]) is None  # slower but above 0.7


def test_the_floor_is_relative_to_the_reference_median_not_to_the_first_window() -> None:
    mon = dict(MON, reference_windows=3)
    # the first window is slow: TR02's own rule (below half of window 1) cannot fire, ours can
    assert _windows(_rules(mon=mon), [0.45, 1.0, 1.0, 0.4]) is None
    got = _windows(_rules(mon=mon), [0.45, 1.0, 1.0, 0.4, 0.4])
    assert got.startswith("throughput_floor:2_windows_below_0.5")


def test_no_throughput_rule_applies_inside_the_reference_windows() -> None:
    assert _windows(_rules(), [1.0, 0.1]) is None
    assert _windows(_rules(), [0.1, 0.1]) is None


def test_a_complete_valid_sequence_still_finishes(tmp_path: Path) -> None:
    r = _rules(microsteps=8)
    events = [
        *_head(),
        _train(4),
        _train(8),
        {
            "kind": "train_total",
            "seconds": 10.0,
            "optimizer_steps": 2,
            "rows_exposed": 8,
            "schedule_complete": True,
        },
        {
            "kind": "adapter",
            "adapter_sha256": A_SHA,
            "config_sha256": C_SHA,
            "bytes": 7,
            "tensors": 32,
            "shapes_ok": True,
            "finite": True,
            "changed_tensors": 32,
            "max_abs_change": 0.01,
            "lora_b_abs_sum": 1.0,
        },
        {
            "kind": "val",
            "tag": "after_training",
            "updates_done": 2,
            "rows": 5,
            "val_loss": 1.2,
            "seconds": 1.0,
            "peak_memory_gb": 5.0,
        },
        {"kind": "done", "peak_memory_gb": 5.0},
    ]
    for ev in events:
        assert r.feed(ev) is None
    assert r.finished and r.val_after == 1.2
    (tmp_path / tr02.ADAPTER_FILE).write_bytes(b"adapter")
    (tmp_path / tr02.ADAPTER_CONFIG).write_bytes(b"{}")
    assert r.finish(tmp_path) is None


def test_a_reload_rule_set_uses_the_same_added_checks() -> None:
    r = drv.TR04Rules(
        SMALL,
        kind="reload",
        phase="reload",
        expect_commit=COMMIT,
        optimizer_updates=791,
        valid_rows=5,
        lora_tensors=32,
        lora_params=LORA_PARAMS,
        reference_loss=1.2,
        reference_adapter_sha256=A_SHA,
        reference_config_sha256=C_SHA,
        mon=MON,
        max_wall_s=900.0,
    )
    assert r.prelaunch(_os()) is None
    assert r.feed(_os(t=2000.0)).startswith("max_wall_time")


def test_build_rules_for_the_real_config_use_the_frozen_caps(
    tmp_path: Path, export_dir: Path
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "num_hidden_layers": 28,
            }
        ),
        "utf-8",
    )
    train = drv.build_rules(CFG, "train", export_dir, model, {}, COMMIT)
    n = len(tr04data.kept_ids()["train"])
    assert train.microsteps == n and train.optimizer_updates == n // 4
    assert train.max_wall_s == CFG["phases"][0]["max_wall_s"]
    assert (train.lora_tensors, train.lora_params) == (32, LORA_PARAMS)
    assert len(train.full_tokens) == len(train.padded_tokens) == n
    assert train.valid_rows == len(tr04data.kept_ids()["validation"])
    assert train.limits.reports_per_window == 25
    prior = {"reference_val_loss": 1.0, "adapter_sha256": A_SHA, "adapter_config_sha256": C_SHA}
    reload_ = drv.build_rules(CFG, "reload", export_dir, model, prior, COMMIT)
    assert reload_.kind == "reload" and reload_.max_wall_s == 900.0
    assert reload_.optimizer_updates == n // 4 and reload_.reference_loss == 1.0


def test_phases_read_only_the_train_and_valid_parts_of_the_export() -> None:
    assert [p.get("data") for p in CFG["phases"]] == ["tr04", None]
    assert set(CFG["inputs"]["export_files"]) == {
        "ROW_IDS.json",
        "tr04/train.jsonl",
        "tr04/valid.jsonl",
    }
    assert all("sealed" in " ".join(CFG["data"]["never_used"]).lower() for _ in [0])
