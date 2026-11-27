"""EXP-TR02 harness without weights: pins, schedule, event audit, supervisor, evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
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
LIM = tr02.Limits.from_config(CFG)
COMMIT = "a" * 40
ADAPTER_BYTES, ADAPTER_CONFIG_BYTES = b"adapter", b"{}"
A_SHA = hashlib.sha256(ADAPTER_BYTES).hexdigest()
C_SHA = hashlib.sha256(ADAPTER_CONFIG_BYTES).hexdigest()
LORA_PARAMS = 458_752  # 8 blocks x (q: 8*(2048+2048) + v: 8*(2048+1024))


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("export")
    export.export(out)
    return out


def _phase(name: str) -> dict:
    return next(p for p in CFG["phases"] if p["name"] == name)


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


def test_all_ten_accepted_checkpoint_files_are_pinned_from_the_exp009_record() -> None:
    record = ROOT / "docs/evidence/experiments/EXP-009-untuned-baselines"
    measured = json.loads(
        (record / "runs_attempt2_qwen3-1.7b/qwen3-1.7b/main_measurements.json").read_text("utf-8")
    )["model"]["files"]
    pinned = CFG["model"]["files"]
    assert len(pinned) == 10 and set(pinned) == set(measured)
    assert all(
        pinned[n] == {"bytes": m["bytes"], "sha256": m["sha256"]} for n, m in measured.items()
    )
    assert {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    } <= set(pinned)
    assert CFG["python"] == [3, 12] and CFG["packages"] == {"mlx": "0.32.4", "mlx-lm": "0.32.2"}


def test_a_fresh_export_matches_every_pinned_hash(export_dir: Path) -> None:
    assert tr02.verify_inputs(export_dir, CFG) == []
    assert tr02.verify_corpus(CFG) == []


def test_verify_inputs_catches_a_changed_missing_or_extra_file(
    export_dir: Path, tmp_path: Path
) -> None:
    copy_ = tmp_path / "e"
    shutil.copytree(export_dir, copy_)
    (copy_ / "smoke" / "train.jsonl").write_bytes(b"{}\n")
    (copy_ / "full" / "valid.jsonl").unlink()
    (copy_ / "smoke" / "test.jsonl").write_bytes(b"{}\n")
    problems = tr02.verify_inputs(copy_, CFG)
    assert "hash differs: smoke/train.jsonl" in problems
    assert "missing: full/valid.jsonl" in problems
    assert "unpinned file: smoke/test.jsonl" in problems
    assert "test file present: smoke/test.jsonl" in problems


def _repo_copy(tmp_path: Path) -> Path:
    """A throw-away git repository holding the corpus files the config pins."""
    repo = tmp_path / "repo"
    rel = Path("data/axbench-tr") / CFG["inputs"]["corpus_version"]
    (repo / rel).mkdir(parents=True)
    for name in (*CFG["inputs"]["corpus_files_sha256"], PER_EXAMPLE.name):
        shutil.copy2(ROOT / rel / name, repo / rel / name)
    _git(repo, "init", "-q")
    return repo


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _commit(repo: Path) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x", "--no-gpg-sign")
    return _git(repo, "rev-parse", "HEAD").strip()


def test_corpus_and_per_example_hashes_are_verified_at_execution_time(tmp_path: Path) -> None:
    repo = _repo_copy(tmp_path)
    assert tr02.verify_corpus(CFG, repo) == []
    rel = repo / "data/axbench-tr" / CFG["inputs"]["corpus_version"]
    (rel / "train.jsonl").write_bytes(b"{}\n")
    (rel / PER_EXAMPLE.name).unlink()
    problems = tr02.verify_corpus(CFG, repo)
    assert "corpus file hash differs: train.jsonl" in problems
    assert "per-example length file missing" in problems


def test_the_schedule_never_reads_an_unverified_per_example_file(
    export_dir: Path, tmp_path: Path
) -> None:
    tampered = tmp_path / "per_example.jsonl"
    tampered.write_bytes(PER_EXAMPLE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned hash"):
        tr02.row_lengths(export_dir, tampered, CFG["inputs"]["per_example_sha256"])
    ok = tr02.row_lengths(export_dir, PER_EXAMPLE, CFG["inputs"]["per_example_sha256"])
    assert len(ok["smoke/train.jsonl"]) == 400


# ----------------------------------------------------------- model inventory, packages, tree


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


def test_model_inventory_accepts_exactly_the_pinned_files(tmp_path: Path) -> None:
    model, cfg = _fake_model(tmp_path)
    assert tr02.verify_model_dir(model, cfg) == []
    (model / ".cache").mkdir()
    (model / ".cache" / "x").write_bytes(b"hub client cache")  # tolerated
    assert tr02.verify_model_dir(model, cfg) == []


def test_model_inventory_rejects_missing_extra_resized_and_changed_files(tmp_path: Path) -> None:
    model, cfg = _fake_model(tmp_path)
    (model / "tokenizer.json").unlink()
    (model / "model.safetensors").write_bytes(b"weightz")  # same size, other bytes
    (model / "config.json").write_bytes(b'{"hidden_size": 16}')  # other size
    (model / "adapter.safetensors").write_bytes(b"x")
    problems = tr02.verify_model_dir(model, cfg)
    assert "missing file: tokenizer.json" in problems
    assert "hash differs: model.safetensors" in problems
    assert "size differs: config.json" in problems
    assert "extra file: adapter.safetensors" in problems


def test_model_inventory_requires_provenance_and_published_weight_hashes(tmp_path: Path) -> None:
    model, cfg = _fake_model(tmp_path)
    prov = json.loads((model / "PROVENANCE.json").read_text("utf-8"))
    prov["files"]["model.safetensors"]["verified_against"] = "self"
    prov["revision"] = "other"
    (model / "PROVENANCE.json").write_text(json.dumps(prov), "utf-8")
    problems = tr02.verify_model_dir(model, cfg)
    assert "revision differs or weights not provisioned" in problems
    assert "weights not verified against a published hash: model.safetensors" in problems
    (model / "PROVENANCE.json").unlink()
    assert "missing file: PROVENANCE.json" in tr02.verify_model_dir(model, cfg)


def test_packages_python_and_offline_mode_are_checked() -> None:
    good = {"python": "3.12", "mlx": "0.32.4", "mlx-lm": "0.32.2"}
    assert tr02.verify_packages(CFG, good) == []
    assert len(tr02.verify_packages(CFG, good | {"mlx": "0.33.0", "python": "3.13"})) == 2
    assert tr02.verify_packages(CFG, {"python": "3.12", "mlx": "not installed"})
    offline = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert tr02.verify_offline(offline) == []
    assert len(tr02.verify_offline({"HF_HUB_OFFLINE": "1"})) == 1
    assert len(tr02.verify_offline({})) == 2


def test_clean_tree_and_exact_commit_are_required(tmp_path: Path) -> None:
    repo = _repo_copy(tmp_path)
    head = _commit(repo)
    assert tr02.verify_clean_tree(head, repo) == []
    assert any("owner-approved" in p for p in tr02.verify_clean_tree("b" * 40, repo))
    assert any("40-character" in p for p in tr02.verify_clean_tree("abc", repo))
    (repo / "scratch.txt").write_text("untracked", "utf-8")
    assert any("not clean" in p for p in tr02.verify_clean_tree(head, repo))
    (repo / "scratch.txt").unlink()
    (repo / "data/axbench-tr" / CFG["inputs"]["corpus_version"] / "train.jsonl").write_bytes(b"")
    assert any("not clean" in p for p in tr02.verify_clean_tree(head, repo))


def _preflight(tmp_path: Path, export_dir: Path, **kw):
    model, cfg = _fake_model(tmp_path)
    repo = _repo_copy(tmp_path)
    head = _commit(repo)
    args = {
        "config": cfg,
        "phase_name": "probe",
        "export_dir": export_dir,
        "model_dir": model,
        "out": tmp_path / "out" / "probe",
        "after": None,
        "expect_commit": head,
        "repo": repo,
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "versions": {"python": "3.12", "mlx": "0.32.4", "mlx-lm": "0.32.2"},
    }
    return tr02.preflight(**(args | kw)), args


def test_preflight_passes_when_everything_is_pinned_clean_and_offline(
    tmp_path: Path, export_dir: Path
) -> None:
    (record, problems), _ = _preflight(tmp_path, export_dir)
    assert problems == []
    assert record["phase"] == "probe" and record["offline"] is True


def test_preflight_names_each_failure(tmp_path: Path, export_dir: Path) -> None:
    (_, problems), args = _preflight(tmp_path, export_dir, env={}, versions={"python": "3.13"})
    text = "; ".join(problems)
    assert "HF_HUB_OFFLINE" in text and "python is 3.13" in text and "mlx is None" in text


def test_preflight_refuses_an_output_directory_inside_the_repository(
    tmp_path: Path, export_dir: Path
) -> None:
    repo = tmp_path / "repo"  # _preflight builds its repository here
    (_, problems), _ = _preflight(tmp_path, export_dir, out=repo / "out" / "probe")
    assert any("inside the repository" in p for p in problems)


def test_a_phase_cannot_start_after_a_failed_or_missing_previous_phase(
    tmp_path: Path, export_dir: Path
) -> None:
    failed = tmp_path / "failed_probe"
    side = tr02.side_dir(failed)
    side.mkdir(parents=True)
    (side / tr02.STOP).write_text("{}", "utf-8")  # a stopped probe: no COMPLETE.json
    (_, problems), _ = _preflight(tmp_path, export_dir, phase_name="smoke", after=failed)
    assert any("no COMPLETE.json" in p for p in problems)
    (_, problems), _ = _preflight(tmp_path / "two", export_dir, phase_name="smoke", after=None)
    assert any("needs --after" in p for p in problems)
    (_, problems), _ = _preflight(tmp_path / "three", export_dir, phase_name="reload", after=None)
    assert any("needs --after" in p for p in problems)


# ------------------------------------------------------------------------------------ schedule


def test_smoke_exposes_each_of_400_rows_exactly_once_in_100_updates(export_dir: Path) -> None:
    plan = tr02.schedule(export_dir, CFG, _phase("smoke"))
    assert plan["rows"] == plan["microsteps"] == 400
    assert plan["optimizer_updates"] == 100
    assert plan["each_row_exposed_exactly_once"]
    assert len({s["id"] for s in plan["steps"]}) == 400
    assert plan["tokens_total"] == sum(s["tokens"] for s in plan["steps"])
    assert tr02.schedule(export_dir, CFG, _phase("smoke")) == plan  # deterministic


def test_probe_runs_the_20_longest_rows_and_shares_none_with_smoke(export_dir: Path) -> None:
    probe = tr02.schedule(export_dir, CFG, _phase("probe"))
    assert probe["optimizer_updates"] == 5 and min(s["tokens"] for s in probe["steps"]) >= 1199
    ids = json.loads((export_dir / "ROW_IDS.json").read_text("utf-8"))
    assert not set(ids["peak_probe/train.jsonl"]) & set(ids["smoke/train.jsonl"])
    assert not set(ids["smoke/valid.jsonl"]) & set(ids["smoke/train.jsonl"])


def test_full_and_padded_token_counts_are_both_scheduled(export_dir: Path) -> None:
    assert tr02.padded_tokens(1343, 2048) == 1344  # 1 + 32*42 = 1345 wide, the model sees 1344
    assert tr02.padded_tokens(26, 2048) == 32
    assert tr02.padded_tokens(2040, 2048) == 2047  # capped at max_seq_length
    plan = tr02.schedule(export_dir, CFG, _phase("probe"))
    assert plan["padded_tokens_total"] > plan["tokens_total"]
    assert all(s["padded_tokens"] >= s["tokens"] - 1 for s in plan["steps"])
    assert plan["padded_tokens_total"] == sum(s["padded_tokens"] for s in plan["steps"])


def test_a_schedule_that_would_repeat_or_skip_rows_is_refused(export_dir: Path) -> None:
    with pytest.raises(ValueError):
        tr02.schedule(export_dir, CFG, {**_phase("smoke"), "microsteps": 200})
    with pytest.raises(ValueError, match="disagree"):
        tr02.schedule(export_dir, CFG, {**_phase("smoke"), "optimizer_updates": 99})


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
        if p.get("kind") == "reload":
            continue
        assert p["microsteps"] == p["optimizer_updates"] * p["grad_accumulation"]
        assert p["microsteps_per_report"] == p["grad_accumulation"]  # one report per update
    assert CFG["hyperparameters"]["max_seq_length"] == 2048
    assert CFG["hyperparameters"]["micro_batch"] == 1
    chain = {p["name"]: p.get("requires") for p in CFG["phases"]}
    assert chain == {"probe": None, "smoke": "probe", "reload": "smoke"}


# -------------------------------------------------------------------- LoRA and adapter checks

MODEL_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 28,
}


def test_config_json_is_byte_identical_to_the_revision_4_pins() -> None:
    # revision 5 changes the name contract and the checks around it, never a pin or a limit.
    # Line endings are normalized: a Windows checkout may convert LF to CRLF in this text file.
    data = tr02.CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert (
        hashlib.sha256(data).hexdigest()
        == "7d31d916621523281d2226de9802671ddc63b05aad219388064a275abaf5301c"
    )


def test_expected_lora_shapes_follow_the_checkpoint_config() -> None:
    shapes = tr02.expected_lora_shapes(MODEL_CONFIG, CFG["hyperparameters"])
    assert len(shapes) == 32 and tr02.trainable_params(shapes) == LORA_PARAMS == 458_752
    # mlx-lm 0.32.2 qwen3.Model nests the blocks under `model`: the contract names carry the prefix
    assert shapes["model.layers.20.self_attn.q_proj.lora_a"] == (2048, 8)
    assert shapes["model.layers.20.self_attn.q_proj.lora_b"] == (8, 2048)
    assert shapes["model.layers.27.self_attn.v_proj.lora_b"] == (8, 1024)
    assert not any(f"model.layers.{n}." in k for n in range(20) for k in shapes)
    assert all(k.startswith("model.layers.") for k in shapes)  # no unprefixed name is accepted


def test_expected_lora_shapes_refuse_a_model_type_they_were_not_derived_for() -> None:
    other = {**MODEL_CONFIG, "model_type": "llama"}
    with pytest.raises(ValueError, match="qwen3"):
        tr02.expected_lora_shapes(other, CFG["hyperparameters"])
    with pytest.raises(ValueError):
        tr02.expected_lora_shapes(
            {k: v for k, v in MODEL_CONFIG.items() if k != "model_type"}, CFG["hyperparameters"]
        )


def test_lora_problems_name_missing_extra_and_misshapen_tensors() -> None:
    want = tr02.expected_lora_shapes(MODEL_CONFIG, CFG["hyperparameters"])
    assert tr02.lora_problems(dict(want), want) == []
    got = dict(want)
    del got["model.layers.20.self_attn.q_proj.lora_a"]
    got["model.layers.19.self_attn.q_proj.lora_a"] = (2048, 8)
    got["model.layers.27.self_attn.v_proj.lora_b"] = (8, 2048)
    text = "; ".join(tr02.lora_problems(got, want))
    assert "missing tensor: model.layers.20.self_attn.q_proj.lora_a" in text
    assert "unexpected tensor: model.layers.19" in text
    assert "model.layers.27.self_attn.v_proj.lora_b is (8, 2048)" in text


def test_the_unprefixed_namespace_that_stopped_the_first_probe_is_rejected() -> None:
    want = tr02.expected_lora_shapes(MODEL_CONFIG, CFG["hyperparameters"])
    bare = {k.removeprefix("model."): v for k, v in want.items()}
    problems = tr02.lora_problems(bare, want)
    assert len([p for p in problems if p.startswith("missing tensor")]) == 32
    assert len([p for p in problems if p.startswith("unexpected tensor")]) == 32
    mixed = dict(want)
    name = "model.layers.20.self_attn.q_proj.lora_a"
    mixed[name.removeprefix("model.")] = mixed.pop(name)
    assert tr02.lora_problems(mixed, want) == [
        f"missing tensor: {name}",
        "unexpected tensor: layers.20.self_attn.q_proj.lora_a",
    ]


def test_lora_report_is_bounded_and_holds_names_and_shapes_only() -> None:
    want = tr02.expected_lora_shapes(MODEL_CONFIG, CFG["hyperparameters"])
    bare = {k.removeprefix("model."): v for k, v in want.items()}
    report = tr02.lora_report(bare, want)
    assert report["observed_count"] == report["expected_count"] == 32
    assert report["missing_count"] == report["unexpected_count"] == 32
    assert report["shape_mismatch_count"] == 0
    assert len(report["observed_sample"]) == len(report["expected_sample"]) == 6
    assert all(k.startswith("layers.") for k in report["observed_sample"])
    assert all(k.startswith("model.layers.") for k in report["expected_sample"])
    assert all(
        isinstance(v, list) and all(isinstance(i, int) for i in v)
        for v in report["observed_sample"].values()
    )
    assert json.dumps(report)  # plain JSON, no tensor contents, no paths


def test_adapter_config_the_harness_writes_is_the_one_mlx_lm_reloads() -> None:
    cfg = tr02.lora_adapter_config(CFG["hyperparameters"])
    assert cfg == {
        "fine_tune_type": "lora",
        "num_layers": 8,
        "lora_parameters": {
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        },
    }
    # the bytes are unchanged from protocol v4 (same keys, same order, default separators)
    assert json.dumps(cfg) == (
        '{"fine_tune_type": "lora", "num_layers": 8, "lora_parameters": {"rank": 8, '
        '"scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}}'
    )


def test_live_adapter_problems_names_missing_extra_shape_nonfinite_and_value_differences() -> None:
    np = pytest.importorskip("numpy")
    a = {"x": np.ones((2, 3), np.float32), "y": np.zeros((3, 2), np.float32)}
    assert tr02.live_adapter_problems(a, {k: v.copy() for k, v in a.items()}) == []
    assert "not in the reloaded model: y" in tr02.live_adapter_problems(a, {"x": a["x"]})[0]
    assert "not in the saved file: z" in tr02.live_adapter_problems(a, {**a, "z": a["x"]})[0]
    assert (
        "reloaded shape"
        in tr02.live_adapter_problems(a, {**a, "y": np.zeros((2, 3), np.float32)})[0]
    )
    bad = a["x"].copy()
    bad[0, 0] = np.nan
    assert "not finite" in tr02.live_adapter_problems(a, {**a, "x": bad})[0]
    diff = a["x"].copy()
    diff[0, 0] = 2.0
    assert "values differ" in tr02.live_adapter_problems(a, {**a, "x": diff})[0]


def test_adapter_summary_detects_no_change_zero_b_and_non_finite() -> None:
    np = pytest.importorskip("numpy")
    init = {"a.lora_a": np.ones((2, 2), "f4"), "a.lora_b": np.zeros((2, 2), "f4")}
    same = tr02.adapter_summary(init, {k: v.copy() for k, v in init.items()})
    assert (
        same["changed_tensors"] == 0 and same["max_abs_change"] == 0 and same["lora_b_abs_sum"] == 0
    )
    trained = {"a.lora_a": init["a.lora_a"] * 1.01, "a.lora_b": np.full((2, 2), 0.5, "f4")}
    out = tr02.adapter_summary(init, trained)
    assert out["finite"] and out["changed_tensors"] == 2 and out["lora_b_abs_sum"] == 2.0
    bad = {**trained, "a.lora_a": np.array([[np.nan, 1.0], [1.0, 1.0]], "f4")}
    assert tr02.adapter_summary(init, bad)["finite"] is False


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


# ------------------------------------------------------------------------- event audit helpers


def _rules(microsteps: int = 8, tokens: list[int] | None = None, **kw):
    tokens = tokens or [500] * microsteps
    base = {
        "kind": "train",
        "phase": "smoke",
        "expect_commit": COMMIT,
        "microsteps": microsteps,
        "microsteps_per_report": 4,
        "optimizer_updates": microsteps // 4,
        "full_tokens": tokens,
        "padded_tokens": [tr02.padded_tokens(t, 2048) for t in tokens],
        "valid_rows": 5,
        "lora_tensors": 32,
        "lora_params": LORA_PARAMS,
    }
    return tr02.StopRules(LIM, **(base | kw))


def _start(phase: str = "smoke") -> dict:
    return {"kind": "start", "phase": phase, "commit": COMMIT, "preflight": {}}


def _load_ev() -> dict:
    return {"kind": "load", "seconds": 2.0, "peak_memory_gb": 3.0}


def _lora_ev() -> dict:
    want = tr02.expected_lora_shapes(MODEL_CONFIG, CFG["hyperparameters"])
    return {
        "kind": "lora",
        "tensors": 32,
        "trainable_params": LORA_PARAMS,
        "shapes_ok": True,
        "problems": [],
        "report": tr02.lora_report(want, want),  # the bounded record the driver now logs
    }


def _val(tag: str, loss: float = 1.5, updates: int = 0, **kw) -> dict:
    ev = {
        "kind": "val",
        "tag": tag,
        "updates_done": updates,
        "rows": 5,
        "val_loss": loss,
        "seconds": 1.0,
        "peak_memory_gb": 4.0,
    }
    return ev | kw


def _train(it: int, rate: float = 1.0, loss: float = 1.0, peak: float = 5.0, **kw) -> dict:
    ev = {
        "kind": "train",
        "iteration": it,
        "optimizer_step": it // 4,
        "train_loss": loss,
        "peak_memory_gb": peak,
        "iterations_per_second": rate,
        "loss_tokens_per_second": 99999.0,  # never used: loss-token throughput is not the rate
    }
    return ev | kw


def _total(updates: int, microsteps: int) -> dict:
    return {
        "kind": "train_total",
        "seconds": 10.0,
        "optimizer_steps": updates,
        "rows_exposed": microsteps,
        "schedule_complete": True,
    }


def _adapter(**kw) -> dict:
    ev = {
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
    }
    return ev | kw


def _events(microsteps: int = 8, rate: float = 1.0) -> list[dict]:
    updates = microsteps // 4
    return [
        _start(),
        _load_ev(),
        _lora_ev(),
        _val("before_training"),
        *[_train(4 * k, rate=rate) for k in range(1, updates + 1)],
        _total(updates, microsteps),
        _adapter(),
        _val("after_training", loss=1.2, updates=updates, peak_memory_gb=5.0),
        {"kind": "done", "peak_memory_gb": 5.0},
    ]


def _feed_all(rules, events: list[dict]) -> str | None:
    for ev in events:
        if reason := rules.feed(ev):
            return reason
    return None


def _os(**kw: object) -> dict:
    base = {
        "kind": "os",
        "rss_gb": 5.0,
        "swap_used_gb": 1.0,
        "pressure_level": 1,
        "on_battery": False,
    }
    return base | kw


# ---------------------------------------------------------------------- event audit: each rule


def test_a_complete_valid_event_sequence_finishes_with_no_stop(tmp_path: Path) -> None:
    r = _rules()
    assert _feed_all(r, _events()) is None
    assert r.finished and r.val_after == 1.2
    (tmp_path / tr02.ADAPTER_FILE).write_bytes(ADAPTER_BYTES)
    (tmp_path / tr02.ADAPTER_CONFIG).write_bytes(ADAPTER_CONFIG_BYTES)
    assert r.finish(tmp_path) is None


def test_validation_and_done_events_are_audited_not_ignored() -> None:
    nan = float("nan")
    assert _feed_all(_rules(), [*_events()[:3], _val("before_training", loss=nan)]) == (
        "non_finite_loss"
    )
    assert _feed_all(_rules(), [*_events()[:3], _val("before_training", loss=-1.0)]) == (
        "out_of_range_val_loss"
    )
    assert _feed_all(_rules(), [*_events()[:3], _val("before_training", rows=4)]) == (
        "validation_did_not_cover_every_row"
    )
    assert _feed_all(_rules(), [*_events()[:3], _val("before_training", updates=1)]) == (
        "val_before_training_updates_not_zero"
    )
    assert _feed_all(_rules(), [*_events()[:3], _val("before_training", seconds=0.0)]) == (
        "out_of_range_seconds"
    )
    ev = _events()
    ev[-2] = _val("after_training", loss=nan, updates=2, peak_memory_gb=5.0)
    assert _feed_all(_rules(), ev) == "non_finite_loss"
    ev = _events()
    ev[-2] = _val("after_training", updates=1, peak_memory_gb=5.0)
    assert _feed_all(_rules(), ev) == "val_after_training_updates_differ"


def test_the_final_mlx_peak_load_and_validation_peaks_are_checked() -> None:
    ev = _events()
    ev[-1] = {"kind": "done", "peak_memory_gb": 10.4}
    assert _feed_all(_rules(), ev).startswith("mlx_peak_above_limit")
    ev = _events()
    ev[1] = {**_load_ev(), "peak_memory_gb": 12.0}
    assert _feed_all(_rules(), ev).startswith("mlx_peak_above_limit")
    ev = _events()
    ev[-2] = _val("after_training", updates=2, peak_memory_gb=10.2)
    assert _feed_all(_rules(), ev).startswith("mlx_peak_above_limit")
    ev = _events()
    ev[-1] = {"kind": "done", "peak_memory_gb": 4.0}  # the peak counter cannot go down
    assert _feed_all(_rules(), ev) == "mlx_peak_decreased"


def test_events_must_arrive_in_the_exact_order_and_nothing_follows_done() -> None:
    ev = _events()
    assert _feed_all(_rules(), [ev[0], ev[2]]).startswith("unexpected_event:lora_expected_load")
    skipped = [e for e in ev if e.get("tag") != "before_training"]
    assert _feed_all(_rules(), skipped).startswith("unexpected_event:train")
    swapped = ev[:-3] + [ev[-2], ev[-3], ev[-1]]
    assert _feed_all(_rules(), swapped).startswith("unexpected_event:val:after_training")
    r = _rules()
    assert _feed_all(r, ev) is None
    assert r.feed(_train(8)).startswith("event_after_done")
    assert _rules().feed(_start("probe")) == "start_phase_mismatch"
    assert _rules().feed({**_start(), "commit": "b" * 40}) == "start_commit_mismatch"


def test_an_incomplete_sequence_is_not_a_success(tmp_path: Path) -> None:
    r = _rules()
    assert _feed_all(r, _events()[:-1]) is None  # no stop reason mid-run ...
    assert not r.finished
    assert r.finish(tmp_path).startswith("incomplete_sequence:")  # ... but it never finished


def test_train_events_check_indices_rates_steps_and_totals() -> None:
    head = _events()[:4]
    assert _feed_all(_rules(), [*head, _train(8)]) == "iteration_index_wrong:8"
    assert _feed_all(_rules(), [*head, _train(4, rate=0.0)]) == "out_of_range_iterations_per_second"
    assert _feed_all(_rules(), [*head, _train(4, rate=float("nan"))]) == (
        "non_finite_iterations_per_second"
    )
    assert _feed_all(_rules(), [*head, _train(4, rate=float("inf"))]) == (
        "non_finite_iterations_per_second"
    )
    assert _feed_all(_rules(), [*head, _train(4, optimizer_step=2)]) == (
        "optimizer_step_not_incrementing:2"
    )
    assert _feed_all(_rules(), [*head, _train(4), _train(8, optimizer_step=1)]) == (
        "optimizer_step_not_incrementing:1"
    )
    assert _feed_all(_rules(), [*head, _train(4, loss=float("nan"))]) == "non_finite_loss"
    assert _feed_all(_rules(), [*head, _train(4, peak=10.5)]).startswith("mlx_peak_above_limit")
    body = [*head, _train(4), _train(8)]
    assert _feed_all(_rules(), [*body, _total(1, 8)]).startswith("optimizer_steps_1_not_2")
    assert _feed_all(_rules(), [*body, _total(2, 7)]) == "rows_exposed_differ_from_schedule"
    assert _feed_all(_rules(), [*body, {**_total(2, 8), "schedule_complete": False}]) == (
        "rows_exposed_differ_from_schedule"
    )


def test_adapter_event_must_prove_a_real_nonzero_change() -> None:
    head = [*_events()[:4], _train(4), _train(8), _total(2, 8)]
    ok = _rules()
    assert _feed_all(ok, [*head, _adapter()]) is None
    assert _feed_all(_rules(), [*head, _adapter(changed_tensors=0)]) == "adapter_did_not_change"
    assert _feed_all(_rules(), [*head, _adapter(max_abs_change=0.0)]) == "adapter_did_not_change"
    assert _feed_all(_rules(), [*head, _adapter(lora_b_abs_sum=0.0)]) == "lora_b_still_zero"
    assert _feed_all(_rules(), [*head, _adapter(finite=False)]) == "adapter_not_finite"
    assert _feed_all(_rules(), [*head, _adapter(tensors=16)]) == "adapter_tensors_or_shapes_differ"
    assert _feed_all(_rules(), [*head, _adapter(shapes_ok=False)]) == (
        "adapter_tensors_or_shapes_differ"
    )
    assert _feed_all(_rules(), [*head, _adapter(adapter_sha256="xyz")]) == (
        "malformed_adapter_sha256"
    )
    lora = [*_events()[:2], {**_lora_ev(), "trainable_params": 1}]
    assert _feed_all(_rules(), lora) == "lora_tensor_count_or_parameters_differ"
    assert _feed_all(_rules(), [*_events()[:2], {**_lora_ev(), "shapes_ok": False}]) == (
        "lora_shapes_not_ok"
    )


def test_the_saved_adapter_files_must_be_the_ones_the_driver_logged(tmp_path: Path) -> None:
    r = _rules()
    assert _feed_all(r, _events()) is None
    assert r.finish(tmp_path).startswith("saved_adapters.safetensors_differs")  # not saved at all
    (tmp_path / tr02.ADAPTER_FILE).write_bytes(b"other bytes")
    (tmp_path / tr02.ADAPTER_CONFIG).write_bytes(ADAPTER_CONFIG_BYTES)
    assert r.finish(tmp_path).startswith("saved_adapters.safetensors_differs")
    (tmp_path / tr02.ADAPTER_FILE).write_bytes(ADAPTER_BYTES)
    assert r.finish(tmp_path) is None


def test_malformed_events_are_a_stop_not_a_crash() -> None:
    head = _events()[:4]
    assert _rules().feed({"nokind": 1}).startswith("malformed_event:KeyError")
    assert _feed_all(_rules(), [*head, {"kind": "train", "iteration": 4}]).startswith(
        "malformed_event:KeyError"
    )
    assert _feed_all(_rules(), [*head, _train(4, loss="1.0")]) == "malformed_train_loss"
    assert _feed_all(_rules(), [*head, _train(4, loss=True)]) == "malformed_train_loss"
    assert _rules().feed({"kind": "mystery"}).startswith("unexpected_event:mystery")
    assert _rules().feed({"kind": "error", "message": "MemoryError"}).startswith("error:")


# ------------------------------------------------------------------------- OS samples and windows


def test_each_os_rule_fires_and_a_healthy_sample_does_not() -> None:
    assert _rules().feed(_os()) is None
    assert _rules().feed(_os(rss_gb=11.5)).startswith("rss_above_limit")
    assert _rules().feed(_os(pressure_level=4)) == "memory_pressure_critical"
    assert _rules().feed(_os(pressure_level=2)) is None  # warning is tolerated, critical is not
    assert _rules().feed(_os(on_battery=True)) == "battery_power"
    r = _rules()
    assert r.feed(_os(swap_used_gb=1.0)) is None  # baseline
    assert r.feed(_os(swap_used_gb=2.9)) is None
    assert r.feed(_os(swap_used_gb=3.1)) == "swap_growth_above_limit"


def test_mlx_peak_and_rss_are_separate_rules_with_separate_limits() -> None:
    assert LIM.mlx_peak_gb == 10.0 and LIM.rss_gb == 11.0
    r = _rules()
    assert r.feed(_os(rss_gb=10.9)) is None  # high RSS alone is not an MLX-peak breach
    assert r.feed(_os(rss_gb=11.1)) is not None
    assert _feed_all(_rules(), [*_events()[:4], _train(4, peak=9.9)]) is None


def test_the_prelaunch_baseline_needs_ac_power_and_normal_pressure() -> None:
    r = _rules()
    assert r.prelaunch(_os(rss_gb=None, swap_used_gb=0.7)) is None
    assert r.swap_baseline == 0.7  # the baseline comes from before the child exists
    assert _rules().prelaunch(_os(on_battery=True)) == "battery_power_before_launch"
    assert _rules().prelaunch(_os(pressure_level=2)) == (
        "memory_pressure_not_normal_before_launch:2"
    )


def _windows(rules, rates: list[float]) -> str | None:
    events = [_start(), _load_ev(), _lora_ev(), _val("before_training")]
    it = 0
    for rate in rates:
        for _ in range(10):  # ten reports (one per update) make a window
            it += 4
            events.append(_train(it, rate=rate))
    return _feed_all(rules, events)


def _w(**kw):
    return _rules(microsteps=400, **kw)


def test_throughput_rule_uses_full_tokens_over_windows_and_needs_three_in_a_row() -> None:
    # a 500-token row at 1.0 it/s per microstep step: first window sets the reference
    assert _windows(_w(), [1.0, 0.4, 0.4]) is None  # only two bad windows
    assert _windows(_w(), [1.0, 0.4, 0.4, 0.4]) == (
        "throughput_below_half_of_first_window_for_3_windows"
    )
    assert _windows(_w(), [1.0, 0.4, 0.4, 0.9, 0.4, 0.4]) is None  # a good window resets
    assert _windows(_w(), [1.0, 0.6, 0.6, 0.6, 0.6]) is None  # slower but above half


def test_window_rates_are_full_and_padded_tokens_per_compute_second() -> None:
    tokens = [100, 300] * 200
    r = _w(tokens=tokens)
    _windows(r, [2.0])
    # 40 microsteps: 20 x 100 + 20 x 300 = 8,000 full tokens in 40 / 2.0 = 20 s of compute
    assert r.window_rates[0] == pytest.approx(400.0)
    # padded: rows pad to 128 and 320 wide, the model computes on one token fewer
    assert r.window_rates_padded[0] == pytest.approx((20 * 128 + 20 * 320) / 20.0)


# ------------------------------------------------------------------------------------ reload


def _reload_rules(**kw):
    base = {
        "kind": "reload",
        "phase": "reload",
        "expect_commit": COMMIT,
        "optimizer_updates": 100,
        "valid_rows": 5,
        "lora_tensors": 32,
        "lora_params": LORA_PARAMS,
        "reference_loss": 1.2,
        "reference_adapter_sha256": A_SHA,
        "reference_config_sha256": C_SHA,
    }
    return tr02.StopRules(LIM, **(base | kw))


def _reload_events(loss: float = 1.2, **kw) -> list[dict]:
    check = {
        "kind": "adapter_check",
        "adapter_sha256": A_SHA,
        "config_sha256": C_SHA,
        "tensors": 32,
        "shapes_ok": True,
        "finite": True,
    }
    return [
        _start("reload"),
        _load_ev(),
        check | kw.get("check", {}),
        _val("reload", loss=loss, updates=100),
        {"kind": "reload", "reference_loss": 1.2, "reload_loss": loss, "abs_diff": abs(loss - 1.2)}
        | kw.get("reload", {}),
        {"kind": "done", "peak_memory_gb": 5.0},
    ]


def test_reload_passes_inside_the_tolerance_and_fails_outside_it() -> None:
    r = _reload_rules()
    assert _feed_all(r, _reload_events()) is None and r.finished
    assert _feed_all(_reload_rules(), _reload_events(loss=1.2 + 5e-5)) is None
    assert _feed_all(_reload_rules(), _reload_events(loss=1.2 + 2e-4)).startswith(
        "reload_loss_mismatch"
    )
    assert _feed_all(_reload_rules(), _reload_events(loss=float("nan"))) == "non_finite_loss"


def test_reload_checks_adapter_hashes_the_reference_and_event_consistency() -> None:
    wrong = "0" * 64
    assert _feed_all(_reload_rules(), _reload_events(check={"adapter_sha256": wrong})) == (
        "reload_adapter_hash_differs"
    )
    assert _feed_all(_reload_rules(), _reload_events(check={"config_sha256": wrong})) == (
        "reload_adapter_config_hash_differs"
    )
    assert _feed_all(_reload_rules(), _reload_events(check={"finite": False})) == (
        "adapter_not_finite"
    )
    assert _feed_all(_reload_rules(), _reload_events(reload={"reference_loss": 1.0})) == (
        "reload_reference_loss_differs"
    )
    assert _feed_all(_reload_rules(), _reload_events(reload={"reload_loss": 1.2001})) == (
        "reload_loss_differs_from_validation_event"
    )
    assert _feed_all(_reload_rules(reference_loss=None), _reload_events()) == (
        "reload_reference_loss_differs"
    )


# ------------------------------------------------------------------------------- supervisor

CHILD = """
import hashlib, json, os, sys, time
out, events = sys.argv[1], json.loads(sys.argv[2])
sleep, code = float(sys.argv[3]), int(sys.argv[4])
os.makedirs(out)
open(os.path.join(os.path.dirname(out), "launched"), "w").write("yes")
if sys.argv[5:] == ["files"]:
    open(os.path.join(out, "adapters.safetensors"), "wb").write(b"adapter")
    open(os.path.join(out, "adapter_config.json"), "wb").write(b"{}")
with open(os.path.join(out, "events.jsonl"), "a", encoding="utf-8") as fh:
    for ev in events:
        fh.write((ev if isinstance(ev, str) else json.dumps(ev)) + "\\n"); fh.flush()
    time.sleep(sleep)
sys.exit(code)
"""


def _cmd(out: Path, events: list, sleep: float = 0.1, code: int = 0, files: bool = True) -> list:
    cmd = [sys.executable, "-c", CHILD, str(out), json.dumps(events), str(sleep), str(code)]
    return cmd + (["files"] if files else [])


def _good(sample: dict | None = None):
    sample = sample or _os()
    return lambda pid: {**sample, "t": 0.0, "rss_gb": None if pid is None else 5.0}


def _run(tmp_path: Path, events: list, sampler=None, rules=None, **kw):
    out = tmp_path / "phase"
    kw.setdefault("sleep", 0.1)
    code = kw.pop("code", 0)
    files = kw.pop("files", True)
    result = tr02.supervise(
        _cmd(out, events, sleep=kw.pop("sleep"), code=code, files=files),
        out,
        rules or _rules(),
        sampler or _good(),
        poll_s=0.05,
        kill_after_s=5,
        **kw,
    )
    return result, out


def test_supervisor_lets_a_complete_valid_child_finish_and_writes_complete_json(
    tmp_path: Path,
) -> None:
    result, out = _run(tmp_path, _events())
    assert result["stopped"] is False and result["child_exit_code"] == 0 and result["complete"]
    side = tr02.side_dir(out)
    done = json.loads((side / tr02.COMPLETE).read_text("utf-8"))
    assert done["phase"] == "smoke" and done["reference_val_loss"] == 1.2
    assert done["adapter_sha256"] == A_SHA and done["adapter_config_sha256"] == C_SHA
    assert "out/events.jsonl" in done["files"] and "supervisor/child.log" in done["files"]
    assert not (side / tr02.STOP).exists()
    rec, problems = tr02.require_complete(out, "smoke")
    assert problems == [] and rec["phase"] == "smoke"


def test_complete_json_gates_the_next_phase(tmp_path: Path) -> None:
    _, out = _run(tmp_path, _events())
    assert tr02.require_complete(out, "probe")[1] == ["previous phase is smoke, need probe"]
    (out / tr02.ADAPTER_FILE).write_bytes(b"tampered after the run")
    assert "files changed since" in tr02.require_complete(out, "smoke")[1][0]
    (out / tr02.ADAPTER_FILE).write_bytes(ADAPTER_BYTES)
    assert tr02.require_complete(out, "smoke")[1] == []
    (out / "extra.txt").write_text("new file", "utf-8")
    assert "files changed since" in tr02.require_complete(out, "smoke")[1][0]
    assert "no COMPLETE.json" in tr02.require_complete(tmp_path / "nowhere", "smoke")[1][0]


def test_a_stopped_phase_leaves_stop_json_and_no_complete_json(tmp_path: Path) -> None:
    ev = _events()
    ev[-2] = _val("after_training", loss=float("nan"), updates=2)
    result, out = _run(tmp_path, ev)
    assert result["stopped"] and result["reason"] == "non_finite_loss"
    side = tr02.side_dir(out)
    assert json.loads((side / tr02.STOP).read_text("utf-8"))["reason"] == "non_finite_loss"
    assert not (side / tr02.COMPLETE).exists()
    assert (out / tr02.EVENTS).exists() and (side / "supervisor.jsonl").exists()  # partials kept
    assert "no COMPLETE.json" in tr02.require_complete(out, "smoke")[1][0]


def test_supervisor_stops_a_child_that_ends_early_or_exits_non_zero(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, _events()[:5])  # events stop part-way, exit code 0
    assert result["reason"].startswith("incomplete_sequence:")
    other = tmp_path / "b"
    other.mkdir()
    result, _ = _run(other, _events(), code=3)
    assert result["reason"] == "child_exit_code_3"
    third = tmp_path / "c"
    third.mkdir()
    result, _ = _run(third, _events(), files=False)  # the logged adapter was never saved
    assert result["reason"].startswith("saved_adapters.safetensors_differs")


def test_supervisor_terminates_the_child_on_a_rule_and_does_not_retry(tmp_path: Path) -> None:
    ev = [*_events()[:4], _train(4, peak=11.0)]
    started = time.monotonic()
    result, out = _run(tmp_path, ev, sleep=30.0)
    assert result["stopped"] and result["reason"].startswith("mlx_peak_above_limit")
    assert result["child_exit_code"] != 0 and time.monotonic() - started < 20
    assert (tr02.side_dir(out) / tr02.STOP).exists()


def test_supervisor_handles_malformed_event_lines_and_events_that_miss_keys(
    tmp_path: Path,
) -> None:
    result, _ = _run(tmp_path, [*_events()[:2], "this is not json"], sleep=30.0)
    assert result["reason"] == "malformed_event:not_json"
    other = tmp_path / "b"
    other.mkdir()
    result, _ = _run(other, [*_events()[:2], '["a list"]'], sleep=30.0)
    assert result["reason"] == "malformed_event:not_an_object"
    third = tmp_path / "c"
    third.mkdir()
    result, _ = _run(third, [_start(), {"kind": "load"}], sleep=30.0)
    assert result["reason"].startswith("malformed_event:KeyError")


def test_supervisor_stops_on_an_os_sample_during_the_run(tmp_path: Path) -> None:
    calls = {"n": 0}

    def sampler(pid):
        calls["n"] += 1
        return _os(on_battery=pid is not None and calls["n"] > 2, rss_gb=None) | {"t": 0.0}

    result, _ = _run(tmp_path, _events(), sampler=sampler, sleep=30.0)
    assert result["reason"] == "battery_power"
    assert result["child_exit_code"] != 0  # the child was still running and was terminated


def test_the_baseline_is_taken_before_the_child_starts_and_refuses_launch(tmp_path: Path) -> None:
    seen: list = []

    def sampler(pid):
        seen.append((pid, (tmp_path / "launched").exists()))
        return {**_os(), "t": 0.0}

    result, _ = _run(tmp_path, _events(), sampler=sampler)
    assert seen[0] == (None, False)  # baseline: no pid, child not yet launched
    assert result["stopped"] is False
    for name, sample, reason in (
        ("battery", _os(on_battery=True), "battery_power_before_launch"),
        ("pressure", _os(pressure_level=2), "memory_pressure_not_normal_before_launch:2"),
    ):
        root = tmp_path / name
        root.mkdir()
        result, out = _run(root, _events(), sampler=_good(sample))
        assert result["reason"] == reason and result["child_exit_code"] is None
        assert not (root / "launched").exists() and not out.exists()  # never launched
        assert (tr02.side_dir(out) / tr02.STOP).exists()


def test_an_unreadable_baseline_probe_refuses_to_launch(tmp_path: Path) -> None:
    def broken(pid):
        raise OSError("sysctl missing")

    result, out = _run(tmp_path, _events(), sampler=broken)
    assert (
        result["reason"] == "baseline_probe_failed:OSError" and not (tmp_path / "launched").exists()
    )


def test_a_probe_that_fails_while_the_child_runs_is_a_stop_not_a_skip(tmp_path: Path) -> None:
    def broken(pid):
        if pid is None:
            return {**_os(), "t": 0.0}
        raise OSError("ps failed")

    result, _ = _run(tmp_path, _events(), sampler=broken, sleep=30.0)
    assert result["reason"] == "probe_failed:OSError" and result["child_exit_code"] != 0


def test_a_probe_failure_after_the_child_exited_is_the_ps_race_not_a_stop(tmp_path: Path) -> None:
    """``ps`` on an exited child fails; that is a normal exit, not a probe failure."""

    def racy(pid):
        if pid is None or not (tmp_path / "launched").exists():
            return {**_os(), "t": 0.0, "rss_gb": None if pid is None else 5.0}
        time.sleep(0.6)  # the child (0.1 s) has exited by now; ps would find no such process
        raise OSError("no such process")

    result, _ = _run(tmp_path, _events(), sampler=racy, sleep=0.1)
    assert result["stopped"] is False and result["child_exit_code"] == 0


def test_an_interrupt_terminates_the_child_and_keeps_the_logs(tmp_path: Path) -> None:
    calls = {"n": 0}

    def interrupted(pid):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return {**_os(), "t": 0.0, "rss_gb": None if pid is None else 5.0}

    started = time.monotonic()
    result, out = _run(tmp_path, _events()[:3], sampler=interrupted, sleep=30.0)
    assert result["reason"] == "interrupted" and result["child_exit_code"] != 0
    assert time.monotonic() - started < 20
    assert (out / tr02.EVENTS).exists() and (tr02.side_dir(out) / "child.log").exists()


def test_output_and_supervisor_directories_must_be_new_and_logs_are_never_overwritten(
    tmp_path: Path,
) -> None:
    result, out = _run(tmp_path, _events())
    first = (tr02.side_dir(out) / "supervisor.jsonl").read_text("utf-8")
    with pytest.raises(FileExistsError):
        tr02.supervise(_cmd(out, _events()), out, _rules(), _good(), poll_s=0.05)
    assert (tr02.side_dir(out) / "supervisor.jsonl").read_text("utf-8") == first
    only_side = tmp_path / "other"
    tr02.side_dir(only_side).mkdir(parents=True)  # a leftover supervisor directory
    with pytest.raises(FileExistsError):
        tr02.supervise(_cmd(only_side, _events()), only_side, _rules(), _good(), poll_s=0.05)
    assert result["complete"]


def test_a_driver_that_creates_its_own_output_directory_is_supervised(tmp_path: Path) -> None:
    out = tmp_path / "phase"
    assert not out.exists()
    result, _ = _run(tmp_path, [_start(), {**_load_ev(), "seconds": float("nan")}], sleep=30.0)
    assert result["reason"] == "non_finite_seconds"


# ----------------------------------------------------------------------------------- evidence


def test_collect_evidence_copies_raw_records_and_hashes_the_adapter(
    tmp_path: Path,
) -> None:
    _, out = _run(tmp_path, _events())
    dest = tmp_path / "evidence"
    index = tr02.collect_evidence([("smoke", out)], dest)
    files = index["phases"]["smoke"]["files"]
    assert {"events.jsonl", "supervisor.jsonl", "child.log", "COMPLETE.json"} <= set(files)
    assert (dest / "smoke" / "events.jsonl").read_bytes() == (out / "events.jsonl").read_bytes()
    assert not (dest / "smoke" / tr02.ADAPTER_FILE).exists()  # weights are hashed, not copied
    assert index["phases"]["smoke"]["not_copied"][tr02.ADAPTER_FILE]["sha256"] == A_SHA
    assert json.loads((dest / "EVIDENCE.json").read_text("utf-8")) == index
    with pytest.raises(FileExistsError):
        tr02.collect_evidence([("smoke", out)], dest)  # never overwritten


def test_collect_evidence_keeps_the_stop_record_of_a_failed_phase(tmp_path: Path) -> None:
    ev = [*_events()[:4], _train(4, peak=11.0)]
    _, out = _run(tmp_path, ev, sleep=30.0)
    index = tr02.collect_evidence([("probe", out)], tmp_path / "evidence")
    assert "STOP.json" in index["phases"]["probe"]["files"]
    assert "COMPLETE.json" not in index["phases"]["probe"]["files"]


def test_collect_evidence_never_copies_a_weight_file_and_hashes_what_it_skips(
    tmp_path: Path,
) -> None:
    _, out = _run(tmp_path, _events())
    (out / "0000100_adapters.safetensors").write_bytes(b"checkpoint")  # a trainer checkpoint
    dest = tmp_path / "evidence"
    index = tr02.collect_evidence([("smoke", out)], dest)
    assert not list(dest.rglob("*.safetensors"))
    skipped = index["phases"]["smoke"]["not_copied"]
    assert set(skipped) >= {tr02.ADAPTER_FILE, "0000100_adapters.safetensors"}
    assert skipped["0000100_adapters.safetensors"]["bytes"] == len(b"checkpoint")


def test_collect_evidence_refuses_a_missing_run_duplicate_names_and_an_existing_destination(
    tmp_path: Path,
) -> None:
    _, out = _run(tmp_path, _events())
    dest = tmp_path / "evidence"
    with pytest.raises(FileNotFoundError):
        tr02.collect_evidence([("smoke", out), ("reload", tmp_path / "never_ran")], dest)
    assert not dest.exists()  # nothing was created before the check
    with pytest.raises(ValueError, match="unique"):
        tr02.collect_evidence([("smoke", out), ("smoke", out)], dest)
    dest.mkdir()
    (dest / "keep.txt").write_text("earlier evidence", "utf-8")
    with pytest.raises(FileExistsError):
        tr02.collect_evidence([("smoke", out)], dest)
    assert (dest / "keep.txt").read_text("utf-8") == "earlier evidence"  # never overwritten


def test_collect_evidence_of_a_stopped_phase_has_no_adapter_and_no_complete(
    tmp_path: Path,
) -> None:
    ev = [*_events()[:3], {"kind": "lora", "shapes_ok": False, "tensors": 32}]
    _, out = _run(tmp_path, ev, sleep=30.0)
    (out / tr02.ADAPTER_FILE).unlink(missing_ok=True)  # a phase that stopped never saved one
    index = tr02.collect_evidence([("probe", out)], tmp_path / "evidence")
    got = index["phases"]["probe"]
    assert "STOP.json" in got["files"] and "COMPLETE.json" not in got["files"]
    assert tr02.ADAPTER_FILE not in got["files"] and tr02.ADAPTER_FILE not in got["not_copied"]


def test_collect_evidence_command_accepts_only_a_destination_inside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out = _run(tmp_path, _events())
    outside = tmp_path / "outside"
    with pytest.raises(SystemExit, match="inside the repository"):
        tr02.main(["collect-evidence", "--dest", str(outside), "--run", f"smoke={out}"])
    assert not outside.exists()
    monkeypatch.setattr(tr02, "REPO", tmp_path / "evidence-checkout")
    inside = tmp_path / "evidence-checkout" / "raw"
    assert tr02.main(["collect-evidence", "--dest", str(inside), "--run", f"smoke={out}"]) == 0
    assert (inside / "EVIDENCE.json").is_file()


HOME = "/Users/someone"


def _raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    (raw / "probe").mkdir(parents=True)
    (raw / "probe" / "child.log").write_text(
        f'File "{HOME}/repo/tr02.py", line 12\nuses /opt/homebrew/bin/python3 and Owner\n'
        "token hf_" + "a" * 24 + "\n",
        "utf-8",
    )
    (raw / "probe" / "events.jsonl").write_text('{"kind": "start"}\n', "utf-8")
    return raw


def test_sanitize_evidence_replaces_private_text_and_records_every_redaction(
    tmp_path: Path,
) -> None:
    raw = _raw_dir(tmp_path)
    dest = tmp_path / "clean"
    manifest = tr02.sanitize_evidence(raw, dest, HOME, {"Owner": "<owner>"})
    text = (dest / "probe" / "child.log").read_text("utf-8")
    assert "$HOME/repo/tr02.py" in text and "<local-path>" in text and "<owner>" in text
    assert "<redacted-credential>" in text and "someone" not in text and "hf_a" not in text
    by_name = {f["original_filename"]: f for f in manifest["files"]}
    log = by_name["probe/child.log"]
    assert log["redaction_counts"] == {
        "home_directory_prefix": 1,
        "other_absolute_local_path": 1,
        "named_term": 1,
        "credential_like_token": 1,
    }
    assert (
        log["original_sha256"]
        == hashlib.sha256((raw / "probe" / "child.log").read_bytes()).hexdigest()
    )
    assert (
        log["sanitized_sha256"]
        == hashlib.sha256((dest / "probe" / "child.log").read_bytes()).hexdigest()
    )
    assert log["original_bytes"] != log["sanitized_bytes"]
    assert json.loads((dest / "REDACTIONS.json").read_text("utf-8")) == manifest


def test_sanitize_evidence_keeps_a_clean_file_byte_identical(tmp_path: Path) -> None:
    raw = _raw_dir(tmp_path)
    dest = tmp_path / "clean"
    manifest = tr02.sanitize_evidence(raw, dest, HOME, {"Owner": "<owner>"})
    clean = next(f for f in manifest["files"] if f["original_filename"] == "probe/events.jsonl")
    assert clean["original_sha256"] == clean["sanitized_sha256"]
    assert (dest / "probe" / "events.jsonl").read_bytes() == (
        raw / "probe" / "events.jsonl"
    ).read_bytes()
    assert sum(clean["redaction_counts"].values()) == 0


def test_sanitize_evidence_refuses_weights_binary_files_and_existing_destinations(
    tmp_path: Path,
) -> None:
    raw = _raw_dir(tmp_path)
    (raw / "weights.safetensors").write_bytes(b"w")
    with pytest.raises(ValueError, match="weight"):
        tr02.sanitize_evidence(raw, tmp_path / "a", HOME)
    assert not (tmp_path / "a").exists()
    (raw / "weights.safetensors").unlink()
    (raw / "blob.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="UTF-8"):
        tr02.sanitize_evidence(raw, tmp_path / "b", HOME)
    (raw / "blob.bin").unlink()
    (tmp_path / "c").mkdir()
    with pytest.raises(FileExistsError):
        tr02.sanitize_evidence(raw, tmp_path / "c", HOME)


def test_sanitize_evidence_fails_when_a_private_path_survives(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "x.log").write_text("path /Users/other/place with the wrong home\n", "utf-8")
    # another user's home is caught by the generic local-path rule
    manifest = tr02.sanitize_evidence(raw, tmp_path / "clean", HOME)
    assert manifest["files"][0]["redaction_counts"]["other_absolute_local_path"] == 1
    (raw / "y.log").write_text("Windows C:\\Users\\someone\\x\n", "utf-8")
    with pytest.raises(ValueError, match="still holds private text"):
        tr02.sanitize_evidence(raw, tmp_path / "clean2", HOME)


def test_sanitize_evidence_command_accepts_only_a_destination_inside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_dir(tmp_path)
    with pytest.raises(SystemExit, match="inside the repository"):
        tr02.main(
            ["sanitize-evidence", "--src", str(raw), "--dest", str(tmp_path / "o"), "--home", HOME]
        )
    monkeypatch.setattr(tr02, "REPO", tmp_path / "wt")
    args = ["sanitize-evidence", "--src", str(raw), "--dest", str(tmp_path / "wt" / "clean")]
    assert tr02.main([*args, "--home", HOME, "--term", "Owner=<owner>"]) == 0
    assert (tmp_path / "wt" / "clean" / "REDACTIONS.json").is_file()


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
    lengths = tr02.row_lengths(export_dir, PER_EXAMPLE, CFG["inputs"]["per_example_sha256"])[
        "smoke/train.jsonl"
    ]
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


def test_check_iterator_agrees_with_the_real_iterator_and_loss_mask(
    export_dir: Path, tokenizer
) -> None:
    pytest.importorskip("mlx_lm")
    where = _tokenizer_dir()
    result = tr02.check_iterator(export_dir, CFG, _phase("probe"), where, mask_rows=2)
    assert result["problems"] == [] and result["problem_count"] == 0
    assert len(result["mask_samples"]) == 2
    assert all(m["stock_mask_tokens"] == m["mask_tokens"] + 1 for m in result["mask_samples"])


def test_exact_loss_masks_exactly_the_answer_one_token_fewer_than_the_stock_loss() -> None:
    mx = pytest.importorskip("mlx.core")
    trainer = pytest.importorskip("mlx_lm.tuner.trainer")
    batch = mx.array([[5, 6, 7, 8, 9, 10, 11, 0, 0]])  # 7 real tokens, then padding
    lengths = mx.array([[3, 7]])  # the answer is tokens 3..6 (four tokens)

    def stub(x):
        return mx.zeros((*x.shape, 16))

    _, exact = tr02.make_exact_loss()(stub, batch, lengths)
    _, stock = trainer.default_loss(stub, batch, lengths)
    assert int(exact.item()) == 4 and int(stock.item()) == 5  # stock also scores one pad target
