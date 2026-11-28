"""EXP-TR02 LoRA namespace, save and reload contract on a real, randomly initialized tiny Qwen3.

These tests build an actual ``mlx_lm.models.qwen3.Model`` with the locked MLX 0.32.4 and mlx-lm
0.32.2 (``envs/model``), run the harness's own LoRA conversion, name and shape contract, safetensors
save and mlx-lm ``load_adapters`` path, and need no download and no pretrained weight. Nothing is
trained: no optimizer step is taken, values are assigned deterministically to test serialization.

They are SKIPPED, not passed, where MLX or mlx-lm is not installed (the Core CI environment installs
neither). They run in the locked model environment, on Linux CPU and on the Apple Silicon M5, and
the same checks run on the M5 as ``tr02.py check-lora`` in the no-weight preflight.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is not installed (Core CI environment): skipped")
pytest.importorskip("mlx_lm", reason="mlx-lm is not installed (Core CI environment): skipped")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.tuner.utils import load_adapters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load():
    if "tr02" in sys.modules:
        return sys.modules["tr02"]
    path = ROOT / "training" / "tr02" / "tr02.py"
    spec = importlib.util.spec_from_file_location("tr02", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tr02"] = module
    spec.loader.exec_module(module)
    return module


tr02 = _load()
HP = tr02.tiny_hyperparameters(tr02.load_config()["hyperparameters"])
WANT = tr02.expected_lora_shapes(tr02.TINY_QWEN3_CONFIG, HP)


def _converted(seed: int = 0):
    model = tr02.build_tiny_qwen3(seed)
    tr02.attach_lora(model, HP)
    return model


def _saved_adapter(tmp_path: Path):
    """A converted tiny model with fixture values, saved the way the trainer saves it."""
    model = _converted(0)
    tr02.assign_fixture_values(model)
    adapter = tmp_path / tr02.ADAPTER_FILE
    mx.save_safetensors(str(adapter), dict(tree_flatten(model.trainable_parameters())))
    (tmp_path / tr02.ADAPTER_CONFIG).write_text(json.dumps(tr02.lora_adapter_config(HP)), "utf-8")
    return model, adapter


def test_the_tiny_model_is_a_real_qwen3_with_the_nested_model_attribute() -> None:
    model = tr02.build_tiny_qwen3(0)
    names = [k for k, _ in tree_flatten(model.parameters())]
    assert type(model).__module__ == "mlx_lm.models.qwen3"
    assert any(k.startswith("model.layers.0.self_attn.q_proj") for k in names)
    assert not any(k.startswith("layers.") for k in names)


def test_production_conversion_yields_exactly_the_contract_names_and_shapes() -> None:
    model = _converted(0)
    flat = dict(tree_flatten(model.trainable_parameters()))
    got = {k: tuple(v.shape) for k, v in flat.items()}
    assert got == WANT  # exact names (with the `model.` prefix) and exact shapes
    assert tr02.lora_problems(got, WANT) == []
    assert len(got) == 8 and tr02.trainable_params(got) == tr02.trainable_params(WANT)
    assert all(k.startswith("model.layers.") for k in got)
    # frozen base: nothing but LoRA tensors is trainable; a base weight is still present but frozen
    assert not any(not k.rsplit(".", 1)[-1].startswith("lora_") for k in got)
    assert "model.layers.0.self_attn.q_proj.weight" in dict(tree_flatten(model.parameters()))
    # lora_b starts at zero (so the converted model equals the base model before any update)
    assert all(float(abs(v).sum()) == 0.0 for k, v in flat.items() if k.endswith("lora_b"))


def test_the_contract_is_derived_from_the_config_and_not_from_the_model() -> None:
    # expected names come from the config alone: a model with other layers changes nothing
    again = tr02.expected_lora_shapes(tr02.TINY_QWEN3_CONFIG, HP)
    assert again == WANT
    assert sorted(WANT) == [
        f"model.layers.{layer}.self_attn.{proj}_proj.lora_{ab}"
        for layer in (2, 3)
        for proj in ("q", "v")
        for ab in ("a", "b")
    ]
    assert WANT["model.layers.2.self_attn.q_proj.lora_a"] == (64, 4)
    assert WANT["model.layers.2.self_attn.q_proj.lora_b"] == (4, 4 * 16)
    assert WANT["model.layers.3.self_attn.v_proj.lora_b"] == (4, 2 * 16)


def test_save_and_reload_round_trip_returns_the_saved_tensors(tmp_path: Path) -> None:
    model, adapter = _saved_adapter(tmp_path)
    saved = {k: v for k, v in tr02.lora_leaves(model).items()}
    on_disk = {
        k: __import__("numpy").array(v.astype(mx.float32)) for k, v in mx.load(str(adapter)).items()
    }
    assert set(on_disk) == set(WANT) == set(saved)
    assert tr02.lora_problems({k: tuple(v.shape) for k, v in on_disk.items()}, WANT) == []
    fresh = tr02.build_tiny_qwen3(1)  # different random base, adapter not applied yet
    load_adapters(fresh, str(tmp_path))  # the path mlx_lm.load(adapter_path=...) takes
    live = tr02.lora_leaves(fresh)
    assert set(live) == set(WANT)
    assert tr02.live_adapter_problems(on_disk, live) == []
    import numpy as np

    assert all(bool(np.isfinite(v).all()) for v in live.values())
    assert all(float(np.abs(v).sum()) > 0.0 for k, v in live.items() if k.endswith("lora_b"))


def test_the_fixture_check_the_m5_preflight_runs_passes(tmp_path: Path) -> None:
    result = tr02.lora_fixture_check(tr02.load_config()["hyperparameters"], tmp_path)
    assert result["problems"] == []
    assert result["tensors"] == 8 and result["trainable_params"] == 1792
    assert result["names"][0] == "model.layers.2.self_attn.q_proj.lora_a"


# ------------------------------------------------------------------------------- negatives


def _problems(names_and_shapes: dict) -> list[str]:
    return tr02.lora_problems(names_and_shapes, WANT)


def _write(tmp_path: Path, tensors: dict, config: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tmp_path / tr02.ADAPTER_FILE), tensors)
    cfg = config if config is not None else tr02.lora_adapter_config(HP)
    (tmp_path / tr02.ADAPTER_CONFIG).write_text(json.dumps(cfg), "utf-8")
    return tmp_path


def _tensors(tmp_path: Path) -> dict:
    _, adapter = _saved_adapter(tmp_path / "good")  # needs the directory to exist
    return dict(mx.load(str(adapter)))


@pytest.fixture
def good(tmp_path: Path) -> dict:
    (tmp_path / "good").mkdir()
    return _tensors(tmp_path)


def test_a_missing_tensor_is_named(good: dict) -> None:
    name = "model.layers.2.self_attn.q_proj.lora_a"
    del good[name]
    assert f"missing tensor: {name}" in _problems({k: tuple(v.shape) for k, v in good.items()})


def test_an_extra_tensor_is_named_and_mlx_lm_refuses_it(good: dict, tmp_path: Path) -> None:
    good["model.layers.2.self_attn.q_proj.lora_c"] = good["model.layers.2.self_attn.q_proj.lora_a"]
    shapes = {k: tuple(v.shape) for k, v in good.items()}
    assert "unexpected tensor: model.layers.2.self_attn.q_proj.lora_c" in _problems(shapes)
    with pytest.raises(ValueError, match="not an adapter parameter"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "out", good)))


def test_a_wrong_layer_is_named_and_mlx_lm_refuses_it(good: dict, tmp_path: Path) -> None:
    good["model.layers.1.self_attn.q_proj.lora_a"] = good.pop(
        "model.layers.2.self_attn.q_proj.lora_a"
    )
    problems = _problems({k: tuple(v.shape) for k, v in good.items()})
    assert "missing tensor: model.layers.2.self_attn.q_proj.lora_a" in problems
    assert "unexpected tensor: model.layers.1.self_attn.q_proj.lora_a" in problems
    with pytest.raises(ValueError, match="not an adapter parameter"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "out", good)))


def test_a_wrong_projection_is_named_and_mlx_lm_refuses_it(good: dict, tmp_path: Path) -> None:
    good["model.layers.3.self_attn.k_proj.lora_a"] = good.pop(
        "model.layers.3.self_attn.v_proj.lora_a"
    )
    problems = _problems({k: tuple(v.shape) for k, v in good.items()})
    assert "missing tensor: model.layers.3.self_attn.v_proj.lora_a" in problems
    assert "unexpected tensor: model.layers.3.self_attn.k_proj.lora_a" in problems
    with pytest.raises(ValueError, match="not an adapter parameter"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "out", good)))


def test_a_wrong_shape_is_named_and_mlx_lm_refuses_it(good: dict, tmp_path: Path) -> None:
    name = "model.layers.2.self_attn.v_proj.lora_b"
    good[name] = mx.zeros((4, 64))  # expected (4, 32)
    problems = _problems({k: tuple(v.shape) for k, v in good.items()})
    assert any(name in p and "expected (4, 32)" in p for p in problems)
    with pytest.raises(ValueError, match="does not match"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "out", good)))


def test_the_unprefixed_or_mixed_namespace_is_rejected(good: dict, tmp_path: Path) -> None:
    bare = {k.removeprefix("model."): v for k, v in good.items()}
    problems = _problems({k: tuple(v.shape) for k, v in bare.items()})
    assert sum(p.startswith("missing tensor") for p in problems) == 8
    assert sum(p.startswith("unexpected tensor") for p in problems) == 8
    with pytest.raises(ValueError, match="not an adapter parameter"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "bare", bare)))
    mixed = dict(good)
    name = "model.layers.3.self_attn.q_proj.lora_b"
    mixed[name.removeprefix("model.")] = mixed.pop(name)
    problems = _problems({k: tuple(v.shape) for k, v in mixed.items()})
    assert problems == [
        f"missing tensor: {name}",
        f"unexpected tensor: {name.removeprefix('model.')}",
    ]
    with pytest.raises(ValueError, match="not an adapter parameter"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "mixed", mixed)))


def test_a_tampered_file_or_a_mismatched_config_is_caught(good: dict, tmp_path: Path) -> None:
    import numpy as np

    saved = {k: np.array(v.astype(mx.float32)) for k, v in good.items()}
    # values differ between the file the harness hashed and what a reload would produce
    tampered = dict(good)
    name = "model.layers.2.self_attn.q_proj.lora_a"
    tampered[name] = good[name] + 1.0
    fresh = tr02.build_tiny_qwen3(1)
    load_adapters(fresh, str(_write(tmp_path / "tampered", tampered)))
    problems = tr02.live_adapter_problems(saved, tr02.lora_leaves(fresh))
    assert problems and name in problems[0] and "values differ" in problems[0]
    # a config with another rank cannot hold the saved tensors: mlx-lm refuses the adapter
    other_rank = tr02.lora_adapter_config({**HP, "lora_rank": 8})
    with pytest.raises(ValueError, match="does not match"):
        load_adapters(tr02.build_tiny_qwen3(1), str(_write(tmp_path / "rank", good, other_rank)))
    # a config converting other layers: the saved tensors are not adapter parameters there
    other_layers = tr02.lora_adapter_config({**HP, "lora_layers": 1})
    with pytest.raises(ValueError, match="not an adapter parameter"):
        load_adapters(
            tr02.build_tiny_qwen3(1), str(_write(tmp_path / "layers", good, other_layers))
        )


def test_a_non_finite_reloaded_tensor_is_caught(good: dict, tmp_path: Path) -> None:
    import numpy as np

    name = "model.layers.3.self_attn.q_proj.lora_a"
    bad = dict(good)
    bad[name] = mx.full(good[name].shape, float("nan"))
    saved = {k: np.array(v.astype(mx.float32)) for k, v in bad.items()}
    fresh = tr02.build_tiny_qwen3(1)
    load_adapters(fresh, str(_write(tmp_path / "nan", bad)))
    problems = tr02.live_adapter_problems(saved, tr02.lora_leaves(fresh))
    assert problems and "not finite" in problems[0]
