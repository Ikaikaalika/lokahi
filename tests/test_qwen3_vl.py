"""Qwen3-VL and Qwen3-VL-MoE: the fixture's guards, and the native engine against the oracle.

The oracle is pinned to MLX-VLM in test_mlx_vlm_crosscheck.py. Correctness
evidence only: generated fixtures, never real weights.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures import TinyQwen3VLSpec, write_tiny_qwen3_vl
from lokahi.fixtures.qwen3_vl_tiny import VL_MOE_TEXT
from lokahi.oracles import Qwen3VLReference
from lokahi.oracles.qwen3_5_vision import Qwen35VisionConfig
from lokahi.oracles.qwen3_vl import text_config

SPEC = TinyQwen3VLSpec()
VARIANTS = {
    "qwen3-vl-mlx": SPEC,
    "qwen3-vl-hf": replace(SPEC, mlx=False),
    "qwen3-vl-moe-mlx": replace(SPEC, text=VL_MOE_TEXT),
}
# Images of 32x64 and 96x32 pixels: merged grids of 2x4 and 6x2 tokens.
SIZES = [(32, 64), (96, 32)]
GRIDS = [(h // 16, w // 16) for h, w in SIZES]
TOKENS = [rows * cols for rows, cols in GRIDS]


def _prompt(spec: TinyQwen3VLSpec = SPEC) -> list[int]:
    t = spec.tower
    start, end, image = t.vision_start_token_id, t.vision_end_token_id, t.image_token_id
    return [2, 5, start] + [image] * TOKENS[0] + [end, 7, 9, start] + [image] * TOKENS[1] + [end, 11, 13]


PROMPT = _prompt()


def _images(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in SIZES]


# Qwen/Qwen3-VL-2B-Instruct's config.json, as published (text fields trimmed
# to those the reference reads).
CONFIG_2B = {
    "image_token_id": 151655, "model_type": "qwen3_vl", "video_token_id": 151656,
    "vision_start_token_id": 151652, "vision_end_token_id": 151653, "tie_word_embeddings": True,
    "text_config": {
        "attention_bias": False, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144,
        "max_position_embeddings": 262144, "model_type": "qwen3_vl_text", "num_attention_heads": 16,
        "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000, "tie_word_embeddings": True, "vocab_size": 151936,
    },
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1024,
        "in_channels": 3, "intermediate_size": 4096, "model_type": "qwen3_vl", "num_heads": 16,
        "num_position_embeddings": 2304, "out_hidden_size": 2048, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return Qwen3VLReference.load(write_tiny_qwen3_vl(tmp_path_factory.mktemp("q3vl") / "model", SPEC))


def test_config_reads_the_published_2b_fields():
    text = text_config(CONFIG_2B)
    assert (text.model_type, text.num_hidden_layers, text.head_dim, text.mrope_section) == ("qwen3", 28, 128, (24, 20, 20))
    axes = text.rope_axes()
    # Sections [temporal 24, rows 20, columns 20]: frequencies 60-63 stay temporal.
    assert axes.size == 64 and list(np.flatnonzero(axes == 1)) == list(range(1, 60, 3))
    assert list(np.flatnonzero(axes == 2)) == list(range(2, 60, 3))
    vision = Qwen35VisionConfig.from_dict(CONFIG_2B, {"size": {"longest_edge": 16777216, "shortest_edge": 65536}})
    assert vision.deepstack_visual_indexes == (5, 11, 17) and vision.factor == 32


def test_fixture_deepstack_and_each_rope_axis_matter(reference):
    # Guards the checks below: deepstack, rows, columns and swapping them each
    # move the logits well beyond the Metal tolerance (2e-2).
    images = _images()
    injected, positions, _, deepstack = reference._inputs(PROMPT, images)
    expected = reference.language.logits(PROMPT, injected, positions, deepstack)
    scale = np.abs(expected).max()

    def moved(other_positions=positions, other_deepstack=deepstack) -> float:
        logits = reference.language.logits(PROMPT, injected, other_positions, other_deepstack)
        return np.abs(logits - expected).max() / scale

    no_rows, no_columns = positions.copy(), positions.copy()
    no_rows[1], no_columns[2] = positions[0], positions[0]
    assert len(deepstack) == 2 and all(len(layer) == sum(TOKENS) for layer in deepstack)
    for name, value in [("rows", moved(no_rows)), ("columns", moved(no_columns)), ("swapped", moved(positions[[0, 2, 1]])),
                        ("no deepstack", moved(other_deepstack=())), ("first layer only", moved(other_deepstack=deepstack[:1])),
                        ("second layer only", moved(other_deepstack=[{}, deepstack[1]]))]:
        assert value > 0.03, name
    # Greedy continuations vary, and with the images.
    greedy = reference.greedy(PROMPT, images, 10)
    assert len(set(greedy)) >= 4 and greedy != reference.greedy(PROMPT, _images(1), 10)


# ---------------------------------------------------------------------------
# The native engine against the oracle.
# ---------------------------------------------------------------------------

def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _run(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _fails(cli: Path, *args: str) -> str:
    result = subprocess.run([str(cli), *args], capture_output=True, text=True)
    assert result.returncode != 0, result.stdout
    return result.stderr


def write_ppm(path: Path, rgb: np.ndarray) -> Path:
    height, width = rgb.shape[:2]
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + np.ascontiguousarray(rgb).tobytes())
    return path


def _image_files(directory: Path, images: list[np.ndarray]) -> str:
    return ",".join(str(write_ppm(directory / f"image{i}.ppm", image)) for i, image in enumerate(images))


def _skip_unsupported(directory: Path, backend: str) -> None:
    if backend == "metal" and "quantization" not in json.loads((directory / "config.json").read_text()):
        pytest.skip("the Metal backend reads quantized checkpoints only")


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def vlm(request, tmp_path_factory):
    directory = write_tiny_qwen3_vl(tmp_path_factory.mktemp(request.param) / "model", VARIANTS[request.param])
    images = _images(seed=3)
    return directory, Qwen3VLReference.load(directory), images, _image_files(directory, images)


@pytest.mark.parametrize("backend", _backends())
def test_image_and_deepstack_features_match_oracle(native_cli, vlm, backend, tmp_path):
    directory, oracle, images, files = vlm
    out = tmp_path / "features.bin"
    report = _run(native_cli, "image-features", "--model", str(directory), "--backend", backend,
                  "--images", files, "--out", str(out))
    width = oracle.language.config.hidden_size
    assert report == {"backend": backend, "tokens": TOKENS, "width": width, "deepstack": 2}
    got = np.fromfile(out, dtype=np.float32).reshape(3, sum(TOKENS), width)  # rows, then each deepstack layer
    encoded = [oracle.encode_all(image) for image in images]
    expected = np.stack([np.concatenate([features for features, _ in encoded])] +
                        [np.concatenate([layers[l] for _, layers in encoded]) for l in range(2)])
    tolerance = 1e-5 if backend == "cpu" else 1e-3
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)


# Chunks of 3 end passes inside both images; the deepstack rows still land on
# every image token.
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("chunk", [512, 7, 3])
def test_logits_with_images_match_oracle(native_cli, vlm, backend, chunk, tmp_path):
    directory, oracle, images, files = vlm
    _skip_unsupported(directory, backend)
    tokens = PROMPT + [44, 45, 46]
    out = tmp_path / "logits.bin"
    report = _run(
        native_cli, "logits", "--model", str(directory), "--backend", backend,
        "--tokens", ",".join(map(str, tokens)), "--prefill", str(len(PROMPT)), "--prefill-chunk", str(chunk),
        "--images", files, "--out", str(out),
    )
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], -1)
    injected, positions, delta, deepstack = oracle._inputs(PROMPT, images)
    later = np.arange(len(PROMPT), len(tokens)) + delta
    full = np.concatenate([positions, np.broadcast_to(later, (3, later.size))], axis=1)
    expected = oracle.language.logits(tokens, injected, full, deepstack)[len(PROMPT) - 1 :]
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))


@pytest.mark.parametrize("backend", _backends())
def test_greedy_with_images_matches_full_recompute(native_cli, vlm, backend):
    directory, oracle, images, files = vlm
    _skip_unsupported(directory, backend)
    report = _run(
        native_cli, "generate", "--model", str(directory), "--backend", backend,
        "--tokens", ",".join(map(str, PROMPT)), "--images", files, "--max-new", "10", "--no-eos-stop",
    )
    assert report["tokens"] == oracle.greedy(PROMPT, images, 10)


@pytest.mark.parametrize("backend", _backends())
def test_text_alone_matches_oracle(native_cli, tmp_path, backend):
    # Without images M-RoPE is plain RoPE and nothing is injected.
    directory = write_tiny_qwen3_vl(tmp_path / "model", SPEC)
    oracle = Qwen3VLReference.load(directory)
    tokens = [2, 5, 9, 33, 100, 7, 7, 400, 3, 11]
    out = tmp_path / "logits.bin"
    report = _run(native_cli, "logits", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, tokens)), "--prefill", "4", "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], -1)
    expected = oracle.language.logits(tokens)[3:]
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))
    info = _run(native_cli, "info", "--model", str(directory), "--backend", backend)
    assert info["architecture"] == "qwen3_vl"


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _set_vision(key, value):
    return lambda data: data["vision_config"].__setitem__(key, value)


def _set_rope(key, value):
    return lambda data: data["text_config"]["rope_scaling"].__setitem__(key, value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set_vision("deepstack_visual_indexes", [2, 0]), "deepstack_visual_indexes must be increasing"),
        (_set_vision("deepstack_visual_indexes", [0, 3]), "deepstack_visual_indexes must be increasing"),
        (_set_vision("deepstack_visual_indexes", [0, 0]), "deepstack_visual_indexes must be increasing"),
        (_set_vision("deepstack_visual_indexes", [0]), "more deepstack mergers than deepstack_visual_indexes"),
        (lambda data: data.__setitem__("model_type", "qwen3_5"), "deepstack features are Qwen3-VL's"),
    ],
)
def test_unsupported_vision_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_qwen3_vl(tmp_path / "model", SPEC)
    _edit(directory / "config.json", change)
    files = _image_files(tmp_path, _images()[:1])
    assert message in _fails(native_cli, "image-features", "--model", str(directory), "--images", files,
                             "--out", str(tmp_path / "x.bin"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set_rope("mrope_interleaved", False), "qwen3_vl: only interleaved M-RoPE is supported"),
        (_set_rope("mrope_section", [8, 8]), "qwen3_vl: mrope_section must have 3 entries"),
        (_set_rope("rope_type", "longrope"), "rope_scaling type 'longrope' is not supported"),
        (lambda data: data["text_config"].pop("rope_scaling"), "qwen3_vl: text_config has no rope_scaling"),
    ],
)
def test_unsupported_text_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_qwen3_vl(tmp_path / "model", SPEC)
    _edit(directory / "config.json", change)
    assert message in _fails(native_cli, "info", "--model", str(directory), "--backend", "cpu")



def test_transposed_moe_experts_fail_closed(native_cli, tmp_path):
    # Transformers' Qwen3-VL-MoE checkpoints fuse each layer's experts into
    # gate_up_proj [experts, hidden, 2 * moe_intermediate] and down_proj
    # [experts, moe_intermediate, hidden], transposed from Qwen3.5-MoE's.
    from lokahi.formats.safetensors import f32_to_bf16, read_safetensors, write_safetensors

    spec = replace(SPEC, text=VL_MOE_TEXT)
    directory = write_tiny_qwen3_vl(tmp_path / "model", spec)
    tensors, metadata = read_safetensors(directory / "model.safetensors")
    text = spec.text
    for layer in range(text.num_hidden_layers):
        base = f"language_model.model.layers.{layer}.mlp"
        for name in [n for n in tensors if n.startswith(f"{base}.switch_mlp.")]:
            del tensors[name]
        shapes = {"gate_up_proj": (text.num_experts, text.hidden_size, 2 * text.moe_intermediate_size),
                  "down_proj": (text.num_experts, text.moe_intermediate_size, text.hidden_size)}
        for name, shape in shapes.items():
            tensors[f"{base}.experts.{name}"] = ("BF16", f32_to_bf16(np.zeros(shape, dtype=np.float32)))
    write_safetensors(directory / "model.safetensors", tensors, metadata=metadata)
    assert "fused, transposed Qwen3-VL-MoE experts are not supported" in _fails(
        native_cli, "info", "--model", str(directory), "--backend", "cpu")
