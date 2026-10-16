"""Qwen2.5-VL and Qwen2-VL: the fixtures' guards, and the native engine against the oracle.

The oracle is pinned to MLX-VLM and Transformers in
test_mlx_vlm_crosscheck.py. Correctness evidence only: generated fixtures,
never real weights.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures import TinyQwen25VLSpec, write_tiny_qwen2_5_vl
from lokahi.oracles import Qwen2VisionConfig, Qwen2VLReference, Qwen25VisionConfig, Qwen25VLReference
from lokahi.oracles.qwen2_5_vl import text_config, window_index

SPEC = TinyQwen25VLSpec()
# Qwen2-VL's tower: mlp_ratio 2, and a seed whose greedy continuations vary.
QWEN2_SPEC = replace(SPEC, family="qwen2_vl", tower=replace(SPEC.tower, intermediate_size=128, seed=335))
VARIANTS = {
    "qwen2.5-vl-mlx": SPEC,
    "qwen2.5-vl-hf": replace(SPEC, layout="hf"),
    "qwen2.5-vl-hf5": replace(SPEC, layout="hf5"),
    "qwen2-vl-mlx": QWEN2_SPEC,
    "qwen2-vl-hf": replace(QWEN2_SPEC, layout="hf"),
    "qwen2-vl-hf5": replace(QWEN2_SPEC, layout="hf5"),
}


def _load(directory: Path, spec: TinyQwen25VLSpec) -> Qwen25VLReference:
    return (Qwen2VLReference if spec.family == "qwen2_vl" else Qwen25VLReference).load(directory)
# Images of 48x80 and 64x32 pixels: merged grids of 3x5 tokens (windows of
# 2x2 tokens: two whole, four cut by the right and bottom edges) and 4x2
# tokens (two whole windows; Transformers pads a whole empty row of them).
SIZES = [(48, 80), (64, 32)]
GRIDS = [(h // 16, w // 16) for h, w in SIZES]
TOKENS = [rows * cols for rows, cols in GRIDS]


def _prompt(spec: TinyQwen25VLSpec = SPEC) -> list[int]:
    t = spec.tower
    start, end, image = t.vision_start_token_id, t.vision_end_token_id, t.image_token_id
    return [2, 5, start] + [image] * TOKENS[0] + [end, 7, 9, start] + [image] * TOKENS[1] + [end, 11, 13]


PROMPT = _prompt()


def _images(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in SIZES]


# Qwen/Qwen2.5-VL-3B-Instruct's config.json and preprocessor_config.json, as
# published (Transformers 4: the text fields at the top level).
CONFIG_3B = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"], "attention_dropout": 0.0, "bos_token_id": 151643,
    "eos_token_id": 151645, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
    "vision_token_id": 151654, "image_token_id": 151655, "video_token_id": 151656, "hidden_act": "silu",
    "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 11008, "max_position_embeddings": 128000,
    "max_window_layers": 70, "model_type": "qwen2_5_vl", "num_attention_heads": 16, "num_hidden_layers": 36,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window": 32768,
    "tie_word_embeddings": True, "torch_dtype": "bfloat16", "transformers_version": "4.41.2", "use_cache": True,
    "use_sliding_window": False,
    "vision_config": {
        "depth": 32, "hidden_act": "silu", "hidden_size": 1280, "intermediate_size": 3420, "num_heads": 16,
        "in_chans": 3, "out_hidden_size": 2048, "patch_size": 14, "spatial_merge_size": 2, "spatial_patch_size": 14,
        "window_size": 112, "fullatt_block_indexes": [7, 15, 23, 31], "tokens_per_second": 2,
        "temporal_patch_size": 2,
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "vocab_size": 151936,
}
PREPROCESSOR_3B = {
    "min_pixels": 3136, "max_pixels": 12845056, "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    "image_processor_type": "Qwen2VLImageProcessor", "processor_class": "Qwen2_5_VLProcessor",
}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return Qwen25VLReference.load(write_tiny_qwen2_5_vl(tmp_path_factory.mktemp("q25vl") / "model", SPEC))


def test_config_reads_the_published_3b_fields():
    text = text_config(CONFIG_3B)
    assert (text.model_type, text.num_hidden_layers, text.head_dim, text.rope_theta) == ("qwen2", 36, 128, 1e6)
    assert text.mrope_section == (16, 24, 24) and not text.mrope_interleaved
    np.testing.assert_array_equal(text.rope_axes(), [0] * 16 + [1] * 24 + [2] * 24)
    vision = Qwen25VisionConfig.from_dict(CONFIG_3B, PREPROCESSOR_3B)
    assert (vision.factor, vision.window, vision.fullatt_block_indexes) == (28, 4, (7, 15, 23, 31))
    assert (vision.min_pixels, vision.max_pixels, vision.in_channels) == (3136, 12845056, 3)
    assert vision.image_mean == tuple(PREPROCESSOR_3B["image_mean"])


def test_window_index_follows_transformers_padding():
    # 3x5: windows of 2x2 tokens row by row, cut at the right and bottom.
    order, lengths = window_index(3, 5, 2)
    assert order.tolist() == [0, 1, 5, 6, 2, 3, 7, 8, 4, 9, 10, 11, 12, 13, 14]
    assert lengths == [4, 4, 2, 2, 2, 1]
    # A grid that is a whole number of windows keeps its order.
    order, lengths = window_index(4, 2, 2)
    assert order.tolist() == list(range(8)) and lengths == [4, 4]
    order, lengths = window_index(2, 3, 4)  # one window holds the whole grid
    assert order.tolist() == list(range(6)) and lengths == [6]


@pytest.mark.parametrize("family", ["qwen2.5-vl", "qwen2-vl"])
def test_layouts_agree(tmp_path, family):
    images = _images()
    names = [f"{family}-{layout}" for layout in ("mlx", "hf", "hf5")]
    oracles = {name: _load(write_tiny_qwen2_5_vl(tmp_path / name, VARIANTS[name]), VARIANTS[name]) for name in names}
    assert [o.tower for o in oracles.values()] == ["vision_tower", "visual", "model.visual"]
    features = {name: [o.encode(image) for image in images] for name, o in oracles.items()}
    for name in names:  # the tower's BF16 weights are the same in every layout
        for got, want in zip(features[name], features[names[0]]):
            np.testing.assert_array_equal(got, want)
    # Transformers 4's flat config and Transformers 5's text_config describe
    # the same text model (both store the dequantized BF16 weights).
    np.testing.assert_array_equal(oracles[names[1]].logits(PROMPT, images), oracles[names[2]].logits(PROMPT, images))


def test_fixture_windows_and_each_rope_axis_matter(reference):
    # Guards the checks below: windows, the full-attention block, rows,
    # columns, swapping them and the chunked layout each move the outputs
    # well beyond the Metal tolerance (2e-2).
    images = _images()
    features = reference.encode(images[0])

    def features_moved(**change) -> float:
        other = Qwen25VLReference(replace(reference.config, **change), reference.w, reference.language)
        return np.abs(other.encode(images[0]) - features).max() / np.abs(features).max()

    depth = reference.config.depth
    for name, value in [("every block global", features_moved(fullatt_block_indexes=tuple(range(depth)))),
                        ("no global block", features_moved(fullatt_block_indexes=())),
                        ("one window", features_moved(window_size=128))]:
        assert value > 0.03, name

    injected, positions, _, _ = reference._inputs(PROMPT, images)
    expected = reference.language.logits(PROMPT, injected, positions)
    scale = np.abs(expected).max()

    def moved(other_positions=positions, language=reference.language) -> float:
        return np.abs(language.logits(PROMPT, injected, other_positions) - expected).max() / scale

    no_rows, no_columns = positions.copy(), positions.copy()
    no_rows[1], no_columns[2] = positions[0], positions[0]
    interleaved = type(reference.language)(replace(reference.language.config, mrope_interleaved=True), reference.w)
    for name, value in [("rows", moved(no_rows)), ("columns", moved(no_columns)),
                        ("swapped", moved(positions[[0, 2, 1]])), ("interleaved", moved(language=interleaved))]:
        assert value > 0.03, name
    # Greedy continuations vary, and with the images.
    greedy = reference.greedy(PROMPT, images, 10)
    assert len(set(greedy)) >= 4 and greedy != reference.greedy(PROMPT, _images(1), 10)


# Qwen/Qwen2-VL-2B-Instruct's config.json, as published (text fields trimmed
# to those the reference reads).
CONFIG_QWEN2_2B = {
    "model_type": "qwen2_vl", "image_token_id": 151655, "vision_start_token_id": 151652,
    "vision_end_token_id": 151653, "hidden_act": "silu", "hidden_size": 1536, "intermediate_size": 8960,
    "num_attention_heads": 12, "num_hidden_layers": 28, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0, "sliding_window": 32768, "tie_word_embeddings": True, "use_sliding_window": False,
    "vision_config": {
        "depth": 32, "embed_dim": 1280, "mlp_ratio": 4, "num_heads": 16, "in_chans": 3, "hidden_size": 1536,
        "patch_size": 14, "spatial_merge_size": 2, "spatial_patch_size": 14, "temporal_patch_size": 2,
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "vocab_size": 151936,
}


def test_config_reads_the_published_qwen2_vl_2b_fields():
    text = text_config(CONFIG_QWEN2_2B)
    assert (text.model_type, text.num_hidden_layers, text.head_dim, text.mrope_section) == ("qwen2", 28, 128,
                                                                                           (16, 24, 24))
    vision = Qwen2VisionConfig.from_dict(CONFIG_QWEN2_2B, PREPROCESSOR_3B)
    assert (vision.hidden_size, vision.intermediate_size, vision.out_hidden_size) == (1280, 5120, 1536)
    assert vision.fullatt_block_indexes == tuple(range(32)) and vision.factor == 28


def test_qwen2_vl_fixture_each_rope_axis_matters(tmp_path):
    reference = _load(write_tiny_qwen2_5_vl(tmp_path / "model", QWEN2_SPEC), QWEN2_SPEC)
    images = _images()
    injected, positions, _, _ = reference._inputs(PROMPT, images)
    expected = reference.language.logits(PROMPT, injected, positions)
    scale = np.abs(expected).max()
    no_rows, no_columns = positions.copy(), positions.copy()
    no_rows[1], no_columns[2] = positions[0], positions[0]
    for name, other in [("rows", no_rows), ("columns", no_columns), ("swapped", positions[[0, 2, 1]])]:
        assert np.abs(reference.language.logits(PROMPT, injected, other) - expected).max() / scale > 0.03, name
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
    directory = write_tiny_qwen2_5_vl(tmp_path_factory.mktemp(request.param) / "model", VARIANTS[request.param])
    images = _images(seed=3)
    return directory, _load(directory, VARIANTS[request.param]), images, _image_files(directory, images)


# 7 query rows per Metal attention pass divides none of the images' patch
# counts, so the last pass of each is partial (a no-op on the CPU).
@pytest.mark.parametrize("attention_rows", ["0", "7"])
@pytest.mark.parametrize("backend", _backends())
def test_image_features_match_oracle(native_cli, vlm, backend, attention_rows, tmp_path):
    directory, oracle, images, files = vlm
    out = tmp_path / "features.bin"
    report = _run(native_cli, "image-features", "--model", str(directory), "--backend", backend,
                  "--images", files, "--out", str(out), "--vision-attention-rows", attention_rows)
    width = oracle.language.config.hidden_size
    assert report == {"backend": backend, "tokens": TOKENS, "width": width}
    got = np.fromfile(out, dtype=np.float32).reshape(sum(TOKENS), width)
    expected = np.concatenate([oracle.encode(image) for image in images])
    tolerance = 1e-5 if backend == "cpu" else 1e-3
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)


# Chunks of 3 end passes inside both images.
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
    injected, positions, delta, _ = oracle._inputs(PROMPT, images)
    later = np.arange(len(PROMPT), len(tokens)) + delta
    full = np.concatenate([positions, np.broadcast_to(later, (3, later.size))], axis=1)
    expected = oracle.language.logits(tokens, injected, full)[len(PROMPT) - 1 :]
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
def test_text_alone_matches_oracle(native_cli, vlm, backend, tmp_path):
    # Without images M-RoPE is plain RoPE and nothing is injected.
    directory, oracle, _, _ = vlm
    _skip_unsupported(directory, backend)
    tokens = [2, 5, 9, 33, 100, 7, 7, 400, 3, 11]
    out = tmp_path / "logits.bin"
    report = _run(native_cli, "logits", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, tokens)), "--prefill", "4", "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], -1)
    expected = oracle.language.logits(tokens)[3:]
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))
    family = json.loads((directory / "config.json").read_text())["model_type"]  # qwen2_5_vl or qwen2_vl
    assert _run(native_cli, "info", "--model", str(directory), "--backend", backend)["architecture"] == family


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _set_vision(key, value):
    return lambda data: data["vision_config"].__setitem__(key, value)


def _set(key, value):
    return lambda data: data.__setitem__(key, value)


def _set_rope(key, value):
    return lambda data: data["rope_scaling"].__setitem__(key, value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set_vision("hidden_act", "gelu"), "qwen2_5_vl vision: unsupported activation gelu"),
        (_set_vision("in_chans", 4), "qwen2_5_vl vision: images must have 3 channels"),
        (_set_vision("fullatt_block_indexes", [1, 1]), "fullatt_block_indexes must be increasing blocks"),
        (_set_vision("fullatt_block_indexes", [3]), "fullatt_block_indexes must be increasing blocks"),
        (_set_vision("window_size", 8), "qwen2_5_vl vision: window_size must cover a merged token"),
        (_set_vision("deepstack_visual_indexes", [0]), "deepstack features are Qwen3-VL's, not qwen2_5_vl's"),
        (_set_vision("rope_parameters", {"rope_type": "axial", "rope_theta": 5000.0}),
         "only axial 2-D RoPE with theta 10000"),
        (_set_vision("out_hidden_size", 64), "out_hidden_size must be the text model's hidden_size"),
    ],
)
def test_unsupported_vision_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_qwen2_5_vl(tmp_path / "model", SPEC)
    _edit(directory / "config.json", change)
    files = _image_files(tmp_path, _images()[:1])
    assert message in _fails(native_cli, "image-features", "--model", str(directory), "--images", files,
                             "--out", str(tmp_path / "x.bin"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set_rope("mrope_interleaved", True), "qwen2_5_vl: only chunked M-RoPE is supported"),
        (_set_rope("mrope_section", [8, 12, 11]), "qwen2_5_vl: mrope_section must cover the head's 32 frequencies"),
        (_set_rope("mrope_section", [16, 16]), "qwen2_5_vl: mrope_section must have 3 entries"),
        (_set_rope("type", "longrope"), "rope_scaling type 'longrope' is not supported"),
        (_set("head_dim", 32), "qwen2_5_vl: head_dim must be hidden_size / num_attention_heads"),
        (_set("use_sliding_window", True), "use_sliding_window is not supported"),
        (lambda data: data.pop("rope_scaling"), "qwen2_5_vl: the text config has no rope_scaling or rope_parameters"),
    ],
)
def test_unsupported_text_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_qwen2_5_vl(tmp_path / "model", SPEC)
    _edit(directory / "config.json", change)
    assert message in _fails(native_cli, "info", "--model", str(directory), "--backend", "cpu")


def test_sliding_layers_fail_closed(native_cli, tmp_path):
    directory = write_tiny_qwen2_5_vl(tmp_path / "model", replace(SPEC, layout="hf5"))
    _edit(directory / "config.json", lambda data: data["text_config"]["layer_types"].__setitem__(2, "sliding_attention"))
    assert "qwen2_5_vl: sliding-window layers are not supported" in _fails(
        native_cli, "info", "--model", str(directory), "--backend", "cpu")


def test_negative_attention_rows_fail_closed(native_cli, tmp_path):
    directory = write_tiny_qwen2_5_vl(tmp_path / "model", SPEC)
    files = _image_files(tmp_path, _images()[:1])
    assert "--vision-attention-rows must not be negative" in _fails(
        native_cli, "image-features", "--model", str(directory), "--images", files, "--out", str(tmp_path / "x.bin"),
        "--vision-attention-rows", "-1")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set_vision("hidden_act", "gelu"), "qwen2_vl vision: unsupported activation gelu"),
        (_set_vision("mlp_ratio", 2.01), "qwen2_vl vision: embed_dim * mlp_ratio must be a whole number"),
        (_set_vision("hidden_size", 64), "out_hidden_size must be the text model's hidden_size"),
        (_set_vision("deepstack_visual_indexes", [0]), "deepstack features are Qwen3-VL's, not qwen2_vl's"),
        (_set_vision("in_chans", 1), "qwen2_vl vision: images must have 3 channels"),
    ],
)
def test_unsupported_qwen2_vl_vision_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_qwen2_5_vl(tmp_path / "model", QWEN2_SPEC)
    _edit(directory / "config.json", change)
    files = _image_files(tmp_path, _images()[:1])
    assert message in _fails(native_cli, "image-features", "--model", str(directory), "--images", files,
                             "--out", str(tmp_path / "x.bin"))
