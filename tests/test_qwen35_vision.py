"""Qwen3.5 vision: the fixture's guards, and the native engine against the oracle.

The oracle itself is pinned to Transformers' image processor and MLX-VLM in
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

from lokahi.fixtures import TinyQwen35VisionSpec, write_tiny_qwen35_vision
from lokahi.fixtures.qwen35_vision_tiny import VISION_TEXT
from lokahi.oracles import Qwen35VisionReference
from lokahi.oracles.qwen3_5_vision import Qwen35VisionConfig

SPEC = TinyQwen35VisionSpec()
VARIANTS = {
    "qwen3.5-vlm-mlx": SPEC,
    "qwen3.5-vlm-hf": replace(SPEC, text=replace(VISION_TEXT, mlx=False, seed=1), seed=301),
    "qwen3.5-moe-vlm-mlx": replace(SPEC, text=replace(VISION_TEXT, model_type="qwen3_5_moe", seed=2), seed=302),
    "qwen3.5-moe-vlm-hf": replace(SPEC, text=replace(VISION_TEXT, model_type="qwen3_5_moe", mlx=False, seed=3),
                                  seed=303),
}
# Images of 32x64 and 96x32 pixels: merged grids of 2x4 and 6x2 tokens (a
# wide one and a tall one, so rows and columns both vary).
SIZES = [(32, 64), (96, 32)]
GRIDS = [(h // 16, w // 16) for h, w in SIZES]


def _prompt(spec: TinyQwen35VisionSpec = SPEC) -> list[int]:
    start, end, image = spec.vision_start_token_id, spec.vision_end_token_id, spec.image_token_id
    first, second = (rows * cols for rows, cols in GRIDS)
    return [2, 5, start] + [image] * first + [end, 7, 9, start] + [image] * second + [end, 11, 13]


PROMPT = _prompt()


def _images(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in SIZES]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return Qwen35VisionReference.load(write_tiny_qwen35_vision(tmp_path_factory.mktemp("q35v") / "model", SPEC))


def test_config_reads_the_published_0_8b_fields():
    # Qwen/Qwen3.5-0.8B's vision fields and preprocessor_config.json.
    raw = {
        "image_token_id": 248056, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
        "vision_config": {
            "deepstack_visual_indexes": [], "depth": 12, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 768,
            "in_channels": 3, "intermediate_size": 3072, "model_type": "qwen3_5", "num_heads": 12,
            "num_position_embeddings": 2304, "out_hidden_size": 1024, "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    processor = {"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16,
                 "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    cfg = Qwen35VisionConfig.from_dict(raw, processor)
    assert (cfg.depth, cfg.hidden_size, cfg.head_dim, cfg.factor) == (12, 768, 64, 32)
    assert (cfg.min_pixels, cfg.max_pixels) == (65536, 16777216)
    assert cfg.target_size(1000, 700) == (992, 704)
    assert cfg.tokens_for(992, 704) == 31 * 22


def test_fixture_images_and_each_rope_axis_matter(reference):
    # Guards the M-RoPE checks below: every part of the image positions moves
    # the logits well beyond the Metal tolerance (2e-2).
    images = _images()
    injected, positions, _, _ = reference._inputs(PROMPT, images)
    expected = reference.language.logits(PROMPT, injected, positions)
    scale = np.abs(expected).max()

    def moved(other: np.ndarray) -> float:
        return np.abs(reference.language.logits(PROMPT, injected, other) - expected).max() / scale

    sequential = np.broadcast_to(np.arange(len(PROMPT)), positions.shape)
    no_rows, no_columns = positions.copy(), positions.copy()
    no_rows[1], no_columns[2] = positions[0], positions[0]
    no_offset = positions.copy()  # text at its index, as if images took one position per token
    text = np.asarray(PROMPT) != SPEC.image_token_id
    no_offset[:, text] = np.arange(len(PROMPT))[text]
    for name, other in [("sequential", sequential), ("rows", no_rows), ("columns", no_columns),
                        ("swapped", positions[[0, 2, 1]]), ("offset", no_offset)]:
        assert moved(other) > 0.03, name
    text_only = reference.language.logits(PROMPT)
    assert np.abs(expected - text_only).max() > 0.3 * scale
    features = reference.encode(images[0])
    text = reference.language.w["model.embed_tokens"][PROMPT]
    assert 0.3 < features.std() / text.std() < 30.0
    assert len(set(reference.greedy(PROMPT, images, 10))) > 5


# ---------------------------------------------------------------------------
# The native engine against the oracle.
# ---------------------------------------------------------------------------

def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _run(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], check=True, capture_output=True, text=True)
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
    directory = write_tiny_qwen35_vision(tmp_path_factory.mktemp(request.param) / "model", VARIANTS[request.param])
    images = _images(seed=3)
    return directory, Qwen35VisionReference.load(directory), images, _image_files(directory, images)


# 7 query rows per Metal attention pass divides none of the images' patch
# counts, so the last pass of each is partial (a no-op on the CPU).
@pytest.mark.parametrize("attention_rows", ["0", "7"])
@pytest.mark.parametrize("backend", _backends())
def test_image_features_match_oracle(native_cli, vlm, backend, attention_rows, tmp_path):
    directory, oracle, images, files = vlm
    out = tmp_path / "features.bin"
    report = _run(native_cli, "image-features", "--model", str(directory), "--backend", backend,
                  "--images", files, "--out", str(out), "--vision-attention-rows", attention_rows)
    tokens = [rows * cols for rows, cols in GRIDS]
    assert report == {"backend": backend, "tokens": tokens, "width": oracle.language.config.hidden_size}
    got = np.fromfile(out, dtype=np.float32).reshape(sum(tokens), -1)
    expected = np.concatenate([oracle.encode(image) for image in images])
    tolerance = 1e-5 if backend == "cpu" else 1e-3
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)


# Chunks of 3 split both images: the images attend causally, so a pass may
# end inside one, and its later tokens keep their grid positions.
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
    # Decoded tokens take M-RoPE positions index + delta on every axis.
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


def test_image_geometry_follows_the_processor(native_cli, tmp_path):
    directory = write_tiny_qwen35_vision(tmp_path / "model", SPEC)
    cfg = Qwen35VisionReference.load(directory).config
    # Sides that round half to even (40 / 16 = 2.5 to 2, 24 / 16 to 2), areas
    # below min_pixels (1024) and above max_pixels (8192), extreme aspects, and
    # a side that shrinks below one patch (20x660 to 16x512).
    for width, height in [(48, 32), (48, 40), (40, 48), (8, 40), (40, 24), (20, 20), (300, 300), (7, 300),
                          (1000, 6), (64, 128), (20, 660)]:
        th, tw = cfg.target_size(height, width)
        report = _run(native_cli, "image-geometry", "--model", str(directory), "--width", str(width),
                      "--height", str(height))
        assert report == {"width": tw, "height": th, "tokens": cfg.tokens_for(th, tw),
                          "image_token_id": SPEC.image_token_id}
    info = _run(native_cli, "info", "--model", str(directory), "--backend", "cpu")
    assert info["vision"] == {"image_size": 0, "image_token_id": SPEC.image_token_id}


def _edit_json(path: Path, change: dict, remove: tuple = ()) -> None:
    data = json.loads(path.read_text())
    data.update(change)
    for key in remove:
        data.pop(key)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("file", "change", "remove", "message"),
    [
        ("config.json", {"vision_config": {**SPEC.vision_config(), "deepstack_visual_indexes": [1]}}, (),
         "deepstack features are Qwen3-VL's, not qwen3_5's"),
        ("config.json", {"vision_config": {**SPEC.vision_config(), "hidden_act": "gelu"}}, (),
         "unsupported activation gelu"),
        ("config.json", {"vision_config": {**SPEC.vision_config(), "out_hidden_size": 64}}, (),
         "out_hidden_size must be the text model's hidden_size"),
        ("config.json", {"vision_config": {**SPEC.vision_config(), "num_position_embeddings": 15}}, (),
         "num_position_embeddings must be a square"),
        ("config.json", {"vision_config": {**SPEC.vision_config(), "num_heads": 32}}, (),
         "the head size must be a multiple of 4"),
        ("config.json", {"vision_config": {**SPEC.vision_config(), "depth": 1}}, (),
         "more blocks than depth"),
        ("preprocessor_config.json", {"merge_size": 4}, (), "patch geometry does not match"),
        ("preprocessor_config.json", {}, ("size",), "gives no shortest_edge"),
        ("preprocessor_config.json", {"size": {"shortest_edge": 9000, "longest_edge": 8192}}, (), "bad pixel bounds"),
    ],
)
def test_unsupported_vision_configurations_fail_closed(native_cli, tmp_path, file, change, remove, message):
    directory = write_tiny_qwen35_vision(tmp_path / "model", SPEC)
    _edit_json(directory / file, change, remove)
    files = _image_files(tmp_path, _images()[:1])
    assert message in _fails(native_cli, "image-features", "--model", str(directory), "--images", files,
                             "--out", str(tmp_path / "x.bin"))


def test_image_inputs_fail_closed(native_cli, tmp_path):
    directory = write_tiny_qwen35_vision(tmp_path / "model", SPEC)
    model = ["--model", str(directory), "--backend", "cpu", "--out", str(tmp_path / "x.bin")]
    odd = write_ppm(tmp_path / "odd.ppm", np.zeros((32, 40, 3), dtype=np.uint8))
    assert "multiples of 16" in _fails(native_cli, "image-features", *model, "--images", str(odd))
    small = write_ppm(tmp_path / "small.ppm", np.zeros((16, 32, 3), dtype=np.uint8))  # 512 pixels < 1024
    assert "outside the processor's 1024 to 8192 pixels" in _fails(native_cli, "image-features", *model,
                                                                    "--images", str(small))
    # smart_resize takes 20x4000 to 16x1280, past max_pixels: refused.
    assert "outside the processor's" in _fails(native_cli, "image-geometry", "--model", str(directory),
                                               "--width", "20", "--height", "4000")
    files = _image_files(tmp_path, _images())
    prompt = ["--tokens", ",".join(map(str, PROMPT))]
    assert "image tokens for 1 image" in _fails(native_cli, "logits", *model, *prompt, "--images",
                                                files.split(",")[0])
    (directory / "preprocessor_config.json").unlink()
    assert "has no preprocessor_config.json" in _fails(native_cli, "image-features", *model, "--images", files)
