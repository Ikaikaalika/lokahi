"""Gemma 4 vision: the oracle, its fixture guards, and the native engine against it.

Correctness evidence only: generated fixtures, never real weights.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures.gemma4_vision_tiny import TEXT, TinyGemma4VisionSpec, image_prompt, write_tiny_gemma4_vision
from lokahi.oracles.gemma4_vision import Gemma4VisionConfig, Gemma4VisionReference, patchify

SPEC = TinyGemma4VisionSpec()
# A sliding window (5) shorter than the second image (8 tokens).
NARROW = replace(SPEC, text=replace(TEXT, sliding_window=5, seed=1), seed=201)
# use_bidirectional_attention unset: image tokens attend causally, as in E2B.
CAUSAL = replace(SPEC, text=replace(TEXT, extra={}, seed=2), seed=202)
# Images of 24x36 and 48x24 pixels: 2x3 and 4x2 pooled tokens.
SIZES = [(24, 36), (48, 24)]
TOKENS = [(h // 12) * (w // 12) for h, w in SIZES]
PROMPT = image_prompt(SPEC, [[2, 5, 9], [33, 100, 7], [11, 90]], TOKENS)

# The vision fields of mlx-community/gemma-4-31b-it-4bit's config.
CONFIG_31B = {
    "image_token_id": 258880, "boi_token_id": 255999, "eoi_token_id": 258882, "vision_soft_tokens_per_image": 280,
    "vision_config": {
        "default_output_length": 280, "head_dim": 72, "hidden_size": 1152, "intermediate_size": 4304,
        "model_type": "gemma4_vision", "num_attention_heads": 16, "num_hidden_layers": 27, "num_key_value_heads": 16,
        "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 10240,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}, "rms_norm_eps": 1e-06,
        "standardize": True, "use_clipped_linears": False,
    },
}


def _images(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in SIZES]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return Gemma4VisionReference.load(write_tiny_gemma4_vision(tmp_path_factory.mktemp("g4v"), SPEC))


def test_config_reads_the_published_31b_fields():
    cfg = Gemma4VisionConfig.from_dict(CONFIG_31B)
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.head_dim, cfg.max_patches) == (1152, 27, 72, 2520)
    assert cfg.standardize and cfg.rope_theta == 100.0 and cfg.image_token_id == 258880
    # The processor's aspect-preserving sizes: sides in multiples of 48 within 2520 patches.
    assert cfg.target_size(1000, 700) == (960, 672)
    assert cfg.target_size(768, 768) == (768, 768)
    assert cfg.target_size(10, 1000) == (48, 8016)
    assert cfg.target_size(1, 1000) == (48, 13440)  # a side rounds to zero: one row, capped
    assert cfg.tokens_for(960, 672) == 280
    with pytest.raises(ValueError):
        cfg.tokens_for(960, 670)
    with pytest.raises(ValueError):
        cfg.tokens_for(1008, 672)  # 2646 patches
    with pytest.raises(NotImplementedError):
        Gemma4VisionConfig.from_dict({**CONFIG_31B, "vision_config": {**CONFIG_31B["vision_config"],
                                                                      "use_clipped_linears": True}})


def test_patchify_orders_patches_row_major_with_xy_positions():
    rgb = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    patches, positions = patchify(rgb, 4)
    assert patches.shape == (6, 48) and positions.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    # Patch 4 is row 1, column 1; flattened as (row, column, channel).
    np.testing.assert_allclose(patches[4], rgb[4:8, 4:8, :].reshape(-1) / np.float32(255.0), rtol=1e-6)


def test_fixture_images_and_blocks_both_matter(reference):
    images = _images()
    injected, blocks = reference.placement(PROMPT, images)
    assert blocks == [(4, 10), (15, 23)] and [len(range(*b)) for b in blocks] == TOKENS
    with_blocks = reference.logits(PROMPT, images)
    causal = reference.language.logits(PROMPT, injected, ())
    text_only = reference.language.logits(PROMPT)
    scale = np.abs(with_blocks).max()
    assert np.abs(with_blocks - causal).max() > 0.05 * scale
    assert np.abs(with_blocks - text_only).max() > 0.5 * scale
    features = reference.encode(images[0])
    text = reference.language.w["model.embed_tokens"][PROMPT] * np.sqrt(SPEC.text.hidden_size)
    assert 0.3 < features.std() / text.std() < 3.0
    # Greedy continuations vary (an untied head), so they discriminate.
    assert len(set(reference.greedy(PROMPT, images, 10))) > 5


def test_fixture_variants_guard_the_mask_rules(tmp_path):
    narrow = Gemma4VisionReference.load(write_tiny_gemma4_vision(tmp_path / "narrow", NARROW))
    assert max(TOKENS) > NARROW.text.sliding_window  # the window cuts into an image block
    causal = Gemma4VisionReference.load(write_tiny_gemma4_vision(tmp_path / "causal", CAUSAL))
    images = _images()
    injected, blocks = causal.placement(PROMPT, images)
    # Without "vision" bidirectionality the blocks change nothing.
    np.testing.assert_array_equal(causal.language.logits(PROMPT, injected, blocks),
                                  causal.language.logits(PROMPT, injected, ()))
    assert narrow.language.config.bidirectional_vision and not causal.language.config.bidirectional_vision


def test_placement_rejects_mismatched_images(reference):
    images = _images()
    with pytest.raises(ValueError, match="image token count"):
        reference.placement(PROMPT, images[:1])
    with pytest.raises(ValueError, match="not an accepted"):
        reference.encode(np.zeros((20, 36, 3), dtype=np.uint8))


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
    path.write_bytes(f"P6\n# lokahi test image\n{width} {height}\n255\n".encode() + np.ascontiguousarray(rgb).tobytes())
    return path


def _image_files(directory: Path, images: list[np.ndarray]) -> str:
    return ",".join(str(write_ppm(directory / f"image{i}.ppm", image)) for i, image in enumerate(images))


VARIANTS = {"g4v": SPEC, "g4v-narrow-window": NARROW, "g4v-causal-images": CAUSAL}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def vlm(request, tmp_path_factory):
    directory = write_tiny_gemma4_vision(tmp_path_factory.mktemp(request.param), VARIANTS[request.param])
    images = _images(seed=3)
    return directory, Gemma4VisionReference.load(directory), images, _image_files(directory, images)


@pytest.mark.parametrize("backend", _backends())
def test_image_features_match_oracle(native_cli, vlm, backend, tmp_path):
    directory, oracle, images, files = vlm
    out = tmp_path / "features.bin"
    report = _run(native_cli, "image-features", "--model", str(directory), "--backend", backend,
                  "--images", files, "--out", str(out))
    assert report == {"backend": backend, "tokens": TOKENS, "width": SPEC.text.hidden_size}
    got = np.fromfile(out, dtype=np.float32).reshape(sum(TOKENS), SPEC.text.hidden_size)
    expected = np.concatenate([oracle.encode(image) for image in images])
    tolerance = 1e-5 if backend == "cpu" else 1e-3
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)


# Chunks of 9 and 8 would cut an image (positions 4-9 and 15-22); passes end
# where an image begins.
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("chunk", [512, 9, 8])
def test_logits_with_images_match_oracle(native_cli, vlm, backend, chunk, tmp_path):
    directory, oracle, images, files = vlm
    tokens = PROMPT + [44, 45]
    out = tmp_path / "logits.bin"
    report = _run(
        native_cli, "logits", "--model", str(directory), "--backend", backend,
        "--tokens", ",".join(map(str, tokens)), "--prefill", str(len(PROMPT)), "--prefill-chunk", str(chunk),
        "--images", files, "--out", str(out),
    )
    vocab = SPEC.text.vocab_size
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], vocab)
    injected, blocks = oracle.placement(PROMPT, images)
    expected = oracle.language.logits(tokens, injected, blocks)[len(PROMPT) - 1 :]
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))


@pytest.mark.parametrize("backend", _backends())
def test_greedy_with_images_matches_full_recompute(native_cli, vlm, backend):
    directory, oracle, images, files = vlm
    report = _run(
        native_cli, "generate", "--model", str(directory), "--backend", backend,
        "--tokens", ",".join(map(str, PROMPT)), "--images", files, "--max-new", "10", "--no-eos-stop",
    )
    assert report["tokens"] == oracle.greedy(PROMPT, images, 10)


def test_image_geometry_follows_the_processor(native_cli, tmp_path):
    directory = write_tiny_gemma4_vision(tmp_path / "model", SPEC)
    cfg = Gemma4VisionConfig.from_dict(json.loads((directory / "config.json").read_text()))
    for width, height in [(700, 1000), (1000, 10), (5, 5000), (36, 24)]:
        th, tw = cfg.target_size(height, width)
        report = _run(native_cli, "image-geometry", "--model", str(directory), "--width", str(width),
                      "--height", str(height))
        assert report == {"width": tw, "height": th, "tokens": cfg.tokens_for(th, tw),
                          "image_token_id": SPEC.image_token_id}
    info = _run(native_cli, "info", "--model", str(directory), "--backend", "cpu")
    assert info["vision"] == {"image_size": 0, "image_token_id": SPEC.image_token_id}


def test_image_inputs_fail_closed(native_cli, tmp_path):
    directory = write_tiny_gemma4_vision(tmp_path / "model", SPEC)
    model = ["--model", str(directory), "--backend", "cpu", "--out", str(tmp_path / "x.bin")]
    prompt = ["--tokens", ",".join(map(str, PROMPT))]
    files = _image_files(tmp_path, _images())
    odd = write_ppm(tmp_path / "odd.ppm", np.zeros((24, 30, 3), dtype=np.uint8))
    assert "multiples of 12" in _fails(native_cli, "image-features", *model, "--images", str(odd))
    big = write_ppm(tmp_path / "big.ppm", np.zeros((36, 48, 3), dtype=np.uint8))  # 108 patches > 72
    assert "8-token budget" in _fails(native_cli, "image-features", *model, "--images", str(big))
    short_table = write_tiny_gemma4_vision(tmp_path / "short-table", replace(SPEC, position_embedding_size=8))
    wide = write_ppm(tmp_path / "wide.ppm", np.zeros((12, 96, 3), dtype=np.uint8))  # 24 columns > 8 positions
    assert "wider than the position table" in _fails(native_cli, "image-features", "--model", str(short_table),
                                                     "--out", str(tmp_path / "w.bin"), "--images", str(wide))
    raw = tmp_path / "raw.rgb"
    raw.write_bytes(b"\0" * (24 * 36 * 3))
    assert "not a PPM" in _fails(native_cli, "image-features", *model, "--images", str(raw))
    assert "image tokens for 1 image" in _fails(native_cli, "logits", *model, *prompt, "--images",
                                                files.split(",")[0])
    # Transformers feeds per-layer inputs at image positions from the pad token.
    per_layer = write_tiny_gemma4_vision(tmp_path / "per-layer",
                                         replace(SPEC, text=replace(TEXT, hidden_size_per_layer_input=32)))
    assert "per-layer inputs are not supported yet" in _fails(
        native_cli, "logits", "--model", str(per_layer), "--backend", "cpu", *prompt, "--images", files,
        "--out", str(tmp_path / "y.bin"))
