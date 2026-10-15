"""Gemma 3 vision: the SigLIP + projector oracle, its fixture guards, and the
native engine against it.

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

from lokahi.fixtures import TinyGemma3Spec, TinyGemma3VisionSpec, write_tiny_gemma3, write_tiny_gemma3_vision
from lokahi.fixtures.gemma3_vision_tiny import image_prompt
from lokahi.oracles import Gemma3VisionReference, SiglipConfig
from lokahi.oracles.siglip import preprocess

SPEC = TinyGemma3VisionSpec()
# The window (3) is shorter than an image (4 tokens): the window still applies
# inside an image block.
NARROW = replace(SPEC, text=replace(SPEC.text, sliding_window=3), seed=101)
# Two images between text runs, as Gemma 3's processor lays them out.
PROMPT = image_prompt(SPEC, [[2, 5, 9], [33, 100, 7], [11, 90]])


def _images(count: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (SPEC.image_size, SPEC.image_size, 3), dtype=np.uint8) for _ in range(count)]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return Gemma3VisionReference.load(write_tiny_gemma3_vision(tmp_path_factory.mktemp("vlm"), SPEC))


def test_siglip_config_reads_the_published_4b_config():
    raw = {
        "image_token_index": 262144, "mm_tokens_per_image": 256, "model_type": "gemma3",
        "vision_config": {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304,
                          "model_type": "siglip_vision_model", "num_attention_heads": 16,
                          "num_hidden_layers": 27, "patch_size": 14, "vision_use_head": False, "skip_vision": True},
    }
    cfg = SiglipConfig.from_dict(raw)
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) == (1152, 27, 16)
    assert (cfg.image_size // cfg.patch_size, cfg.tokens_per_image, cfg.image_token_index) == (64, 256, 262144)
    assert cfg.layer_norm_eps == 1e-6


def test_preprocess_rescales_and_normalizes_to_channels_first():
    rgb = np.array([[[0, 128, 255]]], dtype=np.uint8)
    np.testing.assert_allclose(preprocess(rgb)[:, 0, 0], [-1.0, 128 / 127.5 - 1.0, 1.0], atol=1e-6)


def test_fixture_images_and_blocks_both_matter(reference):
    # Guards the fixture: image features dominate their positions, and the
    # bidirectional image blocks move the logits.
    images = [preprocess(image) for image in _images(2)]
    injected, blocks = reference.placement(PROMPT, images)
    assert blocks == [(4, 8), (13, 17)]
    with_blocks = reference.logits(PROMPT, images)
    causal = reference.language.logits(PROMPT, injected, ())
    text_only = reference.language.logits(PROMPT)
    scale = np.abs(with_blocks).max()
    assert np.abs(with_blocks - causal).max() > 0.05 * scale
    assert np.abs(with_blocks - text_only).max() > 0.5 * scale
    # Features are about as large as the scaled text embeddings.
    features = reference.encode(images[0])
    text = reference.language.w["model.embed_tokens"][PROMPT] * reference.language.config.embed_scale
    assert 0.3 < features.std() / text.std() < 3.0


def test_narrow_variant_has_image_keys_outside_the_window():
    # Guards the narrow variant: an image's first key is further than the
    # window from its last query on sliding layers, where the window wins
    # (Transformers' AND(window, OR(causal, block))).
    assert NARROW.mm_tokens_per_image > NARROW.text.sliding_window
    assert NARROW.text.sliding_window_pattern > 1  # some layers slide


def test_transformers_conv_layout_encodes_the_same(tmp_path):
    images = [preprocess(image) for image in _images(1)]
    mlx = Gemma3VisionReference.load(write_tiny_gemma3_vision(tmp_path / "mlx", SPEC))
    hf = Gemma3VisionReference.load(write_tiny_gemma3_vision(tmp_path / "hf", replace(SPEC, torch_conv_layout=True)))
    np.testing.assert_array_equal(mlx.encode(images[0]), hf.encode(images[0]))
    # Guards the comparison: a transposed image encodes differently.
    assert not np.allclose(mlx.encode(images[0]), mlx.encode(np.ascontiguousarray(images[0].transpose(0, 2, 1))))


def test_placement_rejects_mismatched_and_split_image_runs(reference):
    images = [preprocess(image) for image in _images(2)]
    with pytest.raises(ValueError, match="image token count"):
        reference.placement(PROMPT, images[:1])
    split = list(PROMPT)
    split[5] = 7  # breaks the first image's run
    split[0] = SPEC.image_token_index
    with pytest.raises(ValueError, match="contiguous"):
        reference.placement(split, images)


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


def _image_files(directory: Path, images: list[np.ndarray]) -> str:
    paths = []
    for index, image in enumerate(images):
        path = directory / f"image{index}.rgb"
        path.write_bytes(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
        paths.append(str(path))
    return ",".join(paths)


VARIANTS = {"vlm": SPEC, "vlm-narrow-window": NARROW, "vlm-transformers-conv": replace(SPEC, torch_conv_layout=True)}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def vlm(request, tmp_path_factory):
    directory = write_tiny_gemma3_vision(tmp_path_factory.mktemp(request.param), VARIANTS[request.param])
    images = _images(2, seed=3)
    return directory, Gemma3VisionReference.load(directory), images, _image_files(directory, images)


@pytest.mark.parametrize("backend", _backends())
def test_image_features_match_oracle(native_cli, vlm, backend, tmp_path):
    directory, oracle, images, files = vlm
    out = tmp_path / "features.bin"
    report = _run(native_cli, "image-features", "--model", str(directory), "--backend", backend,
                  "--images", files, "--out", str(out))
    assert report == {"backend": backend, "images": 2, "tokens_per_image": SPEC.mm_tokens_per_image,
                      "width": SPEC.text.hidden_size}
    got = np.fromfile(out, dtype=np.float32).reshape(2, SPEC.mm_tokens_per_image, SPEC.text.hidden_size)
    expected = np.stack([oracle.encode(preprocess(image)) for image in images])
    tolerance = 1e-5 if backend == "cpu" else 1e-3
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)


# 64 patches pooled to 16 tokens: matrix products span several 32-row tiles.
WIDE = replace(SPEC, image_size=112, mm_tokens_per_image=16, seed=102)


@pytest.mark.parametrize("backend", _backends())
def test_image_features_match_oracle_across_tiles(native_cli, backend, tmp_path):
    directory = write_tiny_gemma3_vision(tmp_path / "wide", WIDE)
    oracle = Gemma3VisionReference.load(directory)
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (WIDE.image_size, WIDE.image_size, 3), dtype=np.uint8)
    out = tmp_path / "features.bin"
    _run(native_cli, "image-features", "--model", str(directory), "--backend", backend,
         "--images", _image_files(tmp_path, [image]), "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(WIDE.mm_tokens_per_image, WIDE.text.hidden_size)
    expected = oracle.encode(preprocess(image))
    tolerance = 1e-5 if backend == "cpu" else 1e-3
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)


# Chunks of 5 and 4 would cut an image block (positions 4-7 and 13-16); the
# engine ends those passes where the block begins.
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("chunk", [512, 5, 4])
def test_logits_with_images_match_oracle(native_cli, vlm, backend, chunk, tmp_path):
    directory, oracle, images, files = vlm
    tokens = PROMPT + [44, 45]  # decoded after the prompt's prefill
    out = tmp_path / "logits.bin"
    report = _run(
        native_cli, "logits", "--model", str(directory), "--backend", backend,
        "--tokens", ",".join(map(str, tokens)), "--prefill", str(len(PROMPT)), "--prefill-chunk", str(chunk),
        "--images", files, "--out", str(out),
    )
    assert report["backend"] == backend
    vocab = oracle.language.config.vocab_size
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], vocab)
    injected, blocks = oracle.placement(PROMPT, [preprocess(image) for image in images])
    expected = oracle.language.logits(tokens, injected, blocks)[len(PROMPT) - 1 :]
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))


@pytest.mark.parametrize("backend", _backends())
def test_greedy_with_images_matches_full_recompute(native_cli, vlm, backend):
    directory, oracle, images, files = vlm
    report = _run(
        native_cli, "generate", "--model", str(directory), "--backend", backend,
        "--tokens", ",".join(map(str, PROMPT)), "--images", files, "--max-new", "12", "--no-eos-stop",
    )
    assert report["tokens"] == oracle.greedy(PROMPT, [preprocess(image) for image in images], 12)


def test_vision_checkpoint_runs_text_only_prompts(native_cli, vlm, tmp_path):
    directory, oracle, _, _ = vlm
    out = tmp_path / "logits.bin"
    tokens = [2, 5, 9, 33, 100, 7]
    report = _run(native_cli, "logits", "--model", str(directory), "--backend", "cpu",
                  "--tokens", ",".join(map(str, tokens)), "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], -1)
    expected = oracle.language.logits(tokens)[-1:]
    np.testing.assert_allclose(got, expected, atol=2e-4 * np.abs(expected).max(), rtol=0)
    info = _run(native_cli, "info", "--model", str(directory), "--backend", "cpu")
    assert info["vision"] == {"image_size": SPEC.image_size, "tokens_per_image": SPEC.mm_tokens_per_image,
                              "image_token_id": SPEC.image_token_index}


@pytest.mark.parametrize("backend", _backends())
def test_image_inputs_fail_closed(native_cli, vlm, backend, tmp_path):
    directory, _, images, files = vlm
    model = ["--model", str(directory), "--backend", backend, "--out", str(tmp_path / "x.bin")]
    prompt = ["--tokens", ",".join(map(str, PROMPT))]
    one = files.split(",")[0]
    assert "image tokens for 1 image" in _fails(native_cli, "logits", *model, *prompt, "--images", one)
    # The images' tokens must be in the prefill.
    assert "image tokens for 2 image" in _fails(native_cli, "logits", *model, *prompt, "--prefill", "6",
                                                "--images", files)
    # An image block must fit one prefill pass.
    assert "do not fit one prefill chunk of 3" in _fails(native_cli, "logits", *model, *prompt,
                                                         "--prefill-chunk", "3", "--images", files)
    short = tmp_path / "short.rgb"
    short.write_bytes(b"\0" * 10)
    assert "RGB8 image" in _fails(native_cli, "logits", *model, *prompt, "--images", f"{short},{one}")
    assert "cannot read image" in _fails(native_cli, "logits", *model, *prompt, "--images", f"{one},{tmp_path}/nope")


def test_images_need_a_vision_tower(native_cli, tmp_path):
    directory = write_tiny_gemma3(tmp_path / "text", TinyGemma3Spec())
    image = tmp_path / "image.rgb"
    image.write_bytes(b"\0" * (56 * 56 * 3))
    assert "no vision tower" in _fails(native_cli, "logits", "--model", str(directory), "--tokens", "2,5",
                                       "--images", str(image), "--out", str(tmp_path / "x.bin"))
