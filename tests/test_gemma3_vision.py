"""Gemma 3 vision: the SigLIP + projector oracle and its fixture guards.

Correctness evidence only: generated fixtures, never real weights.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lokahi.fixtures import TinyGemma3VisionSpec, write_tiny_gemma3_vision
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
