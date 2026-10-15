"""Gemma 4 vision: the oracle, its fixture guards, and the native engine against it.

Correctness evidence only: generated fixtures, never real weights.
"""
from __future__ import annotations

from dataclasses import replace

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
