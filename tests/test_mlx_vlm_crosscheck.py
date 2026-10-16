"""Correctness evidence: the Qwen3.5 vision oracle against Transformers and MLX-VLM.

``Qwen35VisionReference`` is pinned to two independent implementations:
Transformers' ``Qwen2VLImageProcessorPil`` (the NumPy backend) for the resize rule and the patch
layout, and MLX-VLM's ``qwen3_5`` / ``qwen3_5_moe`` models for the vision
tower, the merger, the M-RoPE positions and the full model's logits. They need
``mlx``, ``mlx_vlm`` and ``transformers``: hosted macOS CI runs them on its
virtual GPU, and ``mlx[cpu]`` runs them on Linux.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest

from lokahi.fixtures import TinyQwen35VisionSpec, write_tiny_qwen35_vision
from lokahi.fixtures.qwen35_vision_tiny import VISION_TEXT
from lokahi.oracles import Qwen35VisionReference
from lokahi.oracles.qwen3_5_vision import patchify, rope_positions

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("mlx", "mlx_vlm", "transformers")),
    reason="mlx, mlx_vlm and transformers are required",
)

SHAPES = [(32, 48), (48, 32), (64, 64)]
SPECS = [
    replace(VISION_TEXT, seed=0),
    replace(VISION_TEXT, model_type="qwen3_5_moe", seed=0),
    replace(VISION_TEXT, mlx=False, seed=1),
    replace(VISION_TEXT, model_type="qwen3_5_moe", mlx=False, seed=1),
]


def _image(shape, seed):
    return np.random.default_rng(seed).integers(0, 256, shape + (3,), dtype=np.uint8)


def _float32(tree):
    import mlx.core as mx
    from mlx.utils import tree_map

    return tree_map(lambda value: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value, tree)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    directory = write_tiny_qwen35_vision(tmp_path_factory.mktemp("qwen35-vlm") / "model")
    return directory, Qwen35VisionReference.load(directory)


def test_target_size_matches_transformers_smart_resize(reference):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

    cfg = reference[1].config
    # Sides that round half to even (16 / 32 = 0.5, 48 / 32 = 1.5), areas below
    # min_pixels and above max_pixels, and a side that shrinks below a patch.
    for height, width in [(16, 48), (48, 80), (33, 47), (7, 300), (500, 21), (128, 128), (32, 32), (90, 95),
                          (40, 48), (660, 20), (4000, 20)]:
        expected = smart_resize(height, width, factor=cfg.factor, min_pixels=cfg.min_pixels,
                                max_pixels=cfg.max_pixels)
        assert cfg.target_size(height, width) == tuple(expected), (height, width)


@pytest.mark.parametrize("shape", SHAPES)
def test_patchify_matches_transformers_processor(reference, shape):
    from transformers import Qwen2VLImageProcessorPil

    cfg = reference[1].config
    processor = Qwen2VLImageProcessorPil(
        patch_size=cfg.patch_size, temporal_patch_size=cfg.temporal_patch_size, merge_size=cfg.spatial_merge_size,
        min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, image_mean=[0.5] * 3, image_std=[0.5] * 3,
    )
    image = _image(shape, 1)
    out = processor(images=[image], do_resize=False, return_tensors="np")
    np.testing.assert_allclose(out["pixel_values"], patchify(image, cfg), rtol=0, atol=1e-6)
    assert out["image_grid_thw"].tolist() == [[1, *cfg.grid(*shape)]]


@pytest.mark.parametrize("shape", SHAPES)
def test_vision_tower_matches_mlx_vlm(reference, shape):
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5 import VisionConfig, VisionModel

    directory, oracle = reference
    model = VisionModel(VisionConfig.from_dict(json.loads((directory / "config.json").read_text())["vision_config"]))
    prefix = "vision_tower."
    weights = mx.load(str(directory / "model.safetensors"))
    model.load_weights([(name[len(prefix):], value) for name, value in weights.items() if name.startswith(prefix)],
                       strict=True)
    model.update(_float32(model.parameters()))
    image = _image(shape, 1)
    features, _ = model(mx.array(patchify(image, oracle.config)), mx.array([[1, *oracle.config.grid(*shape)]]))
    expected = oracle.encode(image)
    np.testing.assert_allclose(np.array(features.astype(mx.float32)), expected, rtol=0,
                               atol=1e-5 * np.abs(expected).max())


@pytest.mark.parametrize("text", SPECS, ids=lambda s: f"{s.model_type}-{'mlx' if s.mlx else 'hf'}")
def test_oracle_matches_mlx_vlm_with_an_image(tmp_path, text):
    import mlx.core as mx
    from mlx_vlm.utils import load_model

    directory = write_tiny_qwen35_vision(tmp_path / "model", TinyQwen35VisionSpec(text=text))
    if not text.mlx:
        # As in test_mlx_crosscheck: MLX's sanitize adds the zero-centered
        # norms' 1 in BF16, so the Transformers layout runs on float32 copies.
        path = str(directory / "model.safetensors")
        weights = {name: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
                   for name, value in mx.load(path).items()}
        mx.save_safetensors(path, weights, metadata={"format": "pt"})
    oracle = Qwen35VisionReference.load(directory)
    cfg = oracle.config
    image = _image((32, 48), 2)
    rows, cols = oracle.merged_grid(image)
    prompt = [2, 5, cfg.vision_start_token_id] + [cfg.image_token_id] * (rows * cols) + [cfg.vision_end_token_id,
                                                                                        7, 9, 11, 13, 17]
    model = load_model(directory)
    model.update(_float32(model.parameters()))
    grid = mx.array([[1, *cfg.grid(*image.shape[:2])]])
    # One image only: MLX-VLM 0.4.4's get_rope_index sums the indices of
    # every vision_start token, so it places a second image's positions
    # differently from Transformers, which the oracle follows.
    out = model(mx.array([prompt]), pixel_values=mx.array(patchify(image, cfg)), image_grid_thw=grid)
    logits = np.array((out.logits if hasattr(out, "logits") else out)[0].astype(mx.float32))
    expected = oracle.logits(prompt, [image])
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-3 * np.abs(expected).max())
    np.testing.assert_array_equal(logits.argmax(axis=1), expected.argmax(axis=1))

    positions, delta = model.language_model.get_rope_index(mx.array([prompt]), grid)
    ours, our_delta = rope_positions(prompt, [(rows, cols)], cfg.image_token_id)
    np.testing.assert_array_equal(np.array(positions)[:, 0, :], ours)
    assert int(np.array(delta).reshape(-1)[0]) == our_delta
