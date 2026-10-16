"""Correctness evidence: the Qwen3.5, Qwen3-VL and Qwen2.5-VL oracles against Transformers and MLX-VLM.

``Qwen35VisionReference`` is pinned to two independent implementations:
Transformers' ``Qwen2VLImageProcessorPil`` (the NumPy backend) for the resize rule and the patch
layout, and MLX-VLM's ``qwen3_5`` / ``qwen3_5_moe`` models for the vision
tower, the merger, the M-RoPE positions and the full model's logits, and MLX-VLM's
``qwen3_vl`` / ``qwen3_vl_moe`` for deepstack and the Qwen 3 text models.
``Qwen25VLReference`` is pinned to MLX-VLM's ``qwen2_5_vl`` (the window order,
the tower, the positions and the logits) and to Transformers' processor and
``Qwen2_5_VLConfig``. They need
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


# ---------------------------------------------------------------------------
# Qwen3-VL: Qwen3.5's tower with deepstack, in front of Qwen 3 or Qwen 3 MoE.
#
# MLX-VLM 0.4.4 runs the pieces correctly but drops deepstack when they are
# chained: get_input_embeddings keeps the return value of mx.eval (None)
# as the deepstack features, and Qwen3-VL-MoE's _deepstack_process returns
# the unmodified hidden states. Its Qwen3-VL-MoE tower also swaps the two
# GELUs (Transformers uses tanh GELU in the blocks and exact GELU in the
# mergers, for both models). So the dense model is checked piece by piece
# with deepstack passed in, and the MoE text model on the oracle's features.
# ---------------------------------------------------------------------------

def _qwen3_vl(tmp_path, spec):
    import mlx.core as mx
    from mlx_vlm.utils import load_model

    from lokahi.fixtures import write_tiny_qwen3_vl
    from lokahi.oracles import Qwen3VLReference

    directory = write_tiny_qwen3_vl(tmp_path / "model", spec)
    if not spec.mlx:  # float32 copies, as above
        path = str(directory / "model.safetensors")
        weights = {name: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
                   for name, value in mx.load(path).items()}
        mx.save_safetensors(path, weights, metadata={"format": "pt"})
    oracle = Qwen3VLReference.load(directory)
    model = load_model(directory)
    model.update(_float32(model.parameters()))
    image = _image((32, 64), 2)
    rows, cols = oracle.merged_grid(image)
    cfg = oracle.config
    prompt = [2, 5, cfg.vision_start_token_id] + [cfg.image_token_id] * (rows * cols) + [cfg.vision_end_token_id,
                                                                                        7, 9, 11, 13, 17]
    return oracle, model, image, prompt, mx.array([[1, *cfg.grid(*image.shape[:2])]])


def _language_logits(model, prompt, grid, features, deepstack, image_token_id):
    import mlx.core as mx

    ids = mx.array([prompt])
    language = model.language_model
    language._position_ids = language._rope_deltas = None
    embeds = language.model.embed_tokens(ids)
    embeds, mask = model.merge_input_ids_with_image_features(features, embeds, ids, image_token_id, -1)
    out = language(ids, inputs_embeds=embeds, visual_pos_masks=mask[..., 0], deepstack_visual_embeds=deepstack,
                   image_grid_thw=grid)
    return np.array((out.logits if hasattr(out, "logits") else out)[0].astype(mx.float32))


@pytest.mark.parametrize("mlx_layout", [True, False], ids=["mlx", "hf"])
def test_qwen3_vl_matches_mlx_vlm(tmp_path, mlx_layout):
    import mlx.core as mx

    from lokahi.fixtures import TinyQwen3VLSpec

    oracle, model, image, prompt, grid = _qwen3_vl(tmp_path, TinyQwen3VLSpec(mlx=mlx_layout))
    features, deepstack = model.vision_tower(mx.array(patchify(image, oracle.config)), grid)
    expected, expected_deepstack = oracle.encode_all(image)
    assert len(deepstack) == len(expected_deepstack) == 2
    for got, want in zip([features, *deepstack], [expected, *expected_deepstack]):
        np.testing.assert_allclose(np.array(got), want, rtol=0, atol=1e-5 * np.abs(want).max())

    logits = _language_logits(model, prompt, grid, features, deepstack, oracle.config.image_token_id)
    expected = oracle.logits(prompt, [image])
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-3 * np.abs(expected).max())
    np.testing.assert_array_equal(logits.argmax(axis=1), expected.argmax(axis=1))
    positions, delta = model.language_model.get_rope_index(mx.array([prompt]), grid)
    rows, cols = oracle.merged_grid(image)
    ours, our_delta = rope_positions(prompt, [(rows, cols)], oracle.config.image_token_id)
    np.testing.assert_array_equal(np.array(positions)[:, 0, :], ours)
    assert int(np.array(delta).reshape(-1)[0]) == our_delta


def test_qwen3_vl_moe_text_model_matches_mlx_vlm(tmp_path):
    import mlx.core as mx

    from lokahi.fixtures import TinyQwen3VLSpec
    from lokahi.fixtures.qwen3_vl_tiny import VL_MOE_TEXT

    oracle, model, image, prompt, grid = _qwen3_vl(tmp_path, TinyQwen3VLSpec(text=VL_MOE_TEXT))
    features, _ = oracle.encode_all(image)
    logits = _language_logits(model, prompt, grid, mx.array(features), None, oracle.config.image_token_id)
    injected, positions, _, _ = oracle._inputs(prompt, [image])
    expected = oracle.language.logits(prompt, injected, positions)
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-3 * np.abs(expected).max())
    np.testing.assert_array_equal(logits.argmax(axis=1), expected.argmax(axis=1))


# ---------------------------------------------------------------------------
# Qwen2.5-VL: a window-attention tower in front of Qwen 2 with chunked M-RoPE.
# MLX-VLM 0.4.4 cannot read Transformers 5's text_config (its from_dict
# copies the top-level fields over it), so that layout is pinned to
# Transformers' own re-save of the flat config instead.
# ---------------------------------------------------------------------------

def _qwen2_5_vl(directory, layout="mlx"):
    import mlx.core as mx

    from lokahi.fixtures import TinyQwen25VLSpec, write_tiny_qwen2_5_vl
    from lokahi.oracles import Qwen25VLReference

    directory = write_tiny_qwen2_5_vl(directory, TinyQwen25VLSpec(layout=layout))
    if layout != "mlx":  # float32 copies, as above
        path = str(directory / "model.safetensors")
        weights = {name: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
                   for name, value in mx.load(path).items()}
        mx.save_safetensors(path, weights, metadata={"format": "pt"})
    return directory, Qwen25VLReference.load(directory)


def test_qwen2_5_vl_target_size_matches_transformers_smart_resize():
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

    from lokahi.oracles import Qwen25VisionConfig
    from test_qwen2_5_vl import CONFIG_3B, PREPROCESSOR_3B

    cfg = Qwen25VisionConfig.from_dict(CONFIG_3B, PREPROCESSOR_3B)  # factor 28, published pixel bounds
    for height, width in [(14, 42), (42, 70), (33, 47), (7, 300), (500, 21), (1080, 1920), (4000, 3000),
                          (28, 28), (56, 5000), (3000, 20)]:
        expected = smart_resize(height, width, factor=cfg.factor, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        assert cfg.target_size(height, width) == tuple(expected), (height, width)


@pytest.mark.parametrize("shape", [(48, 80), (64, 32)])
def test_qwen2_5_vl_patchify_matches_transformers_processor(tmp_path, shape):
    from transformers import Qwen2VLImageProcessorPil

    _, oracle = _qwen2_5_vl(tmp_path / "model")
    cfg = oracle.config
    processor = Qwen2VLImageProcessorPil(  # CLIP's mean and std, the processor's defaults
        patch_size=cfg.patch_size, temporal_patch_size=cfg.temporal_patch_size, merge_size=cfg.spatial_merge_size,
        min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels,
    )
    image = _image(shape, 1)
    out = processor(images=[image], do_resize=False, return_tensors="np")
    np.testing.assert_allclose(out["pixel_values"], patchify(image, cfg), rtol=0, atol=1e-5)


def test_qwen2_5_vl_window_index_matches_mlx_vlm(tmp_path):
    import mlx.core as mx
    from mlx_vlm.models.qwen2_5_vl import VisionConfig, VisionModel

    from lokahi.oracles.qwen2_5_vl import window_index

    directory, oracle = _qwen2_5_vl(tmp_path / "model")
    model = VisionModel(VisionConfig.from_dict(json.loads((directory / "config.json").read_text())["vision_config"]))
    for rows, cols in [(3, 5), (4, 2), (2, 2), (1, 7), (5, 5), (6, 3)]:
        grid = mx.array([[1, rows * 2, cols * 2]])
        index, cu = model.get_window_index(grid)
        order, lengths = window_index(rows, cols, oracle.config.window)
        assert np.array(index).tolist() == order.tolist(), (rows, cols)
        bounds = sorted(set(np.array(cu).tolist()))
        assert np.diff(bounds).tolist() == [length * 4 for length in lengths], (rows, cols)


@pytest.mark.parametrize("shape", [(48, 80), (64, 32), (64, 64), (32, 96), (80, 48)])
def test_qwen2_5_vl_tower_matches_mlx_vlm(tmp_path, shape):
    import mlx.core as mx
    from mlx_vlm.models.qwen2_5_vl import VisionConfig, VisionModel

    directory, oracle = _qwen2_5_vl(tmp_path / "model")
    model = VisionModel(VisionConfig.from_dict(json.loads((directory / "config.json").read_text())["vision_config"]))
    prefix = "vision_tower."
    weights = mx.load(str(directory / "model.safetensors"))
    model.load_weights([(name[len(prefix):], value) for name, value in weights.items() if name.startswith(prefix)],
                       strict=True)
    model.update(_float32(model.parameters()))
    image = _image(shape, 1)
    features = model(mx.array(patchify(image, oracle.config)), mx.array([[1, *oracle.config.grid(*shape)]]))
    expected = oracle.encode(image)
    np.testing.assert_allclose(np.array(features), expected, rtol=0, atol=1e-5 * np.abs(expected).max())


@pytest.mark.parametrize("layout", ["mlx", "hf"])
def test_qwen2_5_vl_matches_mlx_vlm_with_an_image(tmp_path, layout):
    import mlx.core as mx
    from mlx_vlm.utils import load_model

    directory, oracle = _qwen2_5_vl(tmp_path / "model", layout)
    cfg = oracle.config
    image = _image((48, 80), 2)
    rows, cols = oracle.merged_grid(image)
    prompt = [2, 5, cfg.vision_start_token_id] + [cfg.image_token_id] * (rows * cols) + [cfg.vision_end_token_id,
                                                                                        7, 9, 11, 13, 17]
    model = load_model(directory)
    model.update(_float32(model.parameters()))
    grid = mx.array([[1, *cfg.grid(*image.shape[:2])]])
    # One image only: MLX-VLM 0.4.4's get_rope_index sums the indices of
    # every vision_start token here too.
    out = model(mx.array([prompt]), pixel_values=mx.array(patchify(image, cfg)), image_grid_thw=grid)
    logits = np.array((out.logits if hasattr(out, "logits") else out)[0].astype(mx.float32))
    expected = oracle.logits(prompt, [image])
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-3 * np.abs(expected).max())
    np.testing.assert_array_equal(logits.argmax(axis=1), expected.argmax(axis=1))

    positions, delta = model.language_model.get_rope_index(mx.array([prompt]), grid)
    ours, our_delta = rope_positions(prompt, [(rows, cols)], cfg.image_token_id)
    np.testing.assert_array_equal(np.array(positions)[:, 0, :], ours)
    assert int(np.array(delta).reshape(-1)[0]) == our_delta


def test_qwen2_5_vl_transformers_5_layout_is_what_transformers_saves(tmp_path):
    from transformers import Qwen2_5_VLConfig

    from lokahi.fixtures import TinyQwen25VLSpec, write_tiny_qwen2_5_vl

    flat = json.loads((write_tiny_qwen2_5_vl(tmp_path / "hf", TinyQwen25VLSpec(layout="hf")) / "config.json").read_text())
    nested = json.loads((write_tiny_qwen2_5_vl(tmp_path / "hf5", TinyQwen25VLSpec(layout="hf5")) / "config.json")
                        .read_text())
    saved = Qwen2_5_VLConfig.from_dict(flat).to_diff_dict()
    for key in ("model_type", "image_token_id", "vision_start_token_id", "vision_end_token_id", "tie_word_embeddings"):
        assert nested[key] == saved[key], key
    for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "rms_norm_eps", "vocab_size", "rope_parameters", "use_sliding_window",
                "layer_types", "hidden_act"):
        assert nested["text_config"][key] == saved["text_config"][key], key
    for key in ("model_type", "depth", "hidden_size", "intermediate_size", "num_heads", "in_channels", "patch_size",
                "spatial_merge_size", "temporal_patch_size", "window_size", "out_hidden_size",
                "fullatt_block_indexes", "rope_parameters", "hidden_act"):
        assert nested["vision_config"][key] == saved["vision_config"][key], key
