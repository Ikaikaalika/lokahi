"""Qwen2.5-VL: the fixture's guards, and the native engine against the oracle.

The oracle is pinned to MLX-VLM and Transformers in
test_mlx_vlm_crosscheck.py. Correctness evidence only: generated fixtures,
never real weights.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lokahi.fixtures import TinyQwen25VLSpec, write_tiny_qwen2_5_vl
from lokahi.oracles import Qwen25VisionConfig, Qwen25VLReference
from lokahi.oracles.qwen2_5_vl import text_config, window_index

SPEC = TinyQwen25VLSpec()
VARIANTS = {
    "qwen2.5-vl-mlx": SPEC,
    "qwen2.5-vl-hf": replace(SPEC, layout="hf"),
    "qwen2.5-vl-hf5": replace(SPEC, layout="hf5"),
}
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


def test_layouts_agree(tmp_path):
    images = _images()
    oracles = {name: Qwen25VLReference.load(write_tiny_qwen2_5_vl(tmp_path / name, spec))
               for name, spec in VARIANTS.items()}
    assert [o.tower for o in oracles.values()] == ["vision_tower", "visual", "model.visual"]
    features = {name: [o.encode(image) for image in images] for name, o in oracles.items()}
    for name in VARIANTS:  # the tower's BF16 weights are the same in every layout
        for got, want in zip(features[name], features["qwen2.5-vl-mlx"]):
            np.testing.assert_array_equal(got, want)
    # Transformers 4's flat config and Transformers 5's text_config describe
    # the same text model (both store the dequantized BF16 weights).
    np.testing.assert_array_equal(oracles["qwen2.5-vl-hf"].logits(PROMPT, images),
                                  oracles["qwen2.5-vl-hf5"].logits(PROMPT, images))


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
