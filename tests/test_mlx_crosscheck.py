"""Correctness evidence: the NumPy oracle against MLX and MLX-LM.

These tests pin the oracles' semantics to an independent implementation:
``mx.dequantize`` for the packed affine layout, and MLX-LM's ``gemma3_text``,
``gemma4``, ``llama``, ``qwen2``, ``qwen3`` and ``qwen3_moe`` models for the full decoders. They need
``mlx`` and ``mlx_lm`` and therefore run on Apple Silicon only.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lokahi.fixtures import (
    TinyDenseSpec,
    TinyGemma3Spec,
    TinyGemma4Spec,
    write_tiny_dense,
    write_tiny_gemma3,
    write_tiny_gemma4,
)
from lokahi.formats import QuantSpec, dequantize_affine, pack_affine
from lokahi.oracles import Gemma3Reference, Gemma4Reference, LlamaFamilyReference

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None,
    reason="mlx and mlx_lm are required",
)

TOKENS = [2, 5, 9, 33, 100, 7, 7, 7, 400, 3, 2, 11, 90, 91, 92, 93, 17, 18, 19, 20]


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_affine_layout_matches_mx_dequantize(bits):
    import mlx.core as mx

    rng = np.random.default_rng(bits)
    rows, cols, group = 12, 256, 64
    codes = rng.integers(0, 1 << bits, size=(rows, cols), dtype=np.uint32)
    scales = rng.uniform(0.01, 0.05, size=(rows, cols // group)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.1, size=(rows, cols // group)).astype(np.float32)
    words = pack_affine(codes, bits)
    ours = dequantize_affine(words, scales, biases, QuantSpec(group, bits))
    theirs = mx.dequantize(
        mx.array(words), mx.array(scales), mx.array(biases), group_size=group, bits=bits
    )
    np.testing.assert_allclose(np.array(theirs), ours, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "spec",
    [
        TinyGemma3Spec(),
        TinyGemma3Spec(
            hidden_size=256,
            num_key_value_heads=1,
            head_dim=256,
            tie_word_embeddings=True,
            rope_linear_factor=8.0,
            group_size=64,
            eight_bit_mlp_layers=(),
            seed=1,
        ),
    ],
    ids=["untied-hd64-mixed-bits", "tied-hd256-rope-scaled"],
)
def test_oracle_matches_mlx_lm_gemma3(tmp_path, spec):
    directory = write_tiny_gemma3(tmp_path / "model", spec)
    _assert_mlx_lm_matches(directory, Gemma3Reference.load(directory).logits(TOKENS))


@pytest.mark.parametrize(
    "spec",
    [
        TinyDenseSpec(
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            hidden_size=256,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64,
            },
            seed=3,
        ),
        TinyDenseSpec(model_type="qwen3", head_dim=128, rope_theta=1e6, seed=4),
        TinyDenseSpec(model_type="qwen2", hidden_size=448, num_attention_heads=7, num_key_value_heads=1,
                      group_size=64, rope_theta=1e6, seed=5),
        # MLX-LM runs Mistral through its Llama model, which has no sliding
        # window, so this variant leaves the window unset.
        TinyDenseSpec(model_type="mistral", tie_word_embeddings=False, rope_theta=1e6, seed=6),
        TinyDenseSpec(model_type="qwen3_moe", head_dim=64, mlp_only_layers=(1,), seed=9),
    ],
    ids=["llama3-rope-llama3", "qwen3-qk-norm", "qwen2-bias-group7", "mistral-untied", "qwen3-moe-top2-of-8"],
)
def test_oracle_matches_mlx_lm_llama_family(tmp_path, spec):
    directory = write_tiny_dense(tmp_path / "model", spec)
    _assert_mlx_lm_matches(directory, LlamaFamilyReference.load(directory).logits(TOKENS))


@pytest.mark.parametrize(
    "spec",
    [
        TinyGemma4Spec(),
        TinyGemma4Spec(
            num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True, num_kv_shared_layers=0,
            hidden_size_per_layer_input=0, use_double_wide_mlp=False, tie_word_embeddings=False, nested=False,
            sliding_window=5, global_partial_rotary_factor=0.5, group_size=64, bits=8, seed=7,
        ),
    ],
    ids=["e2b-shape", "k-eq-v-untied-flat"],
)
def test_oracle_matches_mlx_lm_gemma4(tmp_path, spec):
    directory = write_tiny_gemma4(tmp_path / "model", spec)
    # In float32, MLX-LM's scalar constants are not rounded to BF16.
    _assert_mlx_lm_matches(directory, Gemma4Reference.load(directory, bf16_constants=False).logits(TOKENS))


def _assert_mlx_lm_matches(directory, expected):
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.utils import load_model

    model, _ = load_model(directory)
    # Run MLX in float32 so the comparison isolates semantics from BF16 rounding.
    model.update(
        tree_map(
            lambda value: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value,
            model.parameters(),
        )
    )
    logits = np.array(model(mx.array([TOKENS]))[0].astype(mx.float32))
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-3 * np.abs(expected).max())
    np.testing.assert_array_equal(logits.argmax(axis=1), expected.argmax(axis=1))
