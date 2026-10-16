"""Correctness evidence: the NumPy oracle against MLX and MLX-LM.

These tests pin the oracles' semantics to an independent implementation:
``mx.dequantize`` for the packed affine layout, and MLX-LM's ``gemma3_text``,
``gemma4``, ``llama``, ``qwen2``, ``qwen3``, ``qwen3_moe``, ``gpt_oss``, ``glm4_moe``,
``deepseek_v2``, ``deepseek_v3`` and ``glm4_moe_lite`` models for the full decoders. They need
``mlx`` and ``mlx_lm`` and therefore run on Apple Silicon only.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lokahi.fixtures import (
    TinyDeepseekSpec,
    TinyDenseSpec,
    TinyGemma3Spec,
    TinyGemma4Spec,
    write_tiny_deepseek,
    write_tiny_dense,
    write_tiny_gemma3,
    write_tiny_gemma4,
)
from lokahi.formats import QuantSpec, dequantize_affine, pack_affine
from lokahi.fixtures.deepseek_tiny import DEEPSEEK_YARN
from lokahi.oracles import DeepseekReference, Gemma3Reference, Gemma4Reference, LlamaFamilyReference

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


def test_mxfp4_layout_matches_mx_dequantize():
    import mlx.core as mx

    from lokahi.formats import dequantize_mxfp4

    rng = np.random.default_rng(4)
    codes = rng.integers(0, 16, size=(6, 128), dtype=np.uint32)
    exponents = rng.integers(110, 140, size=(6, 4)).astype(np.uint8)
    words = pack_affine(codes, 4)
    theirs = mx.dequantize(mx.array(words), mx.array(exponents), None, group_size=32, bits=4, mode="mxfp4")
    np.testing.assert_array_equal(np.array(theirs.astype(mx.float32)), dequantize_mxfp4(words, exponents))


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
        TinyDenseSpec(model_type="qwen3_moe", head_dim=64, mlp_only_layers=(1,), mxfp4_mlp=True, seed=11),
        TinyDenseSpec(model_type="qwen3", head_dim=64, rope_theta=1e6,
                      rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
                      seed=16),
        # MLX-LM's YaRN always rounds the correction range, so this GPT-OSS
        # variant sets truncate (the engine and oracle honor truncate false).
        TinyDenseSpec(model_type="gpt_oss", head_dim=64, num_hidden_layers=4, num_experts=8, num_experts_per_tok=2,
                      moe_intermediate_size=64, sliding_window=6, rope_theta=150000.0, mxfp4_mlp=True,
                      rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                    "original_max_position_embeddings": 4096, "truncate": True}, seed=14),
        TinyDenseSpec(model_type="glm4_moe", head_dim=64, num_hidden_layers=3, num_experts=8, num_experts_per_tok=2,
                      moe_intermediate_size=64, rope_theta=1e6, tie_word_embeddings=False, seed=18),
        TinyDenseSpec(model_type="glm4_moe", head_dim=64, num_hidden_layers=3, num_experts=8, num_experts_per_tok=3,
                      moe_intermediate_size=64, rope_theta=1e6, n_group=4, topk_group=2, routed_scaling_factor=2.5,
                      use_qk_norm=True, n_shared_experts=2, norm_topk_prob=False, seed=18),
    ],
    ids=["llama3-rope-llama3", "qwen3-qk-norm", "qwen2-bias-group7", "mistral-untied", "qwen3-moe-top2-of-8",
         "qwen3-moe-mxfp4-experts", "qwen3-yarn", "gpt-oss-mxfp4-truncated-yarn", "glm4-moe-air", "glm4-moe-grouped"],
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


@pytest.mark.parametrize(
    "spec",
    [
        # MLX-LM reads the absorbed layout as is (it would requantize a
        # quantized kv_b_proj), always rotates pairs, and scores dropped
        # groups 0, so these biases keep every selection score positive.
        TinyDeepseekSpec(absorbed=True, n_group=4, topk_group=2, num_experts_per_tok=3, routed_scaling_factor=2.5,
                         correction_bias_mean=1.0, seed=20),
        TinyDeepseekSpec(q_lora_rank=None, n_shared_experts=2, routed_scaling_factor=2.446, rms_norm_eps=1e-5,
                         stacked_experts=False, absorbed=True,
                         rope_scaling={**DEEPSEEK_YARN, "factor": 32, "mscale": 0.707, "mscale_all_dim": 1.0}, seed=20),
        TinyDeepseekSpec(model_type="deepseek_v2", q_lora_rank=None, norm_topk_prob=False, n_shared_experts=2,
                         rope_scaling={**DEEPSEEK_YARN, "mscale": 0.707, "mscale_all_dim": 0.707}, seed=37),
        TinyDeepseekSpec(model_type="deepseek_v2", topk_method="group_limited_greedy", n_group=4, topk_group=2,
                         num_experts_per_tok=3, norm_topk_prob=False, n_shared_experts=2, routed_scaling_factor=16.0,
                         rope_scaling={**DEEPSEEK_YARN, "mscale": 0.707, "mscale_all_dim": 0.707}, seed=20),
        # MLX-LM's latent norms take rms_norm_eps (1e-5) where Transformers'
        # take 1e-6; at these magnitudes the difference is far below tolerance.
        TinyDeepseekSpec(model_type="glm4_moe_lite", absorbed=True, qk_nope_head_dim=64, v_head_dim=96,
                         num_experts_per_tok=4, routed_scaling_factor=1.8, rms_norm_eps=1e-5, rope_theta=1e6,
                         rope_scaling=None, seed=21),
    ],
    ids=["v3-absorbed-grouped", "v3-moonlight-factor", "v2-lite", "v2-grouped", "glm4.7-flash"],
)
def test_oracle_matches_mlx_lm_deepseek(tmp_path, spec):
    directory = write_tiny_deepseek(tmp_path / "model", spec)
    _assert_mlx_lm_matches(directory, DeepseekReference.load(directory).logits(TOKENS))


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
