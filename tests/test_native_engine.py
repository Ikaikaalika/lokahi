"""Correctness evidence: the native engine against the independent NumPy oracle.

The native CLI is built once per session with CMake (or taken from
``LOKAHI_NATIVE_BUILD_DIR``). Every comparison uses generated fixtures, so the
results are correctness evidence only, never throughput evidence.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures import (
    TinyDeepseekSpec,
    TinyDenseSpec,
    TinyGemma3Spec,
    TinyGemma4Spec,
    TinyQwen35Spec,
    write_tiny_deepseek,
    write_tiny_dense,
    write_tiny_gemma3,
    write_tiny_gemma4,
    write_tiny_qwen35,
)
from lokahi.fixtures.deepseek_tiny import DEEPSEEK_YARN
from lokahi.formats.safetensors import read_safetensors
from lokahi.oracles import (
    DeepseekReference,
    Gemma3Reference,
    Gemma4Reference,
    LlamaFamilyReference,
    Qwen35Reference,
)
from lokahi.oracles.gemma3 import Gemma3Config

TOKENS = [2, 5, 9, 33, 100, 7, 7, 7, 400, 3, 2, 11, 90, 91, 92, 93, 17, 18, 19, 20, 21, 22]


def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _run(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


# GPT-OSS's YaRN: the correction range is not rounded (truncate false).
GPT_OSS_YARN = {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                "original_max_position_embeddings": 4096, "truncate": False}
# Qwen 3's long-context YaRN (factor 4 over 32k), truncated by default.
QWEN3_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}

LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    # Small enough that the fixture's head_dim spans all three frequency bands.
    "original_max_position_embeddings": 64,
}

GEMMA3_VARIANTS = {
    "gemma3-untied-hd64-mixed-bits": TinyGemma3Spec(),
    "gemma3-tied-hd256-rope-scaled": TinyGemma3Spec(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        tie_word_embeddings=True,
        rope_linear_factor=8.0,
        group_size=64,
        eight_bit_mlp_layers=(),
        seed=1,
    ),
    "gemma3-language-model-prefix": TinyGemma3Spec(prefix="language_model.", seed=2),
}
DENSE_VARIANTS = {
    # Llama 3.2: llama3 RoPE scaling, tied embeddings, 4 query heads per kv head.
    "llama3-tied-rope-llama3": TinyDenseSpec(num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                                             hidden_size=256, rope_scaling=LLAMA3_SCALING, seed=3),
    # Qwen 3: per-head q/k norm and head_dim * heads != hidden_size.
    "qwen3-qk-norm-hd128": TinyDenseSpec(model_type="qwen3", num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=128, rope_theta=1e6, seed=4),
    # Qwen 2.5 0.5B shape: q/k/v biases and 7 query heads per kv head.
    "qwen2-bias-group7": TinyDenseSpec(model_type="qwen2", hidden_size=448, num_attention_heads=7,
                                       num_key_value_heads=1, group_size=64, rope_theta=1e6, seed=5),
    # Mistral: untied head and a sliding window on every layer (64-wide heads,
    # which the Metal kernels cover; 32-wide heads fail closed there).
    "mistral-sliding-untied": TinyDenseSpec(model_type="mistral", hidden_size=256, sliding_window=8,
                                            tie_word_embeddings=False, rope_theta=1e6,
                                            rope_scaling={"rope_type": "linear", "factor": 2.0}, seed=6),
    # Qwen 3 MoE (30B-A3B layout): stacked experts, an 8-bit router, top-2 of 8
    # renormalized, and a dense layer from mlp_only_layers.
    "qwen3-moe-top2-of-8": TinyDenseSpec(model_type="qwen3_moe", head_dim=64, mlp_only_layers=(1,), seed=9),
    # GPT-OSS ships its experts in MXFP4: here stacked experts, and a dense
    # Llama MLP.
    "qwen3-moe-mxfp4-experts": TinyDenseSpec(model_type="qwen3_moe", head_dim=64, mlp_only_layers=(1,),
                                             mxfp4_mlp=True, seed=11),
    "llama-mxfp4-mlp": TinyDenseSpec(mxfp4_mlp=True, eight_bit_mlp_layers=(), seed=12),
    # GPT-OSS 20B's layout: MXFP4 experts by default with affine overrides,
    # biased attention with sinks, alternating windows, untruncated YaRN, a
    # biased router (top-k, then softmax) and biased clamped-SwiGLU experts.
    "gpt-oss-mxfp4-yarn": TinyDenseSpec(
        model_type="gpt_oss", head_dim=64, num_hidden_layers=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, sliding_window=6, rope_theta=150000.0, rope_scaling=GPT_OSS_YARN, mxfp4_mlp=True,
        seed=14,
    ),
    # Affine experts, top-3 of 6, truncated YaRN, a wider window.
    "gpt-oss-affine-top3": TinyDenseSpec(
        model_type="gpt_oss", head_dim=64, num_hidden_layers=4, num_experts=6, num_experts_per_tok=3,
        moe_intermediate_size=64, sliding_window=9, rope_theta=150000.0,
        rope_scaling={**GPT_OSS_YARN, "truncate": True}, seed=15,
    ),
    # GLM-4.5-Air's layout: q/k/v biases, RoPE on half of each head, a leading
    # dense layer, a dense sigmoid router with a correction bias, a shared
    # expert, untied.
    "glm4-moe-air": TinyDenseSpec(model_type="glm4_moe", head_dim=64, num_hidden_layers=3, num_experts=8,
                                  num_experts_per_tok=2, moe_intermediate_size=64, rope_theta=1e6,
                                  tie_word_embeddings=False, seed=18),
    # DeepSeek V3-style group-limited routing (best 2 of 4 groups), q/k norms,
    # two shared experts, raw weights times 2.5.
    "glm4-moe-grouped": TinyDenseSpec(model_type="glm4_moe", head_dim=64, num_hidden_layers=3, num_experts=8,
                                      num_experts_per_tok=3, moe_intermediate_size=64, rope_theta=1e6, n_group=4,
                                      topk_group=2, routed_scaling_factor=2.5, use_qk_norm=True, n_shared_experts=2,
                                      norm_topk_prob=False, seed=18),
    # Qwen 3 with long-context YaRN.
    "qwen3-yarn": TinyDenseSpec(model_type="qwen3", head_dim=64, rope_theta=1e6, rope_scaling=QWEN3_YARN, seed=16),
    # Every other layer sparse, top-3 of 6 unnormalized, Transformers'
    # per-expert tensors, untied (64-wide heads, which Metal covers).
    "qwen3-moe-top3-of-6-per-expert": TinyDenseSpec(
        model_type="qwen3_moe", head_dim=64, num_experts=6, num_experts_per_tok=3, decoder_sparse_step=2,
        norm_topk_prob=False, stacked_experts=False, tie_word_embeddings=False, seed=10,
    ),
}
GEMMA4_VARIANTS = {
    # E2B's shape: sliding and full head sizes, proportional partial RoPE on
    # full layers, the last two layers sharing KV with double-wide MLPs,
    # per-layer inputs, tied embeddings, text_config and language_model.
    "gemma4-e2b-shape": TinyGemma4Spec(),
    # 31B's attention: values are the raw keys on full layers, which have their
    # own kv head count; untied, flat config, 8-bit, no KV sharing or
    # per-layer inputs.
    "gemma4-k-eq-v-untied-flat": TinyGemma4Spec(
        num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True, num_kv_shared_layers=0,
        hidden_size_per_layer_input=0, use_double_wide_mlp=False, tie_word_embeddings=False, nested=False,
        sliding_window=5, global_partial_rotary_factor=0.5, group_size=64, bits=8, seed=7,
    ),
    # Shared sliding and full layers over k_eq_v sources, single-width MLPs,
    # no soft-cap.
    # 26B-A4B: a mixture of experts beside every dense MLP (defined with its
    # fixture guard below).
    "gemma4-26b-a4b-moe": None,
    "gemma4-shared-both-kinds": TinyGemma4Spec(
        num_hidden_layers=8, sliding_window_pattern=4, num_kv_shared_layers=3, use_double_wide_mlp=False,
        final_logit_softcapping=None, attention_k_eq_v=True, num_key_value_heads=2, num_global_key_value_heads=1,
        seed=8,
    ),
    # Gemma 4 12B (gemma4_unified): 31B's attention without per-layer inputs,
    # KV sharing or experts, tied, its vision and audio configs alongside, at
    # lmstudio-community's 5 bits in groups of 64.
    "gemma4-unified-12b-shape": TinyGemma4Spec(
        unified=True, num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True,
        num_kv_shared_layers=0, hidden_size_per_layer_input=0, use_double_wide_mlp=False, bits=5, group_size=64,
        seed=9,
    ),
}
# Gemma 4 26B-A4B's layout: 31B's attention plus a mixture of experts beside
# every dense MLP; untied so greedy continuations discriminate.
GEMMA4_MOE = TinyGemma4Spec(
    num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True, num_kv_shared_layers=0,
    hidden_size_per_layer_input=0, use_double_wide_mlp=False, enable_moe_block=True, tie_word_embeddings=False,
    seed=41,
)
GEMMA4_VARIANTS["gemma4-26b-a4b-moe"] = GEMMA4_MOE
# DeepSeek V2-Lite's YaRN, and one whose attention factor (mscale over
# mscale_all_dim) is not 1, so it scales the RoPE dimensions alone.
DEEPSEEK_V2_YARN = {**DEEPSEEK_YARN, "mscale": 0.707, "mscale_all_dim": 0.707}
DEEPSEEK_FACTOR_YARN = {**DEEPSEEK_YARN, "factor": 32, "mscale": 0.707, "mscale_all_dim": 1.0}
DEEPSEEK_VARIANTS = {
    # V3 (and Kimi K2) as recent MLX conversions store it: a query rank,
    # absorbed embed_q/unembed_out, interleaved RoPE, YaRN, and the best 2 of
    # 4 groups with negative correction biases, so excluding the dropped
    # groups differs from scoring them 0.
    "deepseek-v3-absorbed-grouped": TinyDeepseekSpec(
        absorbed=True, n_group=4, topk_group=2, num_experts_per_tok=3, routed_scaling_factor=2.5,
        correction_bias_mean=-0.3, seed=20),
    # Moonlight's layout plus rope_interleave false: no query rank, kv_b_proj,
    # two shared experts, rms_norm_eps 1e-5 (the latent norms keep 1e-6),
    # Transformers' per-expert modules, and a YaRN attention factor below 1.
    "deepseek-v3-kv-b-rotate-half": TinyDeepseekSpec(
        q_lora_rank=None, rope_interleave=False, rope_scaling=DEEPSEEK_FACTOR_YARN, n_shared_experts=2,
        routed_scaling_factor=2.446, rms_norm_eps=1e-5, stacked_experts=False, seed=20),
    # V2-Lite: no query rank, greedy softmax routing, two shared experts.
    "deepseek-v2-lite": TinyDeepseekSpec(
        model_type="deepseek_v2", q_lora_rank=None, norm_topk_prob=False, n_shared_experts=2,
        rope_scaling=DEEPSEEK_V2_YARN, seed=37),
    # V2's group_limited_greedy times routed_scaling_factor 16, in
    # Transformers' BF16 layout (which the Metal backend refuses).
    "deepseek-v2-grouped-bf16": TinyDeepseekSpec(
        model_type="deepseek_v2", topk_method="group_limited_greedy", n_group=4, topk_group=2,
        num_experts_per_tok=3, norm_topk_prob=False, n_shared_experts=2, routed_scaling_factor=16.0, bits=0,
        stacked_experts=False, rope_scaling=DEEPSEEK_V2_YARN, seed=20),
    # V2-Lite's latent geometry as mlx-community stores it: a 512-wide latent
    # plus 64 rope dimensions (the 576-wide latent of every released model),
    # 128-wide heads, kv_b_proj at 4 bits in groups of 64, eight heads.
    "deepseek-v2-lite-geometry": TinyDeepseekSpec(
        model_type="deepseek_v2", num_attention_heads=8, kv_lora_rank=512, q_lora_rank=None, qk_nope_head_dim=128,
        qk_rope_head_dim=64, v_head_dim=128, norm_topk_prob=False, n_shared_experts=2, rope_scaling=DEEPSEEK_V2_YARN,
        group_size=64, seed=20),
    # GLM-4.7-Flash (glm4_moe_lite): V3 with a query rank, absorbed tensors,
    # values wider than the unrotated keys (as 256 against 192), top-4 times
    # 1.8, rms_norm_eps 1e-5, no YaRN.
    "glm4.7-flash-shape": TinyDeepseekSpec(
        model_type="glm4_moe_lite", absorbed=True, qk_nope_head_dim=64, v_head_dim=96, num_experts_per_tok=4,
        routed_scaling_factor=1.8, rms_norm_eps=1e-5, rope_theta=1e6, rope_scaling=None, seed=21),
    # Latent attention at widths whose codes straddle words: kv_b_proj's
    # per-head absorption at 5 bits, GLM-4.7-Flash's absorbed tensors at 6.
    "deepseek-v2-lite-5bit": TinyDeepseekSpec(
        model_type="deepseek_v2", q_lora_rank=None, norm_topk_prob=False, n_shared_experts=2,
        rope_scaling=DEEPSEEK_V2_YARN, bits=5, seed=37),
    "glm4.7-flash-6bit": TinyDeepseekSpec(
        model_type="glm4_moe_lite", absorbed=True, qk_nope_head_dim=64, v_head_dim=96, num_experts_per_tok=4,
        routed_scaling_factor=1.8, rms_norm_eps=1e-5, rope_theta=1e6, rope_scaling=None, bits=6, seed=21),
}
QWEN35_VARIANTS = {
    # Qwen3.5 as MLX-LM converts it: 4-bit under language_model., norms stored
    # as 1 + w, conv weights [C, K, 1]; Gated DeltaNet layers 0-2 and 4 with
    # two value heads per key head, gated full attention at layer 3, partial
    # RoPE over a quarter of each head.
    "qwen3.5-mlx": TinyQwen35Spec(seed=0),
    # Transformers' BF16 layout under model.language_model. (norms w, conv
    # [C, 1, K]), tied, one value head per key head, keys narrower than values.
    "qwen3.5-hf-tied": TinyQwen35Spec(mlx=False, tie_word_embeddings=True, linear_num_key_heads=4,
                                      linear_key_head_dim=16, seed=1),
    # Qwen3.5-MoE: experts in every layer with a gated shared expert, 8-bit
    # router and gate; and Transformers' fused gate_up_proj experts.
    "qwen3.5-moe-mlx": TinyQwen35Spec(model_type="qwen3_5_moe", seed=0),
    "qwen3.5-moe-hf-fused": TinyQwen35Spec(model_type="qwen3_5_moe", mlx=False, num_experts_per_tok=3, seed=1),
    # Qwen3-Next: in_proj_qkvz and in_proj_ba grouped per key head, a dense
    # layer by mlp_only_layers or by decoder_sparse_step, routing with and
    # without renormalization.
    "qwen3-next-mlx": TinyQwen35Spec(model_type="qwen3_next", mlp_only_layers=(1,), norm_topk_prob=False,
                                     linear_key_head_dim=16, seed=0),
    "qwen3-next-hf": TinyQwen35Spec(model_type="qwen3_next", mlx=False, decoder_sparse_step=2, seed=0),
    # Qwen3.8-27B's 5- and 6-bit MLX builds (group size 64), where codes
    # straddle words, and 3-bit experts.
    "qwen3.5-mlx-5bit": TinyQwen35Spec(bits=5, group_size=64, seed=0),
    "qwen3.5-mlx-6bit": TinyQwen35Spec(bits=6, group_size=64, seed=0),
    "qwen3.5-moe-mlx-3bit": TinyQwen35Spec(model_type="qwen3_5_moe", bits=3, seed=0),
}
VARIANTS = (sorted(GEMMA3_VARIANTS) + sorted(GEMMA4_VARIANTS) + sorted(DENSE_VARIANTS) + sorted(DEEPSEEK_VARIANTS)
            + sorted(QWEN35_VARIANTS))


@pytest.fixture(scope="module", params=VARIANTS)
def fixture_model(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    if request.param in GEMMA3_VARIANTS:
        write_tiny_gemma3(directory, GEMMA3_VARIANTS[request.param])
        return directory, Gemma3Reference.load(directory)
    if request.param in GEMMA4_VARIANTS:
        write_tiny_gemma4(directory, GEMMA4_VARIANTS[request.param])
        return directory, Gemma4Reference.load(directory)
    if request.param in DEEPSEEK_VARIANTS:
        write_tiny_deepseek(directory, DEEPSEEK_VARIANTS[request.param])
        return directory, DeepseekReference.load(directory)
    if request.param in QWEN35_VARIANTS:
        write_tiny_qwen35(directory, QWEN35_VARIANTS[request.param])
        return directory, Qwen35Reference.load(directory)
    write_tiny_dense(directory, DENSE_VARIANTS[request.param])
    return directory, LlamaFamilyReference.load(directory)


CPU_ONLY: tuple[str, ...] = ()  # architectures the Metal backend refuses (none today)


def _dense_mxfp4(config: dict) -> bool:
    """Whether a matrix other than an expert is MXFP4 (Metal reads MXFP4 only for experts)."""

    quant = config.get("quantization") or {}
    overrides = [k for k, v in quant.items() if isinstance(v, dict) and v.get("mode") == "mxfp4"]
    return any(".experts." not in k and ".switch_mlp." not in k for k in overrides)


def _skip_unsupported(directory: Path, backend: str) -> None:
    config = json.loads((directory / "config.json").read_text())
    if backend == "metal" and (config.get("text_config") or config)["model_type"] in CPU_ONLY:
        pytest.skip("the Metal backend does not implement this architecture yet")
    if backend == "metal" and _dense_mxfp4(config):
        pytest.skip("the Metal backend implements mxfp4 only for experts so far")
    if backend == "metal" and "quantization" not in config:
        pytest.skip("the Metal backend reads quantized checkpoints only")


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(("prefill", "chunk"), [(len(TOKENS), 512), (5, 512), (1, 512), (17, 3)])
def test_logits_match_oracle(native_cli, fixture_model, backend, prefill, chunk, tmp_path):
    directory, oracle = fixture_model
    _skip_unsupported(directory, backend)
    out = tmp_path / "logits.bin"
    report = _run(
        native_cli,
        "logits",
        "--model", str(directory),
        "--backend", backend,
        "--tokens", ",".join(map(str, TOKENS)),
        "--prefill", str(prefill),
        "--prefill-chunk", str(chunk),
        "--out", str(out),
    )
    vocab = oracle.config.vocab_size
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], vocab)
    expected = oracle.logits(TOKENS)[prefill - 1 :]
    assert got.shape == expected.shape
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))


def _latent(directory: Path) -> bool:
    config = json.loads((directory / "config.json").read_text())
    text = config.get("text_config") or config
    return bool(text.get("kv_lora_rank"))


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("kv_cache", ["f16", "f32"])
def test_kv_cache_element_type_matches_oracle(native_cli, fixture_model, backend, kv_cache, tmp_path):
    # Keys and values stored as FP16 (Metal's default) or FP32. The CPU rounds
    # them through FP16 as Metal stores them; latent attention keeps FP32.
    directory, oracle = fixture_model
    _skip_unsupported(directory, backend)
    args = ("--model", str(directory), "--backend", backend, "--kv-cache", kv_cache)
    if kv_cache == "f16" and _latent(directory):
        assert "latent attention keeps its cache in f32" in _load_error(native_cli, directory, backend, kv_cache)
        return
    assert _run(native_cli, "info", *args)["kv_cache"] == kv_cache
    out = tmp_path / "logits.bin"
    # Prefill 9 tokens in chunks of 4, then decode the rest one at a time.
    report = _run(native_cli, "logits", *args, "--tokens", ",".join(map(str, TOKENS)),
                  "--prefill", "9", "--prefill-chunk", "4", "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], oracle.config.vocab_size)
    expected = oracle.logits(TOKENS)[8:]
    # FP16 keys and values move these fixtures' logits by 0.006-0.4% of their
    # scale (GPT-OSS's the most), inside Metal's 2% (correctness evidence only).
    tolerance = 2e-2 if backend == "metal" else 1e-2 if kv_cache == "f16" else 2e-4
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))
    if backend == "cpu" and kv_cache == "f16":  # the rounding happened
        exact = tmp_path / "exact.bin"
        _run(native_cli, "logits", "--model", str(directory), "--backend", "cpu", "--tokens",
             ",".join(map(str, TOKENS)), "--prefill", "9", "--prefill-chunk", "4", "--out", str(exact))
        assert not np.array_equal(got, np.fromfile(exact, dtype=np.float32).reshape(got.shape))


@pytest.mark.parametrize("name", ["llama3-tied-rope-llama3", "mistral-sliding-untied"])
def test_cpu_f16_cache_rounds_keys_and_values_as_numpy_does(native_cli, tmp_path, monkeypatch, name):
    # The CPU's FP16 cache rounds keys and values to nearest even, as Metal
    # stores them and as NumPy converts, so it lands much nearer the oracle
    # with keys and values stored as np.float16 than the exact oracle. Values
    # within 1e-6 of a rounding midpoint can round either way, so it is not
    # exactly the rounded oracle: measured 0.09 and 0.21 of the way here, and
    # 1.2 and 1.4 with only the keys rounded.
    import lokahi.oracles.llama as llama

    directory = write_tiny_dense(tmp_path / "model", DENSE_VARIANTS[name])
    exact = LlamaFamilyReference.load(directory).logits(TOKENS)[8:]
    monkeypatch.setattr(llama, "KV_STORAGE", np.float16)
    rounded = LlamaFamilyReference.load(directory).logits(TOKENS)[8:]
    out = tmp_path / "logits.bin"
    _run(native_cli, "logits", "--model", str(directory), "--backend", "cpu", "--kv-cache", "f16",
         "--tokens", ",".join(map(str, TOKENS)), "--prefill", "9", "--prefill-chunk", "4", "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(exact.shape)
    assert np.abs(rounded - exact).max() > 1e-4 * np.abs(exact).max()
    assert np.abs(got - rounded).max() < 0.5 * np.abs(got - exact).max()


@pytest.mark.parametrize("backend", _backends())
def test_kv_cache_defaults_to_f16_on_metal(native_cli, fixture_model, backend):
    directory, _ = fixture_model
    _skip_unsupported(directory, backend)
    expected = "f16" if backend == "metal" and not _latent(directory) else "f32"
    assert _run(native_cli, "info", "--model", str(directory), "--backend", backend)["kv_cache"] == expected


def test_gemma4_unified_runs_text_beside_a_tower_it_does_not(native_cli, tmp_path):
    # Gemma 4 12B's checkpoints carry a gemma4_unified_vision tower and an
    # audio tower. The text model loads and info says the tower is not
    # supported; asking for images fails closed.
    directory = write_tiny_gemma4(tmp_path / "model", GEMMA4_VARIANTS["gemma4-unified-12b-shape"])
    info = _run(native_cli, "info", "--model", str(directory), "--backend", "cpu")
    assert info["architecture"] == "gemma4_unified"
    assert info["vision"] == {"supported": False, "model_type": "gemma4_unified_vision"}
    image = tmp_path / "image.ppm"
    image.write_bytes(b"P6\n16 16\n255\n" + bytes(16 * 16 * 3))
    result = subprocess.run([str(native_cli), "generate", "--model", str(directory), "--backend", "cpu",
                             "--tokens", "2,5,9", "--max-new", "2", "--images", str(image)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "unsupported vision tower 'gemma4_unified_vision'" in result.stderr


_NO_PLE_OR_EXPERTS = "gemma4_unified: per-layer inputs and experts are not part of this model"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hidden_size_per_layer_input": 16}, _NO_PLE_OR_EXPERTS),
        ({"enable_moe_block": True, "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 64},
         _NO_PLE_OR_EXPERTS),
        ({"use_bidirectional_attention": "all"}, "gemma4_unified: use_bidirectional_attention 'all' is not supported"),
        # Transformers would make the last layer full attention; the engine refuses.
        ({"layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
                          "full_attention", "sliding_attention"]},
         "gemma4_unified: the last layer must be full_attention"),
    ],
)
def test_unsupported_gemma4_unified_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_gemma4(tmp_path / "model", GEMMA4_VARIANTS["gemma4-unified-12b-shape"])
    assert message in _load_error(native_cli, _edit_text_config(directory, change))


def test_unknown_kv_cache_type_fails_closed(native_cli, tmp_path):
    directory = write_tiny_qwen35(tmp_path / "model", QWEN35_VARIANTS["qwen3.5-mlx"])
    assert "kv_cache must be auto, f16 or f32, not 'bf16'" in _load_error(native_cli, directory, "cpu", "bf16")


@pytest.mark.parametrize("backend", _backends())
def test_greedy_generation_matches_full_recompute(native_cli, fixture_model, backend):
    directory, oracle = fixture_model
    _skip_unsupported(directory, backend)
    prompt = TOKENS[:6]
    report = _run(
        native_cli,
        "generate",
        "--model", str(directory),
        "--backend", backend,
        "--tokens", ",".join(map(str, prompt)),
        "--max-new", "14",
        "--no-eos-stop",
    )
    assert report["tokens"] == oracle.greedy(prompt, 14)


def test_context_overflow_fails_closed(native_cli, tmp_path):
    directory = write_tiny_gemma3(tmp_path / "model", TinyGemma3Spec())
    result = subprocess.run(
        [
            str(native_cli), "logits", "--model", str(directory), "--backend", "cpu",
            "--tokens", "1,2,3,4,5", "--max-context", "4", "--out", str(tmp_path / "x.bin"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "context length exceeded" in result.stderr


def test_oracle_reads_gemma3_multimodal_config_defaults():
    # mlx-community/gemma-3-4b-it-4bit lists only the text fields that differ
    # from Transformers' defaults (the engine's side: test_gemma3_config).
    raw = {
        "model_type": "gemma3",
        "text_config": {"hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
                        "num_hidden_layers": 34, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                        "sliding_window": 1024},
    }
    cfg = Gemma3Config.from_dict(raw)
    assert (cfg.vocab_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (262208, 8, 4, 256)
    assert cfg.query_pre_attn_scalar == 256 and cfg.rope_linear_factor == 8.0 and cfg.sliding_window == 1024
    assert cfg.layer_is_sliding == tuple(index % 6 != 5 for index in range(34))


def test_reports_the_architecture(native_cli, tmp_path):
    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(model_type="qwen3"))
    assert _run(native_cli, "info", "--model", str(directory), "--backend", "cpu")["architecture"] == "qwen3"


def test_llama3_scaling_changes_the_fixture_output():
    # Guards the fixture: the llama3 variant must exercise scaled frequencies.
    from lokahi.oracles import LlamaFamilyConfig

    spec = DENSE_VARIANTS["llama3-tied-rope-llama3"]
    scaled = LlamaFamilyConfig.from_dict(spec.config()).inv_freq()
    plain = LlamaFamilyConfig.from_dict(replace(spec, rope_scaling=None).config()).inv_freq()
    assert (scaled < plain).any() and (scaled == plain).any()
    assert ((scaled < plain) & (scaled > plain / 32.0)).any()  # the interpolated band


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model_type": "phi3"}, "unsupported architecture 'phi3'"),
        ({"attention_bias": True}, "attention_bias is not supported"),
        ({"mlp_bias": True}, "mlp_bias is not supported"),
        ({"hidden_act": "gelu"}, "unsupported activation gelu"),
        ({"use_sliding_window": True}, "use_sliding_window is not supported"),
        ({"rope_scaling": {"rope_type": "longrope", "factor": 4.0}}, "rope_scaling type 'longrope' is not supported"),
        ({"partial_rotary_factor": 0.5}, "partial_rotary_factor is not supported"),
        ({"tie_word_embeddings": False}, "tie_word_embeddings is false but the checkpoint has no lm_head"),
    ],
)
def test_unsupported_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(extra=change))
    result = subprocess.run(
        [str(native_cli), "info", "--model", str(directory), "--backend", "cpu"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_unexpected_bias_fails_closed(native_cli, tmp_path):
    # A Qwen 2 checkpoint relabeled as Llama carries biases Llama does not have.
    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(model_type="qwen2"))
    config = json.loads((directory / "config.json").read_text())
    config["model_type"] = "llama"
    (directory / "config.json").write_text(json.dumps(config))
    result = subprocess.run(
        [str(native_cli), "info", "--model", str(directory), "--backend", "cpu"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "self_attn.q_proj: unexpected bias for llama" in result.stderr


@pytest.mark.parametrize("backend", _backends())
def test_profile_attributes_every_section_without_changing_tokens(native_cli, fixture_model, backend):
    directory, oracle = fixture_model
    _skip_unsupported(directory, backend)
    prompt, steps = TOKENS[:9], 6
    report = _run(native_cli, "profile", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, prompt)), "--max-new", str(steps))
    # Profiling serializes work per section but must not change the result.
    assert report["tokens"] == oracle.greedy(prompt, steps + 1)
    layers = oracle.config.num_hidden_layers
    per_layer = {"qkv", "rope_kv", "attention", "o_proj"}
    for phase, passes in (("prefill", 1), ("decode", steps)):
        sections = {section["name"]: section for section in report[phase]["sections"]}
        assert per_layer <= sections.keys() and {"embed", "norm", "lm_head"} <= sections.keys()
        for name in per_layer:
            assert sections[name]["calls"] == passes * layers, (phase, name)
        assert sections["lm_head"]["calls"] == passes
        assert all(s["seconds"] >= 0 and s["bytes"] > 0 for s in sections.values())
        total = sum(s["seconds"] for s in sections.values())
        assert total <= report[phase]["wall_seconds"] * 1.05 + 1e-3
    # Every decode step reads every layer weight and the vocabulary projection;
    # with tied embeddings that is all of weight_bytes but the tables only
    # gathered from (Gemma 4's per-layer embeddings) and the experts a token
    # may not use. CPU splits the MLP; Metal fuses it.
    decode = {section["name"]: section for section in report["decode"]["sections"]}
    linear = ("qkv", "o_proj", "mlp_up", "down_proj", "mlp", "per_layer_input", "router", "experts")
    linear_bytes = sum(decode[name]["bytes"] for name in linear if name in decode) + decode["lm_head"]["bytes"]
    tied = oracle.w["lm_head"] is oracle.w["model.embed_tokens"]
    read = report["weight_bytes"] - _gathered_table_bytes(directory)
    assert linear_bytes / steps >= read * (1.0 if tied else 0.5)


def _gathered_table_bytes(directory: Path) -> int:
    total = 0
    sparse = (".embed_tokens_per_layer.", ".switch_mlp.", ".experts.")
    for shard in directory.glob("*.safetensors"):
        tensors, _ = read_safetensors(shard)
        total += sum(array.nbytes for name, (_, array) in tensors.items() if any(s in name for s in sparse))
    return total


def _load_error(native_cli: Path, directory: Path, backend: str = "cpu", kv_cache: str = "auto") -> str:
    result = subprocess.run(
        [str(native_cli), "info", "--model", str(directory), "--backend", backend, "--kv-cache", kv_cache],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    return result.stderr


@pytest.mark.skipif("metal" not in _backends(), reason="needs the Metal backend")
def test_metal_refuses_head_sizes_its_kernels_lack(native_cli, tmp_path):
    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec())  # 128 / 4 heads = 32-wide heads
    result = subprocess.run(
        [str(native_cli), "info", "--model", str(directory), "--backend", "metal"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "metal: head_dim must be 64, 128, 256 or 512" in result.stderr


def test_gemma4_fixtures_exercise_every_feature(tmp_path):
    # Guards the fixtures: each Gemma 4 mechanism must be live in some variant.
    e2b = Gemma4Reference.load(write_tiny_gemma4(tmp_path / "e2b", GEMMA4_VARIANTS["gemma4-e2b-shape"]))
    cfg = e2b.config
    assert [cfg.kv_source(i) for i in range(6)] == [0, 1, 2, 3, 3, 2]
    assert {cfg.head_size(i) for i in range(6)} == {64, 128}
    full = cfg.inv_freq(2)
    assert (full == 0).sum() == 48 and (full > 0).sum() == 16  # 0.25 of 128 dims rotate
    assert "model.layers.4.self_attn.k_proj" not in e2b.w and "model.layers.3.self_attn.k_proj" in e2b.w
    assert e2b.w["model.layers.4.mlp.gate_proj"].shape[0] == 2 * e2b.w["model.layers.3.mlp.gate_proj"].shape[0]
    assert "model.embed_tokens_per_layer" in e2b.w
    assert all(abs(float(e2b.w[f"model.layers.{i}.layer_scalar"][0]) - 1.0) > 1e-3 for i in range(6))

    both = Gemma4Reference.load(write_tiny_gemma4(tmp_path / "both", GEMMA4_VARIANTS["gemma4-shared-both-kinds"]))
    cfg = both.config
    assert [cfg.kv_source(i) for i in range(5, 8)] == [4, 4, 3] and cfg.k_eq_v(3) and cfg.kv_heads(3) == 1
    assert "model.layers.3.self_attn.v_proj" not in both.w and "model.layers.2.self_attn.v_proj" in both.w


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"enable_moe_block": True, "num_experts": 8, "top_k_experts": 9, "moe_intermediate_size": 64},
         "gemma4: invalid mixture-of-experts sizes"),
        ({"num_kv_shared_layers": 4}, "gemma4: a shared layer has no earlier layer of its kind"),
        ({"rope_parameters": {"full_attention": {"rope_type": "yarn", "factor": 4.0}}},
         "gemma4: full rope_type 'yarn' is not supported"),
        ({"rope_parameters": {"sliding_attention": {"rope_type": "default", "partial_rotary_factor": 0.5}}},
         "gemma4: partial rotation on sliding layers"),
        ({"hidden_activation": "silu"}, "gemma4: unsupported activation silu"),
    ],
)
def test_unsupported_gemma4_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_gemma4(tmp_path / "model", TinyGemma4Spec(extra=change))
    assert message in _load_error(native_cli, directory)


def test_gemma4_moe_without_its_router_scales_fails_closed(native_cli, tmp_path):
    directory = write_tiny_gemma4(tmp_path / "model", GEMMA4_MOE)
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    del tensors["language_model.model.layers.2.router.per_expert_scale"]
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "layers.2.router.per_expert_scale" in _load_error(native_cli, directory)


def test_gemma4_missing_layer_scalar_fails_closed(native_cli, tmp_path):
    directory = write_tiny_gemma4(tmp_path / "model", TinyGemma4Spec())
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    del tensors["language_model.model.layers.3.layer_scalar"]
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "layers.3.layer_scalar" in _load_error(native_cli, directory)


@pytest.mark.parametrize("architecture", ["llama-mxfp4"])
def test_cpu_only_models_fall_back_to_the_cpu(native_cli, tmp_path, architecture):
    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(mxfp4_mlp=True, eight_bit_mlp_layers=()))
    # auto falls back to the CPU backend, which implements every architecture and format.
    info = _run(native_cli, "info", "--model", str(directory), "--backend", "auto")
    assert (info["architecture"], info["backend"]) == (architecture.split("-")[0], "cpu")
    if "metal" in _backends():
        message = "is mxfp4, which the Metal kernels" if architecture == "llama-mxfp4" else "the Metal backend does not"
        assert message in _load_error(native_cli, directory, "metal")


def test_mxfp4_matches_its_definition():
    # The OCP MXFP4 element and scale encodings, independently of the packer.
    from lokahi.formats import FP4_E2M1, dequantize_mxfp4, e8m0_to_f32, pack_affine

    assert FP4_E2M1.tolist()[:8] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert (FP4_E2M1[8:] == -FP4_E2M1[:8]).all() and np.signbit(FP4_E2M1[8])
    assert e8m0_to_f32(np.array([127, 128, 0, 254])).tolist() == [1.0, 2.0, 2.0**-127, 2.0**127]
    codes = np.arange(64, dtype=np.uint32).reshape(1, 64) % 16
    matrix = dequantize_mxfp4(pack_affine(codes, 4), np.array([[127, 125]], dtype=np.uint8))
    np.testing.assert_array_equal(matrix[0, :16], FP4_E2M1)
    np.testing.assert_array_equal(matrix[0, 32:48], FP4_E2M1 / 4)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"group_size": 64, "bits": 4, "mode": "mxfp4"}, "mxfp4 needs 4 bits and group size 32"),
        ({"group_size": 16, "bits": 4, "mode": "nvfp4"}, "quantization mode 'nvfp4' is not supported yet"),
    ],
)
def test_unsupported_quantization_fails_closed(native_cli, tmp_path, override, message):
    spec = TinyDenseSpec(mxfp4_mlp=True, eight_bit_mlp_layers=())
    directory = write_tiny_dense(tmp_path / "model", spec)
    config = json.loads((directory / "config.json").read_text())
    config["quantization"]["model.layers.0.mlp.up_proj"] = override
    (directory / "config.json").write_text(json.dumps(config))
    assert message in _load_error(native_cli, directory)


def test_gemma4_moe_fixture_routes_far_from_ties_and_every_part_matters(tmp_path):
    # Guards the fixture: on the router's real inputs the top-k choice is far
    # from ties and uses every expert, and the per-expert scales, the
    # router's input scale and the expert branch each move the outputs.
    directory = write_tiny_gemma4(tmp_path / "moe", GEMMA4_MOE)
    oracle = Gemma4Reference.load(directory)
    margins, used, route = [], set(), oracle.route

    def spy(index, x):
        chosen, weights = route(index, x)
        prefix = f"model.layers.{index}.router"
        routed = (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
        logits = (routed * oracle.w[f"{prefix}.scale"] * np.float32(GEMMA4_MOE.hidden_size**-0.5)) @ (
            oracle.w[f"{prefix}.proj"].T)
        ordered = np.sort(logits, axis=-1)[:, ::-1]
        margins.append(float(np.min(ordered[:, GEMMA4_MOE.top_k_experts - 1] - ordered[:, GEMMA4_MOE.top_k_experts])))
        used.update(int(e) for e in chosen.reshape(-1))
        return chosen, weights

    oracle.route = spy
    expected = oracle.logits(TOKENS)
    assert min(margins) > 1e-2 and used == set(range(GEMMA4_MOE.num_experts))
    scale = np.abs(expected).max()
    for name in ("router.per_expert_scale", "router.scale"):
        changed = Gemma4Reference.load(directory)
        for layer in range(GEMMA4_MOE.num_hidden_layers):
            changed.w[f"model.layers.{layer}.{name}"] = np.ones_like(changed.w[f"model.layers.{layer}.{name}"])
        assert np.abs(changed.logits(TOKENS) - expected).max() > 0.01 * scale, name
    no_experts = Gemma4Reference.load(directory)
    no_experts._experts = lambda index, x: np.zeros_like(x)
    assert np.abs(no_experts.logits(TOKENS) - expected).max() > 0.05 * scale


def test_moe_fixtures_route_far_from_ties(tmp_path):
    # Guards the fixtures: on the router's real inputs, the top-k choice must
    # be stable under float32 error, and every expert must serve some token.
    for name in ("qwen3-moe-top2-of-8", "qwen3-moe-mxfp4-experts", "qwen3-moe-top3-of-6-per-expert",
                 "gpt-oss-mxfp4-yarn", "gpt-oss-affine-top3", "glm4-moe-air", "glm4-moe-grouped"):
        oracle = LlamaFamilyReference.load(write_tiny_dense(tmp_path / name, DENSE_VARIANTS[name]))
        k, route = oracle.config.num_experts_per_tok, oracle.route
        gaps: list[float] = []
        used: set[int] = set()

        def traced(layer, x):
            ordered = -np.sort(-oracle.router_logits(layer, x), axis=-1)
            gaps.append(float((ordered[:, k - 1] - ordered[:, k]).min()))
            chosen, weights = route(layer, x)
            used.update(chosen.ravel().tolist())
            return chosen, weights

        oracle.route = traced
        oracle.logits(TOKENS)
        assert min(gaps) > 1e-3, name
        assert len(used) == oracle.config.num_experts, name


def test_gpt_oss_fixtures_exercise_every_feature(tmp_path):
    # Guards the fixtures: the clamps bind, sinks take probability mass, and
    # untruncated YaRN differs from truncated YaRN.
    from lokahi.oracles import LlamaFamilyConfig

    spec = DENSE_VARIANTS["gpt-oss-mxfp4-yarn"]
    oracle = LlamaFamilyReference.load(write_tiny_dense(tmp_path / "oss", spec))
    gates: list[np.ndarray] = []
    ups: list[np.ndarray] = []
    experts = oracle._experts

    def traced(prefix, x):
        layer = int(prefix.split(".")[2])
        chosen, _ = oracle.route(layer, x)
        for t, row in enumerate(chosen):
            for e in row:
                gates.append(oracle.w[f"{prefix}.experts.gate_proj"][e] @ x[t] + oracle.w[f"{prefix}.experts.gate_proj.bias"][e])
                ups.append(oracle.w[f"{prefix}.experts.up_proj"][e] @ x[t] + oracle.w[f"{prefix}.experts.up_proj.bias"][e])
        return experts(prefix, x)

    oracle._experts = traced
    logits = oracle.logits(TOKENS)
    gate, up = np.concatenate(gates), np.concatenate(ups)
    assert (gate > 7).mean() > 0.005 and (np.abs(up) > 7).mean() > 0.005
    del oracle._experts
    for layer in range(spec.num_hidden_layers):  # a sink of -inf takes no mass
        oracle.w[f"model.layers.{layer}.self_attn.sinks"] = np.full(spec.num_attention_heads, -1e30, np.float32)
    assert np.abs(oracle.logits(TOKENS) - logits).max() > 0.05 * np.abs(logits).max()
    untruncated = LlamaFamilyConfig.from_dict(spec.config())
    truncated = LlamaFamilyConfig.from_dict(replace(spec, rope_scaling={**GPT_OSS_YARN, "truncate": True}).config())
    assert not np.allclose(untruncated.inv_freq(), truncated.inv_freq())
    assert abs(untruncated.attention_factor() - (0.1 * np.log(32.0) + 1.0)) < 1e-12


def test_glm_fixtures_exercise_every_feature(tmp_path):
    # Guards the fixtures: group limits and the correction bias each change
    # some token's experts, and partial RoPE leaves half of each head alone.
    oracle = LlamaFamilyReference.load(write_tiny_dense(tmp_path / "grouped", DENSE_VARIANTS["glm4-moe-grouped"]))
    cfg = oracle.config
    assert cfg.rotary_dim == cfg.head_dim // 2 and cfg.inv_freq().shape == (cfg.head_dim // 4,)
    inputs: list[np.ndarray] = []
    route = oracle.route

    def traced(layer, x):
        inputs.append((layer, x))
        return route(layer, x)

    oracle.route = traced
    oracle.logits(TOKENS)
    k = cfg.num_experts_per_tok
    regrouped = debiased = 0
    for layer, x in inputs:
        chosen = np.sort(route(layer, x)[0], axis=-1)
        raw = 1.0 / (1.0 + np.exp(-(x @ oracle.w[f"model.layers.{layer}.mlp.gate"].T)))
        corrected = raw + oracle.w[f"model.layers.{layer}.mlp.gate.e_score_correction_bias"]
        regrouped += (np.sort(np.argsort(-corrected, axis=-1)[:, :k], axis=-1) != chosen).any(axis=-1).sum()
        debiased += (np.sort(np.argsort(-raw, axis=-1)[:, :k], axis=-1) != chosen).any(axis=-1).sum()
    assert regrouped > 0 and debiased > 0


def test_gpt_oss_without_sinks_fails_closed(native_cli, tmp_path):
    directory = write_tiny_dense(tmp_path / "model", DENSE_VARIANTS["gpt-oss-affine-top3"])
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    del tensors["model.layers.1.self_attn.sinks"]
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "model.layers.1.self_attn.sinks" in _load_error(native_cli, directory)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"shared_expert_intermediate_size": 64}, "qwen3_moe: shared experts are not supported"),
        ({"num_experts_per_tok": 9}, "qwen3_moe: invalid expert configuration"),
        ({"mlp_only_layers": [3]}, "qwen3_moe: mlp_only_layers entry out of range"),
        ({"scoring_func": "softmax"}, "glm4_moe: only sigmoid routing is supported"),
        ({"topk_method": "greedy"}, "glm4_moe: only noaux_tc routing is supported"),
        ({"n_group": 3}, "glm4_moe: invalid expert configuration"),
        ({"partial_rotary_factor": 0.3}, "glm4_moe: partial_rotary_factor must rotate an even number of dimensions"),
    ],
)
def test_unsupported_moe_configurations_fail_closed(native_cli, tmp_path, change, message):
    model_type = message.split(":")[0]
    spec = TinyDenseSpec(model_type=model_type, head_dim=64, extra=change)
    directory = write_tiny_dense(tmp_path / "model", spec)
    assert message in _load_error(native_cli, directory)


def _deepseek_routes(oracle: DeepseekReference) -> list[tuple[int, np.ndarray]]:
    """Each MoE layer's router inputs over TOKENS, as the oracle computes them."""

    inputs: list[tuple[int, np.ndarray]] = []
    route = oracle.route

    def traced(layer, x):
        inputs.append((layer, x))
        return route(layer, x)

    oracle.route = traced
    oracle.logits(TOKENS)
    del oracle.route
    return inputs


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    return np.sort(np.argsort(-scores, axis=-1, kind="stable")[:, :k], axis=-1)


def test_deepseek_fixtures_route_far_from_ties_and_exercise_every_feature(tmp_path):
    # Guards the fixtures: every expert serves some token, the top-k and group
    # choices are stable under float32 error (V2's on logits, which its
    # probabilities only compress), and each routing mechanism changes some
    # token's experts: the V3 correction bias, the group limits, and V3's
    # exclusion of dropped groups rather than scoring them 0.
    for name, spec in DEEPSEEK_VARIANTS.items():
        oracle = DeepseekReference.load(write_tiny_deepseek(tmp_path / name, spec))
        cfg = oracle.config
        k, groups, per_group = cfg.num_experts_per_tok, cfg.n_group, cfg.n_routed_experts // cfg.n_group
        used: set[int] = set()
        changed = {"bias": 0, "groups": 0, "zero": 0}
        for layer, x in _deepseek_routes(oracle):
            chosen = np.sort(oracle.route(layer, x)[0], axis=-1)
            used.update(chosen.ravel().tolist())
            logits = (x @ oracle.w[f"model.layers.{layer}.mlp.gate"].T).astype(np.float64)
            selection = oracle.selection(layer, x)
            if cfg.model_type != "deepseek_v2":
                raw = 1.0 / (1.0 + np.exp(-logits))
                corrected = raw + oracle.w[f"model.layers.{layer}.mlp.gate.e_score_correction_bias"]
                group_scores = -np.sort(-corrected.reshape(len(x), groups, per_group), axis=-1)[..., :2].sum(axis=-1)
                changed["bias"] += (_top(raw, k) != chosen).any(axis=-1).sum()
                changed["groups"] += (_top(corrected, k) != chosen).any(axis=-1).sum()
                zeroed = np.where(np.isinf(selection), 0.0, selection)
                changed["zero"] += (_top(zeroed, k) != chosen).any(axis=-1).sum()
                ranked = selection
            else:
                group_scores = logits.reshape(len(x), groups, per_group).max(axis=-1)
                changed["groups"] += (_top(logits, k) != chosen).any(axis=-1).sum()
                ranked = np.where(selection > 0, logits, -np.inf)
            ordered = -np.sort(-ranked, axis=-1)
            assert (ordered[:, k - 1] - ordered[:, k]).min() > 1e-3, name
            if groups > 1:
                ordered = -np.sort(-group_scores, axis=-1)
                assert (ordered[:, cfg.topk_group - 1] - ordered[:, cfg.topk_group]).min() > 1e-3, name
        assert used == set(range(cfg.n_routed_experts)), name
        if groups > 1:
            assert changed["groups"] > 0, name
        if cfg.model_type != "deepseek_v2":
            assert changed["bias"] > 0, name
            if groups > 1:
                assert changed["zero"] > 0, name


def test_deepseek_attention_features_are_live(tmp_path):
    # Guards the fixtures: YaRN's softmax mscale and its attention factor on
    # the RoPE dimensions each move the outputs, and the RoPE dimensions matter.
    spec = DEEPSEEK_VARIANTS["deepseek-v3-kv-b-rotate-half"]
    directory = write_tiny_deepseek(tmp_path / "model", spec)
    oracle = DeepseekReference.load(directory)
    cfg = oracle.config
    assert abs(cfg.attention_factor() - (0.0707 * np.log(32) + 1) / (0.1 * np.log(32) + 1)) < 1e-12
    assert abs(cfg.softmax_scale() - 64**-0.5 * (0.1 * np.log(32) + 1) ** 2) < 1e-12
    expected = oracle.logits(TOKENS)
    scale = np.abs(expected).max()
    for change in ({"mscale": 1.0}, {"mscale_all_dim": 0.707}):
        oracle.config = replace(cfg, rope_scaling={**DEEPSEEK_FACTOR_YARN, **change})
        assert np.abs(oracle.logits(TOKENS) - expected).max() > 1e-3 * scale, change


@pytest.mark.parametrize("backend", _backends())
def test_latent_norms_keep_their_own_epsilon(native_cli, backend, tmp_path, monkeypatch):
    # The query and latent norms use 1e-6 whatever rms_norm_eps says
    # (Moonlight's is 1e-5). Small latents make the difference visible.
    import lokahi.oracles.deepseek as deepseek

    spec = TinyDeepseekSpec(rms_norm_eps=1e-5, latent_scale=0.01, seed=20)
    directory = write_tiny_deepseek(tmp_path / "model", spec)
    _skip_unsupported(directory, backend)
    oracle = DeepseekReference.load(directory)
    expected = oracle.logits(TOKENS)
    monkeypatch.setattr(deepseek, "LATENT_NORM_EPS", 1e-5)
    assert np.abs(oracle.logits(TOKENS) - expected).max() > 0.05 * np.abs(expected).max()
    out = tmp_path / "logits.bin"
    report = _run(native_cli, "logits", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, TOKENS)), "--prefill", str(len(TOKENS)), "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], spec.vocab_size)
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected[-1:], atol=tolerance * np.abs(expected).max(), rtol=0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"partial_rotary_factor": 0.5}, "glm4_moe_lite: partial_rotary_factor is not supported"),
        ({"first_k_dense_replace": 2}, "glm4_moe_lite: first_k_dense_replace other than 1 needs mlp_layer_types"),
        ({"mlp_layer_types": ["dense", "moe", "sparse"]}, "glm4_moe_lite: mlp_layer_types entries must be dense or sparse"),
        ({"mlp_layer_types": ["dense", "sparse"]}, "glm4_moe_lite: mlp_layer_types length mismatch"),
    ],
)
def test_unsupported_glm4_moe_lite_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_deepseek(tmp_path / "model", TinyDeepseekSpec(model_type="glm4_moe_lite", extra=change))
    assert message in _load_error(native_cli, directory)


def test_glm4_moe_lite_reads_mlp_layer_types(native_cli, tmp_path):
    # Transformers lists GLM-4.7-Flash's dense and sparse layers in
    # mlp_layer_types; here the middle layer is dense.
    spec = replace(DEEPSEEK_VARIANTS["glm4.7-flash-shape"], mlp_layer_types=("dense", "dense", "sparse"))
    directory = write_tiny_deepseek(tmp_path / "model", spec)
    oracle = DeepseekReference.load(directory)
    assert [oracle.config.sparse(i) for i in range(3)] == [False, False, True]
    out = tmp_path / "logits.bin"
    report = _run(native_cli, "logits", "--model", str(directory), "--backend", "cpu",
                  "--tokens", ",".join(map(str, TOKENS)), "--prefill", str(len(TOKENS)), "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], spec.vocab_size)
    expected = oracle.logits(TOKENS)[-1:]
    np.testing.assert_allclose(got, expected, atol=2e-4 * np.abs(expected).max(), rtol=0)


def test_kimi_k2_reads_as_deepseek_v3(native_cli, tmp_path):
    directory = write_tiny_deepseek(tmp_path / "model", DEEPSEEK_VARIANTS["deepseek-v3-absorbed-grouped"])
    config = json.loads((directory / "config.json").read_text())
    config["model_type"] = "kimi_k2"
    (directory / "config.json").write_text(json.dumps(config))
    assert _run(native_cli, "info", "--model", str(directory), "--backend", "cpu")["architecture"] == "deepseek_v3"


@pytest.mark.skipif("metal" not in _backends(), reason="needs the Metal backend")
def test_metal_refuses_unquantized_matrices(native_cli, tmp_path):
    directory = write_tiny_deepseek(tmp_path / "model", DEEPSEEK_VARIANTS["deepseek-v2-grouped-bf16"])
    assert "is not quantized; use --backend cpu" in _load_error(native_cli, directory, "metal")


@pytest.mark.skipif("metal" not in _backends(), reason="needs the Metal backend")
def test_metal_refuses_latent_widths_it_lacks(native_cli, tmp_path):
    spec = TinyDeepseekSpec(kv_lora_rank=128, seed=20)  # a 160-wide latent
    directory = write_tiny_deepseek(tmp_path / "model", spec)
    assert "metal: latent attention is built for kv_lora_rank + qk_rope_head_dim of 576" in _load_error(
        native_cli, directory, "metal")


@pytest.mark.parametrize(
    ("model_type", "change", "message"),
    [
        ("deepseek_v3", {"topk_method": "group_limited_greedy"},
         "deepseek_v3: topk_method 'group_limited_greedy' is not supported"),
        ("deepseek_v3", {"scoring_func": "softmax"}, "deepseek_v3: only sigmoid routing is supported"),
        ("deepseek_v2", {"scoring_func": "sigmoid"}, "deepseek_v2: only softmax routing is supported"),
        ("deepseek_v2", {"topk_method": "noaux_tc"}, "deepseek_v2: topk_method 'noaux_tc' is not supported"),
        ("deepseek_v2", {"norm_topk_prob": True}, "deepseek_v2: norm_topk_prob is not supported"),
        ("deepseek_v3", {"moe_layer_freq": 2}, "deepseek_v3: moe_layer_freq other than 1 is not supported"),
        ("deepseek_v3", {"attention_bias": True}, "deepseek_v3: attention_bias is not supported"),
        ("deepseek_v3", {"rope_scaling": {"type": "linear", "factor": 2.0}},
         "deepseek_v3: rope_scaling type 'linear' is not supported"),
        ("deepseek_v3", {"qk_rope_head_dim": 31}, "deepseek_v3: invalid latent attention sizes"),
        ("deepseek_v3", {"n_group": 3}, "deepseek_v3: invalid expert configuration"),
        ("deepseek_v3", {"num_experts_per_tok": 5, "n_group": 4, "topk_group": 2},
         "deepseek_v3: invalid expert configuration"),
    ],
)
def test_unsupported_deepseek_configurations_fail_closed(native_cli, tmp_path, model_type, change, message):
    directory = write_tiny_deepseek(tmp_path / "model", TinyDeepseekSpec(model_type=model_type, extra=change))
    assert message in _load_error(native_cli, directory)


def test_fp8_checkpoints_fail_closed(native_cli, tmp_path):
    # DeepSeek's and Moonshot's originals ship FP8 blocks; MLX-LM converts them.
    fp8 = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    directory = write_tiny_deepseek(tmp_path / "model", TinyDeepseekSpec(bits=0, extra={"quantization_config": fp8}))
    assert "deepseek_v3: fp8 checkpoints are not supported; convert to MLX (or BF16) first" in _load_error(
        native_cli, directory)


def test_deepseek_without_routing_fields_fails_closed(native_cli, tmp_path):
    # Transformers and MLX-LM default the routing fields differently, so the
    # engine reads them without defaults.
    directory = write_tiny_deepseek(tmp_path / "model", TinyDeepseekSpec())
    config = json.loads((directory / "config.json").read_text())
    del config["routed_scaling_factor"]
    (directory / "config.json").write_text(json.dumps(config))
    assert "routed_scaling_factor" in _load_error(native_cli, directory)


def test_deepseek_unexpected_bias_fails_closed(native_cli, tmp_path):
    directory = write_tiny_deepseek(tmp_path / "model", TinyDeepseekSpec())
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    tensors["model.layers.1.self_attn.kv_a_proj_with_mqa.bias"] = ("BF16", np.zeros(96, dtype=np.uint16))
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "layers.1.self_attn.kv_a_proj_with_mqa: unexpected bias for deepseek_v3" in _load_error(native_cli, directory)


def _edit_text_config(directory: Path, change: dict, remove: tuple[str, ...] = ()) -> Path:
    """Applies `change` to the text configuration (text_config when nested)."""

    path = directory / "config.json"
    config = json.loads(path.read_text())
    text = config.get("text_config") or config
    text.update(change)
    for key in remove:
        del text[key]
    path.write_text(json.dumps(config))
    return directory


def test_qwen35_fixtures_route_far_from_ties(tmp_path):
    # Guards the fixtures: on the router's real inputs the top-k choice is
    # stable under float32 error, and every expert serves some token.
    for name, spec in QWEN35_VARIANTS.items():
        oracle = Qwen35Reference.load(write_tiny_qwen35(tmp_path / name, spec))
        if not oracle.config.num_experts:
            continue
        k, route = oracle.config.num_experts_per_tok, oracle.route
        gaps: list[float] = []
        used: set[int] = set()

        def traced(index, x):
            ordered = -np.sort(-(x @ oracle.w[f"model.layers.{index}.mlp.gate"].T), axis=-1)
            gaps.append(float((ordered[:, k - 1] - ordered[:, k]).min()))
            chosen, weights = route(index, x)
            used.update(chosen.ravel().tolist())
            return chosen, weights

        oracle.route = traced
        oracle.logits(TOKENS)
        assert min(gaps) > 1e-3, name
        assert used == set(range(oracle.config.num_experts)), name


def test_qwen35_fixtures_exercise_every_feature(tmp_path):
    # Guards the fixtures: each Gated DeltaNet and gating mechanism moves the
    # outputs, so parity with the oracle covers it. Removing the decay (A = 0),
    # dt_bias, the conv's earlier taps, the output gate's variation, the shared
    # expert's gate or the zero-centered norms' offset must each change the
    # logits.
    for name in ("qwen3.5-mlx", "qwen3.5-moe-hf-fused", "qwen3-next-hf"):
        directory = write_tiny_qwen35(tmp_path / name, QWEN35_VARIANTS[name])
        oracle = Qwen35Reference.load(directory)
        expected = oracle.logits(TOKENS)
        scale = np.abs(expected).max()
        layers = range(oracle.config.num_hidden_layers)
        linear = [i for i in layers if oracle.config.layer_types[i] == "linear_attention"]
        full = [i for i in layers if i not in linear]

        def changed(edit) -> float:
            mutated = Qwen35Reference.load(directory)
            edit(mutated)
            return float(np.abs(mutated.logits(TOKENS) - expected).max())

        def edit_each(names, layers, fn):
            def apply(mutated):
                for i in layers:
                    for suffix in names:
                        key = f"model.layers.{i}.{suffix}"
                        mutated.w[key] = fn(mutated.w[key])
            return apply

        def last_tap_only(conv):
            out = np.zeros_like(conv)
            if conv.shape[-1] == 1:  # MLX [C, K, 1]
                out[:, -1] = conv[:, -1]
            else:  # Transformers [C, 1, K]
                out[..., -1] = conv[..., -1]
            return out

        def constant_gate(q_proj):
            heads, dim = oracle.config.num_attention_heads, oracle.config.head_dim
            rows = q_proj.reshape(heads, 2, dim, -1).copy()
            rows[:, 1] = 0.0  # sigmoid(0) everywhere: the gate no longer depends on x
            return rows.reshape(q_proj.shape) * np.float32(1.0)

        checks = {
            "decay": edit_each(["linear_attn.A_log"], linear, lambda a: np.full_like(a, -np.inf)),
            "dt_bias": edit_each(["linear_attn.dt_bias"], linear, np.zeros_like),
            "conv history": edit_each(["linear_attn.conv1d"], linear, last_tap_only),
            "output gate": edit_each(["self_attn.q_proj"], full, constant_gate),
        }
        if oracle.config.shared_expert_intermediate_size:
            sparse = [i for i in layers if oracle.config.sparse(i)]
            checks["shared expert gate"] = edit_each(["mlp.shared_expert_gate"], sparse, np.zeros_like)
        for feature, edit in checks.items():
            assert changed(edit) > 0.01 * scale, (name, feature)
        offset = Qwen35Reference.load(directory)
        offset.norm_offset = 1.0 - offset.norm_offset
        assert np.abs(offset.logits(TOKENS) - expected).max() > 0.01 * scale, (name, "norm offset")


@pytest.mark.parametrize("backend", _backends())
def test_qwen35_reset_clears_the_recurrent_state(native_cli, backend, tmp_path):
    # Each benchmark repetition resets the model: the Gated DeltaNet states and
    # conv histories must start from zero again, so every repetition matches
    # the oracle (correctness only; no timing is read).
    directory = write_tiny_qwen35(tmp_path / "model", QWEN35_VARIANTS["qwen3.5-mlx"])
    prompt = TOKENS[:6]
    report = _run(native_cli, "bench", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, prompt)), "--max-new", "8", "--warmups", "0", "--repetitions", "2")
    expected = Qwen35Reference.load(directory).greedy(prompt, 8)
    assert [run["token_ids"] for run in report["repetitions"]] == [expected, expected]


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("name", ["qwen3.5-mlx", "qwen3-next-mlx"])
def test_generation_that_stops_resumes_from_the_kept_tokens(native_cli, backend, name, tmp_path):
    # Metal keeps two decode steps in flight and discards the one launched past
    # a stop token. A recurrent state cannot rewind, so the backend must not
    # let that step's update stand: decoding on after the stop has to continue
    # from the tokens generation kept.
    directory = write_tiny_qwen35(tmp_path / "model", QWEN35_VARIANTS[name])
    _skip_unsupported(directory, backend)
    oracle = Qwen35Reference.load(directory)
    prompt = TOKENS[:6]
    expected = oracle.greedy(prompt, 12)
    stop = next(i for i in range(2, 8) if expected[i] not in expected[:i])  # its first occurrence
    _edit_text_config(directory, {"eos_token_id": expected[stop]})
    report = _run(native_cli, "generate", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, prompt)), "--max-new", "12", "--continue", "4")
    assert report["tokens"] == expected[: stop + 1]
    assert report["continued"] == expected[stop + 1 : stop + 5]


@pytest.mark.parametrize("backend", _backends())
def test_generation_config_eos_replaces_the_config_eos(native_cli, backend, tmp_path):
    # Qwen's checkpoints list <|im_end|> only in generation_config.json, whose
    # eos_token_id Transformers and MLX-LM use instead of config.json's.
    directory = write_tiny_qwen35(tmp_path / "model", QWEN35_VARIANTS["qwen3.5-mlx"])
    _skip_unsupported(directory, backend)
    prompt = TOKENS[:6]
    expected = Qwen35Reference.load(directory).greedy(prompt, 12)
    stop = next(i for i in range(2, 8) if expected[i] not in expected[:i])
    unused = next(t for t in range(64) if t not in expected)
    args = ("generate", "--model", str(directory), "--backend", backend, "--tokens", ",".join(map(str, prompt)),
            "--max-new", "12")

    def tokens(config_eos, generation: dict) -> list[int]:
        _edit_text_config(directory, {"eos_token_id": config_eos})
        (directory / "generation_config.json").write_text(json.dumps(generation))
        return _run(native_cli, *args)["tokens"]

    assert tokens(unused, {"eos_token_id": [unused, expected[stop]]}) == expected[: stop + 1]
    # Its list replaces config.json's rather than adding to it...
    assert tokens(expected[stop], {"eos_token_id": unused}) == expected
    # ...and without an eos_token_id there, config.json's stands.
    assert tokens(expected[stop], {"eos_token_id": None, "top_k": 20}) == expected[: stop + 1]
    (directory / "generation_config.json").write_text("[1]")
    assert "generation_config.json must be an object" in _load_error(native_cli, directory)


@pytest.mark.parametrize(
    ("quantization", "message"),
    [
        # 4-bit rows of 128 codes are 16 words, which hold no whole number of
        # 3-bit codes.
        ({"group_size": 32, "bits": 3}, "weight words do not hold whole codes"),
        ({"group_size": 32, "bits": 7}, "unsupported bit width 7"),
        # 16 three-bit codes end mid-word, so groups would not start on words.
        ({"group_size": 16, "bits": 3}, "invalid group size"),
    ],
)
def test_quantization_the_weights_do_not_match_fails_closed(native_cli, tmp_path, quantization, message):
    directory = write_tiny_qwen35(tmp_path / "model", QWEN35_VARIANTS["qwen3.5-mlx"])
    config = json.loads((directory / "config.json").read_text())
    config["quantization"] = config["quantization_config"] = quantization
    (directory / "config.json").write_text(json.dumps(config))
    assert message in _load_error(native_cli, directory)


_FULL_AT_3 = ["linear_attention"] * 3 + ["full_attention", "linear_attention"]
_QWEN35_ROPE = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
                "mrope_interleaved": True}


@pytest.mark.parametrize(
    ("model_type", "change", "remove", "message"),
    [
        ("qwen3_5", {"attn_output_gate": False}, (), "qwen3_5: attn_output_gate false is not supported"),
        ("qwen3_5", {"hidden_act": "gelu"}, (), "qwen3_5: unsupported activation gelu"),
        ("qwen3_5", {"attention_bias": True}, (), "qwen3_5: attention_bias is not supported"),
        # Qwen3.8 names the Gated DeltaNet norm's gate; it is SiLU ("swish").
        ("qwen3_5", {"output_gate_type": "sigmoid"}, (), "qwen3_5: output_gate_type 'sigmoid' is not supported"),
        ("qwen3_5", {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e7, "factor": 4.0}}, (),
         "qwen3_5: rope type 'yarn' is not supported"),
        ("qwen3_5", {"rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.25}}, (), "rope_theta"),
        ("qwen3_5", {"rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.3}},
         (), "qwen3_5: partial_rotary_factor must rotate an even number of dimensions"),
        # Transformers and MLX-VLM interleave the M-RoPE sections whatever the flag says.
        ("qwen3_5", {"rope_parameters": {**_QWEN35_ROPE, "mrope_interleaved": False}}, (),
         "qwen3_5: only interleaved M-RoPE is supported"),
        ("qwen3_5", {"rope_parameters": {**_QWEN35_ROPE, "mrope_section": [4, 4]}}, (),
         "qwen3_5: mrope_section must have 3 entries"),
        ("qwen3_5", {"rope_parameters": {**_QWEN35_ROPE, "mrope_section": [4, -1, 4]}}, (),
         "qwen3_5: mrope_section entries must not be negative"),
        # MLX-LM places full attention by full_attention_interval alone.
        ("qwen3_5", {"layer_types": ["full_attention"] + ["linear_attention"] * 4}, (),
         "qwen3_5: layer_types must place full attention every full_attention_interval layers"),
        ("qwen3_5", {"layer_types": _FULL_AT_3[:4]}, (), "qwen3_5: layer_types length mismatch"),
        ("qwen3_5", {"layer_types": _FULL_AT_3[:4] + ["mamba"]}, (),
         "qwen3_5: layer_types entries must be linear_attention or full_attention"),
        ("qwen3_5", {"full_attention_interval": 1, "layer_types": ["full_attention"] * 5}, (),
         "qwen3_5: expected Gated DeltaNet layers"),
        ("qwen3_5", {"linear_num_value_heads": 3}, (),
         "qwen3_5: linear_num_value_heads must be a multiple of linear_num_key_heads"),
        ("qwen3_5", {"linear_conv_kernel_dim": 1}, (), "qwen3_5: linear_conv_kernel_dim must be at least 2"),
        ("qwen3_5", {"num_experts": 8, "num_experts_per_tok": 2}, (), "qwen3_5: num_experts needs qwen3_5_moe"),
        ("qwen3_5", {}, ("intermediate_size",), "qwen3_5: dense layers need intermediate_size"),
        ("qwen3_5_moe", {"norm_topk_prob": False}, (), "qwen3_5_moe: norm_topk_prob false is not supported"),
        ("qwen3_5_moe", {"mlp_only_layers": [1]}, (), "qwen3_5_moe: mlp_only_layers is not supported"),
        ("qwen3_5_moe", {"decoder_sparse_step": 2}, (),
         "qwen3_5_moe: decoder_sparse_step other than 1 is not supported"),
        ("qwen3_5_moe", {"shared_expert_intermediate_size": 0}, (), "qwen3_5_moe: invalid expert configuration"),
        # Transformers renormalizes by default and MLX-LM does not.
        ("qwen3_next", {}, ("norm_topk_prob",), "norm_topk_prob"),
        ("qwen3_next", {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, (),
         "qwen3_next: rope_scaling type 'yarn' is not supported"),
        ("qwen3_next", {"use_sliding_window": True}, (), "qwen3_next: use_sliding_window is not supported"),
        ("qwen3_next", {"mlp_only_layers": [5]}, (), "qwen3_next: mlp_only_layers entry out of range"),
    ],
)
def test_unsupported_qwen35_configurations_fail_closed(native_cli, tmp_path, model_type, change, remove, message):
    directory = write_tiny_qwen35(tmp_path / "model", TinyQwen35Spec(model_type=model_type))
    assert message in _load_error(native_cli, _edit_text_config(directory, change, remove))


@pytest.mark.parametrize("backend", _backends())
def test_qwen35_l2_norm_keeps_its_epsilon(native_cli, backend, tmp_path, monkeypatch):
    # q and k are L2-normalized as x * rsqrt(sum(x^2) + 1e-6) (Transformers and
    # MLX-LM). Small q and k make the epsilon visible.
    import lokahi.oracles.qwen3_5 as qwen3_5

    directory = write_tiny_qwen35(tmp_path / "model", TinyQwen35Spec(qk_scale=1e-3, seed=0))
    _skip_unsupported(directory, backend)
    oracle = Qwen35Reference.load(directory)
    expected = oracle.logits(TOKENS)
    monkeypatch.setattr(qwen3_5, "L2_EPS", 0.0)
    assert np.abs(oracle.logits(TOKENS) - expected).max() > 0.05 * np.abs(expected).max()
    out = tmp_path / "logits.bin"
    report = _run(native_cli, "logits", "--model", str(directory), "--backend", backend,
                  "--tokens", ",".join(map(str, TOKENS)), "--prefill", "5", "--out", str(out))
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], oracle.config.vocab_size)
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected[4:], atol=tolerance * np.abs(expected).max(), rtol=0)


def test_qwen35_unexpected_bias_fails_closed(native_cli, tmp_path):
    directory = write_tiny_qwen35(tmp_path / "model", TinyQwen35Spec())
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    tensors["language_model.model.layers.1.linear_attn.in_proj_qkv.bias"] = ("BF16", np.zeros(256, dtype=np.uint16))
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "layers.1.linear_attn.in_proj_qkv: unexpected bias for qwen3_5" in _load_error(native_cli, directory)


def test_qwen35_mixed_conv_layouts_fail_closed(native_cli, tmp_path):
    # The conv layout tells Transformers' norms (w) from MLX's (1 + w), so
    # every layer must agree.
    directory = write_tiny_qwen35(tmp_path / "model", TinyQwen35Spec(mlx=False))
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    name = "model.language_model.layers.1.linear_attn.conv1d.weight"
    dtype, conv = tensors[name]
    tensors[name] = (dtype, np.ascontiguousarray(np.swapaxes(conv, 1, 2)))
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "layers.1: conv1d shape" in _load_error(native_cli, directory)
