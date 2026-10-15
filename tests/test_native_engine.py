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
    TinyDenseSpec,
    TinyGemma3Spec,
    TinyGemma4Spec,
    write_tiny_dense,
    write_tiny_gemma3,
    write_tiny_gemma4,
)
from lokahi.formats.safetensors import read_safetensors
from lokahi.oracles import Gemma3Reference, Gemma4Reference, LlamaFamilyReference

TOKENS = [2, 5, 9, 33, 100, 7, 7, 7, 400, 3, 2, 11, 90, 91, 92, 93, 17, 18, 19, 20, 21, 22]


def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _run(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


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
    # Every other layer sparse, top-3 of 6 unnormalized, Transformers'
    # per-expert tensors, untied.
    "qwen3-moe-top3-of-6-per-expert": TinyDenseSpec(
        model_type="qwen3_moe", num_experts=6, num_experts_per_tok=3, decoder_sparse_step=2, norm_topk_prob=False,
        stacked_experts=False, tie_word_embeddings=False, seed=10,
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
    "gemma4-shared-both-kinds": TinyGemma4Spec(
        num_hidden_layers=8, sliding_window_pattern=4, num_kv_shared_layers=3, use_double_wide_mlp=False,
        hidden_size_per_layer_input=16, group_size=16, final_logit_softcapping=None, attention_k_eq_v=True,
        num_key_value_heads=2, num_global_key_value_heads=1, seed=8,
    ),
}
VARIANTS = sorted(GEMMA3_VARIANTS) + sorted(GEMMA4_VARIANTS) + sorted(DENSE_VARIANTS)


@pytest.fixture(scope="module", params=VARIANTS)
def fixture_model(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    if request.param in GEMMA3_VARIANTS:
        write_tiny_gemma3(directory, GEMMA3_VARIANTS[request.param])
        return directory, Gemma3Reference.load(directory)
    if request.param in GEMMA4_VARIANTS:
        write_tiny_gemma4(directory, GEMMA4_VARIANTS[request.param])
        return directory, Gemma4Reference.load(directory)
    write_tiny_dense(directory, DENSE_VARIANTS[request.param])
    return directory, LlamaFamilyReference.load(directory)


CPU_ONLY = ("gemma4", "gemma4_text", "qwen3_moe")  # the Metal backend refuses these for now


def _uses_mxfp4(config: dict) -> bool:
    quant = config.get("quantization") or {}
    return quant.get("mode") == "mxfp4" or any(isinstance(v, dict) and v.get("mode") == "mxfp4" for v in quant.values())


def _skip_unsupported(directory: Path, backend: str) -> None:
    config = json.loads((directory / "config.json").read_text())
    if backend == "metal" and (config.get("text_config") or config)["model_type"] in CPU_ONLY:
        pytest.skip("the Metal backend does not implement this architecture yet")
    if backend == "metal" and _uses_mxfp4(config):
        pytest.skip("the Metal backend does not implement mxfp4 weights yet")


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
        ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "rope_scaling type 'yarn' is not supported"),
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


def _load_error(native_cli: Path, directory: Path, backend: str = "cpu") -> str:
    result = subprocess.run(
        [str(native_cli), "info", "--model", str(directory), "--backend", backend], capture_output=True, text=True
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
    assert "metal: head_dim must be 64, 128 or 256" in result.stderr


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
        ({"enable_moe_block": True}, "gemma4: mixture-of-experts blocks are not supported yet"),
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


def test_gemma4_missing_layer_scalar_fails_closed(native_cli, tmp_path):
    directory = write_tiny_gemma4(tmp_path / "model", TinyGemma4Spec())
    path = directory / "model.safetensors"
    tensors, metadata = read_safetensors(path)
    del tensors["language_model.model.layers.3.layer_scalar"]
    from lokahi.formats.safetensors import write_safetensors

    write_safetensors(path, tensors, metadata=metadata)
    assert "layers.3.layer_scalar" in _load_error(native_cli, directory)


@pytest.mark.parametrize("architecture", ["gemma4", "qwen3_moe", "llama-mxfp4"])
def test_cpu_only_models_fall_back_to_the_cpu(native_cli, tmp_path, architecture):
    if architecture == "gemma4":
        directory = write_tiny_gemma4(tmp_path / "model", TinyGemma4Spec())
    elif architecture == "qwen3_moe":
        directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(model_type="qwen3_moe"))
    else:
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


def test_moe_fixtures_route_far_from_ties(tmp_path):
    # Guards the fixtures: on the router's real inputs, the top-k choice must
    # be stable under float32 error, and every expert must serve some token.
    for name in ("qwen3-moe-top2-of-8", "qwen3-moe-mxfp4-experts", "qwen3-moe-top3-of-6-per-expert"):
        oracle = LlamaFamilyReference.load(write_tiny_dense(tmp_path / name, DENSE_VARIANTS[name]))
        k, route = oracle.config.num_experts_per_tok, oracle.route
        gaps: list[float] = []
        used: set[int] = set()

        def traced(layer, x):
            ordered = -np.sort(-(x @ oracle.w[f"model.layers.{layer}.mlp.gate"].T), axis=-1)
            gaps.append(float((ordered[:, k - 1] - ordered[:, k]).min()))
            chosen, weights = route(layer, x)
            used.update(chosen.ravel().tolist())
            return chosen, weights

        oracle.route = traced
        oracle.logits(TOKENS)
        assert min(gaps) > 1e-3, name
        assert len(used) == oracle.config.num_experts, name


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"shared_expert_intermediate_size": 64}, "qwen3_moe: shared experts are not supported"),
        ({"num_experts_per_tok": 9}, "qwen3_moe: invalid expert configuration"),
        ({"mlp_only_layers": [3]}, "qwen3_moe: mlp_only_layers entry out of range"),
    ],
)
def test_unsupported_moe_configurations_fail_closed(native_cli, tmp_path, change, message):
    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(model_type="qwen3_moe", extra=change))
    assert message in _load_error(native_cli, directory)
