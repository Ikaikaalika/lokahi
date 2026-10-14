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

from lokahi.fixtures import TinyDenseSpec, TinyGemma3Spec, write_tiny_dense, write_tiny_gemma3
from lokahi.oracles import Gemma3Reference, LlamaFamilyReference

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
    # Mistral: untied head and a sliding window on every layer.
    "mistral-sliding-untied": TinyDenseSpec(model_type="mistral", sliding_window=8, tie_word_embeddings=False,
                                            rope_theta=1e6, rope_scaling={"rope_type": "linear", "factor": 2.0},
                                            seed=6),
}
VARIANTS = sorted(GEMMA3_VARIANTS) + sorted(DENSE_VARIANTS)


@pytest.fixture(scope="module", params=VARIANTS)
def fixture_model(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    if request.param in GEMMA3_VARIANTS:
        write_tiny_gemma3(directory, GEMMA3_VARIANTS[request.param])
        return directory, Gemma3Reference.load(directory)
    write_tiny_dense(directory, DENSE_VARIANTS[request.param])
    return directory, LlamaFamilyReference.load(directory)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(("prefill", "chunk"), [(len(TOKENS), 512), (5, 512), (1, 512), (17, 3)])
def test_logits_match_oracle(native_cli, fixture_model, backend, prefill, chunk, tmp_path):
    directory, oracle = fixture_model
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
    # with tied embeddings that is all of weight_bytes.
    decode = {section["name"]: section for section in report["decode"]["sections"]}
    linear = ("qkv", "o_proj", "mlp_up", "down_proj", "mlp")  # CPU splits the MLP; Metal fuses it
    linear_bytes = sum(decode[name]["bytes"] for name in linear if name in decode) + decode["lm_head"]["bytes"]
    tied = oracle.w["lm_head"] is oracle.w["model.embed_tokens"]
    assert linear_bytes / steps >= report["weight_bytes"] * (1.0 if tied else 0.5)
