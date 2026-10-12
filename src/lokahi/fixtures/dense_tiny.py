"""Write tiny, randomly initialized Llama-family checkpoints in MLX layout.

One writer covers Llama 3.x, Mistral, Qwen 2/2.5 and Qwen 3 with the same
shapes and tensor names as their ``mlx-community`` checkpoints:
affine-quantized embeddings and projections with BF16 scales and biases,
BF16 norm weights, grouped-query attention, optional ``llama3`` or linear
RoPE scaling, Qwen 3's per-head q/k norms, Qwen 2's q/k/v biases, Mistral's
optional sliding window, tied or untied ``lm_head``, and per-module
quantization overrides. It is correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec
from ..formats.safetensors import f32_to_bf16, write_safetensors
from .gemma3_tiny import _quantized

ARCHITECTURES = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}


@dataclass(frozen=True)
class TinyDenseSpec:
    model_type: str = "llama"
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int | None = None  # None: hidden_size // num_attention_heads, not written
    rope_theta: float = 500_000.0
    rope_scaling: dict | None = None
    sliding_window: int | None = None  # Mistral only
    tie_word_embeddings: bool = True
    group_size: int = 32
    bits: int = 4
    eight_bit_mlp_layers: tuple[int, ...] = (0,)
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def resolved_head_dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    def config(self) -> dict:
        if self.model_type not in ARCHITECTURES:
            raise ValueError(f"unknown model_type {self.model_type!r}")
        quant: dict = {"group_size": self.group_size, "bits": self.bits}
        for layer in self.eight_bit_mlp_layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                quant[f"model.layers.{layer}.mlp.{proj}"] = {"group_size": self.group_size, "bits": 8}
        config = {
            "architectures": [ARCHITECTURES[self.model_type]],
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5 if self.model_type in ("llama", "mistral") else 1e-6,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "max_position_embeddings": 4096,
            "bos_token_id": 1,
            "eos_token_id": [2],
            "tie_word_embeddings": self.tie_word_embeddings,
            "torch_dtype": "bfloat16",
            "quantization": quant,
            "quantization_config": quant,
        }
        if self.head_dim is not None or self.model_type == "qwen3":
            config["head_dim"] = self.resolved_head_dim
        if self.model_type in ("llama", "qwen3"):
            config["attention_bias"] = False
        if self.model_type == "llama":
            config["mlp_bias"] = False
        if self.model_type == "mistral":
            config["sliding_window"] = self.sliding_window
        if self.model_type in ("qwen2", "qwen3"):
            config["use_sliding_window"] = False
            config["sliding_window"] = None if self.model_type == "qwen3" else 32768
            config["max_window_layers"] = self.num_hidden_layers
        config.update(self.extra)
        return config


def write_tiny_dense(directory: Path, spec: TinyDenseSpec = TinyDenseSpec()) -> Path:
    """Write ``config.json`` and ``model.safetensors`` into ``directory``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = spec.config()
    rng = np.random.default_rng(spec.seed)
    tensors: dict[str, tuple[str, np.ndarray]] = {}

    def linear(module: str, rows: int, cols: int, std: float) -> None:
        override = config["quantization"].get(module)
        bits = override["bits"] if isinstance(override, dict) else spec.bits
        weight, scales, biases = _quantized(rng, rows, cols, QuantSpec(group_size=spec.group_size, bits=bits), std)
        tensors[f"{module}.weight"] = ("U32", weight)
        tensors[f"{module}.scales"] = ("BF16", scales)
        tensors[f"{module}.biases"] = ("BF16", biases)

    def vector(name: str, size: int, mean: float, std: float) -> None:
        tensors[name] = ("BF16", f32_to_bf16(rng.normal(mean, std, size).astype(np.float32)))

    hidden, inter, head_dim = spec.hidden_size, spec.intermediate_size, spec.resolved_head_dim
    q_out = spec.num_attention_heads * head_dim
    kv_out = spec.num_key_value_heads * head_dim
    linear("model.embed_tokens", spec.vocab_size, hidden, 0.6)
    for layer in range(spec.num_hidden_layers):
        base = f"model.layers.{layer}"
        vector(f"{base}.input_layernorm.weight", hidden, 1.0, 0.2)
        vector(f"{base}.post_attention_layernorm.weight", hidden, 1.0, 0.2)
        if spec.model_type == "qwen3":
            vector(f"{base}.self_attn.q_norm.weight", head_dim, 1.0, 0.2)
            vector(f"{base}.self_attn.k_norm.weight", head_dim, 1.0, 0.2)
        linear(f"{base}.self_attn.q_proj", q_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.k_proj", kv_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.v_proj", kv_out, hidden, hidden**-0.5)
        if spec.model_type == "qwen2":
            vector(f"{base}.self_attn.q_proj.bias", q_out, 0.0, 0.5)
            vector(f"{base}.self_attn.k_proj.bias", kv_out, 0.0, 0.5)
            vector(f"{base}.self_attn.v_proj.bias", kv_out, 0.0, 0.5)
        linear(f"{base}.self_attn.o_proj", hidden, q_out, q_out**-0.5)
        linear(f"{base}.mlp.gate_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.up_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.down_proj", hidden, inter, inter**-0.5)
    vector("model.norm.weight", hidden, 1.0, 0.2)
    if not spec.tie_word_embeddings:
        linear("lm_head", spec.vocab_size, hidden, hidden**-0.5 * 4.0)

    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={"format": "mlx", "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True)},
    )
    return directory
