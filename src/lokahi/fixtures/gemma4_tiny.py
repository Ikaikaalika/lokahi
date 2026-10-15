"""Write a tiny, randomly initialized Gemma 4 text checkpoint in MLX layout.

The default spec has the shape of mlx-community's Gemma 4 E2B, scaled down:
- sliding and full layers with different head sizes, and proportional
  partial RoPE on full layers;
- the last layers sharing earlier layers' keys and values, with
  double-wide MLPs and no k/v weights, as the converted checkpoints store
  them;
- per-layer input embeddings and projections, and a stored layer_scalar
  per layer;
- tied embeddings and a final logit soft-cap;
- nested text_config with the language_model. prefix.

attention_k_eq_v and num_global_key_value_heads cover the 26B/31B
variant. Correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec
from ..formats.safetensors import f32_to_bf16, write_safetensors
from .gemma3_tiny import _quantized


@dataclass(frozen=True)
class TinyGemma4Spec:
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 128
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    num_global_key_value_heads: int | None = None
    head_dim: int = 64
    global_head_dim: int = 128
    global_partial_rotary_factor: float = 0.25
    sliding_window: int = 8
    sliding_window_pattern: int = 3
    num_kv_shared_layers: int = 2
    hidden_size_per_layer_input: int = 32
    use_double_wide_mlp: bool = True
    attention_k_eq_v: bool = False
    tie_word_embeddings: bool = True
    final_logit_softcapping: float | None = 30.0
    nested: bool = True  # text_config and the language_model. prefix, as mlx-community ships it
    group_size: int = 32
    bits: int = 4
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def layer_types(self) -> list[str]:
        cycle = ["sliding_attention"] * (self.sliding_window_pattern - 1) + ["full_attention"]
        return (cycle * (self.num_hidden_layers // len(cycle) + 1))[: self.num_hidden_layers]

    def text_config(self) -> dict:
        config = {
            "model_type": "gemma4_text",
            "vocab_size": self.vocab_size,
            "vocab_size_per_layer_input": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_global_key_value_heads": self.num_global_key_value_heads,
            "head_dim": self.head_dim,
            "global_head_dim": self.global_head_dim,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "layer_types": self.layer_types(),
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": self.global_partial_rotary_factor,
                                   "rope_theta": 1_000_000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            },
            "sliding_window": self.sliding_window,
            "num_kv_shared_layers": self.num_kv_shared_layers,
            "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
            "use_double_wide_mlp": self.use_double_wide_mlp,
            "attention_k_eq_v": self.attention_k_eq_v,
            "enable_moe_block": False,
            "final_logit_softcapping": self.final_logit_softcapping,
            "tie_word_embeddings": self.tie_word_embeddings,
            "eos_token_id": 1,
        }
        config.update(self.extra)
        return config

    def config(self) -> dict:
        quant = {"group_size": self.group_size, "bits": self.bits, "mode": "affine"}
        if not self.nested:
            return {**self.text_config(), "quantization": quant, "quantization_config": quant}
        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": self.text_config(),
            "eos_token_id": [1, 106],
            "tie_word_embeddings": self.tie_word_embeddings,
            "quantization": quant,
            "quantization_config": quant,
        }


def write_tiny_gemma4(directory: Path, spec: TinyGemma4Spec = TinyGemma4Spec()) -> Path:
    """Write ``config.json`` and ``model.safetensors`` into ``directory``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    p = "language_model." if spec.nested else ""
    qspec = QuantSpec(group_size=spec.group_size, bits=spec.bits)

    def linear(module: str, rows: int, cols: int, std: float) -> None:
        weight, scales, biases = _quantized(rng, rows, cols, qspec, std)
        tensors[f"{module}.weight"] = ("U32", weight)
        tensors[f"{module}.scales"] = ("BF16", scales)
        tensors[f"{module}.biases"] = ("BF16", biases)

    def norm(name: str, size: int) -> None:
        # Gemma 4 norms multiply by the weight itself, so it sits around 1.
        tensors[f"{name}.weight"] = ("BF16", f32_to_bf16(rng.normal(1.0, 0.2, size).astype(np.float32)))

    hidden, layers, size = spec.hidden_size, spec.num_hidden_layers, spec.hidden_size_per_layer_input
    types = spec.layer_types()
    first_shared = layers - spec.num_kv_shared_layers
    linear(f"{p}model.embed_tokens", spec.vocab_size, hidden, 0.6)
    if size:
        linear(f"{p}model.embed_tokens_per_layer", spec.vocab_size, layers * size, 0.6)
        linear(f"{p}model.per_layer_model_projection", layers * size, hidden, hidden**-0.5)
        norm(f"{p}model.per_layer_projection_norm", size)
    for layer in range(layers):
        base = f"{p}model.layers.{layer}"
        full = types[layer] == "full_attention"
        dim = spec.global_head_dim if full else spec.head_dim
        k_eq_v = spec.attention_k_eq_v and full
        kv_heads = spec.num_global_key_value_heads if k_eq_v and spec.num_global_key_value_heads else spec.num_key_value_heads
        shared = spec.num_kv_shared_layers > 0 and layer >= first_shared
        inter = spec.intermediate_size * (2 if spec.use_double_wide_mlp and shared else 1)
        q_out = spec.num_attention_heads * dim
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            norm(f"{base}.{name}", hidden)
        norm(f"{base}.self_attn.q_norm", dim)
        linear(f"{base}.self_attn.q_proj", q_out, hidden, hidden**-0.5)
        if not shared:  # the converted checkpoints drop k/v weights of shared layers
            norm(f"{base}.self_attn.k_norm", dim)
            linear(f"{base}.self_attn.k_proj", kv_heads * dim, hidden, hidden**-0.5)
            if not k_eq_v:
                linear(f"{base}.self_attn.v_proj", kv_heads * dim, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.o_proj", hidden, q_out, q_out**-0.5)
        linear(f"{base}.mlp.gate_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.up_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.down_proj", hidden, inter, inter**-0.5)
        if size:
            linear(f"{base}.per_layer_input_gate", size, hidden, hidden**-0.5)
            linear(f"{base}.per_layer_projection", hidden, size, size**-0.5)
            norm(f"{base}.post_per_layer_input_norm", hidden)
        tensors[f"{base}.layer_scalar"] = ("BF16", f32_to_bf16(rng.uniform(0.6, 1.2, 1).astype(np.float32)))
    norm(f"{p}model.norm", hidden)
    if not spec.tie_word_embeddings:
        linear(f"{p}lm_head", spec.vocab_size, hidden, hidden**-0.5 * 4.0)

    (directory / "config.json").write_text(json.dumps(spec.config(), indent=2) + "\n", encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={"format": "mlx", "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True)},
    )
    return directory
