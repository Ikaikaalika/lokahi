"""Write a tiny, randomly initialized Muse Glimmer checkpoint in MLX layout.

The fixture mirrors ``mlx-community/Muse-Glimmer-30B-4bit``: the text model
nested in ``text_config`` with ``layer_types`` and ``layer_rope_theta`` (RoPE
on sliding layers, none on full ones), the decoder under ``language_model.``,
affine-quantized embeddings and projections (attention's ``gate_proj``
included) with BF16 scales and biases, BF16 norms (the four block norms
stored around 0, the final norm around 1), an untied ``lm_head``, per-module
quantization overrides, and optionally vision-tower tensors the text model
must ignore. It is correctness evidence only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec
from ..formats.safetensors import f32_to_bf16, write_safetensors
from .gemma3_tiny import _quantized


@dataclass(frozen=True)
class TinyMuseGlimmerSpec:
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    sliding_window: int = 8
    rope_theta: float = 500_000.0
    explicit_layer_rope_theta: bool = True  # else the references derive it from layer_types
    flat_config: bool = False  # MLX-LM's flattened text config instead of text_config
    group_size: int = 32
    bits: int = 4
    eight_bit_mlp_layers: tuple[int, ...] = (0,)
    vision_tower: bool = True
    seed: int = 0
    prefix: str = "language_model."
    extra: dict = field(default_factory=dict)

    def layer_types(self) -> list[str]:
        return ["full_attention" if i % 4 == 3 else "sliding_attention" for i in range(self.num_hidden_layers)]

    def config(self) -> dict:
        quant: dict = {"group_size": self.group_size, "bits": self.bits, "mode": "affine"}
        for layer in self.eight_bit_mlp_layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                quant[f"{self.prefix}model.layers.{layer}.mlp.{proj}"] = {"group_size": self.group_size, "bits": 8}
        types = self.layer_types()
        text = {
            "model_type": "muse_glimmer_text",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_activation": "silu",
            "attention_bias": False,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": (self.hidden_size / 256) ** -0.5,
            "final_logit_softcapping": 20.0,
            "sliding_window": self.sliding_window,
            "layer_types": types,
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
            "max_position_embeddings": 4096,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "tie_word_embeddings": False,
        }
        if self.explicit_layer_rope_theta:
            text["layer_rope_theta"] = [0 if kind == "full_attention" else self.rope_theta for kind in types]
        if self.flat_config:
            config = {**text, "model_type": "muse_glimmer"}
        else:
            config = {
                "architectures": ["MuseGlimmerForConditionalGeneration"],
                "model_type": "muse_glimmer",
                "eos_token_id": [1, 3],
                "image_token_id": self.vocab_size - 1,
                "text_config": text,
                "vision_config": {"model_type": "muse_glimmer_vision", "hidden_size": 32},
            }
        config["quantization"] = quant
        config["quantization_config"] = quant
        config.update(self.extra)
        return config


def write_tiny_muse_glimmer(directory: Path, spec: TinyMuseGlimmerSpec = TinyMuseGlimmerSpec()) -> Path:
    """Write ``config.json`` and ``model.safetensors`` into ``directory``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = spec.config()
    rng = np.random.default_rng(spec.seed)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    p = spec.prefix

    def linear(module: str, rows: int, cols: int, std: float) -> None:
        override = config["quantization"].get(module)
        bits = override["bits"] if isinstance(override, dict) else spec.bits
        weight, scales, biases = _quantized(rng, rows, cols, QuantSpec(group_size=spec.group_size, bits=bits), std)
        tensors[f"{module}.weight"] = ("U32", weight)
        tensors[f"{module}.scales"] = ("BF16", scales)
        tensors[f"{module}.biases"] = ("BF16", biases)

    def norm(name: str, size: int, center: float) -> None:
        values = center + rng.normal(0.0, 0.2, size)
        tensors[f"{name}.weight"] = ("BF16", f32_to_bf16(values.astype(np.float32)))

    hidden, inter = spec.hidden_size, spec.intermediate_size
    q_out = spec.num_attention_heads * spec.head_dim
    kv_out = spec.num_key_value_heads * spec.head_dim
    linear(f"{p}model.embed_tokens", spec.vocab_size, hidden, 0.6)
    for layer in range(spec.num_hidden_layers):
        base = f"{p}model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            norm(f"{base}.{name}", hidden, 0.0)
        linear(f"{base}.self_attn.q_proj", q_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.k_proj", kv_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.v_proj", kv_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.gate_proj", q_out, hidden, hidden**-0.5 * 2.0)
        linear(f"{base}.self_attn.o_proj", hidden, q_out, q_out**-0.5)
        linear(f"{base}.mlp.gate_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.up_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.down_proj", hidden, inter, inter**-0.5)
    norm(f"{p}model.norm", hidden, 1.0)
    linear(f"{p}lm_head", spec.vocab_size, hidden, hidden**-0.5 * 4.0)
    if spec.vision_tower:
        # Tensors of the image path, which the text decoder must not read.
        tensors["vision_tower.patch_embed.proj.weight"] = (
            "BF16", f32_to_bf16(rng.normal(0.0, 0.1, (32, 3 * 14 * 14)).astype(np.float32)))
        tensors["vision_projection.linear_1.weight"] = (
            "BF16", f32_to_bf16(rng.normal(0.0, 0.1, (hidden, 32)).astype(np.float32)))

    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={"format": "mlx", "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True)},
    )
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(write_tiny_muse_glimmer(args.output, TinyMuseGlimmerSpec(num_hidden_layers=args.layers, seed=args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
