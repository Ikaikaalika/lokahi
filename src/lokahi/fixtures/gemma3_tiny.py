"""Write a tiny, randomly initialized Gemma 3 checkpoint in MLX layout.

The fixture exercises the same code paths as a real ``mlx-community`` Gemma 3
checkpoint: affine-quantized embeddings and projections with BF16 scales and
biases, BF16 norm weights, sliding and global layers, grouped-query attention,
optional linear RoPE scaling, an optional untied ``lm_head``, and per-module
quantization overrides. It is correctness evidence only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec, pack_affine
from ..formats.safetensors import f32_to_bf16, write_safetensors


@dataclass(frozen=True)
class TinyGemma3Spec:
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    sliding_window: int = 8
    sliding_window_pattern: int = 2
    rope_linear_factor: float | None = None
    tie_word_embeddings: bool = False
    group_size: int = 32
    bits: int = 4
    eight_bit_mlp_layers: tuple[int, ...] = (0,)
    seed: int = 0
    prefix: str = ""
    extra: dict = field(default_factory=dict)

    def config(self) -> dict:
        quant: dict = {"group_size": self.group_size, "bits": self.bits, "mode": "affine"}
        for layer in self.eight_bit_mlp_layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                quant[f"{self.prefix}model.layers.{layer}.mlp.{proj}"] = {
                    "group_size": self.group_size,
                    "bits": 8,
                }
        config = {
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "rope_scaling": (
                {"rope_type": "linear", "factor": self.rope_linear_factor}
                if self.rope_linear_factor
                else None
            ),
            "query_pre_attn_scalar": self.head_dim,
            "sliding_window": self.sliding_window,
            "sliding_window_pattern": self.sliding_window_pattern,
            "final_logit_softcapping": None,
            "attn_logit_softcapping": None,
            "max_position_embeddings": 4096,
            "bos_token_id": 2,
            "eos_token_id": [1],
            "tie_word_embeddings": self.tie_word_embeddings,
            "torch_dtype": "bfloat16",
            "quantization": quant,
            "quantization_config": quant,
        }
        config.update(self.extra)
        return config


def _quantized(
    rng: np.random.Generator, rows: int, cols: int, spec: QuantSpec, std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = (1 << spec.bits) - 1
    groups = cols // spec.group_size
    codes = rng.integers(0, levels + 1, size=(rows, cols), dtype=np.uint32)
    # A uniform code in [0, levels] has std ~ levels / sqrt(12).
    scale = rng.uniform(0.5, 1.5, size=(rows, groups)) * std * np.sqrt(12.0) / levels
    bias = -scale * levels / 2.0 + rng.normal(0.0, std * 0.1, size=(rows, groups))
    return (
        pack_affine(codes, spec.bits),
        f32_to_bf16(scale.astype(np.float32)),
        f32_to_bf16(bias.astype(np.float32)),
    )


def _mxfp4(rng: np.random.Generator, rows: int, cols: int, std: float) -> tuple[np.ndarray, np.ndarray]:
    """Random MXFP4 codes and E8M0 exponents whose values have about ``std``."""

    codes = rng.integers(0, 16, size=(rows, cols), dtype=np.uint32)
    # Uniform E2M1 codes have a root mean square of sqrt(8.5625), about 2.93.
    exponent = 127 + int(round(np.log2(std / np.sqrt(8.5625))))
    exponents = exponent + rng.integers(-1, 2, size=(rows, cols // 32))
    return pack_affine(codes, 4), exponents.astype(np.uint8)


def write_tiny_gemma3(directory: Path, spec: TinyGemma3Spec = TinyGemma3Spec()) -> Path:
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
        qspec = QuantSpec(group_size=spec.group_size, bits=bits)
        weight, scales, biases = _quantized(rng, rows, cols, qspec, std)
        tensors[f"{module}.weight"] = ("U32", weight)
        tensors[f"{module}.scales"] = ("BF16", scales)
        tensors[f"{module}.biases"] = ("BF16", biases)

    def norm(name: str, size: int) -> None:
        tensors[f"{name}.weight"] = ("BF16", f32_to_bf16(rng.normal(0.0, 0.2, size).astype(np.float32)))

    hidden, inter = spec.hidden_size, spec.intermediate_size
    q_out = spec.num_attention_heads * spec.head_dim
    kv_out = spec.num_key_value_heads * spec.head_dim
    linear(f"{p}model.embed_tokens", spec.vocab_size, hidden, 0.6)
    for layer in range(spec.num_hidden_layers):
        base = f"{p}model.layers.{layer}"
        norm(f"{base}.input_layernorm", hidden)
        norm(f"{base}.post_attention_layernorm", hidden)
        norm(f"{base}.pre_feedforward_layernorm", hidden)
        norm(f"{base}.post_feedforward_layernorm", hidden)
        norm(f"{base}.self_attn.q_norm", spec.head_dim)
        norm(f"{base}.self_attn.k_norm", spec.head_dim)
        linear(f"{base}.self_attn.q_proj", q_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.k_proj", kv_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.v_proj", kv_out, hidden, hidden**-0.5)
        linear(f"{base}.self_attn.o_proj", hidden, q_out, q_out**-0.5)
        linear(f"{base}.mlp.gate_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.up_proj", inter, hidden, hidden**-0.5)
        linear(f"{base}.mlp.down_proj", hidden, inter, inter**-0.5)
    norm(f"{p}model.norm", hidden)
    if not spec.tie_word_embeddings:
        linear("lm_head", spec.vocab_size, hidden, hidden**-0.5 * 4.0)

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
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--rope-linear-factor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = TinyGemma3Spec(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        head_dim=args.head_dim,
        tie_word_embeddings=args.tied,
        rope_linear_factor=args.rope_linear_factor,
        seed=args.seed,
    )
    print(write_tiny_gemma3(args.output, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
