"""Write tiny, randomly initialized Qwen3.5 and Qwen3-Next text checkpoints.

Two layouts, as the checkpoints the engine reads store them:

* MLX (MLX-LM's conversions): affine-quantized projections with BF16 scales
  and biases, the zero-centered norms stored as ``1 + w``, depthwise conv
  weights as [C, K, 1], stacked ``switch_mlp`` experts with an 8-bit router
  and shared-expert gate; Qwen3.5 under ``language_model.`` (MLX-LM keeps
  the multimodal model's prefix), with its ``text_config``;
* Transformers: BF16 weights, the norms' ``w`` as is, conv weights as
  [C, 1, K]; Qwen3.5 under ``model.language_model.`` with fused
  ``experts.gate_up_proj`` [E, 2I, H] and ``experts.down_proj`` tensors (as
  Transformers saves the multimodal models), Qwen3-Next under ``model.`` with
  one module per expert.

Every model has Gated DeltaNet layers, some full attention, and optionally
mixtures of experts with a shared expert. It is correctness evidence only.
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
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    "qwen3_next": "Qwen3NextForCausalLM",
}


@dataclass(frozen=True)
class TinyQwen35Spec:
    model_type: str = "qwen3_5"
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 5
    full_attention_interval: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    linear_num_key_heads: int = 2
    linear_num_value_heads: int = 4
    linear_key_head_dim: int = 32
    linear_value_head_dim: int = 32
    linear_conv_kernel_dim: int = 4
    rms_norm_eps: float = 1e-6
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 64
    shared_expert_intermediate_size: int = 64
    decoder_sparse_step: int = 1  # Qwen3-Next
    mlp_only_layers: tuple[int, ...] = ()  # Qwen3-Next
    norm_topk_prob: bool = True  # Qwen3-Next
    tie_word_embeddings: bool = False
    qk_scale: float = 1.0  # scales the q and k channels' conv taps; small q and k make the L2 norm's epsilon matter
    mlx: bool = True  # False: Transformers' BF16 layout
    bits: int = 4
    group_size: int = 32
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def moe(self) -> bool:
        return self.model_type in ("qwen3_5_moe", "qwen3_next")

    def layer_types(self) -> list[str]:
        interval = self.full_attention_interval
        return ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(self.num_hidden_layers)]

    def sparse(self, layer: int) -> bool:
        if self.model_type == "qwen3_5_moe":
            return True
        return (self.model_type == "qwen3_next" and layer not in self.mlp_only_layers
                and (layer + 1) % self.decoder_sparse_step == 0)

    @property
    def prefix(self) -> str:
        return "language_model." if self.mlx and self.model_type != "qwen3_next" else ""

    @property
    def stem(self) -> str:
        """Where the decoder's modules sit under the prefix."""

        return "model.language_model." if not self.mlx and self.model_type != "qwen3_next" else "model."

    def text_config(self) -> dict:
        text = {
            "model_type": {"qwen3_5": "qwen3_5_text", "qwen3_5_moe": "qwen3_5_moe_text"}.get(self.model_type,
                                                                                          self.model_type),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "rms_norm_eps": self.rms_norm_eps,
            "layer_types": self.layer_types(),
            "full_attention_interval": self.full_attention_interval,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "attention_bias": False,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": self.tie_word_embeddings,
            "eos_token_id": 2,
        }
        rope = {"rope_type": "default", "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor}
        if self.model_type == "qwen3_next":
            text.update(rope_theta=self.rope_theta, partial_rotary_factor=self.partial_rotary_factor,
                        decoder_sparse_step=self.decoder_sparse_step, mlp_only_layers=list(self.mlp_only_layers),
                        norm_topk_prob=self.norm_topk_prob, intermediate_size=self.intermediate_size)
        else:
            text["rope_parameters"] = {**rope, "mrope_section": [11, 11, 10], "mrope_interleaved": True}
            text.update(attn_output_gate=True, mlp_only_layers=[], mamba_ssm_dtype="float32")
        if self.model_type == "qwen3_5":
            text["intermediate_size"] = self.intermediate_size
        if self.moe:
            text.update(num_experts=self.num_experts, num_experts_per_tok=self.num_experts_per_tok,
                        moe_intermediate_size=self.moe_intermediate_size,
                        shared_expert_intermediate_size=self.shared_expert_intermediate_size)
        return text

    def config(self) -> dict:
        if self.model_type not in ARCHITECTURES:
            raise ValueError(f"unknown model_type {self.model_type!r}")
        text = self.text_config()
        if self.model_type == "qwen3_next":
            config = {"architectures": [ARCHITECTURES["qwen3_next"]], **text}
        else:
            config = {
                "architectures": [ARCHITECTURES[self.model_type]],
                "model_type": self.model_type,
                "text_config": text,
                "tie_word_embeddings": self.tie_word_embeddings,
                "image_token_id": 248056,
                "vision_start_token_id": 248053,
                "vision_end_token_id": 248054,
            }
        if self.mlx:
            quant: dict = {"group_size": self.group_size, "bits": self.bits}
            for layer in range(self.num_hidden_layers):
                if self.sparse(layer):
                    base = f"{self.prefix}model.layers.{layer}.mlp"
                    quant[f"{base}.gate"] = {"group_size": 64, "bits": 8}
                    quant[f"{base}.shared_expert_gate"] = {"group_size": 64, "bits": 8}
            config["quantization"] = quant
            config["quantization_config"] = quant
        config.update(self.extra)
        return config


def write_tiny_qwen35(directory: Path, spec: TinyQwen35Spec = TinyQwen35Spec()) -> Path:
    """Write ``config.json`` and ``model.safetensors`` into ``directory``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = spec.config()
    rng = np.random.default_rng(spec.seed)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    p, stem = spec.prefix, spec.stem

    def linear(module: str, rows: int, cols: int, std: float, stack: int = 0) -> None:
        if not spec.mlx:
            shape = (stack, rows, cols) if stack else (rows, cols)
            tensors[f"{module}.weight"] = ("BF16", f32_to_bf16(rng.normal(0.0, std, shape).astype(np.float32)))
            return
        override = config["quantization"].get(module)
        source = override if isinstance(override, dict) else config["quantization"]
        qspec = QuantSpec(group_size=source["group_size"], bits=source["bits"])
        if not stack:
            weight, scales, biases = _quantized(rng, rows, cols, qspec, std)
        else:
            weight, scales, biases = map(np.stack, zip(*(_quantized(rng, rows, cols, qspec, std) for _ in range(stack))))
        tensors[f"{module}.weight"] = ("U32", weight)
        tensors[f"{module}.scales"] = ("BF16", scales)
        tensors[f"{module}.biases"] = ("BF16", biases)

    def vector(name: str, size: int, mean: float, std: float, dtype: str = "BF16") -> None:
        values = rng.normal(mean, std, size).astype(np.float32)
        tensors[name] = (dtype, f32_to_bf16(values) if dtype == "BF16" else values)

    def norm(name: str, size: int) -> None:
        # Zero-centered: the model scales by 1 + w; MLX stores 1 + w.
        vector(name, size, 1.0 if spec.mlx else 0.0, 0.2)

    hidden, heads, kv_heads, dim = spec.hidden_size, spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim
    hk, hv, dk, dv = (spec.linear_num_key_heads, spec.linear_num_value_heads, spec.linear_key_head_dim,
                      spec.linear_value_head_dim)
    conv_dim, kernel = 2 * hk * dk + hv * dv, spec.linear_conv_kernel_dim
    linear(f"{p}{stem}embed_tokens", spec.vocab_size, hidden, 0.6)
    for layer, kind in enumerate(spec.layer_types()):
        base = f"{p}{stem}layers.{layer}"
        norm(f"{base}.input_layernorm.weight", hidden)
        norm(f"{base}.post_attention_layernorm.weight", hidden)
        if kind == "full_attention":
            attn = f"{base}.self_attn"
            linear(f"{attn}.q_proj", heads * dim * 2, hidden, hidden**-0.5)
            linear(f"{attn}.k_proj", kv_heads * dim, hidden, hidden**-0.5)
            linear(f"{attn}.v_proj", kv_heads * dim, hidden, hidden**-0.5)
            linear(f"{attn}.o_proj", hidden, heads * dim, (heads * dim) ** -0.5)
            norm(f"{attn}.q_norm.weight", dim)
            norm(f"{attn}.k_norm.weight", dim)
        else:
            lin = f"{base}.linear_attn"
            if spec.model_type == "qwen3_next":
                r = hv // hk
                linear(f"{lin}.in_proj_qkvz", hk * (2 * dk + 2 * r * dv), hidden, hidden**-0.5)
                linear(f"{lin}.in_proj_ba", hk * 2 * r, hidden, hidden**-0.5 * 2.0)
            else:
                linear(f"{lin}.in_proj_qkv", conv_dim, hidden, hidden**-0.5)
                linear(f"{lin}.in_proj_z", hv * dv, hidden, hidden**-0.5)
                linear(f"{lin}.in_proj_b", hv, hidden, hidden**-0.5 * 2.0)
                linear(f"{lin}.in_proj_a", hv, hidden, hidden**-0.5 * 2.0)
            conv = rng.normal(0.0, 0.5, (conv_dim, kernel)).astype(np.float32)
            conv[: 2 * hk * dk] *= np.float32(spec.qk_scale)
            conv = conv[:, :, None] if spec.mlx else conv[:, None, :]
            tensors[f"{lin}.conv1d.weight"] = ("BF16", f32_to_bf16(conv))
            # Decays between about 0.4 and 0.95 per token so the state keeps
            # several tokens' worth of history.
            tensors[f"{lin}.A_log"] = ("F32", np.log(rng.uniform(0.05, 0.5, hv)).astype(np.float32))
            vector(f"{lin}.dt_bias", hv, 0.5, 0.5)
            vector(f"{lin}.norm.weight", dv, 1.0, 0.2)
            linear(f"{lin}.out_proj", hidden, hv * dv, (hv * dv) ** -0.5)
        mlp = f"{base}.mlp"
        if not spec.sparse(layer):
            linear(f"{mlp}.gate_proj", spec.intermediate_size, hidden, hidden**-0.5)
            linear(f"{mlp}.up_proj", spec.intermediate_size, hidden, hidden**-0.5)
            linear(f"{mlp}.down_proj", hidden, spec.intermediate_size, spec.intermediate_size**-0.5)
            continue
        experts, moe = spec.num_experts, spec.moe_intermediate_size
        # A wide router keeps the top-k choice far from ties.
        linear(f"{mlp}.gate", experts, hidden, hidden**-0.5 * 4.0)
        if spec.mlx:
            linear(f"{mlp}.switch_mlp.gate_proj", moe, hidden, hidden**-0.5, experts)
            linear(f"{mlp}.switch_mlp.up_proj", moe, hidden, hidden**-0.5, experts)
            linear(f"{mlp}.switch_mlp.down_proj", hidden, moe, moe**-0.5, experts)
        elif spec.model_type == "qwen3_5_moe":  # fused parameters, no .weight suffix
            gate_up = rng.normal(0.0, hidden**-0.5, (experts, 2 * moe, hidden)).astype(np.float32)
            down = rng.normal(0.0, moe**-0.5, (experts, hidden, moe)).astype(np.float32)
            tensors[f"{mlp}.experts.gate_up_proj"] = ("BF16", f32_to_bf16(gate_up))
            tensors[f"{mlp}.experts.down_proj"] = ("BF16", f32_to_bf16(down))
        else:
            for e in range(experts):
                linear(f"{mlp}.experts.{e}.gate_proj", moe, hidden, hidden**-0.5)
                linear(f"{mlp}.experts.{e}.up_proj", moe, hidden, hidden**-0.5)
                linear(f"{mlp}.experts.{e}.down_proj", hidden, moe, moe**-0.5)
        shared = spec.shared_expert_intermediate_size
        linear(f"{mlp}.shared_expert.gate_proj", shared, hidden, hidden**-0.5)
        linear(f"{mlp}.shared_expert.up_proj", shared, hidden, hidden**-0.5)
        linear(f"{mlp}.shared_expert.down_proj", hidden, shared, shared**-0.5)
        linear(f"{mlp}.shared_expert_gate", 1, hidden, hidden**-0.5 * 2.0)
    norm(f"{p}{stem}norm.weight", hidden)
    if not spec.tie_word_embeddings:
        linear(f"{p}lm_head", spec.vocab_size, hidden, hidden**-0.5 * 4.0)

    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={"format": "mlx" if spec.mlx else "pt", "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True)},
    )
    return directory
