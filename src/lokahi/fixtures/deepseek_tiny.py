"""Write tiny, randomly initialized DeepSeek V2 and V3 checkpoints.

The tensor names and layouts are those of the checkpoints the engine reads:
``mlx-community``'s affine-quantized conversions, whose attention keeps
``kv_b_proj`` (conversions before mid-2026) or stores it absorbed as
``embed_q`` [heads, kv_lora_rank, qk_nope_head_dim] and ``unembed_out``
[heads, v_head_dim, kv_lora_rank] (later ones), with stacked ``switch_mlp``
experts; or Transformers' BF16 layout with one module per expert. Both have
an unquantized BF16 router and, for V3, a float32 ``e_score_correction_bias``.
Multi-head latent attention comes with or without a query rank, and the RoPE
dimensions as interleaved pairs or halves. It is correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec
from ..formats.safetensors import f32_to_bf16, write_safetensors
from .gemma3_tiny import _quantized

# DeepSeek V3's YaRN (V2 and V2-Lite use mscale = mscale_all_dim = 0.707).
DEEPSEEK_YARN = {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096, "beta_fast": 32,
                 "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}


@dataclass(frozen=True)
class TinyDeepseekSpec:
    model_type: str = "deepseek_v3"
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    moe_intermediate_size: int = 64
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    kv_lora_rank: int = 64
    q_lora_rank: int | None = 64  # None: a direct q_proj (V2-Lite, Moonlight)
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    v_head_dim: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_scaling: dict | None = field(default_factory=lambda: dict(DEEPSEEK_YARN))
    rope_interleave: bool | None = None  # None: not written (Transformers' default, interleaved)
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    topk_method: str | None = None  # None: the family's (noaux_tc for V3, greedy for V2)
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    correction_bias_mean: float = 0.0
    latent_scale: float = 1.0  # scales q_a_proj and kv_a_proj, so small latents make the norms' epsilon matter
    absorbed: bool = False  # MLX's embed_q and unembed_out instead of kv_b_proj
    stacked_experts: bool = True  # MLX's switch_mlp; False writes experts.{e}
    bits: int = 4  # 0 writes Transformers' BF16 layout
    group_size: int = 32
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def sparse(self, layer: int) -> bool:
        return layer >= self.first_k_dense_replace

    def config(self) -> dict:
        v3 = self.model_type == "deepseek_v3"
        if self.model_type not in ("deepseek_v2", "deepseek_v3"):
            raise ValueError(f"unknown model_type {self.model_type!r}")
        config = {
            "architectures": ["DeepseekV3ForCausalLM" if v3 else "DeepseekV2ForCausalLM"],
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_shared_experts": self.n_shared_experts,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "topk_method": self.topk_method or ("noaux_tc" if v3 else "greedy"),
            "scoring_func": "sigmoid" if v3 else "softmax",
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,
            "first_k_dense_replace": self.first_k_dense_replace,
            "moe_layer_freq": 1,
            "hidden_act": "silu",
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "max_position_embeddings": 163840,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        }
        if v3:
            config["num_nextn_predict_layers"] = 0
        if self.rope_interleave is not None:
            config["rope_interleave"] = self.rope_interleave
        if self.bits:
            quant = {"group_size": self.group_size, "bits": self.bits}
            config["quantization"] = quant
            config["quantization_config"] = quant
        config.update(self.extra)
        return config


def write_tiny_deepseek(directory: Path, spec: TinyDeepseekSpec = TinyDeepseekSpec()) -> Path:
    """Write ``config.json`` and ``model.safetensors`` into ``directory``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = spec.config()
    rng = np.random.default_rng(spec.seed)
    tensors: dict[str, tuple[str, np.ndarray]] = {}

    def linear(module: str, rows: int, cols: int, std: float, stack: int = 0) -> None:
        """A matrix, or ``stack`` matrices stacked [stack, rows, cols] (experts, or heads)."""

        if not spec.bits:
            shape = (stack, rows, cols) if stack else (rows, cols)
            tensors[f"{module}.weight"] = ("BF16", f32_to_bf16(rng.normal(0.0, std, shape).astype(np.float32)))
            return
        qspec = QuantSpec(group_size=spec.group_size, bits=spec.bits)
        if not stack:
            weight, scales, biases = _quantized(rng, rows, cols, qspec, std)
        else:  # each matrix of the stack quantized on its own
            weight, scales, biases = map(np.stack, zip(*(_quantized(rng, rows, cols, qspec, std) for _ in range(stack))))
        tensors[f"{module}.weight"] = ("U32", weight)
        tensors[f"{module}.scales"] = ("BF16", scales)
        tensors[f"{module}.biases"] = ("BF16", biases)

    def vector(name: str, size: int, mean: float, std: float) -> None:
        tensors[name] = ("BF16", f32_to_bf16(rng.normal(mean, std, size).astype(np.float32)))

    hidden, heads = spec.hidden_size, spec.num_attention_heads
    nope, rope, v_dim, rank = spec.qk_nope_head_dim, spec.qk_rope_head_dim, spec.v_head_dim, spec.kv_lora_rank
    q_out = heads * (nope + rope)
    linear("model.embed_tokens", spec.vocab_size, hidden, 0.6)
    for layer in range(spec.num_hidden_layers):
        base = f"model.layers.{layer}"
        attn = f"{base}.self_attn"
        vector(f"{base}.input_layernorm.weight", hidden, 1.0, 0.2)
        vector(f"{base}.post_attention_layernorm.weight", hidden, 1.0, 0.2)
        if spec.q_lora_rank:
            linear(f"{attn}.q_a_proj", spec.q_lora_rank, hidden, hidden**-0.5 * spec.latent_scale)
            vector(f"{attn}.q_a_layernorm.weight", spec.q_lora_rank, 1.0, 0.2)
            linear(f"{attn}.q_b_proj", q_out, spec.q_lora_rank, spec.q_lora_rank**-0.5)
        else:
            linear(f"{attn}.q_proj", q_out, hidden, hidden**-0.5)
        linear(f"{attn}.kv_a_proj_with_mqa", rank + rope, hidden, hidden**-0.5 * spec.latent_scale)
        vector(f"{attn}.kv_a_layernorm.weight", rank, 1.0, 0.2)
        if spec.absorbed:
            linear(f"{attn}.embed_q", rank, nope, nope**-0.5, heads)
            linear(f"{attn}.unembed_out", v_dim, rank, rank**-0.5, heads)
        else:
            linear(f"{attn}.kv_b_proj", heads * (nope + v_dim), rank, rank**-0.5)
        linear(f"{attn}.o_proj", hidden, heads * v_dim, (heads * v_dim) ** -0.5)
        if not spec.sparse(layer):
            linear(f"{base}.mlp.gate_proj", spec.intermediate_size, hidden, hidden**-0.5)
            linear(f"{base}.mlp.up_proj", spec.intermediate_size, hidden, hidden**-0.5)
            linear(f"{base}.mlp.down_proj", hidden, spec.intermediate_size, spec.intermediate_size**-0.5)
            continue
        experts, moe = spec.n_routed_experts, spec.moe_intermediate_size
        # The router is not an nn.Linear, so MLX leaves it unquantized; a
        # wide one keeps the top-k choice far from ties.
        tensors[f"{base}.mlp.gate.weight"] = (
            "BF16", f32_to_bf16(rng.normal(0.0, hidden**-0.5 * 4.0, (experts, hidden)).astype(np.float32)))
        if spec.model_type == "deepseek_v3":
            tensors[f"{base}.mlp.gate.e_score_correction_bias"] = (
                "F32", rng.normal(spec.correction_bias_mean, 0.3, experts).astype(np.float32))
        if spec.stacked_experts:
            linear(f"{base}.mlp.switch_mlp.gate_proj", moe, hidden, hidden**-0.5, experts)
            linear(f"{base}.mlp.switch_mlp.up_proj", moe, hidden, hidden**-0.5, experts)
            linear(f"{base}.mlp.switch_mlp.down_proj", hidden, moe, moe**-0.5, experts)
        else:
            for e in range(experts):
                linear(f"{base}.mlp.experts.{e}.gate_proj", moe, hidden, hidden**-0.5)
                linear(f"{base}.mlp.experts.{e}.up_proj", moe, hidden, hidden**-0.5)
                linear(f"{base}.mlp.experts.{e}.down_proj", hidden, moe, moe**-0.5)
        if spec.n_shared_experts:
            shared = moe * spec.n_shared_experts
            linear(f"{base}.mlp.shared_experts.gate_proj", shared, hidden, hidden**-0.5)
            linear(f"{base}.mlp.shared_experts.up_proj", shared, hidden, hidden**-0.5)
            linear(f"{base}.mlp.shared_experts.down_proj", hidden, shared, shared**-0.5)
    vector("model.norm.weight", hidden, 1.0, 0.2)
    linear("lm_head", spec.vocab_size, hidden, hidden**-0.5 * 4.0)

    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={"format": "mlx" if spec.bits else "pt", "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True)},
    )
    return directory
