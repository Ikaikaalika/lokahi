"""NumPy reference for the Gemma 3 text decoder.

Semantics follow the published Gemma 3 architecture as implemented by Hugging
Face ``Gemma3ForCausalLM`` and MLX-LM ``gemma3_text``:

* token embeddings are multiplied by ``sqrt(hidden_size)`` rounded to BF16;
* RMSNorm scales by ``1 + weight``;
* each layer is ``x += post_attn_norm(attn(input_norm(x)))`` then
  ``x += post_ff_norm(mlp(pre_ff_norm(x)))``;
* queries and keys get per-head RMSNorm before rotate-half RoPE;
* sliding layers use ``rope_local_base_freq`` and attend to the last
  ``sliding_window`` positions; global layers use ``rope_theta`` with optional
  linear ``rope_scaling``;
* attention scale is ``query_pre_attn_scalar ** -0.5``;
* the MLP is GeGLU with the tanh GELU approximation.

Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..formats.mlx_quant import dequantize_affine, quant_spec_for
from ..formats.safetensors import bf16_to_f32, f32_to_bf16, read_safetensors


@dataclass(frozen=True)
class Gemma3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_local_base_freq: float
    rope_linear_factor: float
    query_pre_attn_scalar: float
    sliding_window: int
    layer_is_sliding: tuple[bool, ...]
    final_logit_softcapping: float | None
    attn_logit_softcapping: float | None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Gemma3Config":
        cfg = dict(raw.get("text_config") or raw)
        layers = int(cfg["num_hidden_layers"])
        layer_types = cfg.get("layer_types")
        if layer_types:
            sliding = tuple(kind == "sliding_attention" for kind in layer_types)
        else:
            pattern = int(cfg.get("sliding_window_pattern", 6))
            sliding = tuple((index + 1) % pattern != 0 for index in range(layers))
        if len(sliding) != layers:
            raise ValueError("layer_types length does not match num_hidden_layers")
        scaling = cfg.get("rope_scaling")
        factor = 1.0
        if scaling:
            kind = scaling.get("rope_type", scaling.get("type"))
            if kind == "linear":
                factor = float(scaling["factor"])
            elif kind not in (None, "default"):
                raise NotImplementedError(f"rope_scaling type {kind!r}")
        if cfg.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"activation {cfg.get('hidden_activation')!r}")
        head_dim = int(cfg.get("head_dim", 256))
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=layers,
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
            rope_local_base_freq=float(cfg.get("rope_local_base_freq", 10_000.0)),
            rope_linear_factor=factor,
            query_pre_attn_scalar=float(cfg.get("query_pre_attn_scalar", head_dim)),
            sliding_window=int(cfg.get("sliding_window", 4096)),
            layer_is_sliding=sliding,
            final_logit_softcapping=cfg.get("final_logit_softcapping"),
            attn_logit_softcapping=cfg.get("attn_logit_softcapping"),
        )

    @property
    def embed_scale(self) -> np.float32:
        """``sqrt(hidden_size)`` rounded to BF16, as both references do."""

        return bf16_to_f32(f32_to_bf16(np.array([math.sqrt(self.hidden_size)], np.float32)))[0]


def _tensor_f32(dtype: str, array: np.ndarray) -> np.ndarray:
    if dtype == "BF16":
        return bf16_to_f32(array)
    return np.asarray(array, dtype=np.float32)


def load_dense_weights(model_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load ``config.json`` and every safetensors shard as float32 matrices.

    Returned names have the ``language_model.`` prefix and the quantization
    triple suffixes removed, e.g. ``model.layers.0.mlp.up_proj``. Linear
    biases keep their suffix, e.g. ``model.layers.0.self_attn.q_proj.bias``.
    """

    model_dir = Path(model_dir)
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        shard_tensors, _ = read_safetensors(shard)
        tensors.update(shard_tensors)

    def strip(name: str) -> str:
        return name[len("language_model.") :] if name.startswith("language_model.") else name

    dense: dict[str, np.ndarray] = {}
    for name, (dtype, array) in tensors.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if name.endswith(".bias"):
            dense[strip(name)] = _tensor_f32(dtype, array)
            continue
        if not name.endswith(".weight"):
            continue
        module = name[: -len(".weight")]
        scales = tensors.get(module + ".scales")
        if scales is not None:
            spec = quant_spec_for(raw_config, module)
            if spec is None:
                raise ValueError(f"{module} has scales but no quantization config")
            biases = tensors[module + ".biases"]
            matrix = dequantize_affine(
                array, _tensor_f32(*scales), _tensor_f32(*biases), spec
            )
        else:
            matrix = _tensor_f32(dtype, array)
        dense[strip(module)] = matrix
    return raw_config, dense


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)).astype(np.float32) * (1.0 + weight)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _rope(x: np.ndarray, positions: np.ndarray, theta: float, factor: float) -> np.ndarray:
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)) / factor
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Gemma3Reference:
    """Full-recompute float32 Gemma 3 text model."""

    def __init__(self, config: Gemma3Config, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.w = dict(weights)
        if "lm_head" not in self.w:
            self.w["lm_head"] = self.w["model.embed_tokens"]

    @classmethod
    def load(cls, model_dir: Path) -> "Gemma3Reference":
        raw, weights = load_dense_weights(model_dir)
        return cls(Gemma3Config.from_dict(raw), weights)

    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms_norm(x, self.w[name], self.config.rms_norm_eps)

    def _attention(self, index: int, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        tokens = x.shape[0]
        heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = (x @ self.w[f"{prefix}.q_proj"].T).reshape(tokens, heads, dim)
        k = (x @ self.w[f"{prefix}.k_proj"].T).reshape(tokens, kv_heads, dim)
        v = (x @ self.w[f"{prefix}.v_proj"].T).reshape(tokens, kv_heads, dim)
        q = self._norm(f"{prefix}.q_norm", q)
        k = self._norm(f"{prefix}.k_norm", k)
        sliding = cfg.layer_is_sliding[index]
        theta = cfg.rope_local_base_freq if sliding else cfg.rope_theta
        factor = 1.0 if sliding else cfg.rope_linear_factor
        q = _rope(q, positions, theta, factor)
        k = _rope(k, positions, theta, factor)

        i = positions[:, None]
        j = positions[None, :]
        allowed = j <= i
        if sliding:
            allowed &= (i - j) < cfg.sliding_window
        scale = cfg.query_pre_attn_scalar ** -0.5
        group = heads // kv_heads
        out = np.empty((tokens, heads, dim), dtype=np.float32)
        for head in range(heads):
            kv = head // group
            scores = (q[:, head, :] @ k[:, kv, :].T).astype(np.float64) * scale
            if cfg.attn_logit_softcapping:
                cap = cfg.attn_logit_softcapping
                scores = np.tanh(scores / cap) * cap
            scores = np.where(allowed, scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, head, :] = (probs @ v[:, kv, :]).astype(np.float32)
        return out.reshape(tokens, heads * dim) @ self.w[f"{prefix}.o_proj"].T

    def _mlp(self, index: int, x: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{index}.mlp"
        gate = _gelu_tanh(x @ self.w[f"{prefix}.gate_proj"].T)
        up = x @ self.w[f"{prefix}.up_proj"].T
        return (gate * up) @ self.w[f"{prefix}.down_proj"].T

    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """Return ``[len(token_ids), vocab]`` float32 logits."""

        cfg = self.config
        tokens = np.asarray(token_ids, dtype=np.int64)
        positions = np.arange(tokens.shape[0])
        x = self.w["model.embed_tokens"][tokens] * cfg.embed_scale
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            attn = self._attention(index, self._norm(f"{prefix}.input_layernorm", x), positions)
            x = x + self._norm(f"{prefix}.post_attention_layernorm", attn)
            mlp = self._mlp(index, self._norm(f"{prefix}.pre_feedforward_layernorm", x))
            x = x + self._norm(f"{prefix}.post_feedforward_layernorm", mlp)
        x = self._norm("model.norm", x)
        logits = x @ self.w["lm_head"].T
        if cfg.final_logit_softcapping:
            cap = cfg.final_logit_softcapping
            logits = np.tanh(logits / cap) * cap
        return logits.astype(np.float32)

    def greedy(self, prompt: Sequence[int], steps: int) -> list[int]:
        """Greedy continuation by full recomputation (no KV cache)."""

        tokens = list(prompt)
        generated: list[int] = []
        for _ in range(steps):
            next_token = int(np.argmax(self.logits(tokens)[-1]))
            generated.append(next_token)
            tokens.append(next_token)
        return generated
