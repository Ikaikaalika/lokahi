"""NumPy reference for the Llama family: Llama 3.x, Mistral, Qwen 2/2.5, Qwen 3.

Semantics follow the Hugging Face ``LlamaForCausalLM``, ``MistralForCausalLM``,
``Qwen2ForCausalLM`` and ``Qwen3ForCausalLM`` implementations:

* pre-norm blocks: ``x += attn(norm(x))`` then ``x += mlp(norm(x))``, where
  the norm before the MLP is named ``post_attention_layernorm``;
* RMSNorm scales by ``weight`` (not ``1 + weight``);
* rotate-half RoPE with ``rope_theta`` and optional ``linear`` or ``llama3``
  frequency scaling;
* Qwen 3 applies per-head RMSNorm to queries and keys before RoPE;
* Qwen 2 adds biases to the q, k and v projections;
* Mistral restricts every layer to the last ``sliding_window`` positions
  when the window is set;
* attention scale is ``head_dim ** -0.5`` and the MLP is SiLU-gated.

Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights

FAMILIES = ("llama", "mistral", "qwen2", "qwen3")


@dataclass(frozen=True)
class LlamaFamilyConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Mapping[str, Any] | None
    sliding_window: int | None

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "LlamaFamilyConfig":
        model_type = cfg["model_type"]
        if model_type not in FAMILIES:
            raise NotImplementedError(f"model_type {model_type!r}")
        if cfg.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"activation {cfg.get('hidden_act')!r}")
        heads = int(cfg["num_attention_heads"])
        head_dim = cfg.get("head_dim") or int(cfg["hidden_size"]) // heads
        scaling = cfg.get("rope_scaling")
        if scaling and scaling.get("rope_type", scaling.get("type")) not in ("default", "linear", "llama3"):
            raise NotImplementedError(f"rope_scaling {scaling!r}")
        window = cfg.get("sliding_window") if model_type == "mistral" else None
        return cls(
            model_type=model_type,
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=int(head_dim),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 10_000.0)),
            rope_scaling=scaling,
            sliding_window=int(window) if window else None,
        )

    def inv_freq(self) -> np.ndarray:
        """RoPE inverse frequencies, as Hugging Face's rope initializers compute them."""

        dim = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        scaling = self.rope_scaling or {}
        kind = scaling.get("rope_type", scaling.get("type", "default"))
        if kind == "linear":
            return inv_freq / float(scaling["factor"])
        if kind == "llama3":
            factor = float(scaling["factor"])
            low = float(scaling.get("low_freq_factor", 1.0))
            high = float(scaling.get("high_freq_factor", 4.0))
            old_context = float(scaling.get("original_max_position_embeddings", 8192))
            low_wavelen = old_context / low
            high_wavelen = old_context / high
            wavelen = 2 * math.pi / inv_freq
            scaled = np.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
            smooth = (old_context / wavelen - low) / (high - low)
            smoothed = (1 - smooth) * scaled / factor + smooth * scaled
            medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
            return np.where(medium, smoothed, scaled)
        return inv_freq


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)).astype(np.float32) * weight


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rope(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class LlamaFamilyReference:
    """Full-recompute float32 model for the Llama family."""

    def __init__(self, config: LlamaFamilyConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.w = dict(weights)
        if "lm_head" not in self.w:
            self.w["lm_head"] = self.w["model.embed_tokens"]

    @classmethod
    def load(cls, model_dir: Path) -> "LlamaFamilyReference":
        raw, weights = load_dense_weights(model_dir)
        return cls(LlamaFamilyConfig.from_dict(raw), weights)

    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms_norm(x, self.w[name], self.config.rms_norm_eps)

    def _project(self, name: str, x: np.ndarray) -> np.ndarray:
        out = x @ self.w[name].T
        bias = self.w.get(f"{name}.bias")
        return out + bias if bias is not None else out

    def _attention(self, index: int, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        tokens = x.shape[0]
        heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = self._project(f"{prefix}.q_proj", x).reshape(tokens, heads, dim)
        k = self._project(f"{prefix}.k_proj", x).reshape(tokens, kv_heads, dim)
        v = self._project(f"{prefix}.v_proj", x).reshape(tokens, kv_heads, dim)
        if cfg.model_type == "qwen3":
            q = self._norm(f"{prefix}.q_norm", q)
            k = self._norm(f"{prefix}.k_norm", k)
        inv_freq = cfg.inv_freq()
        q = _rope(q, positions, inv_freq)
        k = _rope(k, positions, inv_freq)

        i = positions[:, None]
        j = positions[None, :]
        allowed = j <= i
        if cfg.sliding_window:
            allowed &= (i - j) < cfg.sliding_window
        scale = dim ** -0.5
        group = heads // kv_heads
        out = np.empty((tokens, heads, dim), dtype=np.float32)
        for head in range(heads):
            kv = head // group
            scores = (q[:, head, :] @ k[:, kv, :].T).astype(np.float64) * scale
            scores = np.where(allowed, scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, head, :] = (probs @ v[:, kv, :]).astype(np.float32)
        return out.reshape(tokens, heads * dim) @ self.w[f"{prefix}.o_proj"].T

    def _mlp(self, index: int, x: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{index}.mlp"
        gate = _silu(x @ self.w[f"{prefix}.gate_proj"].T)
        up = x @ self.w[f"{prefix}.up_proj"].T
        return (gate * up) @ self.w[f"{prefix}.down_proj"].T

    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """Return ``[len(token_ids), vocab]`` float32 logits."""

        cfg = self.config
        tokens = np.asarray(token_ids, dtype=np.int64)
        positions = np.arange(tokens.shape[0])
        x = self.w["model.embed_tokens"][tokens]
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            x = x + self._attention(index, self._norm(f"{prefix}.input_layernorm", x), positions)
            x = x + self._mlp(index, self._norm(f"{prefix}.post_attention_layernorm", x))
        x = self._norm("model.norm", x)
        return (x @ self.w["lm_head"].T).astype(np.float32)

    def greedy(self, prompt: Sequence[int], steps: int) -> list[int]:
        """Greedy continuation by full recomputation (no KV cache)."""

        tokens = list(prompt)
        generated: list[int] = []
        for _ in range(steps):
            next_token = int(np.argmax(self.logits(tokens)[-1]))
            generated.append(next_token)
            tokens.append(next_token)
        return generated
