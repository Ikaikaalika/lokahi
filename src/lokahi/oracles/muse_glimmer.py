"""NumPy reference for Meta's Muse Glimmer text decoder.

Semantics follow Transformers ``modeling_muse_glimmer`` (5.15, Meta's
reference implementation), which MLX-LM ``muse_glimmer`` and MLX-VLM also
implement:

* token embeddings get a scale-free RMSNorm (``rms_norm_eps``);
* each layer is ``x += post_attn_norm(attn(input_norm(x)))`` then
  ``x += post_ff_norm(mlp(pre_ff_norm(x)))``; these four norms scale by
  ``1 + weight``, and the two post-norms use ``post_norm_eps``;
* queries and keys get a scale-free per-head RMSNorm, then queries are
  multiplied by ``qk_scale_factor``; the attention scale is ``head_dim**-0.5``;
* layers whose ``layer_rope_theta`` is nonzero (the sliding ones) apply
  rotate-half RoPE with that theta; the others use no positional encoding;
* sliding layers attend to the last ``sliding_window`` positions;
* the attention output is multiplied by ``sigmoid(gate_proj(x))`` before
  ``o_proj``;
* the MLP is SwiGLU;
* the final norm scales by its weight (no offset), and the logits are
  ``cap * tanh(logits * output_multiplier / cap)``.

Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights


@dataclass(frozen=True)
class MuseGlimmerConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    sliding_window: int
    layer_is_sliding: tuple[bool, ...]
    layer_rope_theta: tuple[float, ...]
    final_logit_softcapping: float | None
    output_multiplier: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MuseGlimmerConfig":
        cfg = dict(raw.get("text_config") or raw)
        layers = int(cfg["num_hidden_layers"])
        layer_types = cfg.get("layer_types") or [
            "full_attention" if index % 4 == 3 else "sliding_attention" for index in range(layers)
        ]
        if len(layer_types) != layers:
            raise ValueError("layer_types length does not match num_hidden_layers")
        rope = cfg.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope.get('rope_type')!r}")
        theta = float(rope.get("rope_theta", cfg.get("rope_theta", 500_000.0)))
        thetas = cfg.get("layer_rope_theta") or [
            0.0 if kind == "full_attention" else theta for kind in layer_types
        ]
        if len(thetas) != layers:
            raise ValueError("layer_rope_theta length does not match num_hidden_layers")
        if cfg.get("hidden_activation", "silu") != "silu":
            raise NotImplementedError(f"activation {cfg.get('hidden_activation')!r}")
        heads = int(cfg["num_attention_heads"])
        cap = cfg.get("final_logit_softcapping", 20.0)
        return cls(
            vocab_size=int(cfg.get("vocab_size", 202048)),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=int(cfg.get("head_dim", int(cfg["hidden_size"]) // heads)),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
            post_norm_eps=float(cfg.get("post_norm_eps", 1e-8)),
            qk_scale_factor=float(cfg.get("qk_scale_factor", 3.87)),
            sliding_window=int(cfg.get("sliding_window", 2048)),
            layer_is_sliding=tuple(kind == "sliding_attention" for kind in layer_types),
            layer_rope_theta=tuple(float(value) for value in thetas),
            final_logit_softcapping=None if cap is None else float(cap),
            output_multiplier=float(cfg.get("output_multiplier", 0.19611613513818404)),
        )


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)).astype(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class MuseGlimmerReference:
    """Full-recompute float32 Muse Glimmer text model."""

    def __init__(self, config: MuseGlimmerConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.w = dict(weights)
        if "lm_head" not in self.w:
            self.w["lm_head"] = self.w["model.embed_tokens"]

    @classmethod
    def load(cls, model_dir: Path) -> "MuseGlimmerReference":
        raw, weights = load_dense_weights(model_dir)
        return cls(MuseGlimmerConfig.from_dict(raw), weights)

    def _centered(self, name: str, x: np.ndarray, eps: float) -> np.ndarray:
        return _rms(x, eps) * (1.0 + self.w[name])

    def _attention(self, index: int, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        tokens = x.shape[0]
        heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = (x @ self.w[f"{prefix}.q_proj"].T).reshape(tokens, heads, dim)
        k = (x @ self.w[f"{prefix}.k_proj"].T).reshape(tokens, kv_heads, dim)
        v = (x @ self.w[f"{prefix}.v_proj"].T).reshape(tokens, kv_heads, dim)
        q = _rms(q, cfg.rms_norm_eps) * np.float32(cfg.qk_scale_factor)
        k = _rms(k, cfg.rms_norm_eps)
        theta = cfg.layer_rope_theta[index]
        if theta:
            q = _rope(q, positions, theta)
            k = _rope(k, positions, theta)

        i = positions[:, None]
        j = positions[None, :]
        allowed = j <= i
        if cfg.layer_is_sliding[index]:
            allowed &= (i - j) < cfg.sliding_window
        scale = cfg.head_dim ** -0.5
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
        out = out.reshape(tokens, heads * dim) * _sigmoid(x @ self.w[f"{prefix}.gate_proj"].T)
        return out @ self.w[f"{prefix}.o_proj"].T

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
        x = _rms(self.w["model.embed_tokens"][tokens], cfg.rms_norm_eps)
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            attn = self._attention(index, self._centered(f"{prefix}.input_layernorm", x, cfg.rms_norm_eps), positions)
            x = x + self._centered(f"{prefix}.post_attention_layernorm", attn, cfg.post_norm_eps)
            mlp = self._mlp(index, self._centered(f"{prefix}.pre_feedforward_layernorm", x, cfg.rms_norm_eps))
            x = x + self._centered(f"{prefix}.post_feedforward_layernorm", mlp, cfg.post_norm_eps)
        x = _rms(x, cfg.rms_norm_eps) * self.w["model.norm"]
        logits = (x @ self.w["lm_head"].T) * np.float32(cfg.output_multiplier)
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
