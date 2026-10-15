"""NumPy reference for the Gemma 4 text decoder (dense, no MoE).

Semantics follow MLX-LM ``gemma4_text`` (v0.31.2), which the mlx-community
checkpoints are converted for:

* token embeddings are multiplied by ``sqrt(hidden_size)`` rounded to BF16,
  as MLX-LM and Transformers both apply it to BF16 weights. The per-layer
  input constants below are rounded too, as MLX-LM's weakly typed scalars
  are (Transformers keeps them in float32; the difference is ~1e-4);
* every RMSNorm scales by ``weight`` (no ``1 +`` offset), and each layer is
  sandwich-normed: ``x += post_attn_norm(attn(input_norm(x)))`` then
  ``x += post_ff_norm(mlp(pre_ff_norm(x)))``;
* layers are sliding or full by ``layer_types``. Full layers can use their
  own head size (``global_head_dim``) and, with ``attention_k_eq_v``, values
  equal to the raw key projection and ``num_global_key_value_heads`` kv heads;
* queries and keys get per-head RMSNorm before rotate-half RoPE; values get
  a scale-free RMSNorm; the attention scale is 1;
* RoPE per layer type from ``rope_parameters``. "proportional" RoPE rotates
  only the first ``partial_rotary_factor`` of the dimensions, with exponents
  taken over the full head size;
* the last ``num_kv_shared_layers`` layers compute no keys or values: each
  attends over those of the last earlier layer of its type. With
  ``use_double_wide_mlp`` their MLPs are twice as wide;
* per-layer inputs (``hidden_size_per_layer_input``): a per-layer embedding
  of the token, scaled by ``sqrt(P)``, plus a normed projection of the scaled
  embedding, scaled by ``hidden_size ** -0.5``, all times ``2 ** -0.5``. Each
  layer gates it with ``gelu(per_layer_input_gate(x))``, projects it back,
  norms it and adds it;
* with ``enable_moe_block`` (26B-A4B), a mixture of experts runs beside each
  dense MLP: ``x += post_ff_norm(post_ff_norm_1(mlp(pre_ff_norm(x))) +
  post_ff_norm_2(experts(pre_ff_norm_2(x))))``. The router reads
  ``rms(x) * scale * hidden ** -0.5``, takes the softmax of its top
  ``top_k_experts`` logits (lower index first on ties) and multiplies each
  weight by ``per_expert_scale``; the experts are GELU-gated;
* each layer's output is multiplied by its ``layer_scalar``;
* the MLP is GeGLU with the tanh GELU, and logits are soft-capped;
* with images (``logits(..., injected, blocks)``), the image features replace
  their tokens' scaled embeddings as they are (with per-layer inputs, those
  positions look up the pad token's per-layer embedding and project the
  features) and, when
  ``use_bidirectional_attention`` is ``"vision"``, each image's tokens attend
  to each other in both directions on sliding layers only, within the window
  (Transformers' ``AND(window, OR(causal, block))``); full layers stay causal.

Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..formats.safetensors import bf16_to_f32, f32_to_bf16, read_safetensors
from .gemma3 import load_dense_weights


@dataclass(frozen=True)
class Gemma4Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int | None
    head_dim: int
    global_head_dim: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    rope: Mapping[str, Mapping[str, Any]]
    sliding_window: int
    attention_k_eq_v: bool
    num_kv_shared_layers: int
    hidden_size_per_layer_input: int
    final_logit_softcapping: float | None
    pad_token_id: int = 0
    bidirectional_vision: bool = False  # use_bidirectional_attention == "vision"
    num_experts: int = 0  # with enable_moe_block
    top_k_experts: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Gemma4Config":
        cfg = dict(raw.get("text_config") or raw)
        moe = bool(cfg.get("enable_moe_block"))
        layers = int(cfg["num_hidden_layers"])
        types = cfg.get("layer_types")
        if not types:
            pattern = int(cfg.get("sliding_window_pattern", 5))
            cycle = ["sliding_attention"] * (pattern - 1) + ["full_attention"]
            types = (cycle * (layers // len(cycle) + 1))[:layers]
        rope = cfg.get("rope_parameters") or {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1e4, "rope_type": "default"},
        }
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=layers,
            num_attention_heads=int(cfg.get("num_attention_heads", 8)),
            num_key_value_heads=int(cfg.get("num_key_value_heads", 1)),
            num_global_key_value_heads=cfg.get("num_global_key_value_heads"),
            head_dim=int(cfg.get("head_dim", 256)),
            global_head_dim=int(cfg.get("global_head_dim") or cfg.get("head_dim", 256)),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            layer_types=tuple(types),
            rope=rope,
            sliding_window=int(cfg.get("sliding_window", 512)),
            attention_k_eq_v=bool(cfg.get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(cfg.get("num_kv_shared_layers", 0)),
            hidden_size_per_layer_input=int(cfg.get("hidden_size_per_layer_input") or 0),
            final_logit_softcapping=cfg.get("final_logit_softcapping"),
            pad_token_id=int(cfg.get("pad_token_id", raw.get("pad_token_id", 0)) or 0),
            bidirectional_vision=cfg.get("use_bidirectional_attention") == "vision",
            num_experts=int(cfg["num_experts"]) if moe else 0,
            top_k_experts=int(cfg["top_k_experts"]) if moe else 0,
        )

    def sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def head_size(self, layer: int) -> int:
        return self.head_dim if self.sliding(layer) else self.global_head_dim

    def k_eq_v(self, layer: int) -> bool:
        return self.attention_k_eq_v and not self.sliding(layer)

    def kv_heads(self, layer: int) -> int:
        if self.k_eq_v(layer) and self.num_global_key_value_heads is not None:
            return int(self.num_global_key_value_heads)
        return self.num_key_value_heads

    def kv_source(self, layer: int) -> int:
        """The layer whose keys and values ``layer`` attends over."""

        first_shared = self.num_hidden_layers - self.num_kv_shared_layers
        if self.num_kv_shared_layers <= 0 or layer < first_shared:
            return layer
        return max(i for i in range(first_shared) if self.layer_types[i] == self.layer_types[layer])

    def inv_freq(self, layer: int) -> np.ndarray:
        params = self.rope["sliding_attention" if self.sliding(layer) else "full_attention"]
        dims = self.head_size(layer)
        theta = float(params.get("rope_theta", 10_000.0))
        kind = params.get("rope_type", "default")
        inv = np.zeros(dims // 2, dtype=np.float64)
        if kind == "proportional":
            # Rotated angle pairs as Transformers counts them; MLX-LM agrees
            # whenever dims * partial_rotary_factor is even.
            pairs = int(dims * float(params.get("partial_rotary_factor", 1.0)) // 2)
            exponents = np.arange(0, 2 * pairs, 2, dtype=np.float64) / dims
            inv[: exponents.shape[0]] = 1.0 / (float(params.get("factor", 1.0)) * theta**exponents)
        elif kind == "default":
            inv[:] = 1.0 / theta ** (np.arange(0, dims, 2, dtype=np.float64) / dims)
        else:
            raise NotImplementedError(f"rope_type {kind!r}")
        return inv


def _bf16(value: float) -> np.float32:
    """A scalar constant as the BF16 references apply it."""

    return bf16_to_f32(f32_to_bf16(np.array([value], np.float32)))[0]


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)).astype(np.float32)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _rope(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _layer_scalars(model_dir: Path) -> dict[str, np.ndarray]:
    """``layer_scalar`` tensors, which have no ``.weight`` suffix."""

    out: dict[str, np.ndarray] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        tensors, _ = read_safetensors(shard)
        for name, (dtype, array) in tensors.items():
            if name.endswith(".layer_scalar"):
                value = bf16_to_f32(array) if dtype == "BF16" else np.asarray(array, dtype=np.float32)
                out[name.removeprefix("language_model.")] = value.reshape(-1)
    return out


class Gemma4Reference:
    """Full-recompute float32 Gemma 4 text model."""

    def __init__(self, config: Gemma4Config, weights: Mapping[str, np.ndarray], bf16_constants: bool = True):
        self.config = config
        self.w = dict(weights)
        if "lm_head" not in self.w:
            self.w["lm_head"] = self.w["model.embed_tokens"]
        # False applies the scalar constants unrounded, as MLX-LM does when it
        # runs in float32.
        self._constant = _bf16 if bf16_constants else np.float32

    @classmethod
    def load(cls, model_dir: Path, bf16_constants: bool = True) -> "Gemma4Reference":
        raw, weights = load_dense_weights(model_dir)
        weights.update(_layer_scalars(model_dir))
        return cls(Gemma4Config.from_dict(raw), weights, bf16_constants)

    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms(x, self.config.rms_norm_eps) * self.w[name]

    def _per_layer_inputs(self, tokens: np.ndarray, embedded: np.ndarray) -> np.ndarray | None:
        cfg = self.config
        size = cfg.hidden_size_per_layer_input
        if not size:
            return None
        n, layers = tokens.shape[0], cfg.num_hidden_layers
        lookup = self.w["model.embed_tokens_per_layer"][tokens] * self._constant(math.sqrt(size))
        projected = (embedded @ self.w["model.per_layer_model_projection"].T) * self._constant(cfg.hidden_size**-0.5)
        projected = self._norm("model.per_layer_projection_norm", projected.reshape(n, layers, size))
        return (projected + lookup.reshape(n, layers, size)) * self._constant(2.0**-0.5)

    def _kv(self, index: int, x: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        n, kv_heads, dim = x.shape[0], cfg.kv_heads(index), cfg.head_size(index)
        k = (x @ self.w[f"{prefix}.k_proj"].T).reshape(n, kv_heads, dim)
        v = k if cfg.k_eq_v(index) else (x @ self.w[f"{prefix}.v_proj"].T).reshape(n, kv_heads, dim)
        k = _rope(self._norm(f"{prefix}.k_norm", k), positions, cfg.inv_freq(index))
        return k, _rms(v, cfg.rms_norm_eps)

    def _attention(self, index: int, x: np.ndarray, positions: np.ndarray, kv: tuple[np.ndarray, np.ndarray],
                   blocks: Sequence[tuple[int, int]] = ()):
        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        n, heads, dim = x.shape[0], cfg.num_attention_heads, cfg.head_size(index)
        q = (x @ self.w[f"{prefix}.q_proj"].T).reshape(n, heads, dim)
        q = _rope(self._norm(f"{prefix}.q_norm", q), positions, cfg.inv_freq(index))
        k, v = kv
        i, j = positions[:, None], positions[None, :]
        allowed = j <= i
        if cfg.sliding(index):
            if cfg.bidirectional_vision:
                for begin, end in blocks:
                    inside = (positions >= begin) & (positions < end)
                    allowed |= inside[:, None] & inside[None, :]
            allowed &= (i - j) < cfg.sliding_window
        group = heads // k.shape[1]
        out = np.empty((n, heads, dim), dtype=np.float32)
        for head in range(heads):
            kv_head = head // group
            scores = (q[:, head, :] @ k[:, kv_head, :].T).astype(np.float64)  # attention scale is 1
            scores = np.where(allowed, scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, head, :] = (probs @ v[:, kv_head, :]).astype(np.float32)
        return out.reshape(n, heads * dim) @ self.w[f"{prefix}.o_proj"].T

    def route(self, index: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The chosen experts ``[n, k]`` and their weights for the layer's input ``x``."""

        cfg = self.config
        prefix = f"model.layers.{index}.router"
        routed = _rms(x, cfg.rms_norm_eps) * self.w[f"{prefix}.scale"] * np.float32(cfg.hidden_size**-0.5)
        logits = (routed @ self.w[f"{prefix}.proj"].T).astype(np.float64)
        k = cfg.top_k_experts
        order = np.argsort(-logits, axis=-1, kind="stable")[:, :k]  # ties to the lower index
        top = np.take_along_axis(logits, order, axis=-1)
        weights = np.exp(top - top.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        return order, (weights * self.w[f"{prefix}.per_expert_scale"][order]).astype(np.float32)

    def _experts(self, index: int, x: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{index}"
        chosen, weights = self.route(index, x)
        h = self._norm(f"{prefix}.pre_feedforward_layernorm_2", x)
        gate = self.w[f"{prefix}.experts.switch_glu.gate_proj"]
        up = self.w[f"{prefix}.experts.switch_glu.up_proj"]
        down = self.w[f"{prefix}.experts.switch_glu.down_proj"]
        out = np.zeros_like(x)
        for t in range(x.shape[0]):
            for e, weight in zip(chosen[t], weights[t]):
                act = _gelu_tanh(gate[e] @ h[t]) * (up[e] @ h[t])
                out[t] += weight * (down[e] @ act)
        return out

    def logits(self, token_ids: Sequence[int], injected: Mapping[int, np.ndarray] | None = None,
               blocks: Sequence[tuple[int, int]] = ()) -> np.ndarray:
        """Return ``[len(token_ids), vocab]`` float32 logits.

        ``injected`` maps prompt positions to rows that replace their scaled
        token embeddings (image features); ``blocks`` are each image's
        ``[begin, end)`` positions.
        """

        cfg = self.config
        tokens = np.asarray(token_ids, dtype=np.int64)
        positions = np.arange(tokens.shape[0])
        x = self.w["model.embed_tokens"][tokens] * self._constant(math.sqrt(cfg.hidden_size))
        lookup = tokens.copy()
        for position, row in (injected or {}).items():
            x[position] = row
            # Transformers looks up image positions' per-layer embeddings for
            # the pad token; their projection reads the image features.
            lookup[position] = cfg.pad_token_id
        per_layer = self._per_layer_inputs(lookup, x)
        kvs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            h = self._norm(f"{prefix}.input_layernorm", x)
            source = cfg.kv_source(index)
            if source == index:
                kvs[index] = self._kv(index, h, positions)
            attention = self._attention(index, h, positions, kvs[source], blocks)
            x = x + self._norm(f"{prefix}.post_attention_layernorm", attention)
            h = self._norm(f"{prefix}.pre_feedforward_layernorm", x)
            mlp = (_gelu_tanh(h @ self.w[f"{prefix}.mlp.gate_proj"].T) * (h @ self.w[f"{prefix}.mlp.up_proj"].T)) @ (
                self.w[f"{prefix}.mlp.down_proj"].T
            )
            if cfg.num_experts:
                mlp = self._norm(f"{prefix}.post_feedforward_layernorm_1", mlp) + self._norm(
                    f"{prefix}.post_feedforward_layernorm_2", self._experts(index, x))
            x = x + self._norm(f"{prefix}.post_feedforward_layernorm", mlp)
            if per_layer is not None:
                gate = _gelu_tanh(x @ self.w[f"{prefix}.per_layer_input_gate"].T) * per_layer[:, index, :]
                x = x + self._norm(f"{prefix}.post_per_layer_input_norm", gate @ self.w[f"{prefix}.per_layer_projection"].T)
            scalar = self.w.get(f"{prefix}.layer_scalar")
            if scalar is not None:
                x = x * scalar
        x = self._norm("model.norm", x)
        logits = x @ self.w["lm_head"].T
        if cfg.final_logit_softcapping:
            cap = float(cfg.final_logit_softcapping)
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
