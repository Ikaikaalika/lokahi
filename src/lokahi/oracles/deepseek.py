"""NumPy reference for DeepSeek V2 and V3: multi-head latent attention and DeepSeek MoE.

Semantics follow Transformers' ``DeepseekV2ForCausalLM`` and
``DeepseekV3ForCausalLM`` (Kimi K2 and Moonlight use the V3 architecture),
and ``Glm4MoeLiteForCausalLM`` (GLM-4.7-Flash), which is V3 with its dense
and sparse layers listed in ``mlp_layer_types`` (by default the first dense):

* multi-head latent attention: each head's query is ``q_proj(x)``, or
  ``q_b_proj(rms(q_a_proj(x)))`` with a query rank, split into
  ``qk_nope_head_dim`` and ``qk_rope_head_dim`` dimensions;
  ``kv_a_proj_with_mqa(x)`` is a ``kv_lora_rank`` latent, RMS-normed, plus
  one RoPE'd key every head shares; ``kv_b_proj`` expands the latent into
  each head's unrotated key and its value; keys are ``[k_nope | k_rope]``;
* the latent norms use epsilon 1e-6 whatever ``rms_norm_eps`` is;
* RoPE covers only the rope dimensions: interleaved pairs (V2 always, as a
  complex product; V3 with ``rope_interleave``, its default, de-interleaving
  as Transformers does), otherwise rotate-half; frequencies span
  ``qk_rope_head_dim``, with YaRN as ``_compute_yarn_parameters`` computes it
  and its attention factor on cos and sin, so on the rope dimensions only;
* the softmax scale is ``(nope + rope) ** -0.5``, times
  ``mscale(factor, mscale_all_dim) ** 2`` under YaRN;
* pre-norm blocks with a dense SiLU MLP before ``first_k_dense_replace`` and a
  mixture of experts plus ``n_shared_experts`` always-on experts after;
* V3 routing (``noaux_tc``): sigmoid scores; groups ranked by the sum of
  their two best ``score + e_score_correction_bias``; experts outside the
  best ``topk_group`` groups are excluded (``-inf``); the top k by corrected
  score are weighted by their raw scores, renormalized with
  ``norm_topk_prob``, times ``routed_scaling_factor``;
* V2 routing: softmax over all experts; ``greedy`` takes the top k, and
  ``group_limited_greedy`` first keeps the ``topk_group`` groups with the
  largest maxima; weights are the probabilities times
  ``routed_scaling_factor``.

MLX's absorbed layout (``embed_q`` [heads, rank, nope] and ``unembed_out``
[heads, v, rank]) is converted back to ``kv_b_proj``, which this reference
expands per head: it never attends in the latent space, as the native engine
does. Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights

FAMILIES = ("deepseek_v2", "deepseek_v3", "glm4_moe_lite")
LATENT_NORM_EPS = 1e-6


def _mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


@dataclass(frozen=True)
class DeepseekConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Mapping[str, Any] | None
    rope_interleave: bool
    max_position_embeddings: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int
    first_k_dense_replace: int
    n_group: int
    topk_group: int
    topk_method: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    mlp_layer_types: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "DeepseekConfig":
        model_type = cfg["model_type"]
        if model_type not in FAMILIES:
            raise NotImplementedError(f"model_type {model_type!r}")
        if cfg.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"activation {cfg.get('hidden_act')!r}")
        scaling = cfg.get("rope_scaling")
        if scaling and scaling.get("rope_type", scaling.get("type")) not in ("default", "yarn"):
            raise NotImplementedError(f"rope_scaling {scaling!r}")
        v3 = model_type != "deepseek_v2"
        if float(cfg.get("partial_rotary_factor", 1.0)) != 1.0:
            raise NotImplementedError("partial_rotary_factor")
        layers = int(cfg["num_hidden_layers"])
        types = cfg.get("mlp_layer_types")
        if model_type == "glm4_moe_lite" and types is None:
            types = ["dense"] + ["sparse"] * (layers - 1)
        q_rank = cfg.get("q_lora_rank")
        return cls(
            model_type=model_type,
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            kv_lora_rank=int(cfg["kv_lora_rank"]),
            q_lora_rank=int(q_rank) if q_rank else None,
            qk_nope_head_dim=int(cfg["qk_nope_head_dim"]),
            qk_rope_head_dim=int(cfg["qk_rope_head_dim"]),
            v_head_dim=int(cfg["v_head_dim"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 10_000.0)),
            rope_scaling=scaling,
            rope_interleave=bool(cfg.get("rope_interleave", True)) if v3 else True,
            max_position_embeddings=int(cfg.get("max_position_embeddings") or 0),
            n_routed_experts=int(cfg.get("n_routed_experts") or 0),
            num_experts_per_tok=int(cfg.get("num_experts_per_tok") or 0),
            moe_intermediate_size=int(cfg.get("moe_intermediate_size") or 0),
            n_shared_experts=int(cfg.get("n_shared_experts", 1 if v3 else 2) or 0),
            first_k_dense_replace=int(cfg.get("first_k_dense_replace") or 0),
            n_group=int(cfg.get("n_group") or 1),
            topk_group=int(cfg.get("topk_group") or 1),
            topk_method=str(cfg.get("topk_method") or ("noaux_tc" if v3 else "greedy")),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", v3)),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor") or 1.0),
            mlp_layer_types=tuple(types or ()),
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def sparse(self, layer: int) -> bool:
        if self.mlp_layer_types:
            return self.mlp_layer_types[layer] == "sparse"
        return self.n_routed_experts > 0 and layer >= self.first_k_dense_replace

    def _yarn(self) -> Mapping[str, Any] | None:
        scaling = self.rope_scaling or {}
        return scaling if scaling.get("rope_type", scaling.get("type")) == "yarn" else None

    def _yarn_factor(self) -> float:
        scaling = self._yarn() or {}
        if scaling.get("factor") is not None:
            return float(scaling["factor"])
        return self.max_position_embeddings / float(scaling["original_max_position_embeddings"])

    def attention_factor(self) -> float:
        """YaRN's scale on cos and sin; 1 without YaRN."""

        scaling = self._yarn()
        if scaling is None:
            return 1.0
        if scaling.get("attention_factor"):
            return float(scaling["attention_factor"])
        factor = self._yarn_factor()
        mscale, all_dim = scaling.get("mscale"), scaling.get("mscale_all_dim")
        if mscale and all_dim:
            return _mscale(factor, mscale) / _mscale(factor, all_dim)
        return _mscale(factor)

    def softmax_scale(self) -> float:
        """``yarn_apply_mscale``: the head scale times mscale(factor, mscale_all_dim) squared."""

        scale = self.qk_head_dim ** -0.5
        scaling = self._yarn()
        if scaling is not None and scaling.get("mscale_all_dim"):
            m = _mscale(float(scaling["factor"]), float(scaling["mscale_all_dim"]))
            scale *= m * m
        return scale

    def inv_freq(self) -> np.ndarray:
        dim = self.qk_rope_head_dim
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        scaling = self._yarn()
        if scaling is None:
            return inv_freq
        factor = self._yarn_factor()
        original = float(scaling["original_max_position_embeddings"])
        fast, slow = scaling.get("beta_fast") or 32, scaling.get("beta_slow") or 1

        def correction(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(self.rope_theta))

        low, high = correction(fast), correction(slow)
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
        extrapolation = 1 - ramp
        return (inv_freq / factor) * (1 - extrapolation) + inv_freq * extrapolation


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)).astype(np.float32) * weight


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class DeepseekReference:
    """Full-recompute float32 model for DeepSeek V2 and V3."""

    def __init__(self, config: DeepseekConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.w = dict(weights)
        if "lm_head" not in self.w:
            self.w["lm_head"] = self.w["model.embed_tokens"]
        heads, nope, v = config.num_attention_heads, config.qk_nope_head_dim, config.v_head_dim
        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}.self_attn"
            if f"{prefix}.kv_b_proj" in self.w:
                continue
            # MLX's absorbed layout back to kv_b_proj: per head, the key rows
            # (embed_q transposed) then the value rows.
            keys = np.transpose(self.w.pop(f"{prefix}.embed_q"), (0, 2, 1))
            values = self.w.pop(f"{prefix}.unembed_out")
            assert keys.shape[:2] == (heads, nope) and values.shape[:2] == (heads, v)
            self.w[f"{prefix}.kv_b_proj"] = np.concatenate([keys, values], axis=1).reshape(heads * (nope + v), -1)

    @classmethod
    def load(cls, model_dir: Path) -> "DeepseekReference":
        raw, weights = load_dense_weights(model_dir)
        return cls(DeepseekConfig.from_dict(raw), weights)

    def _norm(self, name: str, x: np.ndarray, eps: float | None = None) -> np.ndarray:
        return _rms_norm(x, self.w[name], self.config.rms_norm_eps if eps is None else eps)

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """``x`` [tokens, heads, rope]: as each family's apply-rotary function lays the result out."""

        cfg = self.config
        angles = positions[:, None].astype(np.float64) * cfg.inv_freq()[None, :]
        factor = cfg.attention_factor()
        cos = (np.cos(angles) * factor).astype(np.float32)[:, None, :]
        sin = (np.sin(angles) * factor).astype(np.float32)[:, None, :]
        if cfg.model_type == "deepseek_v2":  # view_as_complex(x) * freqs_cis, re-interleaved
            even, odd = x[..., 0::2], x[..., 1::2]
            out = np.empty_like(x)
            out[..., 0::2] = even * cos - odd * sin
            out[..., 1::2] = even * sin + odd * cos
            return out
        if cfg.rope_interleave:  # apply_rotary_pos_emb_interleave: pairs, de-interleaved
            even, odd = x[..., 0::2], x[..., 1::2]
            return np.concatenate([even * cos - odd * sin, odd * cos + even * sin], axis=-1)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def _attention(self, index: int, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        tokens, heads = x.shape[0], cfg.num_attention_heads
        nope, rope, v_dim, rank = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        if cfg.q_lora_rank is None:
            q = x @ self.w[f"{prefix}.q_proj"].T
        else:
            latent = self._norm(f"{prefix}.q_a_layernorm", x @ self.w[f"{prefix}.q_a_proj"].T, LATENT_NORM_EPS)
            q = latent @ self.w[f"{prefix}.q_b_proj"].T
        q = q.reshape(tokens, heads, nope + rope)
        compressed = x @ self.w[f"{prefix}.kv_a_proj_with_mqa"].T
        kv_latent = self._norm(f"{prefix}.kv_a_layernorm", compressed[:, :rank], LATENT_NORM_EPS)
        q_rot = self._rope(q[..., nope:], positions)
        k_rot = self._rope(compressed[:, None, rank:], positions)
        kv = (kv_latent @ self.w[f"{prefix}.kv_b_proj"].T).reshape(tokens, heads, nope + v_dim)
        keys = np.concatenate([kv[..., :nope], np.broadcast_to(k_rot, (tokens, heads, rope))], axis=-1)
        values = kv[..., nope:]
        queries = np.concatenate([q[..., :nope], q_rot], axis=-1)
        allowed = positions[None, :] <= positions[:, None]
        scale = cfg.softmax_scale()
        out = np.empty((tokens, heads, v_dim), dtype=np.float32)
        for head in range(heads):
            scores = (queries[:, head, :] @ keys[:, head, :].T).astype(np.float64) * scale
            scores = np.where(allowed, scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, head, :] = (probs @ values[:, head, :]).astype(np.float32)
        return out.reshape(tokens, heads * v_dim) @ self.w[f"{prefix}.o_proj"].T

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        return (_silu(x @ self.w[f"{prefix}.gate_proj"].T) * (x @ self.w[f"{prefix}.up_proj"].T)) @ (
            self.w[f"{prefix}.down_proj"].T)

    def selection(self, index: int, x: np.ndarray) -> np.ndarray:
        """The scores the top k are taken over, excluded experts at ``-inf`` (V3) or 0 (V2)."""

        cfg = self.config
        logits = (x @ self.w[f"model.layers.{index}.mlp.gate"].T).astype(np.float64)
        tokens, groups = logits.shape[0], cfg.n_group
        if cfg.model_type != "deepseek_v2":
            scores = 1.0 / (1.0 + np.exp(-logits)) + self.w[f"model.layers.{index}.mlp.gate.e_score_correction_bias"]
            group_scores = -np.sort(-scores.reshape(tokens, groups, -1), axis=-1)[..., :2].sum(axis=-1)
            fill = -np.inf
        else:
            scores = np.exp(logits - logits.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)
            if cfg.topk_method == "greedy":
                return scores
            group_scores = scores.reshape(tokens, groups, -1).max(axis=-1)
            fill = 0.0
        kept = np.argsort(-group_scores, axis=-1, kind="stable")[:, : cfg.topk_group]
        mask = np.zeros_like(group_scores, dtype=bool)
        np.put_along_axis(mask, kept, True, axis=-1)
        return np.where(np.repeat(mask, cfg.n_routed_experts // groups, axis=-1), scores, fill)

    def route(self, index: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expert indices ``[tokens, k]`` and their weights for layer ``index``."""

        cfg = self.config
        chosen = np.argsort(-self.selection(index, x), axis=-1, kind="stable")[:, : cfg.num_experts_per_tok]
        logits = (x @ self.w[f"model.layers.{index}.mlp.gate"].T).astype(np.float64)
        if cfg.model_type != "deepseek_v2":
            weights = np.take_along_axis(1.0 / (1.0 + np.exp(-logits)), chosen, axis=-1)
            if cfg.norm_topk_prob:
                weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
        else:
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            weights = np.take_along_axis(probs / probs.sum(axis=-1, keepdims=True), chosen, axis=-1)
        return chosen, weights * cfg.routed_scaling_factor

    def _expert_stack(self, prefix: str, name: str) -> np.ndarray:
        stacked = f"{prefix}.switch_mlp.{name}"
        if stacked in self.w:
            return self.w[stacked]
        return np.stack([self.w[f"{prefix}.experts.{e}.{name}"] for e in range(self.config.n_routed_experts)])

    def _experts(self, index: int, x: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{index}.mlp"
        chosen, weights = self.route(index, x)
        gate, up, down = (self._expert_stack(prefix, name) for name in ("gate_proj", "up_proj", "down_proj"))
        out = np.zeros_like(x, dtype=np.float32)
        for t in range(x.shape[0]):
            for e, p in zip(chosen[t], weights[t]):
                out[t] += np.float32(p) * (down[e] @ (_silu(gate[e] @ x[t]) * (up[e] @ x[t])))
        if self.config.n_shared_experts:
            out = out + self._mlp(f"{prefix}.shared_experts", x)
        return out

    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """Return ``[len(token_ids), vocab]`` float32 logits."""

        cfg = self.config
        tokens = np.asarray(token_ids, dtype=np.int64)
        positions = np.arange(tokens.shape[0])
        x = self.w["model.embed_tokens"][tokens]
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            x = x + self._attention(index, self._norm(f"{prefix}.input_layernorm", x), positions)
            h = self._norm(f"{prefix}.post_attention_layernorm", x)
            x = x + (self._experts(index, h) if cfg.sparse(index) else self._mlp(f"{prefix}.mlp", h))
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
