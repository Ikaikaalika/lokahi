"""NumPy reference for the Llama family: Llama 3.x, Mistral, Qwen 2/2.5, Qwen 3, Qwen 3 MoE, GPT-OSS, GLM-4.5.

Semantics follow the Hugging Face ``LlamaForCausalLM``, ``MistralForCausalLM``,
``Qwen2ForCausalLM``, ``Qwen3ForCausalLM``, ``Qwen3MoeForCausalLM``,
``GptOssForCausalLM`` and ``Glm4MoeForCausalLM`` implementations:

* pre-norm blocks: ``x += attn(norm(x))`` then ``x += mlp(norm(x))``, where
  the norm before the MLP is named ``post_attention_layernorm``;
* RMSNorm scales by ``weight`` (not ``1 + weight``);
* rotate-half RoPE with ``rope_theta`` and optional ``linear`` or ``llama3``
  frequency scaling;
* Qwen 3 applies per-head RMSNorm to queries and keys before RoPE;
* Qwen 2 adds biases to the q, k and v projections;
* Mistral restricts every layer to the last ``sliding_window`` positions
  when the window is set;
* attention scale is ``head_dim ** -0.5`` and the MLP is SiLU-gated;
* Qwen 3 MoE replaces the MLP of sparse layers with ``num_experts`` SiLU-gated
  experts: a softmax over the router logits picks the top
  ``num_experts_per_tok`` (renormalized when ``norm_topk_prob``), and the
  layer outputs the probability-weighted sum of those experts;
* GPT-OSS adds biases to q, k, v and o, a learned per-head sink logit to every
  softmax, alternating sliding and full layers, and experts behind a biased
  router whose top-k logits are softmaxed; experts have biases and the
  clamped SwiGLU ``min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clip(u, -7, 7) + 1)``;
* YaRN RoPE as Transformers' ``_compute_yarn_parameters`` computes it,
  honoring ``truncate``, with q and k scaled by the attention factor;
* GLM-4.5 (``glm4_moe``) adds q/k/v biases, optional q/k norms, RoPE on the
  first ``partial_rotary_factor`` of each head (rotate-half within it), dense
  layers before ``first_k_dense_replace``, and MoE layers whose sigmoid
  router picks experts by score plus ``e_score_correction_bias`` (within the
  best ``topk_group`` of ``n_group`` groups) but weights them by the raw
  scores, renormalized and times ``routed_scaling_factor``, plus a shared
  expert;
* Qwen3-VL's text models (Qwen 3 and Qwen 3 MoE) use interleaved M-RoPE
  (``rope_scaling.mrope_section``): rotation frequency i reads a token's row
  position when ``i % 3 == 1`` and ``i < 3 * section[1]``, its column
  position when ``i % 3 == 2`` and ``i < 3 * section[2]``, and its temporal
  one otherwise; ``logits`` takes the three positions per token, injected
  embeddings, and deepstack rows added to the hidden states after each of
  the first layers;
* Qwen2.5-VL's text model (Qwen 2) uses chunked M-RoPE
  (``mrope_interleaved`` false): the first ``section[0]`` frequencies read
  the temporal position, the next ``section[1]`` the row and the last
  ``section[2]`` the column; the sections cover every frequency.

Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights

FAMILIES = ("llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "gpt_oss", "glm4_moe")
QK_NORM = ("qwen3", "qwen3_moe")


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
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = False
    layer_types: tuple[str, ...] = ()
    swiglu_limit: float = 7.0
    max_position_embeddings: int = 0
    rotary_dim: int = 0  # 0 rotates the whole head
    qk_norm: bool = False
    first_k_dense_replace: int = 0
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    n_shared_experts: int = 0
    mrope_section: tuple[int, ...] = ()  # M-RoPE (Qwen3-VL, Qwen2.5-VL); empty: one position per token
    mrope_interleaved: bool = True  # Qwen3-VL's layout; False: Qwen2.5-VL's chunked sections

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
        if scaling and scaling.get("rope_type", scaling.get("type")) not in ("default", "linear", "llama3", "yarn"):
            raise NotImplementedError(f"rope_scaling {scaling!r}")
        window = cfg.get("sliding_window") if model_type in ("mistral", "gpt_oss") else None
        layers = int(cfg["num_hidden_layers"])
        types = tuple(cfg.get("layer_types") or ["sliding_attention", "full_attention"] * (layers // 2 + 1))[:layers]
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
            num_experts=int(cfg.get("num_experts") or 0) if model_type == "qwen3_moe"
            else int(cfg.get("n_routed_experts") or 0) if model_type == "glm4_moe"
            else int(cfg.get("num_local_experts") or 0),
            num_experts_per_tok=int(cfg.get("num_experts_per_tok") or cfg.get("experts_per_token") or 0),
            decoder_sparse_step=int(cfg.get("decoder_sparse_step") or 1),
            mlp_only_layers=tuple(cfg.get("mlp_only_layers") or ()),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", False)),
            layer_types=types if model_type == "gpt_oss" else (),
            swiglu_limit=float(cfg.get("swiglu_limit", 7.0)),
            max_position_embeddings=int(cfg.get("max_position_embeddings") or 0),
            rotary_dim=int(int(head_dim) * float(cfg.get("partial_rotary_factor", 1.0))) if model_type == "glm4_moe" else 0,
            qk_norm=model_type in QK_NORM or bool(cfg.get("use_qk_norm", False)),
            first_k_dense_replace=int(cfg.get("first_k_dense_replace") or 0),
            n_group=int(cfg.get("n_group") or 1),
            topk_group=int(cfg.get("topk_group") or 1),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor") or 1.0),
            n_shared_experts=int(cfg.get("n_shared_experts") or 0),
            mrope_section=tuple(int(s) for s in (scaling or {}).get("mrope_section") or ()),
            mrope_interleaved=bool((scaling or {}).get("mrope_interleaved", True)),
        )

    def rope_axes(self) -> np.ndarray:
        """The position axis (0 temporal, 1 row, 2 column) each rotation frequency reads."""

        axes = np.zeros((self.rotary_dim or self.head_dim) // 2, dtype=np.int64)
        if self.mrope_section and not self.mrope_interleaved:
            temporal, rows, columns = self.mrope_section
            if temporal + rows + columns != axes.size:
                raise ValueError(f"chunked mrope_section {self.mrope_section} must cover {axes.size} frequencies")
            axes[temporal : temporal + rows] = 1
            axes[temporal + rows :] = 2
            return axes
        for axis in (1, 2) if self.mrope_section else ():
            axes[axis : 3 * self.mrope_section[axis] : 3] = axis
        return axes

    def sliding(self, layer: int) -> bool:
        if self.model_type == "gpt_oss":
            return self.layer_types[layer] == "sliding_attention"
        return bool(self.sliding_window)

    def sparse(self, layer: int) -> bool:
        """Whether ``layer`` uses the mixture of experts instead of the MLP."""

        return (
            self.num_experts > 0
            and layer not in self.mlp_only_layers
            and layer >= self.first_k_dense_replace
            and (layer + 1) % self.decoder_sparse_step == 0
        )

    def attention_factor(self) -> float:
        """YaRN's scale on cos and sin (so on q and k); 1 for other RoPE types."""

        scaling = self.rope_scaling or {}
        if scaling.get("rope_type", scaling.get("type")) != "yarn":
            return 1.0
        if scaling.get("attention_factor"):
            return float(scaling["attention_factor"])
        factor = self._yarn_factor()

        def get_mscale(scale: float, mscale: float = 1.0) -> float:
            return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

        mscale, all_dim = scaling.get("mscale"), scaling.get("mscale_all_dim")
        if mscale and all_dim:
            return get_mscale(factor, mscale) / get_mscale(factor, all_dim)
        return get_mscale(factor)

    def _yarn_factor(self) -> float:
        scaling = self.rope_scaling or {}
        if scaling.get("factor") is not None:
            return float(scaling["factor"])
        return self.max_position_embeddings / float(scaling["original_max_position_embeddings"])

    def inv_freq(self) -> np.ndarray:
        """RoPE inverse frequencies, as Hugging Face's rope initializers compute them."""

        dim = self.rotary_dim or self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        scaling = self.rope_scaling or {}
        kind = scaling.get("rope_type", scaling.get("type", "default"))
        if kind == "linear":
            return inv_freq / float(scaling["factor"])
        if kind == "yarn":
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
    """``positions`` [tokens], or [tokens, frequencies] (each frequency's own position)."""

    half = x.shape[-1] // 2
    positions = positions.astype(np.float64)
    angles = (positions if positions.ndim == 2 else positions[:, None]) * inv_freq[None, :]
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
        """``positions`` [3, tokens]: each token's temporal, row and column RoPE positions."""

        cfg = self.config
        prefix = f"model.layers.{index}.self_attn"
        tokens = x.shape[0]
        heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = self._project(f"{prefix}.q_proj", x).reshape(tokens, heads, dim)
        k = self._project(f"{prefix}.k_proj", x).reshape(tokens, kv_heads, dim)
        v = self._project(f"{prefix}.v_proj", x).reshape(tokens, kv_heads, dim)
        if cfg.qk_norm:
            q = self._norm(f"{prefix}.q_norm", q)
            k = self._norm(f"{prefix}.k_norm", k)
        inv_freq = cfg.inv_freq()
        factor = np.float32(cfg.attention_factor())
        rot = cfg.rotary_dim or dim  # partial RoPE: rotate-half within the first `rot` dimensions
        angles = positions[cfg.rope_axes()].T
        q = np.concatenate([_rope(q[..., :rot], angles, inv_freq), q[..., rot:]], axis=-1) * factor
        k = np.concatenate([_rope(k[..., :rot], angles, inv_freq), k[..., rot:]], axis=-1) * factor

        # Causal by sequence index (M-RoPE positions repeat within an image).
        i = np.arange(tokens)[:, None]
        j = np.arange(tokens)[None, :]
        allowed = j <= i
        if cfg.sliding(index):
            allowed &= (i - j) < cfg.sliding_window
        sinks = self.w.get(f"{prefix}.sinks")
        scale = dim ** -0.5
        group = heads // kv_heads
        out = np.empty((tokens, heads, dim), dtype=np.float32)
        for head in range(heads):
            kv = head // group
            scores = (q[:, head, :] @ k[:, kv, :].T).astype(np.float64) * scale
            scores = np.where(allowed, scores, -np.inf)
            if sinks is not None:  # one more logit per row, which attends to nothing
                scores = np.concatenate([scores, np.full((tokens, 1), float(sinks[head]))], axis=-1)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, head, :] = (probs[:, :tokens] @ v[:, kv, :]).astype(np.float32)
        return self._project(f"{prefix}.o_proj", out.reshape(tokens, heads * dim))

    def _mlp(self, index: int, x: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{index}.mlp"
        if self.config.sparse(index):
            return self._experts(prefix, x)
        gate = _silu(x @ self.w[f"{prefix}.gate_proj"].T)
        up = x @ self.w[f"{prefix}.up_proj"].T
        return (gate * up) @ self.w[f"{prefix}.down_proj"].T

    def router_logits(self, index: int, x: np.ndarray) -> np.ndarray:
        """The scores experts are chosen by (for GLM, sigmoid plus the correction bias)."""

        name = "router" if self.config.model_type == "gpt_oss" else "gate"
        logits = (x @ self.w[f"model.layers.{index}.mlp.{name}"].T).astype(np.float64)
        bias = self.w.get(f"model.layers.{index}.mlp.{name}.bias")
        if bias is not None:
            logits = logits + bias
        if self.config.model_type == "glm4_moe":
            return self._glm_selection(index, 1.0 / (1.0 + np.exp(-logits)))
        return logits

    def _glm_selection(self, index: int, scores: np.ndarray) -> np.ndarray:
        cfg = self.config
        selection = scores + self.w[f"model.layers.{index}.mlp.gate.e_score_correction_bias"]
        if cfg.n_group > 1:  # keep the topk_group groups with the largest top-2 sums; zero the rest
            groups = selection.reshape(selection.shape[0], cfg.n_group, -1)
            group_scores = -np.sort(-groups, axis=-1)[..., :2].sum(axis=-1)
            dropped = np.argsort(group_scores, axis=-1, kind="stable")[:, : cfg.n_group - cfg.topk_group]
            groups = groups.copy()
            np.put_along_axis(groups, dropped[..., None], 0.0, axis=1)
            selection = groups.reshape(selection.shape)
        return selection

    def route(self, index: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expert indices ``[tokens, k]`` and their weights for layer ``index``."""

        logits = self.router_logits(index, x)
        k = self.config.num_experts_per_tok
        if self.config.model_type == "glm4_moe":  # weights are the raw sigmoid scores of the chosen
            chosen = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
            raw = self._project(f"model.layers.{index}.mlp.gate", x).astype(np.float64)
            weights = np.take_along_axis(1.0 / (1.0 + np.exp(-raw)), chosen, axis=-1)
            if k > 1 and self.config.norm_topk_prob:
                weights /= weights.sum(axis=-1, keepdims=True)
            return chosen, weights * self.config.routed_scaling_factor
        chosen = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
        if self.config.model_type == "gpt_oss":  # softmax over the top-k logits
            top = np.take_along_axis(logits, chosen, axis=-1)
            weights = np.exp(top - top.max(axis=-1, keepdims=True))
            return chosen, weights / weights.sum(axis=-1, keepdims=True)
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        weights = np.take_along_axis(probs, chosen, axis=-1)
        if self.config.norm_topk_prob:
            weights /= weights.sum(axis=-1, keepdims=True)
        return chosen, weights

    def _expert_stack(self, prefix: str, name: str) -> np.ndarray:
        """``[experts, rows, cols]``: MLX's stacked tensor or Transformers' per-expert ones."""

        for stacked in (f"{prefix}.switch_mlp.{name}", f"{prefix}.experts.{name}"):
            if stacked in self.w:
                return self.w[stacked]
        return np.stack([self.w[f"{prefix}.experts.{e}.{name}"] for e in range(self.config.num_experts)])

    def _expert_bias(self, prefix: str, name: str, expert: int) -> np.ndarray | float:
        stacked = self.w.get(f"{prefix}.experts.{name}.bias")
        if stacked is not None:
            return stacked[expert]
        return self.w.get(f"{prefix}.experts.{expert}.{name}.bias", 0.0)

    def _experts(self, prefix: str, x: np.ndarray) -> np.ndarray:
        index = int(prefix.split(".")[2])
        chosen, weights = self.route(index, x)
        gate, up, down = (self._expert_stack(prefix, name) for name in ("gate_proj", "up_proj", "down_proj"))
        out = np.zeros_like(x, dtype=np.float32)
        limit = self.config.swiglu_limit
        for t in range(x.shape[0]):
            for e, p in zip(chosen[t], weights[t]):
                g = gate[e] @ x[t] + self._expert_bias(prefix, "gate_proj", e)
                u = up[e] @ x[t] + self._expert_bias(prefix, "up_proj", e)
                if self.config.model_type == "gpt_oss":
                    g = np.minimum(g, limit)
                    hidden = g / (1.0 + np.exp(-1.702 * g)) * (np.clip(u, -limit, limit) + 1.0)
                else:
                    hidden = _silu(g) * u
                y = down[e] @ hidden + self._expert_bias(prefix, "down_proj", e)
                out[t] += np.float32(p) * y
        if self.config.n_shared_experts:
            shared = f"{prefix}.shared_experts"
            gated = _silu(x @ self.w[f"{shared}.gate_proj"].T) * (x @ self.w[f"{shared}.up_proj"].T)
            out = out + gated @ self.w[f"{shared}.down_proj"].T
        return out

    def logits(self, token_ids: Sequence[int], injected: Mapping[int, np.ndarray] | None = None,
               positions: np.ndarray | None = None,
               deepstack: Sequence[Mapping[int, np.ndarray]] = ()) -> np.ndarray:
        """Return ``[len(token_ids), vocab]`` float32 logits.

        ``injected`` replaces the embeddings at some positions (image
        features); ``positions`` [3, tokens] gives the M-RoPE positions
        (default: each token's index on all three axes); ``deepstack[l]``
        adds rows to the hidden states at some positions after layer ``l``.
        """

        cfg = self.config
        tokens = np.asarray(token_ids, dtype=np.int64)
        if positions is None:
            positions = np.broadcast_to(np.arange(tokens.shape[0]), (3, tokens.shape[0]))
        x = self.w["model.embed_tokens"][tokens].copy()
        for position, row in (injected or {}).items():
            x[position] = row
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            x = x + self._attention(index, self._norm(f"{prefix}.input_layernorm", x), positions)
            x = x + self._mlp(index, self._norm(f"{prefix}.post_attention_layernorm", x))
            for position, row in (deepstack[index] if index < len(deepstack) else {}).items():
                x[position] = x[position] + row
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
