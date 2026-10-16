"""NumPy reference for Qwen3.5 and Qwen3-Next text: Gated DeltaNet layers between gated attention.

Semantics follow Transformers' ``Qwen3_5ForCausalLM`` (the text model of
``Qwen3_5ForConditionalGeneration``), ``Qwen3_5MoeForCausalLM`` and
``Qwen3NextForCausalLM``:

* RMSNorms are zero-centered, ``x * rsqrt(mean(x^2) + eps) * (1 + w)``: the
  input, post-attention and final norms and the per-head q/k norms. MLX
  conversions store ``1 + w`` (and their conv1d weights as [C, K, 1] rather
  than [C, 1, K], which is how the reference tells the layouts apart);
* ``layer_types`` lists each layer as ``linear_attention`` or
  ``full_attention`` (by default every ``full_attention_interval``-th, 4, is
  full);
* full attention: ``q_proj`` gives each head its query and an output gate
  (``head_dim`` each, in that order); q and k are RMS-normed per head; RoPE
  rotates halves within the first ``partial_rotary_factor`` of each head;
  grouped-query softmax attention; the output times ``sigmoid(gate)``;
  ``o_proj``. Qwen3.5's RoPE is multimodal (interleaved M-RoPE): each token
  has temporal, height and width positions, and rotation frequency i reads
  the height position when ``i % 3 == 1`` and ``i < 3 * mrope_section[1]``,
  the width position when ``i % 3 == 2`` and ``i < 3 * mrope_section[2]``,
  and the temporal one otherwise. Text tokens' three positions coincide, so
  for text it is plain RoPE; image tokens take their grid coordinates
  (``qwen3_5_vision.rope_positions``);
* Gated DeltaNet: ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b`` and
  ``in_proj_a`` (Qwen3.5), or Qwen3-Next's ``in_proj_qkvz`` and ``in_proj_ba``
  grouped per key head; a depthwise causal conv1d (kernel
  ``linear_conv_kernel_dim``, no bias) over the q, k and v channels, then
  SiLU; ``beta = sigmoid(b)`` and ``g = -exp(A_log) * softplus(a + dt_bias)``;
  q and k L2-normalized (``x * rsqrt(sum(x^2) + 1e-6)``) and q scaled by
  ``dk^-0.5``; value head h reads key head ``h // (Hv / Hk)``; per token the
  state S [dk, dv] decays by ``exp(g)``, takes ``k (beta (v - S^T k))^T`` and
  outputs ``S^T q`` (the recurrent form, which Transformers' chunked prefill
  computes in another order); a gated RMSNorm per value head,
  ``rms(o) * w * silu(z)`` (a plain weight); ``out_proj``;
* MLP: SiLU-gated; mixture-of-experts layers (every layer of Qwen3.5-MoE,
  which always renormalizes; Qwen3-Next's by ``decoder_sparse_step`` and
  ``mlp_only_layers``, renormalized with ``norm_topk_prob``) take the top
  ``num_experts_per_tok`` of a softmax router plus a shared expert times
  ``sigmoid(shared_expert_gate(x))``. Transformers saves Qwen3.5-MoE's
  experts fused, ``gate_up_proj`` [E, 2I, H] with the gate rows first.

Quantized MLX checkpoints are dequantized to float32 on load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights

FAMILIES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next")
L2_EPS = 1e-6


@dataclass(frozen=True)
class Qwen35Config:
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
    partial_rotary_factor: float
    layer_types: tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    mrope_section: tuple[int, ...] = ()  # empty: plain RoPE even for images (Qwen3-Next)
    mrope_interleaved: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Qwen35Config":
        cfg = raw.get("text_config") or raw
        model_type = cfg.get("model_type", raw.get("model_type"))
        if model_type not in FAMILIES:
            raise NotImplementedError(f"model_type {model_type!r}")
        rope = cfg.get("rope_parameters") or {}
        layers = int(cfg["num_hidden_layers"])
        types = cfg.get("layer_types")
        if types is None:
            interval = int(cfg.get("full_attention_interval", 4))
            types = ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(layers)]
        heads = int(cfg["num_attention_heads"])
        moe = model_type.startswith("qwen3_5_moe")
        if not cfg.get("attn_output_gate", True):
            raise NotImplementedError("attn_output_gate false")
        return cls(
            model_type=model_type,
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg.get("intermediate_size") or 0),
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", cfg.get("rope_theta", 10_000.0))),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25))),
            layer_types=tuple(types),
            linear_num_key_heads=int(cfg["linear_num_key_heads"]),
            linear_num_value_heads=int(cfg["linear_num_value_heads"]),
            linear_key_head_dim=int(cfg["linear_key_head_dim"]),
            linear_value_head_dim=int(cfg["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(cfg.get("linear_conv_kernel_dim", 4)),
            num_experts=int(cfg.get("num_experts") or 0),
            num_experts_per_tok=int(cfg.get("num_experts_per_tok") or 0),
            moe_intermediate_size=int(cfg.get("moe_intermediate_size") or 0),
            shared_expert_intermediate_size=int(cfg.get("shared_expert_intermediate_size") or 0),
            norm_topk_prob=True if moe else bool(cfg.get("norm_topk_prob", True)),
            decoder_sparse_step=1 if moe else int(cfg.get("decoder_sparse_step") or 1),
            mlp_only_layers=() if moe else tuple(cfg.get("mlp_only_layers") or ()),
            mrope_section=tuple(int(s) for s in rope.get("mrope_section") or ()),
            mrope_interleaved=bool(rope.get("mrope_interleaved", True)),
        )

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def sparse(self, layer: int) -> bool:
        return (
            self.num_experts > 0
            and layer not in self.mlp_only_layers
            and (layer + 1) % self.decoder_sparse_step == 0
        )

    def rope_axes(self) -> np.ndarray:
        """The position axis (0 temporal, 1 height, 2 width) each rotation frequency reads."""

        axes = np.zeros(self.rotary_dim // 2, dtype=np.int64)
        if self.mrope_section:
            if not self.mrope_interleaved:
                raise NotImplementedError("chunked (non-interleaved) M-RoPE")
            for axis in (1, 2):
                axes[axis : 3 * self.mrope_section[axis] : 3] = axis
        return axes

    def inv_freq(self) -> np.ndarray:
        dim = self.rotary_dim
        return 1.0 / (self.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)).astype(np.float32)


class Qwen35Reference:
    """Full-recompute float32 model for Qwen3.5 and Qwen3-Next text."""

    def __init__(self, config: Qwen35Config, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.w = dict(weights)
        if "lm_head" not in self.w:
            self.w["lm_head"] = self.w["model.embed_tokens"]
        # MLX conversions store the zero-centered norms as 1 + w and their
        # depthwise conv weights as [C, K, 1].
        first = next(i for i, kind in enumerate(config.layer_types) if kind == "linear_attention")
        conv = self.w[f"model.layers.{first}.linear_attn.conv1d"]
        self.mlx_layout = conv.shape[-1] == 1 and config.linear_conv_kernel_dim != 1
        self.norm_offset = 0.0 if self.mlx_layout else 1.0

    @classmethod
    def load(cls, model_dir: Path) -> "Qwen35Reference":
        raw, weights = load_dense_weights(model_dir)
        return cls(Qwen35Config.from_dict(raw), weights)

    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms(x, self.config.rms_norm_eps) * (self.norm_offset + self.w[name])

    def _attention(self, index: int, x: np.ndarray, positions: np.ndarray, prefix: str | None = None,
                   head: "Qwen35Mtp | None" = None) -> np.ndarray:
        """``positions`` [3, tokens]: each token's temporal, height and width RoPE positions.

        ``head``: the weights and norms are an MTP head's, under ``prefix``.
        """

        cfg = self.config
        w, norm = (self.w, self._norm) if head is None else (head.w, head.norm)
        prefix = prefix or f"model.layers.{index}.self_attn"
        tokens, heads, kv_heads, dim = x.shape[0], cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q_and_gate = (x @ w[f"{prefix}.q_proj"].T).reshape(tokens, heads, 2 * dim)
        q, gate = q_and_gate[..., :dim], q_and_gate[..., dim:].reshape(tokens, heads * dim)
        k = (x @ w[f"{prefix}.k_proj"].T).reshape(tokens, kv_heads, dim)
        v = (x @ w[f"{prefix}.v_proj"].T).reshape(tokens, kv_heads, dim)
        q = norm(f"{prefix}.q_norm", q)
        k = norm(f"{prefix}.k_norm", k)
        rot = cfg.rotary_dim
        angles = positions[cfg.rope_axes()].T.astype(np.float64) * cfg.inv_freq()[None, :]
        cos = np.cos(angles).astype(np.float32)[:, None, :]
        sin = np.sin(angles).astype(np.float32)[:, None, :]

        def rope(t: np.ndarray) -> np.ndarray:
            half = rot // 2
            a, b = t[..., :half], t[..., half:rot]
            return np.concatenate([a * cos - b * sin, b * cos + a * sin, t[..., rot:]], axis=-1)

        q, k = rope(q), rope(k)
        order = np.arange(tokens)
        allowed = order[None, :] <= order[:, None]  # causal by sequence index, not RoPE position
        group = heads // kv_heads
        out = np.empty((tokens, heads, dim), dtype=np.float32)
        for head in range(heads):
            scores = (q[:, head] @ k[:, head // group].T).astype(np.float64) * dim**-0.5
            scores = np.where(allowed, scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, head] = (probs @ v[:, head // group]).astype(np.float32)
        out = out.reshape(tokens, heads * dim) * (1.0 / (1.0 + np.exp(-gate)))
        return out @ w[f"{prefix}.o_proj"].T

    def _linear_projections(self, prefix: str, x: np.ndarray) -> tuple[np.ndarray, ...]:
        """``[q | k | v]`` channels, z [tokens, Hv * dv], b and a [tokens, Hv]."""

        cfg = self.config
        hk, hv, dk, dv = (cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
                          cfg.linear_value_head_dim)
        tokens = x.shape[0]
        if f"{prefix}.in_proj_qkv" in self.w:
            return (x @ self.w[f"{prefix}.in_proj_qkv"].T, x @ self.w[f"{prefix}.in_proj_z"].T,
                    x @ self.w[f"{prefix}.in_proj_b"].T, x @ self.w[f"{prefix}.in_proj_a"].T)
        # Qwen3-Next: each key head's [q, k, v, z] rows, then its [b, a].
        r = hv // hk
        qkvz = (x @ self.w[f"{prefix}.in_proj_qkvz"].T).reshape(tokens, hk, 2 * dk + 2 * r * dv)
        ba = (x @ self.w[f"{prefix}.in_proj_ba"].T).reshape(tokens, hk, 2 * r)
        q, k = qkvz[..., :dk], qkvz[..., dk:2 * dk]
        v, z = qkvz[..., 2 * dk:2 * dk + r * dv], qkvz[..., 2 * dk + r * dv:]
        mixed = np.concatenate([q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)], axis=-1)
        return mixed, z.reshape(tokens, -1), ba[..., :r].reshape(tokens, -1), ba[..., r:].reshape(tokens, -1)

    def _linear_attention(self, index: int, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        prefix = f"model.layers.{index}.linear_attn"
        hk, hv, dk, dv = (cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
                          cfg.linear_value_head_dim)
        kernel = cfg.linear_conv_kernel_dim
        tokens = x.shape[0]
        mixed, z, b, a = self._linear_projections(prefix, x)
        weight = self.w[f"{prefix}.conv1d"]
        weight = weight[..., 0] if self.mlx_layout else weight[:, 0, :]  # [C, K]
        padded = np.concatenate([np.zeros((kernel - 1, mixed.shape[1]), np.float32), mixed], axis=0)
        conv = np.stack([np.sum(padded[t:t + kernel] * weight.T, axis=0) for t in range(tokens)])
        conv = _silu(conv)
        q = conv[:, :hk * dk].reshape(tokens, hk, dk).astype(np.float64)
        k = conv[:, hk * dk:2 * hk * dk].reshape(tokens, hk, dk).astype(np.float64)
        v = conv[:, 2 * hk * dk:].reshape(tokens, hv, dv).astype(np.float64)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + L2_EPS) * dk**-0.5
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + L2_EPS)
        beta = 1.0 / (1.0 + np.exp(-b.astype(np.float64)))
        A = np.exp(self.w[f"{prefix}.A_log"].astype(np.float64))
        softplus = np.logaddexp(0.0, a.astype(np.float64) + self.w[f"{prefix}.dt_bias"])
        decay = np.exp(-A * softplus)  # [tokens, Hv]
        ratio = hv // hk
        out = np.zeros((tokens, hv, dv))
        for head in range(hv):
            state = np.zeros((dk, dv))
            kh = head // ratio
            for t in range(tokens):
                state *= decay[t, head]
                delta = (v[t, head] - state.T @ k[t, kh]) * beta[t, head]
                state += np.outer(k[t, kh], delta)
                out[t, head] = state.T @ q[t, kh]
        out = out.astype(np.float32)
        gated = _rms(out, cfg.rms_norm_eps) * self.w[f"{prefix}.norm"] * _silu(z.reshape(tokens, hv, dv))
        return gated.reshape(tokens, hv * dv) @ self.w[f"{prefix}.out_proj"].T

    def _dense_mlp(self, prefix: str, x: np.ndarray, w: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        w = self.w if w is None else w
        return (_silu(x @ w[f"{prefix}.gate_proj"].T) * (x @ w[f"{prefix}.up_proj"].T)) @ w[f"{prefix}.down_proj"].T

    def route(self, index: int, x: np.ndarray, prefix: str | None = None,
              w: Mapping[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Expert indices ``[tokens, k]`` and their weights for layer ``index`` (or the MLP under ``prefix``)."""

        w = self.w if w is None else w
        prefix = prefix or f"model.layers.{index}.mlp"
        logits = (x @ w[f"{prefix}.gate"].T).astype(np.float64)
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        chosen = np.argsort(-probs, axis=-1, kind="stable")[:, : self.config.num_experts_per_tok]
        weights = np.take_along_axis(probs, chosen, axis=-1)
        if self.config.norm_topk_prob:
            weights /= weights.sum(axis=-1, keepdims=True)
        return chosen, weights

    def _experts(self, index: int, x: np.ndarray, prefix: str | None = None,
                 w: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        w = self.w if w is None else w
        prefix = prefix or f"model.layers.{index}.mlp"
        chosen, weights = self.route(index, x, prefix, w)

        def stack(name: str) -> np.ndarray:
            if f"{prefix}.switch_mlp.{name}" in w:
                return w[f"{prefix}.switch_mlp.{name}"]
            return np.stack([w[f"{prefix}.experts.{e}.{name}"] for e in range(self.config.num_experts)])

        if f"{prefix}.experts.gate_up_proj" in w:  # Transformers' fused [E, 2I, H], gate first
            gate, up = np.split(w[f"{prefix}.experts.gate_up_proj"], 2, axis=1)
            down = w[f"{prefix}.experts.down_proj"]
        else:
            gate, up, down = stack("gate_proj"), stack("up_proj"), stack("down_proj")
        out = np.zeros_like(x, dtype=np.float32)
        for t in range(x.shape[0]):
            for e, p in zip(chosen[t], weights[t]):
                out[t] += np.float32(p) * (down[e] @ (_silu(gate[e] @ x[t]) * (up[e] @ x[t])))
        if self.config.shared_expert_intermediate_size:
            shared = self._dense_mlp(f"{prefix}.shared_expert", x, w)
            gate_logit = x @ w[f"{prefix}.shared_expert_gate"].T  # [tokens, 1]
            out = out + shared * (1.0 / (1.0 + np.exp(-gate_logit)))
        return out

    def logits(self, token_ids: Sequence[int], injected: Mapping[int, np.ndarray] | None = None,
               positions: np.ndarray | None = None) -> np.ndarray:
        """Return ``[len(token_ids), vocab]`` float32 logits.

        ``injected`` replaces the embeddings at some positions (image
        features); ``positions`` [3, tokens] gives the M-RoPE positions
        (default: each token's index on all three axes).
        """

        return (self.hidden_states(token_ids, injected, positions) @ self.w["lm_head"].T).astype(np.float32)

    def hidden_states(self, token_ids: Sequence[int], injected: Mapping[int, np.ndarray] | None = None,
                      positions: np.ndarray | None = None) -> np.ndarray:
        """The final-normed hidden states ``[len(token_ids), hidden]`` that ``logits`` projects."""

        cfg = self.config
        tokens = np.asarray(token_ids, dtype=np.int64)
        if positions is None:
            positions = np.broadcast_to(np.arange(tokens.shape[0]), (3, tokens.shape[0]))
        x = self.w["model.embed_tokens"][tokens].copy()
        for position, row in (injected or {}).items():
            x[position] = row
        for index in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{index}"
            h = self._norm(f"{prefix}.input_layernorm", x)
            if cfg.layer_types[index] == "linear_attention":
                x = x + self._linear_attention(index, h)
            else:
                x = x + self._attention(index, h, positions)
            h = self._norm(f"{prefix}.post_attention_layernorm", x)
            x = x + (self._experts(index, h) if cfg.sparse(index) else self._dense_mlp(f"{prefix}.mlp", h))
        return self._norm("model.norm", x)

    def greedy(self, prompt: Sequence[int], steps: int) -> list[int]:
        """Greedy continuation by full recomputation (no cache)."""

        tokens = list(prompt)
        generated: list[int] = []
        for _ in range(steps):
            next_token = int(np.argmax(self.logits(tokens)[-1]))
            generated.append(next_token)
            tokens.append(next_token)
        return generated


class Qwen35Mtp:
    """Qwen3.5's multi-token prediction head, as vLLM's
    ``Qwen3NextMultiTokenPredictor`` and MLX-VLM's ``qwen3_5_mtp`` drafter run it.

    The head's row at position t pairs the target's final-normed hidden state
    h_t with token t + 1: ``x = fc([norm_e(embed(token)) | norm_h(h_t)])``,
    one decoder layer (gated full attention over the head's earlier rows at
    position t, then the target's kind of MLP), then ``norm``. The target's
    ``lm_head`` turns the output into logits for token t + 2, and the output
    stands in for h at the next position when the head drafts again. Every
    row of the context comes from the target's hidden states (vLLM rewrites
    the rows of accepted drafts), so the head's state after any run of
    speculative decoding is this full recomputation's.

    Qwen stores the head's zero-centered norms as w; MLX-VLM's drafter
    directories (and MLX conversions) as 1 + w.
    """

    def __init__(self, target: Qwen35Reference, weights: Mapping[str, np.ndarray], norm_offset: float):
        self.target, self.w, self.norm_offset = target, dict(weights), norm_offset
        self.sparse = "layers.0.mlp.gate" in self.w

    @classmethod
    def from_checkpoint(cls, target: Qwen35Reference) -> "Qwen35Mtp":
        """The ``mtp.*`` head beside the target's decoder."""

        weights = {name[len("mtp."):]: value for name, value in target.w.items() if name.startswith("mtp.")}
        return cls(target, weights, target.norm_offset)

    @classmethod
    def load(cls, target: Qwen35Reference, drafter_dir: Path) -> "Qwen35Mtp":
        """A ``qwen3_5_mtp`` drafter directory's head."""

        raw, weights = load_dense_weights(drafter_dir)
        if raw.get("model_type") != "qwen3_5_mtp":
            raise ValueError(f"{drafter_dir}: expected a qwen3_5_mtp drafter")
        return cls(target, weights, 0.0)

    def norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms(x, self.target.config.rms_norm_eps) * (self.norm_offset + self.w[name])

    def outputs(self, tokens: Sequence[int], hidden: np.ndarray) -> np.ndarray:
        """The head's outputs for rows at positions 0, 1, ...: ``tokens[t]`` after ``hidden[t]``."""

        t = self.target
        count = len(tokens)
        embed = t.w["model.embed_tokens"][np.asarray(tokens, dtype=np.int64)]
        x = np.concatenate([self.norm("pre_fc_norm_embedding", embed), self.norm("pre_fc_norm_hidden", hidden)],
                           axis=-1) @ self.w["fc"].T
        positions = np.broadcast_to(np.arange(count), (3, count))
        h = self.norm("layers.0.input_layernorm", x)
        x = x + t._attention(0, h, positions, prefix="layers.0.self_attn", head=self)
        h = self.norm("layers.0.post_attention_layernorm", x)
        mlp = (t._experts(0, h, prefix="layers.0.mlp", w=self.w) if self.sparse
               else t._dense_mlp("layers.0.mlp", h, self.w))
        return self.norm("norm", x + mlp)

    def drafts(self, context: Sequence[int], steps: int) -> tuple[list[int], np.ndarray]:
        """The head's ``steps`` greedy drafts after ``context`` (whose last
        token the target has not read yet), and their logits ``[steps, vocab]``."""

        context = list(context)
        hidden = list(self.target.hidden_states(context[:-1]))
        tokens = context[1:]
        drafts: list[int] = []
        logits = []
        for _ in range(steps):
            out = self.outputs(tokens, np.stack(hidden))[-1]
            row = (out @ self.target.w["lm_head"].T).astype(np.float32)
            drafts.append(int(np.argmax(row)))
            logits.append(row)
            tokens.append(drafts[-1])
            hidden.append(out)
        return drafts, np.stack(logits)
