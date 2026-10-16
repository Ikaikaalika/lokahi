"""NumPy reference for Qwen2.5-VL: its window-attention tower and merger, and its Qwen 2 text model.

Semantics follow Transformers' ``Qwen2_5_VisionTransformerPretrainedModel``,
``get_vision_window_index`` and ``Qwen2_5_VLModel.get_rope_index``:

* the processor and the patch layout are Qwen3.5's (``qwen3_5_vision``):
  ``smart_resize`` to multiples of ``patch_size * spatial_merge_size``,
  CLIP's mean and std by default, and patches in merge-block order;
* the patch embedding is a Conv3d without a bias, and there are no learned
  position embeddings;
* each block is ``x += proj(attn(norm1(x)))``,
  ``x += down(silu(gate(norm2(x))) * up(norm2(x)))`` with RMSNorms (eps
  1e-6) and biased projections; attention uses Qwen3.5's 2-D RoPE;
* outside ``fullatt_block_indexes`` attention is windowed: the merged grid
  is cut into squares of ``window_size // spatial_merge_size // patch_size``
  tokens from its top-left corner (the right and bottom ones may be
  partial), and a patch sees only the patches of its window. Transformers
  reorders the patches window by window for this, and restores the order
  after the merger; so does this reference. The blocks in
  ``fullatt_block_indexes`` see the whole image;
* the merger RMSNorms each patch (``ln_q``), concatenates each
  ``merge x merge`` square and applies ``mlp.2(gelu(mlp.0(.)))`` with exact
  GELU into the text width;
* the text model is Qwen 2 (``llama``) with chunked M-RoPE (sections
  ``[16, 24, 24]`` by default), and image tokens take Qwen3.5's positions
  (``rope_positions``).

Checkpoints come in three layouts: MLX's (``vision_tower.*``,
``language_model.*``), Qwen's published one (Transformers 4: ``visual.*``,
``model.*``, ``lm_head``) and Transformers 5's (``model.visual.*``,
``model.language_model.*``). Their ``config.json`` keeps the text fields at
the top level, or (Transformers 5) in ``text_config``.

Correctness evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gemma3 import load_dense_weights
from .llama import LlamaFamilyConfig, LlamaFamilyReference, _rms_norm
from .qwen3_5_vision import Qwen35VisionConfig, Qwen35VisionReference, _gelu, merged_order, patchify, processor_fields

DEFAULT_SECTION = (16, 24, 24)


def text_config(raw: Mapping[str, Any]) -> LlamaFamilyConfig:
    """The Llama-family configuration of a Qwen2.5-VL checkpoint's text model."""

    if raw.get("model_type") != "qwen2_5_vl":
        raise NotImplementedError(f"model_type {raw.get('model_type')!r}")
    text = dict(raw.get("text_config") or {key: value for key, value in raw.items() if key != "vision_config"})
    if text.get("use_sliding_window") or any(t != "full_attention" for t in text.get("layer_types") or ()):
        raise NotImplementedError("sliding-window layers")
    heads = int(text["num_attention_heads"])
    if text.get("head_dim") not in (None, int(text["hidden_size"]) // heads):
        # Transformers' Qwen2.5-VL attention splits hidden_size evenly.
        raise NotImplementedError("a head_dim other than hidden_size / num_attention_heads")
    # Transformers reads rope_scaling when present, else rope_parameters, and
    # rope_theta from inside it, else beside it, else 1000000.
    scaling = text.get("rope_scaling") or text.get("rope_parameters") or {}
    if scaling.get("rope_type", scaling.get("type", "default")) not in ("default", "mrope"):
        raise NotImplementedError(f"rope {scaling!r}")
    if scaling.get("mrope_interleaved"):
        raise NotImplementedError("Qwen2.5-VL's M-RoPE sections are chunked, not interleaved")
    text["model_type"] = "qwen2"
    text["rope_scaling"] = {"rope_type": "default", "mrope_interleaved": False,
                            "mrope_section": list(scaling.get("mrope_section") or DEFAULT_SECTION)}
    text["rope_theta"] = scaling.get("rope_theta", text.get("rope_theta", 1_000_000.0))
    return LlamaFamilyConfig.from_dict(text)


@dataclass(frozen=True)
class Qwen25VisionConfig(Qwen35VisionConfig):
    window_size: int = 112
    fullatt_block_indexes: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], processor: Mapping[str, Any] | None = None) -> "Qwen25VisionConfig":
        v = raw["vision_config"]
        if v.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"vision activation {v.get('hidden_act')!r}")
        rope = v.get("rope_parameters") or {}
        if rope.get("rope_type", "axial") != "axial" or float(rope.get("rope_theta", 10000.0)) != 10000.0:
            raise NotImplementedError(f"vision rope {rope!r}")
        text = raw.get("text_config") or raw
        return cls(
            depth=int(v["depth"]),
            hidden_size=int(v["hidden_size"]),
            intermediate_size=int(v["intermediate_size"]),
            num_heads=int(v["num_heads"]),
            patch_size=int(v.get("patch_size", 14)),
            temporal_patch_size=int(v.get("temporal_patch_size", 2)),
            spatial_merge_size=int(v.get("spatial_merge_size", 2)),
            in_channels=int(v.get("in_channels", v.get("in_chans", 3))),
            out_hidden_size=int(v.get("out_hidden_size", text["hidden_size"])),
            num_position_embeddings=0,
            image_token_id=int(raw["image_token_id"]),
            vision_start_token_id=int(raw["vision_start_token_id"]),
            vision_end_token_id=int(raw["vision_end_token_id"]),
            window_size=int(v.get("window_size", 112)),
            fullatt_block_indexes=tuple(int(i) for i in v.get("fullatt_block_indexes", (7, 15, 23, 31))),
            **processor_fields(processor),
        )

    @property
    def window(self) -> int:
        """A window's side in merged tokens."""

        return self.window_size // self.spatial_merge_size // self.patch_size


def window_index(rows: int, cols: int, window: int) -> tuple[np.ndarray, list[int]]:
    """``get_vision_window_index`` for one image's merged grid (rows x cols):
    the merged tokens window by window, and each non-empty window's length."""

    index = np.arange(rows * cols).reshape(rows, cols)
    pad_h, pad_w = window - rows % window, window - cols % window
    padded = np.full((rows + pad_h, cols + pad_w), -100)
    padded[:rows, :cols] = index
    across, down = (cols + pad_w) // window, (rows + pad_h) // window
    padded = padded.reshape(down, window, across, window).transpose(0, 2, 1, 3).reshape(down * across, window**2)
    lengths = (padded != -100).sum(axis=1)
    flat = padded.reshape(-1)
    return flat[flat != -100], [int(length) for length in lengths if length > 0]


class Qwen25VLReference(Qwen35VisionReference):
    """Float32 Qwen2.5-VL: the window-attention tower and merger, then the Qwen 2 text model."""

    def __init__(self, config: Qwen25VisionConfig, weights: Mapping[str, np.ndarray], language: LlamaFamilyReference):
        super().__init__(config, weights, language)
        self.tower = next(t for t in ("vision_tower", "model.visual", "visual") if f"{t}.patch_embed.proj" in self.w)

    @classmethod
    def load(cls, model_dir: Path, processor: Mapping[str, Any] | None = None) -> "Qwen25VLReference":
        import json

        model_dir = Path(model_dir)
        raw, weights = load_dense_weights(model_dir)
        if processor is None and (model_dir / "preprocessor_config.json").exists():
            processor = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
        return cls(Qwen25VisionConfig.from_dict(raw, processor), weights, LlamaFamilyReference(text_config(raw), weights))

    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms_norm(x, self.w[name], 1e-6)

    def _linear(self, name: str, x: np.ndarray) -> np.ndarray:
        return x @ self.w[name].T + self.w[f"{name}.bias"]

    def encode_all(self, rgb: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """The features (and no deepstack features)."""

        cfg, t = self.config, self.tower
        m = cfg.spatial_merge_size
        rows, cols = cfg.grid(*rgb.shape[:2])
        x = (patchify(rgb, cfg) @ self._patch_weight().T).astype(np.float32)
        n, heads, dim, square = x.shape[0], cfg.num_heads, cfg.head_dim, m * m
        # Transformers' window order: merged tokens window by window, each
        # token's merge x merge patches together; RoPE follows the patches.
        order, lengths = window_index(rows // m, cols // m, cfg.window)
        patches = (order[:, None] * square + np.arange(square)[None, :]).reshape(-1)
        x = x[patches]
        r, c = (axis[patches] for axis in merged_order(rows, cols, m))
        bounds = np.cumsum([0] + [length * square for length in lengths])
        for index in range(cfg.depth):
            block = f"{t}.blocks.{index}"
            spans = [(0, n)] if index in cfg.fullatt_block_indexes else list(zip(bounds[:-1], bounds[1:]))
            h = self._norm(f"{block}.norm1", x)
            qkv = self._linear(f"{block}.attn.qkv", h).reshape(n, 3, heads, dim)
            q, k, v = self._rope_at(qkv[:, 0], r, c), self._rope_at(qkv[:, 1], r, c), qkv[:, 2]
            out = np.empty_like(q)
            for begin, end in spans:
                for head in range(heads):
                    scores = (q[begin:end, head] @ k[begin:end, head].T).astype(np.float64) * dim**-0.5
                    scores -= scores.max(axis=-1, keepdims=True)
                    probs = np.exp(scores)
                    probs /= probs.sum(axis=-1, keepdims=True)
                    out[begin:end, head] = (probs @ v[begin:end, head]).astype(np.float32)
            x = x + self._linear(f"{block}.attn.proj", out.reshape(n, heads * dim))
            h = self._norm(f"{block}.norm2", x)
            gate = self._linear(f"{block}.mlp.gate_proj", h)
            up = self._linear(f"{block}.mlp.up_proj", h)
            x = (x + self._linear(f"{block}.mlp.down_proj", gate / (1.0 + np.exp(-gate)) * up)).astype(np.float32)
        h = self._norm(f"{t}.merger.ln_q", x).reshape(-1, x.shape[1] * square)
        h = _gelu(self._linear(f"{t}.merger.mlp.0", h))
        merged = self._linear(f"{t}.merger.mlp.2", h).astype(np.float32)
        return merged[np.argsort(order)], []
