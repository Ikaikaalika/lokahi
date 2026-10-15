"""NumPy reference for Gemma 4's image path: processor, vision tower, embedder.

Semantics follow Transformers' ``Gemma4ImageProcessor``,
``Gemma4VisionModel`` and ``Gemma4MultimodalEmbedder``:

* an image is resized (by the caller) to a size whose sides are multiples of
  ``patch_size * pooling_kernel_size`` within ``max_soft_tokens *
  pooling_kernel_size ** 2`` patches (``target_size``), rescaled to ``[0, 1]``
  and cut into patches, each flattened as (row, column, channel), with (x, y)
  patch coordinates;
* patches are mapped to ``2 * (p - 0.5)`` and projected (no bias); learned x-
  and y-position tables are added;
* each encoder layer is sandwich-normed (RMSNorm scaled by its weight):
  ``x += post_attn_norm(attn(input_norm(x)))`` then
  ``x += post_ff_norm(mlp(pre_ff_norm(x)))``; attention applies per-head
  RMSNorm to queries and keys and a scale-free one to values, rotates queries
  and keys with axial 2-D RoPE (the first half of each head by x, the second
  by y, each rotate-half within its half), uses scale 1 and sees every patch;
  the MLP is ``down(gelu_tanh(gate(x)) * up(x))``;
* the patch grid is average-pooled over ``k x k`` squares (row-major over the
  pooled grid), multiplied by ``sqrt(hidden_size)``, and standardized as
  ``(h - std_bias) * std_scale`` when ``standardize``;
* the embedder applies a scale-free RMSNorm and projects into the text width.

The features replace the image tokens' scaled embeddings as they are. The
prompt lays an image out as begin-of-image, one image token per feature row,
end-of-image. Correctness evidence only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights
from .gemma4 import Gemma4Reference


@dataclass(frozen=True)
class Gemma4VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    patch_size: int
    pooling_kernel_size: int
    max_soft_tokens: int
    position_embedding_size: int
    rope_theta: float
    rms_norm_eps: float
    standardize: bool
    image_token_id: int
    boi_token_id: int
    eoi_token_id: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Gemma4VisionConfig":
        v = raw["vision_config"]
        if v.get("use_clipped_linears"):
            raise NotImplementedError("clipped linears")
        heads = int(v.get("num_attention_heads", 12))
        if int(v.get("num_key_value_heads", heads)) != heads:
            raise NotImplementedError("grouped-query attention in the vision tower")
        return cls(
            hidden_size=int(v["hidden_size"]),
            intermediate_size=int(v["intermediate_size"]),
            num_hidden_layers=int(v["num_hidden_layers"]),
            num_attention_heads=heads,
            head_dim=int(v.get("head_dim", 64)),
            patch_size=int(v.get("patch_size", 16)),
            pooling_kernel_size=int(v.get("pooling_kernel_size", 3)),
            max_soft_tokens=int(raw.get("vision_soft_tokens_per_image", v.get("default_output_length", 280))),
            position_embedding_size=int(v.get("position_embedding_size", 10240)),
            rope_theta=float((v.get("rope_parameters") or {}).get("rope_theta", 100.0)),
            rms_norm_eps=float(v.get("rms_norm_eps", 1e-6)),
            standardize=bool(v.get("standardize", False)),
            image_token_id=int(raw["image_token_id"]),
            boi_token_id=int(raw["boi_token_id"]),
            eoi_token_id=int(raw["eoi_token_id"]),
        )

    @property
    def side_multiple(self) -> int:
        return self.patch_size * self.pooling_kernel_size

    @property
    def max_patches(self) -> int:
        return self.max_soft_tokens * self.pooling_kernel_size**2

    def tokens_for(self, height: int, width: int) -> int:
        """Soft tokens of an image already at an accepted size."""

        m = self.side_multiple
        if height % m or width % m or height * width > self.max_patches * self.patch_size**2:
            raise ValueError(f"a {height}x{width} image is not an accepted Gemma 4 size")
        if max(height, width) // self.patch_size > self.position_embedding_size:
            raise ValueError(f"a {height}x{width} image is wider than the position table")
        return (height // m) * (width // m)

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """The processor's aspect-preserving size (height, width) for an image."""

        target_px = self.max_patches * self.patch_size**2
        factor = math.sqrt(target_px / (height * width))
        m = self.side_multiple
        target_h = int(math.floor(factor * height / m)) * m
        target_w = int(math.floor(factor * width / m)) * m
        if target_h == 0 and target_w == 0:
            raise ValueError("the image would resize to 0 x 0")
        max_side = (self.max_patches // self.pooling_kernel_size**2) * m
        if target_h == 0:
            target_h, target_w = m, min(int(math.floor(width / height)) * m, max_side)
        elif target_w == 0:
            target_w, target_h = m, min(int(math.floor(height / width)) * m, max_side)
        if target_h * target_w > target_px:
            raise ValueError("the resized image exceeds the patch budget")
        return target_h, target_w


def patchify(rgb: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """``[H, W, 3]`` uint8 to patches ``[N, p*p*3]`` in ``[0, 1]`` and (x, y) positions ``[N, 2]``."""

    image = np.asarray(rgb, dtype=np.float32) * np.float32(1.0 / 255.0)
    h, w = image.shape[0] // patch_size, image.shape[1] // patch_size
    patches = image.reshape(h, patch_size, w, patch_size, 3).transpose(0, 2, 1, 3, 4).reshape(h * w, -1)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    positions = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
    return np.ascontiguousarray(patches), positions


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    mean_squared = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(mean_squared + eps)).astype(np.float32)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _rope_2d(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Axial RoPE over ``[N, heads, D]``: the first ``D / 2`` channels by x, the rest by y."""

    dim = x.shape[-1]
    spatial = dim // 2
    half = spatial // 2
    inv_freq = 1.0 / theta ** (np.arange(0, spatial, 2, dtype=np.float64) / spatial)
    out = np.empty_like(x)
    for axis in range(2):
        part = x[..., axis * spatial : (axis + 1) * spatial]
        angles = positions[:, axis : axis + 1].astype(np.float64) * inv_freq[None, :]
        cos = np.cos(angles).astype(np.float32)[:, None, :]
        sin = np.sin(angles).astype(np.float32)[:, None, :]
        a, b = part[..., :half], part[..., half:]
        out[..., axis * spatial : axis * spatial + half] = a * cos - b * sin
        out[..., axis * spatial + half : (axis + 1) * spatial] = b * cos + a * sin
    return out


class Gemma4VisionReference:
    """Float32 Gemma 4 vision tower, embedder and language model."""

    TOWER = "vision_tower"

    def __init__(self, config: Gemma4VisionConfig, weights: Mapping[str, np.ndarray], language: Gemma4Reference):
        self.config = config
        self.w = dict(weights)
        self.language = language

    @classmethod
    def load(cls, model_dir: Path) -> "Gemma4VisionReference":
        raw, weights = load_dense_weights(model_dir)
        return cls(Gemma4VisionConfig.from_dict(raw), weights, Gemma4Reference.load(model_dir))

    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _rms(x, self.config.rms_norm_eps) * self.w[name]

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """An RGB8 image ``[H, W, 3]`` at an accepted size to features ``[tokens, text_hidden]``."""

        cfg, t = self.config, self.TOWER
        height, width = rgb.shape[:2]
        tokens = cfg.tokens_for(height, width)
        patches, positions = patchify(rgb, cfg.patch_size)
        x = (2.0 * (patches - 0.5)).astype(np.float32) @ self.w[f"{t}.patch_embedder.input_proj"].T
        table = self.w[f"{t}.patch_embedder.position_embedding_table"]
        x = (x + table[0][positions[:, 0]] + table[1][positions[:, 1]]).astype(np.float32)
        heads, dim = cfg.num_attention_heads, cfg.head_dim
        n = x.shape[0]
        for index in range(cfg.num_hidden_layers):
            layer = f"{t}.encoder.layers.{index}"
            h = self._norm(f"{layer}.input_layernorm", x)
            q = (h @ self.w[f"{layer}.self_attn.q_proj.linear"].T).reshape(n, heads, dim)
            k = (h @ self.w[f"{layer}.self_attn.k_proj.linear"].T).reshape(n, heads, dim)
            v = (h @ self.w[f"{layer}.self_attn.v_proj.linear"].T).reshape(n, heads, dim)
            q = _rope_2d(self._norm(f"{layer}.self_attn.q_norm", q), positions, cfg.rope_theta)
            k = _rope_2d(self._norm(f"{layer}.self_attn.k_norm", k), positions, cfg.rope_theta)
            v = _rms(v, cfg.rms_norm_eps)
            out = np.empty_like(q)
            for head in range(heads):
                scores = (q[:, head, :] @ k[:, head, :].T).astype(np.float64)  # scale 1
                scores -= scores.max(axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=-1, keepdims=True)
                out[:, head, :] = (probs @ v[:, head, :]).astype(np.float32)
            attn = out.reshape(n, heads * dim) @ self.w[f"{layer}.self_attn.o_proj.linear"].T
            x = x + self._norm(f"{layer}.post_attention_layernorm", attn)
            h = self._norm(f"{layer}.pre_feedforward_layernorm", x)
            gate = _gelu_tanh(h @ self.w[f"{layer}.mlp.gate_proj.linear"].T)
            mlp = (gate * (h @ self.w[f"{layer}.mlp.up_proj.linear"].T)) @ self.w[f"{layer}.mlp.down_proj.linear"].T
            x = x + self._norm(f"{layer}.post_feedforward_layernorm", mlp)
        # Pool k x k squares of the (rows, columns) patch grid, row-major.
        k = cfg.pooling_kernel_size
        rows, cols = height // cfg.patch_size, width // cfg.patch_size
        pooled = x.reshape(rows // k, k, cols // k, k, cfg.hidden_size).mean(axis=(1, 3), dtype=np.float64)
        pooled = pooled.reshape(tokens, cfg.hidden_size).astype(np.float32) * np.float32(math.sqrt(cfg.hidden_size))
        if cfg.standardize:
            pooled = (pooled - self.w[f"{t}.std_bias"]) * self.w[f"{t}.std_scale"]
        return (_rms(pooled, cfg.rms_norm_eps) @ self.w["embed_vision.embedding_projection"].T).astype(np.float32)

    def placement(self, token_ids: Sequence[int], images: Sequence[np.ndarray]):
        """The injected rows and image blocks for images at their token runs, in order."""

        cfg = self.config
        positions = [i for i, t in enumerate(token_ids) if t == cfg.image_token_id]
        features = [self.encode(image) for image in images]
        if len(positions) != sum(f.shape[0] for f in features):
            raise ValueError("image token count does not match the images")
        injected, blocks, start = {}, [], 0
        for image in features:
            run = positions[start : start + image.shape[0]]
            start += image.shape[0]
            if run != list(range(run[0], run[0] + len(run))):
                raise ValueError("each image's tokens must be contiguous")
            blocks.append((run[0], run[-1] + 1))
            injected.update(zip(run, image))
        return injected, blocks

    def logits(self, token_ids: Sequence[int], images: Sequence[np.ndarray]) -> np.ndarray:
        injected, blocks = self.placement(token_ids, images)
        return self.language.logits(token_ids, injected, blocks)

    def greedy(self, prompt: Sequence[int], images: Sequence[np.ndarray], steps: int) -> list[int]:
        """Greedy continuation by full recomputation; images stay at their prompt positions."""

        injected, blocks = self.placement(prompt, images)
        tokens = list(prompt)
        generated: list[int] = []
        for _ in range(steps):
            next_token = int(np.argmax(self.language.logits(tokens, injected, blocks)[-1]))
            generated.append(next_token)
            tokens.append(next_token)
        return generated
