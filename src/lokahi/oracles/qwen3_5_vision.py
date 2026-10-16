"""NumPy reference for Qwen3.5's and Qwen3-VL's image path: processor, tower, mergers, M-RoPE positions.

Semantics follow Transformers' ``Qwen2VLImageProcessorFast`` (as
``Qwen3VLProcessor`` uses it), ``Qwen3VLVisionModel`` (which
``Qwen3_5VisionModel`` is without deepstack) and ``get_rope_index``:

* the processor resizes an image (the caller does, to ``target_size``) so
  both sides are multiples of ``patch_size * spatial_merge_size`` and its
  area lies within ``[min_pixels, max_pixels]`` (``smart_resize``, which
  rounds half to even as Python does), rescales to ``[0, 1]`` and normalizes
  with mean = std = 0.5, repeats the image ``temporal_patch_size`` times
  along time, and cuts it into patches flattened as (channel, time, row,
  column), ordered so each ``merge x merge`` square of patches is contiguous
  (block rows, block columns, then row and column within the block);
* the patch embedding is a Conv3d with stride = kernel, a linear map of each
  flattened patch with a bias; learned position embeddings
  ``[num_position_embeddings, hidden]`` over a square grid are bilinearly
  interpolated to the patch grid (``linspace`` sample points) and added;
* each block is ``x += proj(attn(norm1(x)))``, ``x += fc2(gelu_tanh(fc1(norm2(x))))``
  with LayerNorms (eps 1e-6); attention has a biased qkv projection, 2-D
  RoPE on the whole head (rotate-half; the first quarter of frequencies by
  the patch row, the second by its column, each half repeated) and sees
  every patch of the image;
* the merger LayerNorms each patch, concatenates each block's ``merge^2``
  patches and applies ``fc2(gelu(fc1(.)))`` (exact GELU) into the text width;
  merged tokens follow the merged grid row by row;
* Qwen3-VL's deepstack: after each block in ``deepstack_visual_indexes``, a
  merger of its own (which LayerNorms the concatenated square instead of each
  patch) gives one more feature row per token; the text model adds the i-th
  of them to its hidden states at the image's tokens after its i-th layer;
* in the prompt, image tokens take M-RoPE positions ``(s, s + row, s + col)``
  over the merged grid, ``s`` being the text position where the image
  starts; text after the image continues at ``s + max(rows, cols)``, and
  every later token's positions run ``delta`` ahead of its index.

Correctness evidence only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import load_dense_weights
from .qwen3_5 import Qwen35Reference


@dataclass(frozen=True)
class Qwen35VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    in_channels: int
    out_hidden_size: int
    num_position_embeddings: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    min_pixels: int = 65536
    max_pixels: int = 16777216
    deepstack_visual_indexes: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], processor: Mapping[str, Any] | None = None) -> "Qwen35VisionConfig":
        v = raw["vision_config"]
        if v.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"vision activation {v.get('hidden_act')!r}")
        size = (processor or {}).get("size") or {}
        return cls(
            depth=int(v["depth"]),
            hidden_size=int(v["hidden_size"]),
            intermediate_size=int(v["intermediate_size"]),
            num_heads=int(v["num_heads"]),
            patch_size=int(v.get("patch_size", 16)),
            temporal_patch_size=int(v.get("temporal_patch_size", 2)),
            spatial_merge_size=int(v.get("spatial_merge_size", 2)),
            in_channels=int(v.get("in_channels", 3)),
            out_hidden_size=int(v["out_hidden_size"]),
            num_position_embeddings=int(v["num_position_embeddings"]),
            image_token_id=int(raw["image_token_id"]),
            vision_start_token_id=int(raw["vision_start_token_id"]),
            vision_end_token_id=int(raw["vision_end_token_id"]),
            min_pixels=int(size.get("shortest_edge", 65536)),
            max_pixels=int(size.get("longest_edge", 16777216)),
            deepstack_visual_indexes=tuple(int(i) for i in v.get("deepstack_visual_indexes") or ()),
        )

    @property
    def factor(self) -> int:
        return self.patch_size * self.spatial_merge_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """``smart_resize``: the processor's size (height, width) for an image."""

        factor = self.factor
        if max(height, width) / min(height, width) > 200:
            raise ValueError("the image's aspect ratio exceeds 200")
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def grid(self, height: int, width: int) -> tuple[int, int]:
        """The patch grid (rows, columns) of an image already at an accepted size."""

        if height % self.factor or width % self.factor:
            raise ValueError(f"a {height}x{width} image is not an accepted Qwen3.5 size")
        return height // self.patch_size, width // self.patch_size

    def tokens_for(self, height: int, width: int) -> int:
        rows, cols = self.grid(height, width)
        return rows * cols // self.spatial_merge_size**2


def patchify(rgb: np.ndarray, cfg: Qwen35VisionConfig) -> np.ndarray:
    """``[H, W, 3]`` uint8 to normalized patches ``[N, C * T * P * P]`` in merge-block order."""

    p, m, t = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size
    rows, cols = cfg.grid(*rgb.shape[:2])
    image = (np.asarray(rgb, dtype=np.float32) * np.float32(1.0 / 255.0) - 0.5) / 0.5  # [H, W, C]
    chw = image.transpose(2, 0, 1)
    frames = np.broadcast_to(chw[None], (t,) + chw.shape)  # [T, C, H, W]
    c = chw.shape[0]
    patches = frames.reshape(t, c, rows // m, m, p, cols // m, m, p)
    # (block row, block column, row in block, column in block, channel, time, y, x)
    patches = patches.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return np.ascontiguousarray(patches.reshape(rows * cols, c * t * p * p))


def merged_order(rows: int, cols: int, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """Each patch's (row, column) in the merge-block order patchify uses."""

    r = (np.arange(rows // merge)[:, None, None, None] * merge + np.arange(merge)[None, None, :, None])
    c = (np.arange(cols // merge)[None, :, None, None] * merge + np.arange(merge)[None, None, None, :])
    shape = (rows // merge, cols // merge, merge, merge)
    return np.broadcast_to(r, shape).reshape(-1), np.broadcast_to(c, shape).reshape(-1)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x64 - mean) / np.sqrt(var + eps) * weight + bias).astype(np.float32)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _gelu(x: np.ndarray) -> np.ndarray:
    from math import erf

    return (0.5 * x * (1.0 + np.vectorize(erf)(x.astype(np.float64) / math.sqrt(2.0)))).astype(np.float32)


def rope_positions(token_ids: Sequence[int], grids: Sequence[tuple[int, int]], image_token_id: int,
                   start: int = 0) -> tuple[np.ndarray, int]:
    """M-RoPE positions [3, len(token_ids)] and the delta of every later token.

    ``grids`` are the images' merged grids (rows, columns), in prompt order;
    each image's tokens are a contiguous run of ``image_token_id``.
    ``start`` is the text position of the first token.
    """

    n = len(token_ids)
    out = np.zeros((3, n), dtype=np.int64)
    position, i, image = start, 0, 0
    while i < n:
        if token_ids[i] != image_token_id:
            out[:, i] = position
            position += 1
            i += 1
            continue
        rows, cols = grids[image]
        image += 1
        count = rows * cols
        if list(token_ids[i : i + count]) != [image_token_id] * count:
            raise ValueError("an image's tokens must be a contiguous run of its grid's size")
        r, c = np.divmod(np.arange(count), cols)
        out[0, i : i + count] = position
        out[1, i : i + count] = position + r
        out[2, i : i + count] = position + c
        position += max(rows, cols)
        i += count
    if image != len(grids):
        raise ValueError("more images than image token runs")
    return out, position - (start + n)


class Qwen35VisionReference:
    """Float32 Qwen3.5 vision tower and merger, with the language model."""

    def __init__(self, config: Qwen35VisionConfig, weights: Mapping[str, np.ndarray], language: Qwen35Reference):
        self.config = config
        self.w = dict(weights)
        self.language = language
        self.tower = "vision_tower" if "vision_tower.pos_embed" in self.w else "model.visual"

    @classmethod
    def load(cls, model_dir: Path, processor: Mapping[str, Any] | None = None) -> "Qwen35VisionReference":
        import json

        model_dir = Path(model_dir)
        raw, weights = load_dense_weights(model_dir)
        if processor is None and (model_dir / "preprocessor_config.json").exists():
            processor = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
        return cls(Qwen35VisionConfig.from_dict(raw, processor), weights, Qwen35Reference.load(model_dir))

    def _patch_weight(self) -> np.ndarray:
        """The Conv3d kernel as [hidden, C * T * P * P] over (channel, time, y, x)."""

        w = self.w[f"{self.tower}.patch_embed.proj"]
        if w.shape[1] != self.config.in_channels:  # MLX: [out, T, P, P, C]
            w = w.transpose(0, 4, 1, 2, 3)
        return w.reshape(w.shape[0], -1)

    def _position_embeddings(self, rows: int, cols: int) -> np.ndarray:
        cfg = self.config
        side = int(round(math.sqrt(cfg.num_position_embeddings)))
        table = self.w[f"{self.tower}.pos_embed"]
        h = np.linspace(0, side - 1, rows)
        w = np.linspace(0, side - 1, cols)
        h0, w0 = h.astype(np.int64), w.astype(np.int64)
        h1, w1 = np.minimum(h0 + 1, side - 1), np.minimum(w0 + 1, side - 1)
        dh, dw = (h - h0)[:, None], (w - w0)[None, :]
        grid = ((1 - dh) * (1 - dw))[..., None] * table[h0[:, None] * side + w0[None, :]]
        grid += ((1 - dh) * dw)[..., None] * table[h0[:, None] * side + w1[None, :]]
        grid += (dh * (1 - dw))[..., None] * table[h1[:, None] * side + w0[None, :]]
        grid += (dh * dw)[..., None] * table[h1[:, None] * side + w1[None, :]]
        r, c = merged_order(rows, cols, cfg.spatial_merge_size)
        return grid[r, c].astype(np.float32)

    def _rope(self, x: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """2-D RoPE over ``[N, heads, D]``: frequencies of a D / 2 rotary dimension, rows then columns."""

        cfg = self.config
        dim = cfg.head_dim // 2
        inv_freq = 1.0 / 10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
        r, c = merged_order(rows, cols, cfg.spatial_merge_size)
        freqs = np.concatenate([r[:, None] * inv_freq[None, :], c[:, None] * inv_freq[None, :]], axis=-1)
        cos = np.cos(np.concatenate([freqs, freqs], axis=-1)).astype(np.float32)[:, None, :]
        sin = np.sin(np.concatenate([freqs, freqs], axis=-1)).astype(np.float32)[:, None, :]
        half = cfg.head_dim // 2
        rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cos + rotated * sin

    def _merge(self, name: str, x: np.ndarray, postshuffle: bool) -> np.ndarray:
        """A merger: LayerNorm (per patch, or per concatenated square), then ``fc2(gelu(fc1))``."""

        square = self.config.spatial_merge_size**2
        if postshuffle:
            h = _layer_norm(x.reshape(-1, x.shape[1] * square), self.w[f"{name}.norm"], self.w[f"{name}.norm.bias"])
        else:
            h = _layer_norm(x, self.w[f"{name}.norm"], self.w[f"{name}.norm.bias"]).reshape(-1, x.shape[1] * square)
        h = _gelu(h @ self.w[f"{name}.linear_fc1"].T + self.w[f"{name}.linear_fc1.bias"])
        return (h @ self.w[f"{name}.linear_fc2"].T + self.w[f"{name}.linear_fc2.bias"]).astype(np.float32)

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """An RGB8 image ``[H, W, 3]`` at an accepted size to features ``[tokens, text_hidden]``."""

        return self.encode_all(rgb)[0]

    def encode_all(self, rgb: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """The features and the deepstack features (one ``[tokens, text_hidden]`` per index)."""

        cfg, t = self.config, self.tower
        rows, cols = cfg.grid(*rgb.shape[:2])
        x = patchify(rgb, cfg) @ self._patch_weight().T + self.w[f"{t}.patch_embed.proj.bias"]
        x = (x + self._position_embeddings(rows, cols)).astype(np.float32)
        n, heads, dim = x.shape[0], cfg.num_heads, cfg.head_dim
        deepstack: list[np.ndarray] = []
        for index in range(cfg.depth):
            block = f"{t}.blocks.{index}"
            h = _layer_norm(x, self.w[f"{block}.norm1"], self.w[f"{block}.norm1.bias"])
            qkv = (h @ self.w[f"{block}.attn.qkv"].T + self.w[f"{block}.attn.qkv.bias"]).reshape(n, 3, heads, dim)
            q, k, v = self._rope(qkv[:, 0], rows, cols), self._rope(qkv[:, 1], rows, cols), qkv[:, 2]
            out = np.empty_like(q)
            for head in range(heads):
                scores = (q[:, head] @ k[:, head].T).astype(np.float64) * dim**-0.5
                scores -= scores.max(axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=-1, keepdims=True)
                out[:, head] = (probs @ v[:, head]).astype(np.float32)
            x = x + out.reshape(n, heads * dim) @ self.w[f"{block}.attn.proj"].T + self.w[f"{block}.attn.proj.bias"]
            h = _layer_norm(x, self.w[f"{block}.norm2"], self.w[f"{block}.norm2.bias"])
            h = _gelu_tanh(h @ self.w[f"{block}.mlp.linear_fc1"].T + self.w[f"{block}.mlp.linear_fc1.bias"])
            x = (x + h @ self.w[f"{block}.mlp.linear_fc2"].T + self.w[f"{block}.mlp.linear_fc2.bias"]).astype(np.float32)
            if index in cfg.deepstack_visual_indexes:
                which = cfg.deepstack_visual_indexes.index(index)
                deepstack.append(self._merge(f"{t}.deepstack_merger_list.{which}", x, postshuffle=True))
        return self._merge(f"{t}.merger", x, postshuffle=False), deepstack

    def merged_grid(self, rgb: np.ndarray) -> tuple[int, int]:
        rows, cols = self.config.grid(*rgb.shape[:2])
        m = self.config.spatial_merge_size
        return rows // m, cols // m

    def _inputs(self, token_ids: Sequence[int], images: Sequence[np.ndarray]):
        """The injected rows, M-RoPE positions, later tokens' delta, and per-layer deepstack rows."""

        cfg = self.config
        grids = [self.merged_grid(image) for image in images]
        positions, delta = rope_positions(token_ids, grids, cfg.image_token_id)
        runs = [i for i, token in enumerate(token_ids) if token == cfg.image_token_id]
        injected, at = {}, 0
        deepstack: list[dict[int, np.ndarray]] = [{} for _ in cfg.deepstack_visual_indexes]
        for image in images:
            features, layers = self.encode_all(image)
            where = runs[at : at + features.shape[0]]
            injected.update(zip(where, features))
            for rows, extra in zip(deepstack, layers):
                rows.update(zip(where, extra))
            at += features.shape[0]
        if at != len(runs):
            raise ValueError("image token count does not match the images")
        return injected, positions, delta, deepstack

    def _language_logits(self, tokens, injected, positions, deepstack) -> np.ndarray:
        if deepstack:
            return self.language.logits(tokens, injected, positions, deepstack)
        return self.language.logits(tokens, injected, positions)

    def logits(self, token_ids: Sequence[int], images: Sequence[np.ndarray]) -> np.ndarray:
        injected, positions, _, deepstack = self._inputs(token_ids, images)
        return self._language_logits(token_ids, injected, positions, deepstack)

    def greedy(self, prompt: Sequence[int], images: Sequence[np.ndarray], steps: int) -> list[int]:
        """Greedy continuation by full recomputation; generated tokens take positions index + delta."""

        injected, positions, delta, deepstack = self._inputs(prompt, images)
        tokens = list(prompt)
        generated: list[int] = []
        for _ in range(steps):
            index = np.arange(positions.shape[1], len(tokens))
            full = np.concatenate([positions, np.broadcast_to(index + delta, (3, index.size))], axis=1)
            next_token = int(np.argmax(self._language_logits(tokens, injected, full, deepstack)[-1]))
            generated.append(next_token)
            tokens.append(next_token)
        return generated
