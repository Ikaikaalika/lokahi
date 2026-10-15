"""NumPy reference for Gemma 3's vision path: the SigLIP tower and the projector.

Semantics follow Transformers' ``SiglipVisionModel`` and
``Gemma3MultiModalProjector`` (as MLX-VLM converts them for mlx-community):

* pixels ``[3, H, W]`` are cut into ``patch_size`` squares; each patch,
  flattened as (row, column, channel) to match MLX's ``[out, kH, kW, in]``
  convolution weights, is projected with a bias, and a learned position
  embedding is added;
* each encoder layer is pre-LayerNorm: ``x += attn(ln1(x))`` (bidirectional
  multi-head attention with q/k/v/o biases, scale ``head_dim ** -0.5``), then
  ``x += fc2(gelu_tanh(fc1(ln2(x))))``; a final LayerNorm follows;
* the projector average-pools the patch grid down to
  ``mm_tokens_per_image`` tokens, applies Gemma's RMSNorm (``1 + weight``) and
  multiplies by ``mm_input_projection_weight`` ``[vision_hidden, text_hidden]``.

The features replace the image tokens' embeddings unscaled (the text
embeddings around them keep Gemma's embedding scale), and each image's tokens
attend to each other in both directions during prefill (see
``Gemma3Reference``). MLX-VLM's Gemma 3 departs from Transformers here: it
divides the features by the square root of the top-level ``hidden_size``
(2048 by default) before the language model scales every row, and it builds
no image-block mask; this oracle follows Transformers. Pixels are expected
already resized; ``preprocess`` applies the processor's rescale and
normalization. Correctness evidence only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gemma3 import Gemma3Reference, load_dense_weights


@dataclass(frozen=True)
class SiglipConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    tokens_per_image: int
    image_token_index: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SiglipConfig":
        vision = raw["vision_config"]
        return cls(
            hidden_size=int(vision["hidden_size"]),
            intermediate_size=int(vision["intermediate_size"]),
            num_hidden_layers=int(vision["num_hidden_layers"]),
            num_attention_heads=int(vision["num_attention_heads"]),
            image_size=int(vision["image_size"]),
            patch_size=int(vision["patch_size"]),
            layer_norm_eps=float(vision.get("layer_norm_eps", 1e-6)),
            tokens_per_image=int(raw.get("mm_tokens_per_image", 256)),
            image_token_index=int(raw["image_token_index"]),
        )


def preprocess(rgb: np.ndarray, mean: Sequence[float] = (0.5, 0.5, 0.5), std: Sequence[float] = (0.5, 0.5, 0.5)):
    """``[H, W, 3]`` uint8 to normalized float32 ``[3, H, W]``."""

    x = np.asarray(rgb, dtype=np.float32) * np.float32(1.0 / 255.0)
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x64 - mean) / np.sqrt(var + eps)).astype(np.float32) * weight + bias


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class Gemma3VisionReference:
    """Float32 SigLIP tower, projector and Gemma 3 language model."""

    PREFIX = "vision_tower.vision_model"

    def __init__(self, config: SiglipConfig, weights: Mapping[str, np.ndarray], language: Gemma3Reference):
        self.config = config
        self.w = dict(weights)
        self.language = language

    @classmethod
    def load(cls, model_dir: Path) -> "Gemma3VisionReference":
        raw, weights = load_dense_weights(model_dir)
        return cls(SiglipConfig.from_dict(raw), weights, Gemma3Reference.load(model_dir))

    # Names as load_dense_weights returns them: ".weight" stripped, ".bias" kept.
    def _linear(self, name: str, x: np.ndarray) -> np.ndarray:
        return x @ self.w[name].T + self.w[f"{name}.bias"]

    def _layer_norm(self, name: str, x: np.ndarray) -> np.ndarray:
        return _layer_norm(x, self.w[name], self.w[f"{name}.bias"], self.config.layer_norm_eps)

    def encode(self, pixels: np.ndarray) -> np.ndarray:
        """Normalized pixels ``[3, H, W]`` to ``[tokens_per_image, text_hidden]`` features."""

        cfg, p = self.config, self.PREFIX
        size, patch = cfg.image_size, cfg.patch_size
        side = size // patch
        grid = pixels.reshape(3, side, patch, side, patch).transpose(1, 3, 2, 4, 0)  # [gy, gx, ky, kx, c]
        patches = grid.reshape(side * side, patch * patch * 3)
        conv = self.w[f"{p}.embeddings.patch_embedding"]
        if conv.shape[1:] == (3, patch, patch) and patch != 3:
            conv = conv.transpose(0, 2, 3, 1)  # Transformers' [out, in, kH, kW] to MLX's
        conv = conv.reshape(cfg.hidden_size, -1)
        x = patches @ conv.T + self.w[f"{p}.embeddings.patch_embedding.bias"]
        x = (x + self.w[f"{p}.embeddings.position_embedding"]).astype(np.float32)
        heads = cfg.num_attention_heads
        dim = cfg.hidden_size // heads
        for index in range(cfg.num_hidden_layers):
            layer = f"{p}.encoder.layers.{index}"
            h = self._layer_norm(f"{layer}.layer_norm1", x)
            q = self._linear(f"{layer}.self_attn.q_proj", h).reshape(-1, heads, dim)
            k = self._linear(f"{layer}.self_attn.k_proj", h).reshape(-1, heads, dim)
            v = self._linear(f"{layer}.self_attn.v_proj", h).reshape(-1, heads, dim)
            out = np.empty_like(q)
            for head in range(heads):
                scores = (q[:, head, :] @ k[:, head, :].T).astype(np.float64) * dim**-0.5
                scores -= scores.max(axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=-1, keepdims=True)
                out[:, head, :] = (probs @ v[:, head, :]).astype(np.float32)
            x = x + self._linear(f"{layer}.self_attn.out_proj", out.reshape(-1, cfg.hidden_size))
            h = self._layer_norm(f"{layer}.layer_norm2", x)
            x = x + self._linear(f"{layer}.mlp.fc2", _gelu_tanh(self._linear(f"{layer}.mlp.fc1", h)))
        x = self._layer_norm(f"{p}.post_layernorm", x)
        # Average-pool the patch grid to tokens_per_image tokens (row-major).
        tokens_side = int(round(math.sqrt(cfg.tokens_per_image)))
        kernel = side // tokens_side
        pooled = x.reshape(tokens_side, kernel, tokens_side, kernel, cfg.hidden_size).mean(axis=(1, 3))
        pooled = pooled.reshape(cfg.tokens_per_image, cfg.hidden_size).astype(np.float32)
        variance = np.mean(pooled.astype(np.float64) ** 2, axis=-1, keepdims=True)
        normed = (pooled / np.sqrt(variance + cfg.layer_norm_eps)).astype(np.float32) * (
            1.0 + self.w["multi_modal_projector.mm_soft_emb_norm"])
        return (normed @ self.w["multi_modal_projector.mm_input_projection_weight"]).astype(np.float32)

    def placement(self, token_ids: Sequence[int], images: Sequence[np.ndarray]):
        """The injected rows and bidirectional blocks for images at their token runs."""

        cfg = self.config
        positions = [i for i, t in enumerate(token_ids) if t == cfg.image_token_index]
        if len(positions) != cfg.tokens_per_image * len(images):
            raise ValueError("image token count does not match the images")
        features = [self.encode(image) for image in images]
        injected, blocks = {}, []
        for n, image in enumerate(features):
            run = positions[n * cfg.tokens_per_image:(n + 1) * cfg.tokens_per_image]
            if run != list(range(run[0], run[0] + cfg.tokens_per_image)):
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
