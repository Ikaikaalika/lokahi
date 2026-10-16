"""Write a tiny, randomly initialized Qwen3.5 vision-language checkpoint.

The text model is ``qwen35_tiny``'s, in either layout; the vision tower is
Qwen3-VL's without deepstack, in BF16 as both mlx-community's conversions and
Transformers store it:

* MLX: ``vision_tower.*`` with the patch embedding's Conv3d kernel
  channels-last, [hidden, T, P, P, C];
* Transformers: ``model.visual.*`` with [hidden, C, T, P, P];

plus top-level image token ids, ``vision_config`` and a
``preprocessor_config.json`` (``Qwen2VLImageProcessorFast``'s pixel bounds).
Correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..formats.safetensors import f32_to_bf16, read_safetensors, write_safetensors
from .qwen35_tiny import TinyQwen35Spec, write_tiny_qwen35


# Half of each head rotated, with a low base, so that each M-RoPE axis moves
# the logits well beyond the backends' tolerances.
VISION_TEXT = TinyQwen35Spec(partial_rotary_factor=0.5, mrope_section=(6, 5, 5), rope_theta=100.0)


@dataclass(frozen=True)
class TinyQwen35VisionSpec:
    text: TinyQwen35Spec = VISION_TEXT
    depth: int = 2
    hidden_size: int = 64
    intermediate_size: int = 128
    num_heads: int = 2  # head_dim 32: 2-D RoPE with 4 row and 4 column frequencies
    patch_size: int = 8
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 16  # a 4 x 4 table, interpolated to each image's grid
    image_token_id: int = 500
    vision_start_token_id: int = 498
    vision_end_token_id: int = 499
    min_pixels: int = 1024
    max_pixels: int = 8192
    seed: int = 300

    def vision_config(self) -> dict:
        return {
            "model_type": "qwen3_5",
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": self.intermediate_size,
            "num_heads": self.num_heads,
            "in_channels": 3,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "num_position_embeddings": self.num_position_embeddings,
            "out_hidden_size": self.text.hidden_size,
            "deepstack_visual_indexes": [],
        }

    def processor_config(self) -> dict:
        return {
            "size": {"longest_edge": self.max_pixels, "shortest_edge": self.min_pixels},
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.spatial_merge_size,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
        }


def write_tower(tensors: dict, spec: TinyQwen35VisionSpec, mlx: bool, out_hidden: int,
                deepstack: tuple[int, ...] = ()) -> None:
    """Add the tower's BF16 tensors (and deepstack mergers after the blocks in ``deepstack``)."""

    rng = np.random.default_rng(spec.seed)

    def bf16(name: str, array: np.ndarray) -> None:
        tensors[name] = ("BF16", f32_to_bf16(np.asarray(array, dtype=np.float32)))

    def linear(name: str, rows: int, cols: int) -> None:
        bf16(f"{name}.weight", rng.normal(0.0, cols**-0.5, (rows, cols)))
        bf16(f"{name}.bias", rng.normal(0.0, 0.1, rows))

    def norm(name: str, size: int) -> None:
        bf16(f"{name}.weight", rng.normal(1.0, 0.2, size))
        bf16(f"{name}.bias", rng.normal(0.0, 0.1, size))

    t = "vision_tower" if mlx else "model.visual"
    hidden, inter, p, frames = spec.hidden_size, spec.intermediate_size, spec.patch_size, spec.temporal_patch_size
    kernel = rng.normal(0.0, (3 * frames * p * p) ** -0.5, (hidden, 3, frames, p, p))
    bf16(f"{t}.patch_embed.proj.weight", kernel.transpose(0, 2, 3, 4, 1) if mlx else kernel)
    bf16(f"{t}.patch_embed.proj.bias", rng.normal(0.0, 0.1, hidden))
    bf16(f"{t}.pos_embed.weight", rng.normal(0.0, 0.5, (spec.num_position_embeddings, hidden)))
    for index in range(spec.depth):
        block = f"{t}.blocks.{index}"
        norm(f"{block}.norm1", hidden)
        norm(f"{block}.norm2", hidden)
        linear(f"{block}.attn.qkv", 3 * hidden, hidden)
        linear(f"{block}.attn.proj", hidden, hidden)
        linear(f"{block}.mlp.linear_fc1", inter, hidden)
        linear(f"{block}.mlp.linear_fc2", hidden, inter)
    merged = hidden * spec.spatial_merge_size**2
    norm(f"{t}.merger.norm", hidden)
    linear(f"{t}.merger.linear_fc1", merged, merged)
    linear(f"{t}.merger.linear_fc2", out_hidden, merged)
    for which in range(len(deepstack)):  # these norm the concatenated square
        name = f"{t}.deepstack_merger_list.{which}"
        norm(f"{name}.norm", merged)
        linear(f"{name}.linear_fc1", merged, merged)
        linear(f"{name}.linear_fc2", out_hidden, merged)


def write_tiny_qwen35_vision(directory: Path, spec: TinyQwen35VisionSpec = TinyQwen35VisionSpec()) -> Path:
    """Write ``config.json``, ``preprocessor_config.json`` and ``model.safetensors``."""

    directory = Path(directory)
    if spec.text.model_type == "qwen3_next":
        raise ValueError("Qwen3-Next has no vision tower")
    if max(spec.image_token_id, spec.vision_start_token_id, spec.vision_end_token_id) >= spec.text.vocab_size:
        raise ValueError("the image tokens must be inside the vocabulary")
    write_tiny_qwen35(directory, spec.text)
    tensors, metadata = read_safetensors(directory / "model.safetensors")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config.update(image_token_id=spec.image_token_id, vision_start_token_id=spec.vision_start_token_id,
                  vision_end_token_id=spec.vision_end_token_id, vision_config=spec.vision_config())
    write_tower(tensors, spec, spec.text.mlx, spec.text.hidden_size)

    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "preprocessor_config.json").write_text(json.dumps(spec.processor_config(), indent=2) + "\n",
                                                        encoding="utf-8")
    write_safetensors(directory / "model.safetensors", tensors, metadata=metadata)
    return directory
