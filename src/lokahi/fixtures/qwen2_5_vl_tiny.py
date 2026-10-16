"""Write a tiny, randomly initialized Qwen2.5-VL checkpoint.

The text model is ``dense_tiny``'s Qwen 2 (q/k/v biases) with chunked M-RoPE;
the tower is Qwen2.5-VL's window-attention ViT with RMSNorms, SwiGLU MLPs
and an ``ln_q`` merger, in BF16. Three layouts:

* ``mlx`` (mlx-community's conversions): a flat ``config.json`` (the text
  fields at the top level), the affine-quantized text model under
  ``language_model.``, the tower under ``vision_tower.`` with a
  channels-last Conv3d kernel;
* ``hf`` (Qwen's published checkpoints, saved by Transformers 4): a flat
  ``config.json`` with ``rope_scaling: {"type": "mrope"}`` and ``in_chans``,
  BF16 throughout, the text model under ``model.`` and ``lm_head``, the tower
  under ``visual.``;
* ``hf5`` (Transformers 5's ``save_pretrained``): ``text_config`` with
  ``rope_parameters`` and ``layer_types``, the tower's axial
  ``rope_parameters``, the text model under ``model.language_model.``, the
  tower under ``model.visual.``.

Correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.safetensors import f32_to_bf16, read_safetensors, write_safetensors
from .dense_tiny import TinyDenseSpec, write_tiny_dense
from .qwen3_vl_tiny import text_tensors

# A low RoPE base, so each M-RoPE axis moves the logits well beyond the
# tolerances, an untied head, so greedy continuations vary, and heads of 64
# (hidden_size / num_attention_heads), the smallest the Metal decoder takes.
VL25_TEXT = TinyDenseSpec(model_type="qwen2", vocab_size=512, hidden_size=128, intermediate_size=256,
                          num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, rope_theta=100.0,
                          tie_word_embeddings=False, seed=2)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
LAYOUTS = ("mlx", "hf", "hf5")


@dataclass(frozen=True)
class TinyQwen25VisionSpec:
    depth: int = 3
    hidden_size: int = 64
    intermediate_size: int = 96
    num_heads: int = 2  # head_dim 32: 2-D RoPE with 8 row and 8 column frequencies
    patch_size: int = 8
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    window_size: int = 32  # windows of 2 x 2 merged tokens
    fullatt_block_indexes: tuple[int, ...] = (1,)  # blocks 0 and 2 attend within windows
    image_token_id: int = 500
    vision_start_token_id: int = 498
    vision_end_token_id: int = 499
    min_pixels: int = 1024
    max_pixels: int = 8192
    seed: int = 320


@dataclass(frozen=True)
class TinyQwen25VLSpec:
    text: TinyDenseSpec = VL25_TEXT
    tower: TinyQwen25VisionSpec = field(default_factory=TinyQwen25VisionSpec)
    layout: str = "mlx"
    # 32 frequencies, as Qwen2.5-VL's [16, 24, 24] is for 64: 8 temporal, then
    # 12 on rows and 12 on columns.
    mrope_section: tuple[int, ...] = (8, 12, 12)

    def vision_config(self) -> dict:
        t = self.tower
        config = {
            "depth": t.depth,
            "hidden_act": "silu",
            "hidden_size": t.hidden_size,
            "intermediate_size": t.intermediate_size,
            "num_heads": t.num_heads,
            "out_hidden_size": self.text.hidden_size,
            "patch_size": t.patch_size,
            "spatial_merge_size": t.spatial_merge_size,
            "spatial_patch_size": t.patch_size,
            "window_size": t.window_size,
            "fullatt_block_indexes": list(t.fullatt_block_indexes),
            "tokens_per_second": 2,
            "temporal_patch_size": t.temporal_patch_size,
        }
        if self.layout == "hf5":
            config.update(model_type="qwen2_5_vl_vision", in_channels=3,
                          rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"})
        else:
            config["in_chans"] = 3
        return config

    def processor_config(self) -> dict:
        t = self.tower
        return {
            "min_pixels": t.min_pixels,
            "max_pixels": t.max_pixels,
            "patch_size": t.patch_size,
            "temporal_patch_size": t.temporal_patch_size,
            "merge_size": t.spatial_merge_size,
            "image_mean": CLIP_MEAN,
            "image_std": CLIP_STD,
            "image_processor_type": "Qwen2VLImageProcessor",
            "processor_class": "Qwen2_5_VLProcessor",
        }


def write_tower(tensors: dict, spec: TinyQwen25VLSpec) -> None:
    """Add the tower's BF16 tensors."""

    t, out_hidden = spec.tower, spec.text.hidden_size
    rng = np.random.default_rng(t.seed)

    def bf16(name: str, array: np.ndarray) -> None:
        tensors[name] = ("BF16", f32_to_bf16(np.asarray(array, dtype=np.float32)))

    def linear(name: str, rows: int, cols: int) -> None:
        bf16(f"{name}.weight", rng.normal(0.0, cols**-0.5, (rows, cols)))
        bf16(f"{name}.bias", rng.normal(0.0, 0.1, rows))

    prefix = {"mlx": "vision_tower", "hf": "visual", "hf5": "model.visual"}[spec.layout]
    hidden, inter, p, frames = t.hidden_size, t.intermediate_size, t.patch_size, t.temporal_patch_size
    kernel = rng.normal(0.0, (3 * frames * p * p) ** -0.5, (hidden, 3, frames, p, p))
    bf16(f"{prefix}.patch_embed.proj.weight", kernel.transpose(0, 2, 3, 4, 1) if spec.layout == "mlx" else kernel)
    for index in range(t.depth):
        block = f"{prefix}.blocks.{index}"
        bf16(f"{block}.norm1.weight", rng.normal(1.0, 0.2, hidden))
        bf16(f"{block}.norm2.weight", rng.normal(1.0, 0.2, hidden))
        linear(f"{block}.attn.qkv", 3 * hidden, hidden)
        linear(f"{block}.attn.proj", hidden, hidden)
        linear(f"{block}.mlp.gate_proj", inter, hidden)
        linear(f"{block}.mlp.up_proj", inter, hidden)
        linear(f"{block}.mlp.down_proj", hidden, inter)
    merged = hidden * t.spatial_merge_size**2
    bf16(f"{prefix}.merger.ln_q.weight", rng.normal(1.0, 0.2, hidden))
    linear(f"{prefix}.merger.mlp.0", merged, merged)
    linear(f"{prefix}.merger.mlp.2", out_hidden, merged)


def _rename(name: str, layout: str) -> str:
    if layout == "mlx":
        return "language_model." + name
    if layout == "hf5" and name.startswith("model."):
        return "model.language_model." + name[len("model."):]
    return name


def write_tiny_qwen2_5_vl(directory: Path, spec: TinyQwen25VLSpec = TinyQwen25VLSpec()) -> Path:
    """Write ``config.json``, ``preprocessor_config.json`` and ``model.safetensors``."""

    directory = Path(directory)
    if spec.text.model_type != "qwen2" or spec.text.head_dim is not None:
        raise ValueError("Qwen2.5-VL's text model is Qwen 2, with hidden_size / num_attention_heads per head")
    if spec.layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    t = spec.tower
    if max(t.image_token_id, t.vision_start_token_id, t.vision_end_token_id) >= spec.text.vocab_size:
        raise ValueError("the image tokens must be inside the vocabulary")
    write_tiny_dense(directory, spec.text)
    text = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    source, _ = read_safetensors(directory / "model.safetensors")
    quant = text.pop("quantization")
    text.pop("quantization_config")
    mlx = spec.layout == "mlx"
    tensors = text_tensors(source, quant, mlx, lambda name: _rename(name, spec.layout))
    write_tower(tensors, spec)

    for key in ("architectures", "torch_dtype"):
        text.pop(key, None)
    ids = {"image_token_id": t.image_token_id, "video_token_id": t.image_token_id + 1,
           "vision_start_token_id": t.vision_start_token_id, "vision_end_token_id": t.vision_end_token_id}
    section = list(spec.mrope_section)
    if spec.layout == "hf5":
        text["model_type"] = "qwen2_5_vl_text"
        text["rope_parameters"] = {"type": "mrope", "rope_type": "default", "mrope_section": section,
                                   "rope_theta": text.pop("rope_theta")}
        text["layer_types"] = ["full_attention"] * spec.text.num_hidden_layers
        config = {"architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl",
                  "text_config": text, "vision_config": spec.vision_config(),
                  "tie_word_embeddings": spec.text.tie_word_embeddings, **ids}
    else:
        config = {**text, "architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl",
                  "rope_scaling": {"type": "mrope", "mrope_section": section}, "vision_config": spec.vision_config(),
                  **ids}
    if mlx:
        prefixed = {key if not isinstance(value, dict) else "language_model." + key: value
                    for key, value in quant.items()}
        config["quantization"] = prefixed
        config["quantization_config"] = prefixed
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "preprocessor_config.json").write_text(json.dumps(spec.processor_config(), indent=2) + "\n",
                                                        encoding="utf-8")
    write_safetensors(directory / "model.safetensors", tensors,
                      metadata={"format": "mlx" if mlx else "pt",
                                "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True, default=str)})
    return directory
