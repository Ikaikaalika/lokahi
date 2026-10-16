"""Write a tiny, randomly initialized Qwen3-VL or Qwen3-VL-MoE checkpoint.

The text model is ``dense_tiny``'s Qwen 3 or Qwen 3 MoE with interleaved
M-RoPE (``rope_scaling.mrope_section``), as Qwen3-VL's ``text_config``
describes it; the tower is ``qwen35_vision_tiny``'s with deepstack mergers.
Two layouts:

* MLX (mlx-community's conversions): the affine-quantized text model under
  ``language_model.``, the BF16 tower under ``vision_tower.`` with a
  channels-last Conv3d kernel;
* Transformers (Qwen3-VL only): BF16 throughout, the text model under
  ``model.language_model.`` and ``lm_head``, the tower under
  ``model.visual.``. (Transformers-layout Qwen3-VL-MoE checkpoints store
  fused, transposed experts, which this fixture does not write.)

Correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec, dequantize_affine
from ..formats.safetensors import bf16_to_f32, f32_to_bf16, read_safetensors, write_safetensors
from .dense_tiny import TinyDenseSpec, write_tiny_dense
from .qwen35_vision_tiny import TinyQwen35VisionSpec, write_tower

# A low RoPE base, so each M-RoPE axis moves the logits well beyond the
# tolerances, and an untied head, so greedy continuations vary.
VL_TEXT = TinyDenseSpec(model_type="qwen3", vocab_size=512, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                        rope_theta=100.0, tie_word_embeddings=False, seed=0)
VL_MOE_TEXT = TinyDenseSpec(model_type="qwen3_moe", vocab_size=512, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                            rope_theta=100.0, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
                            tie_word_embeddings=False, seed=1)


@dataclass(frozen=True)
class TinyQwen3VLSpec:
    text: TinyDenseSpec = VL_TEXT
    mlx: bool = True
    # The tower's geometry (its text field is unused); deepstack features
    # after tower blocks 0 and 2 go to the text model's layers 0 and 1.
    tower: TinyQwen35VisionSpec = field(default_factory=lambda: TinyQwen35VisionSpec(depth=3, seed=310))
    deepstack_visual_indexes: tuple[int, ...] = (0, 2)
    mrope_section: tuple[int, ...] = (6, 5, 5)  # head_dim 32: 16 frequencies, 5 on rows and 5 on columns

    @property
    def model_type(self) -> str:
        return {"qwen3": "qwen3_vl", "qwen3_moe": "qwen3_vl_moe"}[self.text.model_type]

    def vision_config(self) -> dict:
        t = self.tower
        return {
            "model_type": self.model_type,
            "depth": t.depth,
            "hidden_size": t.hidden_size,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": t.intermediate_size,
            "num_heads": t.num_heads,
            "in_channels": 3,
            "patch_size": t.patch_size,
            "temporal_patch_size": t.temporal_patch_size,
            "spatial_merge_size": t.spatial_merge_size,
            "num_position_embeddings": t.num_position_embeddings,
            "out_hidden_size": self.text.hidden_size,
            "deepstack_visual_indexes": list(self.deepstack_visual_indexes),
        }


def _rename(name: str, mlx: bool) -> str:
    if mlx:
        return "language_model." + name
    return "model.language_model." + name[len("model."):] if name.startswith("model.") else name


def write_tiny_qwen3_vl(directory: Path, spec: TinyQwen3VLSpec = TinyQwen3VLSpec()) -> Path:
    """Write ``config.json``, ``preprocessor_config.json`` and ``model.safetensors``."""

    directory = Path(directory)
    if spec.text.model_type not in ("qwen3", "qwen3_moe"):
        raise ValueError("Qwen3-VL's text model is Qwen 3 or Qwen 3 MoE")
    if not spec.mlx and spec.text.model_type == "qwen3_moe":
        raise ValueError("the Transformers layout of Qwen3-VL-MoE stores fused, transposed experts")
    t = spec.tower
    if max(t.image_token_id, t.vision_start_token_id, t.vision_end_token_id) >= spec.text.vocab_size:
        raise ValueError("the image tokens must be inside the vocabulary")
    write_tiny_dense(directory, spec.text)
    text = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    source, _ = read_safetensors(directory / "model.safetensors")

    tensors: dict[str, tuple[str, np.ndarray]] = {}
    quant = text.pop("quantization")
    text.pop("quantization_config")
    for name, (dtype, array) in source.items():
        if spec.mlx:
            tensors[_rename(name, True)] = (dtype, array)
        elif name.endswith(".scales") or name.endswith(".biases"):
            continue
        elif name.endswith(".weight") and dtype == "U32":  # dequantize to BF16
            module = name[: -len(".weight")]
            override = quant.get(module)
            source_spec = override if isinstance(override, dict) else quant
            qspec = QuantSpec(group_size=source_spec["group_size"], bits=source_spec["bits"])
            scales = source[f"{module}.scales"]
            biases = source[f"{module}.biases"]
            dense = dequantize_affine(array, bf16_to_f32(scales[1]), bf16_to_f32(biases[1]), qspec)
            tensors[_rename(name, False)] = ("BF16", f32_to_bf16(dense.astype(np.float32)))
        else:
            tensors[_rename(name, False)] = (dtype, array)
    write_tower(tensors, t, spec.mlx, spec.text.hidden_size, spec.deepstack_visual_indexes)

    for key in ("architectures", "quantization", "quantization_config", "torch_dtype"):
        text.pop(key, None)
    text["model_type"] = spec.model_type + "_text"
    text["rope_scaling"] = {"rope_type": "default", "mrope_section": list(spec.mrope_section),
                            "mrope_interleaved": True}
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration" if spec.model_type == "qwen3_vl"
                          else "Qwen3VLMoeForConditionalGeneration"],
        "model_type": spec.model_type,
        "text_config": text,
        "vision_config": spec.vision_config(),
        "image_token_id": t.image_token_id,
        "video_token_id": t.image_token_id + 1,
        "vision_start_token_id": t.vision_start_token_id,
        "vision_end_token_id": t.vision_end_token_id,
        "tie_word_embeddings": spec.text.tie_word_embeddings,
    }
    if spec.mlx:
        prefixed = {key if not isinstance(value, dict) else "language_model." + key: value
                    for key, value in quant.items()}
        config["quantization"] = prefixed
        config["quantization_config"] = prefixed
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "preprocessor_config.json").write_text(json.dumps(t.processor_config(), indent=2) + "\n",
                                                        encoding="utf-8")
    write_safetensors(directory / "model.safetensors", tensors,
                      metadata={"format": "mlx" if spec.mlx else "pt",
                                "lokahi_fixture": json.dumps(asdict(spec), sort_keys=True, default=str)})
    return directory
