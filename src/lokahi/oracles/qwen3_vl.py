"""NumPy reference for Qwen3-VL and Qwen3-VL-MoE.

Qwen3-VL puts Qwen3.5's tower, with deepstack (``qwen3_5_vision``), in
front of a Qwen 3 or Qwen 3 MoE text model (``llama``) whose RoPE is
interleaved M-RoPE. The text configuration's ``model_type`` is
``qwen3_vl_text`` or ``qwen3_vl_moe_text``; its weights live under
``language_model.`` (MLX) or ``model.language_model.`` (Transformers),
and the tower under ``vision_tower.`` or ``model.visual.``.

Correctness evidence only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .gemma3 import load_dense_weights
from .llama import LlamaFamilyConfig, LlamaFamilyReference
from .qwen3_5_vision import Qwen35VisionConfig, Qwen35VisionReference

TEXT_TYPES = {"qwen3_vl": "qwen3", "qwen3_vl_moe": "qwen3_moe"}


def text_config(raw: Mapping[str, Any]) -> LlamaFamilyConfig:
    """The Llama-family configuration of a Qwen3-VL checkpoint's text model."""

    if raw.get("model_type") not in TEXT_TYPES:
        raise NotImplementedError(f"model_type {raw.get('model_type')!r}")
    text = dict(raw["text_config"])
    text["model_type"] = TEXT_TYPES[raw["model_type"]]
    # Transformers reads rope_scaling when present, else rope_parameters, and
    # rope_theta from inside it, else beside it, else 500000.
    scaling = text.get("rope_scaling") or text.get("rope_parameters") or {}
    if not scaling.get("mrope_interleaved", True):
        raise NotImplementedError("chunked (non-interleaved) M-RoPE")
    text["rope_scaling"] = {**scaling, "mrope_section": scaling.get("mrope_section", [24, 20, 20])}
    text["rope_theta"] = scaling.get("rope_theta", text.get("rope_theta", 500_000.0))
    return LlamaFamilyConfig.from_dict(text)


class Qwen3VLReference(Qwen35VisionReference):
    """Float32 Qwen3-VL: the tower with deepstack, then the Qwen 3 (MoE) text model."""

    @classmethod
    def load(cls, model_dir: Path, processor: Mapping[str, Any] | None = None) -> "Qwen3VLReference":
        import json

        model_dir = Path(model_dir)
        raw, weights = load_dense_weights(model_dir)
        if processor is None and (model_dir / "preprocessor_config.json").exists():
            processor = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
        return cls(Qwen35VisionConfig.from_dict(raw, processor), weights, LlamaFamilyReference(text_config(raw), weights))
