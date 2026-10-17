"""Deterministic generated model fixtures (no downloads, no real weights)."""

from .chat_model import write_tiny_chat_model
from .deepseek_tiny import TinyDeepseekSpec, write_tiny_deepseek
from .dense_tiny import TinyDenseSpec, write_tiny_dense
from .gemma3_tiny import TinyGemma3Spec, write_tiny_gemma3
from .gemma3_vision_tiny import TinyGemma3VisionSpec, write_tiny_gemma3_vision
from .gemma4_tiny import TinyGemma4Spec, write_tiny_gemma4
from .gemma4_vision_tiny import TinyGemma4VisionSpec, write_tiny_gemma4_vision
from .muse_glimmer_tiny import TinyMuseGlimmerSpec, write_tiny_muse_glimmer
from .qwen2_5_vl_tiny import TinyQwen25VLSpec, write_tiny_qwen2_5_vl
from .qwen35_tiny import TinyQwen35Spec, write_tiny_qwen35, write_tiny_qwen35_mtp_drafter
from .qwen35_vision_tiny import TinyQwen35VisionSpec, write_tiny_qwen35_vision
from .qwen3_vl_tiny import TinyQwen3VLSpec, write_tiny_qwen3_vl

__all__ = [
    "TinyDeepseekSpec",
    "TinyDenseSpec",
    "TinyGemma3Spec",
    "TinyGemma3VisionSpec",
    "TinyGemma4Spec",
    "TinyGemma4VisionSpec",
    "TinyMuseGlimmerSpec",
    "TinyQwen25VLSpec",
    "TinyQwen35Spec",
    "TinyQwen35VisionSpec",
    "TinyQwen3VLSpec",
    "write_tiny_chat_model",
    "write_tiny_deepseek",
    "write_tiny_dense",
    "write_tiny_gemma3",
    "write_tiny_gemma3_vision",
    "write_tiny_gemma4",
    "write_tiny_gemma4_vision",
    "write_tiny_muse_glimmer",
    "write_tiny_qwen2_5_vl",
    "write_tiny_qwen35",
    "write_tiny_qwen35_mtp_drafter",
    "write_tiny_qwen35_vision",
    "write_tiny_qwen3_vl",
]
