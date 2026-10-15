"""Deterministic generated model fixtures (no downloads, no real weights)."""

from .chat_model import write_tiny_chat_model
from .dense_tiny import TinyDenseSpec, write_tiny_dense
from .gemma3_tiny import TinyGemma3Spec, write_tiny_gemma3
from .gemma3_vision_tiny import TinyGemma3VisionSpec, write_tiny_gemma3_vision
from .gemma4_tiny import TinyGemma4Spec, write_tiny_gemma4

__all__ = [
    "TinyDenseSpec",
    "TinyGemma3Spec",
    "TinyGemma3VisionSpec",
    "TinyGemma4Spec",
    "write_tiny_chat_model",
    "write_tiny_dense",
    "write_tiny_gemma3",
    "write_tiny_gemma3_vision",
    "write_tiny_gemma4",
]
