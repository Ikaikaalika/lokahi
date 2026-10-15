"""Deterministic generated model fixtures (no downloads, no real weights)."""

from .chat_model import write_tiny_chat_model
from .dense_tiny import TinyDenseSpec, write_tiny_dense
from .gemma3_tiny import TinyGemma3Spec, write_tiny_gemma3

__all__ = ["TinyDenseSpec", "TinyGemma3Spec", "write_tiny_chat_model", "write_tiny_dense", "write_tiny_gemma3"]
