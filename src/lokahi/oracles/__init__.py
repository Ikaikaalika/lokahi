"""Independent NumPy reference implementations used as correctness oracles.

Oracles favor clarity over speed: no KV cache, full recomputation, float32
everywhere. Native engines are checked against them; they are never used on
the serving path.
"""

from .gemma3 import Gemma3Config, Gemma3Reference
from .llama import LlamaFamilyConfig, LlamaFamilyReference
from .sampling import SamplingParams

__all__ = ["Gemma3Config", "Gemma3Reference", "LlamaFamilyConfig", "LlamaFamilyReference", "SamplingParams"]
