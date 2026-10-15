"""Independent NumPy reference implementations used as correctness oracles.

Oracles favor clarity over speed: no KV cache, full recomputation, float32
everywhere. Native engines are checked against them; they are never used on
the serving path.
"""

from .gemma3 import Gemma3Config, Gemma3Reference
from .gemma4 import Gemma4Config, Gemma4Reference
from .llama import LlamaFamilyConfig, LlamaFamilyReference
from .sampling import SamplingParams
from .siglip import Gemma3VisionReference, SiglipConfig

__all__ = [
    "Gemma3Config",
    "Gemma3Reference",
    "Gemma3VisionReference",
    "Gemma4Config",
    "Gemma4Reference",
    "LlamaFamilyConfig",
    "LlamaFamilyReference",
    "SamplingParams",
    "SiglipConfig",
]
