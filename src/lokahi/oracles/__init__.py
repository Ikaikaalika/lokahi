"""Independent NumPy reference implementations used as correctness oracles.

Oracles favor clarity over speed: no KV cache, full recomputation, float32
everywhere. Native engines are checked against them; they are never used on
the serving path.
"""

from .deepseek import DeepseekConfig, DeepseekReference
from .gemma3 import Gemma3Config, Gemma3Reference
from .gemma4 import Gemma4Config, Gemma4Reference
from .gemma4_vision import Gemma4VisionConfig, Gemma4VisionReference
from .llama import LlamaFamilyConfig, LlamaFamilyReference
from .qwen3_5 import Qwen35Config, Qwen35Reference
from .qwen3_5_vision import Qwen35VisionConfig, Qwen35VisionReference
from .sampling import SamplingParams
from .siglip import Gemma3VisionReference, SiglipConfig

__all__ = [
    "DeepseekConfig",
    "DeepseekReference",
    "Gemma3Config",
    "Gemma3Reference",
    "Gemma3VisionReference",
    "Gemma4Config",
    "Gemma4Reference",
    "Gemma4VisionConfig",
    "Gemma4VisionReference",
    "LlamaFamilyConfig",
    "LlamaFamilyReference",
    "Qwen35Config",
    "Qwen35Reference",
    "Qwen35VisionConfig",
    "Qwen35VisionReference",
    "SamplingParams",
    "SiglipConfig",
]
