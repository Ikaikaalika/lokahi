"""Write a tiny, randomly initialized Gemma 4 vision-language checkpoint.

The layout is mlx-community's ``Gemma4ForConditionalGeneration`` conversion
(e.g. ``gemma-4-31b-it-4bit``):

* the quantized text model under ``language_model.`` with ``text_config``
  (31B's attention: ``attention_k_eq_v``, no per-layer inputs or KV sharing,
  ``use_bidirectional_attention: "vision"``; untied, unlike 31B, so greedy
  continuations vary instead of echoing the last token);
* a BF16 vision tower (``vision_tower.patch_embedder.*``,
  ``vision_tower.encoder.layers.N.*`` with ``ClippableLinear``'s ``.linear``
  nesting, ``std_bias``/``std_scale``);
* the affine-quantized embedder (``embed_vision.embedding_projection``);
* top-level image token ids and ``vision_soft_tokens_per_image``, plus a
  ``processor_config.json``.

Correctness evidence only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..formats.mlx_quant import QuantSpec
from ..formats.safetensors import f32_to_bf16, read_safetensors, write_safetensors
from .gemma3_tiny import _quantized
from .gemma4_tiny import TinyGemma4Spec, write_tiny_gemma4

TEXT = TinyGemma4Spec(
    num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True, num_kv_shared_layers=0,
    hidden_size_per_layer_input=0, use_double_wide_mlp=False, tie_word_embeddings=False,
    extra={"use_bidirectional_attention": "vision"},
)


@dataclass(frozen=True)
class TinyGemma4VisionSpec:
    text: TinyGemma4Spec = field(default_factory=lambda: TEXT)
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    head_dim: int = 24  # heads * head_dim != hidden_size, as the projections allow
    patch_size: int = 4
    pooling_kernel_size: int = 3
    max_soft_tokens: int = 8
    position_embedding_size: int = 32
    rope_theta: float = 100.0
    standardize: bool = True
    image_token_id: int = 500
    boi_token_id: int = 498
    eoi_token_id: int = 499
    seed: int = 200

    def vision_config(self) -> dict:
        return {
            "model_type": "gemma4_vision",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "global_head_dim": self.head_dim,
            "hidden_activation": "gelu_pytorch_tanh",
            "patch_size": self.patch_size,
            "pooling_kernel_size": self.pooling_kernel_size,
            "default_output_length": self.max_soft_tokens,
            "position_embedding_size": self.position_embedding_size,
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
            "rms_norm_eps": 1e-6,
            "standardize": self.standardize,
            "use_clipped_linears": False,
        }


def write_tiny_gemma4_vision(directory: Path, spec: TinyGemma4VisionSpec = TinyGemma4VisionSpec()) -> Path:
    """Write ``config.json``, ``processor_config.json`` and ``model.safetensors``."""

    directory = Path(directory)
    text = replace(spec.text, nested=True)
    if spec.image_token_id >= text.vocab_size:
        raise ValueError("the image token must be inside the vocabulary")
    if spec.head_dim % 4:
        raise ValueError("axial RoPE needs head_dim divisible by 4")
    write_tiny_gemma4(directory, text)
    tensors, metadata = read_safetensors(directory / "model.safetensors")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(spec.seed)

    def bf16(name: str, array: np.ndarray) -> None:
        tensors[name] = ("BF16", f32_to_bf16(np.asarray(array, dtype=np.float32)))

    hidden, inter, heads, dim = spec.hidden_size, spec.intermediate_size, spec.num_attention_heads, spec.head_dim
    t = "vision_tower"
    patch_width = 3 * spec.patch_size**2
    bf16(f"{t}.patch_embedder.input_proj.weight", rng.normal(0.0, patch_width**-0.5, (hidden, patch_width)))
    bf16(f"{t}.patch_embedder.position_embedding_table",
         rng.normal(0.0, 0.5, (2, spec.position_embedding_size, hidden)))
    for index in range(spec.num_hidden_layers):
        layer = f"{t}.encoder.layers.{index}"
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            bf16(f"{layer}.{name}.weight", rng.normal(1.0, 0.2, hidden))
        bf16(f"{layer}.self_attn.q_norm.weight", rng.normal(1.0, 0.2, dim))
        bf16(f"{layer}.self_attn.k_norm.weight", rng.normal(1.0, 0.2, dim))
        for proj in ("q_proj", "k_proj", "v_proj"):
            bf16(f"{layer}.self_attn.{proj}.linear.weight", rng.normal(0.0, hidden**-0.5, (heads * dim, hidden)))
        bf16(f"{layer}.self_attn.o_proj.linear.weight", rng.normal(0.0, (heads * dim) ** -0.5, (hidden, heads * dim)))
        bf16(f"{layer}.mlp.gate_proj.linear.weight", rng.normal(0.0, hidden**-0.5, (inter, hidden)))
        bf16(f"{layer}.mlp.up_proj.linear.weight", rng.normal(0.0, hidden**-0.5, (inter, hidden)))
        bf16(f"{layer}.mlp.down_proj.linear.weight", rng.normal(0.0, inter**-0.5, (hidden, inter)))
    if spec.standardize:
        bf16(f"{t}.std_bias", rng.normal(0.0, 3.0, hidden))
        bf16(f"{t}.std_scale", rng.uniform(0.5, 1.5, hidden))
    # The embedder's projection is quantized like the text model; features
    # come out about as large as the scaled text embeddings.
    qspec = QuantSpec(group_size=text.group_size, bits=text.bits)
    weight, scales, biases = _quantized(rng, text.hidden_size, hidden, qspec, 0.9)
    tensors["embed_vision.embedding_projection.weight"] = ("U32", weight)
    tensors["embed_vision.embedding_projection.scales"] = ("BF16", scales)
    tensors["embed_vision.embedding_projection.biases"] = ("BF16", biases)

    config.update({
        "image_token_id": spec.image_token_id,
        "boi_token_id": spec.boi_token_id,
        "eoi_token_id": spec.eoi_token_id,
        "vision_soft_tokens_per_image": spec.max_soft_tokens,
        "vision_config": spec.vision_config(),
    })
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    processor = {
        "image_processor": {
            "image_processor_type": "Gemma4ImageProcessor",
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": False,
            "max_soft_tokens": spec.max_soft_tokens,
            "patch_size": spec.patch_size,
            "pooling_kernel_size": spec.pooling_kernel_size,
        },
        "processor_class": "Gemma4Processor",
    }
    (directory / "processor_config.json").write_text(json.dumps(processor, indent=2) + "\n", encoding="utf-8")
    metadata = dict(metadata)
    metadata["lokahi_fixture"] = json.dumps(asdict(spec), sort_keys=True)
    write_safetensors(directory / "model.safetensors", tensors, metadata=metadata)
    return directory


def image_prompt(spec: TinyGemma4VisionSpec, text: list[list[int]], tokens_per_image: list[int]) -> list[int]:
    """Text runs joined by images as Gemma 4's processor lays them out:
    begin-of-image, one image token per soft token, end-of-image."""

    tokens = list(text[0])
    for run, count in zip(text[1:], tokens_per_image):
        tokens += [spec.boi_token_id] + [spec.image_token_id] * count + [spec.eoi_token_id] + list(run)
    return tokens
