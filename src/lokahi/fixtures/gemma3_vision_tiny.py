"""Write a tiny, randomly initialized Gemma 3 vision-language checkpoint.

The layout is mlx-community's ``Gemma3ForConditionalGeneration`` conversion
(e.g. ``gemma-3-4b-it-4bit``): a nested config with ``text_config``,
``vision_config``, ``mm_tokens_per_image`` and ``image_token_index``; the
quantized text model under ``language_model.``; and a BF16 SigLIP tower
(``vision_tower.vision_model.*``, patch convolution in MLX's
``[out, kH, kW, in]`` layout unless ``torch_conv_layout``) plus the projector
(``multi_modal_projector.*``), as ``skip_vision`` conversions keep them. It is
correctness evidence only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..formats.safetensors import f32_to_bf16, read_safetensors, write_safetensors
from .gemma3_tiny import TinyGemma3Spec, write_tiny_gemma3


@dataclass(frozen=True)
class TinyGemma3VisionSpec:
    text: TinyGemma3Spec = field(default_factory=TinyGemma3Spec)
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    image_size: int = 56
    patch_size: int = 14
    mm_tokens_per_image: int = 4
    image_token_index: int = 500
    boi_token_index: int = 498
    eoi_token_index: int = 499
    torch_conv_layout: bool = False
    seed: int = 100

    def config(self) -> dict:
        # The text model's flat config becomes text_config; quantization and
        # the token ids move to the top level, as mlx-community nests them.
        text = replace(self.text, prefix="language_model.").config()
        quant = text.pop("quantization")
        for key in ("quantization_config", "architectures", "eos_token_id", "bos_token_id", "torch_dtype"):
            text.pop(key, None)
        return {
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "boi_token_index": self.boi_token_index,
            "eoi_token_index": self.eoi_token_index,
            "image_token_index": self.image_token_index,
            "mm_tokens_per_image": self.mm_tokens_per_image,
            "eos_token_id": [1],
            "quantization": quant,
            "text_config": text,
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "layer_norm_eps": 1e-6,
                "vision_use_head": False,
                "skip_vision": True,
            },
            "torch_dtype": "bfloat16",
        }


def write_tiny_gemma3_vision(directory: Path, spec: TinyGemma3VisionSpec = TinyGemma3VisionSpec()) -> Path:
    """Write ``config.json``, ``preprocessor_config.json`` and ``model.safetensors``."""

    directory = Path(directory)
    if spec.image_token_index >= spec.text.vocab_size:
        raise ValueError("the image token must be inside the vocabulary")
    side = spec.image_size // spec.patch_size
    tokens_side = int(round(spec.mm_tokens_per_image**0.5))
    if tokens_side**2 != spec.mm_tokens_per_image or side % tokens_side:
        raise ValueError("mm_tokens_per_image must be a square that divides the patch grid")
    write_tiny_gemma3(directory, replace(spec.text, prefix="language_model."))
    tensors, metadata = read_safetensors(directory / "model.safetensors")
    rng = np.random.default_rng(spec.seed)

    def bf16(name: str, array: np.ndarray) -> None:
        tensors[name] = ("BF16", f32_to_bf16(np.asarray(array, dtype=np.float32)))

    hidden, inter = spec.hidden_size, spec.intermediate_size
    p = "vision_tower.vision_model"

    def linear(name: str, rows: int, cols: int, std: float) -> None:
        bf16(f"{name}.weight", rng.normal(0.0, std, (rows, cols)))
        bf16(f"{name}.bias", rng.normal(0.0, 0.1, rows))

    def layer_norm(name: str) -> None:
        bf16(f"{name}.weight", 1.0 + rng.normal(0.0, 0.1, hidden))
        bf16(f"{name}.bias", rng.normal(0.0, 0.1, hidden))

    patch = spec.patch_size
    conv = rng.normal(0.0, (patch * patch * 3) ** -0.5, (hidden, patch, patch, 3))  # MLX [out, kH, kW, in]
    if spec.torch_conv_layout:
        conv = conv.transpose(0, 3, 1, 2)  # Transformers' [out, in, kH, kW]
    bf16(f"{p}.embeddings.patch_embedding.weight", conv)
    bf16(f"{p}.embeddings.patch_embedding.bias", rng.normal(0.0, 0.1, hidden))
    bf16(f"{p}.embeddings.position_embedding.weight", rng.normal(0.0, 0.5, (side * side, hidden)))
    for index in range(spec.num_hidden_layers):
        layer = f"{p}.encoder.layers.{index}"
        layer_norm(f"{layer}.layer_norm1")
        layer_norm(f"{layer}.layer_norm2")
        # Wider than hidden**-0.5 so attention is far from uniform.
        for proj in ("q_proj", "k_proj", "v_proj"):
            linear(f"{layer}.self_attn.{proj}", hidden, hidden, 2.0 * hidden**-0.5)
        linear(f"{layer}.self_attn.out_proj", hidden, hidden, hidden**-0.5)
        linear(f"{layer}.mlp.fc1", inter, hidden, hidden**-0.5)
        linear(f"{layer}.mlp.fc2", hidden, inter, inter**-0.5)
    layer_norm(f"{p}.post_layernorm")
    bf16("multi_modal_projector.mm_soft_emb_norm.weight", rng.normal(0.0, 0.2, hidden))
    # Features about as large as the scaled text embeddings around them.
    bf16("multi_modal_projector.mm_input_projection_weight",
         rng.normal(0.0, 0.75, (hidden, spec.text.hidden_size)))

    config = spec.config()
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    preprocessor = {
        "image_processor_type": "Gemma3ImageProcessor",
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"height": spec.image_size, "width": spec.image_size},
        "image_seq_length": spec.mm_tokens_per_image,
    }
    (directory / "preprocessor_config.json").write_text(json.dumps(preprocessor, indent=2) + "\n", encoding="utf-8")
    metadata = dict(metadata)
    metadata["lokahi_fixture"] = json.dumps(asdict(spec), sort_keys=True)
    write_safetensors(directory / "model.safetensors", tensors, metadata=metadata)
    return directory


def image_prompt(spec: TinyGemma3VisionSpec, text: list[list[int]]) -> list[int]:
    """Text runs joined by images as Gemma 3's processor lays them out:
    begin-of-image, the image tokens, end-of-image."""

    tokens = list(text[0])
    for run in text[1:]:
        tokens += [spec.boi_token_index] + [spec.image_token_index] * spec.mm_tokens_per_image
        tokens += [spec.eoi_token_index] + list(run)
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--torch-conv-layout", action="store_true")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    spec = TinyGemma3VisionSpec(torch_conv_layout=args.torch_conv_layout, seed=args.seed)
    print(write_tiny_gemma3_vision(args.output, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
