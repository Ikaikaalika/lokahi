"""MLX affine quantization layout, implemented independently in NumPy.

MLX stores a quantized ``[out, in]`` matrix as three tensors:

* ``weight``: ``uint32`` ``[out, in * bits / 32]``; value ``k`` of a row sits in
  word ``k // (32 / bits)`` at bit offset ``bits * (k % (32 / bits))``;
* ``scales`` and ``biases``: ``[out, in / group_size]`` in the model dtype.

Dequantization is ``w = scales * q + biases`` per group. Only the 2-, 4- and
8-bit layouts (which divide 32 evenly) are supported here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SUPPORTED_BITS = (2, 4, 8)


@dataclass(frozen=True)
class QuantSpec:
    group_size: int
    bits: int
    mode: str = "affine"

    def __post_init__(self) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(f"unsupported affine bit width {self.bits}")
        if self.group_size <= 0 or self.group_size % (32 // self.bits):
            raise ValueError(f"invalid group size {self.group_size}")
        if self.mode != "affine":
            raise ValueError(f"unsupported quantization mode {self.mode!r}")


def quant_spec_for(config: Mapping[str, Any], module: str) -> QuantSpec | None:
    """Resolve the quantization of ``module`` (e.g. ``model.layers.0.mlp.up_proj``).

    MLX configs carry a default ``group_size``/``bits`` plus optional
    per-module overrides keyed by module path; an override of ``false`` marks
    an unquantized module.
    """

    quant = config.get("quantization")
    if not quant:
        return None
    override = quant.get(module)
    if override is False:
        return None
    source = override if isinstance(override, Mapping) else quant
    return QuantSpec(
        group_size=int(source.get("group_size", quant["group_size"])),
        bits=int(source.get("bits", quant["bits"])),
        mode=str(source.get("mode", quant.get("mode", "affine"))),
    )


def pack_affine(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes ``[rows, cols]`` into MLX ``uint32`` words."""

    q = np.asarray(q, dtype=np.uint32)
    per_word = 32 // bits
    rows, cols = q.shape
    if cols % per_word:
        raise ValueError("columns must fill whole words")
    grouped = q.reshape(rows, cols // per_word, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(1, 1, per_word)
    return np.bitwise_or.reduce(grouped << shifts, axis=2).astype(np.uint32)


def unpack_affine(words: np.ndarray, bits: int) -> np.ndarray:
    """Unpack MLX ``uint32`` words ``[rows, cols*bits/32]`` to codes ``[rows, cols]``."""

    words = np.asarray(words, dtype=np.uint32)
    per_word = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(1, 1, per_word)
    codes = (words[:, :, None] >> shifts) & mask
    return codes.reshape(words.shape[0], words.shape[1] * per_word)


def dequantize_affine(
    words: np.ndarray, scales: np.ndarray, biases: np.ndarray, spec: QuantSpec
) -> np.ndarray:
    """Return the ``float32`` matrix represented by an MLX affine triple.

    Leading dimensions (stacked experts, ``[experts, rows, ...]``) are kept.
    """

    words = np.asarray(words, dtype=np.uint32)
    *lead, rows, _ = words.shape
    codes = unpack_affine(words.reshape(-1, words.shape[-1]), spec.bits).astype(np.float32)
    cols = codes.shape[1]
    groups = cols // spec.group_size
    if scales.shape != (*lead, rows, groups) or biases.shape != (*lead, rows, groups):
        raise ValueError("scales/biases shape does not match weight")
    grouped = codes.reshape(-1, groups, spec.group_size)
    scale = np.asarray(scales, dtype=np.float32).reshape(-1, groups, 1)
    bias = np.asarray(biases, dtype=np.float32).reshape(-1, groups, 1)
    return (grouped * scale + bias).reshape(*lead, rows, cols)
