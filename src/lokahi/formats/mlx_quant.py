"""MLX quantization layouts (affine and MXFP4), implemented independently in NumPy.

MLX stores an affine-quantized ``[out, in]`` matrix as three tensors:

* ``weight``: ``uint32`` ``[out, in * bits / 32]``. Each row is one
  little-endian bit stream: value ``k`` occupies bits ``[k * bits, (k + 1) *
  bits)``. For 2, 4 and 8 bits that is word ``k // (32 / bits)`` at bit offset
  ``bits * (k % (32 / bits))``; 3-, 5- and 6-bit values can straddle words;
* ``scales`` and ``biases``: ``[out, in / group_size]`` in the model dtype.

Dequantization is ``w = scales * q + biases`` per group. Groups start on word
boundaries (``group_size * bits`` is a multiple of 32).

MXFP4 (``mode: "mxfp4"``, OCP microscaling) packs 4-bit E2M1 codes the same
way and stores one ``uint8`` E8M0 exponent per 32 values in ``scales``, with no
biases: ``w = E2M1[q] * 2 ** (e - 127)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)


@dataclass(frozen=True)
class QuantSpec:
    group_size: int
    bits: int
    mode: str = "affine"

    def __post_init__(self) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(f"unsupported affine bit width {self.bits}")
        if self.group_size <= 0 or self.group_size * self.bits % 32:
            raise ValueError(f"invalid group size {self.group_size}")
        if self.mode == "mxfp4" and (self.bits, self.group_size) != (4, 32):
            raise ValueError("mxfp4 needs 4 bits and group size 32")
        if self.mode not in ("affine", "mxfp4"):
            raise ValueError(f"unsupported quantization mode {self.mode!r}")


def quant_spec_for(config: Mapping[str, Any], module: str) -> QuantSpec | None:
    """Resolve the quantization of ``module`` (e.g. ``model.layers.0.mlp.up_proj``).

    MLX configs carry a default ``group_size``/``bits``/``mode`` plus optional
    per-module overrides keyed by module path; an override of ``false`` marks
    an unquantized module, and keys an override leaves out take their mode's
    defaults (affine: 64 and 4 bits; mxfp4: 32 and 4 bits).
    """

    quant = config.get("quantization")
    if not quant:
        return None
    override = quant.get(module)
    if override is False:
        return None
    if isinstance(override, Mapping):
        # As mlx.nn.quantize passes it to to_quantized: missing keys take that
        # mode's defaults, not the model-wide ones.
        mode = str(override.get("mode", "affine"))
        return QuantSpec(
            group_size=int(override.get("group_size", 32 if mode == "mxfp4" else 64)),
            bits=int(override.get("bits", 4)),
            mode=mode,
        )
    return QuantSpec(
        group_size=int(quant["group_size"]),
        bits=int(quant["bits"]),
        mode=str(quant.get("mode", "affine")),
    )


def pack_affine(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes ``[rows, cols]`` into MLX ``uint32`` words."""

    q = np.asarray(q, dtype=np.uint32)
    rows, cols = q.shape
    if cols * bits % 32:
        raise ValueError("columns must fill whole words")
    if 32 % bits:  # one little-endian bit stream per row
        planes = ((q[:, :, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
        packed = np.packbits(planes.reshape(rows, cols * bits), axis=1, bitorder="little")
        return np.ascontiguousarray(packed).view("<u4").astype(np.uint32)
    per_word = 32 // bits
    grouped = q.reshape(rows, cols // per_word, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(1, 1, per_word)
    return np.bitwise_or.reduce(grouped << shifts, axis=2).astype(np.uint32)


def unpack_affine(words: np.ndarray, bits: int) -> np.ndarray:
    """Unpack MLX ``uint32`` words ``[rows, cols*bits/32]`` to codes ``[rows, cols]``."""

    words = np.asarray(words, dtype=np.uint32)
    rows, count = words.shape
    if 32 % bits:
        if count * 32 % bits:
            raise ValueError("words do not hold a whole number of codes")
        stream = np.unpackbits(np.ascontiguousarray(words, dtype="<u4").view(np.uint8), axis=1, bitorder="little")
        planes = stream.reshape(rows, count * 32 // bits, bits).astype(np.uint32)
        return np.bitwise_or.reduce(planes << np.arange(bits, dtype=np.uint32), axis=2).astype(np.uint32)
    per_word = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(1, 1, per_word)
    codes = (words[:, :, None] >> shifts) & mask
    return codes.reshape(rows, count * per_word)


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


# E2M1 values by code, in MLX's order.
FP4_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=np.float32)


def e8m0_to_f32(exponents: np.ndarray) -> np.ndarray:
    """E8M0 scales as float32: ``2 ** (e - 127)`` (``e = 0`` is the subnormal ``2 ** -127``)."""

    return np.ldexp(np.float32(1.0), np.asarray(exponents, dtype=np.int32) - 127).astype(np.float32)


def dequantize_mxfp4(words: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Return the ``float32`` matrix of an MXFP4 pair; leading (expert) dimensions are kept."""

    words = np.asarray(words, dtype=np.uint32)
    *lead, rows, _ = words.shape
    codes = unpack_affine(words.reshape(-1, words.shape[-1]), 4)
    cols = codes.shape[1]
    if np.shape(exponents) != (*lead, rows, cols // 32):
        raise ValueError("mxfp4 scales shape does not match weight")
    values = FP4_E2M1[codes].reshape(-1, cols // 32, 32)
    scale = e8m0_to_f32(exponents).reshape(-1, cols // 32, 1)
    return (values * scale).reshape(*lead, rows, cols)
