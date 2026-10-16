"""Correctness evidence: the Qwen3.5 MTP head's oracle against MLX-VLM's ``qwen3_5_mtp`` drafter.

``Qwen35Mtp`` is pinned to MLX-VLM's speculative drafter (0.7.8), an
independent implementation: MLX-VLM loads the head itself, from a decoder
checkpoint's ``mtp.*`` tensors (its splitter) or from a ``qwen3_5_mtp``
drafter directory, and is fed the oracle's final-normed hidden states and
the decoder's embedding and lm_head. Its first draft's hidden state must
match the oracle's, and its greedy drafts must be the oracle's.

MLX-VLM adds 1 to Transformers' zero-centered norm weights in BF16, which
rounds 1 + w to 8 bits; for Transformers' layout the oracle's head takes
the same rounding so the comparison stays tight.

It needs ``mlx`` and an ``mlx_vlm`` with the ``qwen3_5_mtp`` drafter: hosted
macOS CI runs it on its virtual GPU, and ``mlx[cpu]`` runs it on Linux.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lokahi.fixtures import TinyQwen35Spec, write_tiny_qwen35, write_tiny_qwen35_mtp_drafter
from lokahi.formats.safetensors import f32_to_bf16
from lokahi.oracles.qwen3_5 import Qwen35Mtp, Qwen35Reference


def _has_drafter() -> bool:
    try:
        return importlib.util.find_spec("mlx_vlm.speculative.drafters.qwen3_5_mtp") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _has_drafter(), reason="mlx and an mlx_vlm with the qwen3_5_mtp drafter")

CONTEXT = [2, 5, 9, 33, 100, 7, 7]
LAYOUTS = {
    "qwen3.5-mlx": (TinyQwen35Spec(mtp=True), False),
    "qwen3.5-hf": (TinyQwen35Spec(mlx=False, mtp=True, seed=1), False),
    "qwen3.5-moe-mlx": (TinyQwen35Spec(model_type="qwen3_5_moe", mtp=True), False),
    "qwen3.5-moe-hf-fused": (TinyQwen35Spec(model_type="qwen3_5_moe", mlx=False, mtp=True, seed=1), False),
    "qwen3-next-mlx": (TinyQwen35Spec(model_type="qwen3_next", mtp=True, linear_key_head_dim=16), False),
    "qwen3.5-mlx-drafter": (TinyQwen35Spec(mtp=True), True),
}


def _bf16(values: np.ndarray) -> np.ndarray:
    return (f32_to_bf16(values.astype(np.float32)).astype(np.uint32) << 16).view(np.float32)


def _target_module(target: Qwen35Reference):
    import mlx.core as mx
    import mlx.nn as nn

    class Target(nn.Module):
        """The embedding and lm_head the drafter binds to, in float32."""

        def __init__(self):
            super().__init__()
            embed, head = target.w["model.embed_tokens"], target.w["lm_head"]
            self.embed_tokens = nn.Embedding(*embed.shape)
            self.embed_tokens.weight = mx.array(embed)
            self.lm_head = nn.Linear(head.shape[1], head.shape[0], bias=False)
            self.lm_head.weight = mx.array(head)

    return Target()


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_mtp_head_matches_mlx_vlm(name, tmp_path):
    import mlx.core as mx
    from mlx_vlm.speculative.drafters import load_drafter

    spec, split = LAYOUTS[name]
    directory = write_tiny_qwen35(tmp_path / "model", spec)
    target = Qwen35Reference.load(directory)
    if split:
        drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", spec)
        head = Qwen35Mtp.load(target, drafter)
    else:
        drafter = directory
        head = Qwen35Mtp.from_checkpoint(target)
    if not spec.mlx:
        for key, value in list(head.w.items()):
            if key.endswith(("norm", "layernorm")) or key.startswith("pre_fc_norm"):
                head.w[key] = _bf16(value + 1.0)
        head.norm_offset = 0.0
    model, kind = load_drafter(str(drafter))
    assert kind == "mtp"
    model.reset(_target_module(target))
    hidden = target.hidden_states(CONTEXT[:-1])  # the decoder has read all but the last token
    model.prefill_from_target_hidden(mx.array([CONTEXT[:-1]]), mx.array(hidden[None]), CONTEXT[-1], None,
                                     greedy=True)
    first = np.array(model._seed_hidden.astype(mx.float32)).reshape(-1)
    expected = head.outputs(CONTEXT[1:], hidden)[-1]
    assert np.abs(first - expected).max() < 1e-5 * np.abs(expected).max()
    block = model.draft_block(CONTEXT[-1], None, None, 4, None, greedy=True)
    drafts, _ = head.drafts(CONTEXT, 3)
    assert [int(token) for token in np.array(block).reshape(-1)] == drafts
