"""The MLX-VLM control for speculative benchmarks, on a generated fixture.

Correctness only: the fixture's timings are not evidence. Plain and MTP
greedy runs must produce the oracle's tokens, and the MTP run must count
its own rounds. Needs ``mlx`` and an ``mlx_vlm`` with the ``qwen3_5_mtp``
drafter, like ``test_mlx_vlm_mtp_crosscheck.py``.
"""
from __future__ import annotations

import importlib.util

import pytest

from lokahi.fixtures import TinyQwen35Spec, write_tiny_qwen35, write_tiny_qwen35_mtp_drafter
from lokahi.oracles.qwen3_5 import Qwen35Reference


def _has_drafter() -> bool:
    try:
        return importlib.util.find_spec("mlx_vlm.speculative.drafters.qwen3_5_mtp") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _has_drafter(), reason="mlx and an mlx_vlm with the qwen3_5_mtp drafter")


def test_the_control_decodes_greedily_with_and_without_the_head(tmp_path):
    from benchmarks.benchmark_mlx_vlm_speculative import bench, load_control

    model_dir = write_tiny_qwen35(tmp_path / "model", TinyQwen35Spec())
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", TinyQwen35Spec(mtp=True))
    prompt = [2, 5, 9, 33, 100, 7]
    expected = Qwen35Reference.load(model_dir).greedy(prompt, 12)
    model, processor, draft_model, kind = load_control(model_dir, drafter)
    plain = bench(model, processor, prompt, 12, 1, 2)
    assert [run["token_ids"] for run in plain["repetitions"]] == [expected, expected]
    assert plain["matches_plain_greedy"] is None and "speculation" not in plain["repetitions"][0]
    speculative = bench(model, processor, prompt, 12, 1, 2, draft_model, kind, 3)
    assert speculative["matches_plain_greedy"] is True
    for run in speculative["repetitions"]:
        assert run["token_ids"] == expected
        stats = run["speculation"]
        # Each round drafts up to 3; rounds and drafts are this run's alone.
        assert 0 < stats["passes"] < 12 and stats["passes"] <= stats["drafted"] <= 3 * stats["passes"]
        assert run["inter_token_latency_ms"] == pytest.approx(1000 / run["decode_tokens_per_second"])
