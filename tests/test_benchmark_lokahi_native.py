"""The native benchmark driver's speculation inputs and summary.

Correctness only: the engine runs a tiny generated model, whose timings on a
test machine are not evidence; only arguments, tokens and counts are checked.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.benchmark_lokahi_native import (
    run_native_bench,
    speculation_arguments,
    summarize_speculation,
    text_prompt_tokens,
    validate_drafter_snapshot,
)
from lokahi.fixtures import TinyQwen35Spec, write_tiny_qwen35, write_tiny_qwen35_mtp_drafter
from lokahi.oracles.qwen3_5 import Qwen35Reference

REVISION = "a" * 40


def _snapshot(root: Path, model_type: str, name: str = REVISION) -> Path:
    snapshot = root / "models--fixture" / "snapshots" / name
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    return snapshot


def test_a_drafter_is_a_pinned_qwen3_5_mtp_snapshot_inside_the_approved_root(tmp_path: Path) -> None:
    root = tmp_path / "ssd"
    drafter = _snapshot(root, "qwen3_5_mtp")
    assert validate_drafter_snapshot(drafter, root) == drafter.resolve()
    with pytest.raises(ValueError, match="qwen3_5_mtp drafter, not 'qwen3_5'"):
        validate_drafter_snapshot(_snapshot(root / "decoder", "qwen3_5"), root)
    with pytest.raises(ValueError, match="40-hex snapshot revision"):
        validate_drafter_snapshot(_snapshot(root / "branch", "qwen3_5_mtp", "main"), root)
    outside = _snapshot(tmp_path / "elsewhere", "qwen3_5_mtp")
    with pytest.raises(ValueError, match="inside the approved SSD root"):
        validate_drafter_snapshot(outside, root)


def test_speculation_arguments() -> None:
    assert speculation_arguments(0, "mtp", Path("/d")) == []
    assert speculation_arguments(3, "lookup", None) == ["--speculate", "3", "--drafter", "lookup"]
    assert speculation_arguments(2, "mtp", Path("/d")) == ["--speculate", "2", "--drafter", "mtp",
                                                           "--draft-model", "/d"]


class _Tokenizer:
    def __init__(self, chat_template: str | None):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, add_generation_prompt):
        assert messages == [{"role": "user", "content": "hi"}] and add_generation_prompt
        return [7, 8, 9]

    def encode(self, text):
        return [1, 2]


def test_a_text_prompt_is_one_user_turn_in_the_chat_template() -> None:
    assert text_prompt_tokens(_Tokenizer("{{ messages }}"), "hi") == [7, 8, 9]
    assert text_prompt_tokens(_Tokenizer(None), "hi") == [1, 2]


def test_speculation_summary() -> None:
    summary = summarize_speculation([
        {"speculation": {"passes": 4, "drafted": 12, "accepted": 6}},
        {"speculation": {"passes": 2, "drafted": 6, "accepted": 6}},
    ])
    assert summary["acceptance_rate"]["minimum"] == 0.5 and summary["acceptance_rate"]["mean"] == 0.75
    assert summary["tokens_per_verify_pass"]["minimum"] == 2.5 and summary["tokens_per_verify_pass"]["mean"] == 3.25
    assert summarize_speculation([{"speculation": {"passes": 0, "drafted": 0, "accepted": 0}}])[
        "acceptance_rate"]["median"] == 0.0


def test_the_driver_runs_the_engines_mtp_bench(native_cli, tmp_path: Path) -> None:
    decoder = write_tiny_qwen35(tmp_path / "decoder", TinyQwen35Spec())
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", TinyQwen35Spec(mtp=True))
    prompt = [2, 5, 9, 33, 100, 7]
    native = run_native_bench(native_cli, decoder, prompt, 8, 0, 2, "cpu", 64,
                              extra=speculation_arguments(2, "mtp", drafter))
    assert native["speculation"]["matches_plain_greedy"] is True
    expected = Qwen35Reference.load(decoder).greedy(prompt, 8)
    assert [run["token_ids"] for run in native["repetitions"]] == [expected, expected]
    assert summarize_speculation(native["repetitions"])["tokens_per_verify_pass"]["minimum"] >= 1.0
