from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.compare_mlx_baseline import (
    BaselineComparisonError,
    compare_mlx_baseline,
)


def _report(runner: str, factor: float = 1.0) -> dict:
    higher = {
        "prompt_tokens_per_second": 100.0 * factor,
        "decode_tokens_per_second": 50.0 * factor,
        "aggregate_tokens_per_second_including_prefill": 40.0 * factor,
    }
    lower = {
        "ttft_ms": 100.0 / factor,
        "inter_token_latency_ms": 20.0 / factor,
        "peak_memory_gb": 2.0 / factor,
    }
    return {
        "runner": runner,
        "hardware": {"chip": "Apple Test", "macos_build": "test"},
        "workload": {"prompt_tokens_per_sequence": 8, "sampling": "greedy"},
        "model": {
            "repo": "example/model",
            "revision": "a" * 40,
            "artifact": {
                "weights": [
                    {"filename": "model.safetensors", "bytes": 7, "sha256": "b" * 64}
                ]
            },
        },
        "correctness": {
            "requested_token_count_passed": True,
            "repetition_token_invariance": True,
        },
        "summary": {
            name: {"mean": value}
            for name, value in {**higher, **lower}.items()
        },
        "repetitions": [{"token_ids": [[1, 2, 3]]}],
    }


def test_lokahi_candidate_passes_mvp_gate_with_parity_and_ten_percent_win() -> None:
    report = compare_mlx_baseline(
        _report("mlx-lm-baseline"),
        _report("lokahi-native-metal", factor=1.10),
    )

    assert report["exact_greedy_token_parity"] is True
    assert report["has_ten_percent_win"] is True
    assert report["standalone_mvp_gate_passed"] is True


def test_token_mismatch_blocks_promotion() -> None:
    candidate = _report("lokahi-native-metal", factor=1.20)
    candidate["repetitions"][0]["token_ids"] = [[9]]

    report = compare_mlx_baseline(_report("mlx-lm-baseline"), candidate)

    assert report["exact_greedy_token_parity"] is False
    assert report["standalone_mvp_gate_passed"] is False


def test_incomparable_artifact_is_rejected() -> None:
    candidate = deepcopy(_report("lokahi-native-metal"))
    candidate["model"]["revision"] = "c" * 40

    with pytest.raises(BaselineComparisonError, match="artifact"):
        compare_mlx_baseline(_report("mlx-lm-baseline"), candidate)


def test_native_flat_token_layout_matches_mlx_batch_layout() -> None:
    # The native engine reports one sequence flat; MLX-LM reports a batch.
    candidate = _report("lokahi-native-metal", factor=1.10)
    candidate["repetitions"][0]["token_ids"] = [1, 2, 3]

    report = compare_mlx_baseline(_report("mlx-lm-baseline"), candidate)

    assert report["exact_greedy_token_parity"] is True


def test_thermal_state_is_reported_not_compared() -> None:
    baseline = _report("mlx-lm-baseline")
    baseline["hardware"]["thermal_state_after_run"] = "nominal"
    candidate = _report("lokahi-native-metal", factor=1.10)
    candidate["hardware"]["thermal_state_after_run"] = "fair"

    report = compare_mlx_baseline(baseline, candidate)

    assert report["thermal_state_after_run"] == {"mlx_lm": "nominal", "lokahi": "fair"}


def test_different_os_build_is_rejected() -> None:
    candidate = _report("lokahi-native-metal")
    candidate["hardware"]["macos_build"] = "other"

    with pytest.raises(BaselineComparisonError, match="hardware"):
        compare_mlx_baseline(_report("mlx-lm-baseline"), candidate)
