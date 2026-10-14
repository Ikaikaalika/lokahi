"""Correctness evidence for the improvement loop's scoreboard and attribution.

Scoring, noise thresholds, promotion records and the PR comment are checked
on synthetic reports shaped like the real benchmark outputs. The attribution
reader is also checked against real ``lokahi profile`` output from a CPU
fixture, so the C++ and Python sides agree on the schema. No hardware claim.
"""
from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.attribution import AttributionError, attribute, attribute_phase, peak_bandwidth
from benchmarks.attribution import render_markdown as render_attribution
from benchmarks.scoreboard import (
    MARKER,
    build_scoreboard,
    compare_to_best,
    compact_record,
    exit_status,
    hardware_key,
    render_markdown,
    score_model,
    token_agreement,
    write_outputs,
)

HARDWARE = {"chip": "Apple M1", "unified_memory_bytes": 16 * 2**30, "architecture": "arm64",
            "macos_version": "26.0", "macos_build": "25A100", "thermal_state_after_run": "nominal"}
WORKLOAD = {"prompt_tokens_per_sequence": 512, "output_tokens_per_sequence": 128, "batch_size": 1,
            "sampling": "greedy", "warmups": 2, "repetitions": 5, "prefix_cache": False}


def _native(decode=100.0, ttft=50.0, spread=0.0, tokens=(1, 2, 3, 4), runner="lokahi-native-metal",
            revision="a" * 40, deterministic=True) -> dict:
    reps = []
    for i in range(5):
        jitter = 1.0 + spread * (i - 2) / 2.0
        reps.append({
            "decode_tokens_per_second": decode * jitter,
            "ttft_ms": ttft / jitter,
            "prompt_tokens_per_second": 4000.0 * jitter,
            "peak_memory_gb": 0.6,
            "inter_token_latency_ms": 1000.0 / (decode * jitter),
            "aggregate_tokens_per_second_including_prefill": decode * 0.9 * jitter,
            "token_ids": list(tokens) if runner != "mlx-lm-baseline" else [list(tokens)],
        })
    summary = {name: {"mean": sum(r[name] for r in reps) / 5} for name in reps[0] if name != "token_ids"}
    return {
        "runner": runner,
        "hardware": dict(HARDWARE),
        "workload": dict(WORKLOAD),
        "model": {"model_id": "qwen3-0.6b", "repo": "mlx-community/Qwen3-0.6B-4bit", "revision": revision,
                  "local_snapshot": "/Volumes/SSD/models/x",
                  "artifact": {"weights": [{"filename": "model.safetensors", "bytes": 1, "sha256": "b" * 64}]}},
        "software": {"lokahi_backend": "metal", "lokahi_revision": "c" * 40},
        "correctness": {"requested_token_count_passed": True, "repetition_token_invariance": deterministic},
        "summary": summary,
        "repetitions": reps,
    }


def _profile(steps=4) -> dict:
    def phase(scale):
        return {"wall_seconds": 0.02 * scale, "sections": [
            {"name": "qkv", "seconds": 0.004 * scale, "calls": 28 * scale, "bytes": 0.1e9 * scale},
            {"name": "attention", "seconds": 0.002 * scale, "calls": 28 * scale, "bytes": 0.01e9 * scale},
            {"name": "mlp", "seconds": 0.008 * scale, "calls": 28 * scale, "bytes": 0.4e9 * scale},
            {"name": "lm_head", "seconds": 0.003 * scale, "calls": scale, "bytes": 0.15e9 * scale},
        ]}
    return {"backend": "metal", "architecture": "qwen3", "decode_steps": steps, "weight_bytes": int(0.66e9),
            "prefill": phase(1), "decode": phase(steps)}


def test_peak_bandwidth_lookup_needs_an_unambiguous_chip():
    assert peak_bandwidth("Apple M1") == pytest.approx(68.25e9)
    assert peak_bandwidth("Apple M3 Max", gpu_cores=40) == pytest.approx(400e9)
    assert peak_bandwidth("Apple M3 Max") is None  # two configurations
    assert peak_bandwidth("Apple M99") is None


def test_attribution_ranks_by_time_above_the_bandwidth_floor():
    result = attribute(_profile(), peak_bandwidth("Apple M1"))
    decode = result["decode_per_token"]
    assert decode["ranked_by"] == "recoverable_seconds"
    names = [row["name"] for row in decode["sections"]]
    # mlp takes the longest, but qkv has the most time above its bandwidth
    # floor (4 ms against 1.47 ms at 68.25 GB/s), so qkv is the first target.
    assert names == ["qkv", "mlp", "attention", "lm_head"]
    assert sum(row["share"] for row in decode["sections"]) == pytest.approx(1.0)
    assert decode["section_seconds"] == pytest.approx(0.017)
    for row in decode["sections"]:
        floor = row["bytes"] / 68.25e9
        assert row["recoverable_seconds"] == pytest.approx(max(0.0, row["seconds"] - floor))
    unknown = attribute(_profile(), None)["decode_per_token"]
    assert unknown["ranked_by"] == "seconds" and unknown["sections"][0]["recoverable_seconds"] is None
    assert "peak unknown" in render_attribution(attribute(_profile(), None))
    with pytest.raises(AttributionError):
        attribute_phase({"sections": []}, None)


def test_attribution_reads_real_profile_output(native_cli, tmp_path):
    from lokahi.fixtures import TinyDenseSpec, write_tiny_dense

    directory = write_tiny_dense(tmp_path / "model", TinyDenseSpec(model_type="qwen3"))
    raw = subprocess.run([str(native_cli), "profile", "--model", str(directory), "--backend", "cpu",
                          "--tokens", "1,5,9,33", "--max-new", "4"], capture_output=True, text=True, check=True)
    result = attribute(json.loads(raw.stdout), peak_bandwidth("Apple M1"))
    names = {row["name"] for row in result["decode_per_token"]["sections"]}
    assert {"qkv", "attention", "lm_head"} <= names
    assert "| Section |" in render_attribution(result)


def test_compare_to_best_uses_a_noise_threshold():
    best = _native(decode=100.0)
    assert compare_to_best(_native(decode=110.0), best)["verdict"] == "improved"
    assert compare_to_best(_native(decode=90.0), best)["verdict"] == "regressed"
    assert compare_to_best(_native(decode=102.0), best)["verdict"] == "unchanged"  # under 3%
    # A noisy run widens the threshold: 8% down is not a regression at 20% spread.
    assert compare_to_best(_native(decode=92.0, spread=0.2), best)["verdict"] == "unchanged"
    # TTFT gates too, in the lower-is-better direction.
    assert compare_to_best(_native(ttft=60.0), best)["verdict"] == "regressed"


def test_compare_to_best_refuses_incomparable_records():
    best = _native()
    assert compare_to_best(_native(), None)["verdict"] == "no baseline"
    other = _native(revision="d" * 40)
    assert compare_to_best(other, best)["reason"] == "different model revision"
    shorter = _native()
    shorter["workload"]["output_tokens_per_sequence"] = 32
    assert compare_to_best(shorter, best)["reason"] == "different workload"
    elsewhere = _native()
    elsewhere["hardware"]["chip"] = "Apple M4"
    assert compare_to_best(elsewhere, best)["reason"] == "different chip or memory size"
    assert hardware_key(best) == "apple-m1-16gb"


def test_score_model_promotes_only_deterministic_wins():
    control = _native(decode=80.0, ttft=60.0, tokens=(1, 2, 9, 9), runner="mlx-lm-baseline")
    control["software"] = {}
    win = score_model("qwen3-0.6b", _native(decode=120.0), control, None, _native(decode=100.0))
    assert win["status"] == "improved" and win["promote"]
    assert win["tokens_matching_mlx_lm"] == [2, 4]
    assert win["versus_mlx_lm"]["decode_tokens_per_second"] == pytest.approx(0.5)
    flaky = score_model("qwen3-0.6b", _native(deterministic=False), None, None, None)
    assert flaky["status"] == "regressed" and not flaky["promote"]
    first = score_model("qwen3-0.6b", _native(), None, None, None)
    assert first["status"] == "no baseline" and first["promote"]
    missing = score_model("qwen3-0.6b", None, None, None, None)
    assert missing["status"] == "not staged"
    failed = score_model("qwen3-0.6b", None, None, None, None, ["bench exited 2"])
    assert failed["status"] == "failed"
    assert token_agreement([1, 2, 3], [1, 2, 4, 5]) == 2


def test_compact_record_drops_paths_and_repeated_tokens():
    record = compact_record(_native())
    assert "local_snapshot" not in record["model"]
    assert "token_ids" in record["repetitions"][0]
    assert all("token_ids" not in rep for rep in record["repetitions"][1:])
    assert compare_to_best(_native(), record)["verdict"] == "unchanged"  # still scoreable


def test_scoreboard_end_to_end(tmp_path):
    best_dir = tmp_path / "best"
    (best_dir / "apple-m1-16gb").mkdir(parents=True)
    (best_dir / "apple-m1-16gb" / "llama-3.2-1b.json").write_text(json.dumps(compact_record(_native(decode=100.0))))
    control = _native(decode=80.0, runner="mlx-lm-baseline")
    runs = [
        {"model_id": "qwen3-0.6b", "native": _native(), "control": control, "profile": _profile(), "notes": []},
        {"model_id": "llama-3.2-1b", "native": _native(decode=85.0), "notes": []},
        {"model_id": "gemma-3-1b", "notes": []},
    ]
    scoreboard = build_scoreboard(runs, best_dir, "e" * 40, {"backend": "metal"}, gpu_cores=8)
    statuses = {entry["model_id"]: entry["status"] for entry in scoreboard["models"]}
    assert statuses == {"qwen3-0.6b": "no baseline", "llama-3.2-1b": "regressed", "gemma-3-1b": "not staged"}
    assert exit_status(scoreboard) == 1
    markdown = write_outputs(scoreboard, tmp_path / "out")
    assert markdown.startswith(MARKER)
    assert "| qwen3-0.6b | no baseline |" in markdown and "| llama-3.2-1b | regressed |" in markdown
    assert "Where decode time goes: qwen3-0.6b" in markdown
    promoted = tmp_path / "out" / "promote" / "apple-m1-16gb" / "qwen3-0.6b.json"
    assert json.loads(promoted.read_text())["model"]["revision"] == "a" * 40
    assert not (tmp_path / "out" / "promote" / "apple-m1-16gb" / "llama-3.2-1b.json").exists()
    # The promote block in the comment is the same record, parseable as JSON.
    block = markdown.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(block) == json.loads(promoted.read_text())
    assert json.loads((tmp_path / "out" / "scoreboard.json").read_text())["revision"] == "e" * 40


def test_render_markdown_without_regressions_says_so():
    scoreboard = build_scoreboard([{"model_id": "qwen3-0.6b", "native": _native(), "notes": []}],
                                  Path("/nonexistent"), "f" * 40, {})
    assert exit_status(scoreboard) == 0
    assert "**Result: no regression.**" in render_markdown(deepcopy(scoreboard))


def test_render_markdown_keeps_the_comment_under_the_github_limit(monkeypatch):
    import benchmarks.scoreboard as scoreboard_module

    scoreboard = build_scoreboard([{"model_id": "qwen3-0.6b", "native": _native(), "notes": []}],
                                  Path("/nonexistent"), "f" * 40, {})
    assert "```json" in render_markdown(scoreboard)
    monkeypatch.setattr(scoreboard_module, "COMMENT_LIMIT", 500)
    short = render_markdown(scoreboard)
    assert "```json" not in short and "`qwen3-0.6b.json`" in short and "evidence artifact" in short
