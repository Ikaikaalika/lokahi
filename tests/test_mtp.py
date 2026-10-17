"""Qwen3.5's multi-token prediction head as a speculative drafter, against the NumPy oracle.

Correctness evidence only: generated fixtures with random weights, so the
head's drafts are rarely right and nothing here measures acceptance or speed.
The head's drafts and logits must equal the oracle's full recomputation
(``Qwen35Mtp``) after any run of speculative decoding, and the output must
stay plain greedy decoding's. The CPU must match within 1e-4 of the logits'
scale; Metal, which stores keys and values as FP16, within 2e-2 with the
same drafts, as for its other logits.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures import TinyQwen35Spec, write_tiny_qwen35, write_tiny_qwen35_mtp_drafter
from lokahi.oracles.qwen3_5 import Qwen35Mtp, Qwen35Reference

TOKENS = [2, 5, 9, 33, 100, 7, 7, 7, 400, 3, 2, 11, 90, 91, 92, 93]

VARIANTS = {
    "qwen3.5-mlx": TinyQwen35Spec(mtp=True),
    "qwen3.5-hf": TinyQwen35Spec(mlx=False, mtp=True, seed=1),
    "qwen3.5-mlx-5bit": TinyQwen35Spec(mtp=True, bits=5, group_size=64),
    "qwen3.5-moe-mlx": TinyQwen35Spec(model_type="qwen3_5_moe", mtp=True),
    "qwen3.5-moe-hf-fused": TinyQwen35Spec(model_type="qwen3_5_moe", mlx=False, mtp=True, seed=1),
    "qwen3-next-mlx": TinyQwen35Spec(model_type="qwen3_next", mtp=True, linear_key_head_dim=16),
}


def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _generate(cli: Path, directory: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), "generate", "--model", str(directory), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _error(cli: Path, *args: str) -> str:
    result = subprocess.run([str(cli), *args], capture_output=True, text=True)
    assert result.returncode != 0
    return result.stderr


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def mtp_model(request, tmp_path_factory):
    directory = write_tiny_qwen35(tmp_path_factory.mktemp(request.param), VARIANTS[request.param])
    target = Qwen35Reference.load(directory)
    return directory, target, Qwen35Mtp.from_checkpoint(target)


def _probe(cli, directory, tmp_path, prompt, *args, backend: str = "cpu") -> tuple[dict, np.ndarray]:
    logits = tmp_path / "mtp.f32"
    report = _generate(cli, directory, "--backend", backend, "--tokens", ",".join(map(str, prompt)),
                       "--mtp-probe", "3", "--mtp-logits", str(logits), *args)
    return report, np.fromfile(logits, dtype=np.float32).reshape(3, -1)


def _assert_head_matches(head: Qwen35Mtp, context: list[int], report: dict, logits: np.ndarray,
                         backend: str = "cpu") -> None:
    drafts, expected = head.drafts(context, 3)
    assert report["mtp_drafts"] == drafts
    tolerance = 1e-4 if backend == "cpu" else 2e-2
    assert np.abs(logits - expected).max() < tolerance * np.abs(expected).max()


def _skip_unsupported(directory: Path, backend: str) -> None:
    if backend == "metal" and "quantization" not in json.loads((directory / "config.json").read_text()):
        pytest.skip("the Metal backend reads quantized checkpoints only")


@pytest.mark.parametrize("backend", _backends())
def test_mtp_drafting_is_plain_greedy_and_the_head_matches_the_oracle(native_cli, mtp_model, backend, tmp_path):
    # The head drafts every pass; with random weights it is almost always
    # wrong, so each pass rolls back to the last token. The output is plain
    # greedy decoding's, and afterwards the head's next drafts and their
    # logits equal the oracle's for the whole context.
    directory, target, head = mtp_model
    _skip_unsupported(directory, backend)
    prompt = TOKENS[:6]
    report, logits = _probe(native_cli, directory, tmp_path, prompt, "--max-new", "10", "--no-eos-stop",
                            "--speculate", "3", "--drafter", "mtp", backend=backend)
    assert report["tokens"] == target.greedy(prompt, 10)
    stats = report["speculation"]
    assert stats["passes"] > 0 and stats["drafted"] > stats["passes"]
    _assert_head_matches(head, prompt + report["tokens"], report, logits, backend)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("name", ["qwen3.5-mlx", "qwen3-next-mlx"])
def test_the_head_follows_accepted_runs_stops_and_limits(native_cli, name, backend, tmp_path):
    # Lookup drafts from a reference (the greedy continuation with two tokens
    # changed) are mostly accepted, so passes keep several tokens, reject the
    # rest, stop inside accepted runs and end at limits inside passes. The
    # head follows every pass (--drafter both) and must end where the
    # oracle's recomputation does.
    directory = write_tiny_qwen35(tmp_path / "model", VARIANTS[name])
    target = Qwen35Reference.load(directory)
    head = Qwen35Mtp.from_checkpoint(target)
    prompt = TOKENS[:6]
    plain = target.greedy(prompt, 16)
    reference = list(prompt + plain)
    for at in (len(prompt) + 3, len(prompt) + 9):
        reference[at] = (reference[at] + 1) % 512
    (tmp_path / "reference.txt").write_text(",".join(map(str, reference)))
    args = ("--speculate", "4", "--drafter", "both", "--draft-reference", str(tmp_path / "reference.txt"))
    report, logits = _probe(native_cli, directory, tmp_path, prompt, "--max-new", "16", "--no-eos-stop", *args,
                            backend=backend)
    assert report["tokens"] == plain
    stats = report["speculation"]
    assert 0 < stats["accepted"] < stats["drafted"]
    _assert_head_matches(head, prompt + plain, report, logits, backend)
    for limit in (3, 7, 10):
        report, logits = _probe(native_cli, directory, tmp_path, prompt, "--max-new", str(limit), "--no-eos-stop",
                                *args, backend=backend)
        assert report["tokens"] == plain[:limit]
        _assert_head_matches(head, prompt + plain[:limit], report, logits, backend)
    config = json.loads((directory / "config.json").read_text())
    for stop in [i for i in range(2, 12) if plain[i] not in plain[:i]][:3]:
        (config.get("text_config") or config)["eos_token_id"] = plain[stop]
        (directory / "config.json").write_text(json.dumps(config))
        report, logits = _probe(native_cli, directory, tmp_path, prompt, "--max-new", "16", *args, backend=backend)
        assert report["tokens"] == plain[: stop + 1]
        _assert_head_matches(head, prompt + plain[: stop + 1], report, logits, backend)


def test_a_split_drafter_is_the_head_beside_the_decoder(native_cli, tmp_path):
    # MLX-VLM's qwen3_5_mtp drafter holds the same tensors without the mtp.
    # prefix; paired with the decoder alone it drafts what the head in the
    # checkpoint does.
    spec = VARIANTS["qwen3.5-mlx"]
    with_head = write_tiny_qwen35(tmp_path / "with-head", spec)
    without = write_tiny_qwen35(tmp_path / "without", TinyQwen35Spec())
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", spec)
    info = json.loads(subprocess.run([str(native_cli), "info", "--model", str(without), "--backend", "cpu"],
                                     check=True, capture_output=True, text=True).stdout)
    assert info["mtp"] is False
    info = json.loads(subprocess.run([str(native_cli), "info", "--model", str(without), "--backend", "cpu",
                                      "--draft-model", str(drafter)],
                                     check=True, capture_output=True, text=True).stdout)
    assert info["mtp"] is True
    args = ("--max-new", "8", "--no-eos-stop", "--speculate", "2", "--drafter", "mtp")
    beside, beside_logits = _probe(native_cli, with_head, tmp_path, TOKENS[:6], *args)
    split, split_logits = _probe(native_cli, without, tmp_path, TOKENS[:6], *args, "--draft-model", str(drafter))
    assert split["tokens"] == beside["tokens"] and split["mtp_drafts"] == beside["mtp_drafts"]
    assert np.array_equal(split_logits, beside_logits)
    target = Qwen35Reference.load(without)
    _assert_head_matches(Qwen35Mtp.load(target, drafter), TOKENS[:6] + split["tokens"], split, split_logits)


def test_a_drafter_takes_the_decoders_norm_convention(native_cli, tmp_path):
    # A drafter stores its norms as 1 + w; beside Transformers' decoder, which
    # stores w, the engine reads the drafter's in the decoder's convention.
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", TinyQwen35Spec(mtp=True))
    decoder = write_tiny_qwen35(tmp_path / "decoder", TinyQwen35Spec(mlx=False, seed=1))
    report, logits = _probe(native_cli, decoder, tmp_path, TOKENS[:6], "--max-new", "6", "--no-eos-stop",
                            "--speculate", "2", "--drafter", "mtp", "--draft-model", str(drafter))
    target = Qwen35Reference.load(decoder)
    assert report["tokens"] == target.greedy(TOKENS[:6], 6)
    _assert_head_matches(Qwen35Mtp.load(target, drafter), TOKENS[:6] + report["tokens"], report, logits)


@pytest.mark.parametrize("backend", _backends())
def test_bench_speculates_only_to_reproduce_plain_greedy(native_cli, backend, tmp_path):
    # bench --speculate decodes plainly once, untimed, and checks every
    # repetition against it; a pass emits its tokens together, so the
    # inter-token latency is the mean. The setup is the hardware run's (a
    # decoder and a split drafter), but a tiny random model's timings are not
    # evidence: only tokens and pass counts are checked.
    spec = VARIANTS["qwen3.5-mlx"]
    decoder = write_tiny_qwen35(tmp_path / "decoder", TinyQwen35Spec())
    _skip_unsupported(decoder, backend)
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", spec)
    prompt = TOKENS[:6]
    plain = Qwen35Reference.load(decoder).greedy(prompt, 12)
    reference = prompt + plain
    reference[len(prompt) + 5] = (reference[len(prompt) + 5] + 1) % 512
    (tmp_path / "reference.txt").write_text(",".join(map(str, reference)))
    bench = (str(native_cli), "bench", "--model", str(decoder), "--backend", backend, "--draft-model", str(drafter),
             "--tokens", ",".join(map(str, prompt)), "--max-new", "12", "--warmups", "1", "--repetitions", "2")
    report = json.loads(subprocess.run(bench, check=True, capture_output=True, text=True).stdout)
    assert report["speculation"] is None
    assert [run["token_ids"] for run in report["repetitions"]] == [plain, plain]
    assert not any("speculation" in run for run in report["repetitions"])
    speculate = ("--speculate", "3", "--draft-reference", str(tmp_path / "reference.txt"))
    for drafter_name in ("lookup", "mtp", "both"):
        report = json.loads(subprocess.run([*bench, *speculate, "--drafter", drafter_name],
                                           check=True, capture_output=True, text=True).stdout)
        assert report["speculation"] == {"max_draft": 3, "drafter": drafter_name, "policy": "fixed",
                                         "inter_token_latency": "mean", "matches_plain_greedy": True}
        for run in report["repetitions"]:
            assert run["token_ids"] == plain
            stats = run["speculation"]
            assert stats["passes"] > 0 and stats["drafted"] >= stats["passes"]
            if drafter_name != "mtp":  # lookup follows the reference up to the changed token
                assert stats["accepted"] > 0
            assert run["inter_token_latency_ms"] == pytest.approx(1000 / run["decode_tokens_per_second"], rel=1e-4)


def test_mtp_fails_closed(native_cli, tmp_path):
    plain = write_tiny_qwen35(tmp_path / "plain", TinyQwen35Spec())
    generate = ("generate", "--backend", "cpu", "--tokens", "2,5,9", "--max-new", "4", "--speculate", "2")
    assert "qwen3_5: this checkpoint has no multi-token prediction head" in _error(
        native_cli, *generate, "--model", str(plain), "--drafter", "mtp")
    assert "the drafter must be lookup, mtp or both, not 'eagle'" in _error(
        native_cli, *generate, "--model", str(plain), "--drafter", "eagle")
    # A drafter whose head does not fit the decoder, and a directory that is
    # not a drafter.
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", TinyQwen35Spec(mtp=True, hidden_size=64))
    assert "the drafter's hidden_size does not match the model's" in _error(
        native_cli, "info", "--model", str(plain), "--backend", "cpu", "--draft-model", str(drafter))
    assert "expected a qwen3_5_mtp drafter, not 'qwen3_5'" in _error(
        native_cli, "info", "--model", str(plain), "--backend", "cpu", "--draft-model", str(plain))
    # Heads the engine does not read.
    for change, message in [({"mtp_num_hidden_layers": 2}, "only a one-layer MTP head is supported"),
                            ({"mtp_use_dedicated_embeddings": True}, "an MTP head's own embeddings are not supported")]:
        directory = write_tiny_qwen35(tmp_path / f"head-{len(change)}-{next(iter(change))}",
                                      TinyQwen35Spec(mtp=True))
        config = json.loads((directory / "config.json").read_text())
        config["text_config"].update(change)
        (directory / "config.json").write_text(json.dumps(config))
        assert message in _error(native_cli, "info", "--model", str(directory), "--backend", "cpu")
    # A head with experts beside a dense decoder.
    moe_drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "moe-drafter",
                                                TinyQwen35Spec(model_type="qwen3_5_moe", mtp=True))
    dense_config = json.loads((moe_drafter / "config.json").read_text())
    dense_config["text_config"] = TinyQwen35Spec().text_config()
    (moe_drafter / "config.json").write_text(json.dumps(dense_config))
    assert "the MTP head's MLP must match the decoder's layers" in _error(
        native_cli, "info", "--model", str(plain), "--backend", "cpu", "--draft-model", str(moe_drafter))


@pytest.mark.skipif("metal" not in _backends(), reason="needs the Metal backend")
def test_metal_runs_without_a_head_its_kernels_cannot_read(native_cli, tmp_path):
    # A drafter quantized in groups of 16 leaves the decoder running on
    # Metal; drafting with it fails closed, and the CPU drafts with it.
    decoder = write_tiny_qwen35(tmp_path / "decoder", TinyQwen35Spec())
    drafter = write_tiny_qwen35_mtp_drafter(tmp_path / "drafter", TinyQwen35Spec(mtp=True, group_size=16))
    info = json.loads(subprocess.run([str(native_cli), "info", "--model", str(decoder), "--backend", "metal",
                                      "--draft-model", str(drafter)],
                                     check=True, capture_output=True, text=True).stdout)
    assert info["mtp"] is False
    args = ("generate", "--model", str(decoder), "--draft-model", str(drafter), "--tokens", "2,5,9",
            "--max-new", "4", "--speculate", "2", "--drafter", "mtp")
    assert "the MTP head cannot run on Metal: metal: fc needs cols and group_size divisible by 32" in _error(
        native_cli, *args, "--backend", "metal")
    assert _generate(native_cli, decoder, *args[3:], "--backend", "cpu")["tokens"]
