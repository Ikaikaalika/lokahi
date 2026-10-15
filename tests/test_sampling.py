"""Correctness evidence for seeded sampling (native/engine/src/sampling.h).

Covers:

* the Philox stream against Random123's known answers;
* the native host sampler against the independent float64 oracle
  (``lokahi.oracles.sampling``);
* the drawn distribution against its exact probabilities;
* the sampler on each backend in ``LOKAHI_TEST_BACKENDS``.

Exact tokens are compared only where the oracle shows the decision is
farther from a tie than float32 rounding could move it. Correctness evidence
only, never throughput evidence.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lokahi.oracles.sampling import SamplingParams, expected_distribution, philox4x32_10, sample

# Decisions closer than this are left to float32 rounding and not compared;
# so are draws that change when a filter limit moves by the oracle's deltas.
SCORE_MARGIN = 1e-4

PARAMS = [
    SamplingParams(temperature=1.0, seed=1),
    SamplingParams(temperature=0.7, top_p=0.9, seed=2),
    SamplingParams(temperature=1.3, top_k=20, seed=3),
    SamplingParams(temperature=0.9, min_p=0.05, seed=4),
    SamplingParams(temperature=0.6, top_p=0.8, top_k=40, min_p=0.02, seed=(1 << 64) - 1),
    SamplingParams(temperature=2.0, top_p=0.3, seed=12345678901234),
    SamplingParams(temperature=0.05, top_p=0.95, seed=6),
]


def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _run(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def _native_samples(cli: Path, tmp_path: Path, rows: np.ndarray, params: SamplingParams, steps: int,
                    first_step: int = 0, backend: str = "cpu") -> np.ndarray:
    path = tmp_path / "logits.f32"
    rows.astype(np.float32).tofile(path)
    report = _run(cli, "sample", "--logits-file", str(path), "--vocab", str(rows.shape[1]), *params.cli_args(),
                  "--steps", str(steps), "--first-step", str(first_step), "--backend", backend)
    return np.asarray(report["tokens"])


def test_philox_matches_random123_known_answers():
    cases = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
         (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ]
    for counter, key, expected in cases:
        assert tuple(int(v) for v in philox4x32_10(np.array([counter], dtype=np.uint32), key)[0]) == expected


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("vocab", [7, 1000, 4099, 262_144])
def test_native_sampler_matches_the_oracle(native_cli, tmp_path, vocab, backend):
    """The host sampler and, on Metal, the decoder's GPU kernel itself."""

    rng = np.random.default_rng(vocab)
    # Peaked and flat rows, with repeated values to exercise boundary ties.
    rows = np.concatenate([rng.normal(0, 3, (3, vocab)), rng.normal(0, 0.5, (2, vocab)),
                           np.round(rng.normal(0, 2, (1, vocab)), 1)]).astype(np.float32)
    if vocab > 100_000:  # Gemma's vocabulary size: two rows keep the oracle quick
        rows = rows[[0, 5]]
    steps = 12 if vocab < 100_000 else 3
    compared = total = 0
    for params in PARAMS:
        native = _native_samples(native_cli, tmp_path, rows, params, steps, first_step=5, backend=backend)
        for r, row in enumerate(rows):
            for s in range(steps):
                draw = sample(row, params, 5 + s)
                total += 1
                if draw.score_margin < SCORE_MARGIN or not draw.filters_stable:
                    continue
                compared += 1
                assert native[r, s] == draw.token, (params, r, s)
                assert draw.kept[native[r, s]]
    assert compared >= 0.97 * total


@pytest.mark.parametrize("backend", _backends())
def test_greedy_parameters_take_the_argmax(native_cli, tmp_path, backend):
    rows = np.array([[0.1, 2.0, -1.0, 2.0, 1.9]], dtype=np.float32)
    for params in [SamplingParams(), SamplingParams(temperature=1.0, top_k=1, seed=3)]:
        assert _native_samples(native_cli, tmp_path, rows, params, 20, backend=backend).tolist() == [[1] * 20]


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=1.0, seed=11),
        SamplingParams(temperature=0.5, top_p=0.7, seed=12),
        SamplingParams(temperature=1.5, top_k=3, seed=13),
        SamplingParams(temperature=1.0, min_p=0.2, seed=14),
    ],
)
def test_draws_follow_the_exact_distribution(native_cli, tmp_path, params, backend):
    logits = np.array([[1.0, 0.2, -0.5, 2.0, 1.7, -3.0]], dtype=np.float32)
    steps = 40_000 if backend == "cpu" else 4_000  # one GPU dispatch per draw
    counts = np.bincount(_native_samples(native_cli, tmp_path, logits, params, steps, backend=backend)[0],
                         minlength=6)
    probabilities = expected_distribution(logits[0], params)
    assert np.all(counts[probabilities == 0] == 0), "drew a filtered token"
    support = probabilities > 0
    expected = probabilities[support] * steps
    chi2 = float(((counts[support] - expected) ** 2 / expected).sum())
    dof = int(support.sum()) - 1
    # Mean dof, standard deviation sqrt(2 dof): eight deviations is far past chance.
    assert chi2 < dof + 8 * math.sqrt(2 * dof) + 1, (chi2, counts.tolist(), expected.tolist())


def test_invalid_parameters_fail_closed(native_cli, tmp_path):
    path = tmp_path / "row.f32"
    np.zeros(4, dtype=np.float32).tofile(path)
    for flags in (["--temperature", "-1"], ["--temperature", "1", "--top-p", "0"],
                  ["--temperature", "1", "--min-p", "1.5"], ["--temperature", "nan"]):
        result = subprocess.run([str(native_cli), "sample", "--logits-file", str(path), "--vocab", "4", *flags],
                                capture_output=True, text=True)
        assert result.returncode != 0 and "sampling:" in result.stderr, flags
