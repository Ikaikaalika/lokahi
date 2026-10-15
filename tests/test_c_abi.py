"""Correctness evidence for the C ABI (include/lokahi/lokahi.h) through ctypes.

This is the surface Swift and the Common Compute adapter link against, so it
is exercised directly rather than only through the CLI.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures import TinyGemma3Spec, write_tiny_gemma3
from lokahi.oracles import Gemma3Reference
from lokahi.oracles.sampling import SamplingParams, sample


class LoadOptions(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_char_p),
        ("max_context", ctypes.c_int32),
        ("prefill_chunk", ctypes.c_int32),
        ("cpu_threads", ctypes.c_int32),
    ]


class Timing(ctypes.Structure):
    _fields_ = [("prefill_seconds", ctypes.c_double), ("decode_seconds", ctypes.c_double), ("generated_tokens", ctypes.c_int32)]


class Sampling(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("min_p", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


@pytest.fixture(scope="module")
def lib(native_cli):
    name = "liblokahi.dylib" if sys.platform == "darwin" else "liblokahi.so"
    library = ctypes.CDLL(str(Path(native_cli).parent / name))
    library.lokahi_last_error.restype = ctypes.c_char_p
    library.lokahi_model_backend.restype = ctypes.c_char_p
    return library


def _check(lib, code):
    assert code == 0, lib.lokahi_last_error().decode()


def test_abi_version_and_errors(lib):
    assert lib.lokahi_abi_version() == 2
    handle = ctypes.c_void_p()
    assert lib.lokahi_model_load(b"/nonexistent", None, ctypes.byref(handle)) != 0
    assert b"config.json" in lib.lokahi_last_error()


def test_model_prefill_decode_and_generate(lib, tmp_path):
    directory = write_tiny_gemma3(tmp_path / "model", TinyGemma3Spec())
    oracle = Gemma3Reference.load(directory)
    options = LoadOptions(b"cpu", 64, 512, 2)
    handle = ctypes.c_void_p()
    _check(lib, lib.lokahi_model_load(str(directory).encode(), ctypes.byref(options), ctypes.byref(handle)))
    try:
        assert lib.lokahi_model_backend(handle) == b"cpu"
        vocab = lib.lokahi_model_vocab_size(handle)
        prompt = [2, 5, 9, 33, 100]
        tokens = (ctypes.c_int32 * len(prompt))(*prompt)
        logits = (ctypes.c_float * vocab)()
        _check(lib, lib.lokahi_prefill(handle, tokens, len(prompt), logits))
        np.testing.assert_allclose(np.ctypeslib.as_array(logits), oracle.logits(prompt)[-1], rtol=0, atol=2e-3)
        assert lib.lokahi_model_position(handle) == len(prompt)

        lib.lokahi_model_reset(handle)
        out = (ctypes.c_int32 * 10)()
        count = ctypes.c_int32()
        timing = Timing()
        _check(lib, lib.lokahi_generate_greedy(handle, tokens, len(prompt), 10, 0, out, ctypes.byref(count), ctypes.byref(timing)))
        assert list(out[: count.value]) == oracle.greedy(prompt, 10)
        assert timing.generated_tokens == 10

        # Seeded generation: each token is the reference sampler's draw at its step.
        params = Sampling(temperature=4.0, top_p=1.0, top_k=40, min_p=0.0, seed=2024)
        lib.lokahi_model_reset(handle)
        _check(lib, lib.lokahi_generate(handle, tokens, len(prompt), 10, ctypes.byref(params), 0, out,
                                        ctypes.byref(count), None))
        sampled = list(out[: count.value])
        lib.lokahi_model_reset(handle)
        _check(lib, lib.lokahi_generate(handle, tokens, len(prompt), 10, ctypes.byref(params), 0, out,
                                        ctypes.byref(count), None))
        assert list(out[: count.value]) == sampled
        reference = SamplingParams(temperature=4.0, top_k=40, seed=2024)
        context = list(prompt)
        for step, token in enumerate(sampled):
            draw = sample(oracle.logits(context)[-1], reference, step)
            if draw.score_margin < 2e-2:
                break
            assert token == draw.token, step
            context.append(token)
        lib.lokahi_model_reset(handle)
        _check(lib, lib.lokahi_generate(handle, tokens, len(prompt), 10, None, 0, out, ctypes.byref(count), None))
        assert list(out[: count.value]) == oracle.greedy(prompt, 10)
    finally:
        lib.lokahi_model_free(handle)


def test_reference_sampler_and_parameter_errors(lib):
    logits = np.array([0.5, 3.0, 2.9, -1.0, 3.0, 0.0, 1.0], dtype=np.float32)
    buffer = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    token = ctypes.c_int32()
    params = Sampling(temperature=0.9, top_p=0.95, top_k=0, min_p=0.0, seed=77)
    reference = SamplingParams(temperature=0.9, top_p=0.95, seed=77)
    for step in range(32):
        _check(lib, lib.lokahi_sample(buffer, len(logits), ctypes.byref(params), step, ctypes.byref(token)))
        assert token.value == sample(logits, reference, step).token
    _check(lib, lib.lokahi_sample(buffer, len(logits), None, 0, ctypes.byref(token)))
    assert token.value == 1
    bad = Sampling(temperature=1.0, top_p=0.0, top_k=0, min_p=0.0, seed=0)
    assert lib.lokahi_sample(buffer, len(logits), ctypes.byref(bad), 0, ctypes.byref(token)) != 0
    assert b"top_p" in lib.lokahi_last_error()


def test_tokenizer_round_trip_and_capacity_contract(lib, tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    import test_tokenizer as fixtures

    reference = fixtures.gemma_style()
    reference.save(str(tmp_path / "tokenizer.json"))
    handle = ctypes.c_void_p()
    _check(lib, lib.lokahi_tokenizer_load(str(tmp_path).encode(), ctypes.byref(handle)))
    try:
        text = "Aloha kākou 🌊 <start_of_turn>".encode()
        expected = reference.encode(text.decode()).ids
        count = ctypes.c_int32()
        small = (ctypes.c_int32 * 1)()
        assert lib.lokahi_tokenize(handle, text, len(text), 1, small, 1, ctypes.byref(count)) != 0
        assert count.value == len(expected)
        ids = (ctypes.c_int32 * count.value)()
        _check(lib, lib.lokahi_tokenize(handle, text, len(text), 1, ids, count.value, ctypes.byref(count)))
        assert list(ids) == expected

        length = ctypes.c_size_t()
        assert lib.lokahi_detokenize(handle, ids, len(ids), 0, None, 0, ctypes.byref(length)) != 0
        buffer = ctypes.create_string_buffer(length.value + 1)
        _check(lib, lib.lokahi_detokenize(handle, ids, len(ids), 0, buffer, len(buffer), ctypes.byref(length)))
        assert buffer.value.decode() == reference.decode(expected, skip_special_tokens=False)
    finally:
        lib.lokahi_tokenizer_free(handle)
