"""Correctness evidence for streamed text (native/engine/src/text_stream.h).

The streamer is run over every tokenizer configuration the tokenizer suites
train. Those are the SentencePiece-style ones (byte fallback, Strip,
Metaspace) and the byte-level ones (Qwen 3, Llama 3, GLM, GPT-OSS,
DeepSeek V3, GPT-2). It is fed encoder output and arbitrary id sequences.

Without stop strings, the text pushed out token by token must equal Hugging
Face ``tokenizers``' decode of all the ids. With stop strings, it must equal
that decode cut at the earliest stop string. Nothing may be emitted after
the stop.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

import test_tokenizer as sentencepiece  # noqa: E402
import test_tokenizer_bytelevel as bytelevel  # noqa: E402

TOKENIZERS = {f"sp-{name}": factory for name, factory in sentencepiece.FACTORIES.items()}
TOKENIZERS.update({f"bl-{name}": (lambda n=name: bytelevel._byte_level_tokenizer(n)) for name in bytelevel.NAMES})
TEXTS = sentencepiece.TEXTS[:40] + bytelevel.CORPUS + bytelevel.EDGE_TEXTS


@pytest.fixture(scope="module", params=sorted(TOKENIZERS))
def saved(request, tmp_path_factory):
    from tokenizers import Tokenizer

    directory = tmp_path_factory.mktemp(request.param)
    TOKENIZERS[request.param]().save(str(directory / "tokenizer.json"))
    return directory, Tokenizer.from_file(str(directory / "tokenizer.json"))


def _id_lists(reference) -> list[list[int]]:
    lists = [reference.encode(text).ids for text in TEXTS]
    rng = random.Random(5)
    size = reference.get_vocab_size(with_added_tokens=True)
    lists += [[rng.randrange(size) for _ in range(rng.randint(0, 60))] for _ in range(150)]
    return lists


def _stream(cli: Path, directory: Path, tmp_path: Path, id_lists, stops=None) -> list[dict]:
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(id_lists), encoding="utf-8")
    args = [str(cli), "detokenize", "--model", str(directory), "--ids-file", str(ids_file), "--stream"]
    if stops is not None:
        stops_file = tmp_path / "stops.json"
        stops_file.write_text(json.dumps(stops, ensure_ascii=False), encoding="utf-8")
        args += ["--stops-file", str(stops_file)]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)["streams"]


def test_streamed_text_equals_the_full_decode(native_cli, saved, tmp_path):
    directory, reference = saved
    id_lists = _id_lists(reference)
    streams = _stream(native_cli, directory, tmp_path, id_lists)
    for ids, stream in zip(id_lists, streams):
        expected = reference.decode(ids, skip_special_tokens=True)
        assert "".join(stream["deltas"]) == expected == stream["text"], ids
        assert len(stream["deltas"]) == len(ids) + 1 and stream["stopped_after"] == -1


def test_stop_strings_cut_at_the_earliest_match(native_cli, saved, tmp_path):
    directory, reference = saved
    rng = random.Random(9)
    for trial in range(4):
        id_lists = _id_lists(reference)
        texts = [reference.decode(ids, skip_special_tokens=True) for ids in id_lists]
        # Stops taken from the outputs (so they fire, often mid-token), plus one that never does.
        source = rng.choice([t for t in texts if len(t) > 8])
        stops = []
        for _ in range(3):
            start = rng.randrange(len(source) - 1)
            stops.append(source[start:start + rng.randint(1, 4)])
        stops = list(dict.fromkeys(s for s in stops if s)) + ["\u0000never\u0000"]
        streams = _stream(native_cli, directory, tmp_path, id_lists, stops)
        fired = 0
        for ids, full, stream in zip(id_lists, texts, streams):
            cuts = [full.find(s) for s in stops if s in full]
            expected = full[: min(cuts)] if cuts else full
            assert stream["text"] == expected == "".join(stream["deltas"]), (stops, ids)
            if stream["stopped_after"] >= 0:
                fired += 1
                assert all(delta == "" for delta in stream["deltas"][stream["stopped_after"] + 1:])
        assert fired > 0, stops


def test_text_held_for_a_stop_prefix_is_released_when_ruled_out(native_cli, saved, tmp_path):
    """A stop string's prefix is withheld only until the text rules it out,
    and is kept if generation ends first."""

    directory, reference = saved
    ids = reference.encode("Hello world, this is a test").ids
    full = reference.decode(ids, skip_special_tokens=True)
    stream = _stream(native_cli, directory, tmp_path, [ids], [full + "!"])[0]
    assert stream["text"] == full and stream["stopped_after"] == -1
