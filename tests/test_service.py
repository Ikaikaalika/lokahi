"""Correctness evidence for the wire v1 service (native/engine/src/service.h).

The service runs ``write_tiny_chat_model`` (a tiny generated Gemma 3 with a
byte-level tokenizer and a ChatML chat template), driven through
`lokahi serve` and through the C ABI. Every document it writes must pass the kit's reference verifier
(``verify_kit.validate_document``) and the per-request stream rules:
accepted first at sequence 0, sequences increasing, one terminal receipt.
Outputs are checked against independent references. Prompts are rendered
with jinja2 the way ``transformers`` does and tokenized with Hugging Face
``tokenizers``; tokens come from the NumPy model oracle.

This covers acceptance gate B (stream, cancel, reject, restart), plus gate
C's token and accounting parity, on a fixture model. Correctness evidence
only.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
jinja2 = pytest.importorskip("jinja2")

from lokahi.fixtures.chat_model import CHATML_TEMPLATE, EOS_TOKEN, write_tiny_chat_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "integrations" / "commoncompute" / "lokahi-wire-v1"
sys.path.insert(0, str(KIT / "tools"))
import verify_kit  # noqa: E402

MODEL = {
    "id": "tiny-gemma3",
    "revision": "a" * 40,
    "artifact_digest": "sha256:" + "b" * 64,
    "adapter_id": "gemma3",
    "quantization": "mlx-4bit-group32",
}
RUNTIME = "1" * 64
MESSAGES = [{"role": "system", "content": "Answer concisely."}, {"role": "user", "content": "Aloha kākou! Count to five."}]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Model directory, its NumPy oracle and its reference tokenizer."""

    from tokenizers import Tokenizer

    from lokahi.oracles import Gemma3Reference

    directory = write_tiny_chat_model(tmp_path_factory.mktemp("service-model"))
    return directory, Gemma3Reference.load(directory), Tokenizer.from_file(str(directory / "tokenizer.json"))


def _prompt_ids(reference, messages) -> list[int]:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    text = env.from_string(CHATML_TEMPLATE).render(messages=messages, add_generation_prompt=True, bos_token="",
                                                   eos_token=EOS_TOKEN)
    return reference.encode(text, add_special_tokens=False).ids


def _start(request_id: str, max_output: int = 16, **overrides) -> dict:
    now = int(time.time() * 1000)
    doc = json.loads((KIT / "fixtures" / "valid" / "start-greedy.json").read_text(encoding="utf-8"))
    doc.update(request_id=request_id, task_id=f"task-{request_id}", runtime_revision=RUNTIME, model=dict(MODEL),
               messages=deepcopy(MESSAGES), created_at_unix_ms=now, deadline_unix_ms=now + 120_000)
    doc["objective"].update(context_tokens=4096, max_output_tokens=max_output)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            doc[key].update(value)
        else:
            doc[key] = value
    return doc


def _control(action: str, target: str | None = None) -> dict:
    doc = {"schema_version": 1, "message_type": "control", "control_id": f"control-{action}-{time.time_ns()}",
           "action": action, "issued_at_unix_ms": int(time.time() * 1000)}
    if target is not None:
        doc["target_request_id"] = target
    return doc


def _config(path: Path, directory: Path, **options) -> Path:
    config = {"models": [dict(MODEL, directory=str(directory))], "backend": "cpu", "runtime_revision": RUNTIME,
              "max_context": 4600, **options}
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class Streams:
    """Validates every document and groups them by request."""

    def __init__(self):
        self.by_request: dict[str, list[dict]] = {}

    def add(self, raw: bytes) -> dict:
        doc = json.loads(raw, object_pairs_hook=verify_kit._unique_object)
        kind = verify_kit.validate_document(doc, raw)
        assert kind in ("event", "receipt"), kind
        stream = self.by_request.setdefault(doc["request_id"], [])
        assert not stream or stream[-1]["message_type"] == "event", "document after the receipt"
        sequence = doc["sequence"] if kind == "event" else doc["terminal_sequence"]
        if stream:
            last = stream[-1]["sequence"]
            assert sequence > last, (last, sequence)
        elif kind == "event":
            assert sequence == 0 and doc["kind"] == "accepted", doc
        stream.append(doc)
        return doc

    def receipt(self, request_id: str) -> dict:
        stream = self.by_request[request_id]
        assert stream[-1]["message_type"] == "receipt"
        return stream[-1]

    def text(self, request_id: str) -> str:
        return "".join(d.get("text_delta", "") for d in self.by_request[request_id] if d["message_type"] == "event")


class Server:
    """`lokahi serve` with documents written and read line by line."""

    def __init__(self, cli: Path, config: Path):
        self.process = subprocess.Popen([str(cli), "serve", "--config", str(config)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.streams = Streams()
        self.errors: list[str] = []
        self._err = threading.Thread(target=self._read_errors, daemon=True)
        self._err.start()

    def _read_errors(self):
        for line in self.process.stderr:
            self.errors.append(json.loads(line)["error"])

    def send(self, doc: dict | bytes):
        raw = doc if isinstance(doc, bytes) else json.dumps(doc, ensure_ascii=False).encode()
        self.process.stdin.write(raw + b"\n")
        self.process.stdin.flush()

    def read_until(self, predicate) -> dict:
        for line in self.process.stdout:
            doc = self.streams.add(line.rstrip(b"\n"))
            if predicate(doc):
                return doc
        raise AssertionError(f"stream ended; errors: {self.errors}")

    def receipt(self, request_id: str) -> dict:
        return self.read_until(lambda d: d["message_type"] == "receipt" and d["request_id"] == request_id)

    def close(self) -> int:
        self.process.stdin.close()
        for line in self.process.stdout:
            self.streams.add(line.rstrip(b"\n"))
        code = self.process.wait(timeout=120)
        self._err.join(timeout=10)
        return code


def _serve(cli, config, docs) -> Server:
    server = Server(cli, config)
    for doc in docs:
        server.send(doc)
    assert server.close() == 0
    return server


def _expected_greedy(tiny, steps):
    _, oracle, reference = tiny
    prompt = _prompt_ids(reference, MESSAGES)
    tokens = oracle.greedy(prompt, steps)
    eos = reference.token_to_id(EOS_TOKEN)
    if eos in tokens:
        tokens = tokens[: tokens.index(eos) + 1]
    return prompt, tokens, reference.decode(tokens, skip_special_tokens=True)


def test_greedy_request_streams_the_oracle_output(native_cli, tiny, tmp_path):
    directory = tiny[0]
    prompt, tokens, text = _expected_greedy(tiny, 16)
    server = _serve(native_cli, _config(tmp_path / "c.json", directory, text_flush_ms=0), [_start("req-greedy")])
    receipt = server.streams.receipt("req-greedy")
    assert receipt["status"] == "completed", receipt
    assert receipt["output_text"] == text == server.streams.text("req-greedy")
    assert receipt["usage"] == {"prompt_tokens": len(prompt), "completion_tokens": len(tokens),
                                "total_tokens": len(prompt) + len(tokens)}
    assert receipt["finish_reason"] == ("stop" if len(tokens) < 16 else "length")
    deltas = [d for d in server.streams.by_request["req-greedy"] if d.get("kind") == "text_delta"]
    assert sum(d["generated_tokens_delta"] for d in deltas) <= len(tokens)
    assert receipt["execution"]["backend"] == "cpu" and receipt["metrics"]["maximum_batch_size"] == 1
    assert receipt["metrics"]["ttft_ms"] is not None and not server.errors


def test_text_is_coalesced_into_few_events(native_cli, tiny, tmp_path):
    server = _serve(native_cli, _config(tmp_path / "c.json", tiny[0], text_flush_ms=60_000), [_start("req-coalesce")])
    deltas = [d for d in server.streams.by_request["req-coalesce"] if d.get("kind") == "text_delta"]
    assert len(deltas) <= 1 and server.streams.text("req-coalesce") == server.streams.receipt("req-coalesce")["output_text"]


def test_seeded_sampling_is_reproducible_and_matches_generate(native_cli, tiny, tmp_path):
    directory, _, reference = tiny
    sampling = {"temperature": 1.5, "top_p": 0.95, "seed": -42, "stop": []}
    docs = [_start("req-seed-a", 24, sampling=sampling), _start("req-seed-b", 24, sampling=sampling)]
    server = _serve(native_cli, _config(tmp_path / "c.json", directory), docs)
    a, b = server.streams.receipt("req-seed-a"), server.streams.receipt("req-seed-b")
    assert a["output_text"] == b["output_text"]
    prompt = _prompt_ids(reference, MESSAGES)
    generated = subprocess.run(
        [str(native_cli), "generate", "--model", str(directory), "--backend", "cpu", "--tokens",
         ",".join(map(str, prompt)), "--max-new", "24", "--temperature", "1.5", "--top-p", "0.95",
         "--seed", str((-42) % (1 << 64))], capture_output=True, text=True, check=True)
    tokens = json.loads(generated.stdout)["tokens"]
    eos = reference.token_to_id(EOS_TOKEN)
    if eos in tokens:
        tokens = tokens[: tokens.index(eos) + 1]
    assert a["output_text"] == reference.decode(tokens, skip_special_tokens=True)
    assert a["usage"]["completion_tokens"] == len(tokens)


def test_stop_string_ends_the_output_before_it(native_cli, tiny, tmp_path):
    _, _, text = _expected_greedy(tiny, 16)
    if len(text) < 6:
        pytest.skip("greedy output too short to place a stop string")
    stop = text[len(text) // 2: len(text) // 2 + 2]
    doc = _start("req-stop", 16, sampling={"temperature": 0.0, "top_p": 1.0, "seed": 7, "stop": [stop]})
    server = _serve(native_cli, _config(tmp_path / "c.json", tiny[0], text_flush_ms=0), [doc])
    receipt = server.streams.receipt("req-stop")
    assert receipt["status"] == "completed" and receipt["finish_reason"] == "stop"
    assert receipt["output_text"] == text[: text.find(stop)] == server.streams.text("req-stop")


def test_unservable_requests_get_one_rejected_receipt(native_cli, tiny, tmp_path):
    now = int(time.time() * 1000)
    cases = {
        "req-unknown-model": (_start("req-unknown-model", model=dict(MODEL, revision="c" * 40)),
                              "unsupported-model-revision"),
        "req-old-runtime": (_start("req-old-runtime", runtime_revision="2" * 64), "runtime-revision-mismatch"),
        "req-expired": (_start("req-expired", created_at_unix_ms=now - 10_000, deadline_unix_ms=now - 1_000),
                        "deadline-exceeded"),
        "req-too-long": (_start("req-too-long", 400, objective={"context_tokens": 64}), "context-exceeded"),
        "req-small-memory": (_start("req-small-memory", memory={"max_resident_weight_bytes": 1}), "memory-limit"),
    }
    docs = [doc for doc, _ in cases.values()] + [_control("drain"), _start("req-after-drain")]
    server = _serve(native_cli, _config(tmp_path / "c.json", tiny[0]), docs)
    for request_id, (_, code) in list(cases.items()) + [("req-after-drain", (None, "service-draining"))]:
        stream = server.streams.by_request[request_id]
        assert len(stream) == 1, stream
        receipt = stream[0]
        assert receipt["status"] == "rejected" and receipt["terminal_sequence"] == 0
        assert receipt["failure"]["code"] == code and receipt["failure"]["stage"] == "not_started"


def test_invalid_documents_and_duplicates_produce_no_output(native_cli, tiny, tmp_path):
    server = Server(native_cli, _config(tmp_path / "c.json", tiny[0], text_flush_ms=0))
    invalid = sorted((KIT / "fixtures" / "invalid").glob("*.json"))
    for path in invalid:
        server.send(path.read_bytes().replace(b"\n", b" "))
    server.send(b"not json")
    for path in (KIT / "fixtures" / "valid").glob("event-*.json"):
        server.send(path.read_bytes().replace(b"\n", b" "))  # events are not accepted as input
    long = _start("req-dup", 3000)
    server.send(long)
    server.read_until(lambda d: d.get("kind") == "accepted")
    server.send(long)  # still in flight
    server.send(_control("cancel", "req-dup"))
    receipt = server.receipt("req-dup")
    assert server.close() == 0
    assert receipt["status"] in ("cancelled", "completed")
    assert len(server.errors) == len(invalid) + 1 + 2 + 1, server.errors
    assert any("already in flight" in e for e in server.errors)
    assert set(server.streams.by_request) == {"req-dup"}


def test_cancel_queued_and_running_requests(native_cli, tiny, tmp_path):
    server = Server(native_cli, _config(tmp_path / "c.json", tiny[0], text_flush_ms=0, queue_limit=4))
    server.send(_start("req-running", 4000))
    server.send(_start("req-queued", 8))
    server.read_until(lambda d: d["request_id"] == "req-running" and d.get("kind") == "text_delta")
    server.send(_control("cancel", "req-queued"))
    server.send(_control("cancel", "req-running"))
    server.send(_control("cancel", "req-never-sent"))  # unknown targets are ignored
    assert server.close() == 0
    queued = server.streams.receipt("req-queued")
    assert queued["status"] == "cancelled" and queued["failure"]["stage"] == "not_started"
    assert [d.get("kind") for d in server.streams.by_request["req-queued"][:-1]] == ["accepted", "queue_state"]
    running = server.streams.receipt("req-running")
    assert running["status"] == "cancelled" and running["failure"]["stage"] == "partial_output"
    assert running["output_text"] == server.streams.text("req-running")
    assert 0 < running["usage"]["completion_tokens"] < 4000
    # Cancelling after the receipt is a no-op too.
    again = Server(native_cli, _config(tmp_path / "c2.json", tiny[0]))
    again.send(_start("req-done"))
    again.receipt("req-done")
    again.send(_control("cancel", "req-done"))
    assert again.close() == 0 and len(again.streams.by_request["req-done"]) >= 2 and not again.errors


def test_a_full_queue_rejects_as_retryable(native_cli, tiny, tmp_path):
    server = Server(native_cli, _config(tmp_path / "c.json", tiny[0], text_flush_ms=0, queue_limit=0))
    server.send(_start("req-busy", 4000))
    server.read_until(lambda d: d["request_id"] == "req-busy" and d.get("kind") == "text_delta")
    server.send(_start("req-overflow"))
    rejected = server.receipt("req-overflow")
    server.send(_control("cancel", "req-busy"))
    assert server.close() == 0
    assert rejected["failure"]["code"] == "overloaded" and rejected["failure"]["retryable"] is True


def test_deadline_stops_a_running_request(native_cli, tiny, tmp_path):
    now = int(time.time() * 1000)
    doc = _start("req-deadline", 4000, created_at_unix_ms=now, deadline_unix_ms=now + 1500)
    server = _serve(native_cli, _config(tmp_path / "c.json", tiny[0], text_flush_ms=0), [doc])
    receipt = server.streams.receipt("req-deadline")
    if receipt["status"] == "completed":
        pytest.skip("the fixture finished 4000 tokens inside the deadline on this machine")
    assert receipt["status"] == "failed" and receipt["finish_reason"] == "error"
    assert receipt["failure"]["code"] == "deadline-exceeded"
    assert receipt["failure"]["stage"] in ("running", "partial_output")


ENVELOPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t)


@pytest.fixture(scope="module")
def lib(native_cli):
    name = "liblokahi.dylib" if sys.platform == "darwin" else "liblokahi.so"
    library = ctypes.CDLL(str(Path(native_cli).parent / name))
    library.lokahi_last_error.restype = ctypes.c_char_p
    library.lokahi_service_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    library.lokahi_service_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ENVELOPE,
                                             ctypes.c_void_p]
    library.lokahi_service_control.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.lokahi_service_free.argtypes = [ctypes.c_void_p]
    return library


def test_c_abi_service_streams_and_frees_with_receipts(lib, tiny, tmp_path):
    directory = tiny[0]
    _, _, text = _expected_greedy(tiny, 16)
    config = _config(tmp_path / "c.json", directory, text_flush_ms=0).read_bytes()
    streams = Streams()
    lock = threading.Lock()
    done = threading.Event()

    @ENVELOPE
    def sink(context, data, length):
        with lock:
            doc = streams.add(ctypes.string_at(data, length))
        if doc["message_type"] == "receipt" and doc["request_id"] == "req-abi":
            done.set()

    service = ctypes.c_void_p()
    assert lib.lokahi_service_create(config, len(config), ctypes.byref(service)) == 0, lib.lokahi_last_error()
    try:
        start = json.dumps(_start("req-abi")).encode()
        assert lib.lokahi_service_start(service, start, len(start), sink, None) == 0, lib.lokahi_last_error()
        assert done.wait(60)
        assert streams.receipt("req-abi")["output_text"] == text
        bad = b'{"schema_version": 1}'
        assert lib.lokahi_service_start(service, bad, len(bad), sink, None) != 0
        assert lib.lokahi_service_control(service, bad, len(bad)) != 0
        long = json.dumps(_start("req-abi-long", 4000)).encode()
        assert lib.lokahi_service_start(service, long, len(long), sink, None) == 0
    finally:
        lib.lokahi_service_free(service)  # cancels the long request and delivers its receipt first
    receipt = streams.receipt("req-abi-long")
    assert receipt["status"] in ("cancelled", "completed")
