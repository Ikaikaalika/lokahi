"""Correctness evidence: the native wire v1 codec against the kit's reference verifier.

``lokahi wire-check`` verdicts are compared, document by document, with
``verify_kit.load_document`` followed by ``validate_document``. The comparison
covers the golden fixtures, a seeded corpus of mutations of them, and round
trips through the native encoder. By design (``native/engine/src/wire.h``),
the native codec rejects two classes of document that the reference accepts:
documents that are not UTF-8, and integers outside int64. Every disagreement
must fall in one of those classes. That the Swift decoders accept the encoded
output is argued in ``wire.cpp``; no Swift toolchain runs here.
"""
from __future__ import annotations

import copy
import json
import math
import random
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "integrations/commoncompute/lokahi-wire-v1"
if str(KIT / "tools") not in sys.path:
    sys.path.insert(0, str(KIT / "tools"))
import verify_kit  # noqa: E402

BOM = b"\xef\xbb\xbf"
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
LIMITS = verify_kit.MESSAGE_LIMITS
MARK = "MARKER"
ABSENT = object()  # a member to leave out


# --- Reference and native verdicts -------------------------------------------


class _Raw:
    """Stands in for the Path that verify_kit.load_document reads."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read_bytes(self) -> bytes:
        return self._raw


def reference(raw: bytes) -> str | None:
    """The reference's message type for a document, or None when it rejects it.

    Some rejections surface as TypeError, OverflowError, ValueError or
    RecursionError rather than ContractError (an unhashable kind, a
    4301-digit integer); they are rejections all the same.
    """
    try:
        value, raw = verify_kit.load_document(_Raw(raw))
        return verify_kit.validate_document(value, raw)
    except Exception:  # noqa: BLE001 - any exception is a rejection
        return None


def _run(cli: Path, args: list[str], stdin: bytes | None = None) -> list[dict]:
    result = subprocess.run([str(cli), "wire-check", *args], input=stdin, capture_output=True, check=False)
    assert result.returncode in (0, 1), result.stderr.decode(errors="replace")
    # Only "\n" ends a line: encoded documents may hold U+2028 or U+0085, which splitlines() also splits on.
    return [json.loads(line) for line in result.stdout.split(b"\n")[:-1]]


def native(cli: Path, docs: list[bytes], tmp: Path, *flags: str) -> list[dict]:
    """One verdict per document, in one process for all single-line documents.
    Documents that contain a newline go through files."""
    verdicts: list[dict | None] = [None] * len(docs)
    lines = [i for i, doc in enumerate(docs) if b"\n" not in doc]
    if lines:
        out = _run(cli, [*flags, "--stdin"], b"".join(docs[i] + b"\n" for i in lines))
        assert len(out) == len(lines)
        for i, verdict in zip(lines, out):
            verdicts[i] = verdict
    files = [i for i, doc in enumerate(docs) if b"\n" in doc]
    for start in range(0, len(files), 256):
        chunk = files[start : start + 256]
        paths = []
        for i in chunk:
            path = tmp / f"doc-{i}.json"
            path.write_bytes(docs[i])
            paths.append(str(path))
        out = _run(cli, [*flags, *paths])
        assert len(out) == len(chunk)
        for i, verdict in zip(chunk, out):
            verdicts[i] = verdict
    return verdicts  # type: ignore[return-value]


def _verdict(verdict: dict) -> str | None:
    return verdict["type"] if verdict["valid"] else None


def _integers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _integers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _integers(item)


def by_design(raw: bytes, verdict: dict) -> str | None:
    """Why the native codec rejects a document the reference accepts, when that
    is one of the two deliberate differences."""
    if json.detect_encoding(raw) not in ("utf-8", "utf-8-sig"):
        return "not-utf8"
    try:
        raw.removeprefix(BOM).decode("utf-8")
    except UnicodeDecodeError:
        return "not-utf8"
    if any(not INT64_MIN <= n <= INT64_MAX for n in _integers(json.loads(raw))) and "int64" in verdict["error"]:
        return "beyond-int64"
    return None


# --- Documents ---------------------------------------------------------------


class Raw(str):
    """Text written into a document verbatim: a number lexeme, an escaped name."""


class Pairs(list):
    """An object written member by member, so that names may repeat."""


def _utf16(text: str):
    for char in text:
        code = ord(char)
        if code > 0xFFFF:
            code -= 0x10000
            yield 0xD800 + (code >> 10)
            yield 0xDC00 + (code & 0x3FF)
        else:
            yield code


def _surrogates(text: str) -> bool:
    return any("\ud800" <= char <= "\udfff" for char in text)


def escaped(text: str) -> Raw:
    """`text` as a JSON string with every character written as \\uXXXX."""
    return Raw('"' + "".join(f"\\u{unit:04x}" for unit in _utf16(text)) + '"')


def dump(value, *, ws: str = "", ascii: bool = False, escape_all: bool = False) -> bytes:
    """JSON text with `ws` around every token. Objects may be dicts or Pairs;
    Raw values and names are written as they are."""

    def string(text: str) -> str:
        if isinstance(text, Raw):
            return text
        if escape_all:
            return escaped(text)
        return json.dumps(text, ensure_ascii=ascii or _surrogates(text))

    def encode(item) -> str:
        if isinstance(item, str):
            return item if isinstance(item, Raw) else string(item)
        if isinstance(item, (dict, Pairs)):
            members = item.items() if isinstance(item, dict) else item
            body = f"{ws},{ws}".join(f"{string(k)}{ws}:{ws}{encode(v)}" for k, v in members)
            return "{" + ws + body + ws + "}"
        if isinstance(item, list):
            return "[" + ws + f"{ws},{ws}".join(encode(v) for v in item) + ws + "]"
        return json.dumps(item)

    return (ws + encode(value) + ws).encode("utf-8")


def paths(value, prefix=()):
    """Every path below `value`, parents before children."""
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in items:
        yield prefix + (key,)
        yield from paths(item, prefix + (key,))


def get(value, path):
    for key in path:
        value = value[key]
    return value


def replaced(doc, path, new):
    out = copy.deepcopy(doc)
    if new is ABSENT:
        del get(out, path[:-1])[path[-1]]
    else:
        get(out, path[:-1])[path[-1]] = new
    return out


def renamed(doc, path, name):
    out = copy.deepcopy(doc)
    parent = get(out, path[:-1])
    members = [(name if key == path[-1] else key, value) for key, value in parent.items()]
    parent.clear()
    parent.update(members)
    return out


def with_pairs(doc, path, pairs):
    if not path:
        return Pairs(pairs)
    out = copy.deepcopy(doc)
    get(out, path[:-1])[path[-1]] = Pairs(pairs)
    return out


def _name(path) -> str:
    """The schema name of a path's last step; list items are `name[]`."""
    return path[-1] if isinstance(path[-1], str) else f"{path[-2]}[]"


def _dedupe(values: list) -> list:
    """Order-preserving, and 1, 1.0 and True stay distinct."""
    seen, out = set(), []
    for value in values:
        key = (type(value).__name__, repr(value))
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


FIXTURES = {path.stem: json.loads(path.read_bytes()) for path in sorted((KIT / "fixtures/valid").glob("*.json"))}


def _bases() -> dict[str, dict]:
    """The valid fixtures plus valid variants that reach every optional field and status."""
    docs = copy.deepcopy(FIXTURES)
    accepted, start, receipt = docs["event-accepted"], docs["start-greedy"], docs["receipt-completed"]
    docs["event-progress"] = {**accepted, "sequence": 2, "kind": "progress", "progress": 0.5, "message": "prefill"}
    docs["event-queue"] = {**accepted, "sequence": 3, "kind": "queue_state", "queue_position": 0, "queue_depth": 4}
    del docs["event-queue"]["message"]
    docs["event-warning"] = {**accepted, "sequence": 4, "kind": "warning", "message": "thermal pressure"}
    minimal = copy.deepcopy(start)
    minimal["messages"] = [{"role": "user", "content": "Aloha"}, {"role": "assistant", "content": "Aloha kākou"}]
    minimal["sampling"] = {"temperature": 0.7, "top_p": 0.9, "seed": None, "stop": ["\n\n", "</s>", "Ω"]}
    del minimal["objective"]["target_ttft_ms"], minimal["objective"]["min_decode_tokens_per_second"]
    del minimal["memory"]["max_resident_weight_bytes"]
    minimal["objective"]["workload_class"] = "background"
    docs["start-minimal"] = minimal
    cancelled = copy.deepcopy(receipt)
    cancelled.update(status="cancelled", finish_reason="cancelled", output_text="partial")
    cancelled["metrics"].update(inter_token_latency_ms=12.5, decode_tokens_per_second=80.0)
    cancelled["execution"].update(
        backend="metal", route="native_phase_program", evidence_ids=["e-1", "e:2"],
        fallback_used=True, fallback_reason="resident plan unavailable",
    )
    cancelled["failure"] = {"code": "client-cancelled", "stage": "partial_output", "retryable": False,
                            "message": "Cancelled by the provider."}
    docs["receipt-cancelled"] = cancelled
    docs["receipt-length"] = {**copy.deepcopy(receipt), "finish_reason": "length", "output_text": ""}
    failed = copy.deepcopy(docs["receipt-failed"])
    failed.update(status="failed", finish_reason="error")
    failed["failure"]["stage"] = "running"
    docs["receipt-error"] = failed
    return docs


BASES = _bases()

TYPE_VALUES = [None, True, False, 0, 1, -1, 2, 1.0, 0.5, -0.5, "", "x", "1", "true", [], [1], ["x"], {}, {"x": 1}]
INTEGER_FIELDS = {  # name: (minimum, maximum)
    "schema_version": (1, 1), "batch_size": (1, 1), "created_at_unix_ms": (0, None), "deadline_unix_ms": (1, None),
    "context_tokens": (1, 131072), "max_output_tokens": (1, 8192), "seed": (INT64_MIN, INT64_MAX),
    "max_runtime_memory_bytes": (268435456, None), "max_resident_weight_bytes": (1, None),
    "issued_at_unix_ms": (0, None), "sequence": (0, None), "emitted_at_unix_ms": (0, None),
    "generated_tokens_delta": (0, 8192), "queue_position": (0, None), "queue_depth": (0, None),
    "terminal_sequence": (0, None), "completed_at_unix_ms": (0, None), "prompt_tokens": (0, None),
    "completion_tokens": (0, None), "total_tokens": (0, None), "peak_memory_bytes": (0, None),
    "maximum_batch_size": (1, None),
}
NUMBER_FIELDS = {  # name: (minimum, maximum, exclusive minimum)
    "temperature": (0, 2, False), "top_p": (0, 1, True), "target_ttft_ms": (0, None, True),
    "min_decode_tokens_per_second": (0, None, True), "progress": (0, 1, False), "queue_ms": (0, None, False),
    "ttft_ms": (0, None, False), "inter_token_latency_ms": (0, None, False),
    "decode_tokens_per_second": (0, None, True),
}
TEXT_FIELDS = {  # name: (minimum, maximum) code points
    "quantization": (1, 64), "content": (1, 65536), "stop[]": (1, 128), "text_delta": (1, 65536),
    "message": (1, 1024), "output_text": (0, 1048576), "fallback_reason": (1, 512),
}
IDENTIFIERS = ["a", "Z", "9", "a" * 128, "a" * 129, "", ".a", "_a", ":a", "-a", "a.b_c:d-e", "a/b", "a b", "a\n",
               "a\t", "\na", "a\u0000", "é", "aé", "\uff21", "a" * 127 + "é", "a\u2028", "req-ok"]
HEX = ["0" * 40, "f" * 64, "0" * 39, "0" * 41, "0" * 63, "0" * 65, "A" * 40, "g" * 40, "0" * 40 + "\n",
       " " + "0" * 40, "\uff10" * 40, "", "0" * 128]
DIGESTS = ["sha256:" + "0" * 64, "SHA256:" + "0" * 64, "sha256" + "0" * 64, "sha256:" + "0" * 63,
           "sha256:" + "0" * 65, "sha256:" + "A" * 64, "sha512:" + "0" * 64, "sha256::" + "0" * 63,
           "sha256:" + "0" * 64 + "\n", "sha256: " + "0" * 63]
ADAPTERS = ["a", "a" * 64, "a" * 65, "A", "0abc", ".a", "_a", "a.b", "a_b", "a-b", "a:b", "a b", "aB", "é", ""]
PATTERNS = {
    "revision": HEX, "runtime_revision": HEX, "artifact_digest": DIGESTS, "adapter_id": ADAPTERS,
    **{name: IDENTIFIERS for name in ("request_id", "task_id", "control_id", "target_request_id", "plan_id",
                                      "code", "id", "evidence_ids[]")},
}
ENUMS = {
    "message_type": ["start", "control", "event", "receipt"], "operation": ["llm.generate"],
    "role": ["system", "user", "assistant"], "workload_class": ["interactive", "throughput", "background"],
    "ssd_offload": ["disabled"], "deployment_mode": ["supported"], "action": ["cancel", "drain"],
    "kind": ["accepted", "progress", "text_delta", "queue_state", "warning"],
    "status": ["completed", "failed", "cancelled", "rejected"],
    "finish_reason": ["stop", "length", "cancelled", "error", "rejected"],
    "backend": ["mlx", "metal", "coreml", "ane", "cpu", "hybrid"],
    "route": ["batch_generator", "direct_single_sequence", "native_phase_program"],
    "stage": ["not_started", "running", "partial_output"],
}
FAILURE = {"code": "oom", "stage": "not_started", "retryable": True, "message": "Out of memory."}
OPTIONAL_MEMBERS = {
    "message": "m", "progress": 0.5, "text_delta": "t", "generated_tokens_delta": 1, "queue_position": 1,
    "queue_depth": 2, "target_request_id": "req-1", "output_text": "o", "failure": FAILURE,
    "fallback_reason": "why", "max_resident_weight_bytes": 1 << 30, "target_ttft_ms": 100.0,
    "min_decode_tokens_per_second": 1.0,
}
UNKNOWN_MEMBERS = [("x", 1), ("", None), ("tools", [{"type": "function"}]), ("Schema_version", 1), ("\u0000", 0),
                   ("é", "é")]
EVENT_SAMPLES = {"progress": 0.25, "text_delta": "hi", "generated_tokens_delta": 2, "queue_position": 0,
                 "queue_depth": 1, "message": "m"}
ALPHABETS = [("a", False), ("é", False), ("é", True), ("世", False), ("😀", False), ("😀", True),
             ("e\u0301", False), ("\ud800", True), ("\udfff", True)]
BIG_ALPHABETS = [("a", False), ("é", False), ("😀", False), ("😀", True), ("\ud800", True)]
# Bytes injected into a string: surrogates in UTF-8 (CESU), malformed UTF-8,
# edge code points, raw control characters, and valid and invalid escapes.
RAW_BYTES = [
    b"\xed\xa0\x80", b"\xed\xa0\xbd\xed\xb8\x80", b"\xed\xbf\xbf", b"\xc0\xaf", b"\xe0\x80\xaf", b"\xf0\x80\x80\xaf",
    b"\xff", b"\xfe", b"\x80", b"\xe4\xb8", b"\xf4\x90\x80\x80", b"\xf8\x88\x80\x80\x80", b"\xe9", b"\xed\x9f\xbf",
    b"\xee\x80\x80", b"\xef\xbf\xbf", b"\xf4\x8f\xbf\xbf", BOM, b"\xc2\x85", b"\xe2\x80\xa8", b"\x00", b"\x01",
    b"\x1f", b"\x7f", b"\t", b"\n", b"\r", b"\\", b'"', b"\\u0000", b"\\ud800", b"\\udc00", b"\\ud800\\udc00",
    b"\\udc00\\ud800", b"\\ud800\\u0041", b"\\ud800\\ud800", b"\\uD83D\\uDE00", b"\\ud83d\\u", b"\\u12",
    b"\\u12G4", b"\\x41", b"\\U0001F600", b"\\'", b"\\a", b"\\0", b"\\/", b"\\b\\f\\n\\r\\t",
]
FREE_TEXT = [("messages", 0, "content"), ("message",), ("text_delta",), ("output_text",), ("failure", "message"),
             ("request_id",), ("control_id",)]


def _repeat(alphabet: str, count: int) -> str:
    """`count` code points of `alphabet`, which may be two code points long."""
    if len(alphabet) == 1:
        return alphabet * count
    return alphabet * (count // 2) + alphabet[0] * (count % 2)


def _integer_edges(low: int, high: int | None) -> list:
    values = [low - 1, low, low + 1, 0, -1, 2**31, 2**53 + 1, INT64_MAX - 1, INT64_MAX, INT64_MAX + 1, INT64_MIN,
              INT64_MIN - 1, 2**64, 10**30]
    if high is not None:
        values += [high - 1, high, high + 1]
    values += [float(low), Raw(f"{low}e0"), Raw(f"{low}.0"), Raw(f"{low}.0000000000000000001"), Raw("-0"),
               Raw("1" * 4301), Raw("NaN"), Raw("Infinity"), True, False, str(low), None]
    return _dedupe(values)


def _number_edges(low: float, high: float | None, exclusive: bool) -> list:
    values = [low, float(low), math.nextafter(float(low), -math.inf), math.nextafter(float(low), math.inf), low - 1,
              -0.0, Raw("-0"), 1e-300, 5e-324, 1e308, sys.float_info.max, 10**308, 10**309, 2**64,
              Raw(str(2**1024 - 2**970)), Raw(str(2**1024 - 2**970 - 1)), Raw("1e400"), Raw("-1e400"),
              Raw("1e-400"), Raw("NaN"), Raw("Infinity"), Raw("-Infinity"), True, False, "0.5", None]
    if high is not None:
        values += [high, float(high), math.nextafter(float(high), math.inf), math.nextafter(float(high), -math.inf),
                   high + 1, Raw(f"{high}.0000000000000000001"), Raw(f"{high}E0"), Raw(f"0.{high}e1")]
    if high == 1:  # the halfway point between 1 and the next double rounds to even, which is 1
        values += [Raw("1.00000000000000011102230246251565404236316680908203125"),
                   Raw("1.000000000000000111022302462515654042363166809082031251")]
    return _dedupe(values)


def _structure(doc):
    for path in paths(doc):
        yield "drop", dump(replaced(doc, path, ABSENT))
        for value in TYPE_VALUES:
            yield "retype", dump(replaced(doc, path, value))
        if isinstance(path[-1], str):
            key = path[-1]
            for name in _dedupe([key.upper(), key[:1].upper() + key[1:], key + "_", " " + key, key[:-1],
                                 key.replace("_", "-"), key.replace("e", "\u0435"), escaped(key)]):
                if name != key:
                    yield "rename", dump(renamed(doc, path, name))
    for path in [(), *(p for p in paths(doc) if isinstance(get(doc, p), dict))]:
        members = get(doc, path)
        for key, value in [*UNKNOWN_MEMBERS, *OPTIONAL_MEMBERS.items()]:
            if key not in members:
                yield ("nested-key" if path else "add-key"), dump(with_pairs(doc, path, [*members.items(), (key, value)]))
        pairs = list(members.items())
        for i, (key, value) in enumerate(pairs):
            yield "duplicate", dump(with_pairs(doc, path, pairs[: i + 1] + [(key, value)] + pairs[i + 1 :]))
            yield "duplicate", dump(with_pairs(doc, path, pairs + [(key, TYPE_VALUES[i % len(TYPE_VALUES)])]))
            yield "duplicate", dump(with_pairs(doc, path, pairs + [(escaped(key), value)]))
            yield "escaped-name", dump(with_pairs(doc, path, pairs[:i] + [(escaped(key), value)] + pairs[i + 1 :]))


def _values(doc, big_seen: set):
    for path in paths(doc):
        name = _name(path)
        if name in INTEGER_FIELDS:
            for value in _integer_edges(*INTEGER_FIELDS[name]):
                yield "integer-edge", dump(replaced(doc, path, value))
        elif name in NUMBER_FIELDS:
            for value in _number_edges(*NUMBER_FIELDS[name]):
                yield "number-edge", dump(replaced(doc, path, value))
        if name in PATTERNS:
            for value in PATTERNS[name]:
                yield "pattern", dump(replaced(doc, path, value))
        if name in ENUMS:
            current = get(doc, path)
            for value in ENUMS[name]:
                yield "enum", dump(replaced(doc, path, value))
            for value in _dedupe([current.upper(), current.capitalize(), current + " ", " " + current,
                                  current + "s", current[:-1], current.replace("a", "\u0430"), current + "\u0000",
                                  current + "\u200b", "\ufeff" + current, escaped(current)]):
                yield "enum-near-miss", dump(replaced(doc, path, value))
        if name in TEXT_FIELDS:
            low, high = TEXT_FIELDS[name]
            big = high >= 65536
            if big and (doc["message_type"], name) in big_seen:
                continue
            big_seen.add((doc["message_type"], name))
            for count in (low - 1, low, low + 1, high - 1, high, high + 1):
                for alphabet, ascii in BIG_ALPHABETS if big else ALPHABETS:
                    if count >= 0:
                        raw = dump(replaced(doc, path, _repeat(alphabet, count)), ascii=ascii)
                        if len(raw) <= 3 * 2**20:
                            yield "length", raw
            yield "nul", dump(replaced(doc, path, "a\u0000b"))


def _fill(doc, path, target: int, alphabet: str) -> bytes:
    """`doc` with the string at `path` sized so the document is `target` bytes."""
    need = target - len(dump(replaced(doc, path, "")))
    width = len(alphabet.encode())
    return dump(replaced(doc, path, alphabet * (need // width) + "a" * (need % width)))


def _fill_start(target: int) -> bytes:
    """A start of exactly `target` bytes, spread over messages of four-byte code
    points so no message or the combined content exceeds its limit."""
    empty = {"role": "user", "content": ""}
    per_message = len(dump(empty)) + 1  # with its comma
    need = target - len(dump({**BASES["start-greedy"], "messages": [empty]}))
    contents = []
    while need > 200000 + per_message:
        contents.append("😀" * 50000)
        need -= 200000 + per_message
    contents.append("😀" * (need // 4) + "a" * (need % 4))
    raw = dump({**BASES["start-greedy"], "messages": [{"role": "user", "content": c} for c in contents]})
    assert len(raw) == target
    return raw


def _sizes():
    for name in ("start-greedy", "control-cancel", "event-text-delta", "receipt-completed"):
        raw = dump(BASES[name])
        limit = LIMITS[BASES[name]["message_type"]]
        for target in (limit, limit + 1):
            pad = b" " * (target - len(raw))
            yield "size", raw + pad
            yield "size", pad + raw
            yield "size", BOM + raw + pad[3:]  # the byte order mark counts toward the limit
        yield "size", raw + b" " * (2 * 2**20 + 1 - len(raw))
    for target in (LIMITS["event"], LIMITS["event"] + 1):
        yield "size-content", _fill(BASES["event-text-delta"], ("text_delta",), target, "é")
    for target in (LIMITS["receipt"], LIMITS["receipt"] + 1):
        yield "size-content", _fill(BASES["receipt-completed"], ("output_text",), target, "é")
    for target in (LIMITS["start"], LIMITS["start"] + 1):
        yield "size-content", _fill_start(target)
    start = BASES["start-greedy"]
    for contents in (["a" * 65536] * 8, ["a" * 65536] * 8 + ["b"], ["a" * 65536] * 7 + ["é" * 65535, "x"],
                     ["a" * 65536] * 7 + ["é" * 65536, "x"], ["é" * 65536] * 8):
        doc = {**start, "messages": [{"role": "user", "content": c} for c in contents]}
        yield "combined-content", dump(doc)
        yield "combined-content", dump(doc, ascii=True)


def _cross_field():
    for name in ("start-greedy", "start-minimal"):
        start = BASES[name]
        for created in (0, 1, 1787630400000, 2**62, INT64_MAX - 86400000, INT64_MAX - 1, INT64_MAX):
            for delta in (-1, 0, 1, 86399999, 86400000, 86400001, 2**63):
                yield "deadline", dump({**start, "created_at_unix_ms": created, "deadline_unix_ms": created + delta})
        yield "deadline", dump({**start, "created_at_unix_ms": 0, "deadline_unix_ms": 0})
        for runtime in (268435455, 268435456, 268435457, 4294967296, INT64_MAX, INT64_MAX + 1):
            for resident in (ABSENT, 0, 1, runtime - 1, runtime, runtime + 1, INT64_MAX):
                memory = {"max_runtime_memory_bytes": runtime, "ssd_offload": "disabled"}
                if resident is not ABSENT:
                    memory["max_resident_weight_bytes"] = resident
                yield "memory", dump({**start, "memory": memory})
        for target, rate in ((ABSENT, 25.0), (750.0, ABSENT), (0, 1.0), (0.0, 1.0), (5e-324, 5e-324), (-1.0, 1.0),
                             (None, 1.0), ("1", 1.0), (True, 1.0), (1, 1)):
            objective = {k: v for k, v in start["objective"].items() if k not in ("target_ttft_ms", "min_decode_tokens_per_second")}
            if target is not ABSENT:
                objective["target_ttft_ms"] = target
            if rate is not ABSENT:
                objective["min_decode_tokens_per_second"] = rate
            yield "objective", dump({**start, "objective": objective})
        for stop in ([], ["a"], ["a", "a"], ["a", "b", "c", "d"], ["a", "b", "c", "d", "e"], ["é", "e\u0301"],
                     ["\ud800", "\ud800"], ["\ud83d", "\ude00"], [escaped("😀"), "😀"], [""], ["a" * 128],
                     ["a" * 129], [1], [None], ["a", "A"]):
            yield "stop", dump(replaced(start, ("sampling", "stop"), stop))
    for name in ("receipt-completed", "receipt-cancelled"):
        receipt = BASES[name]
        for prompt, completion, total in ((18, 1, 19), (18, 1, 18), (0, 0, 0), (0, 0, 1), (INT64_MAX, 0, INT64_MAX),
                                          (INT64_MAX, 1, INT64_MAX + 1), (INT64_MAX, 1, INT64_MIN),
                                          (2**62, 2**62, 2**63), (2**62, 2**62 - 1, INT64_MAX),
                                          (INT64_MAX, INT64_MAX, 2**64 - 2), (5, -1, 4), (-1, 5, 4)):
            usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
            yield "usage", dump({**receipt, "usage": usage})
        for used in (True, False, 1, 0, "true", None):
            for reason in (ABSENT, "x", "", None, "r" * 512, "r" * 513, "é" * 512, "é" * 513):
                execution = {k: v for k, v in receipt["execution"].items() if k != "fallback_reason"}
                execution["fallback_used"] = used
                if reason is not ABSENT:
                    execution["fallback_reason"] = reason
                yield "fallback", dump({**receipt, "execution": execution})
        for ids in ([], ["a"] * 2, [f"e{i}" for i in range(16)], [f"e{i}" for i in range(17)], ["a", "A"], ["a b"]):
            yield "evidence", dump(replaced(receipt, ("execution", "evidence_ids"), ids))
    for name in ("control-cancel", "control-drain"):
        for action in ("cancel", "drain", "pause", None):
            for target in (ABSENT, "req-1", "bad id", None, ""):
                doc = {k: v for k, v in BASES[name].items() if k != "target_request_id"}
                doc["action"] = action
                if target is not ABSENT:
                    doc["target_request_id"] = target
                yield "control", dump(doc)
    accepted = {k: v for k, v in BASES["event-accepted"].items() if k != "message"}
    for kind in [*ENUMS["kind"], "bogus"]:
        for mask in range(64):
            fields = {k: v for i, (k, v) in enumerate(EVENT_SAMPLES.items()) if mask >> i & 1}
            yield "event-kind", dump({**accepted, "kind": kind, **fields})
        for field in EVENT_SAMPLES:
            yield "event-null", dump({**accepted, "kind": kind, field: None})
    completed = {k: v for k, v in BASES["receipt-completed"].items() if k != "output_text"}
    for status in [*ENUMS["status"], "done"]:
        for finish in [*ENUMS["finish_reason"], "eos"]:
            for output in (ABSENT, "", "ok", None):
                for failure in (ABSENT, None, *({**FAILURE, "stage": s} for s in ENUMS["stage"])):
                    doc = {**completed, "status": status, "finish_reason": finish}
                    if output is not ABSENT:
                        doc["output_text"] = output
                    if failure is not ABSENT:
                        doc["failure"] = failure
                    yield "receipt-status", dump(doc)


def _formats(doc, rng: random.Random):
    yield "format", json.dumps(doc).encode()
    yield "format", json.dumps(doc, indent=2).encode()
    yield "format", json.dumps(doc, indent="\t", sort_keys=True).encode()
    yield "format", json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    for ws in (" ", "\t", "\n", "\r", "\r\n", " \t\n\r "):
        yield "whitespace", dump(doc, ws=ws)
    for ws in ("\x0b", "\x0c", "\xa0", "\u2028", "\u3000", "\x00", "\ufeff"):
        yield "bad-whitespace", dump(doc, ws=ws)
    yield "escaped", dump(doc, escape_all=True)
    yield "escaped", dump(doc, ascii=True)

    def shuffle(value):
        if isinstance(value, dict):
            items = list(value.items())
            rng.shuffle(items)
            return {k: shuffle(v) for k, v in items}
        return [shuffle(v) for v in value] if isinstance(value, list) else value

    for _ in range(3):
        yield "key-order", dump(shuffle(doc))


def _encodings(doc):
    raw = dump(doc)
    yield "bom", BOM + raw
    yield "bom", BOM + BOM + raw
    for codec in ("utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"):
        yield "utf16-utf32", raw.decode("utf-8").encode(codec)
    path = next(p for p in FREE_TEXT if p[0] in doc and (len(p) == 1 or p[1] in doc[p[0]] or isinstance(p[1], int)))
    marked = dump(replaced(doc, path, f"x{MARK}y"))
    for injected in RAW_BYTES:
        yield "raw-bytes", marked.replace(MARK.encode(), injected)


def _syntax(doc):
    raw = dump(doc)
    for cut in sorted({1, 2, len(raw) // 3, len(raw) // 2, len(raw) - 2, len(raw) - 1}):
        yield "truncated", raw[:cut]
    yield "syntax", raw[:-1] + b",}"
    yield "syntax", raw.replace(b"[]", b"[,]", 1) if b"[]" in raw else raw.replace(b"]", b",]", 1)
    yield "syntax", raw.replace(b",", b"", 1)
    yield "syntax", raw.replace(b":", b" ", 1)
    yield "syntax", raw.replace(b'"', b"'")
    yield "syntax", b"/*c*/" + raw
    yield "syntax", raw + b"//c"
    yield "syntax", raw.replace(b'"schema_version"', b"schema_version")
    for extra in (b"{}", b"x", b"}", b"\x00", b" {}"):
        yield "extra-data", raw + extra
    for top in (b"[]", b"[" + raw + b"]", b'"x"', b"1", b"null", b"true", b"", b" ", b"{}"):
        yield "top-level", top
    for lexeme in (b"01", b"-01", b"+1", b"1.", b".5", b"1e", b"1e+", b"-", b"0x1F", b"1_000", b"-NaN", b"nan",
                   b"+Infinity", b"inf", b"Infinity", b"-Infinity", b"NaN", b"1E400", b"2.", b"1.5.3", b"0e", b"00",
                   b"1e0", b"10E-1", b"0.1e1", b"1.0", b"-0", b"1 ", b" 1", b"\t1"):
        yield "lexeme", raw.replace(b'"schema_version":1', b'"schema_version":' + lexeme, 1)
    for depth in (4, 17, 1000, 100000):
        yield "nesting", dump(Pairs([*doc.items(), ("x", Raw("[" * depth + "]" * depth))]))
        yield "nesting", raw.replace(b'"schema_version":1', b'"schema_version":' + b"[" * depth + b"]" * depth, 1)
    yield "many-values", dump(Pairs([*doc.items(), ("x", Raw("[" + ",".join(["0"] * 2000) + "]"))]))


def _random(rng: random.Random, count: int):
    names = list(BASES)
    keys = _dedupe([p[-1] for doc in BASES.values() for p in paths(doc) if isinstance(p[-1], str)]
                   + list(OPTIONAL_MEMBERS) + ["x", ""])
    values = TYPE_VALUES + [INT64_MAX, INT64_MAX + 1, INT64_MIN, INT64_MIN - 1, 2**64, 1e308, 5e-324, -0.0,
                            Raw("NaN"), Raw("1e400"), Raw("-0"), "a" * 129, "é" * 1025, "\ud800", "req-1", FAILURE,
                            *ENUMS["kind"], *ENUMS["status"], *ENUMS["stage"], "sha256:" + "0" * 64, "0" * 40]
    styles = [{}, {"ws": " "}, {"ascii": True}, {"escape_all": True}, {"ws": "\n"}]

    def pick():  # a copy, since later steps edit documents in place
        return copy.deepcopy(rng.choice(values))

    for _ in range(count):
        doc = copy.deepcopy(BASES[rng.choice(names)])
        for _ in range(rng.randint(1, 3)):
            every = list(paths(doc))
            path = rng.choice(every)
            current = get(doc, path)
            choice = rng.random()
            if choice < 0.3:
                doc = replaced(doc, path, pick())
            elif choice < 0.45:
                doc = replaced(doc, path, ABSENT)
            elif choice < 0.6:
                objects = [p for p in every if isinstance(get(doc, p), dict)]
                target = get(doc, rng.choice(objects)) if objects and rng.random() < 0.7 else doc
                target[rng.choice(keys)] = pick()
            elif choice < 0.75 and isinstance(current, (int, float)) and not isinstance(current, bool):
                doc = replaced(doc, path, current + rng.choice([-1, 1, 2**63, -(2**63), 0.5]))
            elif choice < 0.9 and isinstance(current, str):
                edit = rng.choice(["é", "😀", "\u0000", " ", "A", "\ud800", current, ""])
                doc = replaced(doc, path, rng.choice([current + edit, edit + current, current[:-1], current * 2]))
            elif isinstance(path[-1], str) and path[-1] in ENUMS:
                doc = replaced(doc, path, rng.choice([v for vs in ENUMS.values() for v in vs]))
        yield "random", dump(doc, **rng.choice(styles))


def fuzz_cases(seed: int = 20261015, random_count: int = 8000) -> list[tuple[str, bytes]]:
    """Seeded mutations of every base document, without duplicates."""
    rng = random.Random(seed)
    families = []
    big_seen: set = set()
    for doc in BASES.values():
        families += [_structure(doc), _values(doc, big_seen), _formats(doc, rng), _encodings(doc), _syntax(doc)]
    families += [_sizes(), _cross_field(), _random(rng, random_count)]
    seen, cases = set(), []
    for family in families:
        for category, raw in family:
            if raw not in seen:
                seen.add(raw)
                cases.append((category, raw))
    return cases


# --- Tests -------------------------------------------------------------------


def test_base_documents_are_valid() -> None:
    for name, doc in BASES.items():
        assert reference(dump(doc)) == doc["message_type"], name


def test_golden_fixtures_match_reference(native_cli: Path) -> None:
    valid = sorted((KIT / "fixtures/valid").glob("*.json"))
    invalid = sorted((KIT / "fixtures/invalid").glob("*.json"))
    verdicts = _run(native_cli, [str(path) for path in valid + invalid])

    assert verify_kit.verify_fixture_tree(KIT) == {"valid": len(valid), "invalid": len(invalid)}
    assert [_verdict(v) for v in verdicts] == [reference(p.read_bytes()) for p in valid + invalid]
    assert all(v["valid"] for v in verdicts[: len(valid)])
    assert not any(v["valid"] for v in verdicts[len(valid) :])


def _typed(value):
    """`value` with every number tagged by its JSON type."""
    if isinstance(value, dict):
        return {k: _typed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_typed(v) for v in value]
    if isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool)):
        return (type(value).__name__, value)
    return value


@pytest.mark.parametrize("path", sorted((KIT / "fixtures/valid").glob("*.json")), ids=lambda p: p.stem)
def test_fixture_round_trip(native_cli: Path, path: Path) -> None:
    result = subprocess.run([str(native_cli), "wire-check", "--roundtrip", str(path)], check=True,
                            capture_output=True)
    encoded = result.stdout.rstrip(b"\n")
    original = json.loads(path.read_bytes())

    assert json.loads(encoded) == original
    assert _typed(json.loads(encoded)) == _typed(original)
    assert reference(encoded) == original["message_type"]
    # The fixtures list members in schema order, so compact JSON of them is the encoding.
    assert encoded == json.dumps(original, separators=(",", ":"), ensure_ascii=False).encode()


def _targeted() -> list[tuple[str, bytes, str | None, str | None]]:
    """(case, document, reference verdict, native verdict) for each exactness rule."""
    control, event = BASES["control-drain"], BASES["event-accepted"]
    start, receipt = BASES["start-greedy"], BASES["receipt-completed"]

    def put(doc, path, value, **style):
        return dump(replaced(doc, path, value), **style)

    def raw_message(injected: bytes) -> bytes:
        return put(event, ("message",), f"x{MARK}y").replace(MARK.encode(), injected)

    drain = dump(control)
    queue_ms, top_p = ("metrics", "queue_ms"), ("sampling", "top_p")
    content = ("messages", 0, "content")
    many = {**start, "messages": [{"role": "user", "content": "a" * 65536}] * 8}
    return [
        # Strict JSON.
        ("duplicate key", drain.replace(b'"action":"drain"', b'"action":"drain","action":"drain"'), None, None),
        ("duplicate key spelled with an escape", drain.replace(b'"action":"drain"', b'"action":"drain","\\u0061ction":"drain"'), None, None),
        ("duplicate nested key", dump(with_pairs(start, ("model",), [*start["model"].items(), ("id", "x")])), None, None),
        ("member name spelled with an escape", drain.replace(b'"action"', b'"\\u0061ction"'), "control", "control"),
        ("top level array", b"[" + drain + b"]", None, None),
        ("top level string", b'"control"', None, None),
        ("byte order mark", BOM + drain, "control", "control"),
        ("two byte order marks", BOM + BOM + drain, None, None),
        ("lone high surrogate escape", put(event, ("message",), "\ud800"), "event", "event"),
        ("lone low surrogate escape", put(event, ("message",), "\udc00"), "event", "event"),
        ("reversed surrogate escapes", put(event, ("message",), "\udc00\ud800"), "event", "event"),
        ("surrogate pair escape counts once", put(event, ("message",), "😀" * 1024, ascii=True), "event", "event"),
        ("1025 surrogate pair escapes", put(event, ("message",), "😀" * 1025, ascii=True), None, None),
        ("surrogate in UTF-8 (by design)", raw_message(b"\xed\xa0\x80"), "event", None),
        ("UTF-16 document (by design)", drain.decode().encode("utf-16"), "control", None),
        ("UTF-32 document (by design)", drain.decode().encode("utf-32-le"), "control", None),
        ("overlong UTF-8", raw_message(b"\xc0\xaf"), None, None),
        ("code point above U+10FFFF", raw_message(b"\xf4\x90\x80\x80"), None, None),
        ("raw control character", raw_message(b"\x01"), None, None),
        ("raw DEL", raw_message(b"\x7f"), "event", "event"),
        ("escaped NUL", put(event, ("message",), "a\u0000"), None, None),
        ("nesting far beyond the schema", raw_message(b"").replace(b'"xy"', b"[" * 100000 + b"]" * 100000), None, None),
        # Integers versus numbers.
        ("schema_version 1.0", put(control, ("schema_version",), 1.0), "control", "control"),
        ("schema_version 1e0", put(control, ("schema_version",), Raw("1e0")), "control", "control"),
        ("schema_version true", put(control, ("schema_version",), True), None, None),
        ("batch_size 1.0", put(start, ("objective", "batch_size"), 1.0), "start", "start"),
        ("integer field 1.0", put(control, ("issued_at_unix_ms",), 1.0), None, None),
        ("integer field 1e3", put(control, ("issued_at_unix_ms",), Raw("1e3")), None, None),
        ("integer field true", put(control, ("issued_at_unix_ms",), True), None, None),
        ("integer field -0", put(control, ("issued_at_unix_ms",), Raw("-0")), "control", "control"),
        ("number field integer", put(start, ("sampling", "temperature"), 2), "start", "start"),
        ("number field true", put(start, ("sampling", "temperature"), True), None, None),
        ("seed int64 minimum", put(start, ("sampling", "seed"), INT64_MIN), "start", "start"),
        ("seed int64 maximum", put(start, ("sampling", "seed"), INT64_MAX), "start", "start"),
        ("seed below int64", put(start, ("sampling", "seed"), INT64_MIN - 1), None, None),
        ("seed above int64", put(start, ("sampling", "seed"), INT64_MAX + 1), None, None),
        ("seed beyond double precision", put(start, ("sampling", "seed"), 2**53 + 1), "start", "start"),
        ("sequence int64 maximum", put(event, ("sequence",), INT64_MAX), "event", "event"),
        ("sequence above int64 (by design)", put(event, ("sequence",), INT64_MAX + 1), "event", None),
        ("4301-digit integer", put(control, ("issued_at_unix_ms",), Raw("1" * 4301)), None, None),
        # Non-finite and rounded numbers.
        ("NaN literal", put(start, ("sampling", "temperature"), Raw("NaN")), None, None),
        ("Infinity literal", put(receipt, queue_ms, Raw("Infinity")), None, None),
        ("-Infinity literal", put(receipt, queue_ms, Raw("-Infinity")), None, None),
        ("overflowing float literal", put(receipt, queue_ms, Raw("1e400")), None, None),
        ("overflowing integer literal", put(receipt, queue_ms, 10**400), None, None),
        ("largest float", put(receipt, queue_ms, sys.float_info.max), "receipt", "receipt"),
        ("integer rounding to the largest float", put(receipt, queue_ms, Raw(str(2**1024 - 2**970 - 1))), "receipt", "receipt"),
        ("integer rounding past the largest float", put(receipt, queue_ms, Raw(str(2**1024 - 2**970))), None, None),
        ("top_p halfway above 1 rounds to 1", put(start, top_p, Raw("1.00000000000000011102230246251565404236316680908203125")), "start", "start"),
        ("top_p past halfway above 1", put(start, top_p, Raw("1.000000000000000111022302462515654042363166809082031251")), None, None),
        ("top_p underflowing to 0", put(start, top_p, Raw("1e-400")), None, None),
        ("top_p smallest subnormal", put(start, top_p, 5e-324), "start", "start"),
        # Lengths count code points; limits count bytes.
        ("content of 65536 four-byte code points", put(start, content, "😀" * 65536), "start", "start"),
        ("content of 65537 four-byte code points", put(start, content, "😀" * 65537), None, None),
        ("content of 32768 combining pairs", put(start, content, "e\u0301" * 32768), "start", "start"),
        ("quantization of 64 two-byte code points", put(start, ("model", "quantization"), "é" * 64), "start", "start"),
        ("quantization of 65 two-byte code points", put(start, ("model", "quantization"), "é" * 65), None, None),
        ("combined content 524288", dump(many), "start", "start"),
        ("combined content 524289", dump({**many, "messages": [*many["messages"], {"role": "user", "content": "b"}]}), None, None),
        ("control at its byte limit", drain + b" " * (LIMITS["control"] - len(drain)), "control", "control"),
        ("control one byte over", drain + b" " * (LIMITS["control"] + 1 - len(drain)), None, None),
        ("event text at its byte limit", _fill(BASES["event-text-delta"], ("text_delta",), LIMITS["event"], "é"), "event", "event"),
        ("event text one byte over", _fill(BASES["event-text-delta"], ("text_delta",), LIMITS["event"] + 1, "é"), None, None),
        # Rules.
        ("cancel without target", put(BASES["control-cancel"], ("target_request_id",), ABSENT), None, None),
        ("drain with target", put(control, ("target_request_id",), "req-1"), None, None),
        ("drain with null target", put(control, ("target_request_id",), None), None, None),
        ("accepted with null message", put(event, ("message",), None), None, None),
        ("identifier with trailing newline", put(control, ("control_id",), "control\n"), None, None),
        ("deadline exactly 24 hours later", put(start, ("deadline_unix_ms",), start["created_at_unix_ms"] + 86400000), "start", "start"),
        ("deadline 24 hours and 1 ms later", put(start, ("deadline_unix_ms",), start["created_at_unix_ms"] + 86400001), None, None),
        ("deadline at creation", put(start, ("deadline_unix_ms",), start["created_at_unix_ms"]), None, None),
        ("resident ceiling equal to runtime", put(start, ("memory", "max_resident_weight_bytes"), 4294967296), "start", "start"),
        ("resident ceiling above runtime", put(start, ("memory", "max_resident_weight_bytes"), 4294967297), None, None),
        ("rejected receipt failing while running", put(BASES["receipt-failed"], ("failure", "stage"), "running"), None, None),
        ("cancelled receipt with output", dump(BASES["receipt-cancelled"]), "receipt", "receipt"),
        ("canonically equivalent stop strings", put(start, ("sampling", "stop"), ["é", "e\u0301"]), "start", "start"),
    ]


TARGETED = _targeted()


def test_exactness_rules(native_cli: Path, tmp_path: Path) -> None:
    verdicts = native(native_cli, [doc for _, doc, _, _ in TARGETED], tmp_path)
    for (case, doc, expected_reference, expected_native), verdict in zip(TARGETED, verdicts):
        assert reference(doc) == expected_reference, f"reference changed: {case}"
        assert _verdict(verdict) == expected_native, f"{case}: {verdict}"
        if expected_reference != expected_native:
            assert by_design(doc, verdict), case


@pytest.fixture(scope="module")
def corpus(native_cli: Path, tmp_path_factory) -> tuple[list[tuple[str, bytes]], list[dict]]:
    cases = fuzz_cases()
    return cases, native(native_cli, [raw for _, raw in cases], tmp_path_factory.mktemp("wire-fuzz"))


def test_fuzz_verdicts_match_reference(corpus) -> None:
    cases, verdicts = corpus
    categories, outcomes, mismatches = Counter(), Counter(), []
    for (category, raw), verdict in zip(cases, verdicts):
        categories[category] += 1
        expected, actual = reference(raw), _verdict(verdict)
        if actual == expected:
            outcomes["agree-valid" if expected else "agree-invalid"] += 1
            continue
        reason = by_design(raw, verdict) if expected is not None and actual is None else None
        if reason:
            outcomes[f"by-design-{reason}"] += 1
        else:
            mismatches.append(f"{category}: reference={expected} native={verdict} doc={raw[:240]!r}")
    print(f"\nwire fuzz: {len(cases)} cases; {dict(sorted(outcomes.items()))}")
    print(f"wire fuzz categories: {dict(sorted(categories.items()))}")

    assert len(cases) >= 20000
    assert outcomes["agree-valid"] >= 2000 and outcomes["agree-invalid"] >= 10000
    assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _unencodable(value: dict) -> bool:
    """Valid documents the encoders refuse because the Swift decoders would reject them."""
    if any(_surrogates(text) for text in _strings(value)):
        return True
    stop = value["sampling"]["stop"] if value["message_type"] == "start" else []
    normalized = [unicodedata.normalize("NFC", text) for text in stop]
    return len(set(normalized)) != len(normalized)


def _same(encoded, original) -> bool:
    """JSON equality, except where the reference itself compares numerically:
    an integer in a number field comes back as float(integer), the value the
    reference validates, and schema_version or batch_size written as 1.0 comes
    back as 1. Integer fields cannot take this path because the reference
    rejects floats there, in the original and in the encoding alike."""
    numbers = [isinstance(x, (int, float)) and not isinstance(x, bool) for x in (encoded, original)]
    if all(numbers) and type(encoded) is not type(original):
        return float(encoded) == float(original)
    if isinstance(encoded, dict) and isinstance(original, dict):
        return encoded.keys() == original.keys() and all(_same(encoded[k], original[k]) for k in encoded)
    if isinstance(encoded, list) and isinstance(original, list):
        return len(encoded) == len(original) and all(map(_same, encoded, original))
    return type(encoded) is type(original) and encoded == original


def test_fuzz_round_trip(corpus, native_cli: Path, tmp_path: Path) -> None:
    cases, verdicts = corpus
    accepted = [raw for (_, raw), verdict in zip(cases, verdicts) if verdict["valid"] and b"\n" not in raw]
    encoded, refused = [], 0
    for raw, verdict in zip(accepted, native(native_cli, accepted, tmp_path, "--roundtrip")):
        value = json.loads(raw)
        if _unencodable(value):
            assert verdict["type"] == value["message_type"] and not verdict["valid"], verdict
            assert "cannot encode" in verdict["error"]
            refused += 1
            continue
        assert verdict["valid"], (verdict, raw[:240])
        out = verdict["encoded"].encode("utf-8")
        assert reference(out) == value["message_type"], out[:240]
        assert _same(json.loads(out), value), raw[:240]
        encoded.append(out)
    again = native(native_cli, encoded, tmp_path, "--roundtrip")
    print(f"\nwire round trip: {len(encoded)} re-encoded, {refused} refused as unencodable")

    assert len(encoded) >= 2000 and refused > 0
    assert [verdict["encoded"].encode("utf-8") for verdict in again] == encoded  # encoding is a fixed point


def _event(sequence: int, kind: str = "text_delta", request_id: str = "req-stream-1", **fields) -> dict:
    samples = {"accepted": {}, "progress": {"progress": 0.5}, "text_delta": {"text_delta": "ok", "generated_tokens_delta": 1},
               "queue_state": {"queue_position": 0, "queue_depth": 1}, "warning": {"message": "slow"}}
    return {"schema_version": 1, "message_type": "event", "request_id": request_id, "sequence": sequence,
            "kind": kind, "emitted_at_unix_ms": 1787630400000 + sequence, **(fields or samples[kind])}


def _receipt(sequence: int, request_id: str = "req-stream-1", name: str = "receipt-completed") -> dict:
    return {**BASES[name], "request_id": request_id, "terminal_sequence": sequence}


STREAMS = {
    "fixture order": ([_event(0, "accepted"), _event(1), _receipt(2)], [1, 1, 1]),
    "every kind": ([_event(0, "accepted"), _event(1, "progress"), _event(2, "queue_state"), _event(3, "warning"),
                    _event(4), _event(5), _receipt(6)], [1] * 7),
    "receipt alone": ([_receipt(0, name="receipt-failed")], [1]),
    "cancelled after acceptance": ([_event(0, "accepted"), _receipt(1, name="receipt-cancelled")], [1, 1]),
    "out of order": ([_event(0, "accepted"), _event(1), _event(2), _event(1)], [1, 1, 1, 0]),
    "gap": ([_event(0, "accepted"), _event(2)], [1, 0]),
    "duplicate": ([_event(0, "accepted"), _event(1), _event(1)], [1, 1, 0]),
    "accepted not first": ([_event(0)], [0]),
    "accepted not at zero": ([_event(1, "accepted")], [0]),
    "rejection leaves the state unchanged": ([_event(0), _event(0, "accepted"), _event(1, text_delta=""),
                                              _event(1)], [0, 1, 0, 1]),
    "event after the receipt": ([_event(0, "accepted"), _receipt(1), _event(2)], [1, 1, 0]),
    "second receipt": ([_event(0, "accepted"), _receipt(1), _receipt(2)], [1, 1, 0]),
    "terminal_sequence repeats the last event": ([_event(0, "accepted"), _event(1), _receipt(1)], [1, 1, 0]),
    "terminal_sequence skips": ([_event(0, "accepted"), _event(1), _receipt(3)], [1, 1, 0]),
    "lone receipt not at zero": ([_receipt(5, name="receipt-failed")], [0]),
    "event for another request": ([_event(0, "accepted"), _event(1, request_id="req-other")], [1, 0]),
    "receipt for another request": ([_event(0, "accepted"), _receipt(1, request_id="req-other")], [1, 0]),
    "start in a stream": ([BASES["start-greedy"]], [0]),
}


@pytest.mark.parametrize("name", STREAMS)
def test_stream_validator(native_cli: Path, name: str) -> None:
    messages, expected = STREAMS[name]
    verdicts = _run(native_cli, ["--stream", "req-stream-1", "--stdin"], b"".join(dump(m) + b"\n" for m in messages))

    assert [int(v["valid"]) for v in verdicts] == expected, verdicts
    accepted = sum(expected)
    assert verdicts[-1]["next_sequence"] == accepted
    assert verdicts[-1]["terminated"] == any(
        ok and m["message_type"] == "receipt" for ok, m in zip(expected, messages)
    )


@pytest.mark.parametrize("request_id", ["bad id", "a" * 129, "", ".a", "é"])
def test_stream_needs_a_valid_request_id(native_cli: Path, request_id: str) -> None:
    result = subprocess.run([str(native_cli), "wire-check", "--stream", request_id, "--stdin"], input=b"",
                            capture_output=True)
    assert result.returncode == 2 and b"request_id" in result.stderr


def test_stream_accepts_the_longest_request_id(native_cli: Path) -> None:
    request_id = "a" * 128
    lines = dump(_event(0, "accepted", request_id)) + b"\n" + dump(_receipt(1, request_id)) + b"\n"
    verdicts = _run(native_cli, ["--stream", request_id, "--stdin"], lines)
    assert [v["valid"] for v in verdicts] == [True, True] and verdicts[-1]["terminated"]


def _encode_cases() -> list[tuple[str, bytes, bool]]:
    """(case, lenient JSON for --encode, whether the encoder accepts it)."""
    text, receipt = BASES["event-text-delta"], BASES["receipt-completed"]

    def put(doc, path, value):
        return dump(replaced(doc, path, value))

    raw_text = put(text, ("text_delta",), MARK)
    return [
        ("valid text delta", dump(text), True),
        ("valid progress", dump(BASES["event-progress"]), True),
        ("valid queue state", dump(BASES["event-queue"]), True),
        ("valid completed receipt", dump(receipt), True),
        ("valid cancelled receipt", dump(BASES["receipt-cancelled"]), True),
        ("valid rejected receipt", dump(BASES["receipt-failed"]), True),
        ("empty text_delta", put(text, ("text_delta",), ""), False),
        ("negative sequence", put(text, ("sequence",), -1), False),
        ("negative emitted_at", put(text, ("emitted_at_unix_ms",), -5), False),
        ("empty request_id", put(text, ("request_id",), ""), False),
        ("request_id with a space", put(text, ("request_id",), "req 1"), False),
        ("unknown kind", put(text, ("kind",), "token"), False),
        ("text_delta without its token count", put(text, ("generated_tokens_delta",), ABSENT), False),
        ("token count above 8192", put(text, ("generated_tokens_delta",), 8193), False),
        ("message on a text_delta", dump({**text, "message": "m"}), False),
        ("progress NaN", put(BASES["event-progress"], ("progress",), Raw("NaN")), False),
        ("progress Infinity", put(BASES["event-progress"], ("progress",), Raw("Infinity")), False),
        ("progress above 1", put(BASES["event-progress"], ("progress",), 1.5), False),
        ("negative queue position", put(BASES["event-queue"], ("queue_position",), -1), False),
        ("message of 1025 code points", put(BASES["event-accepted"], ("message",), "é" * 1025), False),
        ("text with a lone surrogate", put(text, ("text_delta",), "\ud800"), False),
        ("text with invalid UTF-8", raw_text.replace(MARK.encode(), b"\xff"), False),
        ("text with NUL", put(text, ("text_delta",), "a\u0000"), False),
        ("completed with a failure", dump({**receipt, "failure": FAILURE}), False),
        ("completed without output_text", put(receipt, ("output_text",), ABSENT), False),
        ("failed without a failure", dump({**receipt, "status": "failed", "finish_reason": "error"}), False),
        ("rejected while running", put(BASES["receipt-failed"], ("failure", "stage"), "running"), False),
        ("status and finish_reason disagree", put(receipt, ("finish_reason",), "error"), False),
        ("total_tokens mismatch", put(receipt, ("usage", "total_tokens"), 18), False),
        ("negative peak memory", put(receipt, ("metrics", "peak_memory_bytes"), -1), False),
        ("maximum_batch_size 0", put(receipt, ("metrics", "maximum_batch_size"), 0), False),
        ("queue_ms NaN", put(receipt, ("metrics", "queue_ms"), Raw("NaN")), False),
        ("negative queue_ms", put(receipt, ("metrics", "queue_ms"), -1.0), False),
        ("decode rate of 0", put(receipt, ("metrics", "decode_tokens_per_second"), 0.0), False),
        ("fallback without a reason", put(receipt, ("execution", "fallback_used"), True), False),
        ("reason without a fallback", put(receipt, ("execution", "fallback_reason"), "why"), False),
        ("duplicate evidence ids", put(receipt, ("execution", "evidence_ids"), ["a", "a"]), False),
        ("17 evidence ids", put(receipt, ("execution", "evidence_ids"), [f"e{i}" for i in range(17)]), False),
        ("unknown backend", put(receipt, ("execution", "backend"), "gpu"), False),
        ("negative terminal_sequence", put(receipt, ("terminal_sequence",), -1), False),
        ("failure code with a space", put(BASES["receipt-failed"], ("failure", "code"), "out of memory"), False),
        ("output_text with a lone surrogate", put(receipt, ("output_text",), "\udfff"), False),
    ]


ENCODE_CASES = _encode_cases()


def test_encoders_fail_closed(native_cli: Path) -> None:
    verdicts = _run(native_cli, ["--encode", "--stdin"], b"".join(doc + b"\n" for _, doc, _ in ENCODE_CASES))

    assert len(verdicts) == len(ENCODE_CASES)
    for (case, doc, accepted), verdict in zip(ENCODE_CASES, verdicts):
        assert verdict["valid"] == accepted, f"{case}: {verdict}"
        if accepted:
            out = verdict["encoded"].encode("utf-8")
            assert reference(out) == json.loads(doc)["message_type"], case
            assert json.loads(out) == json.loads(doc), case
        else:
            assert verdict["encoded"] is None and verdict["error"].startswith("wire: "), case
