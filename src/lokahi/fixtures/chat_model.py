"""A tiny chat model for driving the wire service end to end.

A random Gemma 3 checkpoint (``write_tiny_gemma3``) plus everything a chat
request needs:

* a byte-level ``tokenizer.json`` whose vocabulary is the 256 GPT-2 byte
  symbols with no merges, so it is written directly rather than trained;
* ChatML special tokens;
* a ``tokenizer_config.json`` with a ChatML chat template, ``<|im_end|>`` as
  EOS and no BOS, like Qwen.

Ids past the tokenizer's vocabulary decode to nothing, as in Hugging Face
``tokenizers``. Correctness evidence only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gemma3_tiny import TinyGemma3Spec, write_tiny_gemma3

SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
EOS_TOKEN = "<|im_end|>"
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
VOCAB_SIZE = 288  # 256 bytes + 3 specials, padded to a multiple of 32


def _byte_symbols() -> list[str]:
    """GPT-2's printable stand-in for each byte value, indexed by byte."""

    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    symbols = {b: chr(b) for b in printable}
    extra = 0
    for b in range(256):
        if b not in symbols:
            symbols[b] = chr(256 + extra)
            extra += 1
    return [symbols[b] for b in range(256)]


def tokenizer_json() -> dict:
    vocab = {symbol: index for index, symbol in enumerate(_byte_symbols())}
    byte_level = {"add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 256 + i, "content": token, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for i, token in enumerate(SPECIAL_TOKENS)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", **byte_level},
        "post_processor": {"type": "ByteLevel", **byte_level},
        "decoder": {"type": "ByteLevel", **byte_level},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": []},
    }


def write_tiny_chat_model(directory: Path, seed: int = 0) -> Path:
    directory = Path(directory)
    eos = 256 + SPECIAL_TOKENS.index(EOS_TOKEN)
    write_tiny_gemma3(directory, TinyGemma3Spec(vocab_size=VOCAB_SIZE, seed=seed, extra={"eos_token_id": [eos]}))
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer_json(), ensure_ascii=False), encoding="utf-8")
    config = {"chat_template": CHATML_TEMPLATE, "bos_token": None, "eos_token": EOS_TOKEN}
    (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_tiny_chat_model(args.directory, args.seed)
    print(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
