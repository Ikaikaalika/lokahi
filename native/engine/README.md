# Lōkahi native engine

C++20 inference engine with a Metal GPU backend, a portable CPU backend, a C
ABI ([`include/lokahi/lokahi.h`](include/lokahi/lokahi.h)), and the `lokahi`
CLI. Python is not on the token path.

```sh
cmake -S native/engine -B build/engine -DCMAKE_BUILD_TYPE=Release   # Metal is on by default on macOS
cmake --build build/engine -j
./build/engine/lokahi_engine_tests
./build/engine/lokahi info --model /path/to/mlx-snapshot
./build/engine/lokahi run --model DIR --prompt "Explain unified memory in one paragraph."
./build/engine/lokahi generate --model DIR --prompt "Aloha" --max-new 64     # JSON: tokens + text
./build/engine/lokahi tokenize --model DIR --text "Aloha kākou"
./build/engine/lokahi pretokenize --model DIR --text "Aloha kākou"          # words BPE runs on
./build/engine/lokahi bench --model DIR --tokens-file prompt.txt --max-new 128
```

## Status

| Area | State | Evidence |
|---|---|---|
| Gemma 3 text decoder, CPU backend | Implemented | Correctness: matches the NumPy oracle within 2e-4 relative; greedy tokens exact (`tests/test_native_engine.py`) |
| Gemma 3 text decoder, Metal backend | Implemented, awaiting first hardware run | Builds on hosted macOS CI; execution and parity run on the self-hosted runner |
| MLX affine 2/4/8-bit weights, BF16/F16 scales, per-module bit overrides | Implemented | Unit tests; MLX cross-check on Apple Silicon |
| Tokenizer: SentencePiece-style BPE (Gemma, Llama 2, Mistral) from `tokenizer.json` | Implemented | Correctness: identical ids and text to Hugging Face `tokenizers` on trained fixtures (byte fallback, unknown fusion, added tokens, Metaspace) and a 262k-entry synthetic vocabulary (`tests/test_tokenizer.py`) |
| Tokenizer: byte-level BPE (Llama 3, Qwen, GPT-OSS, DeepSeek, GLM) | Implemented | Correctness: identical words, ids and text to Hugging Face `tokenizers` for the Qwen 3, Llama 3, GLM, GPT-OSS o200k, DeepSeek V3 and GPT-2 configurations on text from every Unicode plane (`tests/test_tokenizer_bytelevel.py`); regex engine fuzzed against the reference's Oniguruma |
| Streaming text output (`lokahi run`) | Implemented | Emits only text later tokens cannot rewrite; property-tested |

No throughput claim is made until the self-hosted runner records hardware
evidence against a fresh MLX-LM control.

## Design

**Loading.** Checkpoints are memory-mapped safetensors (single file or
shards) in MLX layout. The Metal backend copies every matrix once into a few
large shared buffers, aligned to 256 bytes and marked hazard-untracked
because they are read-only.

**Decode step.** One command buffer per token, with roughly 9 dispatches per layer:

1. Embedding gather.
2. Fused q/k/v matrix-vector product (`qmv_qkv`).
3. Fused QK-norm, RoPE and KV-cache write.
4. Split-K online-softmax attention, plus a reduce pass.
5. Output projection.
6. Fused residual and RMSNorm.
7. Fused gate/up and gated activation (`qmv_glu`; GELU for Gemma, SiLU for the Llama family).
8. Down projection.
9. Fused residual and RMSNorm.

The step ends with the last-row vocabulary projection and an argmax on the GPU.

**Pipelined greedy generation.** Each step reads the previous token from GPU
memory, so two steps stay in flight and the host never stalls the GPU between
tokens.

**Prefill.**

- Matrix products use a tiled quantized matmul (`qmm`) built on SIMD-group
  8×8 matrices.
- Chunks of up to `--prefill-chunk` tokens all go into one command buffer.
- The vocabulary projection runs only for the final prompt token.

**KV cache.** Global layers hold `max_context` slots. Sliding-window layers
hold `window + prefill_chunk` slots in a ring, so a whole chunk can be written
before its queries read the preceding window.

**Numerics.** Activations, norms, attention and the KV cache are float32.
Weights stay in their packed 4-bit or 8-bit form and are dequantized inside
the kernels.

**Tokenizer.** `tokenizer.json` is interpreted natively and must agree with
Hugging Face `tokenizers` id for id; components it cannot reproduce exactly
fail at load time. Byte-level models split text with their published regex,
run by a backtracking engine for the Oniguruma subset those patterns use
(Unicode property classes, lookahead, case-insensitive groups, lazy and
possessive quantifiers) with leftmost-first semantics. The reference does not
use one Unicode version: its regex classes follow Unicode 16 while its NFC
follows older data. `tools/gen_unicode_tables.py` therefore extracts the
category, combining-class and composition tables from the pinned reference
release, so new code points classify and normalize exactly as it does.
Streaming holds back tokens that end inside a UTF-8 sequence or a
byte-fallback run, so emitted text is never rewritten.

## Checks on machines without Apple SDKs

`tests/syntax_check.sh` parses the Objective-C++ host and the Metal kernels
with clang against stub headers (`tests/syntax_stubs/`). It catches typos and
C++ type errors on Linux. It is not a Metal compiler.
