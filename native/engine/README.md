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
./build/engine/lokahi generate --model DIR --prompt "Aloha" --temperature 0.7 --top-p 0.9 --seed 7
./build/engine/lokahi tokenize --model DIR --text "Aloha kākou"
./build/engine/lokahi pretokenize --model DIR --text "Aloha kākou"          # words BPE runs on
./build/engine/lokahi bench --model DIR --tokens-file prompt.txt --max-new 128
./build/engine/lokahi profile --model DIR --tokens-file prompt.txt --max-new 32   # JSON: time per section
./build/engine/lokahi serve --config service.json   # Lōkahi wire v1 (Common Compute) as NDJSON on stdin/stdout
```

## Status

| Area | State | Evidence |
|---|---|---|
| Decoders on CPU: Gemma 3, Gemma 4 (dense; E2B/E4B and 31B layouts), Llama 3.x, Mistral, Qwen 2/2.5, Qwen 3, Qwen 3 MoE, GPT-OSS | Implemented | Correctness: matches the NumPy oracles (`oracles/gemma3.py`, `oracles/gemma4.py`, `oracles/llama.py`) within 2e-4 relative; greedy tokens exact (`tests/test_native_engine.py`). Qwen 3 MoE fixtures cover stacked (MLX) and per-expert (Transformers) weights, renormalized and raw top-k, dense layers between sparse ones, and route far from ties on the router's real inputs. GPT-OSS fixtures mirror the 20B layout (MXFP4 experts behind affine overrides, sinks, alternating windows, untruncated YaRN) and are guarded so the SwiGLU clamps bind and the sinks move the logits; the published 20B config parses to the expected layer map, router and YaRN ramp. Gemma 4 fixtures cover KV sharing of both layer kinds, `attention_k_eq_v`, partial RoPE, per-layer inputs and layer scalars, and every non-equivalent mutation of those features fails the suite. The published E2B config parses to the expected layer map (`tests/engine_tests.cpp`) |
| Dense decoders on Metal (Gemma 3 and the dense Llama family) | Implemented; correctness verified on a virtual GPU | Hosted macOS CI runs the Metal suites against the oracles on its paravirtual device (correctness only, no performance claims); hardware runs wait for the self-hosted runner. Gemma 4, mixture-of-experts models and MXFP4 weights are refused (`--backend auto` picks the CPU) until their kernels land |
| Oracles vs MLX-LM (`gemma3_text`, `gemma4`, `llama`, `qwen2`, `qwen3`, `qwen3_moe`, `gpt_oss`) | Written | Runs on the self-hosted Mac (`tests/test_mlx_crosscheck.py`) |
| MLX affine 2/4/8-bit weights, BF16/F16 scales, per-module bit overrides | Implemented | Unit tests; MLX cross-check on Apple Silicon |
| MLX MXFP4 weights (E2M1 codes, E8M0 scale per 32; GPT-OSS's expert format), on CPU | Implemented | Correctness: unit tests against hand-computed rows, the OCP encodings, and oracle parity on MXFP4 MLPs and experts (`tests/test_native_engine.py`); `mx.dequantize` cross-check on Apple Silicon. Per-module overrides follow `mlx.nn.quantize`: keys an override omits take its mode's defaults. Metal refuses MXFP4 for now |
| Tokenizer: SentencePiece-style BPE (Gemma, Llama 2, Mistral) from `tokenizer.json` | Implemented | Correctness: identical ids and text to Hugging Face `tokenizers` on trained fixtures (byte fallback, unknown fusion, added tokens, Metaspace) and a 262k-entry synthetic vocabulary (`tests/test_tokenizer.py`) |
| Tokenizer: byte-level BPE (Llama 3, Qwen, GPT-OSS, DeepSeek, GLM) | Implemented | Correctness: identical words, ids and text to Hugging Face `tokenizers` for the Qwen 3, Llama 3, GLM, GPT-OSS o200k, DeepSeek V3 and GPT-2 configurations on text from every Unicode plane (`tests/test_tokenizer_bytelevel.py`); regex engine fuzzed against the reference's Oniguruma |
| Streaming text with stop strings (`TextStream`, `lokahi run`) | Implemented | Emits only text later tokens cannot rewrite, cut at the earliest stop string, at constant cost per token. Correctness: equals Hugging Face `tokenizers`' full decode (cut at the stop) on every trained tokenizer configuration (`tests/test_text_stream.py`) |
| Common Compute service (`src/service.h`, `lokahi_service_*` C ABI, `lokahi serve`, Swift XPC package in `integrations/commoncompute`) | Implemented | Correctness (acceptance gate B, gate C's parity on a fixture). Every emitted document passes the kit's reference verifier and the stream rules. Output equals the NumPy model with the jinja2-rendered prompt, and usage counts are exact. Covers seeded sampling, stop strings, every rejection path, cancelling queued and running requests, deadlines, drain and duplicates (`tests/test_service.py`, also under ThreadSanitizer and AddressSanitizer). The XPC tests cross a real NSXPC connection on hosted macOS |
| Chat templates: native Jinja subset (`src/chat_template.h`, `lokahi chat-template`) | Implemented | Correctness: byte-identical to jinja2 set up as transformers sets it up, for the real templates of Qwen3, Llama 3.2, Gemma 3, Mistral v0.3, DeepSeek-R1 distill, Qwen2.5, SmolLM3, Phi-4-mini, GLM-4 and GPT-OSS harmony (`tests/test_chat_template.py`, fixtures in `tests/fixtures/chat_templates`). Syntax outside the subset fails at parse time, and operations Python would compute differently (Unicode case mapping, big integers, `%` formatting) fail at render time |
| Lōkahi wire v1 codec and stream rules (`src/wire.h`, `lokahi wire-check`) | Implemented | Correctness: the kit's golden fixtures, and about 22,000 seeded mutations, get the same verdict as `verify_kit.py`, except two stricter rules documented in `wire.h` (UTF-8 only, integers within int64). Encoders round-trip every valid fixture and refuse invalid structs (`tests/test_wire_native.py`) |
| Seeded sampling: temperature, top-p, top-k, min-p (CPU and a GPU kernel) | Implemented; GPU kernel awaiting first hardware run | Correctness: draws equal an independent float64 oracle (`oracles/sampling.py`) and follow the exact distribution (chi-square); Philox matches Random123's known answers (`tests/test_sampling.py`) |

No throughput claim is made until the self-hosted runner records hardware
evidence against a fresh MLX-LM control.

## Design

**Model families.** `DecoderConfig` describes how the supported families
differ. The CPU backend implements every option; the Metal backend
implements every column but Gemma 4, and no mixture of experts yet:

| Option | Gemma 3 | Gemma 4 | Llama 3.x | Mistral | Qwen 2/2.5 | Qwen 3 / Qwen 3 MoE |
|---|---|---|---|---|---|---|
| RMSNorm scale | `1 + w` | `w` | `w` | `w` | `w` | `w` |
| Norms after each sublayer | yes | yes | no | no | no | no |
| Per-head Q/K norm | yes | yes, plus a scale-free V norm; attention scale 1 | no | no | no | yes |
| q/k/v biases | no | no | no | no | yes | no |
| Gated MLP | GELU (tanh) | GELU (tanh), twice as wide in KV-shared layers | SiLU | SiLU | SiLU | SiLU |
| RoPE | local and global bases, linear scaling | per layer kind; "proportional" partial rotation on full layers | `llama3` or linear scaling | linear scaling | default or YaRN | default or YaRN |
| Sliding window | 5 of every 6 layers | 4 of every 5 layers | no | every layer, when set | no | no |
| Per-layer geometry | no | full layers' own head size; with `attention_k_eq_v`, values are the raw keys and full layers have their own kv head count | no | no | no | no |
| KV sharing | no | the last `num_kv_shared_layers` layers attend over the last unshared layer of their kind | no | no | no | no |
| Per-layer inputs | no | a per-layer token embedding plus a projection of the embedding, gated into each layer; each layer's output times its `layer_scalar` | no | no | no | no |
| Mixture of experts | no | not yet (26B-A4B fails at load) | no | no | no | Qwen 3 MoE: a softmax router picks the top `num_experts_per_tok` SiLU-gated experts, renormalized with `norm_topk_prob`, in the layers `decoder_sparse_step` and `mlp_only_layers` select; MLX's stacked `switch_mlp` or per-expert weights |

GPT-OSS is the Llama-family parser plus: q/k/v/o biases; a learned sink
logit per head in every softmax; alternating sliding (128) and full layers;
YaRN as Transformers computes it, honoring `truncate: false` (MLX-LM always
rounds the correction range); and in every layer a biased router whose top-4
logits are softmaxed over 32 biased experts with the clamped SwiGLU
`min(g, 7)·σ(1.702·min(g, 7))·(clip(u, −7, 7) + 1)`. YaRN's attention factor
scales q and k in the references; the engine folds its square into the
attention scale, which gives the same logits on every backend.

A family is a config parser in `decoder.cpp`. Options outside this table
(o_proj or MLP biases outside GPT-OSS, longrope and other RoPE types,
partial rotary outside Gemma 4, other activations, Gemma 4's
mixture-of-experts blocks) fail at load.

Gemma 4's scalar constants (the embedding scale and the per-layer input
scales) are rounded to BF16, as MLX-LM applies them to BF16 activations.
Transformers rounds the embedding scales the same way but keeps the per-layer
input scales in float32; those differ by about 1e-4, and two of the three
cancel in the RMSNorm that follows.

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

**Sampling.** Greedy decoding takes the argmax (lowest id on ties).
Otherwise the step ends with a seeded draw that matches mlx_lm's semantics:

- top-p, min-p and top-k filter the untempered distribution, each as a
  threshold logit, so tokens tied at a boundary are all kept;
- the token is the Gumbel-max over what is left of `logit / T`.

The noise is Philox4x32-10 keyed by the seed and counted by
(generated-token index, token id). The CPU sampler and the GPU kernel
therefore draw the same tokens from the same logits, a seed reproduces its
output regardless of pipelining, and the GPU path keeps two steps in flight
exactly as greedy decoding does. On the GPU, the top-k and top-p thresholds
are found by bisection over the float's order-preserving key, inside the
same single threadgroup that draws the token. `lokahi sample --backend metal`
runs that kernel on arbitrary logits for testing.

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
byte-fallback run, so emitted text is never rewritten. It also holds back
text that could still become a stop string. Each token re-decodes only a
short window that starts at an earlier stable boundary, so the cost per
token does not grow with the output.

## Profiling

`lokahi profile` reports where a forward pass spends its time. It runs warmup
generations, then one prefill and `--max-new` decode steps. Each section is
reported with its total time, its call count and the minimum memory traffic
its math requires. Sections follow each backend's own fused work, so the
names differ:

- **Metal:** embed, norm, qkv, rope_kv, attention, o_proj, residual_norm, mlp,
  lm_head and sample. Each section is its own command buffer, and its time is
  that buffer's GPU time.
- **CPU:** embed, norm, qkv, rope_kv, attention, o_proj, residual, mlp_up,
  down_proj, router and experts (mixture of experts), per_layer_input
  (Gemma 4) and lm_head. Times are host time.

The per-section synchronization makes profiled runs slower than `bench`. Use
the shares to choose work, and `bench` to measure it.
[`benchmarks/attribution.py`](../../benchmarks/attribution.py) ranks the
sections by time above their bandwidth floor
([improvement loop](../../docs/IMPROVEMENT_LOOP.md)). Tokens are unchanged by
profiling (`tests/test_native_engine.py`).

## Checks on machines without Apple SDKs

`tests/syntax_check.sh` parses the Objective-C++ host and the Metal kernels
with clang against stub headers (`tests/syntax_stubs/`). It catches typos and
C++ type errors on Linux. It is not a Metal compiler.
