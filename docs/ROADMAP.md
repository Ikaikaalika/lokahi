# Lōkahi roadmap

Goal: the fastest LLM/VLM runtime on every Apple Silicon Mac, using the CPU,
GPU, Neural Engine, and unified memory together wherever measurements show
the combined plan wins. This page is the build order. Claims follow
[`PERFORMANCE_CONTRACT.md`](PERFORMANCE_CONTRACT.md): nothing here is
promoted until correctness and hardware evidence exist for the exact model,
chip, OS build, and workload. Work is chosen and accepted by the
[improvement loop](IMPROVEMENT_LOOP.md).

## Where things stand (2026-10-12)

| Piece | State | Evidence |
|---|---|---|
| Native C++ engine, C ABI, CLI | Done for Gemma 3, Llama 3.x, Mistral, Qwen 2/2.5 and Qwen 3 text | Correctness: CPU matches the NumPy oracles (2e-4 rel), exact greedy tokens |
| Native tokenizer | Done for SentencePiece-style and byte-level BPE | Correctness: identical to Hugging Face `tokenizers` |
| Metal backend (same families) | Written; compiles on hosted macOS CI | Awaiting first self-hosted run |
| Oracles vs MLX-LM cross-check | Written | Runs on the self-hosted Mac |
| Self-hosted Apple Silicon CI | Written | Needs the runner registered ([setup](APPLE_SILICON_RUNNER.md)) |
| [Improvement loop](IMPROVEMENT_LOOP.md): `lokahi profile`, attribution, scoreboard, PR comment | Written | Correctness of the tooling: the profiler leaves tokens unchanged; scoring is tested on synthetic reports. First scoreboard waits on the runner |
| Common Compute service: wire v1 codec, chat templates, request service, C ABI, Swift XPC package | Written | Correctness: acceptance gate B, and gate C parity on a fixture model (CPU here, XPC on hosted macOS). Gate C on a real pinned model runs on the self-hosted Mac. Wiring into the provider app needs access to its repository |
| Common Compute fleet benchmarks | Spec below | Needs access to the provider app repository |
| MLX-LM M1 baselines | Recorded (Qwen3 0.6B, Llama 3.2 1B, Gemma 3 1B) | Hardware, Aug 2026 |

## Why a native engine can beat MLX-LM

Decode on Apple Silicon is limited by memory bandwidth, so the target is the
fewest bytes moved per token and the least idle time between GPU work. The
levers, in measured-priority order:

1. **No idle gaps.** One command buffer per token, sampling on the GPU, and
   the next step encoded while the current one runs. Small models are
   currently dominated by dispatch and host round trips.
2. **Fusion.** Fusing q/k/v, gate/up/GELU, norm/RoPE/KV-write and
   residual/norm cuts dispatches and intermediate writes per layer.
3. **No wasted work.** Project only the last prompt row onto the
   vocabulary. For Gemma 3 1B the 262k-row vocabulary matrix is about 30% of
   all weights, so computing it for every prompt token would be costly. Lōkahi
   never does; whether a given MLX-LM path does is measured, not assumed.
4. **Per-chip kernels.** Tile sizes, rows per SIMD group, and KV precision are
   tuned per GPU family and recorded as evidence. The M5 GPU's neural
   accelerators (Metal 4 tensor APIs) get their own prefill kernels.
5. **Speculation that uses idle engines.** A small draft model runs on the
   ANE (or the GPU) while the GPU verifies several tokens per weight read,
   which cuts bytes per accepted token.

## Engine plan by compute unit

### GPU (Metal), primary engine

| Step | Work | Exit gate |
|---|---|---|
| G1 | Verify the Metal path on M1 | `LOKAHI_TEST_BACKENDS=cpu,metal` parity green; Gemma 3 1B, Llama 3.2 1B and Qwen3 0.6B benchmarked against fresh MLX-LM controls (the three recorded baselines) |
| G2 | Decode tuning | ≥10% decode win on Gemma 3 1B with exact tokens: qmv rows per SIMD group, vectorized unpack, indirect command buffers or Metal 4 command allocators, residency sets |
| G3 | Prefill attention | Online-softmax tiled attention with 8×8 SIMD-group matrices, head_dim 256 and 512, causal and sliding-block skipping |
| G4 | KV precision | FP16/BF16 KV (then 8-bit) with parity budget; halves long-context attention traffic |
| G5 | Zero-copy load | Page-aligned weight capsule mapped with `newBufferWithBytesNoCopy`, removing the load-time copy and duplicate residency |
| G6 | M5 neural accelerators | Metal 4 tensor / Performance Primitives matmul for prefill; separate evidence per chip |

### Neural Engine (Developer ID builds: direct; Core ML fallback)

Private ANE APIs are allowed in the Developer ID–signed Common Compute app.
Every launch runs a compile/dispatch/readback self-test; if it fails, for
example after a macOS update, the plan falls back to Core ML or the GPU. A
compile cache is not live weight paging, and configuring Core ML is not proof
that work ran on the ANE.

| Step | Placement | Why it can win |
|---|---|---|
| A1 | Make the qualified fixed-shape projection a resident worker (compile once, reuse IOSurfaces) | Removes per-request compile and load |
| A2 | Vision encoders for VLMs | Fixed shapes, run once per image, conv/attention-heavy: a natural ANE fit that frees the GPU for decoding |
| A3 | Draft or MTP model for speculative decoding | Runs concurrently with GPU verification |
| A4 | Long-prompt prefill projections in fixed token lanes | Amortizes dispatch over 64+ tokens |
| A5 | Common Compute side lanes (embeddings, rerank) | Throughput without stealing GPU decode time |

### CPU

| Step | Work |
|---|---|
| C1 | Native tokenizer so the production path has no Python. SentencePiece-style BPE (Gemma, Llama 2, Mistral) and byte-level BPE (Llama 3, Qwen, GPT-OSS, DeepSeek, GLM) are done and match Hugging Face `tokenizers`. Next: chat templates (the Jinja subset in `tokenizer_config.json`, so prompts render natively), an incremental detokenizer in the C ABI for streaming, and faster `tokenizer.json` loading (about 1.2 s for a 262k vocabulary today) |
| C2 | Sampling fused on the GPU with a CPU reference. Done: seeded temperature, top-p, top-k and min-p with mlx_lm semantics, identical across backends. Next: repetition and presence penalties, logit bias, and grammar-constrained decoding |
| C3 | Scheduler: separate prefill and decode queues, continuous batching, prefix cache, cancellation |
| C4 | Matrix compute on the CPU: AMX on M1–M3 through Accelerate (BNNS / BLAS), and SME2 on M4 and later, for small-batch side work such as draft-model layers or embeddings, only where measured to help |

**On "machine code".** Apple does not publish the GPU instruction set, and
macOS always compiles Metal shaders itself, so hand-written GPU machine code
cannot ship. The lowest supported layer is Metal Shading Language with
SIMD-group matrix operations, which is what the engine uses. On the CPU,
ARM64 NEON and SME2 intrinsics or assembly are available. AMX on M1–M3 is
undocumented, so the engine reaches it through Accelerate.

## Model coverage order

Each family lands in the same pattern: config parsing, then the NumPy
oracle, then generated fixtures, then CPU, then Metal, then an MLX-LM
cross-check, then a pinned real model on the runner.

| Wave | Models | New mechanics |
|---|---|---|
| 1 | Gemma 3 (270M–27B) | Done in code; verify on hardware |
| 2 | Dense GQA family: Llama 3.x, Mistral, Qwen 2/2.5, Qwen 3 (and the DeepSeek-R1 distills built on them) | Done in code; verify on hardware. Pulled ahead of Gemma 4 because MLX-LM baselines exist for Llama 3.2 1B and Qwen3 0.6B |
| 3 | More dense: SmolLM3, Phi, GLM dense, long-context Qwen | YaRN and partial rotary RoPE, NoPE layers, other norm placements |
| 4 | Gemma 4 dense (12B) | `global_head_dim` 512 with one global KV head, `attention_k_eq_v`, partial "proportional" RoPE on global layers, final logit softcap 30, 5:1 sliding pattern with a 1024 window. Done on the CPU and Metal backends (Metal verified for correctness on hosted CI's virtual GPU) |
| 5 | Gemma 4 E2B/E4B | Per-layer input embeddings, KV sharing across layers. Done on the CPU and Metal backends with wave 4 |
| 6 | MoE: Qwen3-30B-A3B, GPT-OSS 20B/120B, Gemma 4 26B-A4B, GLM-4.5/4.6-Air | Routers, expert deduplication across the batch, MXFP4 (GPT-OSS), attention sinks, alternating window 128. Qwen3-30B-A3B, GPT-OSS 20B/120B with MXFP4, and GLM-4.5/4.5-Air (with DeepSeek V3's grouped sigmoid routing) done on the CPU and Metal backends (Metal verified for correctness on hosted CI's virtual GPU; routing on the GPU, gathered expert kernels; grouping prefill tokens by expert is next) |
| 7 | Large MoE: DeepSeek V3.x, Kimi K2, GLM-4.6 | MLA attention; needs a 256–512 GB Mac or SSD expert paging |
| 8 | Hybrid attention: Qwen3-Next | Gated DeltaNet linear attention state alongside full attention |
| 9 | VLMs: Gemma 3/4 vision, Qwen2.5/3-VL, GLM-4.5V | Vision encoders (ANE candidate), M-RoPE, image token packing |

Inputs stay portable: MLX safetensors first, then GGUF and plain safetensors.

## Fleet benchmarks (Common Compute)

The self-hosted Mac covers one chip. Fleet runs cover every chip family
(M1–M5, base/Pro/Max/Ultra). Proposed workload, to be wired into the provider
app once this session can read its repository:

```json
{
  "workload_id": "lokahi_bench",
  "payload": {
    "lokahi_revision": "<git sha>",
    "suite": "correctness | benchmark",
    "model": {"repo": "mlx-community/gemma-3-1b-it-qat-4bit", "revision": "<40-hex>"},
    "workload": {"prompt_tokens": 512, "output_tokens": 128, "batch_size": 1,
                  "warmups": 2, "repetitions": 5, "sampling": "greedy"}
  }
}
```

The provider runs a signed Lōkahi build and an MLX-LM control in alternating
order on an idle, powered machine. It returns the same JSON reports the
self-hosted workflow uploads, with the hardware record (chip, memory, macOS
build, thermal state). Reports become evidence per fingerprint; the planner
only promotes a plan for fingerprints that passed.

## Open decisions and actions

| Item | Owner | Notes |
|---|---|---|
| Register the self-hosted runner and set `LOKAHI_HW_RUNNER=enabled` | Tyler | [APPLE_SILICON_RUNNER.md](APPLE_SILICON_RUNNER.md) |
| Grant this project access to the Common Compute provider app repository | Tyler | Needed to add the `lokahi_bench` workload |
| Counsel review of the CLA and the copyright assignment from Tyler to Common Compute LLC | Tyler | No outside contributions until then |
| Rename the GitHub repository to `lokahi` (and consider moving it into the `commoncompute` organization and detaching it from the oLLM fork network) | Tyler | GitHub redirects the old URL; update `pyproject.toml` URLs afterwards |
| Trademark search for "Lōkahi" in software | Tyler | PyPI and the obvious GitHub names were free on 2026-10-09 |
