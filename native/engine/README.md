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
./build/engine/lokahi image-geometry --model VLM --width 1024 --height 768   # the processor's resize target
./build/engine/lokahi generate --model VLM --tokens-file prompt.txt --images photo.ppm --max-new 64
```

## Status

| Area | State | Evidence |
|---|---|---|
| Decoders on CPU: Gemma 3, Gemma 4 (E2B/E4B, 12B (`gemma4_unified`), 31B and 26B-A4B layouts), Llama 3.x, Mistral, Qwen 2/2.5, Qwen 3, Qwen 3 MoE, GPT-OSS, GLM-4.5, GLM-4.7-Flash, DeepSeek V2 and V3 (Kimi K2, Moonlight), Qwen3.5, Qwen3.5-MoE, Qwen3-Next, and the text models of Qwen3-VL, Qwen3-VL-MoE, Qwen2.5-VL and Qwen2-VL | Implemented | Correctness: matches the NumPy oracles (`oracles/gemma3.py`, `oracles/gemma4.py`, `oracles/llama.py`, `oracles/deepseek.py`, `oracles/qwen3_5.py`) within 2e-4 relative; greedy tokens exact (`tests/test_native_engine.py`). Qwen 3 MoE fixtures cover stacked (MLX) and per-expert (Transformers) weights, renormalized and raw top-k, dense layers between sparse ones, and route far from ties on the router's real inputs. GPT-OSS fixtures mirror the 20B layout (MXFP4 experts behind affine overrides, sinks, alternating windows, untruncated YaRN) and are guarded so the SwiGLU clamps bind and the sinks move the logits; the published 20B config parses to the expected layer map, router and YaRN ramp. GLM-4.5 fixtures cover the Air layout and DeepSeek V3-style group-limited routing, and are guarded so the groups and the correction bias each change some token's experts. Gemma 4 fixtures cover KV sharing of both layer kinds, `attention_k_eq_v`, partial RoPE, per-layer inputs and layer scalars, and every non-equivalent mutation of those features fails the suite. The published E2B and 12B configs parse to the expected layer maps, and 12B's defaults are Transformers' `Gemma4UnifiedTextConfig`'s (`tests/engine_tests.cpp`, `tests/test_mlx_crosscheck.py`); 12B's `gemma4_unified_vision` tower is not supported yet, which `info` reports and image requests refuse. DeepSeek fixtures cover both MLA layouts (`kv_b_proj` and MLX's absorbed `embed_q`/`unembed_out`), with and without a query rank, interleaved and rotate-half RoPE, a YaRN attention factor other than 1, V3's grouped routing with negative correction biases (so excluding dropped groups differs from zeroing them) and V2's greedy and group-limited softmax routing, quantized and BF16; the oracle expands keys and values per head as Transformers does while the engine attends in the latent space. Qwen3.5 fixtures cover MLX's and Transformers' layouts (norms stored as `1 + w` or `w`, both conv layouts, `model.language_model.`, fused `gate_up_proj` experts), Qwen3-Next's grouped projections, one and two value heads per key head, prefill split across passes (the conv history and recurrent state carry over) and reset, and are guarded so the decay, `dt_bias`, the conv's earlier taps, the output gate, the shared expert's gate and the norm offset each move the logits; the published Qwen3.5-0.8B, Qwen3.5-35B-A3B and Qwen3-Next-80B-A3B configs parse to the expected layer maps |
| Decoders on Metal (every family above) | Implemented; correctness verified on a virtual GPU | Hosted macOS CI runs the Metal suites against the oracles on its paravirtual device (correctness only, no performance claims); hardware runs wait for the self-hosted runner. Gemma 4 covers 512-wide heads, KV sharing, `attention_k_eq_v`, the V norm, partial RoPE, per-layer inputs and layer scalars. Mixture-of-experts layers route on the GPU (`moe_route`) and run each token's experts with gathered matrix-vector kernels over stacked expert weights, affine or MXFP4. GPT-OSS adds sinks in the attention reduction, attention and expert biases and the clamped SwiGLU; GLM-4.5 adds a dense router kernel, grouped sigmoid routing (`moe_route_grouped`), the shared expert and classic partial RoPE. Gemma 4 26B-A4B runs its experts beside the dense MLP, routing on its own normed input with per-expert weight scales (`moe_route`), and sums the two post-normed branches (`residual_norm_rows`). Qwen3.5 and Qwen3-Next run Gated DeltaNet in five kernels: Qwen3-Next's grouped projections are split (`gdn_split`); the causal conv and SiLU run one thread per channel (`gdn_conv`); q and k are L2-normalized (`gdn_qk_norm`); beta and the decay are computed (`gdn_gates`); each simdgroup runs one value column of a head's state over the pass, with its rows spread across the lanes (`gdn_recurrence`); then the gated norm (`gated_rms_norm_rows`) and `out_proj`. The conv histories and states are double-buffered by position parity, so a decode step launched past a stop token and discarded never overwrites the state the next step reads; full attention gates its output and the shared expert is gated (`mul_sigmoid`). DeepSeek's latent attention absorbs each head's query with the gathered matrix-vector kernel over MLX's `embed_q`, or straight from `kv_b_proj`'s quantized key rows (`qmv_t_heads`), writes normed, rotated latents into a 576-float cache per token (`mla_rope_store`), attends with the split-key kernel four query heads at a time over the one latent key head (`attn_partial_g4_d18`), and expands each head's output with the gathered kernel; V2's group-limited softmax routing (`moe_route`) and V3's group exclusion (`moe_route_grouped`) run on the GPU for up to 512 experts (Kimi K2 has 384). Unquantized (BF16) checkpoints run on the CPU. MXFP4 outside experts is refused (`--backend auto` picks the CPU) until the prefill matrix kernel and embedding gather read it |
| Gemma 3 images on CPU and Metal (SigLIP tower and projector, `src/vision.h`, `src/metal/metal_vision.mm`; `--images`, `lokahi image-features`) | Implemented; Metal verified for correctness on a virtual GPU | Correctness: features, logits and greedy tokens match the NumPy oracle (`oracles/siglip.py`, Transformers' semantics) on a tiny vision-language fixture in mlx-community's layout, with prefill chunks that would cut an image, a window shorter than an image, and both convolution layouts; every mutant of the masks, the injection, the chunking and the tower fails the suite (`tests/test_gemma3_vision.py`). The published 4B config parses (`tests/engine_tests.cpp`). Hosted CI's virtual GPU checks the Metal encoder and the Metal decoder's injections against the same oracle, including matrix products over several tiles (correctness only). The CPU encoder is a correctness path: 896-pixel images cost about 5 TFLOP on its scalar matmul |
| Gemma 4 images on CPU and Metal (variable-resolution tower, `src/gemma4_vision.h`) | Implemented; Metal verified for correctness on a virtual GPU | Correctness: features, logits and greedy tokens match the NumPy oracle (`oracles/gemma4_vision.py`, Transformers' semantics) on a tiny fixture in mlx-community's 31B layout (BF16 tower, quantized embedder): two aspect ratios, chunked prefills that would cut an image, a window shorter than an image, and causal images without `use_bidirectional_attention: "vision"`. Every mutant of the block rule, the 2-D RoPE, the norms, the pooling, the scaling and the patches fails the suite (`tests/test_gemma4_vision.py`); the published 31B config and processor sizes parse. Hosted CI's virtual GPU runs the same tests on the Metal encoder and decoder (correctness only). E2B/E4B (per-layer inputs) look up the pad token's per-layer embedding at image positions and project the features, as Transformers does |
| Qwen3.5 images on CPU and Metal (Qwen3-VL's tower without deepstack, `src/qwen3_5_vision.h`; interleaved M-RoPE in both decoders) | Implemented; passes on hosted CI's virtual GPU (correctness only) | Correctness: features, logits and greedy tokens match the NumPy oracle (`oracles/qwen3_5_vision.py`) for Qwen3.5 and Qwen3.5-MoE in both checkpoint layouts, with two images of opposite aspect, prefill passes that end inside an image, and decoding past them (`tests/test_qwen35_vision.py`). The fixture is guarded so that row positions, column positions, swapping them, and the offset of the text after an image each move the logits by more than the Metal tolerance, and every mutant of the positions, the planner, the config and the tower fails the suite. The oracle matches Transformers' `Qwen2VLImageProcessorPil` (resize rule and patch layout) and MLX-VLM 0.4.4 (tower, positions and single-image logits; `tests/test_mlx_vlm_crosscheck.py`). The published 0.8B vision and RoPE configs parse (`tests/engine_tests.cpp`). Hosted CI's virtual GPU runs the same tests on the Metal tower and decoder (correctness only) |
| Qwen3-VL and Qwen3-VL-MoE images on CPU and Metal (the Qwen3.5 tower with deepstack mergers; deepstack rows in both decoders) | Implemented; passes on hosted CI's virtual GPU (correctness only) | Correctness: features, both deepstack feature sets, logits and greedy tokens match the NumPy oracle (`oracles/qwen3_vl.py`) for Qwen3-VL in MLX's and Transformers' layouts and Qwen3-VL-MoE in MLX's, with two images, prefill passes that end inside an image, and text alone (`tests/test_qwen3_vl.py`). Deepstack, each M-RoPE axis, and each deepstack layer alone move the fixture's logits by 7% or more, and every mutant of the deepstack placement, the mergers, the config and the planner's checks fails the suite. The oracle matches MLX-VLM 0.4.4's pieces (`tests/test_mlx_vlm_crosscheck.py`); the published Qwen3-VL-2B config parses, in both RoPE field layouts (`tests/engine_tests.cpp`). Transformers' fused, transposed Qwen3-VL-MoE experts fail closed |
| Qwen2.5-VL images on CPU and Metal: the window-attention tower (`src/qwen3_5_vision.h`), chunked M-RoPE in both decoders | Implemented; passes on hosted CI's virtual GPU (correctness only) | Correctness: features, logits and greedy tokens match the NumPy oracle (`oracles/qwen2_5_vl.py`) in Qwen's published layout (Transformers 4: a flat config, `visual.*`), MLX's and Transformers 5's, with two images whose windows are whole and cut by the edges, prefill passes that end inside an image, and text alone (`tests/test_qwen2_5_vl.py`). Windows, the full-attention block, each M-RoPE axis and the chunked layout each move the fixture's outputs by 6% or more. The oracle matches MLX-VLM 0.4.4 (window order, tower, positions and single-image logits) and Transformers (`smart_resize` at the published factor 28, the processor, and its re-save of the flat config; `tests/test_mlx_vlm_crosscheck.py`); the published Qwen2.5-VL-3B config parses (`tests/engine_tests.cpp`) |
| Qwen2-VL images on CPU and Metal: the Qwen tower with LayerNorms and QuickGELU MLPs, Qwen2.5-VL's text model | Implemented; the Metal path awaits its first virtual-GPU run | Correctness: features, logits and greedy tokens match the NumPy oracle (`oracles/qwen2_5_vl.py`, `Qwen2VLReference`) in Qwen's published, MLX's and Transformers 5's layouts, with the Qwen2.5-VL tests' images and prompts (`tests/test_qwen2_5_vl.py`). Each M-RoPE axis moves the fixture's logits by 4% or more. The oracle matches MLX-VLM 0.4.4's `qwen2_vl` (tower on five shapes, positions and single-image logits) and Transformers' re-save of the flat config; the published Qwen2-VL-2B config parses (`tests/engine_tests.cpp`) |
| Oracles vs MLX-LM (`gemma3_text`, `gemma4`, `llama`, `qwen2`, `qwen3`, `qwen3_moe`, `gpt_oss`, `glm4_moe`, `glm4_moe_lite`, `deepseek_v2`, `deepseek_v3`, `qwen3_5`, `qwen3_5_moe`, `qwen3_next`) | Implemented | Correctness: hosted macOS CI runs `tests/test_mlx_crosscheck.py` against pinned MLX 0.32.3 and MLX-LM 0.32.0 in float32. MLX-LM adds the zero-centered norms' 1 to Transformers-layout BF16 weights in BF16, so those Qwen3.5 cases run on exact float32 copies of the weights |
| MLX affine 2/3/4/5/6/8-bit weights, BF16/F16 scales, per-module bit overrides | Implemented | Unit tests at every width and group size; MLX cross-check (`mx.quantize`/`mx.dequantize`, MLX-LM logits at 3, 5 and 6 bits) |
| MLX MXFP4 weights (E2M1 codes, E8M0 scale per 32; GPT-OSS's expert format), on CPU and in Metal's matrix-vector and expert kernels | Implemented | Correctness: unit tests against hand-computed rows, the OCP encodings, and oracle parity on MXFP4 MLPs and experts (`tests/test_native_engine.py`); `mx.dequantize` cross-check on Apple Silicon. Per-module overrides follow `mlx.nn.quantize`: keys an override omits take its mode's defaults. Metal reads MXFP4 experts (GPT-OSS) and refuses MXFP4 elsewhere, since its prefill matrix kernel and embedding gather are affine only |
| Tokenizer: SentencePiece-style BPE (Gemma, Llama 2, Mistral) from `tokenizer.json` | Implemented | Correctness: identical ids and text to Hugging Face `tokenizers` on trained fixtures (byte fallback, unknown fusion, added tokens, Metaspace) and a 262k-entry synthetic vocabulary (`tests/test_tokenizer.py`) |
| Tokenizer: byte-level BPE (Llama 3, Qwen, GPT-OSS, DeepSeek, GLM) | Implemented | Correctness: identical words, ids and text to Hugging Face `tokenizers` for the Qwen 3, Llama 3, GLM, GPT-OSS o200k, DeepSeek V3 and GPT-2 configurations on text from every Unicode plane (`tests/test_tokenizer_bytelevel.py`); regex engine fuzzed against the reference's Oniguruma |
| Streaming text with stop strings (`TextStream`, `lokahi run`) | Implemented | Emits only text later tokens cannot rewrite, cut at the earliest stop string, at constant cost per token. Correctness: equals Hugging Face `tokenizers`' full decode (cut at the stop) on every trained tokenizer configuration (`tests/test_text_stream.py`) |
| Common Compute service (`src/service.h`, `lokahi_service_*` C ABI, `lokahi serve`, Swift XPC package in `integrations/commoncompute`) | Implemented | Correctness (acceptance gate B, gate C's parity on a fixture). Every emitted document passes the kit's reference verifier and the stream rules. Output equals the NumPy model with the jinja2-rendered prompt, and usage counts are exact. Covers seeded sampling, stop strings, every rejection path, cancelling queued and running requests, deadlines, drain and duplicates (`tests/test_service.py`, also under ThreadSanitizer and AddressSanitizer). The XPC tests cross a real NSXPC connection on hosted macOS |
| Chat templates: native Jinja subset (`src/chat_template.h`, `lokahi chat-template`) | Implemented | Correctness: byte-identical to jinja2 set up as transformers sets it up, for the real templates of Qwen3, Llama 3.2, Gemma 3, Mistral v0.3, DeepSeek-R1 distill, Qwen2.5, SmolLM3, Phi-4-mini, GLM-4 and GPT-OSS harmony (`tests/test_chat_template.py`, fixtures in `tests/fixtures/chat_templates`). Syntax outside the subset fails at parse time, and operations Python would compute differently (Unicode case mapping, big integers, `%` formatting) fail at render time |
| Lōkahi wire v1 codec and stream rules (`src/wire.h`, `lokahi wire-check`) | Implemented | Correctness: the kit's golden fixtures, and about 22,000 seeded mutations, get the same verdict as `verify_kit.py`, except two stricter rules documented in `wire.h` (UTF-8 only, integers within int64). Encoders round-trip every valid fixture and refuse invalid structs (`tests/test_wire_native.py`) |
| Speculative greedy decoding (`src/speculation.h`, `generate --speculate N`): prompt-lookup drafts from the prompt, the output and an optional reference, verified in one batched pass | Implemented on the CPU and Metal. Metal's verify pass writes the Gated DeltaNet buffers it did not read, so the state before the pass survives without a copy, and a rejection replays the kept tokens from it on the GPU | Correctness: the output equals plain greedy decoding on every fixture, with drafts both accepted and rejected; a stop token inside an accepted run and a limit inside a pass leave the model where plain decoding would (sliding windows, Gated DeltaNet states rebuilt from the state before the pass and the accepted tokens' recorded inputs, Qwen3-Next's grouped projections, and latents); sampled generation does not speculate (`tests/test_native_engine.py`, on both backends). No speedup is claimed yet: it needs Apple Silicon hardware runs |
| Qwen3.5 multi-token prediction drafts (`DecoderWeights::mtp`, `generate --speculate N --drafter mtp|both [--draft-model DIR]`): Qwen's `mtp.*` head beside the decoder, or MLX-VLM's split `qwen3_5_mtp` drafter | Implemented on the CPU (dense, experts, Qwen3-Next). The head follows every pass with only its fc and key/value projections, so its rows always come from the decoder's hidden states (as vLLM rewrites them); each draft is one step of its layer and lm_head. Metal refuses it for now; images fail closed | Correctness: after speculative runs that reject, accept, stop and hit limits, the head's next drafts and logits equal the NumPy oracle's full recomputation (within 1e-4 relative) in MLX, Transformers, quantized-5-bit, MoE and Qwen3-Next layouts and with a split drafter beside MLX's and Transformers' decoders, and the output stays plain greedy decoding's (`tests/test_mtp.py`); the oracle matches MLX-VLM 0.7.8's `qwen3_5_mtp` drafter loading the same checkpoints and drafter directories (`tests/test_mlx_vlm_mtp_crosscheck.py`). Acceptance and speed need real checkpoints on hardware |
| Seeded sampling: temperature, top-p, top-k, min-p (CPU and a GPU kernel) | Implemented; GPU kernel awaiting first hardware run | Correctness: draws equal an independent float64 oracle (`oracles/sampling.py`) and follow the exact distribution (chi-square); Philox matches Random123's known answers (`tests/test_sampling.py`) |

No throughput claim is made until the self-hosted runner records hardware
evidence against a fresh MLX-LM control.

## Design

**Model families.** `DecoderConfig` describes how the supported families
differ. Both backends implement every option:

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
| Mixture of experts | no | 26B-A4B: experts beside every dense MLP, routed on `rms(x) · scale · hidden^-0.5` by the softmax of the top-k logits times `per_expert_scale`, GELU-gated, with their own pre/post norms | no | no | no | Qwen 3 MoE: a softmax router picks the top `num_experts_per_tok` SiLU-gated experts, renormalized with `norm_topk_prob`, in the layers `decoder_sparse_step` and `mlp_only_layers` select; MLX's stacked `switch_mlp` or per-expert weights |

GPT-OSS is the Llama-family parser plus: q/k/v/o biases; a learned sink
logit per head in every softmax; alternating sliding (128) and full layers;
YaRN as Transformers computes it, honoring `truncate: false` (MLX-LM always
rounds the correction range); and in every layer a biased router whose top-4
logits are softmaxed over 32 biased experts with the clamped SwiGLU
`min(g, 7)·σ(1.702·min(g, 7))·(clip(u, −7, 7) + 1)`. YaRN's attention factor
scales q and k in the references; the engine folds its square into the
attention scale, which gives the same logits on every backend.

GLM-4.5 (`glm4_moe`, the Air and full models) is the Llama-family parser plus
q/k/v biases, optional q/k norms, RoPE on the first half of each head
(rotate-half within it), `first_k_dense_replace` dense layers, and MoE layers
whose dense sigmoid router chooses experts by score plus
`e_score_correction_bias` within the best `topk_group` of `n_group` groups
(DeepSeek V3's `noaux_tc`) but weights them by the raw scores, renormalized
and times `routed_scaling_factor`, next to an always-on shared expert.

DeepSeek V2 and V3 (`deepseek_v2`, `deepseek_v3`; Kimi K2's `kimi_k2` and
Moonlight read as V3, and GLM-4.7-Flash's `glm4_moe_lite` is V3 whose dense
and sparse layers may be listed in `mlp_layer_types`) replace attention with
multi-head latent attention.
Each token caches one latent of `kv_lora_rank + qk_rope_head_dim` floats (576
for every released model, against 40,960 for V3's 128 expanded heads): the
RMS-normed output of `kv_a_proj_with_mqa` plus one RoPE'd key every head
shares. Queries come from `q_proj`, or `q_b_proj(rms(q_a_proj(x)))` with a
query rank. The engine never expands keys or values: each head's unrotated
query is absorbed into the latent (`embed_q`, the transposed key rows of
`kv_b_proj`), attention runs over the cache as one key head whose first
`kv_lora_rank` dimensions are also the value, and each head's latent output
is expanded by its value rows (`unembed_out`) before `o_proj`. MLX's recent
conversions store `embed_q` and `unembed_out` as such; `kv_b_proj` is read in
place, its value rows as `unembed_out` and its key rows applied transposed to
absorb the query. RoPE covers only the rope dimensions, as interleaved pairs
(V2, and V3 unless `rope_interleave` is false) or halves; YaRN's attention
factor scales those dimensions alone, and `mscale_all_dim` squared scales the
softmax. The latent norms use epsilon 1e-6 whatever `rms_norm_eps` says.
Leading `first_k_dense_replace` layers are dense; the rest add
`n_shared_experts` always-on experts to the routed ones. V3 routes as GLM-4.5
does, except that experts of dropped groups are excluded (Transformers and
DeepSeek's own code; MLX-LM scores them 0, which differs only when kept
experts score below 0). V2 takes the top probabilities of a softmax
(`greedy`, or within the `topk_group` groups with the largest maxima under
`group_limited_greedy`) times `routed_scaling_factor`. The routing fields are
required, because Transformers and MLX-LM default them differently. FP8
checkpoints (DeepSeek's and Moonshot's originals) need converting first.

Qwen3.5 (`qwen3_5`, `qwen3_5_moe`: the text models of the multimodal
checkpoints) and Qwen3-Next (`qwen3_next`) put three Gated DeltaNet layers
before every full attention layer (`full_attention_interval`; `layer_types`
must agree, as MLX-LM reads only the interval). A Gated DeltaNet layer keeps
no keys or values. Instead it keeps the last `linear_conv_kernel_dim - 1`
inputs of a depthwise causal conv over its `[q | k | v]` channels, and per
value head a `[linear_key_head_dim, linear_value_head_dim]` state. Per token:
the conv, then SiLU; q and k L2-normalized per key head and q scaled by
`linear_key_head_dim^-0.5`; the state decays by `exp(g)`, with
`g = -exp(A_log) · softplus(a + dt_bias)`, takes `k (β (v − Sᵀk))ᵀ`, with
`β = sigmoid(b)`, and outputs `Sᵀq`. A gated RMSNorm `rms(o) · w · silu(z)`
then feeds `out_proj`. Each value head reads key head
`h / (linear_num_value_heads / linear_num_key_heads)`. The full attention
layers gate their output by `sigmoid` of a second half of each head's
`q_proj` rows, norm q and k per head, and rotate halves within the first
`partial_rotary_factor` (0.25) of each head. Qwen3.5's RoPE is interleaved
M-RoPE: rotary frequency i reads a token's row position when `i % 3 == 1`
and `i < 3 · mrope_section[1]`, its column position when `i % 3 == 2` and
`i < 3 · mrope_section[2]`, and its temporal one otherwise. Text tokens'
three positions coincide, so text alone sees plain RoPE. Image tokens take
grid coordinates: an injection's `grids` gives each image's (rows,
columns), and an image starting at RoPE position `s` takes
`(s, s + row, s + column)`, row by row. The text after it continues at
`s + max(rows, columns)`, and from then on RoPE positions run that far
ahead of KV positions (`rope_offset`), through decoding and later
prefills, until `reset`. A later prefill's images take the positions a full
recompute of the conversation gives them (Transformers' cached path gives
them plain positions instead). Every RMSNorm but the gated one is zero-centered,
`x · rms · (1 + w)`; MLX conversions store `1 + w` and their conv weights as
`[C, K, 1]` rather than `[C, 1, K]`, which is how the load tells them apart.
Qwen3.5-MoE has experts in every layer, always renormalized; Qwen3-Next's
follow `decoder_sparse_step` and `mlp_only_layers`, renormalized with
`norm_topk_prob` (required, as Transformers and MLX-LM default it
differently). Both add a shared expert times `sigmoid(shared_expert_gate(x))`.
Transformers saves Qwen3.5 under `model.language_model.` with fused
`experts.gate_up_proj` tensors, both of which the load reads. The states
carry across prefill passes and decode steps until `reset`.

Qwen3-VL (`qwen3_vl`, `qwen3_vl_moe`) is Qwen 3 or Qwen 3 MoE from
`text_config` with the same interleaved M-RoPE, `[24, 20, 20]` by default.
As in Transformers, the sections come from `rope_scaling`, or without it
from Transformers 5's `rope_parameters`, and `rope_theta` from inside that
object, else beside it, else 500000.
Its images add deepstack rows (`PromptInjection::layer_rows`): layer l's
rows are added to the hidden states at the image tokens after decoder layer
l. On Metal, a deepstack layer's residual runs without the fused next-layer
norm, `add_rows_at` adds the rows, then the norm runs. Transformers stores
Qwen3-VL-MoE's experts fused and transposed (`[experts, hidden, 2 · moe]`),
which the load refuses; MLX-VLM's conversions store stacks.

Qwen2.5-VL (`qwen2_5_vl`) is Qwen 2 with chunked M-RoPE: the first
`section[0]` frequencies read the temporal position, the next `section[1]`
the row and the last `section[2]` the column (`[16, 24, 24]` by default),
and the sections must cover every frequency. Qwen's published configs keep
the text fields at the top level with `rope_scaling: {"type": "mrope"}`;
Transformers 5 saves `text_config` with `rope_parameters`, read with
Qwen3-VL's precedence (default theta 1000000). Transformers splits
`hidden_size` evenly across the heads whatever `head_dim` says, so another
`head_dim` fails closed, as do sliding-window layers. Qwen2-VL (`qwen2_vl`)
has the same text model, read the same way.

A family is a config parser in `decoder.cpp`. Options outside this table
(o_proj or MLP biases outside GPT-OSS, longrope and other RoPE types,
partial rotary outside Gemma 4, other activations) fail at load.

Gemma 4's scalar constants (the embedding scale and the per-layer input
scales) are rounded to BF16, as MLX-LM applies them to BF16 activations.
Transformers rounds the embedding scales the same way but keeps the per-layer
input scales in float32; those differ by about 1e-4, and two of the three
cancel in the RMSNorm that follows.

**Images.** A `VisionEncoder` (CPU or Metal, following `--backend`) takes an
image already resized to a size the model accepts (`image-geometry` reports
the processor's choice; PPM files carry their size) and returns one feature row
per image token. Gemma 3's encoder runs the processor's rescale and
normalization on pre-resized RGB8 pixels, then the SigLIP tower (patch
convolution, learned positions, pre-LayerNorm encoder layers with
bidirectional attention, tanh GELU, a final LayerNorm), then the projector
(average-pool to `mm_tokens_per_image`, RMSNorm `1 + w`, projection into the
text width). `place_images` turns the features into a `PromptInjection` at
the prompt's runs of `image_token_index`, which `Model::inject` hands to the
next prefill: the rows replace those tokens' scaled embeddings as they are,
and each image's tokens also attend to its later tokens. The sliding window
still applies, as in Transformers' `AND(window, OR(causal, block))`. A
prefill pass never splits such an image; images longer than
`--prefill-chunk` fail closed. Models whose images stay causal (Gemma 4
without bidirectional vision, Qwen3.5) let passes split images. Both
backends share these checks (`plan_injection`). Gemma 3
applies the image blocks on every layer; Gemma 4 only on sliding layers, and
only with `use_bidirectional_attention: "vision"` (full layers stay causal, as
Transformers' `create_masks_for_vision_model` builds them). Gemma 4's tower
takes any size whose sides are multiples of `patch_size * pooling_kernel_size`
within the soft-token budget: patches mapped to `2 * (p - 0.5)`, learned x and
y positions, axial 2-D RoPE with per-head Q/K norms and scale 1, `k x k`
pooling times `sqrt(hidden)`, standardization and the embedder. The Metal
encoder runs every matrix product through one tiled kernel (`dense_mm`) that
reads the checkpoint's dense F32, F16 or BF16 weights as stored, including
attention's: per head, `Q K^T`, a row softmax, then `P V`. MLX-VLM scales the features differently and builds no image
mask, so the oracle follows Transformers, and there is no MLX cross-check for
this path.

Qwen3.5's tower takes images whose sides are multiples of
`patch_size · spatial_merge_size` (32) and whose area lies within the
processor's `shortest_edge` and `longest_edge` pixels (`image-geometry`
follows `smart_resize`, which rounds half to even). The image is rescaled
and normalized and its patches embedded by the Conv3d kernel summed over its
two frames (an image's frames are equal). Learned positions are interpolated
bilinearly from a square table. Each block applies LayerNorms, biased qkv,
2-D RoPE over the whole head (rows, then columns) with every patch seeing
every patch, and a tanh-GELU MLP; the merger concatenates each 2 × 2 square
of patches through an exact-GELU MLP into the text width. Patches run in
merge-block order, so the image's tokens fill its merged grid row by row.
The encoder reports that grid (`token_grid`), which `place_images` passes
in the injection for the decoder's M-RoPE positions. The Metal tower runs
the same steps with every product in `dense_mm`: q, k and v stay in one
`[patches, 3 · hidden]` buffer that the RoPE (`rope_half_rows`) and the
per-head attention products read with a stride, and the merger's exact GELU
(`gelu_erf_inplace`) uses Numerical Recipes' Chebyshev fit of erfc (error
below 1.2e-7), since the Metal library has no erf. Its buffers grow to the
largest image encoded so far, except one head's attention scores: each
pass covers at most `--vision-attention-rows` query rows (by default as
many as fit 128 MiB of scores), so the processor's largest images (65536
patches) no longer need a 16 GiB score matrix. MLX-VLM 0.4.4 places a second image's
positions differently from Transformers (it sums every `vision_start`
index), so multi-image prompts follow Transformers and are cross-checked
with MLX-VLM for one image only. Qwen3-VL's tower adds deepstack: after
each block in `deepstack_visual_indexes` (increasing, inside the tower), a
merger of its own that LayerNorms each concatenated square instead of each
patch gives one more set of rows, which `encode_features` returns
(`ImageFeatures::deepstack`) and `image-features` writes after the rows.
MLX-VLM 0.4.4 drops deepstack when it chains its pieces (it keeps
`mx.eval`'s `None` as the features, and its MoE model's
`_deepstack_process` returns its input) and swaps the MoE tower's two
GELUs, so the oracle follows Transformers and is cross-checked against
MLX-VLM's pieces with deepstack passed in. Qwen2.5-VL's tower shares the
processor (CLIP's mean and std), the patch order and the 2-D RoPE, but uses
RMSNorms, `down(silu(gate) · up)` MLPs with biases, no patch bias or learned
positions, and an `ln_q` merger; outside `fullatt_block_indexes` each patch
attends only within its window, the merged grid cut into squares of
`window_size // merge // patch` tokens from the top-left corner. The
CPU encoder groups patches by window rather than permuting them as
Transformers does (attention within a window does not depend on order). The
Metal tower permutes them, so each window is a run of rows: the host writes
the patches and RoPE rows window by window with each row's window bounds,
`window_attention` attends each patch over its window in one dispatch per
block (windows of up to 1024 patches; the published ones hold 64), the
full-attention blocks keep the per-head products, and the features return to
the merged grid's order on readback. Qwen2-VL's tower is Qwen3.5's without
learned positions or a patch bias: LayerNorms, `fc2(quick_gelu(fc1))` MLPs
(QuickGELU is `x · sigmoid(1.702 x)`, `quick_gelu_inplace` on Metal),
global attention, and Qwen2.5-VL's merger with a LayerNorm `ln_q`. Its
`vision_config` calls the tower's width `embed_dim`, sizes the MLP by
`mlp_ratio` and gives the text width as `hidden_size`.

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

**KV cache.**
- Global layers hold `max_context` slots. Sliding-window layers hold
  `window + prefill_chunk` slots in a ring, so a whole chunk can be written
  before its queries read the preceding window.
- Keys and values are stored as FP16 by default (`--kv-cache auto`, or
  `"kv_cache"` in the service config), as llama.cpp stores them. Decode reads
  every visible key and value once per token, so FP16 halves that traffic and
  the cache's memory; at long contexts it rivals the weights. `--kv-cache f32`
  keeps float32 for comparison.
- Latent attention's cache stays float32.
- The CPU backend keeps float32 unless `--kv-cache f16` asks it to round keys
  and values as Metal stores them.

**Numerics.** Activations, norms and attention are float32, and the KV cache is
FP16 as above. Weights stay packed (2 to 8 bits, or MXFP4) and are dequantized
inside the kernels.

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

- **Metal:** embed, norm, qkv, rope_kv, attention, o_proj, residual_norm, mlp
  (or router and experts), per_layer_input (Gemma 4), lm_head and sample. Each section is its own command buffer, and its time is
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
