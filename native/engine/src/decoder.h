// Decoder-only transformers (Gemma 3, Gemma 4, Llama 3.x, Mistral, Qwen 2/2.5,
// Qwen 3, Qwen 3 MoE, GPT-OSS, GLM-4.5, DeepSeek V2 and V3, Qwen3.5 and
// Qwen3-Next): configuration, weight binding, and the CPU backend.
// One configuration describes how the families differ (norm style and
// placement, activation, per-head Q/K norm, q/k/v biases, RoPE bases,
// scaling and partial rotation, sliding windows, logit soft-capping,
// Gemma 4's per-layer geometry, KV sharing and per-layer inputs, DeepSeek's
// multi-head latent attention, Qwen3.5's Gated DeltaNet layers and gated
// attention, and mixture-of-experts layers), so a family is a config parser
// rather than a fork. Both backends implement every option; the Metal backend
// refuses only matrix formats its kernels lack.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "model.h"
#include "quant.h"
#include "safetensors.h"

namespace lokahi {

// ClampedSwiglu is GPT-OSS's expert activation:
// min(g, L) * sigmoid(1.702 * min(g, L)) * (clamp(u, -L, L) + 1).
enum class Activation { GeluTanh, Silu, ClampedSwiglu };

// RoPE frequency scaling from `rope_scaling` (Hugging Face semantics).
struct RopeScaling {
  enum class Kind { None, Linear, Llama3, Yarn } kind = Kind::None;
  double factor = 1.0;
  double low_freq_factor = 1.0;      // llama3
  double high_freq_factor = 4.0;     // llama3
  double original_max_positions = 8192.0;  // llama3, yarn
  double beta_fast = 32.0;           // yarn
  double beta_slow = 1.0;            // yarn
  bool truncate = true;              // yarn: round the correction range outward
};

struct DecoderConfig {
  std::string architecture;  // model_type family, e.g. "gemma3" or "qwen3"
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float rms_norm_eps = 1e-6f;
  // RMSNorm multiplies by (norm_offset + weight): Gemma stores weights
  // around 0 and uses an offset of 1, most families use 0.
  float norm_offset = 0.0f;
  // Gemma also normalizes each sublayer's output before the residual add.
  bool sandwich_norms = false;
  bool qk_norm = false;   // per-head RMSNorm on q and k before RoPE
  bool qkv_bias = false;  // biases on the q, k and v projections (Qwen 2)
  Activation activation = Activation::Silu;
  float embed_scale = 1.0f;
  float attn_scale = 0.0f;
  float rope_theta = 1e4f;
  RopeScaling rope_scaling;
  // Gemma 3's sliding layers use their own unscaled RoPE base.
  bool local_rope = false;
  float rope_local_theta = 1e4f;
  int sliding_window = 0;
  std::vector<bool> layer_is_sliding;
  float final_logit_softcapping = 0.0f;  // 0 disables
  float attn_logit_softcapping = 0.0f;
  bool o_bias = false;           // a bias on the attention output projection (GPT-OSS)
  bool attention_sinks = false;  // a learned per-head logit in each softmax that attends to nothing
  // Which layers let an image's tokens see the image's later tokens during
  // prefill: all of them (Gemma 3), sliding layers only (Gemma 4 with
  // use_bidirectional_attention "vision"), or none (images stay causal).
  enum class ImageBlocks { None, AllLayers, SlidingLayers } image_blocks = ImageBlocks::None;
  bool image_blocks_apply(bool sliding) const {
    return image_blocks == ImageBlocks::AllLayers || (image_blocks == ImageBlocks::SlidingLayers && sliding);
  }
  // Multimodal RoPE (Qwen3.5's interleaved M-RoPE): each token has temporal,
  // row and column positions, and rotary frequency i reads the position on
  // axis rope_axes[i] (0, 1 or 2). Empty: one position per token. Text
  // tokens' three positions coincide, so text alone sees plain RoPE.
  std::vector<int> rope_axes;
  // Qwen3-VL's deepstack: injected images also add rows to the hidden states
  // after the first layers.
  bool deepstack = false;
  InjectionRules injection_rules(int rope_offset) const {
    return {image_blocks != ImageBlocks::None, !rope_axes.empty(), rope_offset, deepstack ? num_layers : 0};
  }
  std::vector<int32_t> eos_token_ids;
  std::string weight_prefix;  // "" or "language_model."

  // Gemma 4. Per-layer vectors are empty when every layer uses the fields
  // above.
  std::vector<int> layer_head_dim;
  std::vector<int> layer_kv_heads;
  std::vector<int> layer_intermediate;
  // The layer whose keys and values a layer attends over: itself, or for
  // KV-shared layers the last earlier layer of the same kind.
  std::vector<int> layer_kv_source;
  std::vector<bool> layer_k_eq_v;   // values are the raw key projection
  bool v_norm = false;              // scale-free RMSNorm on values
  // Full layers rotate only this fraction of their dimensions, with
  // exponents over the whole head size ("proportional" RoPE).
  float global_partial_rotary = 1.0f;
  // Classic partial RoPE (GLM): rotate-half within the first rotary_dim
  // dimensions, exponents over rotary_dim, the rest untouched; 0 rotates the
  // whole head.
  int rotary_dim = 0;
  int per_layer_input = 0;          // hidden_size_per_layer_input; 0 disables
  // Per-layer input constants as the references apply them: the lookup times
  // sqrt(P), the projection times hidden^-0.5, their sum times 2^-0.5.
  float per_layer_embed_scale = 1.0f, per_layer_projection_scale = 1.0f, per_layer_input_scale = 1.0f;
  bool layer_scalars = false;       // each layer's output times its layer_scalar
  // The token whose per-layer embedding image positions look up (Gemma 4,
  // as Transformers replaces image tokens with the pad token there).
  int pad_token_id = 0;

  // Mixture of experts (Qwen 3 MoE). In layers marked in layer_moe, a
  // softmax router picks experts_per_token of num_experts gated experts
  // (renormalizing their weights when norm_topk_prob) instead of the MLP.
  int num_experts = 0;
  int experts_per_token = 0;
  int moe_intermediate = 0;
  bool norm_topk_prob = false;
  std::vector<bool> layer_moe;
  // Softmax: softmax over all experts, then the top k, weighted by their
  // probabilities times routed_scaling (Qwen 3 MoE, DeepSeek V2).
  // TopKSoftmax: the top k router logits, then a softmax over them (GPT-OSS).
  // SigmoidGrouped: sigmoid scores plus a correction bias choose the top k
  // within the best topk_group of n_group groups; the raw scores of the
  // chosen experts, renormalized, times routed_scaling, weight them (GLM-4.5,
  // DeepSeek V3's noaux_tc).
  enum class Router { Softmax, TopKSoftmax, SigmoidGrouped } router = Router::Softmax;
  int n_group = 1, topk_group = 1;
  float routed_scaling = 1.0f;
  int shared_intermediate = 0;  // width of the always-on shared expert; 0 for none
  // The shared expert's output times sigmoid(shared_expert_gate(x)), a
  // [1, hidden] projection, as Qwen 2 MoE gates it (Qwen3.5, Qwen3-Next);
  // its modules are then mlp.shared_expert rather than mlp.shared_experts.
  bool shared_expert_gate = false;
  std::string router_module = "mlp.gate";  // relative to the layer
  bool router_bias = false;
  bool expert_bias = false;     // biases on every expert projection
  float swiglu_limit = 7.0f;    // ClampedSwiglu
  // Gemma 4 (26B-A4B): the experts run beside the layer's dense MLP instead
  // of replacing it. The router reads rms(x) * router_input_scale of the
  // layer's residual input x, the experts read expert_norm(x), each branch
  // is post-normed (mlp_post_norm_1, expert_post_norm) and their sum goes
  // through the usual post-feedforward norm. Each routing weight is also
  // multiplied by its expert's per_expert_scale.
  bool experts_beside_mlp = false;
  // Experts of the groups SigmoidGrouped drops can never be chosen
  // (DeepSeek V3, as Transformers and DeepSeek's reference code mask them to
  // -inf) rather than competing with selection score 0 (GLM-4.5, MLX-LM).
  // A Softmax router with n_group > 1 keeps the topk_group groups with the
  // largest probabilities (DeepSeek V2's group_limited_greedy).
  bool exclude_dropped_groups = false;

  // Multi-head latent attention (DeepSeek V2 and V3), enabled by
  // kv_lora_rank > 0. Each token caches one latent: kv_lora_rank RMS-normed
  // dimensions every head shares plus a qk_rope_dim RoPE'd key. A head's
  // query has qk_nope_dim + qk_rope_dim dimensions (head_dim); its unrotated
  // part is absorbed into the latent space, so attention runs over the cache
  // as one key head of latent_width() whose first kv_lora_rank dimensions
  // are also the value, and each head's latent output is expanded to
  // v_head_dim before o_proj. rotary_dim is qk_rope_dim.
  int kv_lora_rank = 0;
  int q_lora_rank = 0;  // the query's own low rank; 0 for a direct q_proj
  int qk_nope_dim = 0, qk_rope_dim = 0, v_head_dim = 0;
  bool rope_pairs = false;   // rotate interleaved pairs rather than halves (MLA only)
  float rope_factor = 1.0f;  // YaRN's attention factor, on the RoPE'd dimensions only (MLA only)
  // The query and latent norms' epsilon, whatever rms_norm_eps is
  // (Transformers and MLX-LM construct them with 1e-6).
  float latent_norm_eps = 1e-6f;
  bool mla() const { return kv_lora_rank > 0; }
  int latent_width() const { return kv_lora_rank + qk_rope_dim; }

  // Gated DeltaNet (Qwen3.5, Qwen3-Next): the layers marked in layer_linear
  // replace attention with a linear-attention recurrence. A depthwise causal
  // conv1d of linear_conv taps, then SiLU, runs over the [q | k | v]
  // channels; q and k are L2-normalized per key head and q is scaled by
  // linear_k_dim^-0.5. Each of the linear_v_heads value heads reads key head
  // h / (linear_v_heads / linear_k_heads) and keeps a [linear_k_dim,
  // linear_v_dim] state S. Per token, S decays by exp(g), with
  // g = -exp(A_log) * softplus(a + dt_bias); it takes k (beta (v - S^T k))^T,
  // with beta = sigmoid(b); and it outputs S^T q. A gated RMSNorm per value
  // head, rms(o) * w * silu(z) with a plain weight, comes before out_proj.
  // The states and the conv's last linear_conv - 1 inputs carry across
  // passes until reset.
  std::vector<bool> layer_linear;
  int linear_k_heads = 0, linear_v_heads = 0, linear_k_dim = 0, linear_v_dim = 0, linear_conv = 0;
  int linear_channels() const { return 2 * linear_k_heads * linear_k_dim + linear_v_heads * linear_v_dim; }
  bool is_linear(int layer) const { return !layer_linear.empty() && layer_linear[static_cast<size_t>(layer)]; }
  bool has_linear() const {
    for (bool linear : layer_linear) {
      if (linear) return true;
    }
    return false;
  }
  // Qwen3.5's full attention: q_proj gives each head its query and then an
  // output gate (head_dim each), and the attention output is multiplied by
  // sigmoid(gate) before o_proj.
  bool attn_output_gate = false;

  static DecoderConfig from_json(const Json& raw);
  // RoPE inverse frequencies (half the layer kind's head size); rotations a
  // partial layer kind skips have frequency 0.
  std::vector<double> rope_inv_freq(bool sliding) const;

  int head_dim_of(int layer) const { return layer_head_dim.empty() ? head_dim : layer_head_dim[static_cast<size_t>(layer)]; }
  int kv_heads_of(int layer) const { return layer_kv_heads.empty() ? num_kv_heads : layer_kv_heads[static_cast<size_t>(layer)]; }
  int intermediate_of(int layer) const {
    return layer_intermediate.empty() ? intermediate_size : layer_intermediate[static_cast<size_t>(layer)];
  }
  int kv_source(int layer) const { return layer_kv_source.empty() ? layer : layer_kv_source[static_cast<size_t>(layer)]; }
  bool shares_kv(int layer) const { return kv_source(layer) != layer; }
  bool k_eq_v(int layer) const { return !layer_k_eq_v.empty() && layer_k_eq_v[static_cast<size_t>(layer)]; }
  bool is_moe(int layer) const { return !layer_moe.empty() && layer_moe[static_cast<size_t>(layer)]; }
};

bool is_supported_decoder(const Json& raw);

struct DecoderLayer {
  // attn_post_norm and mlp_post_norm exist only with sandwich norms;
  // q_norm and k_norm only with qk_norm.
  std::vector<float> attn_norm, attn_post_norm, mlp_norm, mlp_post_norm, q_norm, k_norm;
  std::vector<float> q_bias, k_bias, v_bias;  // only with qkv_bias
  std::vector<float> o_bias, sinks;            // only with o_bias, attention_sinks
  // k_proj and v_proj are empty (rows 0) in KV-shared layers, v_proj with k_eq_v;
  // gate_proj, up_proj and down_proj in mixture-of-experts layers.
  Linear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
  // Mixture of experts: the router [experts, hidden] and one view per expert;
  // biases [experts * rows] only with router_bias and expert_bias.
  Linear router;
  std::vector<Linear> expert_gate, expert_up, expert_down;
  std::vector<float> router_bias, expert_gate_bias, expert_up_bias, expert_down_bias;
  std::vector<float> router_correction;          // SigmoidGrouped's selection bias
  // With experts_beside_mlp: router.scale times hidden^-0.5, the per-expert
  // weight scales, and the extra norms.
  std::vector<float> router_input_scale, per_expert_scale, mlp_post_norm_1, expert_norm, expert_post_norm;
  Linear shared_gate, shared_up, shared_down;    // the shared expert, with shared_intermediate
  // Multi-head latent attention: q_proj is q_b_proj when there is a query
  // rank, after q_a_proj and q_a_norm; kv_a_proj gives the latent and the
  // RoPE key, and kv_a_norm normalizes the latent. Per head, embed_q
  // [kv_lora_rank, qk_nope_dim] absorbs the query into the latent and
  // unembed_out [v_head_dim, kv_lora_rank] expands the latent output: MLX's
  // tensors of those names, or with kv_b_proj [heads * (nope + v), rank]
  // instead, views of its value rows; its key rows then absorb the query
  // applied transposed, and embed_q is empty.
  Linear q_a_proj, kv_a_proj, kv_b_proj;
  std::vector<float> q_a_norm, kv_a_norm;
  std::vector<Linear> embed_q, unembed_out;
  // Gated DeltaNet: Qwen3.5's in_proj_qkv [channels, hidden], in_proj_z
  // [Hv * dv, hidden], in_proj_b and in_proj_a [Hv, hidden], or else
  // Qwen3-Next's in_proj_qkvz and in_proj_ba, whose rows group [q, k, v, z]
  // and [b, a] by key head; the conv's [channels, linear_conv] taps, the
  // decay rates exp(A_log) and dt_bias [Hv], the gated norm's weight [dv],
  // and out_proj [hidden, Hv * dv].
  Linear in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, in_proj_qkvz, in_proj_ba, out_proj;
  std::vector<float> conv, decay_rate, dt_bias, linear_norm;
  Linear shared_expert_gate;  // [1, hidden], with shared_expert_gate
  // Per-layer inputs (Gemma 4): gate [P, hidden], projection [hidden, P].
  Linear per_layer_gate, per_layer_proj;
  std::vector<float> per_layer_norm;
  float layer_scalar = 1.0f;
};

// Weights bound to a mapped checkpoint (zero-copy for matrices).
struct DecoderWeights {
  DecoderConfig config;
  Json raw_config;
  std::shared_ptr<Checkpoint> checkpoint;
  Linear embed;
  Linear lm_head;
  std::vector<float> final_norm;
  // Per-layer inputs (Gemma 4): a [vocab, layers * P] embedding and a
  // [layers * P, hidden] projection of the scaled token embedding.
  Linear embed_per_layer, per_layer_model_proj;
  std::vector<float> per_layer_proj_norm;
  std::vector<DecoderLayer> layers;
  bool tied_lm_head = false;

  static std::shared_ptr<DecoderWeights> load(const std::string& directory);
  size_t matrix_bytes() const;
  // Whether any matrix other than an expert is MXFP4: the Metal kernels read
  // MXFP4 only for experts so far.
  bool dense_mxfp4() const;
};

std::unique_ptr<Model> make_decoder_cpu(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options);

// Head `head`'s query absorption in a latent attention layer: its embed_q,
// or its key rows of kv_b_proj [qk_nope_dim, kv_lora_rank] (applied
// transposed).
Linear latent_absorption(const DecoderConfig& config, const DecoderLayer& layer, int head);

// Minimum memory traffic, in bytes, of forward-pass sections over n tokens,
// shared by every backend's profiler so attributions are comparable.
// Activations are float32; weights count once per call.
// Whether keys and values are stored as FP16, from LoadOptions::kv_cache:
// auto picks FP16 on Metal and FP32 on the CPU; latent attention and an
// unknown value fail closed when FP16 is asked for.
bool kv_cache_f16(const LoadOptions& options, const DecoderConfig& config, bool metal);

namespace traffic {
double linear(const Linear& weights, int n);           // weights, inputs, outputs
double embed(const Linear& embed, int n);              // n gathered rows, outputs
double rows(int64_t width, int n, int arrays);         // `arrays` [n, width] tensors
// q reads, output writes, and every visible key and value read once per kv
// head for tokens at positions [start, start + n).
double attention(const DecoderConfig& config, int layer, int start, int n, double kv_element_bytes = 4.0);
// Latent attention's query and latent projections with every head's
// absorption, and its per-head expansions with o_proj.
double latent_qkv(const DecoderConfig& config, const DecoderLayer& layer, int n);
double latent_out(const DecoderLayer& layer, int n);
// Gated DeltaNet's input projections; its conv, q/k norms and gates; its
// recurrence (every state read and written once per pass, plus each token's
// q, k, v, gates and output); and its gated norm with out_proj.
double linear_projections(const DecoderLayer& layer, int n);
double linear_conv(const DecoderConfig& config, int n);
double linear_recurrence(const DecoderConfig& config, int n);
double linear_out(const DecoderConfig& config, const DecoderLayer& layer, int n);
}  // namespace traffic

}  // namespace lokahi
