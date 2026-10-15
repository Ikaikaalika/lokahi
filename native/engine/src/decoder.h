// Decoder-only transformers (Gemma 3, Gemma 4, Llama 3.x, Mistral, Qwen 2/2.5,
// Qwen 3, Qwen 3 MoE): configuration, weight binding, and the CPU backend.
// One configuration describes how the families differ (norm style and
// placement, activation, per-head Q/K norm, q/k/v biases, RoPE bases,
// scaling and partial rotation, sliding windows, logit soft-capping, and
// Gemma 4's per-layer geometry, KV sharing and per-layer inputs, and
// mixture-of-experts layers), so a family
// is a config parser rather than a fork. Backends that lack a feature refuse
// the config (see uniform_layers()).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "model.h"
#include "quant.h"
#include "safetensors.h"

namespace lokahi {

enum class Activation { GeluTanh, Silu };

// RoPE frequency scaling from `rope_scaling` (Hugging Face semantics).
struct RopeScaling {
  enum class Kind { None, Linear, Llama3 } kind = Kind::None;
  double factor = 1.0;
  double low_freq_factor = 1.0;      // llama3
  double high_freq_factor = 4.0;     // llama3
  double original_max_positions = 8192.0;  // llama3
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
  int per_layer_input = 0;          // hidden_size_per_layer_input; 0 disables
  bool layer_scalars = false;       // each layer's output times its layer_scalar

  // Mixture of experts (Qwen 3 MoE). In layers marked in layer_moe, a
  // softmax router picks experts_per_token of num_experts gated experts
  // (renormalizing their weights when norm_topk_prob) instead of the MLP.
  int num_experts = 0;
  int experts_per_token = 0;
  int moe_intermediate = 0;
  bool norm_topk_prob = false;
  std::vector<bool> layer_moe;

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
  // True when every layer has the same geometry and dense MLP, and none of
  // Gemma 4's additions are used: what the Metal backend implements today.
  bool uniform_layers() const {
    return layer_head_dim.empty() && layer_kv_heads.empty() && layer_intermediate.empty() &&
           layer_kv_source.empty() && layer_k_eq_v.empty() && !v_norm && global_partial_rotary == 1.0f &&
           per_layer_input == 0 && !layer_scalars && num_experts == 0;
  }
};

bool is_supported_decoder(const Json& raw);

struct DecoderLayer {
  // attn_post_norm and mlp_post_norm exist only with sandwich norms;
  // q_norm and k_norm only with qk_norm.
  std::vector<float> attn_norm, attn_post_norm, mlp_norm, mlp_post_norm, q_norm, k_norm;
  std::vector<float> q_bias, k_bias, v_bias;  // only with qkv_bias
  // k_proj and v_proj are empty (rows 0) in KV-shared layers, v_proj with k_eq_v;
  // gate_proj, up_proj and down_proj in mixture-of-experts layers.
  Linear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
  // Mixture of experts: the router [experts, hidden] and one view per expert.
  Linear router;
  std::vector<Linear> expert_gate, expert_up, expert_down;
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
  // Whether any matrix is MXFP4, which only the CPU kernels implement.
  bool uses_mxfp4() const;
};

std::unique_ptr<Model> make_decoder_cpu(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options);

// Minimum memory traffic, in bytes, of forward-pass sections over n tokens,
// shared by every backend's profiler so attributions are comparable.
// Activations are float32; weights count once per call.
namespace traffic {
double linear(const Linear& weights, int n);           // weights, inputs, outputs
double embed(const Linear& embed, int n);              // n gathered rows, outputs
double rows(int64_t width, int n, int arrays);         // `arrays` [n, width] tensors
// q reads, output writes, and every visible key and value read once per kv
// head for tokens at positions [start, start + n).
double attention(const DecoderConfig& config, int layer, int start, int n);
}  // namespace traffic

}  // namespace lokahi
