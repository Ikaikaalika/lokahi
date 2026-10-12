// Dense decoder-only transformers (Gemma 3, Llama 3.x, Mistral, Qwen 2/2.5,
// Qwen 3): configuration, weight binding, and the CPU backend. One
// configuration describes how the families differ (norm style and
// placement, activation, per-head Q/K norm, q/k/v biases, RoPE bases and
// scaling, sliding windows, logit soft-capping), and every backend
// implements all of it, so a family is a config parser rather than a fork.
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

  static DecoderConfig from_json(const Json& raw);
  // RoPE inverse frequencies (head_dim / 2 of them) for a layer kind.
  std::vector<double> rope_inv_freq(bool sliding) const;
  int kv_capacity(int layer, int max_context) const;
};

bool is_supported_decoder(const Json& raw);

struct DecoderLayer {
  // attn_post_norm and mlp_post_norm exist only with sandwich norms;
  // q_norm and k_norm only with qk_norm.
  std::vector<float> attn_norm, attn_post_norm, mlp_norm, mlp_post_norm, q_norm, k_norm;
  std::vector<float> q_bias, k_bias, v_bias;  // only with qkv_bias
  Linear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
};

// Weights bound to a mapped checkpoint (zero-copy for matrices).
struct DecoderWeights {
  DecoderConfig config;
  Json raw_config;
  std::shared_ptr<Checkpoint> checkpoint;
  Linear embed;
  Linear lm_head;
  std::vector<float> final_norm;
  std::vector<DecoderLayer> layers;
  bool tied_lm_head = false;

  static std::shared_ptr<DecoderWeights> load(const std::string& directory);
  size_t matrix_bytes() const;
};

std::unique_ptr<Model> make_decoder_cpu(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options);

}  // namespace lokahi
