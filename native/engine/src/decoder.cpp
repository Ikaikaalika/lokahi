#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include "common.h"
#include "half.h"
#include "thread_pool.h"

namespace lokahi {

static const Json& text_config(const Json& raw) {
  const Json* text = raw.find("text_config");
  return text != nullptr && text->is_object() ? *text : raw;
}

namespace {

// A scalar constant as MLX-LM applies it to BF16 activations: weakly typed,
// so rounded to BF16 first.
float bf16_round(float x) { return bf16_to_f32(f32_to_bf16(x)); }

std::string model_type(const Json& raw) {
  return text_config(raw).string_or("model_type", raw.string_or("model_type", ""));
}

bool is_gemma3(const Json& raw) {
  const std::string type = model_type(raw);
  return type == "gemma3_text" || type == "gemma3";
}

bool is_gemma4(const Json& raw) {
  const std::string type = model_type(raw);
  return type == "gemma4_text" || type == "gemma4";
}

bool is_llama_family(const Json& raw) {
  const std::string type = model_type(raw);
  return type == "llama" || type == "mistral" || type == "qwen2" || type == "qwen3" || type == "qwen3_moe" ||
         type == "gpt_oss" || type == "glm4_moe";
}

bool flag(const Json& cfg, const char* key) {
  const Json* value = cfg.find(key);
  return value != nullptr && value->is_bool() && value->as_bool();
}

std::vector<int32_t> eos_ids(const Json& raw, const Json& cfg) {
  std::vector<int32_t> ids;
  const Json* eos = raw.find("eos_token_id");
  if (eos == nullptr) eos = cfg.find("eos_token_id");
  if (eos != nullptr && eos->is_array()) {
    for (const Json& id : eos->as_array()) ids.push_back(static_cast<int32_t>(id.as_int()));
  } else if (eos != nullptr && eos->is_number()) {
    ids.push_back(static_cast<int32_t>(eos->as_int()));
  }
  return ids;
}

// Gemma 3. Multimodal checkpoints (Gemma3ForConditionalGeneration) list only
// the text fields that differ from Transformers' Gemma3TextConfig, so the
// rest take its defaults.
DecoderConfig gemma3_config(const Json& raw) {
  const Json& cfg = text_config(raw);
  DecoderConfig c;
  c.architecture = "gemma3";
  c.vocab_size = static_cast<int>(cfg.int_or("vocab_size", 262208));
  c.hidden_size = static_cast<int>(cfg.at("hidden_size").as_int());
  c.intermediate_size = static_cast<int>(cfg.at("intermediate_size").as_int());
  c.num_layers = static_cast<int>(cfg.at("num_hidden_layers").as_int());
  c.num_heads = static_cast<int>(cfg.int_or("num_attention_heads", 8));
  c.num_kv_heads = static_cast<int>(cfg.int_or("num_key_value_heads", 4));
  c.head_dim = static_cast<int>(cfg.int_or("head_dim", 256));
  c.rms_norm_eps = static_cast<float>(cfg.number_or("rms_norm_eps", 1e-6));
  c.norm_offset = 1.0f;
  c.sandwich_norms = true;
  c.qk_norm = true;
  c.image_blocks = DecoderConfig::ImageBlocks::AllLayers;
  // sqrt(hidden_size) rounded to BF16, as the reference implementations do.
  c.embed_scale = bf16_round(std::sqrt(static_cast<float>(c.hidden_size)));
  c.attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg.number_or("query_pre_attn_scalar", 256)));
  c.rope_theta = static_cast<float>(cfg.number_or("rope_theta", 1e6));
  c.local_rope = true;
  c.rope_local_theta = static_cast<float>(cfg.number_or("rope_local_base_freq", 1e4));
  c.sliding_window = static_cast<int>(cfg.int_or("sliding_window", 4096));
  if (const Json* scaling = cfg.find("rope_scaling"); scaling != nullptr && scaling->is_object()) {
    std::string kind = scaling->string_or("rope_type", scaling->string_or("type", "default"));
    if (kind == "linear") {
      c.rope_scaling.kind = RopeScaling::Kind::Linear;
      c.rope_scaling.factor = scaling->at("factor").as_number();
    } else {
      LK_CHECK(kind == "default", "gemma3: rope_scaling type '" + kind + "' is not supported");
    }
  }
  std::string activation = cfg.string_or("hidden_activation", "gelu_pytorch_tanh");
  LK_CHECK(activation == "gelu_pytorch_tanh", "gemma3: unsupported activation " + activation);
  c.activation = Activation::GeluTanh;
  if (const Json* types = cfg.find("layer_types"); types != nullptr && types->is_array()) {
    for (const Json& kind : types->as_array()) {
      c.layer_is_sliding.push_back(kind.as_string() == "sliding_attention");
    }
  } else {
    int pattern = static_cast<int>(cfg.int_or("sliding_window_pattern", 6));
    for (int i = 0; i < c.num_layers; ++i) c.layer_is_sliding.push_back((i + 1) % pattern != 0);
  }
  if (const Json* cap = cfg.find("final_logit_softcapping"); cap && cap->is_number()) {
    c.final_logit_softcapping = static_cast<float>(cap->as_number());
  }
  if (const Json* cap = cfg.find("attn_logit_softcapping"); cap && cap->is_number()) {
    c.attn_logit_softcapping = static_cast<float>(cap->as_number());
  }
  c.eos_token_ids = eos_ids(raw, cfg);
  return c;
}

// Gemma 4 (dense; MLX-LM gemma4_text): sandwich norms that scale by the
// weight itself, Q/K norm plus a scale-free V norm and attention scale 1,
// sliding and full layers with their own head size and RoPE (proportional
// partial rotation on full layers), the last num_kv_shared_layers layers
// attending over an earlier layer's keys and values with double-wide MLPs,
// per-layer input embeddings, per-layer output scalars, and a final soft-cap.
DecoderConfig gemma4_config(const Json& raw) {
  const Json& cfg = text_config(raw);
  DecoderConfig c;
  c.architecture = "gemma4";
  LK_CHECK(!flag(cfg, "attention_bias"), "gemma4: attention_bias is not supported");
  LK_CHECK(!flag(cfg, "use_bidirectional_attention"), "gemma4: bidirectional attention is not supported");
  // "vision": image tokens see each other on sliding layers only
  // (Transformers' create_masks_for_vision_model); otherwise they are causal.
  if (cfg.string_or("use_bidirectional_attention", "") == "vision") {
    c.image_blocks = DecoderConfig::ImageBlocks::SlidingLayers;
  }
  c.vocab_size = static_cast<int>(cfg.at("vocab_size").as_int());
  c.hidden_size = static_cast<int>(cfg.at("hidden_size").as_int());
  c.intermediate_size = static_cast<int>(cfg.at("intermediate_size").as_int());
  c.num_layers = static_cast<int>(cfg.at("num_hidden_layers").as_int());
  c.num_heads = static_cast<int>(cfg.int_or("num_attention_heads", 8));
  c.num_kv_heads = static_cast<int>(cfg.int_or("num_key_value_heads", 1));
  c.head_dim = static_cast<int>(cfg.int_or("head_dim", 256));
  c.rms_norm_eps = static_cast<float>(cfg.number_or("rms_norm_eps", 1e-6));
  c.sandwich_norms = true;
  c.qk_norm = true;
  c.v_norm = true;
  c.attn_scale = 1.0f;
  c.embed_scale = bf16_round(std::sqrt(static_cast<float>(c.hidden_size)));
  const std::string activation = cfg.string_or("hidden_activation", "gelu_pytorch_tanh");
  LK_CHECK(activation == "gelu_pytorch_tanh", "gemma4: unsupported activation " + activation);
  c.activation = Activation::GeluTanh;
  c.sliding_window = static_cast<int>(cfg.int_or("sliding_window", 512));
  if (const Json* types = cfg.find("layer_types"); types != nullptr && types->is_array()) {
    for (const Json& kind : types->as_array()) c.layer_is_sliding.push_back(kind.as_string() == "sliding_attention");
  } else {
    const int pattern = static_cast<int>(cfg.int_or("sliding_window_pattern", 5));
    for (int i = 0; i < c.num_layers; ++i) c.layer_is_sliding.push_back((i + 1) % pattern != 0);
  }
  LK_CHECK(static_cast<int>(c.layer_is_sliding.size()) == c.num_layers, "gemma4: layer_types length mismatch");

  // RoPE per layer kind. Sliding layers rotate fully with their own base.
  c.local_rope = true;
  c.rope_local_theta = 1e4f;
  c.rope_theta = 1e6f;
  c.global_partial_rotary = 0.25f;
  if (const Json* rope = cfg.find("rope_parameters"); rope != nullptr && rope->is_object()) {
    if (const Json* local = rope->find("sliding_attention")) {
      const std::string kind = local->string_or("rope_type", "default");
      LK_CHECK(kind == "default", "gemma4: sliding rope_type '" + kind + "' is not supported");
      LK_CHECK(local->number_or("partial_rotary_factor", 1.0) == 1.0, "gemma4: partial rotation on sliding layers");
      c.rope_local_theta = static_cast<float>(local->number_or("rope_theta", 1e4));
    }
    if (const Json* full = rope->find("full_attention")) {
      const std::string kind = full->string_or("rope_type", "default");
      LK_CHECK(kind == "proportional" || kind == "default", "gemma4: full rope_type '" + kind + "' is not supported");
      c.rope_theta = static_cast<float>(full->number_or("rope_theta", 1e6));
      c.global_partial_rotary = static_cast<float>(full->number_or("partial_rotary_factor", 1.0));
      if (kind == "default") LK_CHECK(c.global_partial_rotary == 1.0f, "gemma4: default RoPE with partial rotation");
      const double factor = full->number_or("factor", 1.0);
      if (factor != 1.0) {
        c.rope_scaling.kind = RopeScaling::Kind::Linear;
        c.rope_scaling.factor = factor;
      }
    }
  }

  const int global_dim = static_cast<int>(cfg.int_or("global_head_dim", c.head_dim));
  const bool k_eq_v = flag(cfg, "attention_k_eq_v");
  const Json* global_kv = cfg.find("num_global_key_value_heads");
  const int shared = static_cast<int>(cfg.int_or("num_kv_shared_layers", 0));
  const int first_shared = c.num_layers - shared;
  LK_CHECK(shared >= 0 && first_shared > 0, "gemma4: num_kv_shared_layers out of range");
  const bool double_wide = flag(cfg, "use_double_wide_mlp");
  for (int i = 0; i < c.num_layers; ++i) {
    const bool full = !c.layer_is_sliding[static_cast<size_t>(i)];
    const bool layer_k_eq_v = k_eq_v && full;
    c.layer_head_dim.push_back(full ? global_dim : c.head_dim);
    c.layer_kv_heads.push_back(layer_k_eq_v && global_kv != nullptr && global_kv->is_number()
                                   ? static_cast<int>(global_kv->as_int())
                                   : c.num_kv_heads);
    c.layer_k_eq_v.push_back(layer_k_eq_v);
    int source = i;
    if (shared > 0 && i >= first_shared) {
      source = -1;
      for (int j = 0; j < first_shared; ++j) {
        if (c.layer_is_sliding[static_cast<size_t>(j)] == c.layer_is_sliding[static_cast<size_t>(i)]) source = j;
      }
      LK_CHECK(source >= 0, "gemma4: a shared layer has no earlier layer of its kind");
    }
    c.layer_kv_source.push_back(source);
    c.layer_intermediate.push_back(c.intermediate_size * (double_wide && source != i ? 2 : 1));
  }
  // 26B-A4B: experts beside every dense MLP, routed by the softmax of the
  // top-k logits (Transformers' renormalized top-k of a full softmax).
  if (flag(cfg, "enable_moe_block")) {
    c.num_experts = static_cast<int>(cfg.at("num_experts").as_int());
    c.experts_per_token = static_cast<int>(cfg.at("top_k_experts").as_int());
    const Json* moe = cfg.find("moe_intermediate_size");
    c.moe_intermediate = static_cast<int>(moe != nullptr && moe->is_number() ? moe->as_int()
                                                                              : cfg.at("expert_intermediate_size").as_int());
    LK_CHECK(c.num_experts > 0 && c.experts_per_token > 0 && c.experts_per_token <= c.num_experts && c.moe_intermediate > 0,
             "gemma4: invalid mixture-of-experts sizes");
    c.router = DecoderConfig::Router::TopKSoftmax;
    c.router_module = "router.proj";
    c.experts_beside_mlp = true;
    c.layer_moe.assign(static_cast<size_t>(c.num_layers), true);
  }
  c.per_layer_input = static_cast<int>(cfg.int_or("hidden_size_per_layer_input", 0));
  c.per_layer_embed_scale = bf16_round(std::sqrt(static_cast<float>(c.per_layer_input)));
  c.per_layer_projection_scale = bf16_round(1.0f / std::sqrt(static_cast<float>(c.hidden_size)));
  c.per_layer_input_scale = bf16_round(1.0f / std::sqrt(2.0f));
  c.layer_scalars = true;
  if (const Json* cap = cfg.find("final_logit_softcapping"); cap && cap->is_number()) {
    c.final_logit_softcapping = static_cast<float>(cap->as_number());
  }
  c.eos_token_ids = eos_ids(raw, cfg);
  return c;
}

// Llama 3.x, Mistral, Qwen 2/2.5, Qwen 3, Qwen 3 MoE and GPT-OSS: pre-norm
// blocks with plain RMSNorm, gated MLPs and rotate-half RoPE. Qwen 3 adds
// per-head Q/K norm, Qwen 2 adds q/k/v biases, Mistral may use a sliding
// window on every layer, and Qwen 3 MoE replaces the MLP of its sparse layers
// with routed experts. GPT-OSS adds attention biases and sinks, alternating
// sliding and full layers, a top-k-then-softmax router over biased experts
// with a clamped SwiGLU, and YaRN. GLM-4.5 adds q/k/v biases, optional q/k
// norms, classic partial RoPE, leading dense layers, and a grouped sigmoid
// router over routed experts plus a shared expert. Options these checkpoints
// do not use fail closed.
DecoderConfig llama_family_config(const Json& cfg) {
  DecoderConfig c;
  c.architecture = model_type(cfg);
  const std::string& arch = c.architecture;
  c.vocab_size = static_cast<int>(cfg.at("vocab_size").as_int());
  c.hidden_size = static_cast<int>(cfg.at("hidden_size").as_int());
  c.intermediate_size = static_cast<int>(cfg.at("intermediate_size").as_int());
  c.num_layers = static_cast<int>(cfg.at("num_hidden_layers").as_int());
  c.num_heads = static_cast<int>(cfg.at("num_attention_heads").as_int());
  c.num_kv_heads = static_cast<int>(cfg.int_or("num_key_value_heads", c.num_heads));
  const Json* head_dim = cfg.find("head_dim");
  c.head_dim = head_dim != nullptr && head_dim->is_number() ? static_cast<int>(head_dim->as_int())
                                                             : c.hidden_size / c.num_heads;
  c.rms_norm_eps = static_cast<float>(cfg.number_or("rms_norm_eps", 1e-6));
  c.qk_norm = arch == "qwen3" || arch == "qwen3_moe" || (arch == "glm4_moe" && flag(cfg, "use_qk_norm"));
  c.qkv_bias = arch == "qwen2" || arch == "gpt_oss" || (arch == "glm4_moe" && flag(cfg, "attention_bias"));
  c.o_bias = arch == "gpt_oss";
  c.attention_sinks = arch == "gpt_oss";
  c.attn_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
  c.rope_theta = static_cast<float>(cfg.number_or("rope_theta", 1e4));

  const std::string activation = cfg.string_or("hidden_act", "silu");
  LK_CHECK(activation == "silu", arch + ": unsupported activation " + activation);
  // These add attention or MLP biases, which only GPT-OSS and GLM use.
  LK_CHECK(arch == "gpt_oss" || arch == "glm4_moe" || !flag(cfg, "attention_bias"),
           arch + ": attention_bias is not supported");
  LK_CHECK(!flag(cfg, "mlp_bias"), arch + ": mlp_bias is not supported");
  const double rotary = cfg.number_or("partial_rotary_factor", 1.0);
  LK_CHECK(rotary == 1.0 || arch == "glm4_moe", arch + ": partial_rotary_factor is not supported");
  if (rotary != 1.0) {
    c.rotary_dim = static_cast<int>(c.head_dim * rotary);
    LK_CHECK(c.rotary_dim > 0 && c.rotary_dim <= c.head_dim && c.rotary_dim % 2 == 0,
             arch + ": partial_rotary_factor must rotate an even number of dimensions");
  }

  if (const Json* scaling = cfg.find("rope_scaling"); scaling != nullptr && scaling->is_object()) {
    const std::string kind = scaling->string_or("rope_type", scaling->string_or("type", "default"));
    if (kind == "llama3") {
      c.rope_scaling.kind = RopeScaling::Kind::Llama3;
      c.rope_scaling.factor = scaling->at("factor").as_number();
      c.rope_scaling.low_freq_factor = scaling->number_or("low_freq_factor", 1.0);
      c.rope_scaling.high_freq_factor = scaling->number_or("high_freq_factor", 4.0);
      c.rope_scaling.original_max_positions = scaling->number_or("original_max_position_embeddings", 8192.0);
      LK_CHECK(c.rope_scaling.high_freq_factor > c.rope_scaling.low_freq_factor,
               arch + ": llama3 rope_scaling needs high_freq_factor > low_freq_factor");
    } else if (kind == "linear") {
      c.rope_scaling.kind = RopeScaling::Kind::Linear;
      c.rope_scaling.factor = scaling->at("factor").as_number();
    } else if (kind == "yarn") {
      // Transformers' _compute_yarn_parameters, including `truncate` (which
      // GPT-OSS sets false and MLX-LM ignores).
      RopeScaling& y = c.rope_scaling;
      y.kind = RopeScaling::Kind::Yarn;
      y.original_max_positions = scaling->at("original_max_position_embeddings").as_number();
      const Json* factor = scaling->find("factor");
      y.factor = factor != nullptr && factor->is_number()
                     ? factor->as_number()
                     : cfg.at("max_position_embeddings").as_number() / y.original_max_positions;
      const double fast = scaling->number_or("beta_fast", 0.0), slow = scaling->number_or("beta_slow", 0.0);
      y.beta_fast = fast != 0.0 ? fast : 32.0;
      y.beta_slow = slow != 0.0 ? slow : 1.0;
      const Json* truncate = scaling->find("truncate");
      y.truncate = truncate == nullptr || !truncate->is_bool() || truncate->as_bool();
      auto mscale = [](double scale, double m) { return scale <= 1.0 ? 1.0 : 0.1 * m * std::log(scale) + 1.0; };
      double attention = scaling->number_or("attention_factor", 0.0);
      if (attention == 0.0) {
        const double m = scaling->number_or("mscale", 0.0), all = scaling->number_or("mscale_all_dim", 0.0);
        attention = m != 0.0 && all != 0.0 ? mscale(y.factor, m) / mscale(y.factor, all) : mscale(y.factor, 1.0);
      }
      // The references scale cos and sin, so q and k, by `attention`; that is
      // the same as scaling every attention logit by its square.
      c.attn_scale *= static_cast<float>(attention * attention);
    } else {
      LK_CHECK(kind == "default", arch + ": rope_scaling type '" + kind + "' is not supported");
    }
  }

  // Mistral applies its window to every layer; Qwen's optional per-layer
  // windows (use_sliding_window) are not supported.
  LK_CHECK(!flag(cfg, "use_sliding_window"), arch + ": use_sliding_window is not supported");
  const Json* window = cfg.find("sliding_window");
  const bool sliding = arch == "mistral" && window != nullptr && window->is_number();
  if (sliding) c.sliding_window = static_cast<int>(window->as_int());
  c.layer_is_sliding.assign(static_cast<size_t>(c.num_layers), sliding);
  c.eos_token_ids = eos_ids(cfg, cfg);

  if (arch == "glm4_moe") {
    LK_CHECK(cfg.string_or("scoring_func", "sigmoid") == "sigmoid", arch + ": only sigmoid routing is supported");
    LK_CHECK(cfg.string_or("topk_method", "noaux_tc") == "noaux_tc", arch + ": only noaux_tc routing is supported");
    c.num_experts = static_cast<int>(cfg.at("n_routed_experts").as_int());
    c.experts_per_token = static_cast<int>(cfg.at("num_experts_per_tok").as_int());
    c.moe_intermediate = static_cast<int>(cfg.at("moe_intermediate_size").as_int());
    c.norm_topk_prob = flag(cfg, "norm_topk_prob");
    c.n_group = static_cast<int>(cfg.int_or("n_group", 1));
    c.topk_group = static_cast<int>(cfg.int_or("topk_group", 1));
    c.routed_scaling = static_cast<float>(cfg.number_or("routed_scaling_factor", 1.0));
    c.shared_intermediate = c.moe_intermediate * static_cast<int>(cfg.int_or("n_shared_experts", 0));
    LK_CHECK(c.num_experts > 0 && c.experts_per_token > 0 && c.experts_per_token <= c.num_experts &&
                 c.n_group > 0 && c.num_experts % c.n_group == 0 && c.topk_group > 0 && c.topk_group <= c.n_group &&
                 c.experts_per_token <= c.topk_group * (c.num_experts / c.n_group),
             arch + ": invalid expert configuration");
    const int dense = static_cast<int>(cfg.int_or("first_k_dense_replace", 0));
    for (int i = 0; i < c.num_layers; ++i) c.layer_moe.push_back(i >= dense);
    c.router = DecoderConfig::Router::SigmoidGrouped;
  }

  if (arch == "gpt_oss") {
    // Alternating sliding and full layers by layer_types.
    c.sliding_window = static_cast<int>(cfg.int_or("sliding_window", 128));
    c.layer_is_sliding.clear();
    if (const Json* types = cfg.find("layer_types"); types != nullptr && types->is_array()) {
      for (const Json& kind : types->as_array()) c.layer_is_sliding.push_back(kind.as_string() == "sliding_attention");
    } else {
      for (int i = 0; i < c.num_layers; ++i) c.layer_is_sliding.push_back(i % 2 == 0);
    }
    c.num_experts = static_cast<int>(cfg.at("num_local_experts").as_int());
    c.experts_per_token = static_cast<int>(cfg.int_or("num_experts_per_tok", cfg.int_or("experts_per_token", 4)));
    c.moe_intermediate = c.intermediate_size;
    LK_CHECK(c.num_experts > 0 && c.experts_per_token > 0 && c.experts_per_token <= c.num_experts,
             arch + ": invalid expert configuration");
    c.layer_moe.assign(static_cast<size_t>(c.num_layers), true);
    c.router = DecoderConfig::Router::TopKSoftmax;
    c.router_module = "mlp.router";
    c.router_bias = true;
    c.expert_bias = true;
    c.activation = Activation::ClampedSwiglu;
    c.swiglu_limit = static_cast<float>(cfg.number_or("swiglu_limit", 7.0));
  }

  if (arch == "qwen3_moe") {
    // Qwen 2 MoE's shared expert is a different block.
    LK_CHECK(cfg.int_or("shared_expert_intermediate_size", 0) == 0, arch + ": shared experts are not supported");
    c.num_experts = static_cast<int>(cfg.at("num_experts").as_int());
    c.experts_per_token = static_cast<int>(cfg.at("num_experts_per_tok").as_int());
    c.moe_intermediate = static_cast<int>(cfg.at("moe_intermediate_size").as_int());
    c.norm_topk_prob = flag(cfg, "norm_topk_prob");
    LK_CHECK(c.num_experts > 0 && c.experts_per_token > 0 && c.experts_per_token <= c.num_experts &&
                 c.moe_intermediate > 0,
             arch + ": invalid expert configuration");
    const int step = static_cast<int>(cfg.int_or("decoder_sparse_step", 1));
    LK_CHECK(step > 0, arch + ": decoder_sparse_step must be positive");
    std::vector<bool> dense_only(static_cast<size_t>(c.num_layers), false);
    if (const Json* only = cfg.find("mlp_only_layers"); only != nullptr && only->is_array()) {
      for (const Json& layer : only->as_array()) {
        const int64_t index = layer.as_int();
        LK_CHECK(index >= 0 && index < c.num_layers, arch + ": mlp_only_layers entry out of range");
        dense_only[static_cast<size_t>(index)] = true;
      }
    }
    for (int i = 0; i < c.num_layers; ++i) {
      c.layer_moe.push_back(!dense_only[static_cast<size_t>(i)] && (i + 1) % step == 0);
    }
  }
  return c;
}

}  // namespace

bool is_supported_decoder(const Json& raw) { return is_gemma3(raw) || is_gemma4(raw) || is_llama_family(raw); }

DecoderConfig DecoderConfig::from_json(const Json& raw) {
  LK_CHECK(is_supported_decoder(raw),
           "unsupported architecture '" + model_type(raw) +
               "'; this build supports gemma3, gemma4, llama, mistral, qwen2 and qwen3 text models");
  DecoderConfig c = is_gemma3(raw)   ? gemma3_config(raw)
                    : is_gemma4(raw) ? gemma4_config(raw)
                                     : llama_family_config(raw);
  LK_CHECK(static_cast<int>(c.layer_is_sliding.size()) == c.num_layers,
           c.architecture + ": layer_types length mismatch");
  for (int l = 0; l < c.num_layers; ++l) {
    const int kv = c.kv_heads_of(l), dim = c.head_dim_of(l);
    LK_CHECK(c.num_heads > 0 && kv > 0 && c.num_heads % kv == 0, c.architecture + ": heads must be a multiple of kv heads");
    LK_CHECK(dim > 0 && dim % 2 == 0, c.architecture + ": head_dim must be even");
    const int source = c.kv_source(l);
    LK_CHECK(source >= 0 && source <= l && !c.shares_kv(source) &&
                 c.layer_is_sliding[static_cast<size_t>(source)] == c.layer_is_sliding[static_cast<size_t>(l)] &&
                 c.head_dim_of(source) == dim && c.kv_heads_of(source) == kv,
             c.architecture + ": layer " + std::to_string(l) + " shares keys and values with an incompatible layer");
  }
  LK_CHECK(c.sliding_window > 0 ||
               std::none_of(c.layer_is_sliding.begin(), c.layer_is_sliding.end(), [](bool s) { return s; }),
           c.architecture + ": sliding layers need a positive sliding_window");
  return c;
}

std::vector<double> DecoderConfig::rope_inv_freq(bool sliding) const {
  const bool local = sliding && local_rope;
  const double base = local ? rope_local_theta : rope_theta;
  const RopeScaling none;
  const RopeScaling& scaling = local ? none : rope_scaling;
  constexpr double kPi = 3.14159265358979323846;
  int dims = head_dim;  // the head size of this layer kind
  for (int l = 0; l < num_layers; ++l) {
    if (layer_is_sliding[static_cast<size_t>(l)] == sliding) {
      dims = head_dim_of(l);
      break;
    }
  }
  if (rotary_dim > 0) dims = rotary_dim;  // classic partial RoPE rotates a prefix of the head
  const size_t rotated = sliding ? static_cast<size_t>(dims / 2)
                                 : static_cast<size_t>(static_cast<int>(dims * global_partial_rotary) / 2);
  std::vector<double> inv_freq(static_cast<size_t>(dims / 2), 0.0);
  if (scaling.kind == RopeScaling::Kind::Yarn) {
    // Interpolate (divide by the factor) the low frequencies, extrapolate the
    // high ones, and ramp linearly between the dimensions where a frequency
    // completes beta_fast and beta_slow rotations over the original context.
    auto correction = [&](double rotations) {
      return dims * std::log(scaling.original_max_positions / (rotations * 2.0 * kPi)) / (2.0 * std::log(base));
    };
    double low = correction(scaling.beta_fast), high = correction(scaling.beta_slow);
    if (scaling.truncate) {
      low = std::floor(low);
      high = std::ceil(high);
    }
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(dims - 1));
    if (low == high) high += 0.001;
    for (size_t i = 0; i < rotated; ++i) {
      const double extrapolated = 1.0 / std::pow(base, static_cast<double>(2 * i) / dims);
      const double ramp = std::clamp((static_cast<double>(i) - low) / (high - low), 0.0, 1.0);
      inv_freq[i] = extrapolated / scaling.factor * ramp + extrapolated * (1.0 - ramp);
    }
    return inv_freq;
  }
  for (size_t i = 0; i < rotated; ++i) {
    double f = 1.0 / std::pow(base, static_cast<double>(2 * i) / dims);
    if (scaling.kind == RopeScaling::Kind::Linear) {
      f /= scaling.factor;
    } else if (scaling.kind == RopeScaling::Kind::Llama3) {
      // Hugging Face's _compute_llama3_parameters: long wavelengths are
      // divided by the factor, short ones kept, and the band between them
      // interpolated.
      const double low_wavelen = scaling.original_max_positions / scaling.low_freq_factor;
      const double high_wavelen = scaling.original_max_positions / scaling.high_freq_factor;
      const double wavelen = 2.0 * kPi / f;
      if (wavelen > low_wavelen) {
        f /= scaling.factor;
      } else if (wavelen >= high_wavelen) {
        const double smooth = (scaling.original_max_positions / wavelen - scaling.low_freq_factor) /
                              (scaling.high_freq_factor - scaling.low_freq_factor);
        f = (1.0 - smooth) * f / scaling.factor + smooth * f;
      }
    }
    inv_freq[i] = f;
  }
  return inv_freq;
}

namespace {

// The router and experts of a mixture-of-experts layer: MLX's stacked
// switch_mlp tensors, or one module per expert as Transformers stores them.
void bind_experts(const DecoderWeights& w, DecoderLayer& layer, const std::string& base) {
  const DecoderConfig& c = w.config;
  const Checkpoint& ck = *w.checkpoint;
  const std::string& p = c.weight_prefix;
  const std::string router = p + base + "." + c.router_module;
  layer.router = bind_linear(ck, w.raw_config, router, router);
  LK_CHECK(layer.router.rows == c.num_experts && layer.router.cols == c.hidden_size, base + ": router shape");
  if (c.router_bias) {
    layer.router_bias.resize(static_cast<size_t>(c.num_experts));
    load_vector(ck, router + ".bias", c.num_experts, layer.router_bias.data());
  }
  if (c.router == DecoderConfig::Router::SigmoidGrouped) {
    layer.router_correction.resize(static_cast<size_t>(c.num_experts));
    load_vector(ck, router + ".e_score_correction_bias", c.num_experts, layer.router_correction.data());
  }
  if (c.shared_intermediate > 0) {
    const std::string shared = p + base + ".mlp.shared_experts.";
    layer.shared_gate = bind_linear(ck, w.raw_config, shared + "gate_proj", shared + "gate_proj");
    layer.shared_up = bind_linear(ck, w.raw_config, shared + "up_proj", shared + "up_proj");
    layer.shared_down = bind_linear(ck, w.raw_config, shared + "down_proj", shared + "down_proj");
    LK_CHECK(layer.shared_gate.rows == c.shared_intermediate && layer.shared_gate.cols == c.hidden_size &&
                 layer.shared_up.rows == c.shared_intermediate && layer.shared_up.cols == c.hidden_size &&
                 layer.shared_down.rows == c.hidden_size && layer.shared_down.cols == c.shared_intermediate,
             base + ": shared expert shape");
  }
  std::vector<Linear>* stacks[] = {&layer.expert_gate, &layer.expert_up, &layer.expert_down};
  std::vector<float>* biases[] = {&layer.expert_gate_bias, &layer.expert_up_bias, &layer.expert_down_bias};
  const char* names[] = {"gate_proj", "up_proj", "down_proj"};
  for (int k = 0; k < 3; ++k) {
    // Stacked [experts, rows, cols]: MLX-LM's switch_mlp (Qwen), experts
    // (GPT-OSS) or MLX-VLM's experts.switch_glu (Gemma 4); otherwise one
    // module per expert.
    std::string stacked = p + base + ".mlp.switch_mlp." + names[k];
    if (!ck.contains(stacked + ".weight")) stacked = p + base + ".mlp.experts." + names[k];
    if (!ck.contains(stacked + ".weight")) stacked = p + base + ".experts.switch_glu." + names[k];
    const bool is_stacked = ck.contains(stacked + ".weight");
    if (is_stacked) {
      *stacks[k] = bind_expert_linears(ck, w.raw_config, stacked, stacked);
    } else {
      for (int e = 0; e < c.num_experts; ++e) {
        const std::string module = p + base + ".mlp.experts." + std::to_string(e) + "." + names[k];
        stacks[k]->push_back(bind_linear(ck, w.raw_config, module, module));
      }
    }
    LK_CHECK(static_cast<int>(stacks[k]->size()) == c.num_experts, base + ": expected " +
                                                                     std::to_string(c.num_experts) + " experts");
    const int64_t rows = (*stacks[k])[0].rows;
    if (c.expert_bias) {
      biases[k]->resize(static_cast<size_t>(c.num_experts * rows));
      if (is_stacked) {
        load_vector(ck, stacked + ".bias", c.num_experts * rows, biases[k]->data());
      } else {
        for (int e = 0; e < c.num_experts; ++e) {
          const std::string module = p + base + ".mlp.experts." + std::to_string(e) + "." + names[k];
          load_vector(ck, module + ".bias", rows, biases[k]->data() + static_cast<size_t>(e * rows));
        }
      }
    } else {
      LK_CHECK(!ck.contains((is_stacked ? stacked : p + base + ".mlp.experts.0." + names[k]) + ".bias"),
               base + ": unexpected expert bias for " + c.architecture);
    }
  }
  for (int e = 0; e < c.num_experts; ++e) {
    const Linear& gate = layer.expert_gate[static_cast<size_t>(e)];
    const Linear& up = layer.expert_up[static_cast<size_t>(e)];
    const Linear& down = layer.expert_down[static_cast<size_t>(e)];
    LK_CHECK(gate.rows == c.moe_intermediate && gate.cols == c.hidden_size && up.rows == c.moe_intermediate &&
                 up.cols == c.hidden_size && down.rows == c.hidden_size && down.cols == c.moe_intermediate,
             base + ": expert shape");
  }
}

}  // namespace

std::shared_ptr<DecoderWeights> DecoderWeights::load(const std::string& directory) {
  auto w = std::make_shared<DecoderWeights>();
  w->raw_config = Json::parse(read_text_file(directory + "/config.json"));
  DecoderConfig& c = w->config;
  c = DecoderConfig::from_json(w->raw_config);
  w->checkpoint = std::make_shared<Checkpoint>(Checkpoint::open_directory(directory));
  const Checkpoint& ck = *w->checkpoint;
  if (ck.contains("language_model.model.embed_tokens.weight")) c.weight_prefix = "language_model.";

  const std::string& p = c.weight_prefix;
  const std::string& arch = c.architecture;
  auto linear = [&](const std::string& module) {
    return bind_linear(ck, w->raw_config, p + module, p + module);
  };
  auto vec = [&](const std::string& name, int64_t size) {
    std::vector<float> out(static_cast<size_t>(size));
    load_vector(ck, p + name + ".weight", size, out.data());
    return out;
  };
  auto bias = [&](const std::string& module, int64_t size) {
    std::vector<float> out(static_cast<size_t>(size));
    load_vector(ck, p + module + ".bias", size, out.data());
    return out;
  };

  w->embed = linear("model.embed_tokens");
  LK_CHECK(w->embed.rows == c.vocab_size && w->embed.cols == c.hidden_size, arch + ": embed_tokens shape mismatch");
  if (ck.contains(p + "lm_head.weight")) {
    w->lm_head = linear("lm_head");
  } else if (ck.contains("lm_head.weight")) {
    w->lm_head = bind_linear(ck, w->raw_config, "lm_head", "lm_head");
  } else {
    const Json& cfg = text_config(w->raw_config);
    const Json* tie = cfg.find("tie_word_embeddings");
    LK_CHECK(tie == nullptr || !tie->is_bool() || tie->as_bool(),
             arch + ": tie_word_embeddings is false but the checkpoint has no lm_head");
    w->lm_head = w->embed;
    w->tied_lm_head = true;
  }
  w->final_norm = vec("model.norm", c.hidden_size);
  if (c.per_layer_input > 0) {
    const int64_t width = static_cast<int64_t>(c.num_layers) * c.per_layer_input;
    const Json& cfg = text_config(w->raw_config);
    w->embed_per_layer = linear("model.embed_tokens_per_layer");
    w->per_layer_model_proj = linear("model.per_layer_model_projection");
    w->per_layer_proj_norm = vec("model.per_layer_projection_norm", c.per_layer_input);
    LK_CHECK(w->embed_per_layer.rows == cfg.int_or("vocab_size_per_layer_input", c.vocab_size) &&
                 w->embed_per_layer.cols == width,
             arch + ": embed_tokens_per_layer shape mismatch");
    LK_CHECK(w->per_layer_model_proj.rows == width && w->per_layer_model_proj.cols == c.hidden_size,
             arch + ": per_layer_model_projection shape mismatch");
  }
  for (int i = 0; i < c.num_layers; ++i) {
    const std::string base = "model.layers." + std::to_string(i);
    const int64_t dim = c.head_dim_of(i);
    const int64_t q_out = static_cast<int64_t>(c.num_heads) * dim;
    const int64_t kv_out = static_cast<int64_t>(c.kv_heads_of(i)) * dim;
    const int64_t inter = c.intermediate_of(i);
    // KV-shared layers attend over another layer's keys and values, so their
    // own k/v weights (absent from converted checkpoints) are never bound.
    const bool own_kv = !c.shares_kv(i);
    DecoderLayer layer;
    layer.attn_norm = vec(base + ".input_layernorm", c.hidden_size);
    if (c.sandwich_norms) {
      layer.attn_post_norm = vec(base + ".post_attention_layernorm", c.hidden_size);
      layer.mlp_norm = vec(base + ".pre_feedforward_layernorm", c.hidden_size);
      layer.mlp_post_norm = vec(base + ".post_feedforward_layernorm", c.hidden_size);
    } else {
      // Pre-norm families name the norm before the MLP post_attention_layernorm.
      layer.mlp_norm = vec(base + ".post_attention_layernorm", c.hidden_size);
    }
    if (c.qk_norm) {
      layer.q_norm = vec(base + ".self_attn.q_norm", dim);
      if (own_kv) layer.k_norm = vec(base + ".self_attn.k_norm", dim);
    }
    if (c.qkv_bias) {
      layer.q_bias = bias(base + ".self_attn.q_proj", q_out);
      layer.k_bias = bias(base + ".self_attn.k_proj", kv_out);
      layer.v_bias = bias(base + ".self_attn.v_proj", kv_out);
    }
    // A bias the family does not define means a different model: refuse it
    // rather than silently ignoring it.
    for (const char* module : {".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"}) {
      LK_CHECK(c.qkv_bias || !ck.contains(p + base + module + ".bias"), base + module + ": unexpected bias for " + arch);
    }
    LK_CHECK(c.o_bias || !ck.contains(p + base + ".self_attn.o_proj.bias"),
             base + ".self_attn.o_proj: unexpected bias for " + arch);
    for (const char* module : {".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"}) {
      LK_CHECK(!ck.contains(p + base + module + ".bias"), base + module + ": unexpected bias for " + arch);
    }
    if (c.o_bias) layer.o_bias = bias(base + ".self_attn.o_proj", c.hidden_size);
    if (c.attention_sinks) {
      layer.sinks.resize(static_cast<size_t>(c.num_heads));
      load_vector(ck, p + base + ".self_attn.sinks", c.num_heads, layer.sinks.data());
    }
    layer.q_proj = linear(base + ".self_attn.q_proj");
    if (own_kv) {
      layer.k_proj = linear(base + ".self_attn.k_proj");
      if (!c.k_eq_v(i)) layer.v_proj = linear(base + ".self_attn.v_proj");
      LK_CHECK(layer.k_proj.rows == kv_out && layer.k_proj.cols == c.hidden_size &&
                   (c.k_eq_v(i) || (layer.v_proj.rows == kv_out && layer.v_proj.cols == c.hidden_size)),
               base + ": k/v shape");
    }
    layer.o_proj = linear(base + ".self_attn.o_proj");
    LK_CHECK(layer.q_proj.rows == q_out && layer.q_proj.cols == c.hidden_size, base + ": q_proj shape");
    LK_CHECK(layer.o_proj.rows == c.hidden_size && layer.o_proj.cols == q_out, base + ": o_proj shape");
    if (c.is_moe(i)) bind_experts(*w, layer, base);
    if (c.is_moe(i) && c.experts_beside_mlp) {
      layer.mlp_post_norm_1 = vec(base + ".post_feedforward_layernorm_1", c.hidden_size);
      layer.expert_norm = vec(base + ".pre_feedforward_layernorm_2", c.hidden_size);
      layer.expert_post_norm = vec(base + ".post_feedforward_layernorm_2", c.hidden_size);
      layer.router_input_scale.resize(static_cast<size_t>(c.hidden_size));
      load_vector(ck, p + base + ".router.scale", c.hidden_size, layer.router_input_scale.data());
      const float root = 1.0f / std::sqrt(static_cast<float>(c.hidden_size));
      for (float& scale : layer.router_input_scale) scale *= root;
      layer.per_expert_scale.resize(static_cast<size_t>(c.num_experts));
      load_vector(ck, p + base + ".router.per_expert_scale", c.num_experts, layer.per_expert_scale.data());
    }
    if (!c.is_moe(i) || c.experts_beside_mlp) {
      layer.gate_proj = linear(base + ".mlp.gate_proj");
      layer.up_proj = linear(base + ".mlp.up_proj");
      layer.down_proj = linear(base + ".mlp.down_proj");
      LK_CHECK(layer.gate_proj.rows == inter && layer.up_proj.rows == inter && layer.down_proj.cols == inter &&
                   layer.gate_proj.cols == c.hidden_size && layer.up_proj.cols == c.hidden_size &&
                   layer.down_proj.rows == c.hidden_size,
               base + ": mlp shape");
    }
    if (c.per_layer_input > 0) {
      layer.per_layer_gate = linear(base + ".per_layer_input_gate");
      layer.per_layer_proj = linear(base + ".per_layer_projection");
      layer.per_layer_norm = vec(base + ".post_per_layer_input_norm", c.hidden_size);
      LK_CHECK(layer.per_layer_gate.rows == c.per_layer_input && layer.per_layer_gate.cols == c.hidden_size &&
                   layer.per_layer_proj.rows == c.hidden_size && layer.per_layer_proj.cols == c.per_layer_input,
               base + ": per-layer input shape");
    }
    if (c.layer_scalars) load_vector(ck, p + base + ".layer_scalar", 1, &layer.layer_scalar);
    w->layers.push_back(std::move(layer));
  }
  return w;
}

size_t DecoderWeights::matrix_bytes() const {
  size_t total = embed.weight_bytes() + (tied_lm_head ? 0 : lm_head.weight_bytes()) +
                 embed_per_layer.weight_bytes() + per_layer_model_proj.weight_bytes();
  for (const auto& l : layers) {
    total += l.q_proj.weight_bytes() + l.k_proj.weight_bytes() + l.v_proj.weight_bytes() +
             l.o_proj.weight_bytes() + l.gate_proj.weight_bytes() + l.up_proj.weight_bytes() +
             l.down_proj.weight_bytes() + l.per_layer_gate.weight_bytes() + l.per_layer_proj.weight_bytes() +
             l.router.weight_bytes() + l.shared_gate.weight_bytes() + l.shared_up.weight_bytes() +
             l.shared_down.weight_bytes();
    for (const auto* experts : {&l.expert_gate, &l.expert_up, &l.expert_down}) {
      for (const Linear& expert : *experts) total += expert.weight_bytes();
    }
  }
  return total;
}

bool DecoderWeights::dense_mxfp4() const {
  if (embed.quant.mxfp4() || lm_head.quant.mxfp4() || embed_per_layer.quant.mxfp4() ||
      per_layer_model_proj.quant.mxfp4()) {
    return true;
  }
  for (const auto& l : layers) {
    for (const Linear* m : {&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj, &l.gate_proj, &l.up_proj, &l.down_proj,
                            &l.per_layer_gate, &l.per_layer_proj, &l.router, &l.shared_gate, &l.shared_up,
                            &l.shared_down}) {
      if (m->quant.mxfp4()) return true;
    }
  }
  return false;
}

namespace traffic {

double linear(const Linear& weights, int n) {
  return static_cast<double>(weights.weight_bytes()) + 4.0 * n * static_cast<double>(weights.cols + weights.rows);
}

double embed(const Linear& embed, int n) {
  const double row = embed.quant.quantized()
                         ? 4.0 * static_cast<double>(embed.words_per_row() + embed.groups_per_row())
                         : static_cast<double>(embed.cols) * static_cast<double>(dtype_size(embed.dense_dtype));
  return n * (row + 4.0 * static_cast<double>(embed.cols));
}

double rows(int64_t width, int n, int arrays) { return 4.0 * static_cast<double>(width) * n * arrays; }

double attention(const DecoderConfig& c, int layer, int start, int n) {
  const bool sliding = c.layer_is_sliding[static_cast<size_t>(layer)];
  double keys = 0.0;
  for (int t = 0; t < n; ++t) {
    const int pos = start + t;
    keys += sliding ? std::min(pos + 1, c.sliding_window) : pos + 1;
  }
  const double head_bytes = 4.0 * c.head_dim_of(layer);
  return head_bytes * (2.0 * n * c.num_heads + 2.0 * keys * c.kv_heads_of(layer));
}

}  // namespace traffic

// ---------------------------------------------------------------------------
// CPU backend
// ---------------------------------------------------------------------------

namespace {

void rms_norm(const float* x, const float* weight, int64_t size, float eps, float offset, float* out) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) sum += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(size) + eps));
  if (weight == nullptr) {
    for (int64_t i = 0; i < size; ++i) out[i] = x[i] * inv;
    return;
  }
  for (int64_t i = 0; i < size; ++i) out[i] = x[i] * inv * (offset + weight[i]);
}

float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2 / pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Times the enclosing block into `profiler` as `name` (no-op without one).
class Section {
 public:
  Section(Profiler* profiler, const char* name, double bytes)
      : profiler_(profiler), name_(name), bytes_(bytes), start_(profiler ? now_seconds() : 0.0) {}
  ~Section() {
    if (profiler_) profiler_->add(name_, now_seconds() - start_, bytes_);
  }
  Section(const Section&) = delete;
  Section& operator=(const Section&) = delete;

 private:
  Profiler* profiler_;
  const char* name_;
  double bytes_;
  double start_;
};

class DecoderCpu final : public Model {
 public:
  DecoderCpu(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options)
      : w_(std::move(weights)), c_(w_->config), pool_(options.cpu_threads),
        max_context_(options.max_context), chunk_(std::max(1, options.prefill_chunk)) {
    LK_CHECK(max_context_ > 0, "max_context must be positive");
    const size_t layers = static_cast<size_t>(c_.num_layers);
    kv_k_.resize(layers);
    kv_v_.resize(layers);
    capacity_.resize(layers);
    for (int l = 0; l < c_.num_layers; ++l) {
      max_q_ = std::max<size_t>(max_q_, static_cast<size_t>(c_.num_heads) * c_.head_dim_of(l));
      max_kv_ = std::max<size_t>(max_kv_, static_cast<size_t>(c_.kv_heads_of(l)) * c_.head_dim_of(l));
      max_inter_ = std::max<size_t>(max_inter_, static_cast<size_t>(c_.intermediate_of(l)));
      if (c_.shares_kv(l)) continue;
      // Each pass writes its keys and values before attending, so a sliding
      // ring also holds the window behind the pass's first token: KV-shared
      // layers read the same ring after it has been written.
      const int capacity = c_.layer_is_sliding[static_cast<size_t>(l)]
                               ? std::min(c_.sliding_window + chunk_ - 1, max_context_)
                               : max_context_;
      capacity_[static_cast<size_t>(l)] = capacity;
      const size_t size = static_cast<size_t>(c_.kv_heads_of(l)) * capacity * c_.head_dim_of(l);
      kv_k_[static_cast<size_t>(l)].assign(size, 0.0f);
      kv_v_[static_cast<size_t>(l)].assign(size, 0.0f);
    }
    logits_.resize(static_cast<size_t>(c_.vocab_size));
    inv_freq_local_ = c_.rope_inv_freq(true);
    inv_freq_global_ = c_.rope_inv_freq(false);
  }

  const char* backend() const override { return "cpu"; }
  const char* architecture() const override { return c_.architecture.c_str(); }
  int vocab_size() const override { return c_.vocab_size; }
  int max_context() const override { return max_context_; }
  int position() const override { return position_; }
  size_t weight_bytes() const override { return w_->matrix_bytes(); }
  const std::vector<int32_t>& eos_token_ids() const override { return c_.eos_token_ids; }
  void reset() override {
    position_ = 0;
    injection_ = {};
  }

  const float* prefill(const int32_t* tokens, int n) override {
    LK_CHECK(n > 0, "prefill requires at least one token");
    begin_injection(n);
    for (int start = 0; start < n;) {
      const int count = plan_.pass_length(start, std::min(chunk_, n - start));
      forward(tokens + start, count, start + count == n);
      start += count;
    }
    end_injection();
    return logits_.data();
  }

  const float* decode(int32_t token) override {
    forward(&token, 1, true);
    return logits_.data();
  }

 protected:
  bool accepts_injection() const override { return true; }

 private:
  // Takes the pending injection for a prefill of n tokens at position_.
  void begin_injection(int n) {
    active_ = std::move(injection_);
    injection_ = {};
    if (!active_.empty()) {
      LK_CHECK(c_.per_layer_input == 0, c_.architecture + ": image features with per-layer inputs are not supported yet");
    }
    plan_ = plan_injection(active_, position_, n, c_.hidden_size, chunk_);
  }

  void end_injection() {
    active_ = {};
    plan_ = {};
  }

  // The injected row for position `pos`, or -1.
  int injected_row(int pos) const {
    const int rel = pos - plan_.base;
    return rel >= 0 && rel < static_cast<int>(plan_.row.size()) ? plan_.row[static_cast<size_t>(rel)] : -1;
  }

  // The last key position `pos` attends to: itself, or the last position of
  // its image block.
  int last_key(int pos) const {
    const int rel = pos - plan_.base;
    return rel >= 0 && rel < static_cast<int>(plan_.last_key.size()) ? plan_.last_key[static_cast<size_t>(rel)] : pos;
  }

  // Key or value `kv` of position `pos` in `layer`'s ring.
  const float* cached(const std::vector<std::vector<float>>& cache, int layer, int kv, int pos) const {
    const int capacity = capacity_[static_cast<size_t>(layer)];
    const size_t slot = static_cast<size_t>(kv) * capacity + pos % capacity;
    return cache[static_cast<size_t>(layer)].data() + slot * c_.head_dim_of(layer);
  }

  void store_kv(int layer, int pos, const float* k, const float* v) {
    const int capacity = capacity_[static_cast<size_t>(layer)];
    const int D = c_.head_dim_of(layer);
    for (int kv = 0; kv < c_.kv_heads_of(layer); ++kv) {
      const size_t dst = (static_cast<size_t>(kv) * capacity + pos % capacity) * D;
      std::copy_n(k + static_cast<size_t>(kv) * D, D, kv_k_[static_cast<size_t>(layer)].data() + dst);
      std::copy_n(v + static_cast<size_t>(kv) * D, D, kv_v_[static_cast<size_t>(layer)].data() + dst);
    }
  }

  static void apply_rope(float* vec, int dims, int pos, const std::vector<double>& inv_freq) {
    const int half = dims / 2;
    for (int i = 0; i < half; ++i) {
      const double f = inv_freq[static_cast<size_t>(i)];
      if (f == 0.0) continue;  // a dimension partial rotation leaves alone
      const double angle = static_cast<double>(pos) * f;
      const float cs = static_cast<float>(std::cos(angle));
      const float sn = static_cast<float>(std::sin(angle));
      const float a = vec[i], b = vec[i + half];
      vec[i] = a * cs - b * sn;
      vec[i + half] = b * cs + a * sn;
    }
  }

  void norm(const float* x, const std::vector<float>& weight, int64_t size, float* out) const {
    rms_norm(x, weight.data(), size, c_.rms_norm_eps, c_.norm_offset, out);
  }

  static void add_bias(float* row, const std::vector<float>& bias) {
    for (size_t i = 0; i < bias.size(); ++i) row[i] += bias[i];
  }

  // x += sublayer output r, normalized first when the family uses sandwich norms.
  void add_residual(int n, const std::vector<float>& post_norm) {
    const int64_t hidden = c_.hidden_size;
    for (int t = 0; t < n; ++t) {
      float* xr = x_.data() + t * hidden;
      float* rr = r_.data() + t * hidden;
      if (c_.sandwich_norms) norm(rr, post_norm, hidden, rr);
      for (int64_t i = 0; i < hidden; ++i) xr[i] += rr[i];
    }
  }

  // Gemma 4's per-layer inputs for the pass, [n, layers, P]: the token's
  // per-layer embedding times sqrt(P), plus the normed projection of the
  // scaled token embedding (in x_) times hidden^-0.5, all times 2^-0.5 (the
  // config's constants, rounded to BF16).
  void per_layer_inputs(const int32_t* tokens, int n) {
    const int64_t P = c_.per_layer_input, width = static_cast<int64_t>(c_.num_layers) * P;
    const size_t sn = static_cast<size_t>(n);
    Section section(profiler_, "per_layer_input",
                    profiler_ ? traffic::linear(w_->per_layer_model_proj, n) +
                                    traffic::embed(w_->embed_per_layer, n) + traffic::rows(width, n, 2)
                              : 0.0);
    pl_.resize(sn * width);
    row_.resize(static_cast<size_t>(width));
    matmul(w_->per_layer_model_proj, x_.data(), n, pl_.data(), pool_);
    const float projection_scale = c_.per_layer_projection_scale;
    const float embed_scale = c_.per_layer_embed_scale;
    const float input_scale = c_.per_layer_input_scale;
    for (int t = 0; t < n; ++t) {
      LK_CHECK(tokens[t] < w_->embed_per_layer.rows, "token id has no per-layer embedding");
      dequantize_row(w_->embed_per_layer, tokens[t], row_.data());
      for (int64_t l = 0; l < c_.num_layers; ++l) {
        float* out = pl_.data() + static_cast<size_t>(t) * width + l * P;
        for (int64_t i = 0; i < P; ++i) out[i] *= projection_scale;
        norm(out, w_->per_layer_proj_norm, P, out);
        const float* lookup = row_.data() + l * P;
        for (int64_t i = 0; i < P; ++i) out[i] = (out[i] + lookup[i] * embed_scale) * input_scale;
      }
    }
  }

  void forward(const int32_t* tokens, int n, bool want_logits) {
    LK_CHECK(position_ + n <= max_context_, "context length exceeded");
    LK_CHECK(n <= chunk_, "a forward pass is longer than the prefill chunk");
    const int H = c_.num_heads;
    const int64_t hidden = c_.hidden_size;
    const size_t sn = static_cast<size_t>(n);
    x_.resize(sn * hidden);
    h_.resize(sn * hidden);
    r_.resize(sn * hidden);
    q_.resize(sn * max_q_);
    k_.resize(sn * max_kv_);
    v_.resize(sn * max_kv_);
    attn_.resize(sn * max_q_);
    gate_.resize(sn * max_inter_);
    up_.resize(sn * max_inter_);

    for (int t = 0; t < n; ++t) {
      LK_CHECK(tokens[t] >= 0 && tokens[t] < c_.vocab_size, "token id out of range");
    }
    {
      Section section(profiler_, "embed", profiler_ ? traffic::embed(w_->embed, n) : 0.0);
      for (int t = 0; t < n; ++t) {
        float* row = x_.data() + t * hidden;
        // Injected rows (image features) replace the scaled embedding as is.
        if (const int injected = injected_row(position_ + t); injected >= 0) {
          std::copy_n(active_.rows.data() + static_cast<size_t>(injected) * hidden, hidden, row);
          continue;
        }
        dequantize_row(w_->embed, tokens[t], row);
        if (c_.embed_scale != 1.0f) {
          for (int64_t i = 0; i < hidden; ++i) row[i] *= c_.embed_scale;
        }
      }
    }
    if (c_.per_layer_input > 0) per_layer_inputs(tokens, n);

    const int start = position_;
    // Section traffic is computed only while profiling.
    const bool profiling = profiler_ != nullptr;
    const double norm_bytes = profiling ? traffic::rows(hidden, n, 2) + 4.0 * hidden : 0.0;
    const double residual_bytes = profiling ? traffic::rows(hidden, n, 3) + (c_.sandwich_norms ? 4.0 * hidden : 0.0) : 0.0;
    for (int l = 0; l < c_.num_layers; ++l) {
      const DecoderLayer& L = w_->layers[static_cast<size_t>(l)];
      const bool sliding = c_.layer_is_sliding[static_cast<size_t>(l)];
      const bool own_kv = !c_.shares_kv(l);
      const int D = c_.head_dim_of(l), KV = c_.kv_heads_of(l);
      const int64_t inter = c_.intermediate_of(l);
      {
        Section section(profiler_, "norm", norm_bytes);
        for (int t = 0; t < n; ++t) norm(x_.data() + t * hidden, L.attn_norm, hidden, h_.data() + t * hidden);
      }
      {
        Section section(profiler_, "qkv",
                        profiling ? traffic::linear(L.q_proj, n) +
                                        (own_kv ? traffic::linear(L.k_proj, n) : 0.0) +
                                        (own_kv && !c_.k_eq_v(l) ? traffic::linear(L.v_proj, n) : 0.0)
                                  : 0.0);
        matmul(L.q_proj, h_.data(), n, q_.data(), pool_);
        if (own_kv) {
          matmul(L.k_proj, h_.data(), n, k_.data(), pool_);
          if (c_.k_eq_v(l)) {
            std::copy_n(k_.data(), sn * KV * D, v_.data());  // values are the raw key projection
          } else {
            matmul(L.v_proj, h_.data(), n, v_.data(), pool_);
          }
        }
      }
      {
        Section section(profiler_, "rope_kv", profiling ? traffic::rows(D, n, 2 * H + (own_kv ? (c_.v_norm ? 6 : 4) * KV : 0)) : 0.0);
        if (c_.qkv_bias) {
          for (int t = 0; t < n; ++t) {
            add_bias(q_.data() + static_cast<size_t>(t) * H * D, L.q_bias);
            add_bias(k_.data() + static_cast<size_t>(t) * KV * D, L.k_bias);
            add_bias(v_.data() + static_cast<size_t>(t) * KV * D, L.v_bias);
          }
        }
        const auto& inv_freq = sliding ? inv_freq_local_ : inv_freq_global_;
        const int rotated = c_.rotary_dim > 0 ? c_.rotary_dim : D;
        for (int t = 0; t < n; ++t) {
          for (int hh = 0; hh < H; ++hh) {
            float* qv = q_.data() + (static_cast<size_t>(t) * H + hh) * D;
            if (c_.qk_norm) norm(qv, L.q_norm, D, qv);
            apply_rope(qv, rotated, start + t, inv_freq);
          }
          if (!own_kv) continue;
          for (int kv = 0; kv < KV; ++kv) {
            float* kvec = k_.data() + (static_cast<size_t>(t) * KV + kv) * D;
            if (c_.qk_norm) norm(kvec, L.k_norm, D, kvec);
            apply_rope(kvec, rotated, start + t, inv_freq);
            if (c_.v_norm) {
              float* vvec = v_.data() + (static_cast<size_t>(t) * KV + kv) * D;
              rms_norm(vvec, nullptr, D, c_.rms_norm_eps, 0.0f, vvec);
            }
          }
          store_kv(l, start + t, k_.data() + static_cast<size_t>(t) * KV * D,
                   v_.data() + static_cast<size_t>(t) * KV * D);
        }
      }
      {
        Section section(profiler_, "attention", profiling ? traffic::attention(c_, l, start, n) : 0.0);
        attend(l, start, n, sliding, c_.attention_sinks ? L.sinks.data() : nullptr);
      }
      {
        Section section(profiler_, "o_proj", profiling ? traffic::linear(L.o_proj, n) : 0.0);
        matmul(L.o_proj, attn_.data(), n, r_.data(), pool_);
        if (c_.o_bias) {
          for (int t = 0; t < n; ++t) add_bias(r_.data() + static_cast<size_t>(t) * hidden, L.o_bias);
        }
      }
      {
        Section section(profiler_, "residual", residual_bytes);
        add_residual(n, L.attn_post_norm);
      }
      {
        Section section(profiler_, "norm", norm_bytes);
        for (int t = 0; t < n; ++t) norm(x_.data() + t * hidden, L.mlp_norm, hidden, h_.data() + t * hidden);
      }
      if (c_.is_moe(l) && !c_.experts_beside_mlp) {
        experts(L, n, h_.data(), h_.data());
      } else {
        {
          Section section(profiler_, "mlp_up",
                          profiling ? traffic::linear(L.gate_proj, n) + traffic::linear(L.up_proj, n) +
                                          traffic::rows(inter, n, 1)
                                    : 0.0);
          matmul(L.gate_proj, h_.data(), n, gate_.data(), pool_);
          matmul(L.up_proj, h_.data(), n, up_.data(), pool_);
          gated_activation(gate_.data(), up_.data(), sn * inter);
        }
        Section section(profiler_, "down_proj", profiling ? traffic::linear(L.down_proj, n) : 0.0);
        matmul(L.down_proj, gate_.data(), n, r_.data(), pool_);
      }
      if (c_.is_moe(l) && c_.experts_beside_mlp) experts_beside_mlp(L, n);
      {
        Section section(profiler_, "residual", residual_bytes);
        add_residual(n, L.mlp_post_norm);
      }
      if (c_.per_layer_input > 0) gate_per_layer_input(L, l, n);
      if (L.layer_scalar != 1.0f) {
        Section section(profiler_, "residual", profiling ? traffic::rows(hidden, n, 2) : 0.0);
        for (size_t i = 0; i < sn * hidden; ++i) x_[i] *= L.layer_scalar;
      }
    }
    position_ += n;
    if (!want_logits) return;
    // Only the last row's logits are needed: skip the vocabulary projection
    // for every other prompt token.
    Section section(profiler_, "lm_head", profiling ? traffic::linear(w_->lm_head, 1) + 4.0 * hidden : 0.0);
    std::vector<float> last(static_cast<size_t>(hidden));
    norm(x_.data() + static_cast<size_t>(n - 1) * hidden, w_->final_norm, hidden, last.data());
    matvec(w_->lm_head, last.data(), logits_.data(), pool_);
    if (c_.final_logit_softcapping > 0.0f) {
      const float cap = c_.final_logit_softcapping;
      for (float& v : logits_) v = std::tanh(v / cap) * cap;
    }
  }

  // gate[i] = act(gate[i]) * up[i]
  void gated_activation(float* gate, const float* up, size_t count) const {
    if (c_.activation == Activation::GeluTanh) {
      for (size_t i = 0; i < count; ++i) gate[i] = gelu_tanh(gate[i]) * up[i];
    } else if (c_.activation == Activation::ClampedSwiglu) {
      const float limit = c_.swiglu_limit;
      for (size_t i = 0; i < count; ++i) {
        const float g = std::min(gate[i], limit);
        const float u = std::clamp(up[i], -limit, limit);
        gate[i] = g / (1.0f + std::exp(-1.702f * g)) * (u + 1.0f);
      }
    } else {
      for (size_t i = 0; i < count; ++i) gate[i] = silu(gate[i]) * up[i];
    }
  }

  // GLM-4.5 / DeepSeek V3 (noaux_tc) routing of one token: sigmoid scores
  // plus the correction bias choose the top k within the best topk_group
  // groups (by the sum of each group's two best); the chosen experts' raw
  // scores, renormalized when norm_topk_prob, times routed_scaling, weight
  // them. Ties go to the lower group or expert index.
  void route_sigmoid_grouped(const DecoderLayer& L, const float* logits, int* chosen, float* weights) const {
    const int E = c_.num_experts, K = c_.experts_per_token, groups = c_.n_group, per_group = E / groups;
    std::vector<double> scores(static_cast<size_t>(E)), selection(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) {
      scores[static_cast<size_t>(e)] = 1.0 / (1.0 + std::exp(-static_cast<double>(logits[e])));
      selection[static_cast<size_t>(e)] = scores[static_cast<size_t>(e)] + L.router_correction[static_cast<size_t>(e)];
    }
    if (groups > 1) {
      std::vector<std::pair<double, int>> group_scores;
      for (int g = 0; g < groups; ++g) {
        std::vector<double> member(selection.begin() + g * per_group, selection.begin() + (g + 1) * per_group);
        std::partial_sort(member.begin(), member.begin() + std::min(2, per_group), member.end(), std::greater<>());
        group_scores.push_back({member[0] + (per_group > 1 ? member[1] : 0.0), g});
      }
      // The references zero the n_group - topk_group lowest groups.
      std::stable_sort(group_scores.begin(), group_scores.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      for (int j = 0; j < groups - c_.topk_group; ++j) {
        const int g = group_scores[static_cast<size_t>(j)].second;
        std::fill(selection.begin() + g * per_group, selection.begin() + (g + 1) * per_group, 0.0);
      }
    }
    std::vector<int> order(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) order[static_cast<size_t>(e)] = e;
    std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int b) {
      const double sa = selection[static_cast<size_t>(a)], sb = selection[static_cast<size_t>(b)];
      return sa > sb || (sa == sb && a < b);
    });
    double total = 0.0;
    for (int j = 0; j < K; ++j) total += scores[static_cast<size_t>(order[static_cast<size_t>(j)])];
    const double scale = (K > 1 && c_.norm_topk_prob ? 1.0 / total : 1.0) * c_.routed_scaling;
    for (int j = 0; j < K; ++j) {
      chosen[j] = order[static_cast<size_t>(j)];
      weights[j] = static_cast<float>(scores[static_cast<size_t>(chosen[j])] * scale);
    }
  }

  // rows[t] += bias[e * width ...] for each of n rows of `width` values.
  static void add_expert_bias(float* rows, int n, int64_t width, const std::vector<float>& bias, int e) {
    if (bias.empty()) return;
    const float* b = bias.data() + static_cast<size_t>(e) * width;
    for (int t = 0; t < n; ++t) {
      float* row = rows + static_cast<size_t>(t) * width;
      for (int64_t i = 0; i < width; ++i) row[i] += b[i];
    }
  }

  // r_ = the mixture-of-experts output for the normed rows in h_: each token's
  // top experts by router logit, weighted by their router probabilities
  // (softmax over all experts, renormalized with norm_topk_prob, or a softmax
  // over the chosen logits). Experts run once per pass over the tokens routed
  // to them.
  // Gemma 4's experts beside the dense MLP, whose output is in r_: r_ becomes
  // mlp_post_norm_1(r_) + expert_post_norm(experts(expert_norm(x))), routed
  // on rms(x) * router_input_scale.
  void experts_beside_mlp(const DecoderLayer& L, int n) {
    const int64_t hidden = c_.hidden_size;
    const size_t sn = static_cast<size_t>(n);
    dense_out_.resize(sn * hidden);
    router_in_.resize(sn * hidden);
    {
      Section section(profiler_, "norm", profiler_ ? traffic::rows(hidden, n, 6) + 12.0 * hidden : 0.0);
      for (int t = 0; t < n; ++t) {
        const size_t row = static_cast<size_t>(t) * hidden;
        norm(r_.data() + row, L.mlp_post_norm_1, hidden, dense_out_.data() + row);
        norm(x_.data() + row, L.router_input_scale, hidden, router_in_.data() + row);
        norm(x_.data() + row, L.expert_norm, hidden, h_.data() + row);
      }
    }
    experts(L, n, router_in_.data(), h_.data());
    Section section(profiler_, "norm", profiler_ ? traffic::rows(hidden, n, 3) + 4.0 * hidden : 0.0);
    for (int t = 0; t < n; ++t) {
      float* row = r_.data() + static_cast<size_t>(t) * hidden;
      norm(row, L.expert_post_norm, hidden, row);
      const float* dense = dense_out_.data() + static_cast<size_t>(t) * hidden;
      for (int64_t i = 0; i < hidden; ++i) row[i] += dense[i];
    }
  }

  // The layer's experts into r_: routed on `router_in` rows, run on
  // `expert_in` rows (both [n, hidden]; the shared expert, if any, too).
  void experts(const DecoderLayer& L, int n, const float* router_in, const float* expert_in) {
    const int E = c_.num_experts, K = c_.experts_per_token;
    const int64_t hidden = c_.hidden_size, inter = c_.moe_intermediate;
    const size_t sn = static_cast<size_t>(n);
    route_.resize(sn * E);
    chosen_.resize(sn * K);
    weight_.resize(sn * K);
    {
      Section section(profiler_, "router", profiler_ ? traffic::linear(L.router, n) : 0.0);
      matmul(L.router, router_in, n, route_.data(), pool_);
      std::vector<int> order(static_cast<size_t>(E));
      std::vector<double> probs(static_cast<size_t>(E));
      for (int t = 0; t < n; ++t) {
        float* logits = route_.data() + static_cast<size_t>(t) * E;
        if (c_.router == DecoderConfig::Router::SigmoidGrouped) {
          route_sigmoid_grouped(L, logits, chosen_.data() + static_cast<size_t>(t) * K,
                                weight_.data() + static_cast<size_t>(t) * K);
          continue;
        }
        if (!L.router_bias.empty()) add_bias(logits, L.router_bias);
        const float top = *std::max_element(logits, logits + E);
        double total = 0.0;
        for (int e = 0; e < E; ++e) total += probs[static_cast<size_t>(e)] = std::exp(static_cast<double>(logits[e]) - top);
        for (int e = 0; e < E; ++e) order[static_cast<size_t>(e)] = e;
        std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int b) {
          const double pa = probs[static_cast<size_t>(a)], pb = probs[static_cast<size_t>(b)];
          return pa > pb || (pa == pb && a < b);
        });
        double kept = 0.0;
        for (int j = 0; j < K; ++j) kept += probs[static_cast<size_t>(order[static_cast<size_t>(j)])];
        // Over the chosen experts, a softmax of their logits is their
        // probabilities renormalized.
        const bool renormalize = c_.norm_topk_prob || c_.router == DecoderConfig::Router::TopKSoftmax;
        const double scale = renormalize ? 1.0 / kept : 1.0 / total;
        for (int j = 0; j < K; ++j) {
          const int e = order[static_cast<size_t>(j)];
          chosen_[static_cast<size_t>(t) * K + j] = e;
          float weight = static_cast<float>(probs[static_cast<size_t>(e)] * scale);
          if (!L.per_expert_scale.empty()) weight *= L.per_expert_scale[static_cast<size_t>(e)];
          weight_[static_cast<size_t>(t) * K + j] = weight;
        }
      }
    }
    // Each expert's tokens, gathered so the expert's weights are read once.
    std::vector<std::vector<std::pair<int, float>>> routed(static_cast<size_t>(E));
    for (int t = 0; t < n; ++t) {
      for (int j = 0; j < K; ++j) {
        const size_t slot = static_cast<size_t>(t) * K + j;
        routed[static_cast<size_t>(chosen_[slot])].push_back({t, weight_[slot]});
      }
    }
    double bytes = 0.0;
    if (profiler_) {
      for (int e = 0; e < E; ++e) {
        const int m = static_cast<int>(routed[static_cast<size_t>(e)].size());
        if (m == 0) continue;
        bytes += traffic::linear(L.expert_gate[static_cast<size_t>(e)], m) +
                 traffic::linear(L.expert_up[static_cast<size_t>(e)], m) +
                 traffic::linear(L.expert_down[static_cast<size_t>(e)], m) + traffic::rows(inter, m, 1) +
                 traffic::rows(hidden, m, 2);
      }
    }
    if (profiler_ && c_.shared_intermediate > 0) {
      bytes += traffic::linear(L.shared_gate, n) + traffic::linear(L.shared_up, n) +
               traffic::linear(L.shared_down, n) + traffic::rows(c_.shared_intermediate, n, 1);
    }
    Section section(profiler_, "experts", bytes);
    std::fill(r_.begin(), r_.begin() + static_cast<std::ptrdiff_t>(sn * hidden), 0.0f);
    if (c_.shared_intermediate > 0) {  // the shared expert sees every token
      expert_gate_.resize(sn * c_.shared_intermediate);
      expert_up_.resize(sn * c_.shared_intermediate);
      matmul(L.shared_gate, expert_in, n, expert_gate_.data(), pool_);
      matmul(L.shared_up, expert_in, n, expert_up_.data(), pool_);
      gated_activation(expert_gate_.data(), expert_up_.data(), sn * c_.shared_intermediate);
      matmul(L.shared_down, expert_gate_.data(), n, r_.data(), pool_);
    }
    for (int e = 0; e < E; ++e) {
      const auto& tokens = routed[static_cast<size_t>(e)];
      const int m = static_cast<int>(tokens.size());
      if (m == 0) continue;
      const size_t sm = static_cast<size_t>(m);
      expert_in_.resize(sm * hidden);
      expert_gate_.resize(sm * inter);
      expert_up_.resize(sm * inter);
      expert_out_.resize(sm * hidden);
      for (int i = 0; i < m; ++i) {
        std::copy_n(expert_in + static_cast<size_t>(tokens[static_cast<size_t>(i)].first) * hidden, hidden,
                    expert_in_.data() + static_cast<size_t>(i) * hidden);
      }
      matmul(L.expert_gate[static_cast<size_t>(e)], expert_in_.data(), m, expert_gate_.data(), pool_);
      matmul(L.expert_up[static_cast<size_t>(e)], expert_in_.data(), m, expert_up_.data(), pool_);
      add_expert_bias(expert_gate_.data(), m, inter, L.expert_gate_bias, e);
      add_expert_bias(expert_up_.data(), m, inter, L.expert_up_bias, e);
      gated_activation(expert_gate_.data(), expert_up_.data(), sm * inter);
      matmul(L.expert_down[static_cast<size_t>(e)], expert_gate_.data(), m, expert_out_.data(), pool_);
      add_expert_bias(expert_out_.data(), m, hidden, L.expert_down_bias, e);
      for (int i = 0; i < m; ++i) {
        const auto [t, weight] = tokens[static_cast<size_t>(i)];
        float* out = r_.data() + static_cast<size_t>(t) * hidden;
        const float* y = expert_out_.data() + static_cast<size_t>(i) * hidden;
        for (int64_t d = 0; d < hidden; ++d) out[d] += weight * y[d];
      }
    }
  }

  // x += norm(per_layer_projection(gelu(per_layer_input_gate(x)) * input)),
  // with this layer's slice of the pass's per-layer inputs.
  void gate_per_layer_input(const DecoderLayer& L, int l, int n) {
    const int64_t P = c_.per_layer_input, width = static_cast<int64_t>(c_.num_layers) * P;
    Section section(profiler_, "per_layer_input",
                    profiler_ ? traffic::linear(L.per_layer_gate, n) + traffic::linear(L.per_layer_proj, n) +
                                    traffic::rows(P, n, 1) + traffic::rows(c_.hidden_size, n, 2)
                              : 0.0);
    gate_.resize(std::max(gate_.size(), static_cast<size_t>(n * P)));
    matmul(L.per_layer_gate, x_.data(), n, gate_.data(), pool_);
    for (int t = 0; t < n; ++t) {
      float* g = gate_.data() + static_cast<size_t>(t) * P;
      const float* input = pl_.data() + static_cast<size_t>(t) * width + static_cast<int64_t>(l) * P;
      for (int64_t i = 0; i < P; ++i) g[i] = gelu_tanh(g[i]) * input[i];
    }
    matmul(L.per_layer_proj, gate_.data(), n, r_.data(), pool_);
    const int64_t hidden = c_.hidden_size;
    for (int t = 0; t < n; ++t) {
      float* xr = x_.data() + t * hidden;
      float* rr = r_.data() + t * hidden;
      norm(rr, L.per_layer_norm, hidden, rr);
      for (int64_t i = 0; i < hidden; ++i) xr[i] += rr[i];
    }
  }

  // Attention for the n tokens of this pass (queries in q_) over the keys and
  // values in the ring of layer l's KV source, into attn_.
  void attend(int l, int start, int n, bool sliding, const float* sinks) {
    const int H = c_.num_heads, D = c_.head_dim_of(l);
    const int group = H / c_.kv_heads_of(l);
    const int source = c_.kv_source(l);
    const bool blocks = c_.image_blocks_apply(sliding);
    pool_.parallel_for(static_cast<size_t>(n) * H, [&](size_t begin, size_t end) {
      std::vector<float> scores;
      for (size_t item = begin; item < end; ++item) {
        const int t = static_cast<int>(item / H);
        const int hh = static_cast<int>(item % H);
        const int kv = hh / group;
        const int pos = start + t;
        // Keys [lo, hi]: causal, except that an image block's tokens also see
        // its later positions (all in this pass); the window still applies.
        const int lo = sliding ? std::max(0, pos - c_.sliding_window + 1) : 0;
        const int hi = blocks ? last_key(pos) : pos;
        const float* qv = q_.data() + (static_cast<size_t>(t) * H + hh) * D;
        scores.resize(static_cast<size_t>(hi - lo + 1));
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = lo; j <= hi; ++j) {
          const float* kvec = cached(kv_k_, source, kv, j);
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += qv[d] * kvec[d];
          dot *= c_.attn_scale;
          if (c_.attn_logit_softcapping > 0.0f) {
            dot = std::tanh(dot / c_.attn_logit_softcapping) * c_.attn_logit_softcapping;
          }
          scores[static_cast<size_t>(j - lo)] = dot;
          max_score = std::max(max_score, dot);
        }
        // A sink is one more logit in the softmax that attends to nothing.
        if (sinks != nullptr) max_score = std::max(max_score, sinks[hh]);
        float denom = sinks != nullptr ? std::exp(sinks[hh] - max_score) : 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        float* out = attn_.data() + (static_cast<size_t>(t) * H + hh) * D;
        std::fill(out, out + D, 0.0f);
        for (int j = lo; j <= hi; ++j) {
          const float p = scores[static_cast<size_t>(j - lo)] / denom;
          const float* vvec = cached(kv_v_, source, kv, j);
          for (int d = 0; d < D; ++d) out[d] += p * vvec[d];
        }
      }
    });
  }

  std::shared_ptr<DecoderWeights> w_;
  const DecoderConfig& c_;
  ThreadPool pool_;
  int max_context_;
  int chunk_;
  int position_ = 0;
  size_t max_q_ = 0, max_kv_ = 0, max_inter_ = 0;  // per-token widths, the largest over layers
  std::vector<int> capacity_;                       // ring positions per layer; 0 for KV-shared layers
  // The injection of the prefill in progress and its plan.
  PromptInjection active_;
  InjectionPlan plan_;
  std::vector<std::vector<float>> kv_k_, kv_v_;
  std::vector<double> inv_freq_local_, inv_freq_global_;
  std::vector<float> x_, h_, r_, q_, k_, v_, attn_, gate_, up_, logits_;
  std::vector<float> pl_, row_;  // per-layer inputs [n, layers * P] and an embedding row
  // Mixture of experts: router logits [n, E], each token's experts and
  // weights [n, K], and one expert's gathered inputs and activations.
  std::vector<float> route_, weight_, expert_in_, expert_gate_, expert_up_, expert_out_;
  std::vector<float> dense_out_, router_in_;  // Gemma 4: the dense branch's output, the router's input
  std::vector<int> chosen_;
};

}  // namespace

std::unique_ptr<Model> make_decoder_cpu(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options) {
  return std::make_unique<DecoderCpu>(std::move(weights), options);
}

}  // namespace lokahi
