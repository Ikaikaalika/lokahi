#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common.h"
#include "half.h"
#include "thread_pool.h"

namespace lokahi {

static const Json& text_config(const Json& raw) {
  const Json* text = raw.find("text_config");
  return text != nullptr && text->is_object() ? *text : raw;
}

namespace {

std::string model_type(const Json& raw) {
  return text_config(raw).string_or("model_type", raw.string_or("model_type", ""));
}

bool is_gemma3(const Json& raw) {
  const std::string type = model_type(raw);
  return type == "gemma3_text" || type == "gemma3";
}

bool is_llama_family(const Json& raw) {
  const std::string type = model_type(raw);
  return type == "llama" || type == "mistral" || type == "qwen2" || type == "qwen3";
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

DecoderConfig gemma3_config(const Json& raw) {
  const Json& cfg = text_config(raw);
  DecoderConfig c;
  c.architecture = "gemma3";
  c.vocab_size = static_cast<int>(cfg.at("vocab_size").as_int());
  c.hidden_size = static_cast<int>(cfg.at("hidden_size").as_int());
  c.intermediate_size = static_cast<int>(cfg.at("intermediate_size").as_int());
  c.num_layers = static_cast<int>(cfg.at("num_hidden_layers").as_int());
  c.num_heads = static_cast<int>(cfg.at("num_attention_heads").as_int());
  c.num_kv_heads = static_cast<int>(cfg.at("num_key_value_heads").as_int());
  c.head_dim = static_cast<int>(cfg.int_or("head_dim", 256));
  c.rms_norm_eps = static_cast<float>(cfg.number_or("rms_norm_eps", 1e-6));
  c.norm_offset = 1.0f;
  c.sandwich_norms = true;
  c.qk_norm = true;
  // sqrt(hidden_size) rounded to BF16, as the reference implementations do.
  c.embed_scale = bf16_to_f32(f32_to_bf16(std::sqrt(static_cast<float>(c.hidden_size))));
  c.attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg.number_or("query_pre_attn_scalar", c.head_dim)));
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

// Llama 3.x, Mistral, Qwen 2/2.5 and Qwen 3: pre-norm blocks with plain
// RMSNorm, SiLU-gated MLPs and rotate-half RoPE. Qwen 3 adds per-head Q/K
// norm, Qwen 2 adds q/k/v biases, Mistral may use a sliding window on every
// layer. Options these checkpoints do not use fail closed.
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
  c.qk_norm = arch == "qwen3";
  c.qkv_bias = arch == "qwen2";
  c.attn_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
  c.rope_theta = static_cast<float>(cfg.number_or("rope_theta", 1e4));

  const std::string activation = cfg.string_or("hidden_act", "silu");
  LK_CHECK(activation == "silu", arch + ": unsupported activation " + activation);
  // These add o_proj or MLP biases, which no supported checkpoint uses.
  LK_CHECK(!flag(cfg, "attention_bias"), arch + ": attention_bias is not supported");
  LK_CHECK(!flag(cfg, "mlp_bias"), arch + ": mlp_bias is not supported");
  const double rotary = cfg.number_or("partial_rotary_factor", 1.0);
  LK_CHECK(rotary == 1.0, arch + ": partial_rotary_factor is not supported");

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
  return c;
}

}  // namespace

bool is_supported_decoder(const Json& raw) { return is_gemma3(raw) || is_llama_family(raw); }

DecoderConfig DecoderConfig::from_json(const Json& raw) {
  LK_CHECK(is_supported_decoder(raw),
           "unsupported architecture '" + model_type(raw) +
               "'; this build supports gemma3, llama, mistral, qwen2 and qwen3 text models");
  DecoderConfig c = is_gemma3(raw) ? gemma3_config(raw) : llama_family_config(raw);
  LK_CHECK(static_cast<int>(c.layer_is_sliding.size()) == c.num_layers,
           c.architecture + ": layer_types length mismatch");
  LK_CHECK(c.num_heads > 0 && c.num_kv_heads > 0 && c.num_heads % c.num_kv_heads == 0,
           c.architecture + ": heads must be a multiple of kv heads");
  LK_CHECK(c.head_dim > 0 && c.head_dim % 2 == 0, c.architecture + ": head_dim must be even");
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
  std::vector<double> inv_freq(static_cast<size_t>(head_dim / 2));
  for (size_t i = 0; i < inv_freq.size(); ++i) {
    double f = 1.0 / std::pow(base, static_cast<double>(2 * i) / head_dim);
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

int DecoderConfig::kv_capacity(int layer, int max_context) const {
  return layer_is_sliding[static_cast<size_t>(layer)] ? std::min(sliding_window, max_context)
                                                       : max_context;
}

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
  const int64_t q_out = static_cast<int64_t>(c.num_heads) * c.head_dim;
  const int64_t kv_out = static_cast<int64_t>(c.num_kv_heads) * c.head_dim;
  for (int i = 0; i < c.num_layers; ++i) {
    const std::string base = "model.layers." + std::to_string(i);
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
      layer.q_norm = vec(base + ".self_attn.q_norm", c.head_dim);
      layer.k_norm = vec(base + ".self_attn.k_norm", c.head_dim);
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
    for (const char* module : {".self_attn.o_proj", ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"}) {
      LK_CHECK(!ck.contains(p + base + module + ".bias"), base + module + ": unexpected bias for " + arch);
    }
    layer.q_proj = linear(base + ".self_attn.q_proj");
    layer.k_proj = linear(base + ".self_attn.k_proj");
    layer.v_proj = linear(base + ".self_attn.v_proj");
    layer.o_proj = linear(base + ".self_attn.o_proj");
    layer.gate_proj = linear(base + ".mlp.gate_proj");
    layer.up_proj = linear(base + ".mlp.up_proj");
    layer.down_proj = linear(base + ".mlp.down_proj");
    LK_CHECK(layer.q_proj.rows == q_out && layer.q_proj.cols == c.hidden_size, base + ": q_proj shape");
    LK_CHECK(layer.k_proj.rows == kv_out && layer.v_proj.rows == kv_out, base + ": k/v shape");
    LK_CHECK(layer.o_proj.rows == c.hidden_size && layer.o_proj.cols == q_out, base + ": o_proj shape");
    LK_CHECK(layer.gate_proj.rows == c.intermediate_size && layer.up_proj.rows == c.intermediate_size &&
                 layer.down_proj.cols == c.intermediate_size,
             base + ": mlp shape");
    w->layers.push_back(std::move(layer));
  }
  return w;
}

size_t DecoderWeights::matrix_bytes() const {
  size_t total = embed.weight_bytes() + (tied_lm_head ? 0 : lm_head.weight_bytes());
  for (const auto& l : layers) {
    total += l.q_proj.weight_bytes() + l.k_proj.weight_bytes() + l.v_proj.weight_bytes() +
             l.o_proj.weight_bytes() + l.gate_proj.weight_bytes() + l.up_proj.weight_bytes() +
             l.down_proj.weight_bytes();
  }
  return total;
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
  const double head_bytes = 4.0 * c.head_dim;
  return head_bytes * (2.0 * n * c.num_heads + 2.0 * keys * c.num_kv_heads);
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
    kv_k_.resize(static_cast<size_t>(c_.num_layers));
    kv_v_.resize(static_cast<size_t>(c_.num_layers));
    for (int l = 0; l < c_.num_layers; ++l) {
      size_t size = static_cast<size_t>(c_.num_kv_heads) * c_.kv_capacity(l, max_context_) * c_.head_dim;
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
  void reset() override { position_ = 0; }

  const float* prefill(const int32_t* tokens, int n) override {
    LK_CHECK(n > 0, "prefill requires at least one token");
    for (int start = 0; start < n; start += chunk_) {
      int count = std::min(chunk_, n - start);
      forward(tokens + start, count, start + count == n);
    }
    return logits_.data();
  }

  const float* decode(int32_t token) override {
    forward(&token, 1, true);
    return logits_.data();
  }

 private:
  const float* key_ptr(int layer, int kv, int pos, int chunk_start, const float* chunk_k) const {
    if (c_.layer_is_sliding[static_cast<size_t>(layer)] && pos >= chunk_start) {
      return chunk_k + (static_cast<size_t>(pos - chunk_start) * c_.num_kv_heads + kv) * c_.head_dim;
    }
    int capacity = c_.kv_capacity(layer, max_context_);
    int slot = pos % capacity;
    return kv_k_[static_cast<size_t>(layer)].data() +
           (static_cast<size_t>(kv) * capacity + slot) * c_.head_dim;
  }

  const float* value_ptr(int layer, int kv, int pos, int chunk_start, const float* chunk_v) const {
    if (c_.layer_is_sliding[static_cast<size_t>(layer)] && pos >= chunk_start) {
      return chunk_v + (static_cast<size_t>(pos - chunk_start) * c_.num_kv_heads + kv) * c_.head_dim;
    }
    int capacity = c_.kv_capacity(layer, max_context_);
    int slot = pos % capacity;
    return kv_v_[static_cast<size_t>(layer)].data() +
           (static_cast<size_t>(kv) * capacity + slot) * c_.head_dim;
  }

  void store_kv(int layer, int pos, const float* k, const float* v) {
    int capacity = c_.kv_capacity(layer, max_context_);
    int slot = pos % capacity;
    for (int kv = 0; kv < c_.num_kv_heads; ++kv) {
      size_t dst = (static_cast<size_t>(kv) * capacity + slot) * c_.head_dim;
      std::copy_n(k + static_cast<size_t>(kv) * c_.head_dim, c_.head_dim,
                  kv_k_[static_cast<size_t>(layer)].data() + dst);
      std::copy_n(v + static_cast<size_t>(kv) * c_.head_dim, c_.head_dim,
                  kv_v_[static_cast<size_t>(layer)].data() + dst);
    }
  }

  void apply_rope(float* vec, int pos, const std::vector<double>& inv_freq) const {
    const int half = c_.head_dim / 2;
    for (int i = 0; i < half; ++i) {
      double angle = static_cast<double>(pos) * inv_freq[static_cast<size_t>(i)];
      float cs = static_cast<float>(std::cos(angle));
      float sn = static_cast<float>(std::sin(angle));
      float a = vec[i], b = vec[i + half];
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

  void forward(const int32_t* tokens, int n, bool want_logits) {
    LK_CHECK(position_ + n <= max_context_, "context length exceeded");
    const int H = c_.num_heads, KV = c_.num_kv_heads, D = c_.head_dim;
    const int64_t hidden = c_.hidden_size, inter = c_.intermediate_size;
    const size_t sn = static_cast<size_t>(n);
    x_.resize(sn * hidden);
    h_.resize(sn * hidden);
    r_.resize(sn * hidden);
    q_.resize(sn * H * D);
    k_.resize(sn * KV * D);
    v_.resize(sn * KV * D);
    attn_.resize(sn * H * D);
    gate_.resize(sn * inter);
    up_.resize(sn * inter);

    for (int t = 0; t < n; ++t) {
      LK_CHECK(tokens[t] >= 0 && tokens[t] < c_.vocab_size, "token id out of range");
    }
    {
      Section section(profiler_, "embed", profiler_ ? traffic::embed(w_->embed, n) : 0.0);
      for (int t = 0; t < n; ++t) {
        float* row = x_.data() + t * hidden;
        dequantize_row(w_->embed, tokens[t], row);
        if (c_.embed_scale != 1.0f) {
          for (int64_t i = 0; i < hidden; ++i) row[i] *= c_.embed_scale;
        }
      }
    }

    const int start = position_;
    const int group = H / KV;
    // Section traffic is computed only while profiling.
    const bool profiling = profiler_ != nullptr;
    const double norm_bytes = profiling ? traffic::rows(hidden, n, 2) + 4.0 * hidden : 0.0;
    const double residual_bytes = profiling ? traffic::rows(hidden, n, 3) + (c_.sandwich_norms ? 4.0 * hidden : 0.0) : 0.0;
    for (int l = 0; l < c_.num_layers; ++l) {
      const DecoderLayer& L = w_->layers[static_cast<size_t>(l)];
      const bool sliding = c_.layer_is_sliding[static_cast<size_t>(l)];
      {
        Section section(profiler_, "norm", norm_bytes);
        for (int t = 0; t < n; ++t) norm(x_.data() + t * hidden, L.attn_norm, hidden, h_.data() + t * hidden);
      }
      {
        Section section(profiler_, "qkv",
                        profiling ? traffic::linear(L.q_proj, n) + traffic::linear(L.k_proj, n) +
                                        traffic::linear(L.v_proj, n)
                                  : 0.0);
        matmul(L.q_proj, h_.data(), n, q_.data(), pool_);
        matmul(L.k_proj, h_.data(), n, k_.data(), pool_);
        matmul(L.v_proj, h_.data(), n, v_.data(), pool_);
      }
      {
        Section section(profiler_, "rope_kv", profiling ? traffic::rows(D, n, 2 * H + 4 * KV) : 0.0);
        if (c_.qkv_bias) {
          for (int t = 0; t < n; ++t) {
            add_bias(q_.data() + static_cast<size_t>(t) * H * D, L.q_bias);
            add_bias(k_.data() + static_cast<size_t>(t) * KV * D, L.k_bias);
            add_bias(v_.data() + static_cast<size_t>(t) * KV * D, L.v_bias);
          }
        }
        const auto& inv_freq = sliding ? inv_freq_local_ : inv_freq_global_;
        for (int t = 0; t < n; ++t) {
          for (int hh = 0; hh < H; ++hh) {
            float* qv = q_.data() + (static_cast<size_t>(t) * H + hh) * D;
            if (c_.qk_norm) norm(qv, L.q_norm, D, qv);
            apply_rope(qv, start + t, inv_freq);
          }
          for (int kv = 0; kv < KV; ++kv) {
            float* kvec = k_.data() + (static_cast<size_t>(t) * KV + kv) * D;
            if (c_.qk_norm) norm(kvec, L.k_norm, D, kvec);
            apply_rope(kvec, start + t, inv_freq);
          }
        }
        // Global layers attend through the cache, so write the chunk first.
        // Sliding layers read the chunk directly and update the ring after.
        if (!sliding) {
          for (int t = 0; t < n; ++t) {
            store_kv(l, start + t, k_.data() + static_cast<size_t>(t) * KV * D,
                     v_.data() + static_cast<size_t>(t) * KV * D);
          }
        }
      }
      {
        Section section(profiler_, "attention", profiling ? traffic::attention(c_, l, start, n) : 0.0);
        attend(l, start, n, sliding, group);
        if (sliding) {
          const int first = std::max(0, n - c_.kv_capacity(l, max_context_));
          for (int t = first; t < n; ++t) {
            store_kv(l, start + t, k_.data() + static_cast<size_t>(t) * KV * D,
                     v_.data() + static_cast<size_t>(t) * KV * D);
          }
        }
      }
      {
        Section section(profiler_, "o_proj", profiling ? traffic::linear(L.o_proj, n) : 0.0);
        matmul(L.o_proj, attn_.data(), n, r_.data(), pool_);
      }
      {
        Section section(profiler_, "residual", residual_bytes);
        add_residual(n, L.attn_post_norm);
      }
      {
        Section section(profiler_, "norm", norm_bytes);
        for (int t = 0; t < n; ++t) norm(x_.data() + t * hidden, L.mlp_norm, hidden, h_.data() + t * hidden);
      }
      {
        Section section(profiler_, "mlp_up",
                        profiling ? traffic::linear(L.gate_proj, n) + traffic::linear(L.up_proj, n) +
                                        traffic::rows(inter, n, 1)
                                  : 0.0);
        matmul(L.gate_proj, h_.data(), n, gate_.data(), pool_);
        matmul(L.up_proj, h_.data(), n, up_.data(), pool_);
        if (c_.activation == Activation::GeluTanh) {
          for (size_t i = 0; i < sn * inter; ++i) gate_[i] = gelu_tanh(gate_[i]) * up_[i];
        } else {
          for (size_t i = 0; i < sn * inter; ++i) gate_[i] = silu(gate_[i]) * up_[i];
        }
      }
      {
        Section section(profiler_, "down_proj", profiling ? traffic::linear(L.down_proj, n) : 0.0);
        matmul(L.down_proj, gate_.data(), n, r_.data(), pool_);
      }
      {
        Section section(profiler_, "residual", residual_bytes);
        add_residual(n, L.mlp_post_norm);
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

  // Attention for the n tokens of this pass (queries in q_, keys and values
  // through key_ptr/value_ptr) into attn_.
  void attend(int l, int start, int n, bool sliding, int group) {
    const int H = c_.num_heads, D = c_.head_dim;
    pool_.parallel_for(static_cast<size_t>(n) * H, [&](size_t begin, size_t end) {
      std::vector<float> scores;
      for (size_t item = begin; item < end; ++item) {
        const int t = static_cast<int>(item / H);
        const int hh = static_cast<int>(item % H);
        const int kv = hh / group;
        const int pos = start + t;
        const int lo = sliding ? std::max(0, pos - c_.sliding_window + 1) : 0;
        const float* qv = q_.data() + (static_cast<size_t>(t) * H + hh) * D;
        scores.resize(static_cast<size_t>(pos - lo + 1));
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = lo; j <= pos; ++j) {
          const float* kvec = key_ptr(l, kv, j, start, k_.data());
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += qv[d] * kvec[d];
          dot *= c_.attn_scale;
          if (c_.attn_logit_softcapping > 0.0f) {
            dot = std::tanh(dot / c_.attn_logit_softcapping) * c_.attn_logit_softcapping;
          }
          scores[static_cast<size_t>(j - lo)] = dot;
          max_score = std::max(max_score, dot);
        }
        float denom = 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        float* out = attn_.data() + (static_cast<size_t>(t) * H + hh) * D;
        std::fill(out, out + D, 0.0f);
        for (int j = lo; j <= pos; ++j) {
          const float p = scores[static_cast<size_t>(j - lo)] / denom;
          const float* vvec = value_ptr(l, kv, j, start, v_.data());
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
  std::vector<std::vector<float>> kv_k_, kv_v_;
  std::vector<double> inv_freq_local_, inv_freq_global_;
  std::vector<float> x_, h_, r_, q_, k_, v_, attn_, gate_, up_, logits_;
};

}  // namespace

std::unique_ptr<Model> make_decoder_cpu(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options) {
  return std::make_unique<DecoderCpu>(std::move(weights), options);
}

}  // namespace lokahi
