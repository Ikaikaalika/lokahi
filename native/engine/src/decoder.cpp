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

bool is_gemma3(const Json& raw) {
  std::string type = text_config(raw).string_or("model_type", raw.string_or("model_type", ""));
  return type == "gemma3_text" || type == "gemma3";
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
      c.rope_linear_factor = static_cast<float>(scaling->at("factor").as_number());
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

}  // namespace

bool is_supported_decoder(const Json& raw) { return is_gemma3(raw); }

DecoderConfig DecoderConfig::from_json(const Json& raw) {
  LK_CHECK(is_supported_decoder(raw), "unsupported architecture '" + raw.string_or("model_type", "?") +
                                          "'; this build supports Gemma 3 text models");
  DecoderConfig c = gemma3_config(raw);
  LK_CHECK(static_cast<int>(c.layer_is_sliding.size()) == c.num_layers,
           c.architecture + ": layer_types length mismatch");
  LK_CHECK(c.num_heads > 0 && c.num_kv_heads > 0 && c.num_heads % c.num_kv_heads == 0,
           c.architecture + ": heads must be a multiple of kv heads");
  LK_CHECK(c.head_dim > 0 && c.head_dim % 2 == 0, c.architecture + ": head_dim must be even");
  return c;
}

std::vector<double> DecoderConfig::rope_inv_freq(bool sliding) const {
  const bool local = sliding && local_rope;
  const double base = local ? rope_local_theta : rope_theta;
  const double factor = local ? 1.0 : rope_linear_factor;
  std::vector<double> inv_freq(static_cast<size_t>(head_dim / 2));
  for (size_t i = 0; i < inv_freq.size(); ++i) {
    inv_freq[i] = 1.0 / std::pow(base, static_cast<double>(2 * i) / head_dim) / factor;
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

  w->embed = linear("model.embed_tokens");
  LK_CHECK(w->embed.rows == c.vocab_size && w->embed.cols == c.hidden_size, arch + ": embed_tokens shape mismatch");
  if (ck.contains(p + "lm_head.weight")) {
    w->lm_head = linear("lm_head");
  } else if (ck.contains("lm_head.weight")) {
    w->lm_head = bind_linear(ck, w->raw_config, "lm_head", "lm_head");
  } else {
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
      float* row = x_.data() + t * hidden;
      dequantize_row(w_->embed, tokens[t], row);
      if (c_.embed_scale != 1.0f) {
        for (int64_t i = 0; i < hidden; ++i) row[i] *= c_.embed_scale;
      }
    }

    const int start = position_;
    const int group = H / KV;
    for (int l = 0; l < c_.num_layers; ++l) {
      const DecoderLayer& L = w_->layers[static_cast<size_t>(l)];
      const bool sliding = c_.layer_is_sliding[static_cast<size_t>(l)];
      for (int t = 0; t < n; ++t) norm(x_.data() + t * hidden, L.attn_norm, hidden, h_.data() + t * hidden);
      matmul(L.q_proj, h_.data(), n, q_.data(), pool_);
      matmul(L.k_proj, h_.data(), n, k_.data(), pool_);
      matmul(L.v_proj, h_.data(), n, v_.data(), pool_);
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
      if (sliding) {
        const int first = std::max(0, n - c_.kv_capacity(l, max_context_));
        for (int t = first; t < n; ++t) {
          store_kv(l, start + t, k_.data() + static_cast<size_t>(t) * KV * D,
                   v_.data() + static_cast<size_t>(t) * KV * D);
        }
      }
      matmul(L.o_proj, attn_.data(), n, r_.data(), pool_);
      add_residual(n, L.attn_post_norm);
      for (int t = 0; t < n; ++t) norm(x_.data() + t * hidden, L.mlp_norm, hidden, h_.data() + t * hidden);
      matmul(L.gate_proj, h_.data(), n, gate_.data(), pool_);
      matmul(L.up_proj, h_.data(), n, up_.data(), pool_);
      if (c_.activation == Activation::GeluTanh) {
        for (size_t i = 0; i < sn * inter; ++i) gate_[i] = gelu_tanh(gate_[i]) * up_[i];
      } else {
        for (size_t i = 0; i < sn * inter; ++i) gate_[i] = silu(gate_[i]) * up_[i];
      }
      matmul(L.down_proj, gate_.data(), n, r_.data(), pool_);
      add_residual(n, L.mlp_post_norm);
    }
    position_ += n;
    if (!want_logits) return;
    // Only the last row's logits are needed: skip the vocabulary projection
    // for every other prompt token.
    std::vector<float> last(static_cast<size_t>(hidden));
    norm(x_.data() + static_cast<size_t>(n - 1) * hidden, w_->final_norm, hidden, last.data());
    matvec(w_->lm_head, last.data(), logits_.data(), pool_);
    if (c_.final_logit_softcapping > 0.0f) {
      const float cap = c_.final_logit_softcapping;
      for (float& v : logits_) v = std::tanh(v / cap) * cap;
    }
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
