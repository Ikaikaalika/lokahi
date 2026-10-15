#include "gemma4_vision.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include "common.h"
#include "half.h"
#include "thread_pool.h"
#include "vision_kernels.h"

namespace lokahi {

namespace {

using vision::dense_view;
using vision::gelu_tanh;
using vision::rms_norm;
using vision::vector_of;

int int_or(const Json& object, const char* key, int fallback) { return static_cast<int>(object.int_or(key, fallback)); }

}  // namespace

bool is_gemma4_vision(const Json& config) {
  const Json* vision = config.find("vision_config");
  return vision != nullptr && vision->is_object() && vision->string_or("model_type", "") == "gemma4_vision";
}

Gemma4VisionConfig Gemma4VisionConfig::from_json(const Json& config, const Json* processor) {
  LK_CHECK(is_gemma4_vision(config), "config.json has no Gemma 4 vision_config");
  const Json& v = config.at("vision_config");
  const Json* bool_clip = v.find("use_clipped_linears");
  LK_CHECK(bool_clip == nullptr || !bool_clip->is_bool() || !bool_clip->as_bool(),
           "gemma4 vision: clipped linears are not supported yet");
  LK_CHECK(v.string_or("hidden_activation", "gelu_pytorch_tanh") == "gelu_pytorch_tanh",
           "gemma4 vision: unsupported activation " + v.string_or("hidden_activation", ""));
  Gemma4VisionConfig c;
  c.hidden_size = static_cast<int>(v.at("hidden_size").as_int());
  c.intermediate_size = static_cast<int>(v.at("intermediate_size").as_int());
  c.num_layers = static_cast<int>(v.at("num_hidden_layers").as_int());
  c.num_heads = int_or(v, "num_attention_heads", 12);
  LK_CHECK(int_or(v, "num_key_value_heads", c.num_heads) == c.num_heads,
           "gemma4 vision: grouped-query attention in the tower is not supported yet");
  c.head_dim = int_or(v, "head_dim", 64);
  LK_CHECK(c.head_dim > 0 && c.head_dim % 4 == 0, "gemma4 vision: axial RoPE needs head_dim divisible by 4");
  c.patch_size = int_or(v, "patch_size", 16);
  c.pooling = int_or(v, "pooling_kernel_size", 3);
  c.position_table = int_or(v, "position_embedding_size", 10240);
  if (const Json* rope = v.find("rope_parameters"); rope != nullptr && rope->is_object()) {
    c.rope_theta = rope->number_or("rope_theta", 100.0);
  }
  c.rms_norm_eps = static_cast<float>(v.number_or("rms_norm_eps", 1e-6));
  const Json* standardize = v.find("standardize");
  c.standardize = standardize != nullptr && standardize->is_bool() && standardize->as_bool();
  c.max_soft_tokens = static_cast<int>(config.int_or("vision_soft_tokens_per_image", v.int_or("default_output_length", 280)));
  if (processor != nullptr) {
    if (const Json* images = processor->find("image_processor"); images != nullptr && images->is_object()) {
      c.max_soft_tokens = int_or(*images, "max_soft_tokens", c.max_soft_tokens);
      LK_CHECK(int_or(*images, "patch_size", c.patch_size) == c.patch_size &&
                   int_or(*images, "pooling_kernel_size", c.pooling) == c.pooling,
               "gemma4 vision: the processor's patch geometry does not match vision_config");
    }
  }
  c.image_token_id = static_cast<int>(config.at("image_token_id").as_int());
  const Json* text = config.find("text_config");
  c.text_hidden_size = static_cast<int>((text != nullptr ? *text : config).at("hidden_size").as_int());
  LK_CHECK(c.patch_size > 0 && c.pooling > 0 && c.max_soft_tokens > 0, "gemma4 vision: bad patch geometry");
  return c;
}

std::shared_ptr<Gemma4VisionWeights> Gemma4VisionWeights::load(const std::string& directory, const Json& config) {
  Json processor;
  const std::string processor_path = directory + "/processor_config.json";
  const bool has_processor = std::ifstream(processor_path, std::ios::binary).good();
  if (has_processor) processor = Json::parse(read_text_file(processor_path));
  auto w = std::make_shared<Gemma4VisionWeights>();
  w->config = Gemma4VisionConfig::from_json(config, has_processor ? &processor : nullptr);
  const Gemma4VisionConfig& c = w->config;
  w->checkpoint = std::make_shared<Checkpoint>(Checkpoint::open_directory(directory));
  const Checkpoint& ck = *w->checkpoint;
  const int64_t hidden = c.hidden_size, inter = c.intermediate_size, width = int64_t{c.num_heads} * c.head_dim;
  auto linear = [&](const std::string& module, int64_t rows, int64_t cols) {
    Linear out = bind_linear(ck, config, module, module);
    LK_CHECK(out.rows == rows && out.cols == cols, module + ": unexpected shape");
    return out;
  };
  const std::string t = "vision_tower";
  w->input_proj = linear(t + ".patch_embedder.input_proj", hidden, int64_t{3} * c.patch_size * c.patch_size);
  w->positions = &ck.get(t + ".patch_embedder.position_embedding_table");
  LK_CHECK(w->positions->shape == (std::vector<int64_t>{2, c.position_table, hidden}),
           "gemma4 vision: position_embedding_table has an unexpected shape");
  const DType position_dtype = w->positions->dtype;
  LK_CHECK(position_dtype == DType::F32 || position_dtype == DType::BF16 || position_dtype == DType::F16,
           "gemma4 vision: unsupported position table dtype");
  for (int l = 0; l < c.num_layers; ++l) {
    const std::string base = t + ".encoder.layers." + std::to_string(l);
    Gemma4VisionWeights::Layer L;
    L.input_norm = vector_of(ck, base + ".input_layernorm.weight", hidden);
    L.post_attention_norm = vector_of(ck, base + ".post_attention_layernorm.weight", hidden);
    L.pre_feedforward_norm = vector_of(ck, base + ".pre_feedforward_layernorm.weight", hidden);
    L.post_feedforward_norm = vector_of(ck, base + ".post_feedforward_layernorm.weight", hidden);
    L.q_norm = vector_of(ck, base + ".self_attn.q_norm.weight", c.head_dim);
    L.k_norm = vector_of(ck, base + ".self_attn.k_norm.weight", c.head_dim);
    L.q_proj = linear(base + ".self_attn.q_proj.linear", width, hidden);
    L.k_proj = linear(base + ".self_attn.k_proj.linear", width, hidden);
    L.v_proj = linear(base + ".self_attn.v_proj.linear", width, hidden);
    L.o_proj = linear(base + ".self_attn.o_proj.linear", hidden, width);
    L.gate_proj = linear(base + ".mlp.gate_proj.linear", inter, hidden);
    L.up_proj = linear(base + ".mlp.up_proj.linear", inter, hidden);
    L.down_proj = linear(base + ".mlp.down_proj.linear", hidden, inter);
    w->layers.push_back(std::move(L));
  }
  if (c.standardize) {
    w->std_bias = vector_of(ck, t + ".std_bias", hidden);
    w->std_scale = vector_of(ck, t + ".std_scale", hidden);
  }
  w->projection = linear("embed_vision.embedding_projection", c.text_hidden_size, hidden);
  return w;
}

void Gemma4VisionWeights::position_row(int axis, int index, float* out) const {
  const int64_t hidden = config.hidden_size;
  const int64_t base = (int64_t{axis} * config.position_table + index) * hidden;
  for (int64_t i = 0; i < hidden; ++i) {
    switch (positions->dtype) {
      case DType::F32: out[i] = positions->as<float>()[base + i]; break;
      case DType::BF16: out[i] = bf16_to_f32(positions->as<uint16_t>()[base + i]); break;
      default: out[i] = f16_to_f32(positions->as<uint16_t>()[base + i]); break;
    }
  }
}

std::pair<int, int> Gemma4VisionEncoder::target_size(int width, int height) const {
  // Gemma4ImageProcessor's get_aspect_ratio_preserving_size.
  const Gemma4VisionConfig& c = config();
  LK_CHECK(width > 0 && height > 0, "gemma4 vision: empty image");
  const double target_px = static_cast<double>(c.max_patches()) * c.patch_size * c.patch_size;
  const double factor = std::sqrt(target_px / (static_cast<double>(height) * width));
  const int m = c.side_multiple();
  int th = static_cast<int>(std::floor(factor * height / m)) * m;
  int tw = static_cast<int>(std::floor(factor * width / m)) * m;
  LK_CHECK(th != 0 || tw != 0, "gemma4 vision: the image would resize to 0 x 0");
  const int max_side = (c.max_patches() / (c.pooling * c.pooling)) * m;
  if (th == 0) {
    th = m;
    tw = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * m, max_side);
  } else if (tw == 0) {
    tw = m;
    th = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * m, max_side);
  }
  LK_CHECK(static_cast<double>(th) * tw <= target_px, "gemma4 vision: the resized image exceeds the patch budget");
  return {tw, th};
}

int Gemma4VisionEncoder::tokens_for(int width, int height) const {
  const Gemma4VisionConfig& c = config();
  const int m = c.side_multiple();
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  LK_CHECK(width > 0 && height > 0 && width % m == 0 && height % m == 0,
           "a " + size + " image is not an accepted size: sides must be multiples of " + std::to_string(m) +
               " (resize to target_size)");
  const int64_t patches = int64_t{width / c.patch_size} * (height / c.patch_size);
  LK_CHECK(patches <= c.max_patches(), "a " + size + " image exceeds the " + std::to_string(c.max_soft_tokens) +
                                           "-token budget (resize to target_size)");
  LK_CHECK(width / c.patch_size <= c.position_table && height / c.patch_size <= c.position_table,
           "a " + size + " image is wider than the position table");
  return (width / m) * (height / m);
}

std::vector<float> Gemma4VisionEncoder::patches(const Image& image) const {
  const int p = config().patch_size;
  const int rows = image.height / p, cols = image.width / p, width = 3 * p * p;
  std::vector<float> out(static_cast<size_t>(rows) * cols * width);
  for (int gy = 0; gy < rows; ++gy) {
    for (int gx = 0; gx < cols; ++gx) {
      float* row = out.data() + (static_cast<size_t>(gy) * cols + gx) * width;
      for (int ky = 0; ky < p; ++ky) {
        for (int kx = 0; kx < p; ++kx) {
          for (int ch = 0; ch < 3; ++ch) {
            const uint8_t value = image.rgb[(static_cast<size_t>(gy * p + ky) * image.width + gx * p + kx) * 3 + ch];
            row[(ky * p + kx) * 3 + ch] = 2.0f * (static_cast<float>(value) * (1.0f / 255.0f) - 0.5f);
          }
        }
      }
    }
  }
  return out;
}

namespace {

class Gemma4VisionCpu final : public Gemma4VisionEncoder {
 public:
  Gemma4VisionCpu(std::shared_ptr<const Gemma4VisionWeights> weights, unsigned threads)
      : Gemma4VisionEncoder(std::move(weights)), pool_(threads) {}

  const char* backend() const override { return "cpu"; }
  std::vector<float> encode(const Image& image) override;

 private:
  ThreadPool pool_;
};

// Axial RoPE on one head: the first half of the channels by x, the second by
// y, each rotate-half within its half.
void rope_2d(float* v, int dim, int x, int y, const std::vector<double>& inv_freq) {
  const int spatial = dim / 2, half = spatial / 2;
  for (int axis = 0; axis < 2; ++axis) {
    const double pos = axis == 0 ? x : y;
    float* part = v + axis * spatial;
    for (int i = 0; i < half; ++i) {
      const double angle = pos * inv_freq[static_cast<size_t>(i)];
      const float cs = static_cast<float>(std::cos(angle)), sn = static_cast<float>(std::sin(angle));
      const float a = part[i], b = part[i + half];
      part[i] = a * cs - b * sn;
      part[i + half] = b * cs + a * sn;
    }
  }
}

std::vector<float> Gemma4VisionCpu::encode(const Image& image) {
  const Gemma4VisionConfig& c = config();
  const Gemma4VisionWeights& w = *weights_;
  const int tokens = tokens_for(image.width, image.height);
  const int64_t cols = image.width / c.patch_size, rows = image.height / c.patch_size, N = rows * cols;
  const int64_t hidden = c.hidden_size, inter = c.intermediate_size, heads = c.num_heads, D = c.head_dim;
  const int64_t width = heads * D;
  const size_t sn = static_cast<size_t>(N);

  const std::vector<float> patch_rows = patches(image);
  std::vector<float> x(sn * hidden), h(sn * hidden), q(sn * width), k(sn * width), v(sn * width), attn(sn * width),
      r(sn * hidden), gate(sn * inter), up(sn * inter), row(static_cast<size_t>(hidden));
  matmul(w.input_proj, patch_rows.data(), N, x.data(), pool_);
  for (int64_t t = 0; t < N; ++t) {
    for (int axis = 0; axis < 2; ++axis) {
      w.position_row(axis, static_cast<int>(axis == 0 ? t % cols : t / cols), row.data());
      for (int64_t i = 0; i < hidden; ++i) x[static_cast<size_t>(t * hidden + i)] += row[static_cast<size_t>(i)];
    }
  }

  std::vector<double> inv_freq(static_cast<size_t>(D / 4));
  const double spatial = static_cast<double>(D / 2);
  for (size_t i = 0; i < inv_freq.size(); ++i) inv_freq[i] = 1.0 / std::pow(c.rope_theta, 2.0 * i / spatial);
  auto norm_rows = [&](float* data, int64_t count, int64_t size, const float* weight) {
    for (int64_t t = 0; t < count; ++t) rms_norm(data + t * size, weight, size, c.rms_norm_eps, data + t * size);
  };

  for (const Gemma4VisionWeights::Layer& L : w.layers) {
    for (int64_t t = 0; t < N; ++t) {
      rms_norm(x.data() + t * hidden, L.input_norm.data(), hidden, c.rms_norm_eps, h.data() + t * hidden);
    }
    matmul(L.q_proj, h.data(), N, q.data(), pool_);
    matmul(L.k_proj, h.data(), N, k.data(), pool_);
    matmul(L.v_proj, h.data(), N, v.data(), pool_);
    norm_rows(q.data(), N * heads, D, L.q_norm.data());
    norm_rows(k.data(), N * heads, D, L.k_norm.data());
    norm_rows(v.data(), N * heads, D, nullptr);
    for (int64_t t = 0; t < N; ++t) {
      const int px = static_cast<int>(t % cols), py = static_cast<int>(t / cols);
      for (int64_t hh = 0; hh < heads; ++hh) {
        rope_2d(q.data() + (t * heads + hh) * D, static_cast<int>(D), px, py, inv_freq);
        rope_2d(k.data() + (t * heads + hh) * D, static_cast<int>(D), px, py, inv_freq);
      }
    }
    // Bidirectional attention with scale 1.
    pool_.parallel_for(static_cast<size_t>(N * heads), [&](size_t begin, size_t end) {
      std::vector<float> scores(sn);
      for (size_t item = begin; item < end; ++item) {
        const int64_t t = static_cast<int64_t>(item) / heads, hh = static_cast<int64_t>(item) % heads;
        const float* qv = q.data() + (t * heads + hh) * D;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < N; ++j) {
          const float* kv = k.data() + (j * heads + hh) * D;
          float dot = 0.0f;
          for (int64_t d = 0; d < D; ++d) dot += qv[d] * kv[d];
          scores[static_cast<size_t>(j)] = dot;
          max_score = std::max(max_score, dot);
        }
        float denom = 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        float* out = attn.data() + (t * heads + hh) * D;
        std::fill(out, out + D, 0.0f);
        for (int64_t j = 0; j < N; ++j) {
          const float p = scores[static_cast<size_t>(j)] / denom;
          const float* vv = v.data() + (j * heads + hh) * D;
          for (int64_t d = 0; d < D; ++d) out[d] += p * vv[d];
        }
      }
    });
    matmul(L.o_proj, attn.data(), N, r.data(), pool_);
    for (int64_t t = 0; t < N; ++t) {
      float* rr = r.data() + t * hidden;
      rms_norm(rr, L.post_attention_norm.data(), hidden, c.rms_norm_eps, rr);
      for (int64_t i = 0; i < hidden; ++i) x[static_cast<size_t>(t * hidden + i)] += rr[i];
      rms_norm(x.data() + t * hidden, L.pre_feedforward_norm.data(), hidden, c.rms_norm_eps, h.data() + t * hidden);
    }
    matmul(L.gate_proj, h.data(), N, gate.data(), pool_);
    matmul(L.up_proj, h.data(), N, up.data(), pool_);
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = gelu_tanh(gate[i]) * up[i];
    matmul(L.down_proj, gate.data(), N, r.data(), pool_);
    for (int64_t t = 0; t < N; ++t) {
      float* rr = r.data() + t * hidden;
      rms_norm(rr, L.post_feedforward_norm.data(), hidden, c.rms_norm_eps, rr);
      for (int64_t i = 0; i < hidden; ++i) x[static_cast<size_t>(t * hidden + i)] += rr[i];
    }
  }

  // Average k x k squares of the patch grid (row-major over the pooled grid),
  // scale by sqrt(hidden), standardize, then the embedder.
  const int kp = c.pooling;
  const int64_t pooled_cols = cols / kp;
  const float root = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
  std::vector<float> pooled(static_cast<size_t>(tokens) * hidden);
  for (int token = 0; token < tokens; ++token) {
    const int64_t ty = token / pooled_cols, tx = token % pooled_cols;
    float* out = pooled.data() + static_cast<size_t>(token) * hidden;
    for (int64_t i = 0; i < hidden; ++i) {
      double sum = 0.0;
      for (int ky = 0; ky < kp; ++ky) {
        for (int kx = 0; kx < kp; ++kx) sum += x[static_cast<size_t>(((ty * kp + ky) * cols + tx * kp + kx) * hidden + i)];
      }
      float value = static_cast<float>(sum / (kp * kp)) * root;
      if (c.standardize) value = (value - w.std_bias[static_cast<size_t>(i)]) * w.std_scale[static_cast<size_t>(i)];
      out[i] = value;
    }
    rms_norm(out, nullptr, hidden, c.rms_norm_eps, out);
  }
  std::vector<float> features(static_cast<size_t>(tokens) * c.text_hidden_size);
  matmul(w.projection, pooled.data(), tokens, features.data(), pool_);
  return features;
}

}  // namespace

std::unique_ptr<VisionEncoder> make_gemma4_vision_cpu(std::shared_ptr<const Gemma4VisionWeights> weights,
                                                      unsigned threads) {
  return std::make_unique<Gemma4VisionCpu>(std::move(weights), threads);
}

}  // namespace lokahi
