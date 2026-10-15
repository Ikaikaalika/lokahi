#include "vision.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

#include "common.h"
#include "thread_pool.h"

#if defined(LOKAHI_WITH_METAL)
#include "metal/metal_decoder.h"
#include "metal/metal_vision.h"
#endif

namespace lokahi {

namespace {

const char* kTower = "vision_tower.vision_model";

int int_field(const Json& object, const char* key) { return static_cast<int>(object.at(key).as_int()); }

// Pre-LayerNorm: mean and variance in double, as the oracle computes them.
void layer_norm(const float* x, const std::vector<float>& weight, const std::vector<float>& bias, int64_t size,
                float eps, float* out) {
  double mean = 0.0;
  for (int64_t i = 0; i < size; ++i) mean += x[i];
  mean /= static_cast<double>(size);
  double var = 0.0;
  for (int64_t i = 0; i < size; ++i) var += (x[i] - mean) * (x[i] - mean);
  const double inv = 1.0 / std::sqrt(var / static_cast<double>(size) + eps);
  for (int64_t i = 0; i < size; ++i) {
    out[i] = static_cast<float>((x[i] - mean) * inv) * weight[static_cast<size_t>(i)] + bias[static_cast<size_t>(i)];
  }
}

float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2 / pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

void add_rows(float* x, const std::vector<float>& bias, int64_t n) {
  const size_t width = bias.size();
  for (int64_t t = 0; t < n; ++t) {
    for (size_t i = 0; i < width; ++i) x[static_cast<size_t>(t) * width + i] += bias[i];
  }
}

Linear dense_view(const std::string& name, const std::vector<float>& data, int64_t rows, int64_t cols) {
  Linear linear;
  linear.name = name;
  linear.rows = rows;
  linear.cols = cols;
  linear.dense = data.data();
  linear.dense_dtype = DType::F32;
  return linear;
}

std::vector<float> vector_of(const Checkpoint& checkpoint, const std::string& name, int64_t size) {
  std::vector<float> out(static_cast<size_t>(size));
  load_vector(checkpoint, name, size, out.data());
  return out;
}

}  // namespace

bool has_vision(const Json& config) {
  const Json* vision = config.find("vision_config");
  return vision != nullptr && vision->is_object();
}

VisionConfig VisionConfig::from_json(const Json& config, const Json* preprocessor) {
  LK_CHECK(has_vision(config), "config.json has no vision_config");
  LK_CHECK(config.string_or("model_type", "") == "gemma3",
           "vision: only Gemma 3's image path is supported yet, not '" + config.string_or("model_type", "") + "'");
  const Json& v = config.at("vision_config");
  LK_CHECK(v.string_or("model_type", "siglip_vision_model") == "siglip_vision_model",
           "vision: only SigLIP towers are supported");
  LK_CHECK(v.string_or("hidden_act", "gelu_pytorch_tanh") == "gelu_pytorch_tanh",
           "vision: unsupported activation " + v.string_or("hidden_act", ""));
  LK_CHECK(v.int_or("num_channels", 3) == 3, "vision: images must have 3 channels");
  VisionConfig c;
  c.hidden_size = int_field(v, "hidden_size");
  c.intermediate_size = int_field(v, "intermediate_size");
  c.num_layers = int_field(v, "num_hidden_layers");
  c.num_heads = int_field(v, "num_attention_heads");
  c.image_size = static_cast<int>(v.int_or("image_size", 224));
  c.patch_size = int_field(v, "patch_size");
  c.layer_norm_eps = static_cast<float>(v.number_or("layer_norm_eps", 1e-6));
  c.tokens_per_image = static_cast<int>(config.int_or("mm_tokens_per_image", 256));
  c.image_token_id = static_cast<int>(config.at("image_token_index").as_int());
  const Json* text = config.find("text_config");
  c.text_hidden_size = static_cast<int>((text != nullptr ? *text : config).at("hidden_size").as_int());
  LK_CHECK(c.hidden_size > 0 && c.num_heads > 0 && c.hidden_size % c.num_heads == 0,
           "vision: hidden_size must be a multiple of the head count");
  LK_CHECK(c.patch_size > 0 && c.image_size % c.patch_size == 0, "vision: image_size must be a multiple of patch_size");
  const int side = c.patches_per_side();
  const int tokens_side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(c.tokens_per_image))));
  LK_CHECK(tokens_side * tokens_side == c.tokens_per_image && tokens_side > 0 && side % tokens_side == 0,
           "vision: mm_tokens_per_image must be a square whose side divides the patch grid");
  if (preprocessor != nullptr) {
    if (const Json* size = preprocessor->find("size"); size != nullptr && size->is_object()) {
      LK_CHECK(size->int_or("height", c.image_size) == c.image_size && size->int_or("width", c.image_size) == c.image_size,
               "vision: preprocessor size does not match vision_config.image_size");
    }
    const Json* rescale = preprocessor->find("do_rescale");
    if (rescale == nullptr || !rescale->is_bool() || rescale->as_bool()) {
      c.rescale = static_cast<float>(preprocessor->number_or("rescale_factor", 1.0 / 255.0));
    } else {
      c.rescale = 1.0f;
    }
    const Json* normalize = preprocessor->find("do_normalize");
    if (normalize != nullptr && normalize->is_bool() && !normalize->as_bool()) {
      for (int ch = 0; ch < 3; ++ch) {
        c.mean[ch] = 0.0f;
        c.std[ch] = 1.0f;
      }
    } else {
      for (const auto& [key, out] : {std::pair{"image_mean", c.mean}, std::pair{"image_std", c.std}}) {
        const Json* values = preprocessor->find(key);
        if (values == nullptr || !values->is_array()) continue;
        LK_CHECK(values->as_array().size() == 3, std::string("vision: ") + key + " must have 3 values");
        for (int ch = 0; ch < 3; ++ch) out[ch] = static_cast<float>(values->as_array()[static_cast<size_t>(ch)].as_number());
      }
    }
  }
  return c;
}

int VisionConfig::tokens_per_side() const {
  return static_cast<int>(std::lround(std::sqrt(static_cast<double>(tokens_per_image))));
}

std::shared_ptr<VisionWeights> VisionWeights::load(const std::string& directory) {
  const Json config = Json::parse(read_text_file(directory + "/config.json"));
  if (!has_vision(config)) return nullptr;
  Json preprocessor;
  const std::string preprocessor_path = directory + "/preprocessor_config.json";
  const bool has_preprocessor = std::ifstream(preprocessor_path, std::ios::binary).good();
  if (has_preprocessor) preprocessor = Json::parse(read_text_file(preprocessor_path));
  auto w = std::make_shared<VisionWeights>();
  w->config = VisionConfig::from_json(config, has_preprocessor ? &preprocessor : nullptr);
  const VisionConfig& c = w->config;
  w->checkpoint = std::make_shared<Checkpoint>(Checkpoint::open_directory(directory));
  const Checkpoint& ck = *w->checkpoint;
  const std::string tower = kTower;
  const int64_t hidden = c.hidden_size, inter = c.intermediate_size, P = c.patch_size;
  auto linear = [&](const std::string& module, int64_t rows, int64_t cols) {
    Linear out = bind_linear(ck, config, module, module);
    LK_CHECK(out.rows == rows && out.cols == cols, module + ": unexpected shape");
    return out;
  };

  // The patch convolution, in MLX's [out, kH, kW, in] layout or
  // Transformers' [out, in, kH, kW].
  const std::string conv_name = tower + ".embeddings.patch_embedding.weight";
  const TensorView& conv = ck.get(conv_name);
  std::vector<float> raw = vector_of(ck, conv_name, hidden * P * P * 3);
  if (conv.shape == std::vector<int64_t>{hidden, P, P, 3}) {
    w->patch_weight = std::move(raw);
  } else {
    LK_CHECK(conv.shape == (std::vector<int64_t>{hidden, 3, P, P}), conv_name + ": unexpected shape");
    w->patch_weight.resize(raw.size());
    for (int64_t o = 0; o < hidden; ++o) {
      for (int64_t ch = 0; ch < 3; ++ch) {
        for (int64_t ky = 0; ky < P; ++ky) {
          for (int64_t kx = 0; kx < P; ++kx) {
            w->patch_weight[static_cast<size_t>(((o * P + ky) * P + kx) * 3 + ch)] =
                raw[static_cast<size_t>(((o * 3 + ch) * P + ky) * P + kx)];
          }
        }
      }
    }
  }
  w->patch_bias = vector_of(ck, tower + ".embeddings.patch_embedding.bias", hidden);
  w->positions = vector_of(ck, tower + ".embeddings.position_embedding.weight", c.patches() * hidden);

  for (int l = 0; l < c.num_layers; ++l) {
    const std::string base = tower + ".encoder.layers." + std::to_string(l);
    VisionWeights::Layer L;
    L.ln1_weight = vector_of(ck, base + ".layer_norm1.weight", hidden);
    L.ln1_bias = vector_of(ck, base + ".layer_norm1.bias", hidden);
    L.ln2_weight = vector_of(ck, base + ".layer_norm2.weight", hidden);
    L.ln2_bias = vector_of(ck, base + ".layer_norm2.bias", hidden);
    const std::pair<Linear*, std::vector<float>*> attention[] = {
        {&L.q_proj, &L.q_bias}, {&L.k_proj, &L.k_bias}, {&L.v_proj, &L.v_bias}, {&L.out_proj, &L.out_bias}};
    const char* names[] = {"q_proj", "k_proj", "v_proj", "out_proj"};
    for (int i = 0; i < 4; ++i) {
      const std::string module = base + ".self_attn." + names[i];
      *attention[i].first = linear(module, hidden, hidden);
      *attention[i].second = vector_of(ck, module + ".bias", hidden);
    }
    L.fc1 = linear(base + ".mlp.fc1", inter, hidden);
    L.fc1_bias = vector_of(ck, base + ".mlp.fc1.bias", inter);
    L.fc2 = linear(base + ".mlp.fc2", hidden, inter);
    L.fc2_bias = vector_of(ck, base + ".mlp.fc2.bias", hidden);
    w->layers.push_back(std::move(L));
  }
  w->post_weight = vector_of(ck, tower + ".post_layernorm.weight", hidden);
  w->post_bias = vector_of(ck, tower + ".post_layernorm.bias", hidden);
  w->soft_norm = vector_of(ck, "multi_modal_projector.mm_soft_emb_norm.weight", hidden);
  const std::string projection_name = "multi_modal_projector.mm_input_projection_weight";
  w->projection = &ck.get(projection_name);
  LK_CHECK(w->projection->shape == (std::vector<int64_t>{hidden, c.text_hidden_size}),
           projection_name + ": unexpected shape");
  const DType dtype = w->projection->dtype;
  LK_CHECK(dtype == DType::F32 || dtype == DType::BF16 || dtype == DType::F16,
           projection_name + ": unsupported dtype");
  return w;
}

std::unique_ptr<VisionEncoder> VisionEncoder::load(const std::string& directory, const std::string& requested,
                                                   unsigned threads) {
  auto weights = VisionWeights::load(directory);
  if (weights == nullptr) return nullptr;
  std::string backend = requested;
#if defined(LOKAHI_WITH_METAL)
  if (backend == "auto") backend = metal_vision_supports(*weights) && metal_available() ? "metal" : "cpu";
  if (backend == "metal") return make_vision_metal(std::move(weights));
#else
  if (backend == "auto") backend = "cpu";
  LK_CHECK(backend != "metal", "this build has no Metal backend");
#endif
  LK_CHECK(backend == "cpu", "unknown backend '" + backend + "'");
  return make_vision_cpu(std::move(weights), threads);
}

std::vector<float> VisionEncoder::preprocess(const uint8_t* rgb) const {
  const VisionConfig& c = config();
  const size_t S = static_cast<size_t>(c.image_size);
  std::vector<float> out(3 * S * S);
  for (size_t y = 0; y < S; ++y) {
    for (size_t x = 0; x < S; ++x) {
      for (size_t ch = 0; ch < 3; ++ch) {
        const float value = static_cast<float>(rgb[(y * S + x) * 3 + ch]) * c.rescale;
        out[(ch * S + y) * S + x] = (value - c.mean[ch]) / c.std[ch];
      }
    }
  }
  return out;
}

std::vector<float> VisionEncoder::patches(const float* pixels) const {
  const VisionConfig& c = config();
  const int64_t S = c.image_size, P = c.patch_size, side = c.patches_per_side(), width = P * P * 3;
  std::vector<float> out(static_cast<size_t>(side * side * width));
  for (int64_t gy = 0; gy < side; ++gy) {
    for (int64_t gx = 0; gx < side; ++gx) {
      float* row = out.data() + (gy * side + gx) * width;
      for (int64_t ky = 0; ky < P; ++ky) {
        for (int64_t kx = 0; kx < P; ++kx) {
          for (int64_t ch = 0; ch < 3; ++ch) row[(ky * P + kx) * 3 + ch] = pixels[(ch * S + gy * P + ky) * S + gx * P + kx];
        }
      }
    }
  }
  return out;
}

namespace {

class VisionCpu final : public VisionEncoder {
 public:
  VisionCpu(std::shared_ptr<const VisionWeights> weights, unsigned threads)
      : VisionEncoder(std::move(weights)), pool_(threads) {
    const VisionConfig& c = config();
    const int64_t hidden = c.hidden_size, text_hidden = c.text_hidden_size;
    patch_ = dense_view("patch_embedding", weights_->patch_weight, hidden, c.patch_size * c.patch_size * 3);
    // The projection is [hidden, text_hidden] (features times it); transposed
    // here so it runs as an ordinary [rows = text_hidden, cols = hidden] Linear.
    const std::vector<float> projection =
        vector_of(*weights_->checkpoint, "multi_modal_projector.mm_input_projection_weight", hidden * text_hidden);
    projection_weight_.resize(projection.size());
    for (int64_t i = 0; i < hidden; ++i) {
      for (int64_t j = 0; j < text_hidden; ++j) {
        projection_weight_[static_cast<size_t>(j * hidden + i)] = projection[static_cast<size_t>(i * text_hidden + j)];
      }
    }
    projection_ = dense_view("mm_input_projection_weight", projection_weight_, text_hidden, hidden);
  }

  const char* backend() const override { return "cpu"; }
  std::vector<float> encode(const float* pixels) override;

 private:
  ThreadPool pool_;
  std::vector<float> projection_weight_;
  Linear patch_, projection_;
};

std::vector<float> VisionCpu::encode(const float* pixels) {
  const VisionConfig& c = config();
  const VisionWeights& w = *weights_;
  const int64_t side = c.patches_per_side();
  const int64_t N = side * side, hidden = c.hidden_size, inter = c.intermediate_size;
  const int64_t heads = c.num_heads, D = hidden / heads;
  const size_t sn = static_cast<size_t>(N);

  const std::vector<float> patch_rows = patches(pixels);
  std::vector<float> x(sn * hidden), h(sn * hidden), q(sn * hidden), k(sn * hidden), v(sn * hidden),
      attn(sn * hidden), r(sn * hidden), f(sn * inter);
  matmul(patch_, patch_rows.data(), N, x.data(), pool_);
  add_rows(x.data(), w.patch_bias, N);
  for (size_t i = 0; i < x.size(); ++i) x[i] += w.positions[i];

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (const VisionWeights::Layer& L : w.layers) {
    for (int64_t t = 0; t < N; ++t) {
      layer_norm(x.data() + t * hidden, L.ln1_weight, L.ln1_bias, hidden, c.layer_norm_eps, h.data() + t * hidden);
    }
    matmul(L.q_proj, h.data(), N, q.data(), pool_);
    matmul(L.k_proj, h.data(), N, k.data(), pool_);
    matmul(L.v_proj, h.data(), N, v.data(), pool_);
    add_rows(q.data(), L.q_bias, N);
    add_rows(k.data(), L.k_bias, N);
    add_rows(v.data(), L.v_bias, N);
    // Bidirectional attention: every patch sees every patch.
    pool_.parallel_for(static_cast<size_t>(N * heads), [&](size_t begin, size_t end) {
      std::vector<float> scores(sn);
      for (size_t item = begin; item < end; ++item) {
        const int64_t t = static_cast<int64_t>(item) / heads, hh = static_cast<int64_t>(item) % heads;
        const float* qv = q.data() + t * hidden + hh * D;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < N; ++j) {
          const float* kv = k.data() + j * hidden + hh * D;
          float dot = 0.0f;
          for (int64_t d = 0; d < D; ++d) dot += qv[d] * kv[d];
          scores[static_cast<size_t>(j)] = dot * scale;
          max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
        }
        float denom = 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        float* out = attn.data() + t * hidden + hh * D;
        std::fill(out, out + D, 0.0f);
        for (int64_t j = 0; j < N; ++j) {
          const float p = scores[static_cast<size_t>(j)] / denom;
          const float* vv = v.data() + j * hidden + hh * D;
          for (int64_t d = 0; d < D; ++d) out[d] += p * vv[d];
        }
      }
    });
    matmul(L.out_proj, attn.data(), N, r.data(), pool_);
    add_rows(r.data(), L.out_bias, N);
    for (size_t i = 0; i < x.size(); ++i) x[i] += r[i];
    for (int64_t t = 0; t < N; ++t) {
      layer_norm(x.data() + t * hidden, L.ln2_weight, L.ln2_bias, hidden, c.layer_norm_eps, h.data() + t * hidden);
    }
    matmul(L.fc1, h.data(), N, f.data(), pool_);
    add_rows(f.data(), L.fc1_bias, N);
    for (float& value : f) value = gelu_tanh(value);
    matmul(L.fc2, f.data(), N, r.data(), pool_);
    add_rows(r.data(), L.fc2_bias, N);
    for (size_t i = 0; i < x.size(); ++i) x[i] += r[i];
  }
  for (int64_t t = 0; t < N; ++t) {
    layer_norm(x.data() + t * hidden, w.post_weight, w.post_bias, hidden, c.layer_norm_eps, h.data() + t * hidden);
  }

  // Average-pool the patch grid to tokens_per_image tokens (row-major), then
  // Gemma's RMSNorm (1 + weight) and the projection into the text width.
  const int64_t T = c.tokens_per_image;
  const int64_t tokens_side = c.tokens_per_side();
  const int64_t kernel = side / tokens_side;
  std::vector<float> pooled(static_cast<size_t>(T * hidden), 0.0f);
  const float inv_area = 1.0f / static_cast<float>(kernel * kernel);
  for (int64_t ty = 0; ty < tokens_side; ++ty) {
    for (int64_t tx = 0; tx < tokens_side; ++tx) {
      float* out = pooled.data() + (ty * tokens_side + tx) * hidden;
      for (int64_t i = 0; i < hidden; ++i) {
        double sum = 0.0;
        for (int64_t ky = 0; ky < kernel; ++ky) {
          for (int64_t kx = 0; kx < kernel; ++kx) sum += h[static_cast<size_t>(((ty * kernel + ky) * side + tx * kernel + kx) * hidden + i)];
        }
        out[i] = static_cast<float>(sum) * inv_area;
      }
      double squares = 0.0;
      for (int64_t i = 0; i < hidden; ++i) squares += static_cast<double>(out[i]) * out[i];
      const float inv = static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(hidden) + c.layer_norm_eps));
      for (int64_t i = 0; i < hidden; ++i) out[i] = out[i] * inv * (1.0f + w.soft_norm[static_cast<size_t>(i)]);
    }
  }
  std::vector<float> features(static_cast<size_t>(T * c.text_hidden_size));
  matmul(projection_, pooled.data(), T, features.data(), pool_);
  return features;
}

}  // namespace

std::unique_ptr<VisionEncoder> make_vision_cpu(std::shared_ptr<const VisionWeights> weights, unsigned threads) {
  return std::make_unique<VisionCpu>(std::move(weights), threads);
}

PromptInjection place_images(const std::vector<int32_t>& tokens, int start, const VisionConfig& config,
                             const std::vector<std::vector<float>>& features) {
  const int T = config.tokens_per_image;
  const size_t width = static_cast<size_t>(config.text_hidden_size);
  std::vector<int> runs;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == config.image_token_id) runs.push_back(static_cast<int>(i));
  }
  LK_CHECK(runs.size() == features.size() * static_cast<size_t>(T),
           "the prompt has " + std::to_string(runs.size()) + " image tokens for " + std::to_string(features.size()) +
               " image(s) of " + std::to_string(T));
  PromptInjection out;
  for (size_t n = 0; n < features.size(); ++n) {
    const int first = runs[n * static_cast<size_t>(T)];
    LK_CHECK(runs[(n + 1) * static_cast<size_t>(T) - 1] == first + T - 1, "each image's tokens must be contiguous");
    LK_CHECK(features[n].size() == static_cast<size_t>(T) * width, "image features have the wrong size");
    out.blocks.emplace_back(start + first, start + first + T);
    for (int j = 0; j < T; ++j) out.positions.push_back(start + first + j);
    out.rows.insert(out.rows.end(), features[n].begin(), features[n].end());
  }
  return out;
}

}  // namespace lokahi
