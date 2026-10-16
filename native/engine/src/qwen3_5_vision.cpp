#include "qwen3_5_vision.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include "common.h"
#include "thread_pool.h"
#include "vision_kernels.h"

namespace lokahi {

namespace {

using vision::add_rows;
using vision::dense_view;
using vision::gelu_tanh;
using vision::layer_norm;
using vision::vector_of;

constexpr float kLayerNormEps = 1e-6f;

int positive(const Json& object, const char* key, int fallback) {
  const int value = static_cast<int>(object.int_or(key, fallback));
  LK_CHECK(value > 0, std::string("qwen3_5 vision: ") + key + " must be positive");
  return value;
}

}  // namespace

bool is_qwen3_5_vision(const Json& config) {
  const std::string type = config.string_or("model_type", "");
  const Json* vision = config.find("vision_config");
  return (type == "qwen3_5" || type == "qwen3_5_moe" || type == "qwen3_vl" || type == "qwen3_vl_moe") &&
         vision != nullptr && vision->is_object();
}

Qwen35VisionConfig Qwen35VisionConfig::from_json(const Json& config, const Json* preprocessor) {
  LK_CHECK(is_qwen3_5_vision(config), "config.json has no Qwen3.5 or Qwen3-VL vision_config");
  const Json& v = config.at("vision_config");
  LK_CHECK(v.string_or("hidden_act", "gelu_pytorch_tanh") == "gelu_pytorch_tanh",
           "qwen3_5 vision: unsupported activation " + v.string_or("hidden_act", ""));
  LK_CHECK(v.int_or("in_channels", 3) == 3, "qwen3_5 vision: images must have 3 channels");
  Qwen35VisionConfig c;
  c.depth = positive(v, "depth", 0);
  // Transformers takes the i-th deepstack merger for the i-th listed block
  // but emits features in block order; increasing indexes keep them aligned.
  if (const Json* deepstack = v.find("deepstack_visual_indexes"); deepstack != nullptr && !deepstack->is_null()) {
    LK_CHECK(deepstack->is_array(), "qwen3_5 vision: deepstack_visual_indexes must be a list");
    for (const Json& index : deepstack->as_array()) {
      const int block = static_cast<int>(index.as_int());
      LK_CHECK(block >= 0 && block < c.depth && (c.deepstack.empty() || block > c.deepstack.back()),
               "qwen3_5 vision: deepstack_visual_indexes must be increasing blocks of the tower");
      c.deepstack.push_back(block);
    }
  }
  const std::string type = config.string_or("model_type", "");
  LK_CHECK(c.deepstack.empty() || type == "qwen3_vl" || type == "qwen3_vl_moe",
           "qwen3_5 vision: deepstack features are Qwen3-VL's, not " + type + "'s");
  c.hidden_size = positive(v, "hidden_size", 0);
  c.intermediate_size = positive(v, "intermediate_size", 0);
  c.num_heads = positive(v, "num_heads", 0);
  c.patch_size = positive(v, "patch_size", 16);
  c.temporal_patch_size = positive(v, "temporal_patch_size", 2);
  c.merge = positive(v, "spatial_merge_size", 2);
  const int table = positive(v, "num_position_embeddings", 0);
  c.position_side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(table))));
  LK_CHECK(c.position_side * c.position_side == table, "qwen3_5 vision: num_position_embeddings must be a square");
  // 2-D RoPE splits half of each head between rows and columns.
  LK_CHECK(c.hidden_size % c.num_heads == 0 && c.head_dim() % 4 == 0,
           "qwen3_5 vision: the head size must be a multiple of 4");
  c.text_hidden_size = positive(v, "out_hidden_size", 0);
  const Json* text = config.find("text_config");
  LK_CHECK(text != nullptr && text->is_object() && text->at("hidden_size").as_int() == c.text_hidden_size,
           "qwen3_5 vision: out_hidden_size must be the text model's hidden_size");
  c.image_token_id = static_cast<int>(config.at("image_token_id").as_int());

  // Qwen2VLImageProcessorFast: min_pixels / max_pixels override the size's
  // shortest_edge / longest_edge; the mean and std default to CLIP's.
  LK_CHECK(preprocessor != nullptr, "qwen3_5 vision: the checkpoint has no preprocessor_config.json");
  const Json* size = preprocessor->find("size");
  const bool has_size = size != nullptr && size->is_object();
  const Json* min_pixels = preprocessor->find("min_pixels");
  const Json* max_pixels = preprocessor->find("max_pixels");
  auto bound = [&](const Json* direct, const char* edge) -> int64_t {
    if (direct != nullptr && !direct->is_null()) return direct->as_int();
    LK_CHECK(has_size && size->find(edge) != nullptr,
             std::string("qwen3_5 vision: preprocessor_config.json gives no ") + edge);
    return size->at(edge).as_int();
  };
  c.min_pixels = bound(min_pixels, "shortest_edge");
  c.max_pixels = bound(max_pixels, "longest_edge");
  LK_CHECK(c.min_pixels > 0 && c.min_pixels <= c.max_pixels, "qwen3_5 vision: bad pixel bounds");
  LK_CHECK(preprocessor->int_or("patch_size", c.patch_size) == c.patch_size &&
               preprocessor->int_or("temporal_patch_size", c.temporal_patch_size) == c.temporal_patch_size &&
               preprocessor->int_or("merge_size", c.merge) == c.merge,
           "qwen3_5 vision: the processor's patch geometry does not match vision_config");
  const Json* rescale = preprocessor->find("do_rescale");
  c.rescale = rescale == nullptr || !rescale->is_bool() || rescale->as_bool()
                  ? static_cast<float>(preprocessor->number_or("rescale_factor", 1.0 / 255.0))
                  : 1.0f;
  const float clip_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  const float clip_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  std::copy_n(clip_mean, 3, c.mean);
  std::copy_n(clip_std, 3, c.std);
  const Json* normalize = preprocessor->find("do_normalize");
  if (normalize != nullptr && normalize->is_bool() && !normalize->as_bool()) {
    std::fill_n(c.mean, 3, 0.0f);
    std::fill_n(c.std, 3, 1.0f);
  } else {
    for (const auto& [key, out] : {std::pair{"image_mean", c.mean}, std::pair{"image_std", c.std}}) {
      const Json* values = preprocessor->find(key);
      if (values == nullptr || values->is_null()) continue;
      LK_CHECK(values->is_array() && values->as_array().size() == 3,
               std::string("qwen3_5 vision: ") + key + " must have 3 values");
      for (int ch = 0; ch < 3; ++ch) out[ch] = static_cast<float>(values->as_array()[static_cast<size_t>(ch)].as_number());
    }
  }
  return c;
}

std::shared_ptr<Qwen35VisionWeights> Qwen35VisionWeights::load(const std::string& directory, const Json& config) {
  Json preprocessor;
  const std::string preprocessor_path = directory + "/preprocessor_config.json";
  const bool has_preprocessor = std::ifstream(preprocessor_path, std::ios::binary).good();
  if (has_preprocessor) preprocessor = Json::parse(read_text_file(preprocessor_path));
  auto w = std::make_shared<Qwen35VisionWeights>();
  w->config = Qwen35VisionConfig::from_json(config, has_preprocessor ? &preprocessor : nullptr);
  const Qwen35VisionConfig& c = w->config;
  w->checkpoint = std::make_shared<Checkpoint>(Checkpoint::open_directory(directory));
  const Checkpoint& ck = *w->checkpoint;
  // mlx-community's conversions name the tower vision_tower, Transformers' model.visual.
  const std::string t = ck.contains("vision_tower.pos_embed.weight") ? "vision_tower" : "model.visual";
  const int64_t hidden = c.hidden_size, inter = c.intermediate_size, P = c.patch_size, T = c.temporal_patch_size;
  const int64_t merged = hidden * c.merge * c.merge;
  auto linear = [&](const std::string& module, int64_t rows, int64_t cols, std::vector<float>* bias) {
    Linear out = bind_linear(ck, config, module, module);
    LK_CHECK(out.rows == rows && out.cols == cols, module + ": unexpected shape");
    *bias = vector_of(ck, module + ".bias", rows);
    return out;
  };

  // The Conv3d kernel, Transformers' [out, C, T, P, P] or MLX's channels-last
  // [out, T, P, P, C], summed over time into [out, C * P * P].
  const std::string conv_name = t + ".patch_embed.proj.weight";
  const TensorView& conv = ck.get(conv_name);
  const std::vector<int64_t> torch_shape = {hidden, 3, T, P, P}, mlx_shape = {hidden, T, P, P, 3};
  LK_CHECK(conv.shape == torch_shape || conv.shape == mlx_shape, conv_name + ": unexpected shape");
  LK_CHECK(torch_shape != mlx_shape, conv_name + ": the kernel's layout is ambiguous");
  const bool channels_last = conv.shape == mlx_shape;
  const std::vector<float> raw = vector_of(ck, conv_name, hidden * 3 * T * P * P);
  w->patch_weight.assign(static_cast<size_t>(hidden * 3 * P * P), 0.0f);
  for (int64_t o = 0; o < hidden; ++o) {
    for (int64_t ch = 0; ch < 3; ++ch) {
      for (int64_t y = 0; y < P; ++y) {
        for (int64_t x = 0; x < P; ++x) {
          double sum = 0.0;
          for (int64_t f = 0; f < T; ++f) {
            const int64_t at = channels_last ? (((o * T + f) * P + y) * P + x) * 3 + ch
                                             : (((o * 3 + ch) * T + f) * P + y) * P + x;
            sum += raw[static_cast<size_t>(at)];
          }
          w->patch_weight[static_cast<size_t>(((o * 3 + ch) * P + y) * P + x)] = static_cast<float>(sum);
        }
      }
    }
  }
  w->patch_bias = vector_of(ck, t + ".patch_embed.proj.bias", hidden);
  w->positions = vector_of(ck, t + ".pos_embed.weight", int64_t{c.position_side} * c.position_side * hidden);
  LK_CHECK(ck.get(t + ".pos_embed.weight").shape ==
               (std::vector<int64_t>{int64_t{c.position_side} * c.position_side, hidden}),
           t + ".pos_embed.weight: unexpected shape");

  for (int l = 0; l < c.depth; ++l) {
    const std::string base = t + ".blocks." + std::to_string(l);
    Qwen35VisionWeights::Layer L;
    L.norm1_weight = vector_of(ck, base + ".norm1.weight", hidden);
    L.norm1_bias = vector_of(ck, base + ".norm1.bias", hidden);
    L.norm2_weight = vector_of(ck, base + ".norm2.weight", hidden);
    L.norm2_bias = vector_of(ck, base + ".norm2.bias", hidden);
    L.qkv = linear(base + ".attn.qkv", 3 * hidden, hidden, &L.qkv_bias);
    L.proj = linear(base + ".attn.proj", hidden, hidden, &L.proj_bias);
    L.fc1 = linear(base + ".mlp.linear_fc1", inter, hidden, &L.fc1_bias);
    L.fc2 = linear(base + ".mlp.linear_fc2", hidden, inter, &L.fc2_bias);
    w->layers.push_back(std::move(L));
  }
  LK_CHECK(!ck.contains(t + ".blocks." + std::to_string(c.depth) + ".norm1.weight"),
           "qwen3_5 vision: the checkpoint has more blocks than depth");
  auto merger = [&](const std::string& name, int64_t norm_size) {
    Qwen35VisionWeights::Merger m;
    m.norm_weight = vector_of(ck, name + ".norm.weight", norm_size);
    m.norm_bias = vector_of(ck, name + ".norm.bias", norm_size);
    m.fc1 = linear(name + ".linear_fc1", merged, merged, &m.fc1_bias);
    m.fc2 = linear(name + ".linear_fc2", c.text_hidden_size, merged, &m.fc2_bias);
    return m;
  };
  w->merger = merger(t + ".merger", hidden);
  for (size_t i = 0; i < c.deepstack.size(); ++i) {
    w->deepstack.push_back(merger(t + ".deepstack_merger_list." + std::to_string(i), merged));
  }
  LK_CHECK(!ck.contains(t + ".deepstack_merger_list." + std::to_string(c.deepstack.size()) + ".norm.weight"),
           "qwen3_5 vision: the checkpoint has more deepstack mergers than deepstack_visual_indexes");
  return w;
}

std::pair<int, int> Qwen35VisionEncoder::target_size(int width, int height) const {
  // smart_resize; Python's round() rounds half to even, as nearbyint does.
  const Qwen35VisionConfig& c = config();
  LK_CHECK(width > 0 && height > 0, "qwen3_5 vision: empty image");
  LK_CHECK(static_cast<double>(std::max(width, height)) / std::min(width, height) <= 200.0,
           "qwen3_5 vision: the image's aspect ratio exceeds 200");
  const double factor = c.factor(), h = height, w = width;
  double h_bar = std::nearbyint(h / factor) * factor;
  double w_bar = std::nearbyint(w / factor) * factor;
  if (h_bar * w_bar > static_cast<double>(c.max_pixels)) {
    const double beta = std::sqrt(h * w / static_cast<double>(c.max_pixels));
    h_bar = std::max(factor, std::floor(h / beta / factor) * factor);
    w_bar = std::max(factor, std::floor(w / beta / factor) * factor);
  } else if (h_bar * w_bar < static_cast<double>(c.min_pixels)) {
    const double beta = std::sqrt(static_cast<double>(c.min_pixels) / (h * w));
    h_bar = std::ceil(h * beta / factor) * factor;
    w_bar = std::ceil(w * beta / factor) * factor;
  }
  LK_CHECK(h_bar <= std::numeric_limits<int>::max() && w_bar <= std::numeric_limits<int>::max(),
           "qwen3_5 vision: the resized image is too large");
  return {static_cast<int>(w_bar), static_cast<int>(h_bar)};
}

int Qwen35VisionEncoder::tokens_for(int width, int height) const {
  const auto [rows, cols] = token_grid(width, height);
  return rows * cols;
}

std::pair<int, int> Qwen35VisionEncoder::token_grid(int width, int height) const {
  const Qwen35VisionConfig& c = config();
  const int m = c.factor();
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  LK_CHECK(width > 0 && height > 0 && width % m == 0 && height % m == 0,
           "a " + size + " image is not an accepted size: sides must be multiples of " + std::to_string(m) +
               " (resize to target_size)");
  const int64_t area = int64_t{width} * height;
  LK_CHECK(area >= c.min_pixels && area <= c.max_pixels,
           "a " + size + " image is outside the processor's " + std::to_string(c.min_pixels) + " to " +
               std::to_string(c.max_pixels) + " pixels (resize to target_size)");
  return {height / m, width / m};
}

std::pair<int, int> Qwen35VisionEncoder::patch_position(int index, int cols) const {
  const int m = config().merge, square = m * m;
  const int block = index / square, within = index % square, blocks_per_row = cols / m;
  return {block / blocks_per_row * m + within / m, block % blocks_per_row * m + within % m};
}

std::vector<float> Qwen35VisionEncoder::patches(const Image& image) const {
  const Qwen35VisionConfig& c = config();
  const int p = c.patch_size, rows = image.height / p, cols = image.width / p, width = 3 * p * p;
  std::vector<float> out(static_cast<size_t>(rows) * cols * width);
  for (int index = 0; index < rows * cols; ++index) {
    const auto [gy, gx] = patch_position(index, cols);
    float* row = out.data() + static_cast<size_t>(index) * width;
    for (int ch = 0; ch < 3; ++ch) {
      for (int ky = 0; ky < p; ++ky) {
        for (int kx = 0; kx < p; ++kx) {
          const uint8_t value = image.rgb[(static_cast<size_t>(gy * p + ky) * image.width + gx * p + kx) * 3 + ch];
          row[(ch * p + ky) * p + kx] = (static_cast<float>(value) * c.rescale - c.mean[ch]) / c.std[ch];
        }
      }
    }
  }
  return out;
}

void Qwen35VisionEncoder::add_position(float* x, int index, int rows, int cols) const {
  const Qwen35VisionConfig& c = config();
  const int side = c.position_side;
  const int64_t hidden = c.hidden_size;
  const auto [r, col] = patch_position(index, cols);
  auto sample = [&](int i, int count) { return count > 1 ? static_cast<double>(i) * (side - 1) / (count - 1) : 0.0; };
  const double h = sample(r, rows), w = sample(col, cols);
  const int h0 = static_cast<int>(h), w0 = static_cast<int>(w);
  const int h1 = std::min(h0 + 1, side - 1), w1 = std::min(w0 + 1, side - 1);
  const double dh = h - h0, dw = w - w0;
  const std::pair<int, double> corners[] = {{h0 * side + w0, (1 - dh) * (1 - dw)},
                                            {h0 * side + w1, (1 - dh) * dw},
                                            {h1 * side + w0, dh * (1 - dw)},
                                            {h1 * side + w1, dh * dw}};
  for (int64_t i = 0; i < hidden; ++i) {
    double sum = 0.0;
    for (const auto& [row, weight] : corners) sum += weight * weights_->positions[static_cast<size_t>(row * hidden + i)];
    x[i] += static_cast<float>(sum);
  }
}

void Qwen35VisionEncoder::rope_tables(int rows, int cols, float* cos_table, float* sin_table) const {
  const int64_t D = config().head_dim(), quarter = D / 4, half = D / 2;
  std::vector<double> inv_freq(static_cast<size_t>(quarter));
  for (int64_t i = 0; i < quarter; ++i) inv_freq[static_cast<size_t>(i)] = 1.0 / std::pow(10000.0, 2.0 * i / half);
  for (int64_t t = 0; t < int64_t{rows} * cols; ++t) {
    const auto [pr, pc] = patch_position(static_cast<int>(t), cols);
    for (int64_t i = 0; i < half; ++i) {
      const double angle = (i < quarter ? pr : pc) * inv_freq[static_cast<size_t>(i % quarter)];
      cos_table[t * half + i] = static_cast<float>(std::cos(angle));
      sin_table[t * half + i] = static_cast<float>(std::sin(angle));
    }
  }
}

namespace {

class Qwen35VisionCpu final : public Qwen35VisionEncoder {
 public:
  Qwen35VisionCpu(std::shared_ptr<const Qwen35VisionWeights> weights, unsigned threads)
      : Qwen35VisionEncoder(std::move(weights)), pool_(threads) {
    const Qwen35VisionConfig& c = config();
    patch_ = dense_view("patch_embed", weights_->patch_weight, c.hidden_size, int64_t{3} * c.patch_size * c.patch_size);
  }

  const char* backend() const override { return "cpu"; }
  ImageFeatures encode_features(const Image& image) override;

 private:
  // A merger over the patch rows x [N, hidden], into the text width.
  std::vector<float> merge(const Qwen35VisionWeights::Merger& m, const std::vector<float>& x, int64_t N,
                           bool per_patch);

  ThreadPool pool_;
  Linear patch_;
};

std::vector<float> Qwen35VisionCpu::merge(const Qwen35VisionWeights::Merger& m, const std::vector<float>& x,
                                          int64_t N, bool per_patch) {
  // Each merge x merge square's patches are consecutive, so the rows read as
  // [tokens, merge^2 * hidden]; the norm covers a patch or the whole square.
  const Qwen35VisionConfig& c = config();
  const int64_t hidden = c.hidden_size, tokens = N / (int64_t{c.merge} * c.merge), merged = hidden * c.merge * c.merge;
  const int64_t width = per_patch ? hidden : merged;
  std::vector<float> h(x.size()), f(static_cast<size_t>(tokens * merged));
  for (int64_t row = 0; row < N * hidden / width; ++row) {
    layer_norm(x.data() + row * width, m.norm_weight, m.norm_bias, width, kLayerNormEps, h.data() + row * width);
  }
  matmul(m.fc1, h.data(), tokens, f.data(), pool_);
  add_rows(f.data(), m.fc1_bias, tokens);
  for (float& value : f) value = static_cast<float>(0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0))));
  std::vector<float> features(static_cast<size_t>(tokens) * c.text_hidden_size);
  matmul(m.fc2, f.data(), tokens, features.data(), pool_);
  add_rows(features.data(), m.fc2_bias, tokens);
  return features;
}

ImageFeatures Qwen35VisionCpu::encode_features(const Image& image) {
  const Qwen35VisionConfig& c = config();
  const Qwen35VisionWeights& w = *weights_;
  const auto [token_rows, token_cols] = token_grid(image.width, image.height);
  const int rows = token_rows * c.merge, cols = token_cols * c.merge;
  const int64_t N = int64_t{rows} * cols, hidden = c.hidden_size, inter = c.intermediate_size;
  const int64_t heads = c.num_heads, D = c.head_dim();
  const size_t sn = static_cast<size_t>(N);

  const std::vector<float> patch_rows = patches(image);
  std::vector<float> x(sn * hidden), h(sn * hidden), qkv(sn * 3 * hidden), attn(sn * hidden), r(sn * hidden),
      f(sn * inter);
  matmul(patch_, patch_rows.data(), N, x.data(), pool_);
  add_rows(x.data(), w.patch_bias, N);
  for (int64_t t = 0; t < N; ++t) add_position(x.data() + t * hidden, static_cast<int>(t), rows, cols);

  const int64_t half = D / 2;
  std::vector<float> cos_table(sn * half), sin_table(sn * half);
  rope_tables(rows, cols, cos_table.data(), sin_table.data());
  auto rope = [&](float* v, int64_t t) {
    const float* cosine = cos_table.data() + t * half;
    const float* sine = sin_table.data() + t * half;
    for (int64_t i = 0; i < half; ++i) {
      const float a = v[i], b = v[i + half];
      v[i] = a * cosine[i] - b * sine[i];
      v[i + half] = b * cosine[i] + a * sine[i];
    }
  };

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  ImageFeatures out;
  for (size_t block = 0; block < w.layers.size(); ++block) {
    const Qwen35VisionWeights::Layer& L = w.layers[block];
    for (int64_t t = 0; t < N; ++t) {
      layer_norm(x.data() + t * hidden, L.norm1_weight, L.norm1_bias, hidden, kLayerNormEps, h.data() + t * hidden);
    }
    // qkv rows are [q | k | v], each heads x D.
    matmul(L.qkv, h.data(), N, qkv.data(), pool_);
    add_rows(qkv.data(), L.qkv_bias, N);
    for (int64_t t = 0; t < N; ++t) {
      for (int64_t part = 0; part < 2; ++part) {
        for (int64_t hh = 0; hh < heads; ++hh) rope(qkv.data() + (t * 3 + part) * hidden + hh * D, t);
      }
    }
    // Bidirectional attention: every patch sees every patch of its image.
    pool_.parallel_for(static_cast<size_t>(N * heads), [&](size_t begin, size_t end) {
      std::vector<float> scores(sn);
      for (size_t item = begin; item < end; ++item) {
        const int64_t t = static_cast<int64_t>(item) / heads, hh = static_cast<int64_t>(item) % heads;
        const float* qv = qkv.data() + t * 3 * hidden + hh * D;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < N; ++j) {
          const float* kv = qkv.data() + (j * 3 + 1) * hidden + hh * D;
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
          const float* vv = qkv.data() + (j * 3 + 2) * hidden + hh * D;
          for (int64_t d = 0; d < D; ++d) out[d] += p * vv[d];
        }
      }
    });
    matmul(L.proj, attn.data(), N, r.data(), pool_);
    add_rows(r.data(), L.proj_bias, N);
    for (size_t i = 0; i < x.size(); ++i) x[i] += r[i];
    for (int64_t t = 0; t < N; ++t) {
      layer_norm(x.data() + t * hidden, L.norm2_weight, L.norm2_bias, hidden, kLayerNormEps, h.data() + t * hidden);
    }
    matmul(L.fc1, h.data(), N, f.data(), pool_);
    add_rows(f.data(), L.fc1_bias, N);
    for (float& value : f) value = gelu_tanh(value);
    matmul(L.fc2, f.data(), N, r.data(), pool_);
    add_rows(r.data(), L.fc2_bias, N);
    for (size_t i = 0; i < x.size(); ++i) x[i] += r[i];
    const auto deep = std::find(c.deepstack.begin(), c.deepstack.end(), static_cast<int>(block));
    if (deep != c.deepstack.end()) {
      out.deepstack.push_back(merge(w.deepstack[static_cast<size_t>(deep - c.deepstack.begin())], x, N, false));
    }
  }
  out.rows = merge(w.merger, x, N, true);
  return out;
}

}  // namespace

std::unique_ptr<VisionEncoder> make_qwen3_5_vision_cpu(std::shared_ptr<const Qwen35VisionWeights> weights,
                                                       unsigned threads) {
  return std::make_unique<Qwen35VisionCpu>(std::move(weights), threads);
}

}  // namespace lokahi
