// Gemma 4's image path, as Transformers' Gemma4ImageProcessor,
// Gemma4VisionModel and Gemma4MultimodalEmbedder compute it: images at any
// size whose sides are multiples of patch_size * pooling_kernel_size, patches
// mapped to 2 * (p - 0.5) and projected, learned x and y position tables,
// sandwich-normed encoder layers with per-head Q/K norms, a scale-free V norm,
// axial 2-D RoPE and attention scale 1, a GELU-gated MLP, k x k average
// pooling times sqrt(hidden), optional standardization, and the embedder's
// scale-free RMSNorm plus projection into the text width.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "quant.h"
#include "safetensors.h"
#include "vision.h"

namespace lokahi {

struct Gemma4VisionConfig {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int head_dim = 0;
  int patch_size = 0;
  int pooling = 0;           // pooling_kernel_size
  int max_soft_tokens = 0;   // the processor's budget per image
  int position_table = 0;    // position_embedding_size
  double rope_theta = 100.0;
  float rms_norm_eps = 1e-6f;
  bool standardize = false;
  int image_token_id = -1;
  int text_hidden_size = 0;

  int side_multiple() const { return patch_size * pooling; }
  int max_patches() const { return max_soft_tokens * pooling * pooling; }

  // config.json plus, when given, processor_config.json.
  static Gemma4VisionConfig from_json(const Json& config, const Json* processor);
};

// Whether config.json's vision tower is Gemma 4's.
bool is_gemma4_vision(const Json& config);

struct Gemma4VisionWeights {
  struct Layer {
    std::vector<float> input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    std::vector<float> q_norm, k_norm;  // [head_dim]
    Linear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;  // views into the checkpoint
  };

  Gemma4VisionConfig config;
  std::shared_ptr<Checkpoint> checkpoint;
  Linear input_proj;                         // [hidden, 3 * patch^2]
  const TensorView* positions = nullptr;     // [2, position_table, hidden]: x, then y
  std::vector<Layer> layers;
  std::vector<float> std_bias, std_scale;    // with standardize
  Linear projection;                         // the embedder [text_hidden, hidden]

  static std::shared_ptr<Gemma4VisionWeights> load(const std::string& directory, const Json& config);
  // Row `index` of position table `axis` (0: x, 1: y) as float32.
  void position_row(int axis, int index, float* out) const;
};

// The geometry the CPU and Metal encoders share.
class Gemma4VisionEncoder : public VisionEncoder {
 public:
  int image_token_id() const override { return config().image_token_id; }
  int text_hidden_size() const override { return config().text_hidden_size; }
  std::pair<int, int> target_size(int width, int height) const override;
  int tokens_for(int width, int height) const override;

  const Gemma4VisionConfig& config() const { return weights_->config; }

 protected:
  explicit Gemma4VisionEncoder(std::shared_ptr<const Gemma4VisionWeights> weights) : weights_(std::move(weights)) {}

  // Patches [N, 3 * patch^2] as 2 * (p / 255 - 0.5), row-major over the
  // patch grid, each flattened as (row, column, channel).
  std::vector<float> patches(const Image& image) const;

  std::shared_ptr<const Gemma4VisionWeights> weights_;
};

std::unique_ptr<VisionEncoder> make_gemma4_vision_cpu(std::shared_ptr<const Gemma4VisionWeights> weights,
                                                      unsigned threads);

}  // namespace lokahi
