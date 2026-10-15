// Gemma 3's image path: the processor's rescale and normalization, the SigLIP
// vision tower and the multimodal projector, as Transformers'
// Gemma3ImageProcessor, SiglipVisionModel and Gemma3MultiModalProjector
// compute them. The CPU encoder is the correctness path (an 896-pixel image
// costs about 5 TFLOP, run by its scalar matmul); the Metal encoder
// (metal/metal_vision.h) runs the same math on the GPU.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "quant.h"
#include "safetensors.h"
#include "vision.h"

namespace lokahi {

struct SiglipConfig {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int image_size = 0;  // pixels per side, after resizing
  int patch_size = 0;
  float layer_norm_eps = 1e-6f;
  int tokens_per_image = 0;  // mm_tokens_per_image
  int image_token_id = -1;   // image_token_index
  int text_hidden_size = 0;
  // The processor's per-channel x * rescale, then (x - mean) / std.
  float rescale = 1.0f / 255.0f;
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float std[3] = {0.5f, 0.5f, 0.5f};

  int patches_per_side() const { return image_size / patch_size; }
  int patches() const { return patches_per_side() * patches_per_side(); }
  int tokens_per_side() const;

  // config.json's vision fields; `preprocessor` is preprocessor_config.json
  // when the checkpoint has one.
  static SiglipConfig from_json(const Json& config, const Json* preprocessor);
};

// Whether config.json's vision tower is Gemma 3's SigLIP.
bool is_siglip(const Json& config);

// The tower's and projector's weights, bound to the mapped checkpoint.
struct SiglipWeights {
  struct Layer {
    std::vector<float> ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    Linear q_proj, k_proj, v_proj, out_proj, fc1, fc2;  // views into the checkpoint
    std::vector<float> q_bias, k_bias, v_bias, out_bias, fc1_bias, fc2_bias;
  };

  SiglipConfig config;
  std::shared_ptr<Checkpoint> checkpoint;
  // Owned float32: the patch convolution as [hidden, kH * kW * 3] (MLX's
  // layout, whichever layout the checkpoint stores) and the positions
  // [patches, hidden].
  std::vector<float> patch_weight, patch_bias, positions;
  std::vector<Layer> layers;
  std::vector<float> post_weight, post_bias, soft_norm;
  // multi_modal_projector.mm_input_projection_weight [hidden, text_hidden]
  // (features times it), as stored.
  const TensorView* projection = nullptr;

  static std::shared_ptr<SiglipWeights> load(const std::string& directory, const Json& config);
};

// What the CPU and Metal SigLIP encoders share: the image geometry and the
// processor's preprocessing.
class SiglipEncoder : public VisionEncoder {
 public:
  int image_token_id() const override { return config().image_token_id; }
  int text_hidden_size() const override { return config().text_hidden_size; }
  int fixed_size() const override { return config().image_size; }
  std::pair<int, int> target_size(int, int) const override { return {config().image_size, config().image_size}; }
  int tokens_for(int width, int height) const override;
  std::vector<float> encode(const Image& image) override;

  const SiglipConfig& config() const { return weights_->config; }

  // An RGB8 image [image_size, image_size, 3] to normalized pixels
  // [3, image_size, image_size].
  std::vector<float> preprocess(const uint8_t* rgb) const;

 protected:
  explicit SiglipEncoder(std::shared_ptr<const SiglipWeights> weights) : weights_(std::move(weights)) {}

  // Normalized pixels to the image's features [tokens_per_image, text_hidden].
  virtual std::vector<float> encode_pixels(const float* pixels) = 0;

  // Patches [patches, kH * kW * 3], each flattened as (row, column, channel)
  // to match the convolution.
  std::vector<float> patches(const float* pixels) const;

  std::shared_ptr<const SiglipWeights> weights_;
};

std::unique_ptr<VisionEncoder> make_siglip_cpu(std::shared_ptr<const SiglipWeights> weights, unsigned threads);

}  // namespace lokahi
