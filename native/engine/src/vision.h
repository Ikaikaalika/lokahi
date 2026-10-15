// Gemma 3's image path: the processor's rescale and normalization, the SigLIP
// vision tower and the multimodal projector, as Transformers'
// Gemma3ImageProcessor, SiglipVisionModel and Gemma3MultiModalProjector
// compute them. The CPU encoder is the correctness path (an 896-pixel image
// costs about 5 TFLOP, run by its scalar matmul); the Metal encoder
// (metal/metal_vision.h) runs the same math on the GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "model.h"
#include "quant.h"
#include "safetensors.h"

namespace lokahi {

struct VisionConfig {
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
  static VisionConfig from_json(const Json& config, const Json* preprocessor);
};

// Whether config.json describes a vision tower.
bool has_vision(const Json& config);

// The tower's and projector's weights, bound to the mapped checkpoint.
struct VisionWeights {
  struct Layer {
    std::vector<float> ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    Linear q_proj, k_proj, v_proj, out_proj, fc1, fc2;  // views into the checkpoint
    std::vector<float> q_bias, k_bias, v_bias, out_bias, fc1_bias, fc2_bias;
  };

  VisionConfig config;
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

  // nullptr when the checkpoint has no vision tower.
  static std::shared_ptr<VisionWeights> load(const std::string& directory);
};

class VisionEncoder {
 public:
  virtual ~VisionEncoder() = default;

  // A backend ("cpu", "metal" or "auto") for the checkpoint in `directory`;
  // nullptr when it has no vision tower.
  static std::unique_ptr<VisionEncoder> load(const std::string& directory, const std::string& backend = "auto",
                                             unsigned threads = 0);

  virtual const char* backend() const = 0;
  const VisionConfig& config() const { return weights_->config; }

  // An RGB8 image [image_size, image_size, 3], already resized, to normalized
  // pixels [3, image_size, image_size].
  std::vector<float> preprocess(const uint8_t* rgb) const;

  // Normalized pixels to the image's features [tokens_per_image,
  // text_hidden_size].
  virtual std::vector<float> encode(const float* pixels) = 0;

 protected:
  explicit VisionEncoder(std::shared_ptr<const VisionWeights> weights) : weights_(std::move(weights)) {}

  // Patches [patches, kH * kW * 3], each flattened as (row, column, channel)
  // to match the convolution.
  std::vector<float> patches(const float* pixels) const;

  std::shared_ptr<const VisionWeights> weights_;
};

std::unique_ptr<VisionEncoder> make_vision_cpu(std::shared_ptr<const VisionWeights> weights, unsigned threads);

// Places images' features at the runs of the image token in `tokens`, a
// prompt whose first token sits at position `start`: image i fills the i-th
// run of tokens_per_image tokens, which becomes its bidirectional block.
PromptInjection place_images(const std::vector<int32_t>& tokens, int start, const VisionConfig& config,
                             const std::vector<std::vector<float>>& features);

}  // namespace lokahi
