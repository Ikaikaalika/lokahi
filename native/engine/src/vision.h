// Gemma 3's image path on the CPU: the processor's rescale and normalization,
// the SigLIP vision tower and the multimodal projector, as Transformers'
// Gemma3ImageProcessor, SiglipVisionModel and Gemma3MultiModalProjector
// compute them. This is the correctness path; an 896-pixel image costs about
// 5 TFLOP through the 27-layer tower of the published models.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "model.h"
#include "quant.h"
#include "safetensors.h"
#include "thread_pool.h"

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

  // config.json's vision fields; `preprocessor` is preprocessor_config.json
  // when the checkpoint has one.
  static VisionConfig from_json(const Json& config, const Json* preprocessor);
};

// Whether config.json describes a vision tower.
bool has_vision(const Json& config);

class VisionEncoder {
 public:
  // nullptr when the checkpoint has no vision tower.
  static std::unique_ptr<VisionEncoder> load(const std::string& directory, unsigned threads = 0);

  const VisionConfig& config() const { return c_; }

  // An RGB8 image [image_size, image_size, 3], already resized, to normalized
  // pixels [3, image_size, image_size].
  std::vector<float> preprocess(const uint8_t* rgb) const;

  // Normalized pixels to the image's features [tokens_per_image,
  // text_hidden_size].
  std::vector<float> encode(const float* pixels);

 private:
  struct Layer {
    std::vector<float> ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    Linear q_proj, k_proj, v_proj, out_proj, fc1, fc2;
    std::vector<float> q_bias, k_bias, v_bias, out_bias, fc1_bias, fc2_bias;
  };

  VisionEncoder(VisionConfig config, unsigned threads) : c_(config), pool_(threads) {}

  VisionConfig c_;
  ThreadPool pool_;
  std::shared_ptr<Checkpoint> checkpoint_;
  // Owned float32 copies: the patch convolution as [hidden, kH * kW * 3]
  // (MLX's layout), positions [patches, hidden] and the projection transposed
  // to [text_hidden, hidden]; the Linears below view them.
  std::vector<float> patch_weight_, positions_, projection_weight_;
  Linear patch_, projection_;
  std::vector<float> patch_bias_, post_weight_, post_bias_, soft_norm_;
  std::vector<Layer> layers_;
};

// Places images' features at the runs of the image token in `tokens`, a
// prompt whose first token sits at position `start`: image i fills the i-th
// run of tokens_per_image tokens, which becomes its bidirectional block.
PromptInjection place_images(const std::vector<int32_t>& tokens, int start, const VisionConfig& config,
                             const std::vector<std::vector<float>>& features);

}  // namespace lokahi
