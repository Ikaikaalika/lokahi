// Qwen3.5's image path, as Transformers' Qwen2VLImageProcessorFast (which
// Qwen3VLProcessor uses) and Qwen3_5VisionModel (Qwen3-VL's tower without
// deepstack) compute it:
//
// * the processor resizes an image (the caller does, to target_size) so both
//   sides are multiples of patch_size * spatial_merge_size and its area lies
//   in [min_pixels, max_pixels], rescales and normalizes it, and repeats it
//   temporal_patch_size times along time;
// * a Conv3d with stride = kernel embeds each patch (here the kernel is
//   summed over time, since an image's frames are equal); learned position
//   embeddings over a square table are bilinearly interpolated to the patch
//   grid and added;
// * each block is x += proj(attn(norm1(x))), x += fc2(gelu_tanh(fc1(norm2(x))))
//   with LayerNorms (eps 1e-6), a biased qkv projection, and 2-D RoPE over
//   the whole head (the first quarter of the frequencies by patch row, the
//   second by column, each half repeated); every patch sees every patch;
// * the merger LayerNorms each patch, concatenates each merge x merge square
//   and applies fc2(gelu(fc1(.))) with exact GELU into the text width.
//
// Patches run in merge-block order (block rows, block columns, then row and
// column within the block), so merged tokens follow the merged grid row by
// row: token_grid is that grid, which the decoder's M-RoPE positions follow.
// The CPU encoder is the correctness path.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "json.h"
#include "quant.h"
#include "safetensors.h"
#include "vision.h"

namespace lokahi {

struct Qwen35VisionConfig {
  int depth = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_heads = 0;
  int patch_size = 0;
  int temporal_patch_size = 0;
  int merge = 0;            // spatial_merge_size
  int position_side = 0;    // sqrt(num_position_embeddings)
  int text_hidden_size = 0;  // out_hidden_size
  int image_token_id = -1;
  int64_t min_pixels = 65536;     // the processor's size.shortest_edge
  int64_t max_pixels = 16777216;  // and size.longest_edge
  float rescale = 1.0f / 255.0f;
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float std[3] = {0.5f, 0.5f, 0.5f};

  int head_dim() const { return hidden_size / num_heads; }
  int factor() const { return patch_size * merge; }

  // config.json plus, when given, preprocessor_config.json.
  static Qwen35VisionConfig from_json(const Json& config, const Json* preprocessor);
};

// Whether config.json describes a Qwen3.5 (or Qwen3.5-MoE) vision-language model.
bool is_qwen3_5_vision(const Json& config);

struct Qwen35VisionWeights {
  struct Layer {
    std::vector<float> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    Linear qkv, proj, fc1, fc2;  // views into the checkpoint
    std::vector<float> qkv_bias, proj_bias, fc1_bias, fc2_bias;
  };

  Qwen35VisionConfig config;
  std::shared_ptr<Checkpoint> checkpoint;
  // Owned float32: the patch embedding summed over time, [hidden, 3 * P * P]
  // over (channel, row, column), whichever layout the checkpoint stores; its
  // bias; the position table [side * side, hidden].
  std::vector<float> patch_weight, patch_bias, positions;
  std::vector<Layer> layers;
  std::vector<float> merger_norm_weight, merger_norm_bias;
  Linear merger_fc1, merger_fc2;
  std::vector<float> merger_fc1_bias, merger_fc2_bias;

  static std::shared_ptr<Qwen35VisionWeights> load(const std::string& directory, const Json& config);
};

// The geometry and preprocessing every Qwen3.5 encoder shares.
class Qwen35VisionEncoder : public VisionEncoder {
 public:
  int image_token_id() const override { return config().image_token_id; }
  int text_hidden_size() const override { return config().text_hidden_size; }
  std::pair<int, int> target_size(int width, int height) const override;
  int tokens_for(int width, int height) const override;
  std::pair<int, int> token_grid(int width, int height) const override;

  const Qwen35VisionConfig& config() const { return weights_->config; }

 protected:
  explicit Qwen35VisionEncoder(std::shared_ptr<const Qwen35VisionWeights> weights) : weights_(std::move(weights)) {}

  // Normalized patches [rows * cols, 3 * P * P] in merge-block order, each
  // flattened as (channel, row, column).
  std::vector<float> patches(const Image& image) const;
  // Patch i's (row, column) in merge-block order.
  std::pair<int, int> patch_position(int index, int cols) const;
  // Adds patch `index`'s position embedding, the table bilinearly
  // interpolated to a rows x cols patch grid (linspace sample points), to x.
  void add_position(float* x, int index, int rows, int cols) const;
  // The 2-D RoPE tables [rows * cols, head_dim / 2] in merge-block order:
  // head_dim / 4 frequencies over a head_dim / 2 rotary dimension; channel
  // i < head_dim / 4 turns by the patch row, the next head_dim / 4 by its
  // column, and channel i + head_dim / 2 is i's rotate-half partner.
  void rope_tables(int rows, int cols, float* cos_table, float* sin_table) const;

  std::shared_ptr<const Qwen35VisionWeights> weights_;
};

std::unique_ptr<VisionEncoder> make_qwen3_5_vision_cpu(std::shared_ptr<const Qwen35VisionWeights> weights,
                                                       unsigned threads);

}  // namespace lokahi
