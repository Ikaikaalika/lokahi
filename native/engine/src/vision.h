// Image inputs: the vision encoders' common interface, image files, and the
// placement of image features in a prompt.
//
// Each model family's encoder computes its processor's normalization, its
// vision tower and its projector into the text width, as Transformers does:
// Gemma 3's SigLIP (siglip.h) and Gemma 4's tower (gemma4_vision.h), on the
// CPU or, through metal/metal_vision.h, the GPU, Qwen3.5's and Qwen3-VL's
// (qwen3_5_vision.h) likewise, and Qwen2.5-VL's (also qwen3_5_vision.h) on
// the CPU. Images arrive already resized to a size the
// model takes (target_size says which).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "json.h"
#include "model.h"

namespace lokahi {

// An RGB8 image, rows top to bottom.
struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;  // [height, width, 3]
};

// An image's features: one row per image token, and for Qwen3-VL one more
// set of rows per deepstack layer.
struct ImageFeatures {
  std::vector<float> rows;  // [tokens, text_hidden_size]
  std::vector<std::vector<float>> deepstack;
};

// A binary PPM (P6, maxval 255) file, or raw RGB8 bytes of a square image of
// side `raw_side` (0: raw files are refused).
Image read_image(const std::string& path, int raw_side);

// Whether config.json describes a vision tower.
bool has_vision(const Json& config);

class VisionEncoder {
 public:
  virtual ~VisionEncoder() = default;

  // The encoder for the checkpoint in `directory` on `backend` ("cpu",
  // "metal" or "auto"); nullptr when it has no vision tower.
  static std::unique_ptr<VisionEncoder> load(const std::string& directory, const std::string& backend = "auto",
                                             unsigned threads = 0);

  virtual const char* backend() const = 0;
  virtual int image_token_id() const = 0;
  virtual int text_hidden_size() const = 0;
  // The side every image has, for models with one square size; 0 when sizes vary.
  virtual int fixed_size() const { return 0; }
  // The size (width, height) the model's processor resizes a width x height
  // image to.
  virtual std::pair<int, int> target_size(int width, int height) const = 0;
  // Feature rows of an image of this size; fails closed on a size the model
  // does not take.
  virtual int tokens_for(int width, int height) const = 0;
  // The (rows, columns) an image's tokens fill, row by row, for models whose
  // decoder positions follow it (Qwen3.5's M-RoPE); {0, 0} for the others.
  virtual std::pair<int, int> token_grid(int /*width*/, int /*height*/) const { return {0, 0}; }
  // An image at an accepted size to its features [tokens_for(width, height),
  // text_hidden_size].
  virtual std::vector<float> encode(const Image& image) = 0;
  // The features with any deepstack rows.
  virtual ImageFeatures encode_features(const Image& image) { return {encode(image), {}}; }
};

// Places images' features at the runs of `image_token` in `tokens`, a prompt
// whose first token sits at position `start`: image i fills the next
// images[i].rows.size() / width tokens, which become its block, and its
// deepstack rows go to the same positions. `grids`, when not empty, gives
// each image's token_grid.
PromptInjection place_images(const std::vector<int32_t>& tokens, int start, int image_token, int width,
                             const std::vector<ImageFeatures>& images,
                             const std::vector<std::pair<int, int>>& grids = {});

}  // namespace lokahi
