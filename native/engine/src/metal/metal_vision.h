// Metal backend for the vision encoders (implemented in metal_vision.mm).
#pragma once

#include <memory>

#include "gemma4_vision.h"
#include "qwen3_5_vision.h"
#include "siglip.h"

namespace lokahi {

// Whether the Metal SigLIP encoder reads these weights: dense (F32, F16 or
// BF16) matrices, as mlx-community's skip_vision conversions keep them.
bool metal_siglip_supports(const SiglipWeights& weights);

std::unique_ptr<VisionEncoder> make_siglip_metal(std::shared_ptr<const SiglipWeights> weights);

// Whether the Metal Gemma 4 encoder reads these weights: a dense tower (the
// embedder's projection may be quantized; it is read as float32).
bool metal_gemma4_vision_supports(const Gemma4VisionWeights& weights);

std::unique_ptr<VisionEncoder> make_gemma4_vision_metal(std::shared_ptr<const Gemma4VisionWeights> weights);

// Whether the Metal Qwen3.5 encoder reads these weights: dense block and
// merger matrices, as mlx-community's conversions keep them.
bool metal_qwen3_5_vision_supports(const Qwen35VisionWeights& weights);

std::unique_ptr<VisionEncoder> make_qwen3_5_vision_metal(std::shared_ptr<const Qwen35VisionWeights> weights);

}  // namespace lokahi
