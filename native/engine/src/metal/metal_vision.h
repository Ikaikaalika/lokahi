// Metal backend for the vision encoder (implemented in metal_vision.mm).
#pragma once

#include <memory>

#include "vision.h"

namespace lokahi {

// Whether the Metal encoder reads these weights: dense (F32, F16 or BF16)
// matrices, as mlx-community's skip_vision conversions keep them.
bool metal_vision_supports(const VisionWeights& weights);

std::unique_ptr<VisionEncoder> make_vision_metal(std::shared_ptr<const VisionWeights> weights);

}  // namespace lokahi
