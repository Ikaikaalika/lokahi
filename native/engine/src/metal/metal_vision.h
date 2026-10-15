// Metal backend for the vision encoders (implemented in metal_vision.mm).
#pragma once

#include <memory>

#include "siglip.h"

namespace lokahi {

// Whether the Metal SigLIP encoder reads these weights: dense (F32, F16 or
// BF16) matrices, as mlx-community's skip_vision conversions keep them.
bool metal_siglip_supports(const SiglipWeights& weights);

std::unique_ptr<VisionEncoder> make_siglip_metal(std::shared_ptr<const SiglipWeights> weights);

}  // namespace lokahi
