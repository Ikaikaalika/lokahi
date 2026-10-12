// Metal GPU backend for dense decoders (implemented in metal_decoder.mm).
#pragma once

#include <memory>
#include <string>

#include "decoder.h"
#include "model.h"

namespace lokahi {

bool metal_available();
std::string metal_device_name();

std::unique_ptr<Model> make_decoder_metal(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options);

}  // namespace lokahi
