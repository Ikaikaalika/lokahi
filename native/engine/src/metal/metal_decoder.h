// Metal GPU backend for dense decoders (implemented in metal_decoder.mm).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "decoder.h"
#include "model.h"

namespace lokahi {

bool metal_available();
std::string metal_device_name();

std::unique_ptr<Model> make_decoder_metal(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options);

// Runs the decoder's GPU token-selection kernel (argmax or seeded sampling)
// on rows of host logits, for checking it against the host reference:
// result[row * steps + i] is the token drawn at step first_step + i.
std::vector<int32_t> metal_sample_rows(const std::vector<float>& logits, int vocab, const SamplingParams& params,
                                       int first_step, int steps);

}  // namespace lokahi
