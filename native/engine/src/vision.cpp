#include "vision.h"

#include <cctype>
#include <fstream>
#include <iterator>

#include "common.h"
#include "gemma4_vision.h"
#include "safetensors.h"
#include "siglip.h"

#if defined(LOKAHI_WITH_METAL)
#include "metal/metal_decoder.h"
#include "metal/metal_vision.h"
#endif

namespace lokahi {

namespace {

// The next whitespace-separated integer of a PPM header, skipping comments.
int ppm_field(const std::vector<uint8_t>& data, size_t& at, const std::string& path) {
  while (at < data.size()) {
    if (data[at] == '#') {
      while (at < data.size() && data[at] != '\n') ++at;
    } else if (std::isspace(data[at])) {
      ++at;
    } else {
      break;
    }
  }
  LK_CHECK(at < data.size() && std::isdigit(data[at]), path + ": malformed PPM header");
  long value = 0;
  while (at < data.size() && std::isdigit(data[at])) {
    value = value * 10 + (data[at++] - '0');
    LK_CHECK(value <= 1 << 16, path + ": PPM dimension too large");
  }
  return static_cast<int>(value);
}

}  // namespace

Image read_image(const std::string& path, int raw_side) {
  std::ifstream in(path, std::ios::binary);
  LK_CHECK(in.good(), "cannot read image " + path);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  Image image;
  if (data.size() >= 2 && data[0] == 'P' && data[1] == '6') {
    size_t at = 2;
    image.width = ppm_field(data, at, path);
    image.height = ppm_field(data, at, path);
    LK_CHECK(ppm_field(data, at, path) == 255, path + ": only 8-bit PPM (maxval 255) is supported");
    LK_CHECK(at < data.size() && std::isspace(data[at]), path + ": malformed PPM header");
    ++at;  // exactly one whitespace byte before the pixels
    const size_t bytes = static_cast<size_t>(image.width) * image.height * 3;
    LK_CHECK(image.width > 0 && image.height > 0 && data.size() - at == bytes,
             path + ": PPM pixel data does not match its size");
    image.rgb.assign(data.begin() + static_cast<std::ptrdiff_t>(at), data.end());
    return image;
  }
  LK_CHECK(raw_side > 0, path + " is not a PPM (P6) image; this model's images vary in size");
  const size_t bytes = static_cast<size_t>(raw_side) * raw_side * 3;
  LK_CHECK(data.size() == bytes, path + " is not a " + std::to_string(raw_side) + "x" + std::to_string(raw_side) +
                                     " RGB8 image (" + std::to_string(bytes) + " bytes) or a PPM");
  image.width = image.height = raw_side;
  image.rgb = std::move(data);
  return image;
}

bool has_vision(const Json& config) {
  const Json* vision = config.find("vision_config");
  return vision != nullptr && vision->is_object();
}

std::unique_ptr<VisionEncoder> VisionEncoder::load(const std::string& directory, const std::string& requested,
                                                   unsigned threads) {
  const Json config = Json::parse(read_text_file(directory + "/config.json"));
  if (!has_vision(config)) return nullptr;
  std::string backend = requested;
  if (is_gemma4_vision(config)) {
    auto weights = Gemma4VisionWeights::load(directory, config);
    if (backend == "auto") backend = "cpu";
    LK_CHECK(backend != "metal", "metal: Gemma 4's vision tower runs on the CPU only so far; use --backend cpu");
    LK_CHECK(backend == "cpu", "unknown backend '" + backend + "'");
    return make_gemma4_vision_cpu(std::move(weights), threads);
  }
  LK_CHECK(is_siglip(config), "vision: unsupported vision tower '" +
                                  config.at("vision_config").string_or("model_type", "") + "'");
  auto weights = SiglipWeights::load(directory, config);
#if defined(LOKAHI_WITH_METAL)
  if (backend == "auto") backend = metal_siglip_supports(*weights) && metal_available() ? "metal" : "cpu";
  if (backend == "metal") return make_siglip_metal(std::move(weights));
#endif
  if (backend == "auto") backend = "cpu";
  LK_CHECK(backend == "cpu", "unknown or unavailable backend '" + backend + "'");
  return make_siglip_cpu(std::move(weights), threads);
}

PromptInjection place_images(const std::vector<int32_t>& tokens, int start, int image_token, int width,
                             const std::vector<std::vector<float>>& features) {
  const size_t w = static_cast<size_t>(width);
  std::vector<int> runs;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == image_token) runs.push_back(static_cast<int>(i));
  }
  size_t rows = 0;
  for (const auto& image : features) {
    LK_CHECK(!image.empty() && image.size() % w == 0, "image features have the wrong size");
    rows += image.size() / w;
  }
  LK_CHECK(runs.size() == rows, "the prompt has " + std::to_string(runs.size()) + " image tokens for " +
                                    std::to_string(features.size()) + " image(s) of " + std::to_string(rows) +
                                    " tokens in all");
  PromptInjection out;
  size_t next = 0;
  for (const auto& image : features) {
    const int count = static_cast<int>(image.size() / w);
    const int first = runs[next];
    LK_CHECK(runs[next + static_cast<size_t>(count) - 1] == first + count - 1, "each image's tokens must be contiguous");
    next += static_cast<size_t>(count);
    out.blocks.emplace_back(start + first, start + first + count);
    for (int j = 0; j < count; ++j) out.positions.push_back(start + first + j);
    out.rows.insert(out.rows.end(), image.begin(), image.end());
  }
  return out;
}

}  // namespace lokahi
