#include "model.h"

#include <algorithm>

#include "common.h"
#include "decoder.h"

#if defined(LOKAHI_WITH_METAL)
#include "metal/metal_decoder.h"
#endif

namespace lokahi {

void Profiler::add(const std::string& name, double seconds, double bytes) {
  for (ProfileSection& section : sections_) {
    if (section.name == name) {
      section.seconds += seconds;
      section.calls += 1;
      section.bytes += bytes;
      return;
    }
  }
  sections_.push_back({name, seconds, 1, bytes});
}

int32_t argmax(const float* values, int count) {
  int32_t best = 0;
  for (int i = 1; i < count; ++i) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

std::vector<int32_t> Model::generate_greedy(const std::vector<int32_t>& prompt, int max_new,
                                            bool stop_at_eos, GenerationTiming* timing,
                                            const TokenCallback& on_token) {
  std::vector<int32_t> out;
  if (max_new <= 0) return out;
  const auto& eos = eos_token_ids();
  const double start = now_seconds();
  int32_t token = argmax(prefill(prompt.data(), static_cast<int>(prompt.size())), vocab_size());
  const double first = now_seconds();
  out.push_back(token);
  if (timing) {
    timing->prefill_seconds = first - start;
    timing->token_times = {first};
  }
  bool keep_going = !on_token || on_token(token);
  while (keep_going && static_cast<int>(out.size()) < max_new) {
    if (stop_at_eos && std::find(eos.begin(), eos.end(), token) != eos.end()) break;
    token = argmax(decode(token), vocab_size());
    out.push_back(token);
    if (timing) timing->token_times.push_back(now_seconds());
    keep_going = !on_token || on_token(token);
  }
  if (timing) timing->decode_seconds = timing->token_times.back() - first;
  return out;
}

std::unique_ptr<Model> load_model(const std::string& directory, const LoadOptions& options) {
  auto weights = DecoderWeights::load(directory);
  std::string backend = options.backend;
#if defined(LOKAHI_WITH_METAL)
  if (backend == "auto") backend = metal_available() ? "metal" : "cpu";
  if (backend == "metal") return make_decoder_metal(std::move(weights), options);
#else
  if (backend == "auto") backend = "cpu";
  LK_CHECK(backend != "metal", "this build has no Metal backend");
#endif
  LK_CHECK(backend == "cpu", "unknown backend '" + backend + "'");
  return make_decoder_cpu(std::move(weights), options);
}

}  // namespace lokahi
