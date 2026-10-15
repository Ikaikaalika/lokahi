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

int InjectionPlan::pass_length(int start, int count) const {
  for (const auto& [begin, end] : blocks) {
    const int b = begin - base;
    if (b > start && b < start + count && end - base > start + count) count = b - start;
  }
  return count;
}

InjectionPlan plan_injection(const PromptInjection& injection, int base, int n, int width, int chunk) {
  InjectionPlan plan;
  plan.base = base;
  if (injection.empty()) return plan;
  LK_CHECK(injection.rows.size() == injection.positions.size() * static_cast<size_t>(width),
           "injected rows must be hidden_size wide");
  plan.row.assign(static_cast<size_t>(n), -1);
  plan.last_key.resize(static_cast<size_t>(n));
  for (int t = 0; t < n; ++t) plan.last_key[static_cast<size_t>(t)] = base + t;
  for (size_t i = 0; i < injection.positions.size(); ++i) {
    const int rel = injection.positions[i] - base;
    LK_CHECK(rel >= 0 && rel < n, "image features must fall inside the prefill");
    LK_CHECK(plan.row[static_cast<size_t>(rel)] < 0, "an image position is given twice");
    plan.row[static_cast<size_t>(rel)] = static_cast<int>(i);
  }
  std::vector<bool> taken(static_cast<size_t>(n), false);
  for (const auto& [begin, end] : injection.blocks) {
    LK_CHECK(begin >= base && begin < end && end <= base + n, "image blocks must fall inside the prefill");
    LK_CHECK(end - begin <= chunk, "an image's " + std::to_string(end - begin) +
                                       " tokens do not fit one prefill chunk of " + std::to_string(chunk));
    for (int pos = begin; pos < end; ++pos) {
      LK_CHECK(!taken[static_cast<size_t>(pos - base)], "image blocks overlap");
      taken[static_cast<size_t>(pos - base)] = true;
      plan.last_key[static_cast<size_t>(pos - base)] = end - 1;
    }
    plan.reach = std::max(plan.reach, end - 1 - begin);
    plan.blocks.emplace_back(begin, end);
  }
  return plan;
}

void Model::inject(PromptInjection injection) {
  LK_CHECK(accepts_injection(), std::string("the ") + backend() + " backend does not take image features yet");
  injection_ = std::move(injection);
}

int32_t argmax(const float* values, int count) {
  int32_t best = 0;
  for (int i = 1; i < count; ++i) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

std::vector<int32_t> Model::generate(const std::vector<int32_t>& prompt, int max_new,
                                     const SamplingParams& sampling, bool stop_at_eos, GenerationTiming* timing,
                                     const TokenCallback& on_token) {
  std::vector<int32_t> out;
  if (max_new <= 0) return out;
  validate_sampling(sampling);
  const auto& eos = eos_token_ids();
  auto pick = [&](const float* logits) {
    return sample_token(logits, vocab_size(), sampling, static_cast<int>(out.size()));
  };
  const double start = now_seconds();
  int32_t token = pick(prefill(prompt.data(), static_cast<int>(prompt.size())));
  const double first = now_seconds();
  out.push_back(token);
  if (timing) {
    timing->prefill_seconds = first - start;
    timing->token_times = {first};
  }
  bool keep_going = !on_token || on_token(token);
  while (keep_going && static_cast<int>(out.size()) < max_new) {
    if (stop_at_eos && std::find(eos.begin(), eos.end(), token) != eos.end()) break;
    token = pick(decode(token));
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
  // Metal reads MXFP4 only for experts so far.
  if (backend == "auto") {
    const bool metal = !weights->dense_mxfp4();
    backend = metal && metal_available() ? "metal" : "cpu";
  }
  if (backend == "metal") return make_decoder_metal(std::move(weights), options);
#else
  if (backend == "auto") backend = "cpu";
  LK_CHECK(backend != "metal", "this build has no Metal backend");
#endif
  LK_CHECK(backend == "cpu", "unknown backend '" + backend + "'");
  return make_decoder_cpu(std::move(weights), options);
}

}  // namespace lokahi
