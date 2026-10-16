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

InjectionPlan plan_injection(const PromptInjection& injection, int base, int n, int width, int chunk,
                             const InjectionRules& rules) {
  InjectionPlan plan;
  plan.base = base;
  plan.rope_offset = rules.rope_offset;
  if (injection.empty()) return plan;
  LK_CHECK(injection.rows.size() == injection.positions.size() * static_cast<size_t>(width),
           "injected rows must be hidden_size wide");
  LK_CHECK(rules.mrope || injection.grids.empty(), "this model's positions do not follow image grids");
  LK_CHECK(static_cast<int>(injection.layer_rows.size()) <= rules.deepstack_layers,
           "this model takes deepstack rows for at most " + std::to_string(rules.deepstack_layers) + " layers, not " +
               std::to_string(injection.layer_rows.size()));
  for (const std::vector<float>& layer : injection.layer_rows) {
    LK_CHECK(layer.size() == injection.rows.size(), "each layer's deepstack rows must match the injected rows");
  }
  LK_CHECK(!rules.mrope || injection.grids.size() == injection.blocks.size(),
           "this model's positions follow each image's grid: give one grid per image block");
  plan.row.assign(static_cast<size_t>(n), -1);
  plan.last_key.resize(static_cast<size_t>(n));
  for (int t = 0; t < n; ++t) plan.last_key[static_cast<size_t>(t)] = base + t;
  for (size_t i = 0; i < injection.positions.size(); ++i) {
    const int rel = injection.positions[i] - base;
    LK_CHECK(rel >= 0 && rel < n, "image features must fall inside the prefill");
    LK_CHECK(plan.row[static_cast<size_t>(rel)] < 0, "an image position is given twice");
    plan.row[static_cast<size_t>(rel)] = static_cast<int>(i);
  }
  std::vector<int> block_of(static_cast<size_t>(n), -1);
  for (size_t b = 0; b < injection.blocks.size(); ++b) {
    const auto [begin, end] = injection.blocks[b];
    LK_CHECK(begin >= base && begin < end && end <= base + n, "image blocks must fall inside the prefill");
    for (int pos = begin; pos < end; ++pos) {
      LK_CHECK(block_of[static_cast<size_t>(pos - base)] < 0, "image blocks overlap");
      block_of[static_cast<size_t>(pos - base)] = static_cast<int>(b);
    }
    if (rules.mrope) {
      const auto [rows, cols] = injection.grids[b];
      LK_CHECK(rows > 0 && cols > 0 && int64_t{rows} * cols == end - begin,
               "an image's grid must cover its block's " + std::to_string(end - begin) + " tokens");
    }
    if (!rules.bidirectional_blocks) continue;
    LK_CHECK(end - begin <= chunk, "an image's " + std::to_string(end - begin) +
                                       " tokens do not fit one prefill chunk of " + std::to_string(chunk));
    for (int pos = begin; pos < end; ++pos) plan.last_key[static_cast<size_t>(pos - base)] = end - 1;
    plan.reach = std::max(plan.reach, end - 1 - begin);
    plan.blocks.emplace_back(begin, end);
  }
  if (rules.mrope) {
    plan.rope.resize(static_cast<size_t>(n) * 3);
    int offset = rules.rope_offset;
    for (int t = 0; t < n;) {
      int32_t* out = plan.rope.data() + static_cast<size_t>(t) * 3;
      const int b = block_of[static_cast<size_t>(t)];
      if (b < 0) {
        LK_CHECK(plan.row[static_cast<size_t>(t)] < 0, "this model's image features must lie in image blocks");
        out[0] = out[1] = out[2] = base + t + offset;
        ++t;
        continue;
      }
      // An image starting at s takes (s, s + row, s + column) over its grid.
      const auto [rows, cols] = injection.grids[static_cast<size_t>(b)];
      const int start = base + t + offset;
      for (int j = 0; j < rows * cols; ++j, ++t) {
        int32_t* at = plan.rope.data() + static_cast<size_t>(t) * 3;
        at[0] = start;
        at[1] = start + j / cols;
        at[2] = start + j % cols;
      }
      offset += std::max(rows, cols) - rows * cols;
    }
    plan.rope_offset = offset;
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

const float* Model::verify(const int32_t*, int) {
  throw Error(std::string(backend()) + ": speculative decoding is not supported on this backend");
}

void Model::accept(int) {
  throw Error(std::string(backend()) + ": speculative decoding is not supported on this backend");
}

std::vector<int32_t> Model::generate(const std::vector<int32_t>& prompt, int max_new,
                                     const SamplingParams& sampling, bool stop_at_eos, GenerationTiming* timing,
                                     const TokenCallback& on_token) {
  std::vector<int32_t> out;
  if (max_new <= 0) return out;
  validate_sampling(sampling);
  speculation_stats_ = {};
  if (speculation_.max_draft > 0 && sampling.greedy() && can_verify()) {
    return generate_speculative(prompt, max_new, sampling, stop_at_eos, timing, on_token);
  }
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

std::vector<int32_t> Model::generate_speculative(const std::vector<int32_t>& prompt, int max_new,
                                                 const SamplingParams& sampling, bool stop_at_eos,
                                                 GenerationTiming* timing, const TokenCallback& on_token) {
  std::vector<int32_t> out;
  const auto& eos = eos_token_ids();
  const int vocab = vocab_size();
  auto pick = [&](const float* logits) { return sample_token(logits, vocab, sampling, static_cast<int>(out.size())); };
  PromptLookupDrafter drafter(speculation_.reference);
  std::vector<int32_t> context = prompt;
  bool keep_going = true;
  // Emits one generated token; false once generation must stop after it.
  auto emit = [&](int32_t token) {
    out.push_back(token);
    context.push_back(token);
    if (timing) timing->token_times.push_back(now_seconds());
    keep_going = !on_token || on_token(token);
    const bool stop = stop_at_eos && std::find(eos.begin(), eos.end(), token) != eos.end();
    return keep_going && !stop && static_cast<int>(out.size()) < max_new;
  };
  const double start = now_seconds();
  if (timing) timing->token_times.clear();
  int32_t token = pick(prefill(prompt.data(), static_cast<int>(prompt.size())));
  const double first = now_seconds();
  bool more = emit(token);
  if (timing) {
    timing->prefill_seconds = first - start;
    timing->token_times = {first};
  }
  while (more) {
    // A pass yields up to one token beyond its drafts, so it drafts at most
    // the tokens still wanted minus one.
    const int room = std::min(speculation_.max_draft, max_new - static_cast<int>(out.size()) - 1);
    const std::vector<int32_t> draft = drafter.propose(context, room);
    if (draft.empty()) {
      token = pick(decode(token));
      more = emit(token);
      continue;
    }
    std::vector<int32_t> inputs = {token};
    inputs.insert(inputs.end(), draft.begin(), draft.end());
    const float* logits = verify(inputs.data(), static_cast<int>(inputs.size()));
    ++speculation_stats_.passes;
    speculation_stats_.drafted += static_cast<int>(draft.size());
    // Row i's greedy choice follows inputs[i]; while it equals the next
    // draft, that draft's keys and values (and states) stand.
    int kept = 1;
    for (size_t i = 0; i < inputs.size(); ++i) {
      token = pick(logits + i * static_cast<size_t>(vocab));
      more = emit(token);
      if (!more || i == draft.size() || token != draft[i]) break;
      ++kept;
      ++speculation_stats_.accepted;
    }
    accept(kept);
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
