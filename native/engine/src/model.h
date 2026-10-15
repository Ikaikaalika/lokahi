// Backend-neutral model interface used by the C ABI and the CLI.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sampling.h"

namespace lokahi {

struct LoadOptions {
  std::string backend = "auto";  // auto | cpu | metal
  int max_context = 4096;
  int prefill_chunk = 512;
  unsigned cpu_threads = 0;  // 0: hardware concurrency
};

struct GenerationTiming {
  double prefill_seconds = 0.0;      // admitted prompt to first token available
  double decode_seconds = 0.0;       // first token available to last token available
  std::vector<double> token_times;   // host time each generated token became available
};

// Time and memory traffic per section of the forward pass (embedding, norms,
// projections, attention, ...), for attributing where a step's time goes.
// Bytes are the minimum traffic the section must move: weights it reads plus
// activations and KV it reads and writes.
struct ProfileSection {
  std::string name;
  double seconds = 0.0;
  int64_t calls = 0;
  double bytes = 0.0;
};

class Profiler {
 public:
  void add(const std::string& name, double seconds, double bytes);
  // Sections in the order they were first seen.
  const std::vector<ProfileSection>& sections() const { return sections_; }
  void clear() { sections_.clear(); }

 private:
  std::vector<ProfileSection> sections_;
};

// Rows that replace the scaled token embeddings at some prompt positions
// (image features, hidden_size wide) and [begin, end) position ranges whose
// tokens also attend to each other's later positions during prefill (one per
// image, as Gemma 3's image blocks do). Positions are absolute.
struct PromptInjection {
  std::vector<int> positions;
  std::vector<float> rows;  // [positions.size(), hidden_size]
  std::vector<std::pair<int, int>> blocks;
  bool empty() const { return positions.empty() && blocks.empty(); }
};

class Model {
 public:
  virtual ~Model() = default;

  virtual const char* backend() const = 0;
  virtual const char* architecture() const = 0;
  virtual int vocab_size() const = 0;
  virtual int max_context() const = 0;
  virtual int position() const = 0;
  virtual size_t weight_bytes() const = 0;
  virtual const std::vector<int32_t>& eos_token_ids() const = 0;

  virtual void reset() = 0;

  // While set, the backend times every section of each forward pass into
  // `profiler` (nullptr turns it off). Profiling serializes GPU work per
  // section, so it measures attribution, not the fast path's wall time.
  void set_profiler(Profiler* profiler) { profiler_ = profiler; }

  // Append `n` tokens at the current position. Returns the logits of the last
  // token; the pointer stays valid until the next call.
  virtual const float* prefill(const int32_t* tokens, int n) = 0;

  // Applies `injection` to the next prefill (also the one generate starts
  // with), whose tokens must cover all of it; reset drops it. Backends that
  // cannot take one fail closed here.
  void inject(PromptInjection injection);
  virtual const float* decode(int32_t token) = 0;

  // Called with each generated token as it becomes available on the host;
  // returning false stops generation after that token.
  using TokenCallback = std::function<bool(int32_t)>;

  // Generates up to max_new tokens with `sampling` (sampling.h); generated
  // token i is drawn with step i. Backends may keep tokens on the device and
  // pipeline steps; the default runs prefill + decode and samples on the host.
  virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_new,
                                        const SamplingParams& sampling, bool stop_at_eos,
                                        GenerationTiming* timing, const TokenCallback& on_token = {});

  std::vector<int32_t> generate_greedy(const std::vector<int32_t>& prompt, int max_new, bool stop_at_eos,
                                       GenerationTiming* timing, const TokenCallback& on_token = {}) {
    return generate(prompt, max_new, SamplingParams{}, stop_at_eos, timing, on_token);
  }

 protected:
  virtual bool accepts_injection() const { return false; }

  Profiler* profiler_ = nullptr;
  PromptInjection injection_;  // pending for the next prefill
};

int32_t argmax(const float* values, int count);

std::unique_ptr<Model> load_model(const std::string& directory, const LoadOptions& options);

}  // namespace lokahi
