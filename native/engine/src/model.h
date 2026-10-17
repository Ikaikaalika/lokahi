// Backend-neutral model interface used by the C ABI and the CLI.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sampling.h"
#include "speculation.h"

namespace lokahi {

struct LoadOptions {
  std::string backend = "auto";  // auto | cpu | metal
  int max_context = 4096;
  int prefill_chunk = 512;
  unsigned cpu_threads = 0;  // 0: hardware concurrency
  // The keys' and values' element type: auto (f16 on Metal, f32 on the CPU
  // reference backend), f16 or f32. Latent attention keeps f32.
  std::string kv_cache = "auto";
  // A qwen3_5_mtp drafter directory: a multi-token prediction head split
  // from the checkpoint (DecoderWeights::mtp).
  std::string draft_model;
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
// (image features, hidden_size wide) and [begin, end) position ranges, one
// per image. In models whose image blocks are bidirectional (Gemma 3's) a
// block's tokens also attend to each other's later positions during prefill.
// Positions are absolute.
//
// `grids` gives each block's (rows, columns), for models whose RoPE follows
// an image's layout (Qwen3.5's and Qwen3-VL's M-RoPE): the block's tokens
// fill the grid row by row. It is empty for every other model.
//
// `layer_rows[l]` (Qwen3-VL's deepstack), laid out like `rows`, is added to
// the hidden states at `positions` after decoder layer l.
struct PromptInjection {
  std::vector<int> positions;
  std::vector<float> rows;  // [positions.size(), hidden_size]
  std::vector<std::pair<int, int>> blocks;
  std::vector<std::pair<int, int>> grids;
  std::vector<std::vector<float>> layer_rows;
  bool empty() const { return positions.empty() && blocks.empty(); }
};

// How a decoder reads an injection.
struct InjectionRules {
  bool bidirectional_blocks = false;  // a block's tokens see each other, so no pass may split one
  bool mrope = false;                 // blocks carry grids, and RoPE positions follow them
  // M-RoPE: how far RoPE positions run ahead of the KV positions before this
  // prefill (images so far have moved them).
  int rope_offset = 0;
  int deepstack_layers = 0;  // the most layers that may take layer_rows (Qwen3-VL)
};

// A PromptInjection checked against a prefill of `n` tokens whose first sits
// at position `base`, indexed by offset into the prefill.
struct InjectionPlan {
  int base = 0;
  std::vector<int> row;       // [n] index into the injection's rows, or -1
  std::vector<int> last_key;  // [n] the last position each token attends to
  int reach = 0;              // the most any token looks ahead: max(last_key[t] - base - t)
  std::vector<std::pair<int, int>> blocks;  // bidirectional blocks only
  // M-RoPE: [n, 3] each token's (temporal, row, column) RoPE positions, and
  // the offset of the tokens after the prefill (text positions continue
  // past an image at its start plus max(rows, columns)).
  std::vector<int32_t> rope;
  int rope_offset = 0;

  bool empty() const { return row.empty(); }
  // How many of `count` tokens from offset `start` a pass should take so it
  // never splits a block: it ends where a block begins.
  int pass_length(int start, int count) const;
};

// Fails closed unless every row and block lies inside the prefill, rows are
// `width` wide, blocks do not overlap, each bidirectional block fits a pass
// of `chunk` tokens, grids are given exactly when the rules use them, one
// per block, each covering its block, and layer rows match the rows, for no
// more layers than the rules allow.
InjectionPlan plan_injection(const PromptInjection& injection, int base, int n, int width, int chunk,
                             const InjectionRules& rules);

class Model {
 public:
  virtual ~Model() = default;

  virtual const char* backend() const = 0;
  virtual const char* architecture() const = 0;
  virtual int vocab_size() const = 0;
  virtual int max_context() const = 0;
  virtual int position() const = 0;
  virtual size_t weight_bytes() const = 0;
  // The KV cache's element type ("f16" or "f32").
  virtual const char* kv_cache() const { return "f32"; }
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

  // Speculative decoding. verify appends n tokens at the current position, as
  // prefill does, and returns every row's logits ([n, vocab_size]); accept
  // then keeps the first `kept` (1 to n) of them and rolls the rest back,
  // recurrent states included. Backends that cannot roll back say so here,
  // and generate decodes one token at a time instead.
  virtual bool can_verify() const { return false; }
  virtual const float* verify(const int32_t* tokens, int n);
  virtual void accept(int kept);
  // About what one more verified token adds to a pass, in decode steps; the
  // adaptive draft policy weighs drafts against it (DraftBudget).
  virtual double verify_token_cost() const { return 0.25; }

  // Multi-token prediction (Qwen3.5's MTP head). While the head follows the
  // model, which must start from an empty context, every pass also writes
  // the head's rows; mtp_draft then drafts up to `count` greedy tokens after
  // `token`, the last token generated (not yet fed), and appends each
  // draft's [vocab] logits to `logits` when given. Greedy generation with an
  // MTP drafter arranges the following itself.
  virtual bool has_mtp() const { return false; }
  virtual void follow_mtp(bool on);
  virtual std::vector<int32_t> mtp_draft(int32_t token, int count, std::vector<float>* logits = nullptr);

  // Greedy generation drafts up to options.max_draft tokens per verify pass,
  // fewer under the adaptive policy (speculation.h); other sampling decodes
  // one token at a time.
  void set_speculation(SpeculationOptions options) { speculation_ = std::move(options); }
  const SpeculationStats& speculation_stats() const { return speculation_stats_; }

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

  // The speculative loop behind generate, for greedy sampling.
  std::vector<int32_t> generate_speculative(const std::vector<int32_t>& prompt, int max_new,
                                            const SamplingParams& sampling, bool stop_at_eos,
                                            GenerationTiming* timing, const TokenCallback& on_token);

  Profiler* profiler_ = nullptr;
  PromptInjection injection_;  // pending for the next prefill
  SpeculationOptions speculation_;
  SpeculationStats speculation_stats_;
};

int32_t argmax(const float* values, int count);

std::unique_ptr<Model> load_model(const std::string& directory, const LoadOptions& options);

}  // namespace lokahi
