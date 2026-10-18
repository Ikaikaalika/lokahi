// Metal backend for decoders (Gemma 3, Gemma 4, the Llama family, DeepSeek and
// Qwen3.5).
//
// One serial compute encoder per command buffer runs a whole prefill (all
// chunks) or one decode step: embedding, every layer, the last-row
// vocabulary projection, and device-side argmax or seeded sampling.
// Generation keeps two decode steps in flight; each step reads the previous
// token from GPU memory, so the host never sits on the critical path between
// tokens.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/metal_decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "decoder_kernels_source.h"
#include "metal/metal_common.h"

namespace lokahi {

std::string ns_error(NSError* error) {
  return error ? std::string([[error localizedDescription] UTF8String]) : std::string("unknown error");
}

id<MTLLibrary> compile_kernels(id<MTLDevice> device) {
  MTLCompileOptions* compile = [MTLCompileOptions new];
  compile.languageVersion = MTLLanguageVersion3_0;
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:@(kDecoderKernelSource) options:compile error:&error];
  LK_CHECK(library != nil, "metal: kernel compilation failed: " + ns_error(error));
  return library;
}

namespace {

struct BufRef {
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0;
};

struct GpuLinear {
  BufRef words, scales, biases;
  int rows = 0;
  int cols = 0;
  int bits = 0;
  int group = 0;
  bool bf16 = true;
  bool mxfp4 = false;  // scales hold E8M0 exponents; no biases

  bool same_format(const GpuLinear& other) const {
    return bits == other.bits && group == other.group && bf16 == other.bf16 && cols == other.cols &&
           mxfp4 == other.mxfp4;
  }
};

// A stack of same-shape expert matrices stored contiguously, expert-major.
struct GpuExperts {
  BufRef words, scales, biases;
  int rows = 0, cols = 0, bits = 0, group = 0;
  bool bf16 = true, mxfp4 = false;
  BufRef bias;  // [experts, rows] (GPT-OSS), or the placeholder
  GpuLinear format() const {
    GpuLinear l;
    l.rows = rows;
    l.cols = cols;
    l.bits = bits;
    l.group = group;
    l.bf16 = bf16;
    l.mxfp4 = mxfp4;
    return l;
  }
};

// A dense matrix (GLM's unquantized router): dtype 0 F32, 1 BF16, 2 F16.
struct GpuDense {
  BufRef data;
  int rows = 0, cols = 0, dtype = 0;
};

struct GpuLayer {
  GpuLinear q, k, v, o, gate, up, down;
  GpuLinear router;                          // mixture-of-experts layers
  GpuDense dense_router;                     // when the router is not quantized
  bool router_dense = false;
  GpuLinear shared_gate, shared_up, shared_down;  // GLM's shared expert
  BufRef router_correction;                  // GLM's selection bias, or the placeholder
  GpuExperts expert_gate, expert_up, expert_down;
  bool moe = false;
  BufRef o_bias, sinks, router_bias;          // GPT-OSS, or the placeholder
  GpuLinear pl_gate, pl_proj;  // Gemma 4's per-layer input gate and projection
  // Norms and biases a family does not have are bound to a placeholder and
  // never read.
  BufRef attn_norm, attn_post_norm, mlp_norm, mlp_post_norm, q_norm, k_norm, q_bias, k_bias, v_bias, pl_norm;
  // Gemma 4's experts beside the dense MLP: the router's input scale, the
  // per-expert weight scales, and the extra norms.
  BufRef router_input_scale, per_expert_scale, mlp_post_norm_1, expert_norm, expert_post_norm;
  // Multi-head latent attention: the query's down projection (q is q_b_proj
  // then) and the latent projection with their norms; the query absorption,
  // stacked per head (embed_q) or as kv_b_proj's key rows; and the per-head
  // expansion, stacked (unembed_out, or kv_b_proj's value rows copied
  // contiguous). The latent cache is k_cache, and also v_cache.
  GpuLinear q_a, kv_a, kv_b;
  GpuExperts embed_q, unembed_out;
  bool absorbed = false;  // embed_q, rather than kv_b_proj's key rows
  BufRef q_a_norm, kv_a_norm;
  // Gated DeltaNet (Qwen3.5, Qwen3-Next): Qwen3.5's four input projections or
  // Qwen3-Next's two grouped ones, the conv taps, decay rates exp(A_log),
  // dt_bias and the gated norm's weight, out_proj; and the conv history and
  // states, double-buffered by position parity (see gdn_recurrence).
  bool linear = false, grouped = false;
  GpuLinear in_qkv, in_z, in_b, in_a, in_qkvz, in_ba, out_proj;
  BufRef conv_taps, decay_rate, dt_bias, linear_norm;
  std::array<id<MTLBuffer>, 2> conv_history = {nil, nil};
  std::array<id<MTLBuffer>, 2> delta_state = {nil, nil};
  GpuLinear shared_expert_gate;  // Qwen's [1, hidden] gate on the shared expert
  // A KV-shared layer's caches and capacity are its source layer's.
  id<MTLBuffer> k_cache = nil;
  id<MTLBuffer> v_cache = nil;
  int capacity = 0;
  bool sliding = false;
  int head_dim = 0, kv_heads = 0, inter = 0;
  bool own_kv = true;  // computes and stores its own keys and values
  bool k_eq_v = false;  // values are the raw key projection
  float scalar = 1.0f;  // Gemma 4's layer_scalar
};

// Packs read-only host regions into a few large GPU buffers.
class WeightArena {
 public:
  explicit WeightArena(id<MTLDevice> device) : device_(device) {}

  size_t add(const void* data, size_t bytes) {
    items_.push_back({data, bytes, {}});
    return items_.size() - 1;
  }

  void finalize() {
    const size_t limit = std::min<size_t>(static_cast<size_t>([device_ maxBufferLength]), size_t(1) << 32);
    size_t i = 0;
    while (i < items_.size()) {
      size_t total = 0;
      size_t j = i;
      while (j < items_.size() && total + align_up(items_[j].bytes, 256) <= limit) {
        total += align_up(items_[j].bytes, 256);
        ++j;
      }
      LK_CHECK(j > i, "metal: tensor exceeds maxBufferLength");
      id<MTLBuffer> buffer =
          [device_ newBufferWithLength:total
                               options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
      LK_CHECK(buffer != nil, "metal: weight allocation failed");
      uint8_t* base = static_cast<uint8_t*>([buffer contents]);
      size_t offset = 0;
      for (size_t k = i; k < j; ++k) {
        std::memcpy(base + offset, items_[k].data, items_[k].bytes);
        items_[k].ref = {buffer, offset};
        offset += align_up(items_[k].bytes, 256);
      }
      buffers_.push_back(buffer);
      bytes_ += total;
      i = j;
    }
  }

  BufRef get(size_t index) const { return items_[index].ref; }
  size_t bytes() const { return bytes_; }

 private:
  struct Item {
    const void* data;
    size_t bytes;
    BufRef ref;
  };
  id<MTLDevice> device_;
  std::vector<Item> items_;
  std::vector<id<MTLBuffer>> buffers_;
  size_t bytes_ = 0;
};

// Parameter blocks; layouts match the structs in decoder_kernels.metal.
struct EmbedParams { int cols; int n; float scale; };
struct NormParams { int cols; float eps; float offset; };
struct ResidualNormParams { int cols; float eps; float offset; int post_norm; int has_next; float scale; };
struct QmvParams { int rows; int cols; };
struct GluParams { int rows; int cols; int silu; };
struct QkvParams { int q_rows; int kv_rows; int cols; };
struct QmmParams { int n; int rows; int cols; };
struct QmvNParams { int rows; int cols; int n; };
constexpr int kQmvNMax = 8;  // QMV_N_MAX in decoder_kernels.metal
struct ElementwiseParams { int count; };
struct GluMulParams { int count; int silu; };
struct RopeParams {
  int heads; int kv_heads; int head_dim; int start_pos; int capacity; float eps; float offset; int qk_norm;
  int qkv_bias; int v_norm; int rotary; int mrope; int rope_given; int rope_offset; int first_head;
};
struct PerLayerCombineParams { int size; int layers; float eps; float offset; float proj_scale; float input_scale; };
struct PerLayerGateParams { int size; int layers; int layer; int count; };
struct MoeRouteParams {
  int experts; int k; int topk_softmax; int norm_topk; int has_bias; int has_scale; int groups; int topk_groups;
  float scaling;
};
struct GatherParams { int rows; int cols; int k; int input_rows; int act; int has_bias; float limit; int x_stride; int y_stride; };
struct AbsorbParams { int heads; int nope; int head_rows; int cols; int x_stride; int y_stride; };
struct MlaRopeParams { int heads; int nope; int rope; int rank; int start_pos; float eps; int pairs; float factor; };
struct MoeCombineParams { int cols; int k; int count; int has_shared; };
struct GdnSplitParams { int key_heads; int key_dim; int ratio; int value_dim; int n; };
struct GdnConvParams { int channels; int taps; int n; };
struct GdnNormParams { int key_heads; int key_dim; int channels; float q_scale; };
struct GdnGateParams { int value_heads; int count; };
struct GdnRecurrenceParams { int key_heads; int value_heads; int key_dim; int value_dim; int channels; int n; };
struct GdnHistoryParams { int channels; int taps; int kept; };
struct SplitGateParams { int head_dim; int count; };
struct SigmoidScaleParams { int cols; int count; };
struct MoeGroupedParams { int experts; int k; int groups; int topk_groups; int norm_topk; float scaling; int exclude; };
struct DenseParams { int rows; int cols; int dtype; };
struct AttnParams {
  int heads; int kv_heads; int head_dim; int start_pos; int window; int capacity; int block; int n_blocks;
  float scale; float softcap; int has_last_key; int groups_per_kv;
};
struct AttnMmParams {
  int heads; int kv_heads; int head_dim; int start_pos; int row0; int rows; int key_base; int keys; int window;
  int capacity; float scale; float softcap; int has_last_key;
};
struct ReplaceParams { int cols; int n; };
struct ReduceParams { int heads; int head_dim; int n_blocks; int has_sinks; };
struct BiasParams { int cols; int count; };
struct ArgmaxParams { int vocab; int history_index; };
struct SampleParams {
  int vocab;
  int history_index;  // also the step that keys the sampling noise
  float inv_temperature;
  float top_p;      // >= 1 disables
  int top_k;        // <= 0 or >= vocab disables
  float log_min_p;  // -inf disables
  uint32_t seed_lo;
  uint32_t seed_hi;
};

SampleParams sample_params(const SamplingParams& sp, int vocab, int history_index) {
  return SampleParams{vocab, history_index, 1.0f / sp.temperature, sp.top_p, sp.top_k,
                      sp.min_p > 0.0f ? std::log(sp.min_p) : -INFINITY, static_cast<uint32_t>(sp.seed),
                      static_cast<uint32_t>(sp.seed >> 32)};
}

constexpr int kDecodeMaxBlocks = 64;
constexpr int kPrefillMaxBlocks = 8;
// Passes of at least this many tokens attend on simdgroup matrices (attn_qk,
// softmax_rows, attn_pv), their scores held in at most this many floats.
constexpr int kAttnMmMinRows = 16;
constexpr size_t kAttnScoresBudget = size_t{1} << 24;
constexpr int kInflightSteps = 2;

class DecoderMetal final : public Model {
 public:
  DecoderMetal(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options)
      : w_(std::move(weights)), c_(w_->config), max_context_(options.max_context),
        chunk_(std::max(1, std::min(options.prefill_chunk, options.max_context))),
        kv_f16_(kv_cache_f16(options, c_, true)) {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      LK_CHECK(device_ != nil, "metal: no Metal device");
      queue_ = [device_ newCommandQueue];
      library_ = compile_kernels(device_);
      validate();
      upload();
      allocate_state();
      zero_states();
    }
  }

  ~DecoderMetal() override { drain(); }

  const char* backend() const override { return "metal"; }
  const char* architecture() const override { return c_.architecture.c_str(); }
  int vocab_size() const override { return c_.vocab_size; }
  int max_context() const override { return max_context_; }
  const char* kv_cache() const override { return kv_f16_ ? "f16" : "f32"; }
  double kv_bytes() const { return kv_f16_ ? 2.0 : 4.0; }
  int position() const override { return position_; }
  size_t weight_bytes() const override { return w_->matrix_bytes(); }
  const std::vector<int32_t>& eos_token_ids() const override { return c_.eos_token_ids; }

  void reset() override {
    drain();
    position_ = 0;
    rope_offset_ = 0;
    injection_ = {};
    verify_n_ = 0;
    parity_ = 0;
    zero_states();
  }

  const float* prefill(const int32_t* tokens, int n) override {
    @autoreleasepool {
      drain();
      verify_n_ = 0;
      run_prefill(tokens, n);
      return static_cast<const float*>([logits_ contents]);
    }
  }

  const float* decode(int32_t token) override {
    @autoreleasepool {
      drain();
      verify_n_ = 0;
      LK_CHECK(token >= 0 && token < c_.vocab_size, "token id out of range");
      LK_CHECK(position_ + 1 <= max_context_, "context length exceeded");
      static_cast<int32_t*>([token_ contents])[0] = token;
      id<MTLCommandBuffer> cb = encode_decode_step(position_, 0);
      [cb commit];
      wait(cb);
      position_ += 1;
      return static_cast<const float*>([logits_ contents]);
    }
  }

  bool can_verify() const override { return true; }

  // One pass over the tokens with every row's logits. Keys and values rewind
  // by position. The pass reads each Gated DeltaNet layer's buffers for the
  // current position and writes the other ones, which then hold the
  // position after the pass, as a prefill's would; the state before it
  // survives, with the pass's conv inputs and recurrence inputs recorded,
  // for accept.
  const float* verify(const int32_t* tokens, int n) override {
    @autoreleasepool {
      drain();
      verify_n_ = 0;
      LK_CHECK(n > 0 && n <= chunk_, "verify takes 1 to prefill_chunk tokens");
      LK_CHECK(injection_.empty(), "verify cannot take an injection");
      LK_CHECK(position_ + n <= max_context_, "context length exceeded");
      for (int i = 0; i < n; ++i) LK_CHECK(tokens[i] >= 0 && tokens[i] < c_.vocab_size, "token id out of range");
      reserve_verify(n);
      std::memcpy([prompt_ contents], tokens, sizeof(int32_t) * static_cast<size_t>(n));
      struct Clear {
        bool& flag;
        ~Clear() { flag = false; }
      } clear{verifying_};
      verifying_ = true;
      active_cb_ = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [active_cb_ computeCommandEncoder];
      encode_forward(enc, prompt_, 0, n, position_, false, 0);
      // The last layer's residual leaves every row final-normed in h_.
      encode_qmm(enc, lm_head_, h_, n, verify_logits_);
      if (c_.final_logit_softcapping > 0.0f) {
        [enc setComputePipelineState:pipeline("softcap")];
        bind(enc, verify_logits_, 0, 0);
        bytes(enc, c_.final_logit_softcapping, 1);
        bytes(enc, ElementwiseParams{n * c_.vocab_size}, 2);
        encode_elementwise(enc, n * c_.vocab_size);
      }
      section(enc, "lm_head", profiler_ != nullptr ? traffic::linear(w_->lm_head, n) : 0.0);
      [enc endEncoding];
      id<MTLCommandBuffer> cb = active_cb_;
      [cb commit];
      wait(cb);
      verify_read_ = state_buffer(position_);
      verify_start_ = position_;
      verify_n_ = n;
      position_ += n;
      parity_ = static_cast<int>((1 - verify_read_ + static_cast<size_t>(position_)) % 2);
      return static_cast<const float*>([verify_logits_ contents]);
    }
  }

  // Keeps the first `kept` tokens of the last verify. When it rejects some,
  // each Gated DeltaNet layer's written buffers are rebuilt from the ones the
  // pass read and the kept tokens' recorded inputs, and then hold the kept
  // position's state.
  void accept(int kept) override {
    LK_CHECK(verify_n_ > 0 && kept >= 1 && kept <= verify_n_, "accept needs 1 to n tokens of the last verify");
    const int n = verify_n_;
    verify_n_ = 0;
    const size_t read = verify_read_, written = 1 - read;
    const bool replay = kept < n && c_.has_linear();
    if (replay || mtp_on_) {
      @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (replay) encode_replay(enc, kept, read, written);
        // The kept row's hidden state is pending for the MTP head; rows past
        // it are rewritten before a draft reads them.
        const size_t row = static_cast<size_t>(c_.hidden_size) * sizeof(float);
        if (mtp_on_) encode_copy(enc, mtp_h_, mtp_h_, c_.hidden_size, row * static_cast<size_t>(kept), 0);
        [enc endEncoding];
        [cb commit];
        wait(cb);
      }
    }
    position_ = verify_start_ + kept;
    parity_ = static_cast<int>((written + static_cast<size_t>(position_)) % 2);
  }

  bool has_mtp() const override { return layers_.size() > static_cast<size_t>(c_.num_layers); }

  void follow_mtp(bool on) override {
    if (on == mtp_on_) return;
    LK_CHECK(!on || w_->mtp.has_value(), c_.architecture + ": this checkpoint has no multi-token prediction head");
    LK_CHECK(!on || mtp_refused_.empty(), "the MTP head cannot run on Metal: " + mtp_refused_);
    LK_CHECK(!on || position_ == 0, "the MTP head must follow the model from an empty context");
    mtp_on_ = on;
  }

  // The whole block in one command buffer: each draft's step runs the head's
  // layer on one row at the next position, then its norm, lm_head and an
  // argmax on the device, whose token the next step embeds.
  std::vector<int32_t> mtp_draft(int32_t token, int count, std::vector<float>* logits) override {
    @autoreleasepool {
      drain();
      LK_CHECK(mtp_on_ && position_ > 0, "the MTP head drafts once it has followed a pass");
      LK_CHECK(token >= 0 && token < c_.vocab_size, "token id out of range");
      LK_CHECK(count >= 0 && position_ - 1 + count <= max_context_, "context length exceeded");
      if (logits != nullptr) logits->clear();
      if (count == 0) return {};
      const size_t vocab = static_cast<size_t>(c_.vocab_size);
      if (logits != nullptr && count > mtp_logits_rows_) {
        mtp_logits_ = shared_buffer(static_cast<size_t>(count) * vocab * sizeof(float));
        mtp_logits_rows_ = count;
      }
      static_cast<int32_t*>([token_ contents])[0] = token;
      const size_t layer = static_cast<size_t>(c_.num_layers);
      const GpuLayer& G = layers_[layer];
      active_cb_ = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [active_cb_ computeCommandEncoder];
      id<MTLComputePipelineState> pick = pipeline("argmax_vocab");
      for (int step = 0; step < count; ++step) {
        // The first draft starts from the pending hidden state, each later
        // one from the head's output (h_).
        encode_mtp_input(enc, token_, 0, step == 0 ? mtp_h_ : h_, 0, 1);
        encode_qmm(enc, mtp_fc_, mtp_in_, 1, x_);
        encode_rms_norm(enc, x_, G.attn_norm, h_, 1);
        encode_layer(enc, layer, 1, position_ - 1 + step, &mtp_final_norm_);
        encode_qmv(enc, lm_head_, h_, 0, logits_, 0);
        if (c_.final_logit_softcapping > 0.0f) {
          [enc setComputePipelineState:pipeline("softcap")];
          bind(enc, logits_, 0, 0);
          bytes(enc, c_.final_logit_softcapping, 1);
          bytes(enc, ElementwiseParams{c_.vocab_size}, 2);
          encode_elementwise(enc, c_.vocab_size);
        }
        section(enc, "lm_head", profiler_ != nullptr ? traffic::linear(w_->lm_head, 1) : 0.0);
        if (logits != nullptr) {
          encode_copy(enc, logits_, mtp_logits_, c_.vocab_size, 0, static_cast<size_t>(step) * vocab * sizeof(float));
        }
        [enc setComputePipelineState:pick];
        bind(enc, logits_, 0, 0);
        bind(enc, token_, 0, 1);
        bind(enc, history_, 0, 2);
        bytes(enc, ArgmaxParams{c_.vocab_size, step}, 3);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(1024, [pick maxTotalThreadsPerThreadgroup] / 32 * 32),
                                              1, 1)];
        section(enc, "sample", profiler_ != nullptr ? 4.0 * c_.vocab_size : 0.0);
      }
      [enc endEncoding];
      id<MTLCommandBuffer> cb = active_cb_;
      [cb commit];
      wait(cb);
      const int32_t* history = static_cast<const int32_t*>([history_ contents]);
      if (logits != nullptr) {
        const float* rows = static_cast<const float*>([mtp_logits_ contents]);
        logits->assign(rows, rows + static_cast<size_t>(count) * vocab);
      }
      return std::vector<int32_t>(history, history + count);
    }
  }

  std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_new, const SamplingParams& sampling,
                                bool stop_at_eos, GenerationTiming* timing, const TokenCallback& on_token) override {
    std::vector<int32_t> out;
    if (max_new <= 0) return out;
    validate_sampling(sampling);
    @autoreleasepool {
      drain();
      // Steps encoded during this call sample with these parameters.
      struct Restore {
        SamplingParams& target;
        ~Restore() { target = SamplingParams{}; }
      } restore{sampling_};
      sampling_ = sampling;
      const int prompt_len = static_cast<int>(prompt.size());
      LK_CHECK(position_ + prompt_len + max_new - 1 <= max_context_, "context length exceeded");
      verify_n_ = 0;
      speculation_stats_ = {};
      // Greedy generation with speculation verifies drafts; it waits for each
      // pass instead of pipelining steps.
      if (speculation_.max_draft > 0 && sampling.greedy()) {
        return generate_speculative(prompt, max_new, sampling, stop_at_eos, timing, on_token);
      }
      const auto& eos = c_.eos_token_ids;
      auto is_eos = [&](int32_t t) { return std::find(eos.begin(), eos.end(), t) != eos.end(); };
      const int32_t* history = static_cast<const int32_t*>([history_ contents]);

      const double start = now_seconds();
      run_prefill(prompt.data(), prompt_len);
      const double first = now_seconds();
      out.push_back(history[0]);
      std::vector<double> times = {first};

      const int base = position_;  // position of the first generated token
      std::deque<std::pair<id<MTLCommandBuffer>, int>> inflight;
      int next_step = 1;
      bool stopped = (stop_at_eos && is_eos(out[0])) || (on_token && !on_token(out[0]));
      while (!stopped && static_cast<int>(out.size()) < max_new) {
        while (next_step < max_new && static_cast<int>(inflight.size()) < kInflightSteps) {
          id<MTLCommandBuffer> cb = encode_decode_step(base + next_step - 1, next_step);
          [cb commit];
          inflight.emplace_back(cb, next_step);
          ++next_step;
        }
        auto [cb, step] = inflight.front();
        inflight.pop_front();
        wait(cb);
        times.push_back(now_seconds());
        out.push_back(history[step]);
        if (stop_at_eos && is_eos(out.back())) stopped = true;
        if (on_token && !on_token(out.back())) stopped = true;
      }
      for (auto& [cb, step] : inflight) wait(cb);
      // A discarded step has moved the MTP head's pending state on, so the
      // head stops following (drafting with it fails closed until reset).
      if (!inflight.empty()) mtp_on_ = false;
      // Steps launched past a stop token are discarded; only consumed tokens advance.
      position_ = base + static_cast<int>(out.size()) - 1;
      if (timing) {
        timing->prefill_seconds = first - start;
        timing->decode_seconds = times.back() - first;
        timing->token_times = std::move(times);
      }
    }
    return out;
  }

 protected:
  bool accepts_injection() const override { return true; }

 private:
  // ---- setup --------------------------------------------------------------

  void validate() {
    if (c_.has_linear()) {
      // The conv keeps its taps in registers, and each lane of the recurrence
      // at most 8 rows of a state column.
      LK_CHECK(c_.linear_conv <= 8, "metal: Gated DeltaNet is built for at most 8 conv taps");
      LK_CHECK(c_.linear_k_dim <= 256 && c_.linear_v_dim <= 1024,
               "metal: Gated DeltaNet is built for key heads up to 256 and value heads up to 1024 wide");
    }
    // Experts run only through the matrix-vector kernels, which also read
    // MXFP4; other matrices must be affine.
    auto check_any = [&](const Linear& l) {
      LK_CHECK(l.quant.quantized(), "metal: " + l.name + " is not quantized; use --backend cpu");
      LK_CHECK(l.cols % 32 == 0 && l.quant.group_size % 32 == 0,
               "metal: " + l.name + " needs cols and group_size divisible by 32");
    };
    auto check = [&](const Linear& l) {
      check_any(l);
      LK_CHECK(!l.quant.mxfp4(), "metal: " + l.name + " is mxfp4, which the Metal kernels implement only for "
               "experts so far; use --backend cpu");
    };
    check(w_->embed);
    check(w_->lm_head);
    if (c_.per_layer_input > 0) {
      check(w_->embed_per_layer);
      check(w_->per_layer_model_proj);
    }
    LK_CHECK(c_.num_experts <= 512, "metal: at most 512 experts");
    LK_CHECK(c_.n_group <= 32, "metal: at most 32 expert groups");
    if (c_.mla()) {
      // Four query heads per attention threadgroup over the latent, whose
      // width the kernels are instantiated for.
      const int width = c_.latent_width();
      LK_CHECK(c_.num_heads % 4 == 0, "metal: latent attention needs a multiple of 4 heads");
      LK_CHECK(width == 96 || width == 576, "metal: latent attention is built for kv_lora_rank + "
                                            "qk_rope_head_dim of 576 (or 96); this model has " + std::to_string(width));
    }
    auto check_layer = [&](const DecoderLayer& L, int i) {
      if (c_.is_linear(i)) {
        const bool grouped = L.in_proj_qkvz.rows > 0;
        for (const Linear* l : {&L.in_proj_qkvz, &L.in_proj_ba, &L.in_proj_qkv, &L.in_proj_z, &L.in_proj_b,
                                &L.in_proj_a}) {
          if ((l == &L.in_proj_qkvz || l == &L.in_proj_ba) == grouped) check(*l);
        }
        check(L.out_proj);
      } else {
        for (const Linear* l : {&L.q_proj, &L.o_proj}) check(*l);
      }
      if (c_.mla()) {
        if (c_.q_lora_rank > 0) check(L.q_a_proj);
        check(L.kv_a_proj);
        for (const auto* heads : {&L.embed_q, &L.unembed_out}) {
          for (const Linear& h : *heads) check(h);
        }
        if (L.embed_q.empty()) {
          check(L.kv_b_proj);
        }
      }
      if (c_.is_moe(i)) {
        if (L.router.quant.quantized()) {
          check(L.router);
        } else {
          LK_CHECK(L.router.dense_dtype == DType::F32 || L.router.dense_dtype == DType::BF16 ||
                       L.router.dense_dtype == DType::F16,
                   "metal: " + L.router.name + " has an unsupported dense dtype");
        }
        if (c_.shared_intermediate > 0) {
          for (const Linear* l : {&L.shared_gate, &L.shared_up, &L.shared_down}) check(*l);
        }
        if (c_.shared_expert_gate) check(L.shared_expert_gate);
        for (const auto* experts : {&L.expert_gate, &L.expert_up, &L.expert_down}) {
          for (const Linear& e : *experts) check_any(e);
        }
        const Linear &g = L.expert_gate[0], &u = L.expert_up[0];
        LK_CHECK(g.quant.bits == u.quant.bits && g.quant.group_size == u.quant.group_size &&
                     g.quant.mode == u.quant.mode && g.scale_dtype == u.scale_dtype,
                 "metal: expert gate and up projections must share a format");
      }
      if (!c_.is_moe(i) || c_.experts_beside_mlp) {
        for (const Linear* l : {&L.gate_proj, &L.up_proj, &L.down_proj}) check(*l);
      }
      if (c_.per_layer_input > 0) {
        check(L.per_layer_gate);
        check(L.per_layer_proj);
      }
      if (c_.mla() || c_.is_linear(i)) return;  // no k/v projections
      if (!c_.shares_kv(i)) check(L.k_proj);
      if (!c_.shares_kv(i) && !c_.k_eq_v(i)) check(L.v_proj);
      const int dim = c_.head_dim_of(i), dpl = dim / 32;
      LK_CHECK(dim % 32 == 0 && (dpl == 2 || dpl == 4 || dpl == 8 || dpl == 16),
               "metal: head_dim must be 64, 128, 256 or 512");
      const int group = c_.num_heads / c_.kv_heads_of(i);
      LK_CHECK(group >= 1 && group <= 8, "metal: heads per kv head must be between 1 and 8");
    };
    for (int i = 0; i < c_.num_layers; ++i) check_layer(w_->layers[static_cast<size_t>(i)], i);
    // An MTP head the kernels cannot read leaves the decoder running here and
    // refuses MTP drafting with the reason.
    if (w_->mtp) {
      try {
        check(w_->mtp->fc);
        check_layer(w_->mtp->layer, c_.num_layers);
      } catch (const Error& e) {
        // The reason is quoted inside another error, so without its prefix.
        const std::string reason = e.what(), prefix = "lokahi: ";
        mtp_refused_ = reason.rfind(prefix, 0) == 0 ? reason.substr(prefix.size()) : reason;
      }
    }
  }

  GpuLinear add_linear(WeightArena& arena, const Linear& l, std::vector<std::pair<GpuLinear*, std::array<size_t, 3>>>& fix,
                       GpuLinear* target) {
    GpuLinear g;
    g.rows = static_cast<int>(l.rows);
    g.cols = static_cast<int>(l.cols);
    g.bits = l.quant.bits;
    g.group = l.quant.group_size;
    g.bf16 = l.scale_dtype == DType::BF16;
    const size_t words = static_cast<size_t>(l.rows * l.words_per_row()) * 4;
    const size_t scales = static_cast<size_t>(l.rows * l.groups_per_row()) * 2;
    fix.push_back({target, {arena.add(l.words, words), arena.add(l.scales, scales), arena.add(l.biases, scales)}});
    return g;
  }

  // Copies a stack of expert matrices into contiguous host staging (MLX's
  // stacked tensors already are; per-expert modules are not) for the arena.
  void add_experts(WeightArena& arena, const std::vector<Linear>& experts, GpuExperts* target) {
    const Linear& first = experts[0];
    target->rows = static_cast<int>(first.rows);
    target->cols = static_cast<int>(first.cols);
    target->bits = first.quant.bits;
    target->group = first.quant.group_size;
    target->bf16 = first.scale_dtype == DType::BF16;
    target->mxfp4 = first.quant.mxfp4();
    const size_t words = static_cast<size_t>(first.rows * first.words_per_row()) * 4;
    // MXFP4 groups carry one exponent byte and no biases (a 4-byte stand-in).
    const size_t scales = static_cast<size_t>(first.rows * first.groups_per_row()) * (target->mxfp4 ? 1 : 2);
    std::array<size_t, 3> ids{};
    const std::array<size_t, 3> sizes = {words, scales, target->mxfp4 ? 0 : scales};
    for (int part = 0; part < 3; ++part) {
      std::vector<uint8_t>& stage = staged_.emplace_back(std::max<size_t>(4, sizes[part] * experts.size()));
      for (size_t e = 0; e < experts.size() && sizes[part] > 0; ++e) {
        const void* source = part == 0   ? static_cast<const void*>(experts[e].words)
                             : part == 1 ? (target->mxfp4 ? static_cast<const void*>(experts[e].exponents)
                                                          : static_cast<const void*>(experts[e].scales))
                                         : static_cast<const void*>(experts[e].biases);
        std::memcpy(stage.data() + e * sizes[part], source, sizes[part]);
      }
      ids[static_cast<size_t>(part)] = arena.add(stage.data(), stage.size());
    }
    expert_fix_.push_back({target, ids});
  }

  void upload() {
    WeightArena arena(device_);
    std::vector<std::pair<GpuLinear*, std::array<size_t, 3>>> linear_fix;
    std::vector<std::pair<BufRef*, size_t>> vector_fix;
    auto add_vector = [&](const std::vector<float>& v, BufRef* target) {
      vector_fix.push_back({target, arena.add(v.data(), v.size() * sizeof(float))});
    };

    embed_ = add_linear(arena, w_->embed, linear_fix, &embed_);
    if (w_->tied_lm_head) {
      tied_ = true;
    } else {
      lm_head_ = add_linear(arena, w_->lm_head, linear_fix, &lm_head_);
    }
    add_vector(w_->final_norm, &final_norm_);
    if (c_.per_layer_input > 0) {
      embed_per_layer_ = add_linear(arena, w_->embed_per_layer, linear_fix, &embed_per_layer_);
      per_layer_model_proj_ = add_linear(arena, w_->per_layer_model_proj, linear_fix, &per_layer_model_proj_);
      add_vector(w_->per_layer_proj_norm, &per_layer_proj_norm_);
    }
    for (int i = 0; i < c_.num_layers; ++i) {
      max_head_dim_ = std::max(max_head_dim_, c_.head_dim_of(i));
      max_kv_width_ = std::max(max_kv_width_, c_.kv_heads_of(i) * c_.head_dim_of(i));
      max_inter_ = std::max(max_inter_, c_.intermediate_of(i));
    }
    if (c_.mla()) {  // attention runs over latent-wide heads
      max_head_dim_ = std::max(max_head_dim_, c_.latent_width());
      max_kv_width_ = std::max(max_kv_width_, c_.latent_width());
    }
    // Bound in place of norms a family does not have; kernels never read it.
    placeholder_data_.assign(static_cast<size_t>(std::max({c_.hidden_size, max_head_dim_, c_.num_heads * max_head_dim_,
                                                            c_.per_layer_input})),
                             0.0f);
    add_vector(placeholder_data_, &placeholder_);

    for (const double f : c_.rope_inv_freq(true)) inv_freq_local_.push_back(static_cast<float>(f));
    for (const double f : c_.rope_inv_freq(false)) inv_freq_global_.push_back(static_cast<float>(f));
    add_vector(inv_freq_local_, &inv_freq_local_buf_);
    add_vector(inv_freq_global_, &inv_freq_global_buf_);

    std::vector<std::pair<BufRef*, BufRef*>> placeholders;
    auto add_optional = [&](const std::vector<float>& v, BufRef* target) {
      if (v.empty()) placeholders.push_back({target, &placeholder_});
      else add_vector(v, target);
    };
    // The MTP head's layer, when it runs here, follows the decoder's.
    const bool mtp = w_->mtp.has_value() && mtp_refused_.empty();
    layers_.resize(w_->layers.size() + (mtp ? 1 : 0));
    for (size_t i = 0; i < layers_.size(); ++i) {
      const DecoderLayer& L = i < w_->layers.size() ? w_->layers[i] : w_->mtp->layer;
      GpuLayer& G = layers_[i];
      const int layer = static_cast<int>(i);
      G.head_dim = c_.head_dim_of(layer);
      G.kv_heads = c_.kv_heads_of(layer);
      G.inter = c_.intermediate_of(layer);
      G.own_kv = !c_.shares_kv(layer);
      G.k_eq_v = c_.k_eq_v(layer);
      G.scalar = L.layer_scalar;
      G.linear = c_.is_linear(layer);
      if (G.linear) {
        G.grouped = L.in_proj_qkvz.rows > 0;
        if (G.grouped) {
          G.in_qkvz = add_linear(arena, L.in_proj_qkvz, linear_fix, &G.in_qkvz);
          G.in_ba = add_linear(arena, L.in_proj_ba, linear_fix, &G.in_ba);
        } else {
          G.in_qkv = add_linear(arena, L.in_proj_qkv, linear_fix, &G.in_qkv);
          G.in_z = add_linear(arena, L.in_proj_z, linear_fix, &G.in_z);
          G.in_b = add_linear(arena, L.in_proj_b, linear_fix, &G.in_b);
          G.in_a = add_linear(arena, L.in_proj_a, linear_fix, &G.in_a);
        }
        G.out_proj = add_linear(arena, L.out_proj, linear_fix, &G.out_proj);
      } else {
        G.q = add_linear(arena, L.q_proj, linear_fix, &G.q);
        if (G.own_kv && !c_.mla()) G.k = add_linear(arena, L.k_proj, linear_fix, &G.k);
        if (G.own_kv && !G.k_eq_v && !c_.mla()) G.v = add_linear(arena, L.v_proj, linear_fix, &G.v);
        G.o = add_linear(arena, L.o_proj, linear_fix, &G.o);
      }
      add_optional(L.conv, &G.conv_taps);
      add_optional(L.decay_rate, &G.decay_rate);
      add_optional(L.dt_bias, &G.dt_bias);
      add_optional(L.linear_norm, &G.linear_norm);
      if (c_.mla()) {
        if (c_.q_lora_rank > 0) G.q_a = add_linear(arena, L.q_a_proj, linear_fix, &G.q_a);
        G.kv_a = add_linear(arena, L.kv_a_proj, linear_fix, &G.kv_a);
        G.absorbed = !L.embed_q.empty();
        if (G.absorbed) {
          add_experts(arena, L.embed_q, &G.embed_q);
        } else {
          G.kv_b = add_linear(arena, L.kv_b_proj, linear_fix, &G.kv_b);
        }
        add_experts(arena, L.unembed_out, &G.unembed_out);
      }
      add_optional(L.q_a_norm, &G.q_a_norm);
      add_optional(L.kv_a_norm, &G.kv_a_norm);
      G.moe = c_.is_moe(layer);
      if (G.moe) {
        G.router_dense = !L.router.quant.quantized();
        if (G.router_dense) {
          G.dense_router.rows = static_cast<int>(L.router.rows);
          G.dense_router.cols = static_cast<int>(L.router.cols);
          G.dense_router.dtype = L.router.dense_dtype == DType::BF16 ? 1 : L.router.dense_dtype == DType::F16 ? 2 : 0;
          const size_t bytes = static_cast<size_t>(L.router.rows * L.router.cols) * dtype_size(L.router.dense_dtype);
          dense_fix_.push_back({&G.dense_router.data, arena.add(L.router.dense, bytes)});
        } else {
          G.router = add_linear(arena, L.router, linear_fix, &G.router);
        }
        if (c_.shared_intermediate > 0) {
          G.shared_gate = add_linear(arena, L.shared_gate, linear_fix, &G.shared_gate);
          G.shared_up = add_linear(arena, L.shared_up, linear_fix, &G.shared_up);
          G.shared_down = add_linear(arena, L.shared_down, linear_fix, &G.shared_down);
        }
        if (c_.shared_expert_gate) {
          G.shared_expert_gate = add_linear(arena, L.shared_expert_gate, linear_fix, &G.shared_expert_gate);
        }
        add_experts(arena, L.expert_gate, &G.expert_gate);
        add_experts(arena, L.expert_up, &G.expert_up);
        add_experts(arena, L.expert_down, &G.expert_down);
      }
      add_optional(L.router_bias, &G.router_bias);
      add_optional(L.router_correction, &G.router_correction);
      add_optional(L.expert_gate_bias, &G.expert_gate.bias);
      add_optional(L.expert_up_bias, &G.expert_up.bias);
      add_optional(L.expert_down_bias, &G.expert_down.bias);
      add_optional(L.o_bias, &G.o_bias);
      add_optional(L.sinks, &G.sinks);
      if (c_.per_layer_input > 0) {
        G.pl_gate = add_linear(arena, L.per_layer_gate, linear_fix, &G.pl_gate);
        G.pl_proj = add_linear(arena, L.per_layer_proj, linear_fix, &G.pl_proj);
      }
      add_optional(L.per_layer_norm, &G.pl_norm);
      add_optional(L.router_input_scale, &G.router_input_scale);
      add_optional(L.per_expert_scale, &G.per_expert_scale);
      add_optional(L.mlp_post_norm_1, &G.mlp_post_norm_1);
      add_optional(L.expert_norm, &G.expert_norm);
      add_optional(L.expert_post_norm, &G.expert_post_norm);
      if (!G.moe || c_.experts_beside_mlp) {
        G.gate = add_linear(arena, L.gate_proj, linear_fix, &G.gate);
        G.up = add_linear(arena, L.up_proj, linear_fix, &G.up);
        G.down = add_linear(arena, L.down_proj, linear_fix, &G.down);
      }
      add_vector(L.attn_norm, &G.attn_norm);
      add_vector(L.mlp_norm, &G.mlp_norm);
      add_optional(L.attn_post_norm, &G.attn_post_norm);
      add_optional(L.mlp_post_norm, &G.mlp_post_norm);
      add_optional(L.q_norm, &G.q_norm);
      add_optional(L.k_norm, &G.k_norm);
      add_optional(L.q_bias, &G.q_bias);
      add_optional(L.k_bias, &G.k_bias);
      add_optional(L.v_bias, &G.v_bias);
    }
    if (mtp) {
      mtp_fc_ = add_linear(arena, w_->mtp->fc, linear_fix, &mtp_fc_);
      add_vector(w_->mtp->embedding_norm, &mtp_embedding_norm_);
      add_vector(w_->mtp->hidden_norm, &mtp_hidden_norm_);
      add_vector(w_->mtp->final_norm, &mtp_final_norm_);
    }
    arena.finalize();
    staged_.clear();  // the arena has copied the staged expert stacks
    for (auto& [target, ids] : expert_fix_) {
      target->words = arena.get(ids[0]);
      target->scales = arena.get(ids[1]);
      target->biases = arena.get(ids[2]);
    }
    expert_fix_.clear();
    for (auto& [target, id] : dense_fix_) *target = arena.get(id);
    dense_fix_.clear();
    for (auto& [target, ids] : linear_fix) {
      target->words = arena.get(ids[0]);
      target->scales = arena.get(ids[1]);
      target->biases = arena.get(ids[2]);
    }
    for (auto& [target, id] : vector_fix) *target = arena.get(id);
    for (auto& [target, source] : placeholders) *target = *source;
    if (tied_) lm_head_ = embed_;
    device_weight_bytes_ = arena.bytes();
  }

  id<MTLBuffer> private_buffer(size_t bytes) {
    id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<size_t>(bytes, 16)
                                                options:MTLResourceStorageModePrivate];
    LK_CHECK(buffer != nil, "metal: allocation failed");
    return buffer;
  }

  id<MTLBuffer> shared_buffer(size_t bytes) {
    id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<size_t>(bytes, 16)
                                                options:MTLResourceStorageModeShared];
    LK_CHECK(buffer != nil, "metal: allocation failed");
    return buffer;
  }

  void allocate_state() {
    const size_t f = sizeof(float);
    const size_t H = static_cast<size_t>(c_.num_heads);
    const size_t D = static_cast<size_t>(max_head_dim_), hidden = static_cast<size_t>(c_.hidden_size);
    const size_t chunk = static_cast<size_t>(chunk_);
    for (size_t i = 0; i < layers_.size(); ++i) {
      GpuLayer& G = layers_[i];
      G.sliding = c_.layer_is_sliding[i];
      if (G.linear) {  // no keys or values: the conv history and the states, two of each
        const size_t history = static_cast<size_t>(c_.linear_conv - 1) * static_cast<size_t>(c_.linear_channels());
        const size_t state = static_cast<size_t>(c_.linear_v_heads) * static_cast<size_t>(c_.linear_k_dim) *
                             static_cast<size_t>(c_.linear_v_dim);
        for (int parity = 0; parity < 2; ++parity) {
          G.conv_history[static_cast<size_t>(parity)] = private_buffer(history * f);
          G.delta_state[static_cast<size_t>(parity)] = private_buffer(state * f);
        }
        continue;
      }
      if (c_.mla()) {  // one latent per position, whose prefix is also the value
        G.capacity = max_context_;
        G.k_cache = private_buffer(static_cast<size_t>(c_.latent_width()) * static_cast<size_t>(G.capacity) * f);
        G.v_cache = G.k_cache;
        continue;
      }
      if (!G.own_kv) {  // reads its source layer's caches, allocated earlier
        const GpuLayer& source = layers_[static_cast<size_t>(c_.kv_source(static_cast<int>(i)))];
        G.k_cache = source.k_cache;
        G.v_cache = source.v_cache;
        G.capacity = source.capacity;
        continue;
      }
      // Sliding layers keep window + chunk slots so a whole chunk can be
      // written before its queries read the preceding window.
      G.capacity = G.sliding ? std::min(max_context_, c_.sliding_window + chunk_) : max_context_;
      const size_t width = static_cast<size_t>(G.kv_heads) * static_cast<size_t>(G.head_dim);
      const size_t element = kv_f16_ ? sizeof(uint16_t) : sizeof(float);
      G.k_cache = private_buffer(width * static_cast<size_t>(G.capacity) * element);
      G.v_cache = private_buffer(width * static_cast<size_t>(G.capacity) * element);
    }
    if (has_mtp()) {
      mtp_h_ = private_buffer((chunk + 1) * hidden * f);
      mtp_e_ = private_buffer(chunk * hidden * f);
      mtp_in_ = private_buffer(chunk * 2 * hidden * f);
    }
    x_ = private_buffer(chunk * hidden * f);
    h_ = private_buffer(chunk * hidden * f);
    r_ = private_buffer(chunk * hidden * f);
    q_ = private_buffer(chunk * H * D * f);
    kt_ = private_buffer(chunk * static_cast<size_t>(max_kv_width_) * f);
    vt_ = private_buffer(chunk * static_cast<size_t>(max_kv_width_) * f);
    attn_ = private_buffer(chunk * H * D * f);
    const size_t widest = static_cast<size_t>(std::max(max_inter_, c_.shared_intermediate));
    gate_ = private_buffer(chunk * widest * f);
    up_ = private_buffer(chunk * widest * f);
    if (c_.shared_intermediate > 0) shared_out_ = private_buffer(chunk * hidden * f);
    if (c_.experts_beside_mlp) {
      router_in_ = private_buffer(chunk * hidden * f);
      moe_out_ = private_buffer(chunk * hidden * f);
    }
    if (c_.num_experts > 0) {
      const size_t slots = chunk * static_cast<size_t>(c_.experts_per_token);
      route_ = private_buffer(chunk * static_cast<size_t>(c_.num_experts) * f);
      chosen_ = shared_buffer(slots * sizeof(int32_t));  // shared: profiling reads which experts ran
      expert_weights_ = private_buffer(slots * f);
      expert_hidden_ = private_buffer(slots * static_cast<size_t>(c_.moe_intermediate) * f);
      expert_out_ = private_buffer(slots * hidden * f);
    }
    if (c_.per_layer_input > 0) {
      const size_t width = static_cast<size_t>(c_.num_layers) * static_cast<size_t>(c_.per_layer_input);
      per_layer_ = private_buffer(chunk * width * f);
      per_layer_lookup_ = private_buffer(chunk * width * f);
      per_layer_proj_ = private_buffer(chunk * width * f);
      per_layer_gate_ = private_buffer(chunk * static_cast<size_t>(c_.per_layer_input) * f);
    }
    if (c_.has_linear()) {
      const size_t values = static_cast<size_t>(c_.linear_v_heads) * static_cast<size_t>(c_.linear_v_dim);
      const size_t heads = static_cast<size_t>(c_.linear_v_heads);
      mixed_ = private_buffer(chunk * static_cast<size_t>(c_.linear_channels()) * f);
      lin_z_ = private_buffer(chunk * values * f);
      lin_out_ = private_buffer(chunk * values * f);
      lin_b_ = private_buffer(chunk * heads * f);
      lin_a_ = private_buffer(chunk * heads * f);
      grouped_ = private_buffer(chunk * (static_cast<size_t>(c_.linear_channels()) + values) * f);
      grouped_ba_ = private_buffer(chunk * 2 * heads * f);
    }
    if (c_.attn_output_gate) {
      q_gate_ = private_buffer(chunk * H * D * 2 * f);
      out_gate_ = private_buffer(chunk * H * D * f);
    }
    if (c_.shared_expert_gate) shared_scale_ = private_buffer(chunk * f);
    if (c_.mla()) {
      const size_t width = static_cast<size_t>(c_.latent_width());
      latent_q_ = private_buffer(chunk * H * width * f);
      latent_out_ = private_buffer(chunk * H * width * f);
      if (c_.q_lora_rank > 0) q_latent_ = private_buffer(chunk * static_cast<size_t>(c_.q_lora_rank) * f);
      // Each (token, head) pair's head, the expert index of the per-head
      // gathered kernels.
      head_index_ = shared_buffer(chunk * H * sizeof(int32_t));
      int32_t* heads = static_cast<int32_t*>([head_index_ contents]);
      for (size_t i = 0; i < chunk * H; ++i) heads[i] = static_cast<int32_t>(i % H);
    }
    const size_t partials = std::max(H * kDecodeMaxBlocks, chunk * H * kPrefillMaxBlocks);
    part_o_ = private_buffer(partials * D * f);
    part_ml_ = private_buffer(partials * 2 * f);
    logits_ = shared_buffer(static_cast<size_t>(c_.vocab_size) * f);
    token_ = shared_buffer(sizeof(int32_t));
    history_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
    prompt_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
    // An injection's row index and last key per prompt token; always bound.
    inject_index_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
    if (c_.per_layer_input > 0) per_layer_tokens_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
    last_key_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
    if (!c_.rope_axes.empty()) {
      // M-RoPE: each prompt token's three positions, and each frequency's axis.
      rope_pos_ = shared_buffer(static_cast<size_t>(max_context_) * 3 * sizeof(int32_t));
      rope_axes_ = shared_buffer(c_.rope_axes.size() * sizeof(int32_t));
      int32_t* axes = static_cast<int32_t*>([rope_axes_ contents]);
      for (size_t i = 0; i < c_.rope_axes.size(); ++i) axes[i] = c_.rope_axes[i];
    }
  }

  // Zeroes every conv history and state (private buffers start undefined).
  void zero_states() {
    if (!c_.has_linear()) return;
    const int history = (c_.linear_conv - 1) * c_.linear_channels();
    const int state = c_.linear_v_heads * c_.linear_k_dim * c_.linear_v_dim;
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline("zero_floats")];
    for (const GpuLayer& G : layers_) {
      if (!G.linear) continue;
      for (size_t parity = 0; parity < 2; ++parity) {
        for (auto [buffer, count] : {std::pair{G.conv_history[parity], history}, std::pair{G.delta_state[parity], state}}) {
          bind(enc, buffer, 0, 0);
          bytes(enc, ElementwiseParams{count}, 1);
          encode_elementwise(enc, count);
        }
      }
    }
    [enc endEncoding];
    [cb commit];
    wait(cb);
  }

  id<MTLComputePipelineState> pipeline(const std::string& name) {
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) return it->second;
    NSError* error = nil;
    id<MTLFunction> function = [library_ newFunctionWithName:@(name.c_str())];
    LK_CHECK(function != nil, "metal: missing kernel " + name);
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:function error:&error];
    LK_CHECK(pso != nil, "metal: pipeline " + name + ": " + ns_error(error));
    pipelines_[name] = pso;
    return pso;
  }

  id<MTLComputePipelineState> quant_pipeline(const std::string& name, const GpuLinear& l) {
    const std::string key = name + "/" + std::to_string(l.bits) + "/" + std::to_string(l.group) + "/" +
                            (l.bf16 ? "bf16" : "f16") + (l.mxfp4 ? "/mxfp4" : "");
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    int bits = l.bits, group = l.group;
    bool bf16 = l.bf16, mxfp4 = l.mxfp4;
    [values setConstantValue:&bits type:MTLDataTypeInt atIndex:0];
    [values setConstantValue:&group type:MTLDataTypeInt atIndex:1];
    [values setConstantValue:&bf16 type:MTLDataTypeBool atIndex:2];
    [values setConstantValue:&mxfp4 type:MTLDataTypeBool atIndex:3];
    NSError* error = nil;
    id<MTLFunction> function = [library_ newFunctionWithName:@(name.c_str()) constantValues:values error:&error];
    LK_CHECK(function != nil, "metal: kernel " + key + ": " + ns_error(error));
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:function error:&error];
    LK_CHECK(pso != nil, "metal: pipeline " + key + ": " + ns_error(error));
    pipelines_[key] = pso;
    return pso;
  }

  // ---- encoding helpers ---------------------------------------------------

  static void bind(id<MTLComputeCommandEncoder> enc, const BufRef& ref, NSUInteger index) {
    [enc setBuffer:ref.buffer offset:ref.offset atIndex:index];
  }
  static void bind(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buffer, size_t offset, NSUInteger index) {
    [enc setBuffer:buffer offset:offset atIndex:index];
  }
  template <typename T>
  static void bytes(id<MTLComputeCommandEncoder> enc, const T& value, NSUInteger index) {
    [enc setBytes:&value length:sizeof(T) atIndex:index];
  }
  static void bind_weights(id<MTLComputeCommandEncoder> enc, const GpuLinear& l, NSUInteger first) {
    bind(enc, l.words, first);
    bind(enc, l.scales, first + 1);
    bind(enc, l.biases, first + 2);
  }

  static NSUInteger qmv_groups(int rows) { return static_cast<NSUInteger>((rows + 7) / 8); }

  void encode_qmv(id<MTLComputeCommandEncoder> enc, const GpuLinear& l, id<MTLBuffer> x, size_t x_off,
                  id<MTLBuffer> y, size_t y_off) {
    [enc setComputePipelineState:quant_pipeline("qmv", l)];
    bind_weights(enc, l, 0);
    bind(enc, x, x_off, 3);
    bind(enc, y, y_off, 4);
    bytes(enc, QmvParams{l.rows, l.cols}, 5);
    [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(l.rows), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }

  void encode_qmm(id<MTLComputeCommandEncoder> enc, const GpuLinear& l, id<MTLBuffer> x, int n,
                  id<MTLBuffer> y) {
    if (n == 1) {
      encode_qmv(enc, l, x, 0, y, 0);
      return;
    }
    if (n <= kQmvNMax && !l.mxfp4) {  // a short pass: read the weights once for all n tokens
      [enc setComputePipelineState:quant_pipeline("qmv_n", l)];
      bind_weights(enc, l, 0);
      bind(enc, x, 0, 3);
      bind(enc, y, 0, 4);
      bytes(enc, QmvNParams{l.rows, l.cols, n}, 5);
      [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(l.rows), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      return;
    }
    [enc setComputePipelineState:quant_pipeline("qmm", l)];
    bind_weights(enc, l, 0);
    bind(enc, x, 0, 3);
    bind(enc, y, 0, 4);
    bytes(enc, QmmParams{n, l.rows, l.cols}, 5);
    // 64x64 tiles, as QMM_TILE_M and QMM_TILE_N in decoder_kernels.metal.
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((l.rows + 63) / 64),
                                          static_cast<NSUInteger>((n + 63) / 64), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }

  void encode_elementwise(id<MTLComputeCommandEncoder> enc, int count) {
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  NSUInteger norm_threads(id<MTLComputePipelineState> pso) const {
    NSUInteger wanted = c_.hidden_size >= 4096 ? 1024 : 256;
    return std::min(wanted, [pso maxTotalThreadsPerThreadgroup] / 32 * 32);
  }

  void encode_rms_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, const BufRef& weight,
                       id<MTLBuffer> out, int n) {
    encode_rms_norm(enc, in, weight, out, n, NormParams{c_.hidden_size, c_.rms_norm_eps, c_.norm_offset});
  }

  // Rows of params.cols values (in place when in is out).
  void encode_rms_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, const BufRef& weight,
                       id<MTLBuffer> out, int n, const NormParams& params) {
    id<MTLComputePipelineState> pso = pipeline("rms_norm_rows");
    [enc setComputePipelineState:pso];
    bind(enc, in, 0, 0);
    bind(enc, weight, 1);
    bind(enc, out, 0, 2);
    bytes(enc, params, 3);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(norm_threads(pso), 1, 1)];
  }

  void encode_residual_norm(id<MTLComputeCommandEncoder> enc, const BufRef& w_post, const BufRef* w_next, int n,
                            float scale = 1.0f) {
    id<MTLComputePipelineState> pso = pipeline("residual_norm_rows");
    [enc setComputePipelineState:pso];
    bind(enc, x_, 0, 0);
    bind(enc, r_, 0, 1);
    bind(enc, w_post, 2);
    bind(enc, w_next ? *w_next : w_post, 3);
    bind(enc, h_, 0, 4);
    bytes(enc, ResidualNormParams{c_.hidden_size, c_.rms_norm_eps, c_.norm_offset, c_.sandwich_norms ? 1 : 0,
                                  w_next ? 1 : 0, scale},
          5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(norm_threads(pso), 1, 1)];
  }

  // into += norm(rows) * (offset + weight), row by row.
  void encode_add_normed(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> into, id<MTLBuffer> rows, const BufRef& weight,
                         int n) {
    id<MTLComputePipelineState> pso = pipeline("residual_norm_rows");
    [enc setComputePipelineState:pso];
    bind(enc, into, 0, 0);
    bind(enc, rows, 0, 1);
    bind(enc, weight, 2);
    bind(enc, weight, 3);
    bind(enc, h_, 0, 4);  // unread without a next norm
    bytes(enc, ResidualNormParams{c_.hidden_size, c_.rms_norm_eps, c_.norm_offset, 1, 0, 1.0f}, 5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(norm_threads(pso), 1, 1)];
  }

  // q_proj into q_ (or, with an output gate, into q_gate_ and then split into
  // q_ and out_gate_), and k and v into kt_ and vt_.
  void encode_qkv(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    encode_qkv_into(enc, G, n, c_.attn_output_gate ? q_gate_ : q_);
    if (!c_.attn_output_gate) return;
    [enc setComputePipelineState:pipeline("split_query_gate")];
    bind(enc, q_gate_, 0, 0);
    bind(enc, q_, 0, 1);
    bind(enc, out_gate_, 0, 2);
    const int count = n * c_.num_heads * G.head_dim;
    bytes(enc, SplitGateParams{G.head_dim, count}, 3);
    encode_elementwise(enc, count);
  }

  void encode_qkv_into(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, id<MTLBuffer> q_out) {
    if (!G.own_kv || G.k_eq_v) {  // queries only, or queries and the keys that double as values
      encode_qmm(enc, G.q, h_, n, q_out);
      if (G.own_kv) encode_qmm(enc, G.k, h_, n, kt_);
      return;
    }
    if (n == 1 && G.q.same_format(G.k) && G.q.same_format(G.v)) {
      [enc setComputePipelineState:quant_pipeline("qmv_qkv", G.q)];
      bind_weights(enc, G.q, 0);
      bind_weights(enc, G.k, 3);
      bind_weights(enc, G.v, 6);
      bind(enc, h_, 0, 9);
      bind(enc, q_out, 0, 10);
      bind(enc, kt_, 0, 11);
      bind(enc, vt_, 0, 12);
      bytes(enc, QkvParams{G.q.rows, G.k.rows, G.q.cols}, 13);
      [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(G.q.rows + 2 * G.k.rows), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      return;
    }
    encode_qmm(enc, G.q, h_, n, q_out);
    encode_qmm(enc, G.k, h_, n, kt_);
    encode_qmm(enc, G.v, h_, n, vt_);
  }

  // x[i] *= sigmoid(scales[i / cols]) over `count` values.
  void encode_mul_sigmoid(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> scales, int cols, int count) {
    [enc setComputePipelineState:pipeline("mul_sigmoid")];
    bind(enc, x, 0, 0);
    bind(enc, scales, 0, 1);
    bytes(enc, SigmoidScaleParams{cols, count}, 2);
    encode_elementwise(enc, count);
  }

  // ---- Gated DeltaNet -----------------------------------------------------

  // The input projections into mixed_ ([q | k | v] channels), lin_z_, and the
  // raw b and a in lin_b_ and lin_a_; Qwen3-Next's grouped rows are split.
  void encode_gdn_projections(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    if (!G.grouped) {
      encode_qmm(enc, G.in_qkv, h_, n, mixed_);
      encode_qmm(enc, G.in_z, h_, n, lin_z_);
      encode_qmm(enc, G.in_b, h_, n, lin_b_);
      encode_qmm(enc, G.in_a, h_, n, lin_a_);
      return;
    }
    encode_qmm(enc, G.in_qkvz, h_, n, grouped_);
    encode_qmm(enc, G.in_ba, h_, n, grouped_ba_);
    const int hk = c_.linear_k_heads, ratio = c_.linear_v_heads / hk;
    const int per_head = 2 * c_.linear_k_dim + 2 * ratio * c_.linear_v_dim + 2 * ratio;
    [enc setComputePipelineState:pipeline("gdn_split")];
    bind(enc, grouped_, 0, 0);
    bind(enc, grouped_ba_, 0, 1);
    bind(enc, mixed_, 0, 2);
    bind(enc, lin_z_, 0, 3);
    bind(enc, lin_b_, 0, 4);
    bind(enc, lin_a_, 0, 5);
    bytes(enc, GdnSplitParams{hk, c_.linear_k_dim, ratio, c_.linear_v_dim, n}, 6);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(hk * per_head), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  // The buffer of each Gated DeltaNet pair holding the conv history and
  // state at position p.
  size_t state_buffer(int p) const { return static_cast<size_t>((p + parity_) % 2); }
  // The buffer a pass from `start` writes: the one for its end position, or
  // during a verify pass always the other one.
  size_t written_buffer(int start, int n) const {
    return verifying_ ? 1 - state_buffer(start) : state_buffer(start + n);
  }

  // The conv (history from the start position's buffer to the written one)
  // and SiLU, the q/k L2 norms, and beta and the decay.
  void encode_gdn_conv(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    const int channels = c_.linear_channels();
    [enc setComputePipelineState:pipeline("gdn_conv")];
    bind(enc, mixed_, 0, 0);
    bind(enc, G.conv_taps, 1);
    bind(enc, G.conv_history[state_buffer(start)], 0, 2);
    bind(enc, G.conv_history[written_buffer(start, n)], 0, 3);
    bytes(enc, GdnConvParams{channels, c_.linear_conv, n}, 4);
    encode_elementwise(enc, channels);
    [enc setComputePipelineState:pipeline("gdn_qk_norm")];
    bind(enc, mixed_, 0, 0);
    bytes(enc, GdnNormParams{c_.linear_k_heads, c_.linear_k_dim, channels,
                             1.0f / std::sqrt(static_cast<float>(c_.linear_k_dim))},
          1);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(2 * c_.linear_k_heads), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(c_.linear_k_dim), 1, 1)];
    [enc setComputePipelineState:pipeline("gdn_gates")];
    bind(enc, lin_b_, 0, 0);
    bind(enc, lin_a_, 0, 1);
    bind(enc, G.decay_rate, 2);
    bind(enc, G.dt_bias, 3);
    const int count = n * c_.linear_v_heads;
    bytes(enc, GdnGateParams{c_.linear_v_heads, count}, 4);
    encode_elementwise(enc, count);
  }

  void encode_gdn_recurrence(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    encode_delta_rule(enc, mixed_, lin_b_, lin_a_, G.delta_state[state_buffer(start)],
                      G.delta_state[written_buffer(start, n)], n);
  }

  // The gated delta rule over n tokens' rows (q, k, v; normed), betas and
  // decays, from state_in into state_out; the outputs go to lin_out_.
  void encode_delta_rule(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> rows, id<MTLBuffer> beta,
                         id<MTLBuffer> decay, id<MTLBuffer> state_in, id<MTLBuffer> state_out, int n) {
    [enc setComputePipelineState:pipeline("gdn_recurrence")];
    bind(enc, rows, 0, 0);
    bind(enc, beta, 0, 1);
    bind(enc, decay, 0, 2);
    bind(enc, state_in, 0, 3);
    bind(enc, state_out, 0, 4);
    bind(enc, lin_out_, 0, 5);
    bytes(enc, GdnRecurrenceParams{c_.linear_k_heads, c_.linear_v_heads, c_.linear_k_dim, c_.linear_v_dim,
                                   c_.linear_channels(), n},
          6);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((c_.linear_v_dim + 3) / 4),
                                          static_cast<NSUInteger>(c_.linear_v_heads), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }

  // dst[0, count) = src[0, count), from byte offsets into each.
  void encode_copy(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, id<MTLBuffer> dst, int count,
                   size_t src_offset = 0, size_t dst_offset = 0) {
    [enc setComputePipelineState:pipeline("copy_floats")];
    bind(enc, src, src_offset, 0);
    bind(enc, dst, dst_offset, 1);
    bytes(enc, ElementwiseParams{count}, 2);
    encode_elementwise(enc, count);
  }

  // ---- multi-token prediction ---------------------------------------------

  // mtp_in_'s `rows` rows: [norm_e(embedding) | norm_h(hidden)] for the
  // tokens at tokens + tokens_offset and the hidden states at `hidden` +
  // hidden_offset.
  void encode_mtp_input(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> tokens, size_t tokens_offset,
                        id<MTLBuffer> hidden, size_t hidden_offset, int rows) {
    [enc setComputePipelineState:quant_pipeline("embed_rows", embed_)];
    bind_weights(enc, embed_, 0);
    bind(enc, tokens, tokens_offset, 3);
    bind(enc, mtp_e_, 0, 4);
    bytes(enc, EmbedParams{c_.hidden_size, rows, c_.embed_scale}, 5);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(c_.hidden_size), static_cast<NSUInteger>(rows), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    id<MTLComputePipelineState> pso = pipeline("mtp_concat_norms");
    [enc setComputePipelineState:pso];
    bind(enc, mtp_e_, 0, 0);
    bind(enc, hidden, hidden_offset, 1);
    bind(enc, mtp_embedding_norm_, 2);
    bind(enc, mtp_hidden_norm_, 3);
    bind(enc, mtp_in_, 0, 4);
    bytes(enc, NormParams{c_.hidden_size, c_.rms_norm_eps, c_.norm_offset}, 5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(norm_threads(pso), 1, 1)];
  }

  // The head follows the pass (DecoderCpu::mtp_follow): its rows for
  // positions [start - 1, start + n - 1) pair each token with the hidden
  // state before it, and only their keys and values are computed; the
  // pass's last hidden state is left pending in mtp_h_'s row 0.
  void encode_mtp_follow(__strong id<MTLComputeCommandEncoder>& enc, id<MTLBuffer> tokens, size_t tokens_offset,
                         int n, int start) {
    const GpuLayer& G = layers_[static_cast<size_t>(c_.num_layers)];
    const DecoderWeights::Mtp& M = *w_->mtp;
    const int hidden = c_.hidden_size;
    const size_t row = static_cast<size_t>(hidden) * sizeof(float);
    encode_copy(enc, h_, mtp_h_, n * hidden, 0, row);
    const int first = start == 0 ? 1 : 0;  // position 0's token follows no hidden state
    const int rows = n - first;
    if (rows > 0) {
      encode_mtp_input(enc, tokens, tokens_offset + sizeof(int32_t) * static_cast<size_t>(first), mtp_h_,
                       row * static_cast<size_t>(first), rows);
      encode_qmm(enc, mtp_fc_, mtp_in_, rows, x_);
      encode_rms_norm(enc, x_, G.attn_norm, r_, rows);
      encode_qmm(enc, G.k, r_, rows, kt_);
      encode_qmm(enc, G.v, r_, rows, vt_);
      encode_rope(enc, G, rows, start - 1 + first, true);
    }
    encode_copy(enc, mtp_h_, mtp_h_, hidden, row * static_cast<size_t>(n), 0);
    section(enc, "mtp",
            profiler_ != nullptr && rows > 0
                ? traffic::linear(M.fc, rows) + traffic::linear(M.layer.k_proj, rows) +
                      traffic::linear(M.layer.v_proj, rows) + traffic::embed(w_->embed, rows)
                : 0.0);
  }

  // Each Gated DeltaNet layer's conv history and state after the first
  // `kept` tokens of the last verify, from the buffers it read into the ones
  // it wrote.
  void encode_replay(id<MTLComputeCommandEncoder> enc, int kept, size_t read, size_t written) {
    const int channels = c_.linear_channels(), history = c_.linear_conv - 1;
    for (size_t l = 0; l < layers_.size(); ++l) {
      const GpuLayer& G = layers_[l];
      if (!G.linear) continue;
      const VerifyRecord& record = records_[l];
      if (history > 0) {
        [enc setComputePipelineState:pipeline("gdn_history_replay")];
        bind(enc, G.conv_history[read], 0, 0);
        bind(enc, record.raw, 0, 1);
        bind(enc, G.conv_history[written], 0, 2);
        bytes(enc, GdnHistoryParams{channels, c_.linear_conv, kept}, 3);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(channels), static_cast<NSUInteger>(history), 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      encode_delta_rule(enc, record.rows, record.beta, record.decay, G.delta_state[read], G.delta_state[written],
                        kept);
    }
  }

  // Room for a verify pass of n tokens: every row's logits, and each Gated
  // DeltaNet layer's records.
  void reserve_verify(int n) {
    if (n <= verify_capacity_) return;
    const size_t f = sizeof(float), rows = static_cast<size_t>(n);
    verify_logits_ = shared_buffer(rows * static_cast<size_t>(c_.vocab_size) * f);
    records_.assign(layers_.size(), {});
    for (size_t l = 0; l < layers_.size(); ++l) {
      if (!layers_[l].linear) continue;
      const size_t channels = rows * static_cast<size_t>(c_.linear_channels()) * f;
      const size_t gates = rows * static_cast<size_t>(c_.linear_v_heads) * f;
      records_[l] = {private_buffer(channels), private_buffer(channels), private_buffer(gates), private_buffer(gates)};
    }
    verify_capacity_ = n;
  }

  // The gated norm per value head, then out_proj into r_.
  void encode_gdn_out(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    id<MTLComputePipelineState> pso = pipeline("gated_rms_norm_rows");
    [enc setComputePipelineState:pso];
    bind(enc, lin_out_, 0, 0);
    bind(enc, G.linear_norm, 1);
    bind(enc, lin_z_, 0, 2);
    bytes(enc, NormParams{c_.linear_v_dim, c_.rms_norm_eps, 0.0f}, 3);
    const NSUInteger threads = std::min<NSUInteger>(256, static_cast<NSUInteger>((c_.linear_v_dim + 31) / 32 * 32));
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n * c_.linear_v_heads), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    encode_qmm(enc, G.out_proj, lin_out_, n, r_);
  }

  // With kv_only, only the keys and values are normed, rotated and stored.
  void encode_rope(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start, bool kv_only = false) {
    [enc setComputePipelineState:pipeline(kv_f16_ ? "qk_norm_rope_store_h" : "qk_norm_rope_store")];
    bind(enc, q_, 0, 0);
    bind(enc, kt_, 0, 1);
    bind(enc, vt_, 0, 2);
    bind(enc, G.k_cache, 0, 3);
    bind(enc, G.v_cache, 0, 4);
    bind(enc, G.q_norm, 5);
    bind(enc, G.k_norm, 6);
    bind(enc, G.sliding ? inv_freq_local_buf_ : inv_freq_global_buf_, 7);
    // An M-RoPE prefill with images reads the plan's positions; every other
    // pass offsets the KV positions by rope_offset_.
    const bool given = !plan_.rope.empty();
    bytes(enc, RopeParams{c_.num_heads, G.kv_heads, G.head_dim, start, G.capacity, c_.rms_norm_eps,
                          c_.norm_offset, c_.qk_norm ? 1 : 0, c_.qkv_bias ? 1 : 0, c_.v_norm ? 1 : 0,
                          c_.rotary_dim > 0 ? c_.rotary_dim : G.head_dim, rope_axes_ != nil ? 1 : 0, given ? 1 : 0,
                          rope_offset_, kv_only ? c_.num_heads : 0},
          8);
    bind(enc, G.q_bias, 9);
    bind(enc, G.k_bias, 10);
    bind(enc, G.v_bias, 11);
    if (given) {
      bind(enc, rope_pos_, sizeof(int32_t) * 3 * static_cast<size_t>(start - plan_.base), 12);
    } else {
      bind(enc, placeholder_, 12);
    }
    if (rope_axes_ != nil) {
      bind(enc, rope_axes_, 0, 13);
    } else {
      bind(enc, placeholder_, 13);
    }
    if (G.k_eq_v) bind(enc, kt_, 0, 2);  // values are the raw keys, before k_norm
    // KV-shared layers only rotate their queries.
    const int heads = kv_only ? G.kv_heads : c_.num_heads + (G.own_kv ? G.kv_heads : 0);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(heads), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(G.head_dim / 2), 1, 1)];
  }

  void encode_attention(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    const int window = G.sliding ? c_.sliding_window : (1 << 30);
    // Image blocks let a query also see up to plan_.reach later keys.
    const bool blocks = !plan_.empty() && plan_.reach > 0 && c_.image_blocks_apply(G.sliding);
    if (n >= kAttnMmMinRows && G.head_dim % 64 == 0 && !c_.attention_sinks &&
        encode_attention_mm(enc, G, n, start, window, blocks)) {
      return;
    }
    const int keys_max = std::min(window + (blocks ? plan_.reach : 0), start + n);  // the most keys a query sees
    const int max_blocks = n == 1 ? kDecodeMaxBlocks : kPrefillMaxBlocks;
    const int min_block = n == 1 ? 32 : 64;
    const int block = std::max(min_block, (keys_max + max_blocks - 1) / max_blocks);
    const int n_blocks = (keys_max + block - 1) / block;
    const int group = c_.num_heads / G.kv_heads;
    const std::string name =
        "attn_partial_g" + std::to_string(group) + "_d" + std::to_string(G.head_dim / 32) + (kv_f16_ ? "_h" : "");
    [enc setComputePipelineState:pipeline(name)];
    bind(enc, q_, 0, 0);
    bind(enc, G.k_cache, 0, 1);
    bind(enc, G.v_cache, 0, 2);
    bind(enc, part_o_, 0, 3);
    bind(enc, part_ml_, 0, 4);
    bytes(enc,
          AttnParams{c_.num_heads, G.kv_heads, G.head_dim, start, window, G.capacity, block, n_blocks,
                     c_.attn_scale, c_.attn_logit_softcapping, blocks ? 1 : 0, 1},
          5);
    bind(enc, last_key_, blocks ? sizeof(int32_t) * static_cast<size_t>(start - plan_.base) : 0, 6);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_blocks),
                                          static_cast<NSUInteger>(G.kv_heads), static_cast<NSUInteger>(n))
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc setComputePipelineState:pipeline("attn_reduce")];
    bind(enc, part_o_, 0, 0);
    bind(enc, part_ml_, 0, 1);
    bind(enc, attn_, 0, 2);
    bytes(enc, ReduceParams{c_.num_heads, G.head_dim, n_blocks, c_.attention_sinks ? 1 : 0}, 3);
    bind(enc, G.sinks, 4);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(c_.num_heads), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(G.head_dim), 1, 1)];
  }

  // A long pass's attention as S = Q K^T, softmax, P V on simdgroup matrices,
  // over query rows in sub-passes whose scores fit kAttnScoresBudget; false
  // (leaving attn_partial to it) when 64 rows would not.
  bool encode_attention_mm(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start, int window,
                           bool blocks) {
    const int H = c_.num_heads, D = G.head_dim;
    const int key_base = std::max(0, start - window + 1);  // the earliest key any row of the pass sees
    const int keys = start + n - key_base;
    const size_t per_row = static_cast<size_t>(H) * static_cast<size_t>(keys);
    const int fit = static_cast<int>(std::min<size_t>(kAttnScoresBudget / per_row, static_cast<size_t>(n)));
    const int sub = fit >= n ? n : fit / 64 * 64;
    if (sub < std::min(n, 64)) return false;
    const size_t scores_bytes = per_row * static_cast<size_t>(sub) * sizeof(float);
    if (attn_scores_bytes_ < scores_bytes) {
      attn_scores_ = private_buffer(scores_bytes);
      attn_scores_bytes_ = scores_bytes;
    }
    const std::string kv = kv_f16_ ? "_h" : "";
    for (int row0 = 0; row0 < n; row0 += sub) {
      const int rows = std::min(sub, n - row0);
      const AttnMmParams params{H, G.kv_heads, D, start, row0, rows, key_base, keys, window, G.capacity,
                                c_.attn_scale, c_.attn_logit_softcapping, blocks ? 1 : 0};
      const NSUInteger row_tiles = static_cast<NSUInteger>((rows + 63) / 64);
      [enc setComputePipelineState:pipeline("attn_qk" + kv)];
      bind(enc, q_, 0, 0);
      bind(enc, G.k_cache, 0, 1);
      bind(enc, attn_scores_, 0, 2);
      bytes(enc, params, 3);
      bind(enc, last_key_, blocks ? sizeof(int32_t) * static_cast<size_t>(start - plan_.base) : 0, 4);
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((keys + 63) / 64), row_tiles,
                                            static_cast<NSUInteger>(H))
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc setComputePipelineState:pipeline("softmax_rows")];
      bind(enc, attn_scores_, 0, 0);
      bytes(enc, ElementwiseParams{keys}, 1);
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(H * rows), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc setComputePipelineState:pipeline("attn_pv" + kv)];
      bind(enc, attn_scores_, 0, 0);
      bind(enc, G.v_cache, 0, 1);
      bind(enc, attn_, 0, 2);
      bytes(enc, params, 3);
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(D / 64), row_tiles, static_cast<NSUInteger>(H))
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    return true;
  }

  // ---- multi-head latent attention ---------------------------------------

  // The query (through the query rank when there is one) and the latent
  // projection into kt_, then each head's query absorbed into
  // latent_q_[t, h, :rank]: through the stacked embed_q, or straight from
  // kv_b_proj's key rows.
  void encode_latent_qkv(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    if (c_.q_lora_rank > 0) {
      encode_qmm(enc, G.q_a, h_, n, q_latent_);
      encode_rms_norm(enc, q_latent_, G.q_a_norm, q_latent_, n, NormParams{c_.q_lora_rank, c_.latent_norm_eps, 0.0f});
      encode_qmm(enc, G.q, q_latent_, n, q_);
    } else {
      encode_qmm(enc, G.q, h_, n, q_);
    }
    encode_qmm(enc, G.kv_a, h_, n, kt_);
    const int H = c_.num_heads, nope = c_.qk_nope_dim, rank = c_.kv_lora_rank, width = c_.latent_width();
    const NSUInteger pairs = static_cast<NSUInteger>(n * H);
    if (G.absorbed) {
      [enc setComputePipelineState:quant_pipeline("qmv_gather", G.embed_q.format())];
      bind(enc, G.embed_q.words, 0);
      bind(enc, G.embed_q.scales, 1);
      bind(enc, G.embed_q.biases, 2);
      bind(enc, q_, 0, 3);
      bind(enc, latent_q_, 0, 4);
      bytes(enc, GatherParams{rank, nope, 1, 0, 0, 0, 0.0f, c_.head_dim, width}, 5);
      bind(enc, head_index_, 0, 6);
      bind(enc, placeholder_, 7);
      [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(rank), pairs, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      return;
    }
    id<MTLComputePipelineState> pso = quant_pipeline("qmv_t_heads", G.kv_b);
    [enc setComputePipelineState:pso];
    bind_weights(enc, G.kv_b, 0);
    bind(enc, q_, 0, 3);
    bind(enc, latent_q_, 0, 4);
    bytes(enc, AbsorbParams{H, nope, nope + c_.v_head_dim, rank, c_.head_dim, width}, 5);
    // One thread per word, or per 16 codes when codes straddle words.
    const NSUInteger units = static_cast<NSUInteger>(32 % G.kv_b.bits == 0 ? rank * G.kv_b.bits / 32 : rank / 16);
    const NSUInteger threads =
        std::min<NSUInteger>((units + 31) / 32 * 32, std::min<NSUInteger>(256, [pso maxTotalThreadsPerThreadgroup] / 32 * 32));
    [enc dispatchThreadgroups:MTLSizeMake(pairs, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  }

  // RoPE on the rope queries (into latent_q_[t, h, rank:]) and the pass's
  // latents normed and rotated into the cache.
  void encode_latent_rope(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    [enc setComputePipelineState:pipeline("mla_rope_store")];
    bind(enc, q_, 0, 0);
    bind(enc, latent_q_, 0, 1);
    bind(enc, kt_, 0, 2);
    bind(enc, G.k_cache, 0, 3);
    bind(enc, G.kv_a_norm, 4);
    bind(enc, inv_freq_global_buf_, 5);
    bytes(enc, MlaRopeParams{c_.num_heads, c_.qk_nope_dim, c_.qk_rope_dim, c_.kv_lora_rank, start,
                             c_.latent_norm_eps, c_.rope_pairs ? 1 : 0, c_.rope_factor},
          6);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(c_.num_heads + 1), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }

  // Causal attention of the absorbed queries over the cached latents (one
  // key head, values its first kv_lora_rank dimensions), four query heads
  // per threadgroup, into latent_out_ (whose rope dimensions are unused).
  void encode_latent_attention(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    constexpr int kHeadsPerGroup = 4;
    const int H = c_.num_heads, width = c_.latent_width();
    const int keys_max = start + n;
    const int max_blocks = n == 1 ? kDecodeMaxBlocks : kPrefillMaxBlocks;
    const int block = std::max(n == 1 ? 32 : 64, (keys_max + max_blocks - 1) / max_blocks);
    const int n_blocks = (keys_max + block - 1) / block;
    [enc setComputePipelineState:pipeline("attn_partial_g4_d" + std::to_string(width / 32))];
    bind(enc, latent_q_, 0, 0);
    bind(enc, G.k_cache, 0, 1);
    bind(enc, G.k_cache, 0, 2);
    bind(enc, part_o_, 0, 3);
    bind(enc, part_ml_, 0, 4);
    bytes(enc, AttnParams{H, 1, width, start, 1 << 30, G.capacity, block, n_blocks, c_.attn_scale, 0.0f, 0,
                          H / kHeadsPerGroup},
          5);
    bind(enc, last_key_, 0, 6);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_blocks), static_cast<NSUInteger>(H / kHeadsPerGroup),
                                          static_cast<NSUInteger>(n))
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc setComputePipelineState:pipeline("attn_reduce")];
    bind(enc, part_o_, 0, 0);
    bind(enc, part_ml_, 0, 1);
    bind(enc, latent_out_, 0, 2);
    bytes(enc, ReduceParams{H, width, n_blocks, 0}, 3);
    bind(enc, placeholder_, 4);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(H), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(width), 1, 1)];
  }

  // Each head's latent output expanded (unembed_out) into attn_ [n, heads *
  // v_head_dim], then o_proj into r_.
  void encode_latent_out(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    const int V = c_.v_head_dim;
    [enc setComputePipelineState:quant_pipeline("qmv_gather", G.unembed_out.format())];
    bind(enc, G.unembed_out.words, 0);
    bind(enc, G.unembed_out.scales, 1);
    bind(enc, G.unembed_out.biases, 2);
    bind(enc, latent_out_, 0, 3);
    bind(enc, attn_, 0, 4);
    bytes(enc, GatherParams{V, c_.kv_lora_rank, 1, 0, 0, 0, 0.0f, c_.latent_width(), 0}, 5);
    bind(enc, head_index_, 0, 6);
    bind(enc, placeholder_, 7);
    [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(V), static_cast<NSUInteger>(n * c_.num_heads), 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    encode_qmm(enc, G.o, attn_, n, r_);
  }

  // Gemma 4's per-layer inputs for the pass into per_layer_: the per-layer
  // embedding lookup (times sqrt(P)) plus the normed projection of the
  // scaled embedding in x_.
  void encode_per_layer_inputs(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> tokens, size_t tokens_offset, int n) {
    const int width = c_.num_layers * c_.per_layer_input;
    [enc setComputePipelineState:quant_pipeline("embed_rows", embed_per_layer_)];
    bind_weights(enc, embed_per_layer_, 0);
    bind(enc, tokens, tokens_offset, 3);
    bind(enc, per_layer_lookup_, 0, 4);
    bytes(enc, EmbedParams{width, n, c_.per_layer_embed_scale}, 5);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_qmm(enc, per_layer_model_proj_, x_, n, per_layer_proj_);
    id<MTLComputePipelineState> pso = pipeline("per_layer_combine");
    [enc setComputePipelineState:pso];
    bind(enc, per_layer_proj_, 0, 0);
    bind(enc, per_layer_lookup_, 0, 1);
    bind(enc, per_layer_proj_norm_, 2);
    bind(enc, per_layer_, 0, 3);
    bytes(enc, PerLayerCombineParams{c_.per_layer_input, c_.num_layers, c_.rms_norm_eps, c_.norm_offset,
                                     c_.per_layer_projection_scale, c_.per_layer_input_scale},
          4);
    const NSUInteger threads = std::min<NSUInteger>(
        std::max<NSUInteger>(32, static_cast<NSUInteger>((c_.per_layer_input + 31) / 32 * 32)),
        std::min<NSUInteger>(256, [pso maxTotalThreadsPerThreadgroup] / 32 * 32));
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(c_.num_layers), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  }

  // r_ = per_layer_projection(gelu(per_layer_input_gate(x)) * this layer's input).
  void encode_per_layer_gate(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int layer, int n) {
    encode_qmm(enc, G.pl_gate, x_, n, per_layer_gate_);
    [enc setComputePipelineState:pipeline("per_layer_gate")];
    bind(enc, per_layer_gate_, 0, 0);
    bind(enc, per_layer_, 0, 1);
    const int count = n * c_.per_layer_input;
    bytes(enc, PerLayerGateParams{c_.per_layer_input, c_.num_layers, layer, count}, 2);
    encode_elementwise(enc, count);
    encode_qmm(enc, G.pl_proj, per_layer_gate_, n, r_);
  }

  // Routing of the rows in `input` (h_ unless Gemma 4's router input). It
  // runs on the GPU, so the host never waits for the chosen experts.
  void encode_router(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, id<MTLBuffer> input) {
    if (G.router_dense) {
      [enc setComputePipelineState:pipeline("dense_rows")];
      bind(enc, G.dense_router.data, 0);
      bind(enc, input, 0, 1);
      bind(enc, route_, 0, 2);
      bytes(enc, DenseParams{G.dense_router.rows, G.dense_router.cols, G.dense_router.dtype}, 3);
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((G.dense_router.rows + 1) / 2),
                                            static_cast<NSUInteger>(n), 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      encode_qmm(enc, G.router, input, n, route_);
    }
    if (c_.router == DecoderConfig::Router::SigmoidGrouped) {
      [enc setComputePipelineState:pipeline("moe_route_grouped")];
      bind(enc, route_, 0, 0);
      bind(enc, chosen_, 0, 1);
      bind(enc, expert_weights_, 0, 2);
      bytes(enc, MoeGroupedParams{c_.num_experts, c_.experts_per_token, c_.n_group, c_.topk_group,
                                  c_.norm_topk_prob ? 1 : 0, c_.routed_scaling, c_.exclude_dropped_groups ? 1 : 0},
            3);
      bind(enc, G.router_correction, 4);
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      return;
    }
    [enc setComputePipelineState:pipeline("moe_route")];
    bind(enc, route_, 0, 0);
    bind(enc, chosen_, 0, 1);
    bind(enc, expert_weights_, 0, 2);
    bytes(enc, MoeRouteParams{c_.num_experts, c_.experts_per_token,
                              c_.router == DecoderConfig::Router::TopKSoftmax ? 1 : 0, c_.norm_topk_prob ? 1 : 0,
                              c_.router_bias ? 1 : 0, c_.experts_beside_mlp ? 1 : 0, c_.n_group, c_.topk_group,
                              c_.routed_scaling},
          3);
    bind(enc, G.router_bias, 4);
    bind(enc, G.per_expert_scale, 5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  }

  // `out` = the experts of the rows in `input`, weighted (plus the shared expert).
  void encode_experts(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, id<MTLBuffer> input,
                      id<MTLBuffer> out) {
    const int K = c_.experts_per_token, I = c_.moe_intermediate, pairs = n * K;
    const int act = c_.activation == Activation::ClampedSwiglu ? 2 : c_.activation == Activation::Silu ? 1 : 0;
    const int has_bias = c_.expert_bias ? 1 : 0;
    [enc setComputePipelineState:quant_pipeline("qmv_gather_glu", G.expert_gate.format())];
    bind(enc, G.expert_gate.words, 0);
    bind(enc, G.expert_gate.scales, 1);
    bind(enc, G.expert_gate.biases, 2);
    bind(enc, G.expert_up.words, 3);
    bind(enc, G.expert_up.scales, 4);
    bind(enc, G.expert_up.biases, 5);
    bind(enc, input, 0, 6);
    bind(enc, expert_hidden_, 0, 7);
    bytes(enc, GatherParams{I, c_.hidden_size, K, 1, act, has_bias, c_.swiglu_limit, 0, 0}, 8);
    bind(enc, chosen_, 0, 9);
    bind(enc, G.expert_gate.bias, 10);
    bind(enc, G.expert_up.bias, 11);
    [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(I), static_cast<NSUInteger>(pairs), 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc setComputePipelineState:quant_pipeline("qmv_gather", G.expert_down.format())];
    bind(enc, G.expert_down.words, 0);
    bind(enc, G.expert_down.scales, 1);
    bind(enc, G.expert_down.biases, 2);
    bind(enc, expert_hidden_, 0, 3);
    bind(enc, expert_out_, 0, 4);
    bytes(enc, GatherParams{c_.hidden_size, I, K, 0, 0, has_bias, c_.swiglu_limit, 0, 0}, 5);
    bind(enc, chosen_, 0, 6);
    bind(enc, G.expert_down.bias, 7);
    [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(c_.hidden_size), static_cast<NSUInteger>(pairs), 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    const bool shared = c_.shared_intermediate > 0;
    if (shared) encode_shared_expert(enc, G, n);
    if (shared && c_.shared_expert_gate) {  // times sigmoid(shared_expert_gate(x))
      encode_qmm(enc, G.shared_expert_gate, input, n, shared_scale_);
      encode_mul_sigmoid(enc, shared_out_, shared_scale_, c_.hidden_size, n * c_.hidden_size);
    }
    [enc setComputePipelineState:pipeline("moe_combine")];
    bind(enc, expert_out_, 0, 0);
    bind(enc, expert_weights_, 0, 1);
    bind(enc, out, 0, 2);
    const int count = n * c_.hidden_size;
    bytes(enc, MoeCombineParams{c_.hidden_size, K, count, shared ? 1 : 0}, 3);
    bind(enc, shared ? shared_out_ : out, 0, 4);
    encode_elementwise(enc, count);
  }

  // shared_out_ = the shared expert (a dense SiLU-gated MLP) of h_.
  void encode_shared_expert(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    const int silu = c_.activation == Activation::Silu ? 1 : 0;
    if (n == 1 && G.shared_gate.same_format(G.shared_up)) {
      [enc setComputePipelineState:quant_pipeline("qmv_glu", G.shared_gate)];
      bind_weights(enc, G.shared_gate, 0);
      bind_weights(enc, G.shared_up, 3);
      bind(enc, h_, 0, 6);
      bind(enc, gate_, 0, 7);
      bytes(enc, GluParams{G.shared_gate.rows, G.shared_gate.cols, silu}, 8);
      [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(G.shared_gate.rows), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      encode_qmm(enc, G.shared_gate, h_, n, gate_);
      encode_qmm(enc, G.shared_up, h_, n, up_);
      [enc setComputePipelineState:pipeline("glu_mul")];
      bind(enc, gate_, 0, 0);
      bind(enc, up_, 0, 1);
      const int count = n * c_.shared_intermediate;
      bytes(enc, GluMulParams{count, silu}, 2);
      encode_elementwise(enc, count);
    }
    encode_qmm(enc, G.shared_down, gate_, n, shared_out_);
  }

  // Bytes the experts of this pass read: each distinct routed expert once
  // (known after the router section has run, which profiling waits for).
  double expert_traffic(const DecoderLayer& L, int n) const {
    const int32_t* chosen = static_cast<const int32_t*>([chosen_ contents]);
    std::vector<int> count(static_cast<size_t>(c_.num_experts), 0);
    for (int i = 0; i < n * c_.experts_per_token; ++i) {
      if (chosen[i] >= 0 && chosen[i] < c_.num_experts) ++count[static_cast<size_t>(chosen[i])];
    }
    double bytes = 0.0;
    for (int e = 0; e < c_.num_experts; ++e) {
      const int m = count[static_cast<size_t>(e)];
      if (m == 0) continue;
      bytes += traffic::linear(L.expert_gate[static_cast<size_t>(e)], m) +
               traffic::linear(L.expert_up[static_cast<size_t>(e)], m) +
               traffic::linear(L.expert_down[static_cast<size_t>(e)], m);
    }
    if (c_.shared_intermediate > 0) {
      bytes += traffic::linear(L.shared_gate, n) + traffic::linear(L.shared_up, n) + traffic::linear(L.shared_down, n) +
               (c_.shared_expert_gate ? traffic::linear(L.shared_expert_gate, n) : 0.0);
    }
    return bytes + traffic::rows(c_.hidden_size, n, 1);
  }

  void encode_mlp(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    const int silu = c_.activation == Activation::Silu ? 1 : 0;
    if (n == 1 && G.gate.same_format(G.up)) {
      [enc setComputePipelineState:quant_pipeline("qmv_glu", G.gate)];
      bind_weights(enc, G.gate, 0);
      bind_weights(enc, G.up, 3);
      bind(enc, h_, 0, 6);
      bind(enc, gate_, 0, 7);
      bytes(enc, GluParams{G.gate.rows, G.gate.cols, silu}, 8);
      [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(G.gate.rows), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      encode_qmm(enc, G.gate, h_, n, gate_);
      encode_qmm(enc, G.up, h_, n, up_);
      [enc setComputePipelineState:pipeline("glu_mul")];
      bind(enc, gate_, 0, 0);
      bind(enc, up_, 0, 1);
      const int count = n * G.inter;
      bytes(enc, GluMulParams{count, silu}, 2);
      encode_elementwise(enc, count);
    }
    encode_qmm(enc, G.down, gate_, n, r_);
  }

  // While profiling, ends the work encoded since the previous section in its
  // own command buffer, waits for it, and books its GPU time as `name`; the
  // caller continues with a fresh encoder. A no-op otherwise.
  void section(__strong id<MTLComputeCommandEncoder>& enc, const char* name, double traffic_bytes) {
    if (profiler_ == nullptr) return;
    [enc endEncoding];
    [active_cb_ commit];
    [active_cb_ waitUntilCompleted];
    LK_CHECK([active_cb_ status] == MTLCommandBufferStatusCompleted,
             "metal: command buffer failed: " + ns_error([active_cb_ error]));
    profiler_->add(name, [active_cb_ GPUEndTime] - [active_cb_ GPUStartTime], traffic_bytes);
    active_cb_ = [queue_ commandBuffer];
    enc = [active_cb_ computeCommandEncoder];
  }

  // Layer l over the pass's n rows (h_ holds their norm), at positions
  // [start, start + n); its residual leaves norm `next` of x_ in h_.
  void encode_layer(__strong id<MTLComputeCommandEncoder>& enc, size_t l, int n, int start, const BufRef* next) {
    const bool profiling = profiler_ != nullptr;
    const int64_t hidden = c_.hidden_size;
    const GpuLayer& G = layers_[l];
    const DecoderLayer& L = l < w_->layers.size() ? w_->layers[l] : w_->mtp->layer;
    const int layer = static_cast<int>(l);
    // The residual kernel adds the sublayer output (normalized first with
    // sandwich norms) and applies the next norm in one pass.
    const double residual_bytes = profiling ? traffic::rows(hidden, n, 4) + 8.0 * hidden : 0.0;
    if (G.linear) {
      encode_gdn_projections(enc, G, n);
      section(enc, "qkv", profiling ? traffic::linear_projections(L, n) : 0.0);
      // A verify pass records its conv inputs, then its recurrence's
      // inputs, for accept (each copy read and written once).
      const int channels = c_.linear_channels(), gates = n * c_.linear_v_heads;
      const VerifyRecord* record = verifying_ ? &records_[l] : nullptr;
      if (record != nullptr) encode_copy(enc, mixed_, record->raw, n * channels);
      encode_gdn_conv(enc, G, n, start);
      section(enc, "rope_kv",
              profiling ? traffic::linear_conv(c_, n) + (record ? traffic::rows(channels, n, 2) : 0.0) : 0.0);
      if (record != nullptr) {
        encode_copy(enc, mixed_, record->rows, n * channels);
        encode_copy(enc, lin_b_, record->beta, gates);
        encode_copy(enc, lin_a_, record->decay, gates);
      }
      encode_gdn_recurrence(enc, G, n, start);
      section(enc, "attention",
              profiling ? traffic::linear_recurrence(c_, n) +
                              (record ? traffic::rows(channels, n, 2) + traffic::rows(c_.linear_v_heads, n, 4) : 0.0)
                        : 0.0);
      encode_gdn_out(enc, G, n);
      section(enc, "o_proj", profiling ? traffic::linear_out(c_, L, n) : 0.0);
    } else if (c_.mla()) {
      encode_latent_qkv(enc, G, n);
      section(enc, "qkv", profiling ? traffic::latent_qkv(c_, L, n) : 0.0);
      encode_latent_rope(enc, G, n, start);
      section(enc, "rope_kv", profiling ? traffic::rows(c_.latent_width(), n, 2 * c_.num_heads + 3) : 0.0);
      encode_latent_attention(enc, G, n, start);
      section(enc, "attention", profiling ? traffic::attention(c_, layer, start, n) : 0.0);
      encode_latent_out(enc, G, n);
      section(enc, "o_proj", profiling ? traffic::latent_out(L, n) : 0.0);
    } else {
      encode_qkv(enc, G, n);
      section(enc, "qkv",
              profiling ? traffic::linear(L.q_proj, n) + (G.own_kv ? traffic::linear(L.k_proj, n) : 0.0) +
                              (G.own_kv && !G.k_eq_v ? traffic::linear(L.v_proj, n) : 0.0)
                        : 0.0);
      encode_rope(enc, G, n, start);
      // q in and out, k and v in, and k and v written to the cache.
      section(enc, "rope_kv",
              profiling ? traffic::rows(G.head_dim, n, 2 * c_.num_heads + (G.own_kv ? 2 * G.kv_heads : 0)) +
                              (G.own_kv ? kv_bytes() / 4.0 * traffic::rows(G.head_dim, n, 2 * G.kv_heads) : 0.0)
                        : 0.0);
      encode_attention(enc, G, n, start);
      section(enc, "attention", profiling ? traffic::attention(c_, layer, start, n, kv_bytes()) : 0.0);
      const int gated = n * c_.num_heads * G.head_dim;
      if (c_.attn_output_gate) encode_mul_sigmoid(enc, attn_, out_gate_, 1, gated);
      encode_qmm(enc, G.o, attn_, n, r_);
      if (c_.o_bias) {
        [enc setComputePipelineState:pipeline("add_bias_rows")];
        bind(enc, r_, 0, 0);
        bind(enc, G.o_bias, 1);
        const int count = n * c_.hidden_size;
        bytes(enc, BiasParams{c_.hidden_size, count}, 2);
        encode_elementwise(enc, count);
      }
      section(enc, "o_proj",
              profiling ? traffic::linear(L.o_proj, n) + (c_.attn_output_gate ? traffic::rows(gated, 1, 3) : 0.0)
                        : 0.0);
    }
    encode_residual_norm(enc, G.attn_post_norm, &G.mlp_norm, n);
    section(enc, "residual_norm", residual_bytes);
    if (G.moe && c_.experts_beside_mlp) {
      // Gemma 4: r_ = mlp_post_norm_1(mlp(h_)) + expert_post_norm(experts(expert_norm(x))),
      // routed on rms(x) * router_input_scale.
      encode_mlp(enc, G, n);
      section(enc, "mlp",
              profiling ? traffic::linear(L.gate_proj, n) + traffic::linear(L.up_proj, n) +
                              traffic::linear(L.down_proj, n) + traffic::rows(G.inter, n, 1)
                        : 0.0);
      encode_rms_norm(enc, r_, G.mlp_post_norm_1, r_, n);
      encode_rms_norm(enc, x_, G.router_input_scale, router_in_, n);
      encode_rms_norm(enc, x_, G.expert_norm, h_, n);
      section(enc, "norm", profiling ? traffic::rows(c_.hidden_size, n, 6) + 12.0 * c_.hidden_size : 0.0);
      encode_router(enc, G, n, router_in_);
      section(enc, "router", profiling ? traffic::linear(L.router, n) + traffic::rows(c_.num_experts, n, 1) : 0.0);
      const double expert_bytes = profiling ? expert_traffic(L, n) : 0.0;
      encode_experts(enc, G, n, h_, moe_out_);
      section(enc, "experts", expert_bytes);
      encode_add_normed(enc, r_, moe_out_, G.expert_post_norm, n);
      section(enc, "norm", profiling ? traffic::rows(c_.hidden_size, n, 3) + 4.0 * c_.hidden_size : 0.0);
    } else if (G.moe) {
      encode_router(enc, G, n, h_);
      section(enc, "router", profiling ? traffic::linear(L.router, n) + traffic::rows(c_.num_experts, n, 1) : 0.0);
      // While profiling the router has finished here, so its choices are readable.
      const double expert_bytes = profiling ? expert_traffic(L, n) : 0.0;
      encode_experts(enc, G, n, h_, r_);
      section(enc, "experts", expert_bytes);
    } else {
      encode_mlp(enc, G, n);
      section(enc, "mlp",
              profiling ? traffic::linear(L.gate_proj, n) + traffic::linear(L.up_proj, n) +
                              traffic::linear(L.down_proj, n) + traffic::rows(G.inter, n, 1)
                        : 0.0);
    }
    if (c_.per_layer_input > 0) {
      // x += post_ff_norm(mlp); x += norm(per-layer input); x *= scalar.
      encode_residual_norm(enc, G.mlp_post_norm, nullptr, n);
      section(enc, "residual_norm", residual_bytes);
      encode_per_layer_gate(enc, G, layer, n);
      section(enc, "per_layer_input",
              profiling ? traffic::linear(L.per_layer_gate, n) + traffic::linear(L.per_layer_proj, n) +
                              traffic::rows(c_.per_layer_input, n, 2)
                        : 0.0);
      encode_residual_norm(enc, G.pl_norm, next, n, G.scalar);
    } else if (!plan_.empty() && l < active_.layer_rows.size()) {
      // Deepstack: x += mlp, then this layer's rows onto the image tokens,
      // then the next layer's norm.
      encode_residual_norm(enc, G.mlp_post_norm, nullptr, n, G.scalar);
      [enc setComputePipelineState:pipeline("add_rows_at")];
      bind(enc, layer_rows_, sizeof(float) * l * active_.rows.size(), 0);
      bind(enc, inject_index_, sizeof(int32_t) * static_cast<size_t>(start - plan_.base), 1);
      bind(enc, x_, 0, 2);
      bytes(enc, ReplaceParams{c_.hidden_size, n}, 3);
      [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(c_.hidden_size), static_cast<NSUInteger>(n), 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      encode_rms_norm(enc, x_, *next, h_, n);
    } else {
      encode_residual_norm(enc, G.mlp_post_norm, next, n, G.scalar);
    }
    section(enc, "residual_norm", residual_bytes);
  }

  // Encodes n tokens at positions [start, start + n) whose ids are at
  // tokens + tokens_offset. When `last`, also the final projection and argmax.
  void encode_forward(__strong id<MTLComputeCommandEncoder>& enc, id<MTLBuffer> tokens, size_t tokens_offset, int n,
                      int start, bool last, int history_index) {
    const bool profiling = profiler_ != nullptr;
    const int64_t hidden = c_.hidden_size;
    [enc setComputePipelineState:quant_pipeline("embed_rows", embed_)];
    bind_weights(enc, embed_, 0);
    bind(enc, tokens, tokens_offset, 3);
    bind(enc, x_, 0, 4);
    bytes(enc, EmbedParams{c_.hidden_size, n, c_.embed_scale}, 5);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(c_.hidden_size), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (!plan_.empty()) {  // injected rows (image features) replace the scaled embeddings
      [enc setComputePipelineState:pipeline("replace_rows")];
      bind(enc, inject_rows_, 0, 0);
      bind(enc, inject_index_, sizeof(int32_t) * static_cast<size_t>(start - plan_.base), 1);
      bind(enc, x_, 0, 2);
      bytes(enc, ReplaceParams{c_.hidden_size, n}, 3);
      [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(c_.hidden_size), static_cast<NSUInteger>(n), 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    section(enc, "embed", profiling ? traffic::embed(w_->embed, n) : 0.0);
    if (c_.per_layer_input > 0) {
      // During an image prefill the lookup reads the pad-substituted tokens,
      // laid out like the prompt.
      encode_per_layer_inputs(enc, plan_.empty() ? tokens : per_layer_tokens_, tokens_offset, n);
      const int64_t width = static_cast<int64_t>(c_.num_layers) * c_.per_layer_input;
      section(enc, "per_layer_input",
              profiling ? traffic::linear(w_->per_layer_model_proj, n) + traffic::embed(w_->embed_per_layer, n) +
                              traffic::rows(width, n, 2)
                        : 0.0);
    }
    encode_rms_norm(enc, x_, layers_[0].attn_norm, h_, n);
    section(enc, "norm", profiling ? traffic::rows(hidden, n, 2) + 4.0 * hidden : 0.0);

    const size_t decoder_layers = static_cast<size_t>(c_.num_layers);
    auto next_norm = [&](size_t l) { return l + 1 < decoder_layers ? &layers_[l + 1].attn_norm : &final_norm_; };
    // KV-shared trailing layers write no cache: a plain pass skips them
    // without logits and runs them on its last row with (shared_suffix()).
    const size_t full = verifying_ || mtp_on_ ? decoder_layers : static_cast<size_t>(c_.shared_suffix());
    for (size_t l = 0; l < full; ++l) encode_layer(enc, l, n, start, next_norm(l));
    if (full < decoder_layers) {
      if (!last) return;
      if (n > 1) {  // the last row of x, of its normed copy and of the per-layer inputs into row 0
        const size_t row = static_cast<size_t>(n - 1) * c_.hidden_size * sizeof(float);
        encode_copy(enc, x_, x_, c_.hidden_size, row, 0);
        encode_copy(enc, h_, h_, c_.hidden_size, row, 0);
        if (c_.per_layer_input > 0) {
          const int width = c_.num_layers * c_.per_layer_input;
          encode_copy(enc, per_layer_, per_layer_, width, static_cast<size_t>(n - 1) * width * sizeof(float), 0);
        }
      }
      for (size_t l = full; l < decoder_layers; ++l) encode_layer(enc, l, 1, start + n - 1, next_norm(l));
      n = 1;
    }
    if (mtp_on_) encode_mtp_follow(enc, tokens, tokens_offset, n, start);
    if (!last) return;
    const size_t last_row = static_cast<size_t>(n - 1) * c_.hidden_size * sizeof(float);
    encode_qmv(enc, lm_head_, h_, last_row, logits_, 0);
    if (c_.final_logit_softcapping > 0.0f) {
      [enc setComputePipelineState:pipeline("softcap")];
      bind(enc, logits_, 0, 0);
      bytes(enc, c_.final_logit_softcapping, 1);
      bytes(enc, ElementwiseParams{c_.vocab_size}, 2);
      encode_elementwise(enc, c_.vocab_size);
    }
    section(enc, "lm_head", profiling ? traffic::linear(w_->lm_head, 1) : 0.0);
    id<MTLComputePipelineState> pick = pipeline(sampling_.greedy() ? "argmax_vocab" : "sample_vocab");
    [enc setComputePipelineState:pick];
    bind(enc, logits_, 0, 0);
    bind(enc, token_, 0, 1);
    bind(enc, history_, 0, 2);
    if (sampling_.greedy()) {
      bytes(enc, ArgmaxParams{c_.vocab_size, history_index}, 3);
    } else {
      bytes(enc, sample_params(sampling_, c_.vocab_size, history_index), 3);
    }
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(1024, [pick maxTotalThreadsPerThreadgroup] / 32 * 32), 1, 1)];
    section(enc, "sample", profiling ? 4.0 * c_.vocab_size : 0.0);
  }

  void run_prefill(const int32_t* tokens, int n) {
    // The pending injection applies to this prefill only.
    active_ = std::move(injection_);
    injection_ = {};
    struct Clear {
      DecoderMetal& self;
      ~Clear() {
        self.active_ = {};
        self.plan_ = {};
      }
    } clear{*this};
    LK_CHECK(n > 0, "prefill requires at least one token");
    LK_CHECK(position_ + n <= max_context_, "context length exceeded");
    for (int i = 0; i < n; ++i) LK_CHECK(tokens[i] >= 0 && tokens[i] < c_.vocab_size, "token id out of range");
    LK_CHECK(!mtp_on_ || active_.empty(), "the MTP head does not take image features yet");
    plan_ = plan_injection(active_, position_, n, c_.hidden_size, chunk_, c_.injection_rules(rope_offset_));
    rope_offset_ = plan_.rope_offset;  // for the tokens after this prefill
    if (!plan_.rope.empty()) std::memcpy([rope_pos_ contents], plan_.rope.data(), sizeof(int32_t) * plan_.rope.size());
    if (!plan_.empty()) {
      if (c_.per_layer_input > 0) {
        // Image positions look up the pad token's per-layer embedding.
        LK_CHECK(c_.pad_token_id >= 0 && c_.pad_token_id < embed_per_layer_.rows, "metal: pad token out of range");
        int32_t* lookup = static_cast<int32_t*>([per_layer_tokens_ contents]);
        for (int i = 0; i < n; ++i) lookup[i] = plan_.row[static_cast<size_t>(i)] >= 0 ? c_.pad_token_id : tokens[i];
      }
      std::memcpy([inject_index_ contents], plan_.row.data(), sizeof(int32_t) * static_cast<size_t>(n));
      std::memcpy([last_key_ contents], plan_.last_key.data(), sizeof(int32_t) * static_cast<size_t>(n));
      inject_rows_ = [device_ newBufferWithBytes:active_.rows.data()
                                          length:std::max<size_t>(sizeof(float) * active_.rows.size(), 16)
                                         options:MTLResourceStorageModeShared];
      LK_CHECK(inject_rows_ != nil, "metal: allocation failed");
      if (!active_.layer_rows.empty()) {  // every deepstack layer's rows, one after another
        layer_rows_ = shared_buffer(sizeof(float) * active_.rows.size() * active_.layer_rows.size());
        float* at = static_cast<float*>([layer_rows_ contents]);
        for (const std::vector<float>& layer : active_.layer_rows) at = std::copy(layer.begin(), layer.end(), at);
      }
    }
    std::memcpy([prompt_ contents], tokens, sizeof(int32_t) * static_cast<size_t>(n));
    active_cb_ = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [active_cb_ computeCommandEncoder];
    for (int begin = 0; begin < n;) {
      // A pass never splits an image block.
      const int count = plan_.pass_length(begin, std::min(chunk_, n - begin));
      encode_forward(enc, prompt_, sizeof(int32_t) * static_cast<size_t>(begin), count, position_ + begin,
                     begin + count == n, 0);
      begin += count;
    }
    [enc endEncoding];
    id<MTLCommandBuffer> cb = active_cb_;
    [cb commit];
    wait(cb);
    position_ += n;
  }

  id<MTLCommandBuffer> encode_decode_step(int pos, int history_index) {
    active_cb_ = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [active_cb_ computeCommandEncoder];
    encode_forward(enc, token_, 0, 1, pos, true, history_index);
    [enc endEncoding];
    pending_.push_back(active_cb_);
    return active_cb_;
  }

  void wait(id<MTLCommandBuffer> cb) {
    [cb waitUntilCompleted];
    pending_.erase(std::remove(pending_.begin(), pending_.end(), cb), pending_.end());
    LK_CHECK([cb status] == MTLCommandBufferStatusCompleted, "metal: command buffer failed: " + ns_error([cb error]));
  }

  void drain() {
    std::vector<id<MTLCommandBuffer>> pending;
    pending.swap(pending_);
    for (id<MTLCommandBuffer> cb : pending) [cb waitUntilCompleted];
  }

  std::shared_ptr<DecoderWeights> w_;
  const DecoderConfig& c_;
  int max_context_;
  int chunk_;
  bool kv_f16_;  // standard attention's keys and values stored as FP16
  int position_ = 0;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;

  GpuLinear embed_, lm_head_, embed_per_layer_, per_layer_model_proj_;
  bool tied_ = false;
  BufRef final_norm_, placeholder_, inv_freq_local_buf_, inv_freq_global_buf_, per_layer_proj_norm_;
  int max_head_dim_ = 0, max_kv_width_ = 0, max_inter_ = 0;  // the largest over layers
  std::vector<float> inv_freq_local_, inv_freq_global_, placeholder_data_;
  std::vector<GpuLayer> layers_;
  size_t device_weight_bytes_ = 0;

  id<MTLBuffer> x_ = nil, h_ = nil, r_ = nil, q_ = nil, kt_ = nil, vt_ = nil, attn_ = nil, gate_ = nil, up_ = nil;
  id<MTLBuffer> part_o_ = nil, part_ml_ = nil;
  id<MTLBuffer> attn_scores_ = nil;  // [heads, rows, keys] for encode_attention_mm, grown on demand
  size_t attn_scores_bytes_ = 0;
  // Mixture of experts: router logits [chunk, E], each token's experts and
  // weights [chunk, K], and each slot's expert activations and outputs.
  id<MTLBuffer> route_ = nil, chosen_ = nil, expert_weights_ = nil, expert_hidden_ = nil, expert_out_ = nil;
  // Host copies of expert stacks, alive until the arena has copied them.
  std::deque<std::vector<uint8_t>> staged_;
  std::vector<std::pair<GpuExperts*, std::array<size_t, 3>>> expert_fix_;
  std::vector<std::pair<BufRef*, size_t>> dense_fix_;
  id<MTLBuffer> shared_out_ = nil;  // GLM's shared expert output [chunk, hidden]
  id<MTLBuffer> router_in_ = nil, moe_out_ = nil;  // Gemma 4: the router's input and the experts' output
  // Gemma 4's per-layer inputs [chunk, layers * P], their two parts, and one
  // layer's gate [chunk, P].
  id<MTLBuffer> per_layer_ = nil, per_layer_lookup_ = nil, per_layer_proj_ = nil, per_layer_gate_ = nil;
  id<MTLBuffer> logits_ = nil, token_ = nil, history_ = nil, prompt_ = nil;
  // The prefill in progress's injection and plan; its rows, and per prompt
  // token the row index and last key (shared, written before each prefill).
  PromptInjection active_;
  InjectionPlan plan_;
  id<MTLBuffer> inject_rows_ = nil, inject_index_ = nil, last_key_ = nil;
  id<MTLBuffer> layer_rows_ = nil;  // deepstack rows [layers, injected rows, hidden]
  // M-RoPE: the prefill's positions [n, 3] (shared, written before each
  // prefill), each frequency's axis, and how far RoPE positions run ahead
  // of KV positions until reset().
  id<MTLBuffer> rope_pos_ = nil, rope_axes_ = nil;
  int rope_offset_ = 0;
  id<MTLBuffer> per_layer_tokens_ = nil;  // the prompt with image tokens replaced by the pad token
  // Multi-head latent attention: absorbed queries and latent outputs [chunk,
  // heads, latent width], the query latent [chunk, q_lora_rank], and each
  // (token, head) pair's head.
  id<MTLBuffer> latent_q_ = nil, latent_out_ = nil, q_latent_ = nil, head_index_ = nil;
  // Gated DeltaNet: the [q | k | v] channels, z and the recurrence's outputs
  // [chunk, Hv * dv], b (then beta) and a (then the decay) [chunk, Hv], and
  // Qwen3-Next's grouped projections.
  id<MTLBuffer> mixed_ = nil, lin_z_ = nil, lin_out_ = nil, lin_b_ = nil, lin_a_ = nil, grouped_ = nil, grouped_ba_ = nil;
  // Gated attention: q_proj's [query | gate] rows and the gates; the shared
  // expert's gate logits [chunk].
  id<MTLBuffer> q_gate_ = nil, out_gate_ = nil, shared_scale_ = nil;
  std::vector<id<MTLCommandBuffer>> pending_;
  // The command buffer encode_forward is appending to (sections replace it
  // while profiling).
  id<MTLCommandBuffer> active_cb_ = nil;
  // How the final step picks a token; greedy outside generate().
  SamplingParams sampling_;
  // Speculative decoding. Each Gated DeltaNet pair holds position p's conv
  // history and state in buffer (p + parity_) % 2; verify and accept set the
  // parity so that the buffers a verify pass wrote hold the current
  // position's.
  int parity_ = 0;
  bool verifying_ = false;  // encode_forward is encoding a verify pass
  int verify_start_ = 0, verify_n_ = 0, verify_capacity_ = 0;
  size_t verify_read_ = 0;  // the buffer of each pair the last verify read
  id<MTLBuffer> verify_logits_ = nil;  // [verify capacity, vocab]
  // Per Gated DeltaNet layer, the last verify's conv inputs and its
  // recurrence's rows [capacity, channels], betas and decays [capacity, Hv].
  struct VerifyRecord {
    id<MTLBuffer> raw = nil, rows = nil, beta = nil, decay = nil;
  };
  std::vector<VerifyRecord> records_;
  // Multi-token prediction (DecoderWeights::mtp): the head's layer is
  // layers_[c_.num_layers] when it runs here, and mtp_refused_ says why it
  // does not otherwise. mtp_h_ [chunk + 1, hidden] holds the hidden state
  // each of a pass's head rows starts from: row 0 the one the previous pass
  // left pending, rows 1 to n the pass's own final-normed rows. mtp_e_
  // [chunk, hidden] holds token embeddings, mtp_in_ [chunk, 2 * hidden] the
  // head's inputs, and mtp_logits_ a draft block's logits.
  bool mtp_on_ = false;
  std::string mtp_refused_;
  GpuLinear mtp_fc_;
  BufRef mtp_embedding_norm_, mtp_hidden_norm_, mtp_final_norm_;
  id<MTLBuffer> mtp_h_ = nil, mtp_e_ = nil, mtp_in_ = nil, mtp_logits_ = nil;
  int mtp_logits_rows_ = 0;
};

}  // namespace

bool metal_available() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}

std::string metal_device_name() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device ? std::string([[device name] UTF8String]) : std::string();
  }
}

std::unique_ptr<Model> make_decoder_metal(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options) {
  return std::make_unique<DecoderMetal>(std::move(weights), options);
}

std::vector<int32_t> metal_sample_rows(const std::vector<float>& logits, int vocab, const SamplingParams& params,
                                       int first_step, int steps) {
  validate_sampling(params);
  LK_CHECK(vocab > 0 && !logits.empty() && logits.size() % static_cast<size_t>(vocab) == 0,
           "metal sample: logits are not whole rows");
  LK_CHECK(first_step >= 0 && steps > 0, "metal sample: bad step range");
  std::vector<int32_t> out;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    LK_CHECK(device != nil, "metal: no Metal device");
    id<MTLLibrary> library = compile_kernels(device);
    const bool greedy = params.greedy();
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:greedy ? @"argmax_vocab" : @"sample_vocab"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function error:&error];
    LK_CHECK(pso != nil, "metal: sampling pipeline: " + ns_error(error));
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> rows = [device newBufferWithBytes:logits.data()
                                             length:logits.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> token = [device newBufferWithLength:sizeof(int32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> history = [device newBufferWithLength:sizeof(int32_t) * static_cast<size_t>(first_step + steps)
                                                options:MTLResourceStorageModeShared];
    const NSUInteger threads = std::min<NSUInteger>(1024, [pso maxTotalThreadsPerThreadgroup] / 32 * 32);
    const size_t row_count = logits.size() / static_cast<size_t>(vocab);
    for (size_t row = 0; row < row_count; ++row) {
      for (int step = first_step; step < first_step + steps; ++step) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:rows offset:row * static_cast<size_t>(vocab) * sizeof(float) atIndex:0];
        [enc setBuffer:token offset:0 atIndex:1];
        [enc setBuffer:history offset:0 atIndex:2];
        if (greedy) {
          const ArgmaxParams p{vocab, step};
          [enc setBytes:&p length:sizeof(p) atIndex:3];
        } else {
          const SampleParams p = sample_params(params, vocab, step);
          [enc setBytes:&p length:sizeof(p) atIndex:3];
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        LK_CHECK([cb status] == MTLCommandBufferStatusCompleted, "metal: command buffer failed: " + ns_error([cb error]));
        out.push_back(static_cast<const int32_t*>([history contents])[step]);
      }
    }
  }
  return out;
}

}  // namespace lokahi
