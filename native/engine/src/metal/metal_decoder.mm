// Metal backend for dense decoders.
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

namespace lokahi {
namespace {

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

  bool same_format(const GpuLinear& other) const {
    return bits == other.bits && group == other.group && bf16 == other.bf16 && cols == other.cols;
  }
};

struct GpuLayer {
  GpuLinear q, k, v, o, gate, up, down;
  // Norms and biases a family does not have are bound to a placeholder and
  // never read.
  BufRef attn_norm, attn_post_norm, mlp_norm, mlp_post_norm, q_norm, k_norm, q_bias, k_bias, v_bias;
  id<MTLBuffer> k_cache = nil;
  id<MTLBuffer> v_cache = nil;
  int capacity = 0;
  bool sliding = false;
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
struct ResidualNormParams { int cols; float eps; float offset; int post_norm; int has_next; };
struct QmvParams { int rows; int cols; };
struct GluParams { int rows; int cols; int silu; };
struct QkvParams { int q_rows; int kv_rows; int cols; };
struct QmmParams { int n; int rows; int cols; };
struct ElementwiseParams { int count; };
struct GluMulParams { int count; int silu; };
struct RopeParams {
  int heads; int kv_heads; int head_dim; int start_pos; int capacity; float eps; float offset; int qk_norm;
  int qkv_bias;
};
struct AttnParams {
  int heads; int kv_heads; int head_dim; int start_pos; int window; int capacity; int block; int n_blocks;
  float scale; float softcap;
};
struct ReduceParams { int heads; int head_dim; int n_blocks; };
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
constexpr int kInflightSteps = 2;

class DecoderMetal final : public Model {
 public:
  DecoderMetal(std::shared_ptr<DecoderWeights> weights, const LoadOptions& options)
      : w_(std::move(weights)), c_(w_->config), max_context_(options.max_context),
        chunk_(std::max(1, std::min(options.prefill_chunk, options.max_context))) {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      LK_CHECK(device_ != nil, "metal: no Metal device");
      queue_ = [device_ newCommandQueue];
      library_ = compile_kernels(device_);
      validate();
      upload();
      allocate_state();
    }
  }

  ~DecoderMetal() override { drain(); }

  const char* backend() const override { return "metal"; }
  const char* architecture() const override { return c_.architecture.c_str(); }
  int vocab_size() const override { return c_.vocab_size; }
  int max_context() const override { return max_context_; }
  int position() const override { return position_; }
  size_t weight_bytes() const override { return w_->matrix_bytes(); }
  const std::vector<int32_t>& eos_token_ids() const override { return c_.eos_token_ids; }

  void reset() override {
    drain();
    position_ = 0;
  }

  const float* prefill(const int32_t* tokens, int n) override {
    @autoreleasepool {
      drain();
      run_prefill(tokens, n);
      return static_cast<const float*>([logits_ contents]);
    }
  }

  const float* decode(int32_t token) override {
    @autoreleasepool {
      drain();
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

 private:
  // ---- setup --------------------------------------------------------------

  void validate() const {
    auto check = [&](const Linear& l) {
      LK_CHECK(l.quant.quantized(), "metal: " + l.name + " is not quantized; use --backend cpu");
      LK_CHECK(l.cols % 32 == 0 && l.quant.group_size % 32 == 0,
               "metal: " + l.name + " needs cols and group_size divisible by 32");
    };
    check(w_->embed);
    check(w_->lm_head);
    for (const auto& L : w_->layers) {
      for (const Linear* l : {&L.q_proj, &L.k_proj, &L.v_proj, &L.o_proj, &L.gate_proj, &L.up_proj,
                              &L.down_proj}) {
        check(*l);
      }
    }
    const int dpl = c_.head_dim / 32;
    LK_CHECK(c_.head_dim % 32 == 0 && (dpl == 2 || dpl == 4 || dpl == 8),
             "metal: head_dim must be 64, 128 or 256");
    const int group = c_.num_heads / c_.num_kv_heads;
    LK_CHECK(group >= 1 && group <= 8, "metal: heads per kv head must be between 1 and 8");
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
    // Bound in place of norms a family does not have; kernels never read it.
    placeholder_data_.assign(static_cast<size_t>(
                                 std::max({c_.hidden_size, c_.head_dim, c_.num_heads * c_.head_dim})),
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
    layers_.resize(w_->layers.size());
    for (size_t i = 0; i < w_->layers.size(); ++i) {
      const DecoderLayer& L = w_->layers[i];
      GpuLayer& G = layers_[i];
      G.q = add_linear(arena, L.q_proj, linear_fix, &G.q);
      G.k = add_linear(arena, L.k_proj, linear_fix, &G.k);
      G.v = add_linear(arena, L.v_proj, linear_fix, &G.v);
      G.o = add_linear(arena, L.o_proj, linear_fix, &G.o);
      G.gate = add_linear(arena, L.gate_proj, linear_fix, &G.gate);
      G.up = add_linear(arena, L.up_proj, linear_fix, &G.up);
      G.down = add_linear(arena, L.down_proj, linear_fix, &G.down);
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
    arena.finalize();
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
    const size_t H = static_cast<size_t>(c_.num_heads), KV = static_cast<size_t>(c_.num_kv_heads);
    const size_t D = static_cast<size_t>(c_.head_dim), hidden = static_cast<size_t>(c_.hidden_size);
    const size_t chunk = static_cast<size_t>(chunk_);
    for (size_t i = 0; i < layers_.size(); ++i) {
      GpuLayer& G = layers_[i];
      G.sliding = c_.layer_is_sliding[i];
      // Sliding layers keep window + chunk slots so a whole chunk can be
      // written before its queries read the preceding window.
      G.capacity = G.sliding ? std::min(max_context_, c_.sliding_window + chunk_) : max_context_;
      G.k_cache = private_buffer(KV * static_cast<size_t>(G.capacity) * D * f);
      G.v_cache = private_buffer(KV * static_cast<size_t>(G.capacity) * D * f);
    }
    x_ = private_buffer(chunk * hidden * f);
    h_ = private_buffer(chunk * hidden * f);
    r_ = private_buffer(chunk * hidden * f);
    q_ = private_buffer(chunk * H * D * f);
    kt_ = private_buffer(chunk * KV * D * f);
    vt_ = private_buffer(chunk * KV * D * f);
    attn_ = private_buffer(chunk * H * D * f);
    gate_ = private_buffer(chunk * static_cast<size_t>(c_.intermediate_size) * f);
    up_ = private_buffer(chunk * static_cast<size_t>(c_.intermediate_size) * f);
    const size_t partials = std::max(H * kDecodeMaxBlocks, chunk * H * kPrefillMaxBlocks);
    part_o_ = private_buffer(partials * D * f);
    part_ml_ = private_buffer(partials * 2 * f);
    logits_ = shared_buffer(static_cast<size_t>(c_.vocab_size) * f);
    token_ = shared_buffer(sizeof(int32_t));
    history_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
    prompt_ = shared_buffer(static_cast<size_t>(max_context_) * sizeof(int32_t));
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
                            (l.bf16 ? "bf16" : "f16");
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    int bits = l.bits, group = l.group;
    bool bf16 = l.bf16;
    [values setConstantValue:&bits type:MTLDataTypeInt atIndex:0];
    [values setConstantValue:&group type:MTLDataTypeInt atIndex:1];
    [values setConstantValue:&bf16 type:MTLDataTypeBool atIndex:2];
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
    [enc setComputePipelineState:quant_pipeline("qmm", l)];
    bind_weights(enc, l, 0);
    bind(enc, x, 0, 3);
    bind(enc, y, 0, 4);
    bytes(enc, QmmParams{n, l.rows, l.cols}, 5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((l.rows + 31) / 32),
                                          static_cast<NSUInteger>((n + 31) / 32), 1)
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
    id<MTLComputePipelineState> pso = pipeline("rms_norm_rows");
    [enc setComputePipelineState:pso];
    bind(enc, in, 0, 0);
    bind(enc, weight, 1);
    bind(enc, out, 0, 2);
    bytes(enc, NormParams{c_.hidden_size, c_.rms_norm_eps, c_.norm_offset}, 3);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(norm_threads(pso), 1, 1)];
  }

  void encode_residual_norm(id<MTLComputeCommandEncoder> enc, const BufRef& w_post, const BufRef* w_next, int n) {
    id<MTLComputePipelineState> pso = pipeline("residual_norm_rows");
    [enc setComputePipelineState:pso];
    bind(enc, x_, 0, 0);
    bind(enc, r_, 0, 1);
    bind(enc, w_post, 2);
    bind(enc, w_next ? *w_next : w_post, 3);
    bind(enc, h_, 0, 4);
    bytes(enc, ResidualNormParams{c_.hidden_size, c_.rms_norm_eps, c_.norm_offset, c_.sandwich_norms ? 1 : 0,
                                  w_next ? 1 : 0},
          5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(norm_threads(pso), 1, 1)];
  }

  void encode_qkv(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    if (n == 1 && G.q.same_format(G.k) && G.q.same_format(G.v)) {
      [enc setComputePipelineState:quant_pipeline("qmv_qkv", G.q)];
      bind_weights(enc, G.q, 0);
      bind_weights(enc, G.k, 3);
      bind_weights(enc, G.v, 6);
      bind(enc, h_, 0, 9);
      bind(enc, q_, 0, 10);
      bind(enc, kt_, 0, 11);
      bind(enc, vt_, 0, 12);
      bytes(enc, QkvParams{G.q.rows, G.k.rows, G.q.cols}, 13);
      [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(G.q.rows + 2 * G.k.rows), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      return;
    }
    encode_qmm(enc, G.q, h_, n, q_);
    encode_qmm(enc, G.k, h_, n, kt_);
    encode_qmm(enc, G.v, h_, n, vt_);
  }

  void encode_rope(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    [enc setComputePipelineState:pipeline("qk_norm_rope_store")];
    bind(enc, q_, 0, 0);
    bind(enc, kt_, 0, 1);
    bind(enc, vt_, 0, 2);
    bind(enc, G.k_cache, 0, 3);
    bind(enc, G.v_cache, 0, 4);
    bind(enc, G.q_norm, 5);
    bind(enc, G.k_norm, 6);
    bind(enc, G.sliding ? inv_freq_local_buf_ : inv_freq_global_buf_, 7);
    bytes(enc, RopeParams{c_.num_heads, c_.num_kv_heads, c_.head_dim, start, G.capacity, c_.rms_norm_eps,
                          c_.norm_offset, c_.qk_norm ? 1 : 0, c_.qkv_bias ? 1 : 0},
          8);
    bind(enc, G.q_bias, 9);
    bind(enc, G.k_bias, 10);
    bind(enc, G.v_bias, 11);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(c_.num_heads + c_.num_kv_heads),
                                          static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(c_.head_dim / 2), 1, 1)];
  }

  void encode_attention(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    const int window = G.sliding ? c_.sliding_window : (1 << 30);
    const int keys_max = std::min(window, start + n);  // keys seen by the last query
    const int max_blocks = n == 1 ? kDecodeMaxBlocks : kPrefillMaxBlocks;
    const int min_block = n == 1 ? 32 : 64;
    const int block = std::max(min_block, (keys_max + max_blocks - 1) / max_blocks);
    const int n_blocks = (keys_max + block - 1) / block;
    const int group = c_.num_heads / c_.num_kv_heads;
    const std::string name = "attn_partial_g" + std::to_string(group) + "_d" + std::to_string(c_.head_dim / 32);
    [enc setComputePipelineState:pipeline(name)];
    bind(enc, q_, 0, 0);
    bind(enc, G.k_cache, 0, 1);
    bind(enc, G.v_cache, 0, 2);
    bind(enc, part_o_, 0, 3);
    bind(enc, part_ml_, 0, 4);
    bytes(enc,
          AttnParams{c_.num_heads, c_.num_kv_heads, c_.head_dim, start, window, G.capacity, block, n_blocks,
                     c_.attn_scale, c_.attn_logit_softcapping},
          5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_blocks),
                                          static_cast<NSUInteger>(c_.num_kv_heads), static_cast<NSUInteger>(n))
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc setComputePipelineState:pipeline("attn_reduce")];
    bind(enc, part_o_, 0, 0);
    bind(enc, part_ml_, 0, 1);
    bind(enc, attn_, 0, 2);
    bytes(enc, ReduceParams{c_.num_heads, c_.head_dim, n_blocks}, 3);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(c_.num_heads), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(c_.head_dim), 1, 1)];
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
      const int count = n * c_.intermediate_size;
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
    section(enc, "embed", profiling ? traffic::embed(w_->embed, n) : 0.0);
    encode_rms_norm(enc, x_, layers_[0].attn_norm, h_, n);
    section(enc, "norm", profiling ? traffic::rows(hidden, n, 2) + 4.0 * hidden : 0.0);

    // The residual kernel adds the sublayer output (normalized first with
    // sandwich norms) and applies the next norm in one pass.
    const double residual_bytes = profiling ? traffic::rows(hidden, n, 4) + 8.0 * hidden : 0.0;
    for (size_t l = 0; l < layers_.size(); ++l) {
      const GpuLayer& G = layers_[l];
      const DecoderLayer& L = w_->layers[l];
      const int layer = static_cast<int>(l);
      encode_qkv(enc, G, n);
      section(enc, "qkv",
              profiling ? traffic::linear(L.q_proj, n) + traffic::linear(L.k_proj, n) + traffic::linear(L.v_proj, n)
                        : 0.0);
      encode_rope(enc, G, n, start);
      section(enc, "rope_kv",
              profiling ? traffic::rows(c_.head_dim, n, 2 * c_.num_heads + 4 * c_.num_kv_heads) : 0.0);
      encode_attention(enc, G, n, start);
      section(enc, "attention", profiling ? traffic::attention(c_, layer, start, n) : 0.0);
      encode_qmm(enc, G.o, attn_, n, r_);
      section(enc, "o_proj", profiling ? traffic::linear(L.o_proj, n) : 0.0);
      encode_residual_norm(enc, G.attn_post_norm, &G.mlp_norm, n);
      section(enc, "residual_norm", residual_bytes);
      encode_mlp(enc, G, n);
      section(enc, "mlp",
              profiling ? traffic::linear(L.gate_proj, n) + traffic::linear(L.up_proj, n) +
                              traffic::linear(L.down_proj, n) + traffic::rows(c_.intermediate_size, n, 1)
                        : 0.0);
      const BufRef* next = l + 1 < layers_.size() ? &layers_[l + 1].attn_norm : &final_norm_;
      encode_residual_norm(enc, G.mlp_post_norm, next, n);
      section(enc, "residual_norm", residual_bytes);
    }
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
    LK_CHECK(n > 0, "prefill requires at least one token");
    LK_CHECK(position_ + n <= max_context_, "context length exceeded");
    for (int i = 0; i < n; ++i) LK_CHECK(tokens[i] >= 0 && tokens[i] < c_.vocab_size, "token id out of range");
    std::memcpy([prompt_ contents], tokens, sizeof(int32_t) * static_cast<size_t>(n));
    active_cb_ = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [active_cb_ computeCommandEncoder];
    for (int begin = 0; begin < n; begin += chunk_) {
      const int count = std::min(chunk_, n - begin);
      encode_forward(enc, prompt_, sizeof(int32_t) * static_cast<size_t>(begin), count, position_ + begin,
                     begin + count == n, 0);
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
  int position_ = 0;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;

  GpuLinear embed_, lm_head_;
  bool tied_ = false;
  BufRef final_norm_, placeholder_, inv_freq_local_buf_, inv_freq_global_buf_;
  std::vector<float> inv_freq_local_, inv_freq_global_, placeholder_data_;
  std::vector<GpuLayer> layers_;
  size_t device_weight_bytes_ = 0;

  id<MTLBuffer> x_ = nil, h_ = nil, r_ = nil, q_ = nil, kt_ = nil, vt_ = nil, attn_ = nil, gate_ = nil, up_ = nil;
  id<MTLBuffer> part_o_ = nil, part_ml_ = nil;
  id<MTLBuffer> logits_ = nil, token_ = nil, history_ = nil, prompt_ = nil;
  std::vector<id<MTLCommandBuffer>> pending_;
  // The command buffer encode_forward is appending to (sections replace it
  // while profiling).
  id<MTLCommandBuffer> active_cb_ = nil;
  // How the final step picks a token; greedy outside generate().
  SamplingParams sampling_;
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
