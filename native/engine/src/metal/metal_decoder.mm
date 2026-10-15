// Metal backend for dense decoders (Gemma 3, Gemma 4 and the Llama family).
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
struct ElementwiseParams { int count; };
struct GluMulParams { int count; int silu; };
struct RopeParams {
  int heads; int kv_heads; int head_dim; int start_pos; int capacity; float eps; float offset; int qk_norm;
  int qkv_bias; int v_norm; int rotary;
};
struct PerLayerCombineParams { int size; int layers; float eps; float offset; float proj_scale; float input_scale; };
struct PerLayerGateParams { int size; int layers; int layer; int count; };
struct MoeRouteParams { int experts; int k; int topk_softmax; int norm_topk; int has_bias; };
struct GatherParams { int rows; int cols; int k; int input_rows; int act; int has_bias; float limit; };
struct MoeCombineParams { int cols; int k; int count; int has_shared; };
struct MoeGroupedParams { int experts; int k; int groups; int topk_groups; int norm_topk; float scaling; };
struct DenseParams { int rows; int cols; int dtype; };
struct AttnParams {
  int heads; int kv_heads; int head_dim; int start_pos; int window; int capacity; int block; int n_blocks;
  float scale; float softcap;
};
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
    LK_CHECK(c_.num_experts <= 256, "metal: at most 256 experts");
    for (int i = 0; i < c_.num_layers; ++i) {
      const DecoderLayer& L = w_->layers[static_cast<size_t>(i)];
      for (const Linear* l : {&L.q_proj, &L.o_proj}) check(*l);
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
        for (const auto* experts : {&L.expert_gate, &L.expert_up, &L.expert_down}) {
          for (const Linear& e : *experts) check_any(e);
        }
        const Linear &g = L.expert_gate[0], &u = L.expert_up[0];
        LK_CHECK(g.quant.bits == u.quant.bits && g.quant.group_size == u.quant.group_size &&
                     g.quant.mode == u.quant.mode && g.scale_dtype == u.scale_dtype,
                 "metal: expert gate and up projections must share a format");
      } else {
        for (const Linear* l : {&L.gate_proj, &L.up_proj, &L.down_proj}) check(*l);
      }
      if (!c_.shares_kv(i)) check(L.k_proj);
      if (!c_.shares_kv(i) && !c_.k_eq_v(i)) check(L.v_proj);
      if (c_.per_layer_input > 0) {
        check(L.per_layer_gate);
        check(L.per_layer_proj);
      }
      const int dim = c_.head_dim_of(i), dpl = dim / 32;
      LK_CHECK(dim % 32 == 0 && (dpl == 2 || dpl == 4 || dpl == 8 || dpl == 16),
               "metal: head_dim must be 64, 128, 256 or 512");
      const int group = c_.num_heads / c_.kv_heads_of(i);
      LK_CHECK(group >= 1 && group <= 8, "metal: heads per kv head must be between 1 and 8");
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
    layers_.resize(w_->layers.size());
    for (size_t i = 0; i < w_->layers.size(); ++i) {
      const DecoderLayer& L = w_->layers[i];
      GpuLayer& G = layers_[i];
      const int layer = static_cast<int>(i);
      G.head_dim = c_.head_dim_of(layer);
      G.kv_heads = c_.kv_heads_of(layer);
      G.inter = c_.intermediate_of(layer);
      G.own_kv = !c_.shares_kv(layer);
      G.k_eq_v = c_.k_eq_v(layer);
      G.scalar = L.layer_scalar;
      G.q = add_linear(arena, L.q_proj, linear_fix, &G.q);
      if (G.own_kv) G.k = add_linear(arena, L.k_proj, linear_fix, &G.k);
      if (G.own_kv && !G.k_eq_v) G.v = add_linear(arena, L.v_proj, linear_fix, &G.v);
      G.o = add_linear(arena, L.o_proj, linear_fix, &G.o);
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
      if (!G.moe) {
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
      G.k_cache = private_buffer(width * static_cast<size_t>(G.capacity) * f);
      G.v_cache = private_buffer(width * static_cast<size_t>(G.capacity) * f);
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

  void encode_qkv(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    if (!G.own_kv || G.k_eq_v) {  // queries only, or queries and the keys that double as values
      encode_qmm(enc, G.q, h_, n, q_);
      if (G.own_kv) encode_qmm(enc, G.k, h_, n, kt_);
      return;
    }
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
    bytes(enc, RopeParams{c_.num_heads, G.kv_heads, G.head_dim, start, G.capacity, c_.rms_norm_eps,
                          c_.norm_offset, c_.qk_norm ? 1 : 0, c_.qkv_bias ? 1 : 0, c_.v_norm ? 1 : 0,
                          c_.rotary_dim > 0 ? c_.rotary_dim : G.head_dim},
          8);
    bind(enc, G.q_bias, 9);
    bind(enc, G.k_bias, 10);
    bind(enc, G.v_bias, 11);
    if (G.k_eq_v) bind(enc, kt_, 0, 2);  // values are the raw keys, before k_norm
    // KV-shared layers only rotate their queries.
    const int heads = c_.num_heads + (G.own_kv ? G.kv_heads : 0);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(heads), static_cast<NSUInteger>(n), 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(G.head_dim / 2), 1, 1)];
  }

  void encode_attention(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n, int start) {
    const int window = G.sliding ? c_.sliding_window : (1 << 30);
    const int keys_max = std::min(window, start + n);  // keys seen by the last query
    const int max_blocks = n == 1 ? kDecodeMaxBlocks : kPrefillMaxBlocks;
    const int min_block = n == 1 ? 32 : 64;
    const int block = std::max(min_block, (keys_max + max_blocks - 1) / max_blocks);
    const int n_blocks = (keys_max + block - 1) / block;
    const int group = c_.num_heads / G.kv_heads;
    const std::string name = "attn_partial_g" + std::to_string(group) + "_d" + std::to_string(G.head_dim / 32);
    [enc setComputePipelineState:pipeline(name)];
    bind(enc, q_, 0, 0);
    bind(enc, G.k_cache, 0, 1);
    bind(enc, G.v_cache, 0, 2);
    bind(enc, part_o_, 0, 3);
    bind(enc, part_ml_, 0, 4);
    bytes(enc,
          AttnParams{c_.num_heads, G.kv_heads, G.head_dim, start, window, G.capacity, block, n_blocks,
                     c_.attn_scale, c_.attn_logit_softcapping},
          5);
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

  // r_ = the mixture-of-experts output for the normed rows in h_. Routing
  // runs on the GPU, so the host never waits for the chosen experts.
  void encode_router(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
    if (G.router_dense) {
      [enc setComputePipelineState:pipeline("dense_rows")];
      bind(enc, G.dense_router.data, 0);
      bind(enc, h_, 0, 1);
      bind(enc, route_, 0, 2);
      bytes(enc, DenseParams{G.dense_router.rows, G.dense_router.cols, G.dense_router.dtype}, 3);
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((G.dense_router.rows + 1) / 2),
                                            static_cast<NSUInteger>(n), 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      encode_qmm(enc, G.router, h_, n, route_);
    }
    if (c_.router == DecoderConfig::Router::SigmoidGrouped) {
      [enc setComputePipelineState:pipeline("moe_route_grouped")];
      bind(enc, route_, 0, 0);
      bind(enc, chosen_, 0, 1);
      bind(enc, expert_weights_, 0, 2);
      bytes(enc, MoeGroupedParams{c_.num_experts, c_.experts_per_token, c_.n_group, c_.topk_group,
                                  c_.norm_topk_prob ? 1 : 0, c_.routed_scaling},
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
                              c_.router_bias ? 1 : 0},
          3);
    bind(enc, G.router_bias, 4);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  }

  void encode_experts(id<MTLComputeCommandEncoder> enc, const GpuLayer& G, int n) {
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
    bind(enc, h_, 0, 6);
    bind(enc, expert_hidden_, 0, 7);
    bytes(enc, GatherParams{I, c_.hidden_size, K, 1, act, has_bias, c_.swiglu_limit}, 8);
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
    bytes(enc, GatherParams{c_.hidden_size, I, K, 0, 0, has_bias, c_.swiglu_limit}, 5);
    bind(enc, chosen_, 0, 6);
    bind(enc, G.expert_down.bias, 7);
    [enc dispatchThreadgroups:MTLSizeMake(qmv_groups(c_.hidden_size), static_cast<NSUInteger>(pairs), 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    const bool shared = c_.shared_intermediate > 0;
    if (shared) encode_shared_expert(enc, G, n);
    [enc setComputePipelineState:pipeline("moe_combine")];
    bind(enc, expert_out_, 0, 0);
    bind(enc, expert_weights_, 0, 1);
    bind(enc, r_, 0, 2);
    const int count = n * c_.hidden_size;
    bytes(enc, MoeCombineParams{c_.hidden_size, K, count, shared ? 1 : 0}, 3);
    bind(enc, shared ? shared_out_ : r_, 0, 4);
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
      bytes += traffic::linear(L.shared_gate, n) + traffic::linear(L.shared_up, n) + traffic::linear(L.shared_down, n);
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
    if (c_.per_layer_input > 0) {
      encode_per_layer_inputs(enc, tokens, tokens_offset, n);
      const int64_t width = static_cast<int64_t>(c_.num_layers) * c_.per_layer_input;
      section(enc, "per_layer_input",
              profiling ? traffic::linear(w_->per_layer_model_proj, n) + traffic::embed(w_->embed_per_layer, n) +
                              traffic::rows(width, n, 2)
                        : 0.0);
    }
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
              profiling ? traffic::linear(L.q_proj, n) + (G.own_kv ? traffic::linear(L.k_proj, n) : 0.0) +
                              (G.own_kv && !G.k_eq_v ? traffic::linear(L.v_proj, n) : 0.0)
                        : 0.0);
      encode_rope(enc, G, n, start);
      section(enc, "rope_kv",
              profiling ? traffic::rows(G.head_dim, n, 2 * c_.num_heads + (G.own_kv ? 4 * G.kv_heads : 0)) : 0.0);
      encode_attention(enc, G, n, start);
      section(enc, "attention", profiling ? traffic::attention(c_, layer, start, n) : 0.0);
      encode_qmm(enc, G.o, attn_, n, r_);
      if (c_.o_bias) {
        [enc setComputePipelineState:pipeline("add_bias_rows")];
        bind(enc, r_, 0, 0);
        bind(enc, G.o_bias, 1);
        const int count = n * c_.hidden_size;
        bytes(enc, BiasParams{c_.hidden_size, count}, 2);
        encode_elementwise(enc, count);
      }
      section(enc, "o_proj", profiling ? traffic::linear(L.o_proj, n) : 0.0);
      encode_residual_norm(enc, G.attn_post_norm, &G.mlp_norm, n);
      section(enc, "residual_norm", residual_bytes);
      if (G.moe) {
        encode_router(enc, G, n);
        section(enc, "router", profiling ? traffic::linear(L.router, n) + traffic::rows(c_.num_experts, n, 1) : 0.0);
        // While profiling the router has finished here, so its choices are readable.
        const double expert_bytes = profiling ? expert_traffic(L, n) : 0.0;
        encode_experts(enc, G, n);
        section(enc, "experts", expert_bytes);
      } else {
        encode_mlp(enc, G, n);
        section(enc, "mlp",
                profiling ? traffic::linear(L.gate_proj, n) + traffic::linear(L.up_proj, n) +
                                traffic::linear(L.down_proj, n) + traffic::rows(G.inter, n, 1)
                          : 0.0);
      }
      const BufRef* next = l + 1 < layers_.size() ? &layers_[l + 1].attn_norm : &final_norm_;
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
      } else {
        encode_residual_norm(enc, G.mlp_post_norm, next, n, G.scalar);
      }
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

  GpuLinear embed_, lm_head_, embed_per_layer_, per_layer_model_proj_;
  bool tied_ = false;
  BufRef final_norm_, placeholder_, inv_freq_local_buf_, inv_freq_global_buf_, per_layer_proj_norm_;
  int max_head_dim_ = 0, max_kv_width_ = 0, max_inter_ = 0;  // the largest over layers
  std::vector<float> inv_freq_local_, inv_freq_global_, placeholder_data_;
  std::vector<GpuLayer> layers_;
  size_t device_weight_bytes_ = 0;

  id<MTLBuffer> x_ = nil, h_ = nil, r_ = nil, q_ = nil, kt_ = nil, vt_ = nil, attn_ = nil, gate_ = nil, up_ = nil;
  id<MTLBuffer> part_o_ = nil, part_ml_ = nil;
  // Mixture of experts: router logits [chunk, E], each token's experts and
  // weights [chunk, K], and each slot's expert activations and outputs.
  id<MTLBuffer> route_ = nil, chosen_ = nil, expert_weights_ = nil, expert_hidden_ = nil, expert_out_ = nil;
  // Host copies of expert stacks, alive until the arena has copied them.
  std::deque<std::vector<uint8_t>> staged_;
  std::vector<std::pair<GpuExperts*, std::array<size_t, 3>>> expert_fix_;
  std::vector<std::pair<BufRef*, size_t>> dense_fix_;
  id<MTLBuffer> shared_out_ = nil;  // GLM's shared expert output [chunk, hidden]
  // Gemma 4's per-layer inputs [chunk, layers * P], their two parts, and one
  // layer's gate [chunk, P].
  id<MTLBuffer> per_layer_ = nil, per_layer_lookup_ = nil, per_layer_proj_ = nil, per_layer_gate_ = nil;
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
