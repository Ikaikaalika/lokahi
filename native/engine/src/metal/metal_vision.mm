// Metal backend for the vision encoders: Gemma 3's SigLIP tower and
// projector, Gemma 4's tower and Qwen3.5's, one command buffer per image.
//
// Every matrix product, including attention's (scores = Q K^T per head, then
// a row softmax and P V), runs through dense_mm. That kernel reads the
// checkpoint's dense F32, F16 or BF16 weights as they are, and the
// activations stay float32 throughout.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/metal_vision.h"

#include "gemma4_vision.h"
#include "qwen3_5_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "metal/metal_common.h"

namespace lokahi {
namespace {

// Parameter blocks; layouts match the structs in decoder_kernels.metal.
struct MmParams { int m; int n; int k; int lda; int ldb; int ldc; float alpha; int has_bias; int accumulate; };
struct LayerNormParams { int cols; float eps; };
struct NormParams { int cols; float eps; float offset; };
struct ElementwiseParams { int count; };
struct PoolParams { int cols; int side; int tokens_side; };
struct GluMulParams { int count; int silu; };
struct WindowAttentionParams { int dim; int stride; int hidden; float scale; };
struct SplitHeadsParams { int heads; int dim; int stride; int rows; };
struct AttnMmParams {
  int heads; int kv_heads; int head_dim; int start_pos; int row0; int rows; int key_base; int keys; int window;
  int capacity; float scale; float softcap; int has_last_key; int head_rows; int block_keys; int has_bounds;
  int batched;
};
constexpr int kMaxWindowKeys = 1024;  // as decoder_kernels.metal
// One page of the placeholder buffer bound for unused optional inputs.
constexpr size_t kDummyBytes = 16384;
// The Qwen tower's attention scores held at once by default, in floats
// (128 MiB): a 1344 x 1344 Qwen2.5-VL image has 9216 patches, which would
// need 324 MiB for one head's full score matrix.
constexpr size_t kScoreBudget = size_t{1} << 25;

// A matrix on the device with the dense_mm variant that reads it.
struct GpuMatrix {
  id<MTLBuffer> buffer = nil;
  std::string dtype;  // f32, f16 or bf16
};

// The dense_mm variant that reads a dense matrix of this dtype, or nullptr.
const char* mm_variant(DType dtype) {
  switch (dtype) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    default: return nullptr;
  }
}

// The device, pipelines and dispatch helpers both encoders use.
class MetalKit {
 protected:
  MetalKit() {
    device_ = MTLCreateSystemDefaultDevice();
    LK_CHECK(device_ != nil, "metal: no Metal device");
    queue_ = [device_ newCommandQueue];
    library_ = compile_kernels(device_);
    check_fragment_layout(device_, queue_, library_);
    // Bound where a kernel's optional input is unused, and filled with NaN so
    // a kernel that reads it anyway (a bias flagged present but nil) fails
    // every check instead of adding whatever the recycled page held.
    dummy_ = shared_buffer(kDummyBytes);
    std::fill_n(static_cast<float*>([dummy_ contents]), kDummyBytes / sizeof(float), std::nanf(""));
  }

  id<MTLBuffer> buffer_of(const void* data, size_t bytes) {
    id<MTLBuffer> buffer = [device_ newBufferWithBytes:data length:std::max<size_t>(bytes, 16)
                                               options:MTLResourceStorageModeShared];
    LK_CHECK(buffer != nil, "metal: allocation failed");
    return buffer;
  }
  id<MTLBuffer> buffer_of(const std::vector<float>& values) {
    return buffer_of(values.data(), sizeof(float) * values.size());
  }
  id<MTLBuffer> shared_buffer(size_t bytes) {
    id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<size_t>(bytes, 16)
                                                options:MTLResourceStorageModeShared];
    LK_CHECK(buffer != nil, "metal: allocation failed");
    return buffer;
  }
  id<MTLBuffer> private_buffer(size_t bytes) {
    id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<size_t>(bytes, 16)
                                                options:MTLResourceStorageModePrivate];
    LK_CHECK(buffer != nil, "metal: allocation failed");
    return buffer;
  }
  GpuMatrix matrix_of(const Linear& linear) {
    return GpuMatrix{buffer_of(linear.dense, linear.weight_bytes()), mm_variant(linear.dense_dtype)};
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

  template <typename T>
  static void bytes(id<MTLComputeCommandEncoder> enc, const T& value, NSUInteger index) {
    [enc setBytes:&value length:sizeof(T) atIndex:index];
  }

  // C = alpha * A B' (+ bias) (+ C); offsets in floats (or B's elements).
  void mm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, size_t a_offset, const GpuMatrix& b, size_t b_offset,
          bool trans_b, id<MTLBuffer> c, size_t c_offset, MmParams p, id<MTLBuffer> bias = nil) {
    const size_t element = b.dtype == "f32" ? sizeof(float) : sizeof(uint16_t);
    [enc setComputePipelineState:pipeline("dense_mm_" + b.dtype + (trans_b ? "_nt" : "_nn"))];
    [enc setBuffer:a offset:a_offset * sizeof(float) atIndex:0];
    [enc setBuffer:b.buffer offset:b_offset * element atIndex:1];
    [enc setBuffer:c offset:c_offset * sizeof(float) atIndex:2];
    bytes(enc, p, 3);
    [enc setBuffer:(bias != nil ? bias : dummy_) offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((p.n + 31) / 32), static_cast<NSUInteger>((p.m + 31) / 32), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }

  // An elementwise kernel over `count` floats whose parameters sit at `index`.
  void elementwise(id<MTLComputeCommandEncoder> enc, const char* name, int count, NSUInteger index) {
    [enc setComputePipelineState:pipeline(name)];
    bytes(enc, ElementwiseParams{count}, index);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  // Encodes one command buffer with `body`, runs it and waits.
  template <typename Body>
  void run(Body&& body) {
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    body(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    LK_CHECK([cb status] == MTLCommandBufferStatusCompleted, "metal: command buffer failed: " + ns_error([cb error]));
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;
  id<MTLBuffer> dummy_ = nil;
};

class SiglipMetal final : public SiglipEncoder, private MetalKit {
 public:
  explicit SiglipMetal(std::shared_ptr<const SiglipWeights> weights) : SiglipEncoder(std::move(weights)) {
    @autoreleasepool {
      LK_CHECK(metal_siglip_supports(*weights_), "metal: the vision tower has quantized matrices; use --backend cpu");
      upload();
      allocate();
    }
  }

  const char* backend() const override { return "metal"; }

 protected:
  std::vector<float> encode_pixels(const float* pixels) override {
    @autoreleasepool {
      const SiglipConfig& c = config();
      const std::vector<float> rows = patches(pixels);
      std::memcpy([patches_ contents], rows.data(), sizeof(float) * rows.size());
      run([&](id<MTLComputeCommandEncoder> enc) { encode_tower(enc); });
      const float* out = static_cast<const float*>([features_ contents]);
      return std::vector<float>(out, out + static_cast<size_t>(c.tokens_per_image) * c.text_hidden_size);
    }
  }

 private:
  struct Layer {
    id<MTLBuffer> ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    GpuMatrix q, k, v, out, fc1, fc2;
    id<MTLBuffer> q_bias, k_bias, v_bias, out_bias, fc1_bias, fc2_bias;
  };

  void upload() {
    const SiglipWeights& w = *weights_;
    patch_ = GpuMatrix{buffer_of(w.patch_weight), "f32"};
    patch_bias_ = buffer_of(w.patch_bias);
    positions_ = buffer_of(w.positions);
    for (const SiglipWeights::Layer& L : w.layers) {
      Layer G;
      G.ln1_weight = buffer_of(L.ln1_weight);
      G.ln1_bias = buffer_of(L.ln1_bias);
      G.ln2_weight = buffer_of(L.ln2_weight);
      G.ln2_bias = buffer_of(L.ln2_bias);
      G.q = matrix_of(L.q_proj);
      G.k = matrix_of(L.k_proj);
      G.v = matrix_of(L.v_proj);
      G.out = matrix_of(L.out_proj);
      G.fc1 = matrix_of(L.fc1);
      G.fc2 = matrix_of(L.fc2);
      G.q_bias = buffer_of(L.q_bias);
      G.k_bias = buffer_of(L.k_bias);
      G.v_bias = buffer_of(L.v_bias);
      G.out_bias = buffer_of(L.out_bias);
      G.fc1_bias = buffer_of(L.fc1_bias);
      G.fc2_bias = buffer_of(L.fc2_bias);
      layers_.push_back(G);
    }
    post_weight_ = buffer_of(w.post_weight);
    post_bias_ = buffer_of(w.post_bias);
    soft_norm_ = buffer_of(w.soft_norm);
    projection_ = GpuMatrix{buffer_of(w.projection->data, w.projection->nbytes), mm_variant(w.projection->dtype)};
  }

  void allocate() {
    const SiglipConfig& c = config();
    const size_t f = sizeof(float), N = static_cast<size_t>(c.patches());
    const size_t hidden = static_cast<size_t>(c.hidden_size), T = static_cast<size_t>(c.tokens_per_image);
    const size_t width = static_cast<size_t>(c.patch_size) * c.patch_size * 3;
    patches_ = shared_buffer(N * width * f);
    x_ = private_buffer(N * hidden * f);
    h_ = private_buffer(N * hidden * f);
    q_ = private_buffer(N * hidden * f);
    k_ = private_buffer(N * hidden * f);
    v_ = private_buffer(N * hidden * f);
    attn_ = private_buffer(N * hidden * f);
    f_ = private_buffer(N * static_cast<size_t>(c.intermediate_size) * f);
    scores_ = private_buffer(N * N * f);  // one head's [patches, patches]
    pooled_ = private_buffer(T * hidden * f);
    normed_ = private_buffer(T * hidden * f);
    features_ = shared_buffer(T * static_cast<size_t>(c.text_hidden_size) * f);
  }

  // A [m, k] times a [n, k] weight, plus its bias, into C (or onto it).
  void linear(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, const GpuMatrix& w, id<MTLBuffer> bias,
              id<MTLBuffer> c, int m, int n, int k, bool accumulate) {
    mm(enc, a, 0, w, 0, true, c, 0, MmParams{m, n, k, k, k, n, 1.0f, bias != nil ? 1 : 0, accumulate ? 1 : 0}, bias);
  }

  void layer_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> weight, id<MTLBuffer> bias,
                  id<MTLBuffer> out, int rows) {
    const SiglipConfig& c = config();
    [enc setComputePipelineState:pipeline("layer_norm_rows")];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:bias offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    bytes(enc, LayerNormParams{c.hidden_size, c.layer_norm_eps}, 4);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  void encode_tower(id<MTLComputeCommandEncoder> enc) {
    const SiglipConfig& c = config();
    const int N = c.patches(), hidden = c.hidden_size, inter = c.intermediate_size;
    const int heads = c.num_heads, D = hidden / heads, T = c.tokens_per_image;
    const int width = c.patch_size * c.patch_size * 3;

    // x = positions, then x += patches W^T + bias.
    [enc setBuffer:positions_ offset:0 atIndex:0];
    [enc setBuffer:x_ offset:0 atIndex:1];
    elementwise(enc, "copy_floats", N * hidden, 2);
    linear(enc, patches_, patch_, patch_bias_, x_, N, hidden, width, true);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const GpuMatrix keys{k_, "f32"}, values{v_, "f32"};
    for (const Layer& L : layers_) {
      layer_norm(enc, x_, L.ln1_weight, L.ln1_bias, h_, N);
      linear(enc, h_, L.q, L.q_bias, q_, N, hidden, hidden, false);
      linear(enc, h_, L.k, L.k_bias, k_, N, hidden, hidden, false);
      linear(enc, h_, L.v, L.v_bias, v_, N, hidden, hidden, false);
      // Bidirectional attention, one head at a time through scores_.
      for (int head = 0; head < heads; ++head) {
        const size_t column = static_cast<size_t>(head) * D;
        mm(enc, q_, column, keys, column, true, scores_, 0, MmParams{N, N, D, hidden, hidden, N, scale, 0, 0});
        [enc setComputePipelineState:pipeline("softmax_rows")];
        [enc setBuffer:scores_ offset:0 atIndex:0];
        bytes(enc, ElementwiseParams{N}, 1);
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(N), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        mm(enc, scores_, 0, values, column, false, attn_, column, MmParams{N, D, N, N, hidden, hidden, 1.0f, 0, 0});
      }
      linear(enc, attn_, L.out, L.out_bias, x_, N, hidden, hidden, true);
      layer_norm(enc, x_, L.ln2_weight, L.ln2_bias, h_, N);
      linear(enc, h_, L.fc1, L.fc1_bias, f_, N, inter, hidden, false);
      [enc setBuffer:f_ offset:0 atIndex:0];
      elementwise(enc, "gelu_inplace", N * inter, 1);
      linear(enc, f_, L.fc2, L.fc2_bias, x_, N, hidden, inter, true);
    }
    layer_norm(enc, x_, post_weight_, post_bias_, h_, N);

    // Pool the patch grid, then Gemma's RMSNorm (1 + w) and the projection
    // [hidden, text_hidden] (stored [k, n]).
    [enc setComputePipelineState:pipeline("avg_pool_tokens")];
    [enc setBuffer:h_ offset:0 atIndex:0];
    [enc setBuffer:pooled_ offset:0 atIndex:1];
    bytes(enc, PoolParams{hidden, c.patches_per_side(), c.tokens_per_side()}, 2);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(hidden), static_cast<NSUInteger>(T), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc setComputePipelineState:pipeline("rms_norm_rows")];
    [enc setBuffer:pooled_ offset:0 atIndex:0];
    [enc setBuffer:soft_norm_ offset:0 atIndex:1];
    [enc setBuffer:normed_ offset:0 atIndex:2];
    bytes(enc, NormParams{hidden, c.layer_norm_eps, 1.0f}, 3);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(T), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    const int text_hidden = c.text_hidden_size;
    mm(enc, normed_, 0, projection_, 0, false, features_, 0,
       MmParams{T, text_hidden, hidden, hidden, text_hidden, text_hidden, 1.0f, 0, 0});
  }

  GpuMatrix patch_, projection_;
  id<MTLBuffer> patch_bias_ = nil, positions_ = nil, post_weight_ = nil, post_bias_ = nil, soft_norm_ = nil;
  std::vector<Layer> layers_;
  // Patches (shared, written per image), activations, one head's scores, the
  // pooled tokens and the features (shared, read back).
  id<MTLBuffer> patches_ = nil, x_ = nil, h_ = nil, q_ = nil, k_ = nil, v_ = nil, attn_ = nil, f_ = nil;
  id<MTLBuffer> scores_ = nil, pooled_ = nil, normed_ = nil, features_ = nil;
};

// Gemma 4's tower, sized for the processor's largest image (max_patches).
class Gemma4VisionMetal final : public Gemma4VisionEncoder, private MetalKit {
 public:
  explicit Gemma4VisionMetal(std::shared_ptr<const Gemma4VisionWeights> weights)
      : Gemma4VisionEncoder(std::move(weights)) {
    @autoreleasepool {
      LK_CHECK(metal_gemma4_vision_supports(*weights_),
               "metal: the vision tower has quantized matrices; use --backend cpu");
      upload();
      allocate();
    }
  }

  const char* backend() const override { return "metal"; }

  std::vector<float> encode(const Image& image) override {
    @autoreleasepool {
      const Gemma4VisionConfig& c = config();
      const Gemma4VisionWeights& w = *weights_;
      const int tokens = tokens_for(image.width, image.height);
      const int cols = image.width / c.patch_size, rows = image.height / c.patch_size, N = rows * cols;
      const int hidden = c.hidden_size, freqs = c.head_dim / 4;
      // Host side: the centered patches, each patch's x + y position rows
      // (into x_, which the projection then accumulates onto), the (x, y)
      // pairs, and cos/sin tables in double.
      const std::vector<float> rows_data = patches(image);
      std::memcpy([patches_ contents], rows_data.data(), sizeof(float) * rows_data.size());
      float* x = static_cast<float*>([x_ contents]);
      int* xy = static_cast<int*>([xy_ contents]);
      std::vector<float> row(static_cast<size_t>(hidden));
      for (int t = 0; t < N; ++t) {
        xy[2 * t] = t % cols;
        xy[2 * t + 1] = t / cols;
        float* out = x + static_cast<size_t>(t) * hidden;
        w.position_row(0, t % cols, out);
        w.position_row(1, t / cols, row.data());
        for (int i = 0; i < hidden; ++i) out[i] += row[static_cast<size_t>(i)];
      }
      float* cos_table = static_cast<float*>([cos_ contents]);
      float* sin_table = static_cast<float*>([sin_ contents]);
      for (int pos = 0; pos < std::max(rows, cols); ++pos) {
        for (int i = 0; i < freqs; ++i) {
          const double angle = pos / std::pow(c.rope_theta, 2.0 * i / (c.head_dim / 2));
          cos_table[pos * freqs + i] = static_cast<float>(std::cos(angle));
          sin_table[pos * freqs + i] = static_cast<float>(std::sin(angle));
        }
      }
      run([&](id<MTLComputeCommandEncoder> enc) { encode_tower(enc, rows, cols, tokens); });
      const float* out = static_cast<const float*>([features_ contents]);
      return std::vector<float>(out, out + static_cast<size_t>(tokens) * c.text_hidden_size);
    }
  }

 private:
  struct Layer {
    id<MTLBuffer> input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm, q_norm, k_norm;
    GpuMatrix q, k, v, o, gate, up, down;
  };
  struct ResidualNormParams { int cols; float eps; float offset; int post_norm; int has_next; float scale; };
  struct GluMulParams { int count; int silu; };
  struct AxialRopeParams { int heads; int dim; int freqs; };
  struct PoolStandardizeParams {
    int cols; int patch_cols; int pooled_cols; int kernel_size; int tokens; float scale; int standardize;
  };

  void upload() {
    const Gemma4VisionWeights& w = *weights_;
    const Gemma4VisionConfig& c = config();
    input_proj_ = matrix_of(w.input_proj);
    for (const Gemma4VisionWeights::Layer& L : w.layers) {
      layers_.push_back(Layer{buffer_of(L.input_norm), buffer_of(L.post_attention_norm),
                              buffer_of(L.pre_feedforward_norm), buffer_of(L.post_feedforward_norm),
                              buffer_of(L.q_norm), buffer_of(L.k_norm), matrix_of(L.q_proj), matrix_of(L.k_proj),
                              matrix_of(L.v_proj), matrix_of(L.o_proj), matrix_of(L.gate_proj),
                              matrix_of(L.up_proj), matrix_of(L.down_proj)});
    }
    std_bias_ = c.standardize ? buffer_of(w.std_bias) : dummy_;
    std_scale_ = c.standardize ? buffer_of(w.std_scale) : dummy_;
    // The embedder's projection is quantized in published checkpoints: read
    // it as float32 rows.
    std::vector<float> projection(static_cast<size_t>(w.projection.rows * w.projection.cols));
    for (int64_t r = 0; r < w.projection.rows; ++r) {
      dequantize_row(w.projection, r, projection.data() + r * w.projection.cols);
    }
    projection_ = GpuMatrix{buffer_of(projection), "f32"};
    // RMSNorm without a scale is (1 + 0) * x.
    zeros_ = buffer_of(std::vector<float>(static_cast<size_t>(std::max(c.hidden_size, c.head_dim)), 0.0f));
  }

  void allocate() {
    const Gemma4VisionConfig& c = config();
    const size_t f = sizeof(float), N = static_cast<size_t>(c.max_patches());
    const size_t hidden = static_cast<size_t>(c.hidden_size), width = static_cast<size_t>(c.num_heads) * c.head_dim;
    const size_t T = static_cast<size_t>(c.max_soft_tokens), freqs = static_cast<size_t>(c.head_dim / 4);
    patches_ = shared_buffer(N * static_cast<size_t>(3 * c.patch_size * c.patch_size) * f);
    x_ = shared_buffer(N * hidden * f);  // the host writes the position rows
    xy_ = shared_buffer(N * 2 * sizeof(int32_t));
    cos_ = shared_buffer(N * freqs * f);
    sin_ = shared_buffer(N * freqs * f);
    h_ = private_buffer(N * hidden * f);
    r_ = private_buffer(N * hidden * f);
    q_ = private_buffer(N * width * f);
    k_ = private_buffer(N * width * f);
    v_ = private_buffer(N * width * f);
    attn_ = private_buffer(N * width * f);
    gate_ = private_buffer(N * static_cast<size_t>(c.intermediate_size) * f);
    up_ = private_buffer(N * static_cast<size_t>(c.intermediate_size) * f);
    scores_ = private_buffer(N * N * f);
    pooled_ = private_buffer(T * hidden * f);
    features_ = shared_buffer(T * static_cast<size_t>(c.text_hidden_size) * f);
  }

  // rows of `cols` floats normalized by RMS times (offset + weight), in place or not.
  void rms_rows(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> weight, float offset,
                id<MTLBuffer> out, int rows, int cols) {
    [enc setComputePipelineState:pipeline("rms_norm_rows")];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    bytes(enc, NormParams{cols, config().rms_norm_eps, offset}, 3);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(std::min(256, (cols + 31) / 32 * 32)), 1, 1)];
  }

  // x += norm(r) * w_post, then (if w_next) h = norm(x) * w_next.
  void residual(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> w_post, id<MTLBuffer> w_next, int rows) {
    const Gemma4VisionConfig& c = config();
    [enc setComputePipelineState:pipeline("residual_norm_rows")];
    [enc setBuffer:x_ offset:0 atIndex:0];
    [enc setBuffer:r_ offset:0 atIndex:1];
    [enc setBuffer:w_post offset:0 atIndex:2];
    [enc setBuffer:(w_next != nil ? w_next : dummy_) offset:0 atIndex:3];
    [enc setBuffer:h_ offset:0 atIndex:4];
    bytes(enc, ResidualNormParams{c.hidden_size, c.rms_norm_eps, 0.0f, 1, w_next != nil ? 1 : 0, 1.0f}, 5);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  void rope(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> rows, int N) {
    const Gemma4VisionConfig& c = config();
    [enc setComputePipelineState:pipeline("rope_axial")];
    [enc setBuffer:rows offset:0 atIndex:0];
    [enc setBuffer:xy_ offset:0 atIndex:1];
    [enc setBuffer:cos_ offset:0 atIndex:2];
    [enc setBuffer:sin_ offset:0 atIndex:3];
    bytes(enc, AxialRopeParams{c.num_heads, c.head_dim, c.head_dim / 4}, 4);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(c.head_dim / 2), static_cast<NSUInteger>(c.num_heads),
                                     static_cast<NSUInteger>(N))
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(std::min(c.head_dim / 2, 64)), 1, 1)];
  }

  void encode_tower(id<MTLComputeCommandEncoder> enc, int rows, int cols, int tokens) {
    const Gemma4VisionConfig& c = config();
    const int N = rows * cols, hidden = c.hidden_size, inter = c.intermediate_size;
    const int heads = c.num_heads, D = c.head_dim, width = heads * D;
    const int patch_width = 3 * c.patch_size * c.patch_size;
    const GpuMatrix keys{k_, "f32"}, values{v_, "f32"};

    // x (the position rows) += patches input_proj^T, then h = input_norm(x).
    mm(enc, patches_, 0, input_proj_, 0, true, x_, 0, MmParams{N, hidden, patch_width, patch_width, patch_width, hidden,
                                                               1.0f, 0, 1});
    rms_rows(enc, x_, layers_[0].input_norm, 0.0f, h_, N, hidden);
    for (size_t l = 0; l < layers_.size(); ++l) {
      const Layer& L = layers_[l];
      for (auto [w, out] : {std::pair{&L.q, q_}, std::pair{&L.k, k_}, std::pair{&L.v, v_}}) {
        mm(enc, h_, 0, *w, 0, true, out, 0, MmParams{N, width, hidden, hidden, hidden, width, 1.0f, 0, 0});
      }
      rms_rows(enc, q_, L.q_norm, 0.0f, q_, N * heads, D);
      rms_rows(enc, k_, L.k_norm, 0.0f, k_, N * heads, D);
      rms_rows(enc, v_, zeros_, 1.0f, v_, N * heads, D);
      rope(enc, q_, N);
      rope(enc, k_, N);
      // Bidirectional attention with scale 1, one head at a time.
      for (int head = 0; head < heads; ++head) {
        const size_t column = static_cast<size_t>(head) * D;
        mm(enc, q_, column, keys, column, true, scores_, 0, MmParams{N, N, D, width, width, N, 1.0f, 0, 0});
        [enc setComputePipelineState:pipeline("softmax_rows")];
        [enc setBuffer:scores_ offset:0 atIndex:0];
        bytes(enc, ElementwiseParams{N}, 1);
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(N), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        mm(enc, scores_, 0, values, column, false, attn_, column, MmParams{N, D, N, N, width, width, 1.0f, 0, 0});
      }
      mm(enc, attn_, 0, L.o, 0, true, r_, 0, MmParams{N, hidden, width, width, width, hidden, 1.0f, 0, 0});
      residual(enc, L.post_attention_norm, L.pre_feedforward_norm, N);
      mm(enc, h_, 0, L.gate, 0, true, gate_, 0, MmParams{N, inter, hidden, hidden, hidden, inter, 1.0f, 0, 0});
      mm(enc, h_, 0, L.up, 0, true, up_, 0, MmParams{N, inter, hidden, hidden, hidden, inter, 1.0f, 0, 0});
      [enc setComputePipelineState:pipeline("glu_mul")];
      [enc setBuffer:gate_ offset:0 atIndex:0];
      [enc setBuffer:up_ offset:0 atIndex:1];
      bytes(enc, GluMulParams{N * inter, 0}, 2);  // 0: tanh GELU
      [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N * inter), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      mm(enc, gate_, 0, L.down, 0, true, r_, 0, MmParams{N, hidden, inter, inter, inter, hidden, 1.0f, 0, 0});
      residual(enc, L.post_feedforward_norm, l + 1 < layers_.size() ? layers_[l + 1].input_norm : nil, N);
    }

    // Pool, scale by sqrt(hidden), standardize, then the embedder.
    [enc setComputePipelineState:pipeline("pool_standardize")];
    [enc setBuffer:x_ offset:0 atIndex:0];
    [enc setBuffer:pooled_ offset:0 atIndex:1];
    [enc setBuffer:std_bias_ offset:0 atIndex:2];
    [enc setBuffer:std_scale_ offset:0 atIndex:3];
    bytes(enc, PoolStandardizeParams{hidden, cols, cols / c.pooling, c.pooling, tokens,
                                     static_cast<float>(std::sqrt(static_cast<double>(hidden))), c.standardize ? 1 : 0},
          4);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(hidden), static_cast<NSUInteger>(tokens), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    rms_rows(enc, pooled_, zeros_, 1.0f, pooled_, tokens, hidden);
    mm(enc, pooled_, 0, projection_, 0, true, features_, 0,
       MmParams{tokens, c.text_hidden_size, hidden, hidden, hidden, c.text_hidden_size, 1.0f, 0, 0});
  }

  GpuMatrix input_proj_, projection_;
  std::vector<Layer> layers_;
  id<MTLBuffer> std_bias_ = nil, std_scale_ = nil, zeros_ = nil;
  // Host-written inputs (shared), activations, one head's scores, and the
  // features (shared, read back).
  id<MTLBuffer> patches_ = nil, x_ = nil, xy_ = nil, cos_ = nil, sin_ = nil;
  id<MTLBuffer> h_ = nil, r_ = nil, q_ = nil, k_ = nil, v_ = nil, attn_ = nil, gate_ = nil, up_ = nil;
  id<MTLBuffer> scores_ = nil, pooled_ = nil, features_ = nil;
};

// Qwen3.5's, Qwen3-VL's, Qwen2.5-VL's and Qwen2-VL's tower. Buffers grow to the largest
// image encoded so far, since the processor's bounds allow images far larger
// than most. Qwen2.5-VL's patches run window by window, as Transformers
// orders them, so each window is a run of rows; the features return to the
// merged grid's order on readback.
class Qwen35VisionMetal final : public Qwen35VisionEncoder, private MetalKit {
 public:
  explicit Qwen35VisionMetal(std::shared_ptr<const Qwen35VisionWeights> weights)
      : Qwen35VisionEncoder(std::move(weights)) {
    @autoreleasepool {
      LK_CHECK(metal_qwen3_5_vision_supports(*weights_),
               "metal: the vision tower has quantized matrices or windows of over " + std::to_string(kMaxWindowKeys) +
                   " patches; use --backend cpu");
      upload();
    }
  }

  const char* backend() const override { return "metal"; }

  void set_attention_rows(int rows) override {
    LK_CHECK(rows >= 0, "metal: attention rows must not be negative");
    attention_rows_ = rows;
  }

  ImageFeatures encode_features(const Image& image) override {
    @autoreleasepool {
      const Qwen35VisionConfig& c = config();
      const auto [token_rows, token_cols] = token_grid(image.width, image.height);
      const int rows = token_rows * c.merge, cols = token_cols * c.merge, N = rows * cols;
      const int tokens = token_rows * token_cols, hidden = c.hidden_size;
      reserve(N);
      // Row `slot` runs patch order[slot] of the merge-block order (for
      // Qwen2.5-VL, window by window, with its window's rows bounded by
      // window_[2 * slot] and window_[2 * slot + 1]).
      std::vector<int32_t> bounds;
      const std::vector<int> order = run_order(rows, cols, c.qwen2_5() ? &bounds : nullptr);
      if (c.qwen2_5()) std::memcpy([window_ contents], bounds.data(), sizeof(int32_t) * bounds.size());
      // Host side: the normalized patches, the interpolated position rows
      // (into x_, which the patch embedding then accumulates onto), and the
      // 2-D RoPE tables in double, each in run order.
      const size_t width = static_cast<size_t>(3 * c.patch_size * c.patch_size);
      const size_t half = static_cast<size_t>(c.head_dim() / 2);
      const std::vector<float> rows_data = patches(image);
      std::vector<float> cos_rows(static_cast<size_t>(N) * half), sin_rows(static_cast<size_t>(N) * half);
      rope_tables(rows, cols, cos_rows.data(), sin_rows.data());
      float* patch_out = static_cast<float*>([patches_ contents]);
      float* cos_out = static_cast<float*>([cos_ contents]);
      float* sin_out = static_cast<float*>([sin_ contents]);
      float* x = static_cast<float*>([x_ contents]);
      std::fill(x, x + static_cast<size_t>(N) * hidden, 0.0f);
      for (size_t slot = 0; slot < static_cast<size_t>(N); ++slot) {
        const size_t t = static_cast<size_t>(order[slot]);
        std::copy_n(rows_data.data() + t * width, width, patch_out + slot * width);
        std::copy_n(cos_rows.data() + t * half, half, cos_out + slot * half);
        std::copy_n(sin_rows.data() + t * half, half, sin_out + slot * half);
        if (c.position_side > 0) add_position(x + slot * hidden, static_cast<int>(t), rows, cols);
      }
      run([&](id<MTLComputeCommandEncoder> enc) { encode_tower(enc, N); });
      // Feature row r is the merged token of the patches in rows r * merge^2
      // onward.
      const size_t square = static_cast<size_t>(c.merge * c.merge), text = static_cast<size_t>(c.text_hidden_size);
      const size_t size = static_cast<size_t>(tokens) * text;
      auto read = [&](id<MTLBuffer> buffer) {
        const float* out = static_cast<const float*>([buffer contents]);
        std::vector<float> features(size);
        for (size_t row = 0; row < static_cast<size_t>(tokens); ++row) {
          const size_t token = static_cast<size_t>(order[row * square]) / square;
          std::copy_n(out + row * text, text, features.data() + token * text);
        }
        return features;
      };
      ImageFeatures features{read(features_), {}};
      for (id<MTLBuffer> buffer : deepstack_features_) features.deepstack.push_back(read(buffer));
      return features;
    }
  }

 private:
  // Qwen3.5's MLP is fc1 / fc2, Qwen2.5-VL's gate / up / down; its norms
  // have no biases and its patch embedding none (nil buffers).
  struct Layer {
    id<MTLBuffer> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    GpuMatrix qkv, proj, fc1, fc2, gate, up, down;
    id<MTLBuffer> qkv_bias, proj_bias, fc1_bias, fc2_bias, gate_bias, up_bias, down_bias;
  };
  struct Merger {
    id<MTLBuffer> norm_weight, norm_bias;
    GpuMatrix fc1, fc2;
    id<MTLBuffer> fc1_bias, fc2_bias;
  };
  struct HalfRopeParams { int heads; int dim; int stride; };

  // A buffer of `values`, or nil when there are none.
  id<MTLBuffer> optional(const std::vector<float>& values) { return values.empty() ? nil : buffer_of(values); }
  GpuMatrix optional(const Linear& linear) { return linear.cols > 0 ? matrix_of(linear) : GpuMatrix{}; }

  void upload() {
    const Qwen35VisionWeights& w = *weights_;
    patch_ = GpuMatrix{buffer_of(w.patch_weight), "f32"};
    patch_bias_ = optional(w.patch_bias);
    for (const Qwen35VisionWeights::Layer& L : w.layers) {
      layers_.push_back(Layer{buffer_of(L.norm1_weight), optional(L.norm1_bias), buffer_of(L.norm2_weight),
                              optional(L.norm2_bias), matrix_of(L.qkv), matrix_of(L.proj), optional(L.fc1),
                              optional(L.fc2), optional(L.gate), optional(L.up), optional(L.down),
                              buffer_of(L.qkv_bias), buffer_of(L.proj_bias), optional(L.fc1_bias), optional(L.fc2_bias),
                              optional(L.gate_bias), optional(L.up_bias), optional(L.down_bias)});
    }
    auto merger = [&](const Qwen35VisionWeights::Merger& m) {
      return Merger{buffer_of(m.norm_weight), optional(m.norm_bias), matrix_of(m.fc1), matrix_of(m.fc2),
                    buffer_of(m.fc1_bias), buffer_of(m.fc2_bias)};
    };
    merger_ = merger(w.merger);
    for (const Qwen35VisionWeights::Merger& m : w.deepstack) deepstack_.push_back(merger(m));
  }

  // Activation buffers for images of up to N patches.
  void reserve(int N) {
    if (N <= capacity_) return;
    const Qwen35VisionConfig& c = config();
    const size_t f = sizeof(float), n = static_cast<size_t>(N), hidden = static_cast<size_t>(c.hidden_size);
    const size_t tokens = n / static_cast<size_t>(c.merge * c.merge);
    patches_ = shared_buffer(n * static_cast<size_t>(3 * c.patch_size * c.patch_size) * f);
    x_ = shared_buffer(n * hidden * f);  // the host writes the position rows
    cos_ = shared_buffer(n * static_cast<size_t>(c.head_dim() / 2) * f);
    sin_ = shared_buffer(n * static_cast<size_t>(c.head_dim() / 2) * f);
    h_ = private_buffer(n * hidden * f);
    qkv_ = private_buffer(n * 3 * hidden * f);
    attn_ = private_buffer(n * hidden * f);
    if (flash()) {  // attn_flash's layouts of q, k and v
      q_heads_ = private_buffer(n * hidden * f);
      k_heads_ = private_buffer(n * hidden * f);
      v_heads_ = private_buffer(n * hidden * f);
    }
    f_ = private_buffer(n * static_cast<size_t>(c.intermediate_size) * f);
    if (c.qwen2_5()) {
      up_ = private_buffer(n * static_cast<size_t>(c.intermediate_size) * f);
      window_ = shared_buffer(n * 2 * sizeof(int32_t));
    }
    merged_ = private_buffer(n * hidden * f);  // [tokens, merge^2 * hidden]
    features_ = shared_buffer(tokens * static_cast<size_t>(c.text_hidden_size) * f);
    deepstack_features_.clear();
    for (size_t i = 0; i < deepstack_.size(); ++i) {
      deepstack_features_.push_back(shared_buffer(tokens * static_cast<size_t>(c.text_hidden_size) * f));
    }
    capacity_ = N;
  }

  // [m, k] rows times a [n, k] weight, plus its bias, into C (or onto it).
  void linear(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, const GpuMatrix& w, id<MTLBuffer> bias,
              id<MTLBuffer> c, int m, int n, int k, bool accumulate) {
    mm(enc, a, 0, w, 0, true, c, 0, MmParams{m, n, k, k, k, n, 1.0f, bias != nil ? 1 : 0, accumulate ? 1 : 0}, bias);
  }

  void layer_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> weight, id<MTLBuffer> bias,
                  id<MTLBuffer> out, int rows, int cols) {
    [enc setComputePipelineState:pipeline("layer_norm_rows")];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:bias offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    bytes(enc, LayerNormParams{cols, 1e-6f}, 4);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  // The tower's norm: Qwen2.5-VL's RMSNorm (no bias), else a LayerNorm.
  void norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> weight, id<MTLBuffer> bias,
            id<MTLBuffer> out, int rows, int cols) {
    if (!config().qwen2_5()) {
      layer_norm(enc, in, weight, bias, out, rows, cols);
      return;
    }
    [enc setComputePipelineState:pipeline("rms_norm_rows")];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    bytes(enc, NormParams{cols, 1e-6f, 0.0f}, 3);
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  // 2-D RoPE on the q (part 0) or k (part 1) heads inside qkv_ [N, 3 * hidden].
  void rope(id<MTLComputeCommandEncoder> enc, int part, int N) {
    const Qwen35VisionConfig& c = config();
    const int D = c.head_dim();
    [enc setComputePipelineState:pipeline("rope_half_rows")];
    [enc setBuffer:qkv_ offset:sizeof(float) * static_cast<size_t>(part) * c.hidden_size atIndex:0];
    [enc setBuffer:cos_ offset:0 atIndex:1];
    [enc setBuffer:sin_ offset:0 atIndex:2];
    bytes(enc, HalfRopeParams{c.num_heads, D, 3 * c.hidden_size}, 3);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(D / 2), static_cast<NSUInteger>(c.num_heads),
                                     static_cast<NSUInteger>(N))
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(std::min(D / 2, 64)), 1, 1)];
  }

  void encode_tower(id<MTLComputeCommandEncoder> enc, int N) {
    const Qwen35VisionConfig& c = config();
    const int hidden = c.hidden_size, inter = c.intermediate_size, heads = c.num_heads, D = c.head_dim();
    const int patch_width = 3 * c.patch_size * c.patch_size, stride = 3 * hidden;

    // x (the position rows) += patches W^T + bias.
    linear(enc, patches_, patch_, patch_bias_, x_, N, hidden, patch_width, true);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const GpuMatrix fused{qkv_, "f32"};
    const int pass_rows = score_rows(N);
    const size_t scores = static_cast<size_t>(pass_rows) * static_cast<size_t>(N);
    if (!flash() && scores > score_capacity_) {
      scores_ = private_buffer(scores * sizeof(float));
      score_capacity_ = scores;
    }
    for (const Layer& L : layers_) {
      const int index = static_cast<int>(&L - layers_.data());
      norm(enc, x_, L.norm1_weight, L.norm1_bias, h_, N, hidden);
      linear(enc, h_, L.qkv, L.qkv_bias, qkv_, N, stride, hidden, false);
      rope(enc, 0, N);
      rope(enc, 1, N);
      const bool global = !c.qwen2_5() || std::binary_search(c.full_attention.begin(), c.full_attention.end(), index);
      if (flash()) {
        // attn_flash over all heads: every patch over every patch (its rows
        // placed at the last position so they see every key), or over its
        // window's rows, which run consecutively (window_'s bounds).
        [enc setComputePipelineState:pipeline("split_qkv_heads")];
        [enc setBuffer:qkv_ offset:0 atIndex:0];
        [enc setBuffer:q_heads_ offset:0 atIndex:1];
        [enc setBuffer:k_heads_ offset:0 atIndex:2];
        [enc setBuffer:v_heads_ offset:0 atIndex:3];
        bytes(enc, SplitHeadsParams{heads, D, stride, N}, 4);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(hidden), static_cast<NSUInteger>(N), 1)
            threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
        [enc setComputePipelineState:pipeline("attn_flash_d" + std::to_string(D))];
        [enc setBuffer:q_heads_ offset:0 atIndex:0];
        [enc setBuffer:k_heads_ offset:0 atIndex:1];
        [enc setBuffer:v_heads_ offset:0 atIndex:2];
        [enc setBuffer:attn_ offset:0 atIndex:3];
        bytes(enc, AttnMmParams{heads, heads, D, N - 1, 0, N, 0, N, 1 << 30, N, scale, 0.0f, 0, 0, 0, global ? 0 : 1, 0},
              4);
        [enc setBuffer:(global ? dummy_ : window_) offset:0 atIndex:5];
        [enc setBuffer:dummy_ offset:0 atIndex:6];
        [enc setBuffer:dummy_ offset:0 atIndex:7];
        [enc setBuffer:dummy_ offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((N + 31) / 32), static_cast<NSUInteger>(heads), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (!global) {  // each patch over its window's rows, every head at once
        [enc setComputePipelineState:pipeline("window_attention")];
        [enc setBuffer:qkv_ offset:0 atIndex:0];
        [enc setBuffer:window_ offset:0 atIndex:1];
        [enc setBuffer:attn_ offset:0 atIndex:2];
        bytes(enc, WindowAttentionParams{D, stride, hidden, scale}, 3);
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(heads), static_cast<NSUInteger>(N), 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      // Without it, one head and up to pass_rows query rows at a time
      // through scores_ [pass_rows, N]; each qkv_ row is [q | k | v].
      for (int head = 0; global && !flash() && head < heads; ++head) {
        const size_t column = static_cast<size_t>(head) * D;
        for (int first = 0; first < N; first += pass_rows) {
          const int count = std::min(pass_rows, N - first);
          const size_t row = static_cast<size_t>(first);
          mm(enc, qkv_, row * stride + column, fused, hidden + column, true, scores_, 0,
             MmParams{count, N, D, stride, stride, N, scale, 0, 0});
          [enc setComputePipelineState:pipeline("softmax_rows")];
          [enc setBuffer:scores_ offset:0 atIndex:0];
          bytes(enc, ElementwiseParams{N}, 1);
          [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          mm(enc, scores_, 0, fused, 2 * hidden + column, false, attn_, row * hidden + column,
             MmParams{count, D, N, N, stride, hidden, 1.0f, 0, 0});
        }
      }
      linear(enc, attn_, L.proj, L.proj_bias, x_, N, hidden, hidden, true);
      norm(enc, x_, L.norm2_weight, L.norm2_bias, h_, N, hidden);
      if (c.qwen2_5()) {  // x += down(silu(gate(h)) * up(h))
        linear(enc, h_, L.gate, L.gate_bias, f_, N, inter, hidden, false);
        linear(enc, h_, L.up, L.up_bias, up_, N, inter, hidden, false);
        [enc setComputePipelineState:pipeline("glu_mul")];
        [enc setBuffer:f_ offset:0 atIndex:0];
        [enc setBuffer:up_ offset:0 atIndex:1];
        bytes(enc, GluMulParams{N * inter, 1}, 2);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N) * static_cast<NSUInteger>(inter), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        linear(enc, f_, L.down, L.down_bias, x_, N, hidden, inter, true);
      } else {
        linear(enc, h_, L.fc1, L.fc1_bias, f_, N, inter, hidden, false);
        [enc setBuffer:f_ offset:0 atIndex:0];
        elementwise(enc, c.qwen3_5() ? "gelu_inplace" : "quick_gelu_inplace", N * inter, 1);  // Qwen2-VL: QuickGELU
        linear(enc, f_, L.fc2, L.fc2_bias, x_, N, hidden, inter, true);
      }
      const auto deep = std::find(c.deepstack.begin(), c.deepstack.end(), index);
      if (deep != c.deepstack.end()) {
        const size_t which = static_cast<size_t>(deep - c.deepstack.begin());
        merge(enc, deepstack_[which], false, deepstack_features_[which], N);
      }
    }
    merge(enc, merger_, true, features_, N);
  }

  // Whether full attention runs as attn_flash (the head widths it is built
  // for), else through scores_.
  bool flash() const {
    const int D = config().head_dim();
    return D == 32 || D == 64 || D == 72 || D == 80 || D == 128;
  }

  // The query rows of each attention pass over N patches.
  int score_rows(int N) const {
    const size_t budget = std::max<size_t>(32, kScoreBudget / static_cast<size_t>(N));
    const size_t rows = attention_rows_ > 0 ? static_cast<size_t>(attention_rows_) : budget;
    return static_cast<int>(std::min(rows, static_cast<size_t>(N)));
  }

  // A merger over the patch rows in x_, into `out` [tokens, text width]. Each
  // merge x merge square's patches are consecutive rows, so they read as
  // [tokens, merge^2 * hidden]; the norm covers a patch or the whole square.
  void merge(id<MTLComputeCommandEncoder> enc, const Merger& m, bool per_patch, id<MTLBuffer> out, int N) {
    const Qwen35VisionConfig& c = config();
    const int square = c.merge * c.merge, tokens = N / square, merged = c.hidden_size * square;
    norm(enc, x_, m.norm_weight, m.norm_bias, h_, per_patch ? N : tokens, per_patch ? c.hidden_size : merged);
    linear(enc, h_, m.fc1, m.fc1_bias, merged_, tokens, merged, merged, false);
    [enc setBuffer:merged_ offset:0 atIndex:0];
    elementwise(enc, "gelu_erf_inplace", tokens * merged, 1);
    linear(enc, merged_, m.fc2, m.fc2_bias, out, tokens, c.text_hidden_size, merged, false);
  }

  GpuMatrix patch_;
  id<MTLBuffer> patch_bias_ = nil;
  Merger merger_;
  std::vector<Merger> deepstack_;
  std::vector<id<MTLBuffer>> deepstack_features_;  // shared, read back
  std::vector<Layer> layers_;
  int capacity_ = 0;  // patches the activation buffers hold
  // Host-written inputs (shared), activations, one head's scores, the merged
  // rows and the features (shared, read back).
  id<MTLBuffer> patches_ = nil, x_ = nil, cos_ = nil, sin_ = nil;
  id<MTLBuffer> h_ = nil, qkv_ = nil, attn_ = nil, f_ = nil, scores_ = nil, merged_ = nil, features_ = nil;
  id<MTLBuffer> up_ = nil, window_ = nil;  // Qwen2.5-VL: the MLP's up rows, each slot's window bounds
  id<MTLBuffer> q_heads_ = nil, k_heads_ = nil, v_heads_ = nil;  // flash(): q, k and v as attn_flash reads them
  int attention_rows_ = 0;     // set_attention_rows; 0: kScoreBudget's share
  size_t score_capacity_ = 0;  // floats scores_ holds
};

}  // namespace

bool metal_siglip_supports(const SiglipWeights& weights) {
  for (const SiglipWeights::Layer& L : weights.layers) {
    for (const Linear* l : {&L.q_proj, &L.k_proj, &L.v_proj, &L.out_proj, &L.fc1, &L.fc2}) {
      if (l->quant.quantized() || mm_variant(l->dense_dtype) == nullptr) return false;
    }
  }
  return true;
}

std::unique_ptr<VisionEncoder> make_siglip_metal(std::shared_ptr<const SiglipWeights> weights) {
  return std::make_unique<SiglipMetal>(std::move(weights));
}

bool metal_gemma4_vision_supports(const Gemma4VisionWeights& weights) {
  if (weights.input_proj.quant.quantized() || mm_variant(weights.input_proj.dense_dtype) == nullptr) return false;
  for (const Gemma4VisionWeights::Layer& L : weights.layers) {
    for (const Linear* l : {&L.q_proj, &L.k_proj, &L.v_proj, &L.o_proj, &L.gate_proj, &L.up_proj, &L.down_proj}) {
      if (l->quant.quantized() || mm_variant(l->dense_dtype) == nullptr) return false;
    }
  }
  return true;
}

std::unique_ptr<VisionEncoder> make_gemma4_vision_metal(std::shared_ptr<const Gemma4VisionWeights> weights) {
  return std::make_unique<Gemma4VisionMetal>(std::move(weights));
}

bool metal_qwen3_5_vision_supports(const Qwen35VisionWeights& weights) {
  const Qwen35VisionConfig& c = weights.config;
  if (c.qwen2_5() && c.window * c.window * c.merge * c.merge > kMaxWindowKeys) return false;
  std::vector<const Qwen35VisionWeights::Merger*> mergers = {&weights.merger};
  for (const Qwen35VisionWeights::Merger& m : weights.deepstack) mergers.push_back(&m);
  for (const Qwen35VisionWeights::Merger* m : mergers) {
    for (const Linear* l : {&m->fc1, &m->fc2}) {
      if (l->quant.quantized() || mm_variant(l->dense_dtype) == nullptr) return false;
    }
  }
  for (const Qwen35VisionWeights::Layer& L : weights.layers) {
    for (const Linear* l : {&L.qkv, &L.proj, &L.fc1, &L.fc2, &L.gate, &L.up, &L.down}) {
      if (l->cols > 0 && (l->quant.quantized() || mm_variant(l->dense_dtype) == nullptr)) return false;
    }
  }
  return true;
}

std::unique_ptr<VisionEncoder> make_qwen3_5_vision_metal(std::shared_ptr<const Qwen35VisionWeights> weights) {
  return std::make_unique<Qwen35VisionMetal>(std::move(weights));
}

}  // namespace lokahi
