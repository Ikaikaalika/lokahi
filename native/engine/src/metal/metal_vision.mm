// Metal backend for the vision encoder: Gemma 3's SigLIP tower and
// projector, one command buffer per image.
//
// Every matrix product, including attention's (scores = Q K^T per head, then
// a row softmax and P V), runs through dense_mm. That kernel reads the
// checkpoint's dense F32, F16 or BF16 weights as they are, and the
// activations stay float32 throughout.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/metal_vision.h"

#include "gemma4_vision.h"

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
    dummy_ = shared_buffer(16);  // bound where a kernel's optional input is unused
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
    mm(enc, a, 0, w, 0, true, c, 0, MmParams{m, n, k, k, k, n, 1.0f, 1, accumulate ? 1 : 0}, bias);
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

}  // namespace lokahi
