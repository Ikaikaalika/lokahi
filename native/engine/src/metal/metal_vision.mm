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

class SiglipMetal final : public SiglipEncoder {
 public:
  explicit SiglipMetal(std::shared_ptr<const SiglipWeights> weights) : SiglipEncoder(std::move(weights)) {
    @autoreleasepool {
      LK_CHECK(metal_siglip_supports(*weights_), "metal: the vision tower has quantized matrices; use --backend cpu");
      device_ = MTLCreateSystemDefaultDevice();
      LK_CHECK(device_ != nil, "metal: no Metal device");
      queue_ = [device_ newCommandQueue];
      library_ = compile_kernels(device_);
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
      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      encode_tower(enc);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      LK_CHECK([cb status] == MTLCommandBufferStatusCompleted, "metal: command buffer failed: " + ns_error([cb error]));
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

  id<MTLBuffer> buffer_of(const void* data, size_t bytes) {
    id<MTLBuffer> buffer = [device_ newBufferWithBytes:data length:std::max<size_t>(bytes, 16)
                                               options:MTLResourceStorageModeShared];
    LK_CHECK(buffer != nil, "metal: allocation failed");
    return buffer;
  }
  id<MTLBuffer> buffer_of(const std::vector<float>& values) {
    return buffer_of(values.data(), sizeof(float) * values.size());
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
    patches_ = [device_ newBufferWithLength:N * width * f options:MTLResourceStorageModeShared];
    LK_CHECK(patches_ != nil, "metal: allocation failed");
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
    features_ = [device_ newBufferWithLength:T * static_cast<size_t>(c.text_hidden_size) * f
                                     options:MTLResourceStorageModeShared];
    LK_CHECK(features_ != nil, "metal: allocation failed");
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
    [enc setBuffer:(bias != nil ? bias : patch_bias_) offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((p.n + 31) / 32), static_cast<NSUInteger>((p.m + 31) / 32), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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

  // An elementwise kernel over `count` floats whose parameters sit at `index`.
  void elementwise(id<MTLComputeCommandEncoder> enc, const char* name, int count, NSUInteger index) {
    [enc setComputePipelineState:pipeline(name)];
    bytes(enc, ElementwiseParams{count}, index);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;

  GpuMatrix patch_, projection_;
  id<MTLBuffer> patch_bias_ = nil, positions_ = nil, post_weight_ = nil, post_bias_ = nil, soft_norm_ = nil;
  std::vector<Layer> layers_;
  // Patches (shared, written per image), activations, one head's scores, the
  // pooled tokens and the features (shared, read back).
  id<MTLBuffer> patches_ = nil, x_ = nil, h_ = nil, q_ = nil, k_ = nil, v_ = nil, attn_ = nil, f_ = nil;
  id<MTLBuffer> scores_ = nil, pooled_ = nil, normed_ = nil, features_ = nil;
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

}  // namespace lokahi
