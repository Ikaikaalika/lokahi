// Linear-layer weights: MLX affine- or MXFP4-quantized, or dense, plus CPU
// kernels.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "json.h"
#include "safetensors.h"
#include "thread_pool.h"

namespace lokahi {

// Affine: code * scale + bias per group, with BF16/F16 scales and biases.
// MXFP4 (OCP microscaling): 4-bit E2M1 codes times a per-32 E8M0 scale.
enum class QuantMode { Affine, Mxfp4 };

struct QuantSpec {
  int group_size = 0;
  int bits = 0;  // 0 means unquantized.
  QuantMode mode = QuantMode::Affine;
  bool quantized() const { return bits != 0; }
  bool mxfp4() const { return quantized() && mode == QuantMode::Mxfp4; }
};

// Resolves an MLX `quantization` config (default plus per-module overrides,
// where `false` marks an unquantized module) for `module`.
QuantSpec quant_spec_for(const Json& config, const std::string& module);

// A [rows, cols] matrix used as y = W x. Either quantized (packed uint32
// codes with per-group scales, plus biases for affine) or dense.
struct Linear {
  std::string name;
  int64_t rows = 0;
  int64_t cols = 0;
  QuantSpec quant;
  const uint32_t* words = nullptr;  // quantized: [rows, cols * bits / 32]
  const uint16_t* scales = nullptr;  // affine: [rows, cols / group_size]
  const uint16_t* biases = nullptr;
  const uint8_t* exponents = nullptr;  // mxfp4: [rows, cols / 32] E8M0 scales
  DType scale_dtype = DType::BF16;  // BF16 or F16
  const void* dense = nullptr;       // dense: [rows, cols]
  DType dense_dtype = DType::F32;

  int64_t words_per_row() const { return cols * quant.bits / 32; }
  int64_t groups_per_row() const { return cols / quant.group_size; }
  size_t weight_bytes() const;
};

// Binds `module` (e.g. "model.layers.0.mlp.up_proj") from the checkpoint.
Linear bind_linear(const Checkpoint& checkpoint, const Json& config, const std::string& module,
                   const std::string& quant_module);

// Binds a stack of expert matrices stored as one [experts, rows, cols]
// tensor (MLX switch layers), as one zero-copy view per expert.
std::vector<Linear> bind_expert_linears(const Checkpoint& checkpoint, const Json& config, const std::string& module,
                                        const std::string& quant_module);

// Rows [first, first + count) of `linear` as a zero-copy view.
Linear row_slice(const Linear& linear, int64_t first, int64_t count);

// Dense 1-D vector (norm weights) as float32.
void load_vector(const Checkpoint& checkpoint, const std::string& name, int64_t size, float* out);

float scale_value(const Linear& linear, int64_t row, int64_t group);
float bias_value(const Linear& linear, int64_t row, int64_t group);

// out[cols] = W[row, :]
void dequantize_row(const Linear& linear, int64_t row, float* out);

// y[rows] = W x
void matvec(const Linear& linear, const float* x, float* y, ThreadPool& pool);

// Y[n, rows] = X[n, cols] W^T
void matmul(const Linear& linear, const float* x, int64_t n, float* y, ThreadPool& pool);

}  // namespace lokahi
