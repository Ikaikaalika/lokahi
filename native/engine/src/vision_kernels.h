// CPU helpers shared by the vision encoders (internal).
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "quant.h"
#include "safetensors.h"

namespace lokahi::vision {

// LayerNorm with mean and variance in double, as the oracles compute them.
inline void layer_norm(const float* x, const std::vector<float>& weight, const std::vector<float>& bias, int64_t size,
                       float eps, float* out) {
  double mean = 0.0;
  for (int64_t i = 0; i < size; ++i) mean += x[i];
  mean /= static_cast<double>(size);
  double var = 0.0;
  for (int64_t i = 0; i < size; ++i) var += (x[i] - mean) * (x[i] - mean);
  const double inv = 1.0 / std::sqrt(var / static_cast<double>(size) + eps);
  for (int64_t i = 0; i < size; ++i) {
    out[i] = static_cast<float>((x[i] - mean) * inv) * weight[static_cast<size_t>(i)] + bias[static_cast<size_t>(i)];
  }
}

// RMSNorm times `weight` (nullptr: no scale), the mean square in double.
inline void rms_norm(const float* x, const float* weight, int64_t size, float eps, float* out) {
  double squares = 0.0;
  for (int64_t i = 0; i < size; ++i) squares += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(size) + eps));
  for (int64_t i = 0; i < size; ++i) out[i] = x[i] * inv * (weight != nullptr ? weight[i] : 1.0f);
}

inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2 / pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

inline void add_rows(float* x, const std::vector<float>& bias, int64_t n) {
  const size_t width = bias.size();
  for (int64_t t = 0; t < n; ++t) {
    for (size_t i = 0; i < width; ++i) x[static_cast<size_t>(t) * width + i] += bias[i];
  }
}

// A float32 [rows, cols] Linear over `data`.
inline Linear dense_view(const std::string& name, const std::vector<float>& data, int64_t rows, int64_t cols) {
  Linear linear;
  linear.name = name;
  linear.rows = rows;
  linear.cols = cols;
  linear.dense = data.data();
  linear.dense_dtype = DType::F32;
  return linear;
}

inline std::vector<float> vector_of(const Checkpoint& checkpoint, const std::string& name, int64_t size) {
  std::vector<float> out(static_cast<size_t>(size));
  load_vector(checkpoint, name, size, out.data());
  return out;
}

}  // namespace lokahi::vision
