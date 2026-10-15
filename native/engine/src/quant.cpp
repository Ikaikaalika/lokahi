#include "quant.h"

#include <algorithm>
#include <vector>

#include "common.h"
#include "half.h"

namespace lokahi {

QuantSpec quant_spec_for(const Json& config, const std::string& module) {
  const Json* quant = config.find("quantization");
  if (quant == nullptr || !quant->is_object()) return {};
  QuantSpec spec;
  spec.group_size = static_cast<int>(quant->int_or("group_size", 64));
  spec.bits = static_cast<int>(quant->int_or("bits", 4));
  std::string mode = quant->string_or("mode", "affine");
  if (const Json* override_value = quant->find(module)) {
    if (override_value->is_bool() && !override_value->as_bool()) return {};
    if (override_value->is_object()) {
      spec.group_size = static_cast<int>(override_value->int_or("group_size", spec.group_size));
      spec.bits = static_cast<int>(override_value->int_or("bits", spec.bits));
      mode = override_value->string_or("mode", mode);
    }
  }
  LK_CHECK(mode == "affine", "quantization mode '" + mode + "' is not supported yet (" + module + ")");
  LK_CHECK(spec.bits == 2 || spec.bits == 4 || spec.bits == 8,
           "unsupported bit width " + std::to_string(spec.bits) + " for " + module);
  LK_CHECK(spec.group_size > 0 && spec.group_size % (32 / spec.bits) == 0,
           "invalid group size for " + module);
  return spec;
}

size_t Linear::weight_bytes() const {
  if (!quant.quantized()) return static_cast<size_t>(rows * cols) * dtype_size(dense_dtype);
  return static_cast<size_t>(rows * words_per_row()) * 4 +
         static_cast<size_t>(rows * groups_per_row()) * 2 * 2;
}

Linear bind_linear(const Checkpoint& checkpoint, const Json& config, const std::string& module,
                   const std::string& quant_module) {
  Linear linear;
  linear.name = module;
  const TensorView& weight = checkpoint.get(module + ".weight");
  const TensorView* scales = checkpoint.find(module + ".scales");
  LK_CHECK(weight.shape.size() == 2, module + ": weight must be 2-D");
  if (scales == nullptr) {
    LK_CHECK(weight.dtype == DType::F32 || weight.dtype == DType::F16 || weight.dtype == DType::BF16,
             module + ": unsupported dense dtype");
    linear.rows = weight.dim(0);
    linear.cols = weight.dim(1);
    linear.dense = weight.data;
    linear.dense_dtype = weight.dtype;
    return linear;
  }
  const TensorView& biases = checkpoint.get(module + ".biases");
  linear.quant = quant_spec_for(config, quant_module);
  LK_CHECK(linear.quant.quantized(), module + ": has scales but config marks it unquantized");
  LK_CHECK(weight.dtype == DType::U32, module + ": quantized weight must be U32");
  LK_CHECK(scales->dtype == biases.dtype &&
               (scales->dtype == DType::BF16 || scales->dtype == DType::F16),
           module + ": scales/biases must be BF16 or F16");
  linear.rows = weight.dim(0);
  linear.cols = weight.dim(1) * 32 / linear.quant.bits;
  LK_CHECK(scales->shape.size() == 2 && scales->dim(0) == linear.rows &&
               scales->dim(1) == linear.cols / linear.quant.group_size,
           module + ": scales shape does not match weight and group size");
  LK_CHECK(biases.shape == scales->shape, module + ": biases shape mismatch");
  linear.words = weight.as<uint32_t>();
  linear.scales = scales->as<uint16_t>();
  linear.biases = biases.as<uint16_t>();
  linear.scale_dtype = scales->dtype;
  return linear;
}

std::vector<Linear> bind_expert_linears(const Checkpoint& checkpoint, const Json& config, const std::string& module,
                                        const std::string& quant_module) {
  const TensorView& weight = checkpoint.get(module + ".weight");
  const TensorView* scales = checkpoint.find(module + ".scales");
  LK_CHECK(weight.shape.size() == 3, module + ": expert weights must be 3-D");
  const int64_t experts = weight.dim(0);
  LK_CHECK(experts > 0, module + ": no experts");
  Linear first;
  first.name = module;
  first.rows = weight.dim(1);
  size_t weight_stride = 0, scale_stride = 0;  // bytes per expert
  if (scales == nullptr) {
    LK_CHECK(weight.dtype == DType::F32 || weight.dtype == DType::F16 || weight.dtype == DType::BF16,
             module + ": unsupported dense dtype");
    first.cols = weight.dim(2);
    first.dense = weight.data;
    first.dense_dtype = weight.dtype;
    weight_stride = static_cast<size_t>(first.rows * first.cols) * dtype_size(weight.dtype);
  } else {
    const TensorView& biases = checkpoint.get(module + ".biases");
    first.quant = quant_spec_for(config, quant_module);
    LK_CHECK(first.quant.quantized(), module + ": has scales but config marks it unquantized");
    LK_CHECK(weight.dtype == DType::U32, module + ": quantized weight must be U32");
    LK_CHECK(scales->dtype == biases.dtype && (scales->dtype == DType::BF16 || scales->dtype == DType::F16),
             module + ": scales/biases must be BF16 or F16");
    first.cols = weight.dim(2) * 32 / first.quant.bits;
    LK_CHECK(scales->shape.size() == 3 && scales->dim(0) == experts && scales->dim(1) == first.rows &&
                 scales->dim(2) == first.cols / first.quant.group_size,
             module + ": scales shape does not match weight and group size");
    LK_CHECK(biases.shape == scales->shape, module + ": biases shape mismatch");
    first.words = weight.as<uint32_t>();
    first.scales = scales->as<uint16_t>();
    first.biases = biases.as<uint16_t>();
    first.scale_dtype = scales->dtype;
    weight_stride = static_cast<size_t>(first.rows * first.words_per_row()) * 4;
    scale_stride = static_cast<size_t>(first.rows * first.groups_per_row()) * 2;
  }
  std::vector<Linear> out(static_cast<size_t>(experts), first);
  for (int64_t e = 0; e < experts; ++e) {
    Linear& linear = out[static_cast<size_t>(e)];
    const size_t offset = static_cast<size_t>(e);
    if (linear.quant.quantized()) {
      linear.words = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(first.words) + offset * weight_stride);
      linear.scales = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(first.scales) + offset * scale_stride);
      linear.biases = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(first.biases) + offset * scale_stride);
    } else {
      linear.dense = static_cast<const uint8_t*>(first.dense) + offset * weight_stride;
    }
  }
  return out;
}

void load_vector(const Checkpoint& checkpoint, const std::string& name, int64_t size, float* out) {
  const TensorView& view = checkpoint.get(name);
  LK_CHECK(view.numel() == size, name + ": expected " + std::to_string(size) + " elements");
  for (int64_t i = 0; i < size; ++i) {
    switch (view.dtype) {
      case DType::F32: out[i] = view.as<float>()[i]; break;
      case DType::BF16: out[i] = bf16_to_f32(view.as<uint16_t>()[i]); break;
      case DType::F16: out[i] = f16_to_f32(view.as<uint16_t>()[i]); break;
      default: fail(name + ": unsupported vector dtype");
    }
  }
}

static inline float half_value(const uint16_t* data, DType dtype, int64_t index) {
  return dtype == DType::BF16 ? bf16_to_f32(data[index]) : f16_to_f32(data[index]);
}

float scale_value(const Linear& linear, int64_t row, int64_t group) {
  return half_value(linear.scales, linear.scale_dtype, row * linear.groups_per_row() + group);
}

float bias_value(const Linear& linear, int64_t row, int64_t group) {
  return half_value(linear.biases, linear.scale_dtype, row * linear.groups_per_row() + group);
}

static inline float dense_value(const Linear& linear, int64_t index) {
  switch (linear.dense_dtype) {
    case DType::F32: return static_cast<const float*>(linear.dense)[index];
    case DType::BF16: return bf16_to_f32(static_cast<const uint16_t*>(linear.dense)[index]);
    default: return f16_to_f32(static_cast<const uint16_t*>(linear.dense)[index]);
  }
}

void dequantize_row(const Linear& linear, int64_t row, float* out) {
  if (!linear.quant.quantized()) {
    for (int64_t c = 0; c < linear.cols; ++c) out[c] = dense_value(linear, row * linear.cols + c);
    return;
  }
  const int bits = linear.quant.bits;
  const int per_word = 32 / bits;
  const uint32_t mask = (1u << bits) - 1u;
  const uint32_t* words = linear.words + row * linear.words_per_row();
  const int64_t group_size = linear.quant.group_size;
  for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
    const float scale = scale_value(linear, row, g);
    const float bias = bias_value(linear, row, g);
    for (int64_t c = g * group_size; c < (g + 1) * group_size; ++c) {
      uint32_t word = words[c / per_word];
      uint32_t code = (word >> (bits * (c % per_word))) & mask;
      out[c] = scale * static_cast<float>(code) + bias;
    }
  }
}

// Dot of one quantized row with x, using per-group sums of x so the bias term
// costs one multiply per group: sum(s*q*x + b*x) = s*sum(q*x) + b*sum(x).
template <int Bits>
static float quant_row_dot(const Linear& linear, int64_t row, const float* x, const float* xsum) {
  constexpr int kPerWord = 32 / Bits;
  constexpr uint32_t kMask = (1u << Bits) - 1u;
  const int64_t group_size = linear.quant.group_size;
  const int64_t words_per_group = group_size / kPerWord;
  const uint32_t* words = linear.words + row * linear.words_per_row();
  float total = 0.0f;
  for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
    const uint32_t* gw = words + g * words_per_group;
    const float* gx = x + g * group_size;
    float acc = 0.0f;
    for (int64_t w = 0; w < words_per_group; ++w) {
      uint32_t word = gw[w];
      const float* wx = gx + w * kPerWord;
      for (int j = 0; j < kPerWord; ++j) {
        acc += static_cast<float>((word >> (Bits * j)) & kMask) * wx[j];
      }
    }
    total += scale_value(linear, row, g) * acc + bias_value(linear, row, g) * xsum[g];
  }
  return total;
}

static float row_dot(const Linear& linear, int64_t row, const float* x, const float* xsum,
                     float* scratch) {
  if (!linear.quant.quantized()) {
    float acc = 0.0f;
    const int64_t base = row * linear.cols;
    if (linear.dense_dtype == DType::F32) {
      const float* w = static_cast<const float*>(linear.dense) + base;
      for (int64_t c = 0; c < linear.cols; ++c) acc += w[c] * x[c];
      return acc;
    }
    dequantize_row(linear, row, scratch);
    for (int64_t c = 0; c < linear.cols; ++c) acc += scratch[c] * x[c];
    return acc;
  }
  switch (linear.quant.bits) {
    case 2: return quant_row_dot<2>(linear, row, x, xsum);
    case 4: return quant_row_dot<4>(linear, row, x, xsum);
    default: return quant_row_dot<8>(linear, row, x, xsum);
  }
}

static std::vector<float> group_sums(const Linear& linear, const float* x) {
  if (!linear.quant.quantized()) return {};
  std::vector<float> sums(static_cast<size_t>(linear.groups_per_row()), 0.0f);
  const int64_t group_size = linear.quant.group_size;
  for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
    float acc = 0.0f;
    for (int64_t c = g * group_size; c < (g + 1) * group_size; ++c) acc += x[c];
    sums[static_cast<size_t>(g)] = acc;
  }
  return sums;
}

void matvec(const Linear& linear, const float* x, float* y, ThreadPool& pool) {
  std::vector<float> sums = group_sums(linear, x);
  pool.parallel_for(
      static_cast<size_t>(linear.rows),
      [&](size_t begin, size_t end) {
        std::vector<float> scratch(linear.quant.quantized() ? 0 : static_cast<size_t>(linear.cols));
        for (size_t r = begin; r < end; ++r) {
          y[r] = row_dot(linear, static_cast<int64_t>(r), x, sums.data(), scratch.data());
        }
      },
      64);
}

void matmul(const Linear& linear, const float* x, int64_t n, float* y, ThreadPool& pool) {
  if (n == 1) {
    matvec(linear, x, y, pool);
    return;
  }
  // Dequantize each weight row once and reuse it for every token.
  pool.parallel_for(
      static_cast<size_t>(linear.rows),
      [&](size_t begin, size_t end) {
        std::vector<float> row(static_cast<size_t>(linear.cols));
        for (size_t r = begin; r < end; ++r) {
          dequantize_row(linear, static_cast<int64_t>(r), row.data());
          for (int64_t t = 0; t < n; ++t) {
            const float* xt = x + t * linear.cols;
            float acc = 0.0f;
            for (int64_t c = 0; c < linear.cols; ++c) acc += row[static_cast<size_t>(c)] * xt[c];
            y[t * linear.rows + static_cast<int64_t>(r)] = acc;
          }
        }
      },
      16);
}

}  // namespace lokahi
