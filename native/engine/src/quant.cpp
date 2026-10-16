#include "quant.h"

#include <algorithm>
#include <cstring>
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
      // As mlx.nn.quantize passes it to to_quantized: missing keys take that
      // mode's defaults, not the model-wide ones.
      mode = override_value->string_or("mode", "affine");
      spec.group_size = static_cast<int>(override_value->int_or("group_size", mode == "mxfp4" ? 32 : 64));
      spec.bits = static_cast<int>(override_value->int_or("bits", 4));
    }
  }
  if (mode == "mxfp4") {
    LK_CHECK(spec.bits == 4 && spec.group_size == 32, "mxfp4 needs 4 bits and group size 32 (" + module + ")");
    spec.mode = QuantMode::Mxfp4;
    return spec;
  }
  LK_CHECK(mode == "affine", "quantization mode '" + mode + "' is not supported yet (" + module + ")");
  LK_CHECK(spec.bits == 2 || spec.bits == 3 || spec.bits == 4 || spec.bits == 5 || spec.bits == 6 || spec.bits == 8,
           "unsupported bit width " + std::to_string(spec.bits) + " for " + module);
  // Groups start on word boundaries, as in every layout MLX writes.
  LK_CHECK(spec.group_size > 0 && spec.group_size * spec.bits % 32 == 0, "invalid group size for " + module);
  return spec;
}

size_t Linear::weight_bytes() const {
  if (!quant.quantized()) return static_cast<size_t>(rows * cols) * dtype_size(dense_dtype);
  // Affine groups carry a 2-byte scale and bias; MXFP4 groups a 1-byte exponent.
  return static_cast<size_t>(rows * words_per_row()) * 4 +
         static_cast<size_t>(rows * groups_per_row()) * (quant.mxfp4() ? 1 : 4);
}

namespace {

// Binds the per-group scales of a quantized module whose codes are bound:
// affine BF16/F16 scales and biases, or MXFP4's U8 exponents. `experts` > 0
// expects a leading expert dimension.
void bind_scales(const Checkpoint& checkpoint, const std::string& module, int64_t experts, Linear& linear) {
  const TensorView& scales = checkpoint.get(module + ".scales");
  std::vector<int64_t> shape = {linear.rows, linear.cols / linear.quant.group_size};
  if (experts > 0) shape.insert(shape.begin(), experts);
  LK_CHECK(scales.shape == shape, module + ": scales shape does not match weight and group size");
  if (linear.quant.mxfp4()) {
    LK_CHECK(scales.dtype == DType::U8, module + ": mxfp4 scales must be U8");
    LK_CHECK(checkpoint.find(module + ".biases") == nullptr, module + ": mxfp4 has no biases");
    linear.exponents = scales.as<uint8_t>();
    return;
  }
  const TensorView& biases = checkpoint.get(module + ".biases");
  LK_CHECK(scales.dtype == biases.dtype && (scales.dtype == DType::BF16 || scales.dtype == DType::F16),
           module + ": scales/biases must be BF16 or F16");
  LK_CHECK(biases.shape == scales.shape, module + ": biases shape mismatch");
  linear.scales = scales.as<uint16_t>();
  linear.biases = biases.as<uint16_t>();
  linear.scale_dtype = scales.dtype;
}

}  // namespace

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
  linear.quant = quant_spec_for(config, quant_module);
  LK_CHECK(linear.quant.quantized(), module + ": has scales but config marks it unquantized");
  LK_CHECK(weight.dtype == DType::U32, module + ": quantized weight must be U32");
  linear.rows = weight.dim(0);
  LK_CHECK(weight.dim(1) * 32 % linear.quant.bits == 0, module + ": weight words do not hold whole codes");
  linear.cols = weight.dim(1) * 32 / linear.quant.bits;
  linear.words = weight.as<uint32_t>();
  bind_scales(checkpoint, module, 0, linear);
  return linear;
}

std::vector<Linear> bind_expert_linears(const Checkpoint& checkpoint, const Json& config, const std::string& module,
                                        const std::string& quant_module, const std::string& suffix) {
  const TensorView& weight = checkpoint.get(module + suffix);
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
    first.quant = quant_spec_for(config, quant_module);
    LK_CHECK(first.quant.quantized(), module + ": has scales but config marks it unquantized");
    LK_CHECK(weight.dtype == DType::U32, module + ": quantized weight must be U32");
    LK_CHECK(weight.dim(2) * 32 % first.quant.bits == 0, module + ": weight words do not hold whole codes");
    first.cols = weight.dim(2) * 32 / first.quant.bits;
    first.words = weight.as<uint32_t>();
    bind_scales(checkpoint, module, experts, first);
    weight_stride = static_cast<size_t>(first.rows * first.words_per_row()) * 4;
    scale_stride = static_cast<size_t>(first.rows * first.groups_per_row()) * (first.quant.mxfp4() ? 1 : 2);
  }
  std::vector<Linear> out(static_cast<size_t>(experts), first);
  for (int64_t e = 0; e < experts; ++e) {
    Linear& linear = out[static_cast<size_t>(e)];
    const size_t offset = static_cast<size_t>(e);
    if (linear.quant.mxfp4()) {
      linear.words = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(first.words) + offset * weight_stride);
      linear.exponents = first.exponents + offset * scale_stride;
    } else if (linear.quant.quantized()) {
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

// E2M1 values by code, and an E8M0 exponent as MLX decodes it: 2^(e - 127),
// built from its bits (e = 0 is the subnormal 2^-127).
constexpr float kFp4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

static float e8m0_value(uint8_t e) {
  const uint32_t bits = e == 0 ? 0x00400000u : static_cast<uint32_t>(e) << 23;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

Linear row_slice(const Linear& linear, int64_t first, int64_t count) {
  LK_CHECK(first >= 0 && count > 0 && first + count <= linear.rows, linear.name + ": row slice out of range");
  Linear out = linear;
  out.rows = count;
  if (!linear.quant.quantized()) {
    out.dense = static_cast<const uint8_t*>(linear.dense) +
                static_cast<size_t>(first * linear.cols) * dtype_size(linear.dense_dtype);
    return out;
  }
  out.words = linear.words + first * linear.words_per_row();
  if (linear.quant.mxfp4()) {
    out.exponents = linear.exponents + first * linear.groups_per_row();
  } else {
    out.scales = linear.scales + first * linear.groups_per_row();
    out.biases = linear.biases + first * linear.groups_per_row();
  }
  return out;
}

namespace {

// Reads codes in order from MLX's packing: each row (and each group, which
// starts on a word boundary) is one little-endian bit stream of `bits`-bit
// codes, so 3-, 5- and 6-bit codes straddle words.
class CodeStream {
 public:
  CodeStream(const uint32_t* words, int bits) : words_(words), bits_(bits), mask_((1u << bits) - 1u) {}
  uint32_t next() {
    if (have_ < bits_) {
      buffer_ |= static_cast<uint64_t>(*words_++) << have_;
      have_ += 32;
    }
    const uint32_t code = static_cast<uint32_t>(buffer_) & mask_;
    buffer_ >>= bits_;
    have_ -= bits_;
    return code;
  }

 private:
  const uint32_t* words_;
  int bits_;
  uint32_t mask_;
  uint64_t buffer_ = 0;
  int have_ = 0;
};

}  // namespace

void dequantize_row(const Linear& linear, int64_t row, float* out) {
  if (!linear.quant.quantized()) {
    for (int64_t c = 0; c < linear.cols; ++c) out[c] = dense_value(linear, row * linear.cols + c);
    return;
  }
  if (linear.quant.mxfp4()) {
    const uint32_t* words = linear.words + row * linear.words_per_row();
    for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
      const float scale = e8m0_value(linear.exponents[row * linear.groups_per_row() + g]);
      for (int64_t c = g * 32; c < (g + 1) * 32; ++c) out[c] = scale * kFp4[(words[c / 8] >> (4 * (c % 8))) & 0xF];
    }
    return;
  }
  const int bits = linear.quant.bits;
  const uint32_t* words = linear.words + row * linear.words_per_row();
  const int64_t group_size = linear.quant.group_size;
  for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
    const float scale = scale_value(linear, row, g);
    const float bias = bias_value(linear, row, g);
    CodeStream codes(words + g * group_size * bits / 32, bits);
    for (int64_t c = g * group_size; c < (g + 1) * group_size; ++c) {
      out[c] = scale * static_cast<float>(codes.next()) + bias;
    }
  }
}

// Dot of one quantized row with x, using per-group sums of x so the bias term
// costs one multiply per group: sum(s*q*x + b*x) = s*sum(q*x) + b*sum(x).
template <int Bits>
static float quant_row_dot(const Linear& linear, int64_t row, const float* x, const float* xsum) {
  const int64_t group_size = linear.quant.group_size;
  const int64_t words_per_group = group_size * Bits / 32;
  const uint32_t* words = linear.words + row * linear.words_per_row();
  float total = 0.0f;
  for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
    const uint32_t* gw = words + g * words_per_group;
    const float* gx = x + g * group_size;
    float acc = 0.0f;
    if constexpr (32 % Bits == 0) {  // whole codes per word
      constexpr int kPerWord = 32 / Bits;
      constexpr uint32_t kMask = (1u << Bits) - 1u;
      for (int64_t w = 0; w < words_per_group; ++w) {
        uint32_t word = gw[w];
        const float* wx = gx + w * kPerWord;
        for (int j = 0; j < kPerWord; ++j) {
          acc += static_cast<float>((word >> (Bits * j)) & kMask) * wx[j];
        }
      }
    } else {  // 3, 5 or 6 bits: codes straddle words
      CodeStream codes(gw, Bits);
      for (int64_t c = 0; c < group_size; ++c) acc += static_cast<float>(codes.next()) * gx[c];
    }
    total += scale_value(linear, row, g) * acc + bias_value(linear, row, g) * xsum[g];
  }
  return total;
}

// Dot of one MXFP4 row with x: per group, the scale times the sum of
// E2M1 values times x, as MLX accumulates it.
static float mxfp4_row_dot(const Linear& linear, int64_t row, const float* x) {
  const uint32_t* words = linear.words + row * linear.words_per_row();
  const uint8_t* exponents = linear.exponents + row * linear.groups_per_row();
  float total = 0.0f;
  for (int64_t g = 0; g < linear.groups_per_row(); ++g) {
    const uint32_t* gw = words + g * 4;
    const float* gx = x + g * 32;
    float acc = 0.0f;
    for (int w = 0; w < 4; ++w) {
      const uint32_t word = gw[w];
      for (int j = 0; j < 8; ++j) acc += kFp4[(word >> (4 * j)) & 0xF] * gx[w * 8 + j];
    }
    total += e8m0_value(exponents[g]) * acc;
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
  if (linear.quant.mxfp4()) return mxfp4_row_dot(linear, row, x);
  switch (linear.quant.bits) {
    case 2: return quant_row_dot<2>(linear, row, x, xsum);
    case 3: return quant_row_dot<3>(linear, row, x, xsum);
    case 4: return quant_row_dot<4>(linear, row, x, xsum);
    case 5: return quant_row_dot<5>(linear, row, x, xsum);
    case 6: return quant_row_dot<6>(linear, row, x, xsum);
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
