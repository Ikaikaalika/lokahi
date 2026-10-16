// Scalar BF16/FP16 <-> FP32 conversions (portable, no compiler extensions).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace lokahi {

inline float bf16_to_f32(uint16_t bits) {
  uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &widened, sizeof(out));
  return out;
}

inline uint16_t f32_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if (std::isnan(value)) return 0x7FC0;
  uint32_t rounding = ((bits >> 16) & 1u) + 0x7FFFu;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

inline float f16_to_f32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      exponent = 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// To the nearest FP16, ties to even, as Metal converts float to half; values
// from 65520 up become infinity.
inline uint16_t f32_to_f16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t magnitude = bits & 0x7FFFFFFFu;
  if (magnitude >= 0x7F800000u) return static_cast<uint16_t>(sign | (magnitude > 0x7F800000u ? 0x7E00u : 0x7C00u));
  if (magnitude >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);
  if (magnitude < 0x38800000u) {  // below 2^-14: a subnormal (or zero) in units of 2^-24
    float scaled;
    std::memcpy(&scaled, &magnitude, sizeof(scaled));
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(scaled * 16777216.0f)));
  }
  uint32_t half = (magnitude - 0x38000000u) >> 13;  // rebiased exponent and the top 10 mantissa bits
  const uint32_t rest = magnitude & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;  // a carry moves to the next exponent
  return static_cast<uint16_t>(sign | half);
}

}  // namespace lokahi
