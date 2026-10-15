// Seeded token sampling shared by every backend.
//
// Semantics follow mlx_lm's sampler: top-p, min-p and top-k filter the
// untempered distribution, then a token is drawn from what is left with
// probability proportional to exp(logit / temperature). Each filter keeps
// every token at or above a threshold logit, so the kept set is the
// intersection of the three and ties at a boundary are all kept.
//
// Draws use the Gumbel-max trick with a counter-based generator
// (Philox4x32-10): the token is the argmax over kept i of
// logit_i / T + G(seed, step, i), where step is the index of the generated
// token within the request. Noise depends only on (seed, step, token id), so
// the CPU sampler and the GPU kernel draw the same tokens without sharing
// state, and pipelining or batching cannot change a seeded result.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace lokahi {

struct SamplingParams {
  float temperature = 0.0f;  // 0 is greedy
  float top_p = 1.0f;        // nucleus mass in (0, 1]; 1 disables
  int top_k = 0;             // 0 disables
  float min_p = 0.0f;        // [0, 1]; 0 disables
  uint64_t seed = 0;

  bool greedy() const { return temperature <= 0.0f || top_k == 1; }
};

// Throws lokahi::Error when a parameter is out of range or not finite.
void validate_sampling(const SamplingParams& params);

using PhiloxBlock = std::array<uint32_t, 4>;
PhiloxBlock philox4x32_10(PhiloxBlock counter, std::array<uint32_t, 2> key);

// Separates sampling noise from any other use of the generator.
inline constexpr uint32_t kSamplingStream = 0x6C6F6B61u;  // "loka"

// Order-preserving map from a float to a uint32: a > b implies key(a) > key(b).
inline uint32_t order_key(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Gumbel noise for one token at one step; the GPU kernel computes the same.
float gumbel_noise(uint64_t seed, int step, int32_t token);

// Samples one token from `vocab` logits. Greedy parameters return the argmax
// (lowest index on ties). The reference for every backend's sampler.
int32_t sample_token(const float* logits, int vocab, const SamplingParams& params, int step);

}  // namespace lokahi
