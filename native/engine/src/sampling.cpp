#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "common.h"

namespace lokahi {

void validate_sampling(const SamplingParams& p) {
  LK_CHECK(std::isfinite(p.temperature) && p.temperature >= 0.0f, "sampling: temperature must be finite and >= 0");
  LK_CHECK(std::isfinite(p.top_p) && p.top_p > 0.0f && p.top_p <= 1.0f, "sampling: top_p must be in (0, 1]");
  LK_CHECK(p.top_k >= 0, "sampling: top_k must be >= 0");
  LK_CHECK(std::isfinite(p.min_p) && p.min_p >= 0.0f && p.min_p <= 1.0f, "sampling: min_p must be in [0, 1]");
}

PhiloxBlock philox4x32_10(PhiloxBlock c, std::array<uint32_t, 2> k) {
  constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u, kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k[0] += kW0;
      k[1] += kW1;
    }
    const uint64_t p0 = uint64_t{kM0} * c[0];
    const uint64_t p1 = uint64_t{kM1} * c[2];
    c = {static_cast<uint32_t>(p1 >> 32) ^ c[1] ^ k[0], static_cast<uint32_t>(p1),
         static_cast<uint32_t>(p0 >> 32) ^ c[3] ^ k[1], static_cast<uint32_t>(p0)};
  }
  return c;
}

namespace {

// One Philox call covers four consecutive token ids.
PhiloxBlock noise_block(uint64_t seed, int step, int32_t block) {
  return philox4x32_10({static_cast<uint32_t>(block), static_cast<uint32_t>(step), kSamplingStream, 0u},
                       {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)});
}

// 23 random bits mapped strictly inside (0, 1), where float32 can represent
// both u and 1 - u, so neither logarithm reaches zero.
float gumbel_from_bits(uint32_t bits) {
  const float u = (static_cast<float>(bits >> 9) + 0.5f) * 0x1p-23f;
  return -std::log(-std::log(u));
}

int32_t argmax_lowest(const float* values, int count) {
  int32_t best = 0;
  for (int i = 1; i < count; ++i) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

// Smallest key k with mass(keys > k) < top_p * Z: a token stays in the
// nucleus while the tokens strictly more likely than it hold less than top_p
// of the mass (mlx_lm's rule, with boundary ties kept). Tokens below `floor`
// are already filtered out, so the walk stops there. The nucleus is usually
// small, so tokens are ordered in growing windows instead of sorting them all.
uint32_t top_p_threshold(const float* logits, const std::vector<uint32_t>& keys, float max_logit, float top_p,
                         uint32_t floor) {
  const int vocab = static_cast<int>(keys.size());
  double z = 0.0;
  for (int i = 0; i < vocab; ++i) z += std::exp(static_cast<double>(logits[i]) - max_logit);
  const double target = static_cast<double>(top_p) * z;
  std::vector<int32_t> order;
  for (int i = 0; i < vocab; ++i) {
    if (keys[i] >= floor) order.push_back(i);
  }
  auto before = [&](int32_t a, int32_t b) { return keys[a] != keys[b] ? keys[a] > keys[b] : a < b; };
  size_t window = std::min<size_t>(order.size(), 64);
  for (;;) {
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(window), order.end(), before);
    double mass = 0.0;
    size_t i = 0;
    while (i < window) {
      size_t j = i;
      double group = 0.0;
      while (j < window && keys[order[j]] == keys[order[i]]) {
        group += std::exp(static_cast<double>(logits[order[j]]) - max_logit);
        ++j;
      }
      if (j == window && window < order.size()) break;  // the tie group may continue past the window
      mass += group;
      if (mass >= target) return keys[order[i]];
      i = j;
    }
    if (window == order.size()) return floor;
    window = std::min(order.size(), window * 4);
  }
}

}  // namespace

float gumbel_noise(uint64_t seed, int step, int32_t token) {
  return gumbel_from_bits(noise_block(seed, step, token >> 2)[static_cast<size_t>(token & 3)]);
}

int32_t sample_token(const float* logits, int vocab, const SamplingParams& p, int step) {
  LK_CHECK(vocab > 0, "sampling: empty vocabulary");
  if (p.greedy()) return argmax_lowest(logits, vocab);
  validate_sampling(p);
  std::vector<uint32_t> keys(static_cast<size_t>(vocab));
  float max_logit = logits[0];
  for (int i = 0; i < vocab; ++i) {
    keys[static_cast<size_t>(i)] = order_key(logits[i]);
    max_logit = std::max(max_logit, logits[i]);
  }
  // Every filter is a threshold; the strictest one wins.
  uint32_t threshold = 0;
  if (p.min_p > 0.0f) threshold = std::max(threshold, order_key(max_logit + std::log(p.min_p)));
  if (p.top_k > 0 && p.top_k < vocab) {
    std::vector<uint32_t> sorted = keys;
    std::nth_element(sorted.begin(), sorted.begin() + (p.top_k - 1), sorted.end(), std::greater<>());
    threshold = std::max(threshold, sorted[static_cast<size_t>(p.top_k - 1)]);
  }
  if (p.top_p < 1.0f) threshold = std::max(threshold, top_p_threshold(logits, keys, max_logit, p.top_p, threshold));

  const float inv_temperature = 1.0f / p.temperature;
  float best = 0.0f;
  int32_t best_index = -1;
  for (int begin = 0; begin < vocab; begin += 4) {
    const int end = std::min(vocab, begin + 4);
    bool any = false;
    for (int i = begin; i < end; ++i) any = any || keys[static_cast<size_t>(i)] >= threshold;
    if (!any) continue;
    const PhiloxBlock bits = noise_block(p.seed, step, begin >> 2);
    for (int i = begin; i < end; ++i) {
      if (keys[static_cast<size_t>(i)] < threshold) continue;
      const float score = logits[i] * inv_temperature + gumbel_from_bits(bits[static_cast<size_t>(i - begin)]);
      if (best_index < 0 || score > best) {
        best = score;
        best_index = i;
      }
    }
  }
  return best_index;
}

}  // namespace lokahi
