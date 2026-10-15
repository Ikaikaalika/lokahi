// Lokahi Metal kernels for dense decoders (Gemma 3, Gemma 4 and the Llama
// family).
//
// Activations are float32. Linear weights are MLX affine-quantized codes with
// BF16 or F16 per-group scales and biases, or (in the matrix-vector kernels)
// MXFP4: E2M1 codes with one E8M0 exponent per 32 values. Quantization
// parameters are function constants so each pipeline is specialized at load
// time.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant int QBITS [[function_constant(0)]];
constant int QGROUP [[function_constant(1)]];
constant bool QSCALE_BF16 [[function_constant(2)]];
constant bool QMXFP4 [[function_constant(3)]];

// E2M1 values by code, in MLX's order, and an E8M0 exponent as MLX decodes it.
constant float kFp4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8m0(uchar e) { return as_type<float>(e == 0 ? 0x00400000u : uint(e) << 23); }

// Dot product of 16 consecutive E2M1 codes with x.
inline float fp4dot16(const device uint* w, thread const float* x) {
  const uint2 ww = *reinterpret_cast<const device uint2*>(w);
  float acc = 0.0f;
  for (int j = 0; j < 8; ++j) acc += kFp4[(ww.x >> (4 * j)) & 0xFu] * x[j];
  for (int j = 0; j < 8; ++j) acc += kFp4[(ww.y >> (4 * j)) & 0xFu] * x[8 + j];
  return acc;
}

inline float scale_at(const device ushort* values, int index) {
  ushort bits = values[index];
  if (QSCALE_BF16) return as_type<float>(uint(bits) << 16);
  return float(as_type<half>(bits));
}

// Dot product of 16 consecutive codes (starting at a 16-value boundary) with x.
inline float dot16(const device uint* w, thread const float* x) {
  float acc = 0.0f;
  if (QBITS == 4) {
    uint2 ww = *reinterpret_cast<const device uint2*>(w);
    for (int j = 0; j < 8; ++j) acc += float((ww.x >> (4 * j)) & 0xFu) * x[j];
    for (int j = 0; j < 8; ++j) acc += float((ww.y >> (4 * j)) & 0xFu) * x[8 + j];
  } else if (QBITS == 8) {
    uint4 ww = *reinterpret_cast<const device uint4*>(w);
    for (int j = 0; j < 4; ++j) acc += float((ww.x >> (8 * j)) & 0xFFu) * x[j];
    for (int j = 0; j < 4; ++j) acc += float((ww.y >> (8 * j)) & 0xFFu) * x[4 + j];
    for (int j = 0; j < 4; ++j) acc += float((ww.z >> (8 * j)) & 0xFFu) * x[8 + j];
    for (int j = 0; j < 4; ++j) acc += float((ww.w >> (8 * j)) & 0xFFu) * x[12 + j];
  } else {
    uint ww = w[0];
    for (int j = 0; j < 16; ++j) acc += float((ww >> (2 * j)) & 0x3u) * x[j];
  }
  return acc;
}

inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;
  return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) { return x / (1.0f + precise::exp(-x)); }

// The gated MLP's activation: act 1 is SiLU, 0 tanh GELU, 2 GPT-OSS's
// clamped SwiGLU min(g, L) * sigmoid(1.702 * min(g, L)) * (clip(u, -L, L) + 1).
inline float glu(float gate, float up, int act, float limit = 7.0f) {
  if (act == 2) {
    const float g = min(gate, limit);
    return g / (1.0f + precise::exp(-1.702f * g)) * (clamp(up, -limit, limit) + 1.0f);
  }
  return (act != 0 ? silu(gate) : gelu_tanh(gate)) * up;
}

// Threadgroup-wide sum. `scratch` holds one float per simdgroup.
inline float threadgroup_sum(float value, threadgroup float* scratch, uint sg, uint lane,
                             uint simdgroups) {
  value = simd_sum(value);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < simdgroups; ++i) total += scratch[i];
  return total;
}

// ---------------------------------------------------------------------------
// Embedding gather: out[t, c] = dequant(W[token[t], c]) * scale
// ---------------------------------------------------------------------------

struct EmbedParams {
  int cols;
  int n;
  float scale;
};

kernel void embed_rows(const device uint* w [[buffer(0)]],
                       const device ushort* scales [[buffer(1)]],
                       const device ushort* biases [[buffer(2)]],
                       const device int* tokens [[buffer(3)]],
                       device float* out [[buffer(4)]],
                       constant EmbedParams& p [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int t = int(gid.y);
  if (c >= p.cols || t >= p.n) return;
  const int per_word = 32 / QBITS;
  const int words_per_row = p.cols / per_word;
  const int groups_per_row = p.cols / QGROUP;
  const int token = tokens[t];
  const uint word = w[token * words_per_row + c / per_word];
  const uint code = (word >> (QBITS * (c % per_word))) & ((1u << QBITS) - 1u);
  const int g = token * groups_per_row + c / QGROUP;
  out[t * p.cols + c] = (scale_at(scales, g) * float(code) + scale_at(biases, g)) * p.scale;
}

struct ReplaceParams {
  int cols;
  int n;
};

// x[t, :] = rows[index[t], :] wherever index[t] >= 0: injected rows (image
// features) over the scaled embeddings.
kernel void replace_rows(const device float* rows [[buffer(0)]],
                         const device int* index [[buffer(1)]],
                         device float* x [[buffer(2)]],
                         constant ReplaceParams& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int t = int(gid.y);
  if (c >= p.cols || t >= p.n) return;
  const int r = index[t];
  if (r >= 0) x[t * p.cols + c] = rows[r * p.cols + c];
}

// ---------------------------------------------------------------------------
// RMSNorm scaled by (offset + w): Gemma's offset is 1, most families use 0.
// One threadgroup per row.
// ---------------------------------------------------------------------------

struct NormParams {
  int cols;
  float eps;
  float offset;
};

kernel void rms_norm_rows(const device float* in [[buffer(0)]],
                          const device float* weight [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant NormParams& p [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tpg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const device float* x = in + row * p.cols;
  float ss = 0.0f;
  for (int i = int(tid); i < p.cols; i += int(tpg)) ss += x[i] * x[i];
  const float total = threadgroup_sum(ss, scratch, sg, lane, (tpg + 31) / 32);
  const float inv = precise::rsqrt(total / float(p.cols) + p.eps);
  device float* y = out + row * p.cols;
  for (int i = int(tid); i < p.cols; i += int(tpg)) y[i] = x[i] * inv * (p.offset + weight[i]);
}

// x = (x + r) * scale, with r first normalized by w_post when post_norm
// (sandwich norms) and scale Gemma 4's layer scalar (else 1); then, if
// has_next, h = norm(x) * (offset + w_next).
struct ResidualNormParams {
  int cols;
  float eps;
  float offset;
  int post_norm;
  int has_next;
  float scale;
};

kernel void residual_norm_rows(device float* x_rows [[buffer(0)]],
                               const device float* r_rows [[buffer(1)]],
                               const device float* w_post [[buffer(2)]],
                               const device float* w_next [[buffer(3)]],
                               device float* h_rows [[buffer(4)]],
                               constant ResidualNormParams& p [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpg [[threads_per_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const uint simdgroups = (tpg + 31) / 32;
  device float* x = x_rows + row * p.cols;
  const device float* r = r_rows + row * p.cols;
  float xs = 0.0f;
  if (p.post_norm != 0) {  // Uniform across the threadgroup.
    float ss = 0.0f;
    for (int i = int(tid); i < p.cols; i += int(tpg)) ss += r[i] * r[i];
    const float inv_r = precise::rsqrt(threadgroup_sum(ss, scratch, sg, lane, simdgroups) / float(p.cols) + p.eps);
    for (int i = int(tid); i < p.cols; i += int(tpg)) {
      const float v = (x[i] + r[i] * inv_r * (p.offset + w_post[i])) * p.scale;
      x[i] = v;
      xs += v * v;
    }
  } else {
    for (int i = int(tid); i < p.cols; i += int(tpg)) {
      const float v = (x[i] + r[i]) * p.scale;
      x[i] = v;
      xs += v * v;
    }
  }
  if (p.has_next == 0) return;  // Uniform across the threadgroup.
  const float inv_x = precise::rsqrt(threadgroup_sum(xs, scratch, sg, lane, simdgroups) / float(p.cols) + p.eps);
  device float* h = h_rows + row * p.cols;
  for (int i = int(tid); i < p.cols; i += int(tpg)) h[i] = x[i] * inv_x * (p.offset + w_next[i]);
}

// ---------------------------------------------------------------------------
// Quantized matrix-vector products for single-token decode.
//
// Each simdgroup produces QMV_ROWS consecutive output rows; each lane covers
// 16 input values per iteration, so a simdgroup reads 512 inputs per step.
// Requires cols % 16 == 0 and QGROUP % 16 == 0.
// ---------------------------------------------------------------------------

constant constexpr int QMV_ROWS = 4;
constant constexpr int QMV_SIMDGROUPS = 2;

struct QmvParams {
  int rows;
  int cols;
};

inline void qmv_accumulate(const device uint* w, const device ushort* scales,
                           const device ushort* biases, const device float* x, int row0,
                           int rows, int cols, uint lane, thread float* acc) {
  const int words_per_row = cols * QBITS / 32;
  const int groups_per_row = cols / QGROUP;
  for (int k0 = int(lane) * 16; k0 < cols; k0 += 32 * 16) {
    float xv[16];
    const device float4* x4 = reinterpret_cast<const device float4*>(x + k0);
    for (int j = 0; j < 4; ++j) {
      const float4 v = x4[j];
      xv[4 * j] = v.x;
      xv[4 * j + 1] = v.y;
      xv[4 * j + 2] = v.z;
      xv[4 * j + 3] = v.w;
    }
    const int g = k0 / QGROUP;
    if (QMXFP4) {  // no biases; one exponent byte per group
      const device uchar* exponents = reinterpret_cast<const device uchar*>(scales);
      for (int r = 0; r < QMV_ROWS; ++r) {
        const int row = row0 + r;
        if (row < rows) {
          const device uint* wr = w + row * words_per_row + k0 * QBITS / 32;
          acc[r] += e8m0(exponents[row * groups_per_row + g]) * fp4dot16(wr, xv);
        }
      }
      continue;
    }
    float xsum = 0.0f;
    for (int j = 0; j < 16; ++j) xsum += xv[j];
    for (int r = 0; r < QMV_ROWS; ++r) {
      const int row = row0 + r;
      if (row < rows) {
        const device uint* wr = w + row * words_per_row + k0 * QBITS / 32;
        const int gi = row * groups_per_row + g;
        acc[r] += scale_at(scales, gi) * dot16(wr, xv) + scale_at(biases, gi) * xsum;
      }
    }
  }
}

kernel void qmv(const device uint* w [[buffer(0)]],
                const device ushort* scales [[buffer(1)]],
                const device ushort* biases [[buffer(2)]],
                const device float* x [[buffer(3)]],
                device float* y [[buffer(4)]],
                constant QmvParams& p [[buffer(5)]],
                uint tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
  const int row0 = (int(tg) * QMV_SIMDGROUPS + int(sg)) * QMV_ROWS;
  if (row0 >= p.rows) return;
  float acc[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  qmv_accumulate(w, scales, biases, x, row0, p.rows, p.cols, lane, acc);
  for (int r = 0; r < QMV_ROWS; ++r) {
    const float total = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < p.rows) y[row0 + r] = total;
  }
}

// q, k and v projections of one token in one dispatch. Row space is the
// concatenation [q_rows | kv_rows | kv_rows]; every 4-row block lies in one
// matrix because q_rows and kv_rows are multiples of 4.
struct QkvParams {
  int q_rows;
  int kv_rows;
  int cols;
};

kernel void qmv_qkv(const device uint* wq [[buffer(0)]],
                    const device ushort* sq [[buffer(1)]],
                    const device ushort* bq [[buffer(2)]],
                    const device uint* wk [[buffer(3)]],
                    const device ushort* sk [[buffer(4)]],
                    const device ushort* bk [[buffer(5)]],
                    const device uint* wv [[buffer(6)]],
                    const device ushort* sv [[buffer(7)]],
                    const device ushort* bv [[buffer(8)]],
                    const device float* x [[buffer(9)]],
                    device float* q_out [[buffer(10)]],
                    device float* k_out [[buffer(11)]],
                    device float* v_out [[buffer(12)]],
                    constant QkvParams& p [[buffer(13)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  int row0 = (int(tg) * QMV_SIMDGROUPS + int(sg)) * QMV_ROWS;
  const device uint* w = wq;
  const device ushort* s = sq;
  const device ushort* b = bq;
  device float* y = q_out;
  int rows = p.q_rows;
  if (row0 >= p.q_rows + p.kv_rows) {
    row0 -= p.q_rows + p.kv_rows;
    w = wv; s = sv; b = bv; y = v_out; rows = p.kv_rows;
  } else if (row0 >= p.q_rows) {
    row0 -= p.q_rows;
    w = wk; s = sk; b = bk; y = k_out; rows = p.kv_rows;
  }
  if (row0 >= rows) return;
  float acc[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  qmv_accumulate(w, s, b, x, row0, rows, p.cols, lane, acc);
  for (int r = 0; r < QMV_ROWS; ++r) {
    const float total = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < rows) y[row0 + r] = total;
  }
}

// out = act(gate(x)) * up(x) for one token.
struct GluParams {
  int rows;
  int cols;
  int silu;
};

kernel void qmv_glu(const device uint* wg [[buffer(0)]],
                      const device ushort* sg_scales [[buffer(1)]],
                      const device ushort* bg [[buffer(2)]],
                      const device uint* wu [[buffer(3)]],
                      const device ushort* su [[buffer(4)]],
                      const device ushort* bu [[buffer(5)]],
                      const device float* x [[buffer(6)]],
                      device float* y [[buffer(7)]],
                      constant GluParams& p [[buffer(8)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const int row0 = (int(tg) * QMV_SIMDGROUPS + int(sg)) * QMV_ROWS;
  if (row0 >= p.rows) return;
  float gate[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  qmv_accumulate(wg, sg_scales, bg, x, row0, p.rows, p.cols, lane, gate);
  qmv_accumulate(wu, su, bu, x, row0, p.rows, p.cols, lane, up);
  for (int r = 0; r < QMV_ROWS; ++r) {
    const float g = simd_sum(gate[r]);
    const float u = simd_sum(up[r]);
    if (lane == 0 && row0 + r < p.rows) y[row0 + r] = glu(g, u, p.silu);
  }
}

// ---------------------------------------------------------------------------
// Mixture of experts. moe_route picks each token's experts from its router
// logits; qmv_gather_glu and qmv_gather run the chosen experts' projections
// for every (token, slot) pair, reading the expert index from device memory
// and offsetting into the stacked [experts, rows, ...] weights; moe_combine
// sums the slots by their router weights.
// ---------------------------------------------------------------------------

struct MoeRouteParams {
  int experts;
  int k;
  int topk_softmax;   // 1: softmax over the chosen logits; 0: softmax over all experts
  int norm_topk;      // with softmax over all experts, renormalize over the chosen
  int has_bias;       // add the router bias first (GPT-OSS)
  int has_scale;      // multiply each weight by its expert's scale (Gemma 4)
};

// One simdgroup per token. Ties go to the lower expert index.
kernel void moe_route(device float* logits [[buffer(0)]],
                      device int* chosen [[buffer(1)]],
                      device float* weights [[buffer(2)]],
                      constant MoeRouteParams& p [[buffer(3)]],
                      const device float* bias [[buffer(4)]],
                      const device float* expert_scale [[buffer(5)]],
                      uint t [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
  device float* l = logits + int(t) * p.experts;
  if (p.has_bias != 0) {
    for (int e = int(lane); e < p.experts; e += 32) l[e] += bias[e];
    simdgroup_barrier(mem_flags::mem_device);
  }
  float top = -INFINITY;
  for (int e = int(lane); e < p.experts; e += 32) top = max(top, l[e]);
  top = simd_max(top);
  float total = 0.0f;
  for (int e = int(lane); e < p.experts; e += 32) total += exp(l[e] - top);
  total = simd_sum(total);
  // Experts already chosen, as a bit set over up to 256 experts.
  uint taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float kept = 0.0f;
  for (int j = 0; j < p.k; ++j) {
    float best = -INFINITY;
    int index = 1 << 30;
    for (int e = int(lane); e < p.experts; e += 32) {
      const bool used = (taken[e >> 5] >> (e & 31)) & 1u;
      if (!used && (l[e] > best || (l[e] == best && e < index))) {
        best = l[e];
        index = e;
      }
    }
    const float winner = simd_max(best);
    // NaN logits would leave no winner; never hand an out-of-range expert on.
    const int e = min(simd_min(best == winner ? index : (1 << 30)), p.experts - 1);
    taken[e >> 5] |= 1u << (e & 31);
    const float weight = exp(l[e] - top);
    kept += weight;
    if (lane == 0) {
      chosen[int(t) * p.k + j] = e;
      weights[int(t) * p.k + j] = weight;
    }
  }
  const float scale = (p.topk_softmax != 0 || p.norm_topk != 0) ? 1.0f / kept : 1.0f / total;
  if (lane == 0) {
    for (int j = 0; j < p.k; ++j) {
      const int slot = int(t) * p.k + j;
      weights[slot] *= p.has_scale != 0 ? scale * expert_scale[chosen[slot]] : scale;
    }
  }
}

// Expert e's scales in a stacked tensor: MXFP4 exponents are bytes, affine
// scales and biases two-byte values.
inline const device ushort* expert_scales(const device ushort* scales, int e, int groups) {
  if (QMXFP4) return reinterpret_cast<const device ushort*>(reinterpret_cast<const device uchar*>(scales) + e * groups);
  return scales + e * groups;
}

struct GatherParams {
  int rows;
  int cols;
  int k;            // slots per token
  int input_rows;   // 1: input row t = pair / k; 0: input row = pair
  int act;          // glu's activation (gate and up kernel)
  int has_bias;     // per-expert output biases [experts, rows]
  float limit;      // the clamped SwiGLU's limit
};

// y[pair, :] = act(gate_e(x)) * up_e(x) with e = chosen[pair].
kernel void qmv_gather_glu(const device uint* wg [[buffer(0)]],
                           const device ushort* sg_scales [[buffer(1)]],
                           const device ushort* bg [[buffer(2)]],
                           const device uint* wu [[buffer(3)]],
                           const device ushort* su [[buffer(4)]],
                           const device ushort* bu [[buffer(5)]],
                           const device float* x [[buffer(6)]],
                           device float* y [[buffer(7)]],
                           constant GatherParams& p [[buffer(8)]],
                           const device int* chosen [[buffer(9)]],
                           const device float* gate_bias [[buffer(10)]],
                           const device float* up_bias [[buffer(11)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const int row0 = (int(tg.x) * QMV_SIMDGROUPS + int(sg)) * QMV_ROWS;
  const int pair = int(tg.y);
  if (row0 >= p.rows) return;
  const int e = chosen[pair];
  const int words = p.rows * (p.cols * QBITS / 32), groups = p.rows * (p.cols / QGROUP);
  const device float* xr = x + (p.input_rows != 0 ? pair / p.k : pair) * p.cols;
  float gate[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  qmv_accumulate(wg + e * words, expert_scales(sg_scales, e, groups), bg + e * groups, xr, row0, p.rows, p.cols, lane,
                 gate);
  qmv_accumulate(wu + e * words, expert_scales(su, e, groups), bu + e * groups, xr, row0, p.rows, p.cols, lane, up);
  for (int r = 0; r < QMV_ROWS; ++r) {
    float g = simd_sum(gate[r]);
    float u = simd_sum(up[r]);
    const int row = row0 + r;
    if (lane == 0 && row < p.rows) {
      if (p.has_bias != 0) {
        g += gate_bias[e * p.rows + row];
        u += up_bias[e * p.rows + row];
      }
      y[pair * p.rows + row] = glu(g, u, p.act, p.limit);
    }
  }
}

// y[pair, :] = W_e x[pair or pair / k] with e = chosen[pair].
kernel void qmv_gather(const device uint* w [[buffer(0)]],
                       const device ushort* scales [[buffer(1)]],
                       const device ushort* biases [[buffer(2)]],
                       const device float* x [[buffer(3)]],
                       device float* y [[buffer(4)]],
                       constant GatherParams& p [[buffer(5)]],
                       const device int* chosen [[buffer(6)]],
                       const device float* out_bias [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const int row0 = (int(tg.x) * QMV_SIMDGROUPS + int(sg)) * QMV_ROWS;
  const int pair = int(tg.y);
  if (row0 >= p.rows) return;
  const int e = chosen[pair];
  const int words = p.rows * (p.cols * QBITS / 32), groups = p.rows * (p.cols / QGROUP);
  const device float* xr = x + (p.input_rows != 0 ? pair / p.k : pair) * p.cols;
  float acc[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  qmv_accumulate(w + e * words, expert_scales(scales, e, groups), biases + e * groups, xr, row0, p.rows, p.cols, lane,
                 acc);
  for (int r = 0; r < QMV_ROWS; ++r) {
    const float total = simd_sum(acc[r]);
    const int row = row0 + r;
    if (lane == 0 && row < p.rows) y[pair * p.rows + row] = total + (p.has_bias != 0 ? out_bias[e * p.rows + row] : 0.0f);
  }
}

// GLM-4.5 / DeepSeek V3 (noaux_tc) routing, one simdgroup per token:
// sigmoid scores plus the correction bias choose the top k within the best
// topk_groups groups (by each group's two best); the chosen experts' raw
// scores, renormalized when norm_topk (and k > 1), times scaling, weight
// them. Ties go to the lower group or expert index. Lane 0 does the
// selection: at most 256 experts, so it is cheap next to the experts.
struct MoeGroupedParams {
  int experts;
  int k;
  int groups;
  int topk_groups;
  int norm_topk;
  float scaling;
};

kernel void moe_route_grouped(const device float* logits [[buffer(0)]],
                              device int* chosen [[buffer(1)]],
                              device float* weights [[buffer(2)]],
                              constant MoeGroupedParams& p [[buffer(3)]],
                              const device float* correction [[buffer(4)]],
                              uint t [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float score[256];
  threadgroup float selection[256];
  const device float* l = logits + int(t) * p.experts;
  for (int e = int(lane); e < p.experts; e += 32) {
    score[e] = 1.0f / (1.0f + precise::exp(-l[e]));
    selection[e] = score[e] + correction[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane != 0) return;
  const int per_group = p.experts / p.groups;
  if (p.groups > 1) {
    float group_score[32];
    for (int g = 0; g < p.groups; ++g) {
      float first = -INFINITY, second = -INFINITY;
      for (int e = g * per_group; e < (g + 1) * per_group; ++e) {
        const float v = selection[e];
        if (v > first) {
          second = first;
          first = v;
        } else if (v > second) {
          second = v;
        }
      }
      group_score[g] = first + (per_group > 1 ? second : 0.0f);
    }
    uint dropped = 0;
    for (int j = 0; j < p.groups - p.topk_groups; ++j) {  // zero the lowest groups
      int worst = -1;
      for (int g = 0; g < p.groups; ++g) {
        if (!((dropped >> g) & 1u) && (worst < 0 || group_score[g] < group_score[worst])) worst = g;
      }
      dropped |= 1u << worst;
      for (int e = worst * per_group; e < (worst + 1) * per_group; ++e) selection[e] = 0.0f;
    }
  }
  uint taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float total = 0.0f;
  for (int j = 0; j < p.k; ++j) {
    int best = -1;
    for (int e = 0; e < p.experts; ++e) {
      if (!((taken[e >> 5] >> (e & 31)) & 1u) && (best < 0 || selection[e] > selection[best])) best = e;
    }
    taken[best >> 5] |= 1u << (best & 31);
    chosen[int(t) * p.k + j] = best;
    weights[int(t) * p.k + j] = score[best];
    total += score[best];
  }
  const float scale = (p.k > 1 && p.norm_topk != 0 ? 1.0f / total : 1.0f) * p.scaling;
  for (int j = 0; j < p.k; ++j) weights[int(t) * p.k + j] *= scale;
}

// y[t, r] = W[r, :] . x[t, :] for a dense matrix (dtype 0 F32, 1 BF16, 2
// F16): GLM's unquantized router. Grid: threadgroups (row pairs, n_tokens),
// two simdgroups of one row each.
struct DenseParams {
  int rows;
  int cols;
  int dtype;
};

inline float dense_at(const device uchar* w, int index, int dtype) {
  if (dtype == 1) return as_type<float>(uint(reinterpret_cast<const device ushort*>(w)[index]) << 16);
  if (dtype == 2) return float(reinterpret_cast<const device half*>(w)[index]);
  return reinterpret_cast<const device float*>(w)[index];
}

kernel void dense_rows(const device uchar* w [[buffer(0)]],
                       const device float* x [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       constant DenseParams& p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tg.x) * 2 + int(sg);
  const int t = int(tg.y);
  if (row >= p.rows) return;
  float acc = 0.0f;
  for (int c = int(lane); c < p.cols; c += 32) acc += dense_at(w, row * p.cols + c, p.dtype) * x[t * p.cols + c];
  acc = simd_sum(acc);
  if (lane == 0) y[t * p.rows + row] = acc;
}

struct BiasParams {
  int cols;
  int count;  // n_rows * cols
};

// rows[t, :] += bias (GPT-OSS's attention output bias).
kernel void add_bias_rows(device float* rows [[buffer(0)]],
                          const device float* bias [[buffer(1)]],
                          constant BiasParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) rows[gid] += bias[int(gid) % p.cols];
}

struct MoeCombineParams {
  int cols;   // hidden
  int k;
  int count;  // n_tokens * hidden
  int has_shared;
};

// out[t, :] = sum_j weights[t, j] * slots[t * k + j, :] (+ shared[t, :])
kernel void moe_combine(const device float* slots [[buffer(0)]],
                        const device float* weights [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant MoeCombineParams& p [[buffer(3)]],
                        const device float* shared [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.count) return;
  const int t = int(gid) / p.cols;
  const int d = int(gid) % p.cols;
  float total = p.has_shared != 0 ? shared[gid] : 0.0f;
  for (int j = 0; j < p.k; ++j) total += weights[t * p.k + j] * slots[(t * p.k + j) * p.cols + d];
  out[gid] = total;
}

// ---------------------------------------------------------------------------
// Quantized matrix-matrix product for prefill: Y[n, rows] = X[n, cols] W^T.
//
// 32x32 output tiles, 4 simdgroups each owning a 16x16 quadrant of 8x8
// simdgroup matrices; K advances 32 at a time with W dequantized into
// threadgroup memory. Requires cols % 32 == 0 and QGROUP % 32 == 0.
// ---------------------------------------------------------------------------

constant constexpr int QMM_BM = 32;
constant constexpr int QMM_BN = 32;
constant constexpr int QMM_BK = 32;
constant constexpr int QMM_PAD = 4;

struct QmmParams {
  int n;
  int rows;
  int cols;
};

kernel void qmm(const device uint* w [[buffer(0)]],
                const device ushort* scales [[buffer(1)]],
                const device ushort* biases [[buffer(2)]],
                const device float* x [[buffer(3)]],
                device float* y [[buffer(4)]],
                constant QmmParams& p [[buffer(5)]],
                uint2 tg [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float xs[QMM_BM][QMM_BK + QMM_PAD];
  threadgroup float ws[QMM_BN][QMM_BK + QMM_PAD];
  threadgroup float out_tile[QMM_BM][QMM_BN + QMM_PAD];

  const int m0 = int(tg.y) * QMM_BM;  // first token of the tile
  const int n0 = int(tg.x) * QMM_BN;  // first output row of the tile
  const int sm = int(sg / 2) * 16;
  const int sn = int(sg % 2) * 16;
  const int words_per_row = p.cols * QBITS / 32;
  const int groups_per_row = p.cols / QGROUP;

  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  // Each of the 128 threads dequantizes 8 consecutive weights of one row.
  const int w_row = int(tid) * 8 / QMM_BK;
  const int w_col = int(tid) * 8 % QMM_BK;

  for (int k0 = 0; k0 < p.cols; k0 += QMM_BK) {
    for (int i = int(tid); i < QMM_BM * QMM_BK; i += 128) {
      const int r = i / QMM_BK;
      const int c = i % QMM_BK;
      const int token = m0 + r;
      xs[r][c] = token < p.n ? x[token * p.cols + k0 + c] : 0.0f;
    }
    {
      const int row = n0 + w_row;
      const int col = k0 + w_col;
      float vals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      if (row < p.rows) {
        const int gi = row * groups_per_row + col / QGROUP;
        const float s = scale_at(scales, gi);
        const float b = scale_at(biases, gi);
        const device uint* wr = w + row * words_per_row;
        if (QBITS == 4) {
          const uint word = wr[col / 8];
          for (int j = 0; j < 8; ++j) vals[j] = s * float((word >> (4 * j)) & 0xFu) + b;
        } else if (QBITS == 8) {
          const uint lo = wr[col / 4];
          const uint hi = wr[col / 4 + 1];
          for (int j = 0; j < 4; ++j) vals[j] = s * float((lo >> (8 * j)) & 0xFFu) + b;
          for (int j = 0; j < 4; ++j) vals[4 + j] = s * float((hi >> (8 * j)) & 0xFFu) + b;
        } else {
          const uint word = wr[col / 16];
          const int base = (col % 16) * 2;
          for (int j = 0; j < 8; ++j) vals[j] = s * float((word >> (base + 2 * j)) & 0x3u) + b;
        }
      }
      for (int j = 0; j < 8; ++j) ws[w_row][w_col + j] = vals[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < QMM_BK; kk += 8) {
      simdgroup_float8x8 a[2];
      simdgroup_float8x8 b[2];
      for (int i = 0; i < 2; ++i) simdgroup_load(a[i], &xs[sm + 8 * i][kk], QMM_BK + QMM_PAD);
      for (int j = 0; j < 2; ++j)
        simdgroup_load(b[j], &ws[sn + 8 * j][kk], QMM_BK + QMM_PAD, ulong2(0, 0), true);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], &out_tile[sm + 8 * i][sn + 8 * j], QMM_BN + QMM_PAD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < QMM_BM * QMM_BN; i += 128) {
    const int r = i / QMM_BN;
    const int c = i % QMM_BN;
    const int token = m0 + r;
    const int row = n0 + c;
    if (token < p.n && row < p.rows) y[token * p.rows + row] = out_tile[r][c];
  }
}

struct ElementwiseParams {
  int count;
};

struct GluMulParams {
  int count;
  int silu;
};

kernel void glu_mul(device float* gate [[buffer(0)]],
                    const device float* up [[buffer(1)]],
                    constant GluMulParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) gate[gid] = glu(gate[gid], up[gid], p.silu);
}

kernel void softcap(device float* values [[buffer(0)]],
                    constant float& cap [[buffer(1)]],
                    constant ElementwiseParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) values[gid] = precise::tanh(values[gid] / cap) * cap;
}

// ---------------------------------------------------------------------------
// Optional q/k/v biases, optional per-head RMSNorm, rotate-half RoPE for q
// and k, an optional scale-free RMSNorm on v (Gemma 4), and KV-cache writes.
// Grid: threadgroups (heads + kv_heads, n_tokens), or heads alone for layers
// that share another layer's keys and values; head_dim / 2 threads each.
// Rotations with inverse frequency 0 (partial RoPE) leave values unchanged.
// ---------------------------------------------------------------------------

struct RopeParams {
  int heads;
  int kv_heads;
  int head_dim;
  int start_pos;
  int capacity;
  float eps;
  float offset;
  int qk_norm;
  int qkv_bias;
  int v_norm;
  int rotary;  // dimensions RoPE rotates: head_dim, or fewer for classic partial RoPE (GLM)
};

// Classic partial RoPE for value j of a normed head in `head`: rotate-half
// within the first 2 * rh dimensions; the rest pass through.
inline float rotate_partial(threadgroup const float* head, int j, int rh, int pos, const device float* inv_freq) {
  if (j >= 2 * rh) return head[j];
  const int k = j < rh ? j : j - rh;
  const float angle = float(pos) * inv_freq[k];
  const float cs = precise::cos(angle);
  const float sn = precise::sin(angle);
  return j < rh ? head[j] * cs - head[j + rh] * sn : head[j] * cs + head[j - rh] * sn;
}

kernel void qk_norm_rope_store(device float* q [[buffer(0)]],
                               const device float* k_in [[buffer(1)]],
                               const device float* v_in [[buffer(2)]],
                               device float* k_cache [[buffer(3)]],
                               device float* v_cache [[buffer(4)]],
                               const device float* q_norm [[buffer(5)]],
                               const device float* k_norm [[buffer(6)]],
                               const device float* inv_freq [[buffer(7)]],
                               constant RopeParams& p [[buffer(8)]],
                               const device float* q_bias [[buffer(9)]],
                               const device float* k_bias [[buffer(10)]],
                               const device float* v_bias [[buffer(11)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               uint2 tid2 [[thread_position_in_threadgroup]],
                               uint2 tpg2 [[threads_per_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  // Position builtins must all share one vector width, hence uint2 + .x.
  threadgroup float scratch[32];
  threadgroup float normed[512];  // one head, for classic partial RoPE
  const uint tid = tid2.x;
  const uint tpg = tpg2.x;
  const int head = int(tgp.x);
  const int t = int(tgp.y);
  const int half_dim = p.head_dim / 2;
  const int pos = p.start_pos + t;
  const bool is_q = head < p.heads;
  const int kvh = head - p.heads;
  const device float* src = is_q ? q + (t * p.heads + head) * p.head_dim
                                 : k_in + (t * p.kv_heads + kvh) * p.head_dim;
  const int i = int(tid);
  float a = src[i];
  float b = src[i + half_dim];
  if (p.qkv_bias != 0) {
    const device float* bias = is_q ? q_bias + head * p.head_dim : k_bias + kvh * p.head_dim;
    a += bias[i];
    b += bias[i + half_dim];
  }
  float na = a;
  float nb = b;
  if (p.qk_norm != 0) {  // Uniform across the threadgroup.
    const float ss = threadgroup_sum(a * a + b * b, scratch, sg, lane, (tpg + 31) / 32);
    const float inv = precise::rsqrt(ss / float(p.head_dim) + p.eps);
    const device float* nw = is_q ? q_norm : k_norm;
    na = a * inv * (p.offset + nw[i]);
    nb = b * inv * (p.offset + nw[i + half_dim]);
  }
  float ra, rb;
  if (p.rotary == p.head_dim) {
    const float angle = float(pos) * inv_freq[i];
    const float cs = precise::cos(angle);
    const float sn = precise::sin(angle);
    ra = na * cs - nb * sn;
    rb = nb * cs + na * sn;
  } else {  // Uniform across the threadgroup.
    normed[i] = na;
    normed[i + half_dim] = nb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ra = rotate_partial(normed, i, p.rotary / 2, pos, inv_freq);
    rb = rotate_partial(normed, i + half_dim, p.rotary / 2, pos, inv_freq);
  }
  if (is_q) {
    device float* dst = q + (t * p.heads + head) * p.head_dim;
    dst[i] = ra;
    dst[i + half_dim] = rb;
    return;
  }
  const int slot = pos % p.capacity;
  device float* kd = k_cache + (kvh * p.capacity + slot) * p.head_dim;
  kd[i] = ra;
  kd[i + half_dim] = rb;
  const device float* vs = v_in + (t * p.kv_heads + kvh) * p.head_dim;
  device float* vd = v_cache + (kvh * p.capacity + slot) * p.head_dim;
  const device float* vb = v_bias + kvh * p.head_dim;
  float v_lo = vs[i] + (p.qkv_bias != 0 ? vb[i] : 0.0f);
  float v_hi = vs[i + half_dim] + (p.qkv_bias != 0 ? vb[i + half_dim] : 0.0f);
  if (p.v_norm != 0) {  // Uniform across the threadgroup (every kv head).
    const float ss = threadgroup_sum(v_lo * v_lo + v_hi * v_hi, scratch, sg, lane, (tpg + 31) / 32);
    const float inv = precise::rsqrt(ss / float(p.head_dim) + p.eps);
    v_lo *= inv;
    v_hi *= inv;
  }
  vd[i] = v_lo;
  vd[i + half_dim] = v_hi;
}

// ---------------------------------------------------------------------------
// Gemma 4's per-layer inputs. per_layer_combine turns the pass's projection
// of the scaled embedding and its per-layer embedding lookup (already times
// sqrt(P)) into out[t, l, :] = (norm(proj * proj_scale) * w + lookup) *
// input_scale; grid: threadgroups (layers, n_tokens). per_layer_gate then
// gates a layer: g[t, :] = gelu(g[t, :]) * inputs[t, layer, :].
// ---------------------------------------------------------------------------

struct PerLayerCombineParams {
  int size;    // P
  int layers;
  float eps;
  float offset;
  float proj_scale;
  float input_scale;
};

kernel void per_layer_combine(const device float* proj [[buffer(0)]],
                              const device float* lookup [[buffer(1)]],
                              const device float* weight [[buffer(2)]],
                              device float* out [[buffer(3)]],
                              constant PerLayerCombineParams& p [[buffer(4)]],
                              uint2 tgp [[threadgroup_position_in_grid]],
                              uint2 tid2 [[thread_position_in_threadgroup]],
                              uint2 tpg2 [[threads_per_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const uint tid = tid2.x;
  const uint tpg = tpg2.x;
  const int row = int(tgp.y) * p.layers + int(tgp.x);
  const device float* x = proj + row * p.size;
  float ss = 0.0f;
  for (int i = int(tid); i < p.size; i += int(tpg)) {
    const float v = x[i] * p.proj_scale;
    ss += v * v;
  }
  const float inv = precise::rsqrt(threadgroup_sum(ss, scratch, sg, lane, (tpg + 31) / 32) / float(p.size) + p.eps);
  const device float* e = lookup + row * p.size;
  device float* y = out + row * p.size;
  for (int i = int(tid); i < p.size; i += int(tpg)) {
    y[i] = (x[i] * p.proj_scale * inv * (p.offset + weight[i]) + e[i]) * p.input_scale;
  }
}

struct PerLayerGateParams {
  int size;    // P
  int layers;
  int layer;
  int count;   // n_tokens * P
};

kernel void per_layer_gate(device float* g [[buffer(0)]],
                           const device float* inputs [[buffer(1)]],
                           constant PerLayerGateParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.count) return;
  const int t = int(gid) / p.size;
  const int i = int(gid) % p.size;
  g[gid] = gelu_tanh(g[gid]) * inputs[(t * p.layers + p.layer) * p.size + i];
}

// ---------------------------------------------------------------------------
// Attention with online softmax, split over key blocks.
//
// Grid: threadgroups (n_blocks, kv_heads, n_tokens), 4 simdgroups each. A
// threadgroup handles every query head that shares one kv head for one
// token and one block of keys; its simdgroups stride over the block's keys
// and merge in threadgroup memory. attn_reduce combines the blocks.
// ---------------------------------------------------------------------------

constant constexpr int ATT_SIMDGROUPS = 4;

struct AttnParams {
  int heads;
  int kv_heads;
  int head_dim;
  int start_pos;  // position of token 0 of this submission
  int window;     // keys allowed per query (sliding window or a large value)
  int capacity;   // KV slots for this layer
  int block;      // keys per threadgroup
  int n_blocks;
  float scale;
  float softcap;  // 0 disables
  int has_last_key;  // last_key[t] extends query t's keys (Gemma 3's image blocks)
};

template <int G, int DPL>
[[kernel]] void attn_partial(const device float* q [[buffer(0)]],
                             const device float* k_cache [[buffer(1)]],
                             const device float* v_cache [[buffer(2)]],
                             device float* part_o [[buffer(3)]],
                             device float* part_ml [[buffer(4)]],
                             constant AttnParams& p [[buffer(5)]],
                             const device int* last_key [[buffer(6)]],
                             uint3 tgp [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_o[G * DPL * 32];
  threadgroup float sh_m[G];
  threadgroup float sh_l[G];

  const int blk = int(tgp.x);
  const int kvh = int(tgp.y);
  const int t = int(tgp.z);
  const int D = p.head_dim;
  const int pos = p.start_pos + t;
  const int lo = max(0, pos - p.window + 1);
  const int hi = p.has_last_key != 0 ? max(pos, last_key[t]) : pos;
  const int kb = lo + blk * p.block;
  const int ke = min(kb + p.block, hi + 1);

  float qv[G][DPL];
  for (int g = 0; g < G; ++g) {
    const device float* qh = q + (t * p.heads + kvh * G + g) * D + int(lane) * DPL;
    for (int j = 0; j < DPL; ++j) qv[g][j] = qh[j];
  }
  float m[G], l[G], acc[G][DPL];
  for (int g = 0; g < G; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0f;
    for (int j = 0; j < DPL; ++j) acc[g][j] = 0.0f;
  }

  for (int key = kb + int(sg); key < ke; key += ATT_SIMDGROUPS) {
    const int slot = key % p.capacity;
    const device float* kp = k_cache + (kvh * p.capacity + slot) * D + int(lane) * DPL;
    const device float* vp = v_cache + (kvh * p.capacity + slot) * D + int(lane) * DPL;
    float kv[DPL], vv[DPL];
    for (int j = 0; j < DPL; ++j) {
      kv[j] = kp[j];
      vv[j] = vp[j];
    }
    for (int g = 0; g < G; ++g) {
      float partial = 0.0f;
      for (int j = 0; j < DPL; ++j) partial += qv[g][j] * kv[j];
      float s = simd_sum(partial) * p.scale;
      if (p.softcap > 0.0f) s = precise::tanh(s / p.softcap) * p.softcap;
      const float m_new = max(m[g], s);
      const float correction = exp(m[g] - m_new);
      const float weight = exp(s - m_new);
      l[g] = l[g] * correction + weight;
      for (int j = 0; j < DPL; ++j) acc[g][j] = acc[g][j] * correction + weight * vv[j];
      m[g] = m_new;
    }
  }

  // Merge simdgroups 1..3 into simdgroup 0 through threadgroup memory.
  for (int src = 1; src < ATT_SIMDGROUPS; ++src) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(sg) == src) {
      for (int g = 0; g < G; ++g) {
        if (lane == 0) {
          sh_m[g] = m[g];
          sh_l[g] = l[g];
        }
        for (int j = 0; j < DPL; ++j) sh_o[(g * 32 + int(lane)) * DPL + j] = acc[g][j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      for (int g = 0; g < G; ++g) {
        const float om = sh_m[g];
        const float ol = sh_l[g];
        if (ol == 0.0f) continue;
        const float m_new = max(m[g], om);
        const float ca = exp(m[g] - m_new);
        const float cb = exp(om - m_new);
        l[g] = l[g] * ca + ol * cb;
        for (int j = 0; j < DPL; ++j)
          acc[g][j] = acc[g][j] * ca + sh_o[(g * 32 + int(lane)) * DPL + j] * cb;
        m[g] = m_new;
      }
    }
  }
  if (sg != 0) return;
  for (int g = 0; g < G; ++g) {
    const int head = kvh * G + g;
    const int index = (t * p.heads + head) * p.n_blocks + blk;
    device float* o = part_o + index * D + int(lane) * DPL;
    for (int j = 0; j < DPL; ++j) o[j] = acc[g][j];
    if (lane == 0) {
      part_ml[2 * index] = m[g];
      part_ml[2 * index + 1] = l[g];
    }
  }
}

// The Linux syntax check (tests/syntax_stubs) parses this file as C++, which
// does not accept attributes on explicit instantiations; it blanks this macro.
#ifndef LOKAHI_INSTANCE_ATTRS
#define LOKAHI_INSTANCE_ATTRS(name) [[host_name(name)]] [[kernel]]
#endif
#define LOKAHI_ATTN(G, DPL)                                                 \
  template LOKAHI_INSTANCE_ATTRS("attn_partial_g" #G "_d" #DPL)             \
  decltype(attn_partial<G, DPL>) attn_partial<G, DPL>;

// Query heads per kv head (G) seen in supported checkpoints run from 1 to 8
// (Qwen 2.5 uses 6 and 7, Qwen 3 14B uses 5); head_dim is 32 * DPL, up to
// Gemma 4's 512-wide full-attention heads.
#define LOKAHI_ATTN_ALL_DPL(G) LOKAHI_ATTN(G, 2) LOKAHI_ATTN(G, 4) LOKAHI_ATTN(G, 8) LOKAHI_ATTN(G, 16)
LOKAHI_ATTN_ALL_DPL(1)
LOKAHI_ATTN_ALL_DPL(2)
LOKAHI_ATTN_ALL_DPL(3)
LOKAHI_ATTN_ALL_DPL(4)
LOKAHI_ATTN_ALL_DPL(5)
LOKAHI_ATTN_ALL_DPL(6)
LOKAHI_ATTN_ALL_DPL(7)
LOKAHI_ATTN_ALL_DPL(8)

struct ReduceParams {
  int heads;
  int head_dim;
  int n_blocks;
  int has_sinks;  // one more logit per head that attends to nothing (GPT-OSS)
};

// Grid: threadgroups (heads, n_tokens), head_dim threads each.
kernel void attn_reduce(const device float* part_o [[buffer(0)]],
                        const device float* part_ml [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant ReduceParams& p [[buffer(3)]],
                        const device float* sinks [[buffer(4)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint d = tid2.x;
  const int head = int(tgp.x);
  const int t = int(tgp.y);
  const int base = (t * p.heads + head) * p.n_blocks;
  float m_max = p.has_sinks != 0 ? sinks[head] : -INFINITY;
  for (int b = 0; b < p.n_blocks; ++b) {
    if (part_ml[2 * (base + b) + 1] > 0.0f) m_max = max(m_max, part_ml[2 * (base + b)]);
  }
  float numer = 0.0f;
  float denom = p.has_sinks != 0 ? exp(sinks[head] - m_max) : 0.0f;
  for (int b = 0; b < p.n_blocks; ++b) {
    const float l = part_ml[2 * (base + b) + 1];
    if (l == 0.0f) continue;
    const float c = exp(part_ml[2 * (base + b)] - m_max);
    numer += c * part_o[(base + b) * p.head_dim + int(d)];
    denom += c * l;
  }
  out[(t * p.heads + head) * p.head_dim + int(d)] = numer / denom;
}

// ---------------------------------------------------------------------------
// Greedy sampling: argmax over the vocabulary (ties resolve to the lowest id).
// One threadgroup.
// ---------------------------------------------------------------------------

struct ArgmaxParams {
  int vocab;
  int history_index;
};

kernel void argmax_vocab(const device float* logits [[buffer(0)]],
                         device int* token [[buffer(1)]],
                         device int* history [[buffer(2)]],
                         constant ArgmaxParams& p [[buffer(3)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float best_v[32];
  threadgroup int best_i[32];
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  for (int i = int(tid); i < p.vocab; i += int(tpg)) {
    const float v = logits[i];
    if (v > bv) {
      bv = v;
      bi = i;
    }
  }
  for (ushort offset = 16; offset > 0; offset /= 2) {
    const float ov = simd_shuffle_down(bv, offset);
    const int oi = simd_shuffle_down(bi, offset);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if (lane == 0) {
    best_v[sg] = bv;
    best_i[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint groups = (tpg + 31) / 32;
    for (uint s = 1; s < groups; ++s) {
      if (best_v[s] > bv || (best_v[s] == bv && best_i[s] < bi)) {
        bv = best_v[s];
        bi = best_i[s];
      }
    }
    token[0] = bi;
    history[p.history_index] = bi;
  }
}

// ---------------------------------------------------------------------------
// Seeded sampling (sampling.h has the semantics and the host reference).
// Filters are thresholds on the order-preserving float key: min-p directly,
// top-k and top-p as the smallest key whose count or mass above it is under
// the limit, found by bisection. The draw is Gumbel-max with Philox4x32-10
// noise keyed by (seed, step, token id), the same noise the CPU uses.
// One threadgroup whose size is a multiple of 32.
// ---------------------------------------------------------------------------

struct SampleParams {
  int vocab;
  int history_index;  // also the step that keys the noise
  float inv_temperature;
  float top_p;
  int top_k;
  float log_min_p;
  uint seed_lo;
  uint seed_hi;
};

constant uint kSamplingStream = 0x6C6F6B61u;

inline uint order_key(float value) {
  const uint bits = as_type<uint>(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline uint4 philox4x32_10(uint4 c, uint2 k) {
  for (int round = 0; round < 10; ++round) {
    if (round > 0) k += uint2(0x9E3779B9u, 0xBB67AE85u);
    const uint hi0 = mulhi(0xD2511F53u, c.x);
    const uint lo0 = 0xD2511F53u * c.x;
    const uint hi1 = mulhi(0xCD9E8D57u, c.z);
    const uint lo1 = 0xCD9E8D57u * c.z;
    c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
  }
  return c;
}

// Threadgroup-wide reductions; every thread receives the result.
inline float tg_max(float v, threadgroup float* scratch, uint sg, uint lane, uint groups) {
  v = simd_max(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_max(lane < groups ? scratch[lane] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

inline float tg_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint groups) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_sum(lane < groups ? scratch[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

inline uint tg_count(uint v, threadgroup uint* scratch, uint sg, uint lane, uint groups) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_sum(lane < groups ? scratch[lane] : 0u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

kernel void sample_vocab(const device float* logits [[buffer(0)]],
                         device int* token [[buffer(1)]],
                         device int* history [[buffer(2)]],
                         constant SampleParams& p [[buffer(3)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch_f[32];
  threadgroup uint scratch_u[32];
  threadgroup int scratch_i[32];
  const uint groups = tpg / 32;

  float local_max = -INFINITY;
  for (int i = int(tid); i < p.vocab; i += int(tpg)) local_max = max(local_max, logits[i]);
  const float m = tg_max(local_max, scratch_f, sg, lane, groups);
  const uint top = order_key(m);

  // Each search finds the smallest key in [threshold, top] meeting its limit,
  // which equals max(threshold, that filter's own threshold).
  uint threshold = 0;
  if (p.log_min_p > -INFINITY) threshold = max(threshold, order_key(m + p.log_min_p));
  if (p.top_k > 0 && p.top_k < p.vocab) {
    uint lo = threshold, hi = top;
    while (lo < hi) {
      const uint mid = lo + (hi - lo) / 2;
      uint count = 0;
      for (int i = int(tid); i < p.vocab; i += int(tpg)) count += order_key(logits[i]) > mid ? 1u : 0u;
      if (tg_count(count, scratch_u, sg, lane, groups) < uint(p.top_k)) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    threshold = lo;
  }
  if (p.top_p < 1.0f) {
    float z = 0.0f;
    for (int i = int(tid); i < p.vocab; i += int(tpg)) z += precise::exp(logits[i] - m);
    const float target = p.top_p * tg_sum(z, scratch_f, sg, lane, groups);
    uint lo = threshold, hi = top;
    while (lo < hi) {
      const uint mid = lo + (hi - lo) / 2;
      float mass = 0.0f;
      for (int i = int(tid); i < p.vocab; i += int(tpg)) {
        const float l = logits[i];
        if (order_key(l) > mid) mass += precise::exp(l - m);
      }
      if (tg_sum(mass, scratch_f, sg, lane, groups) < target) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    threshold = lo;
  }

  // Gumbel-max over kept tokens; one Philox call covers four token ids.
  const uint2 key = uint2(p.seed_lo, p.seed_hi);
  float best = -INFINITY;
  int best_i = 0x7FFFFFFF;
  for (int base = int(tid) * 4; base < p.vocab; base += int(tpg) * 4) {
    uint4 bits = uint4(0u);
    bool drawn = false;
    for (int j = 0; j < 4 && base + j < p.vocab; ++j) {
      const float l = logits[base + j];
      if (order_key(l) < threshold) continue;
      if (!drawn) {
        bits = philox4x32_10(uint4(uint(base >> 2), uint(p.history_index), kSamplingStream, 0u), key);
        drawn = true;
      }
      const float u = (float(bits[j] >> 9) + 0.5f) * 0x1p-23f;
      const float score = l * p.inv_temperature - precise::log(-precise::log(u));
      if (best_i == 0x7FFFFFFF || score > best) {
        best = score;
        best_i = base + j;
      }
    }
  }
  for (ushort offset = 16; offset > 0; offset /= 2) {
    const float ov = simd_shuffle_down(best, offset);
    const int oi = simd_shuffle_down(best_i, offset);
    if (ov > best || (ov == best && oi < best_i)) {
      best = ov;
      best_i = oi;
    }
  }
  if (lane == 0) {
    scratch_f[sg] = best;
    scratch_i[sg] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (uint s = 1; s < groups; ++s) {
      if (scratch_f[s] > best || (scratch_f[s] == best && scratch_i[s] < best_i)) {
        best = scratch_f[s];
        best_i = scratch_i[s];
      }
    }
    token[0] = best_i;
    history[p.history_index] = best_i;
  }
}

// ---------------------------------------------------------------------------
// Vision encoder (SigLIP and Gemma 3's projector): dense matrix products,
// LayerNorm, GELU, row softmax and average pooling. Float32 activations;
// weights F32, F16 or BF16.
// ---------------------------------------------------------------------------

// C[m, n] = alpha * A[m, k] B' (+ bias[j]) (+ C[i, j]), where B' is B^T for
// a B stored [n, k] (TRANS_B) and B for one stored [k, n]. Leading
// dimensions are in elements.
struct MmParams {
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
  float alpha;
  int has_bias;
  int accumulate;
};

struct Bf16 {
  ushort bits;
};
inline float load_value(const device float* p, int i) { return p[i]; }
inline float load_value(const device half* p, int i) { return float(p[i]); }
inline float load_value(const device Bf16* p, int i) { return as_type<float>(uint(p[i].bits) << 16); }

// The qmm tiling (32x32 tiles, four simdgroups of 16x16) with bounds checks
// on every edge, so m, n and k are arbitrary.
template <typename TB, bool TRANS_B>
[[kernel]] void dense_mm(const device float* a [[buffer(0)]],
                         const device TB* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant MmParams& p [[buffer(3)]],
                         const device float* bias [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float as[QMM_BM][QMM_BK + QMM_PAD];
  threadgroup float bs[QMM_BN][QMM_BK + QMM_PAD];
  threadgroup float out_tile[QMM_BM][QMM_BN + QMM_PAD];

  const int m0 = int(tg.y) * QMM_BM;
  const int n0 = int(tg.x) * QMM_BN;
  const int sm = int(sg / 2) * 16;
  const int sn = int(sg % 2) * 16;

  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (int k0 = 0; k0 < p.k; k0 += QMM_BK) {
    for (int i = int(tid); i < QMM_BM * QMM_BK; i += 128) {
      const int r = i / QMM_BK;
      const int kk = i % QMM_BK;
      const int row = m0 + r;
      const int col = k0 + kk;
      as[r][kk] = row < p.m && col < p.k ? a[row * p.lda + col] : 0.0f;
    }
    for (int i = int(tid); i < QMM_BN * QMM_BK; i += 128) {
      // Consecutive threads read consecutive addresses in either layout.
      const int r = TRANS_B ? i / QMM_BK : i % QMM_BN;
      const int kk = TRANS_B ? i % QMM_BK : i / QMM_BN;
      const int row = n0 + r;
      const int col = k0 + kk;
      bs[r][kk] = row < p.n && col < p.k ? load_value(b, TRANS_B ? row * p.ldb + col : col * p.ldb + row) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < QMM_BK; kk += 8) {
      simdgroup_float8x8 x[2];
      simdgroup_float8x8 y[2];
      for (int i = 0; i < 2; ++i) simdgroup_load(x[i], &as[sm + 8 * i][kk], QMM_BK + QMM_PAD);
      for (int j = 0; j < 2; ++j)
        simdgroup_load(y[j], &bs[sn + 8 * j][kk], QMM_BK + QMM_PAD, ulong2(0, 0), true);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], x[i], y[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], &out_tile[sm + 8 * i][sn + 8 * j], QMM_BN + QMM_PAD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < QMM_BM * QMM_BN; i += 128) {
    const int r = i / QMM_BN;
    const int col = i % QMM_BN;
    const int row = m0 + r;
    const int j = n0 + col;
    if (row >= p.m || j >= p.n) continue;
    float value = p.alpha * out_tile[r][col];
    if (p.has_bias != 0) value += bias[j];
    device float* out = c + row * p.ldc + j;
    *out = p.accumulate != 0 ? value + *out : value;
  }
}

#define LOKAHI_MM(NAME, TB, TRANS_B) \
  template LOKAHI_INSTANCE_ATTRS(NAME) decltype(dense_mm<TB, TRANS_B>) dense_mm<TB, TRANS_B>;
LOKAHI_MM("dense_mm_f32_nt", float, true)
LOKAHI_MM("dense_mm_f16_nt", half, true)
LOKAHI_MM("dense_mm_bf16_nt", Bf16, true)
LOKAHI_MM("dense_mm_f32_nn", float, false)
LOKAHI_MM("dense_mm_f16_nn", half, false)
LOKAHI_MM("dense_mm_bf16_nn", Bf16, false)

// Threadgroup-wide maximum. `scratch` holds one float per simdgroup.
inline float threadgroup_max(float value, threadgroup float* scratch, uint sg, uint lane, uint simdgroups) {
  value = simd_max(value);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = -INFINITY;
  for (uint i = 0; i < simdgroups; ++i) total = max(total, scratch[i]);
  return total;
}

struct LayerNormParams {
  int cols;
  float eps;
};

// out = (x - mean) / sqrt(var + eps) * weight + bias. One threadgroup per row.
kernel void layer_norm_rows(const device float* in [[buffer(0)]],
                            const device float* weight [[buffer(1)]],
                            const device float* bias [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant LayerNormParams& p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tpg [[threads_per_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const device float* x = in + row * p.cols;
  const uint simdgroups = (tpg + 31) / 32;
  float sum = 0.0f;
  for (int i = int(tid); i < p.cols; i += int(tpg)) sum += x[i];
  const float mean = threadgroup_sum(sum, scratch, sg, lane, simdgroups) / float(p.cols);
  float squares = 0.0f;
  for (int i = int(tid); i < p.cols; i += int(tpg)) squares += (x[i] - mean) * (x[i] - mean);
  const float inv = precise::rsqrt(threadgroup_sum(squares, scratch, sg, lane, simdgroups) / float(p.cols) + p.eps);
  device float* y = out + row * p.cols;
  for (int i = int(tid); i < p.cols; i += int(tpg)) y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

// Rows of s become softmax(s). One threadgroup per row.
kernel void softmax_rows(device float* s [[buffer(0)]],
                         constant ElementwiseParams& p [[buffer(1)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device float* x = s + row * p.count;
  const uint simdgroups = (tpg + 31) / 32;
  float m = -INFINITY;
  for (int i = int(tid); i < p.count; i += int(tpg)) m = max(m, x[i]);
  m = threadgroup_max(m, scratch, sg, lane, simdgroups);
  float sum = 0.0f;
  for (int i = int(tid); i < p.count; i += int(tpg)) {
    const float e = precise::exp(x[i] - m);
    x[i] = e;
    sum += e;
  }
  const float inv = 1.0f / threadgroup_sum(sum, scratch, sg, lane, simdgroups);
  for (int i = int(tid); i < p.count; i += int(tpg)) x[i] *= inv;
}

kernel void gelu_inplace(device float* x [[buffer(0)]],
                         constant ElementwiseParams& p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) x[gid] = gelu_tanh(x[gid]);
}

kernel void copy_floats(const device float* src [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant ElementwiseParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) dst[gid] = src[gid];
}

struct PoolParams {
  int cols;
  int side;         // patches per side
  int tokens_side;  // pooled tokens per side
};

// The patch grid [side * side, cols] averaged over kernel x kernel squares
// into [tokens_side^2, cols], both row-major.
kernel void avg_pool_tokens(const device float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant PoolParams& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int token = int(gid.y);
  if (c >= p.cols || token >= p.tokens_side * p.tokens_side) return;
  const int kernel_size = p.side / p.tokens_side;
  const int ty = token / p.tokens_side;
  const int tx = token % p.tokens_side;
  float sum = 0.0f;
  for (int ky = 0; ky < kernel_size; ++ky)
    for (int kx = 0; kx < kernel_size; ++kx)
      sum += in[((ty * kernel_size + ky) * p.side + tx * kernel_size + kx) * p.cols + c];
  out[token * p.cols + c] = sum / float(kernel_size * kernel_size);
}

// ---------------------------------------------------------------------------
// Gemma 4's vision tower: axial 2-D RoPE and pooling with standardization.
// ---------------------------------------------------------------------------

struct AxialRopeParams {
  int heads;
  int dim;    // head size; each axis rotates dim / 2 channels
  int freqs;  // dim / 4 frequencies per axis
};

// In place on rows [tokens, heads, dim]: the first half of each head rotates
// by its token's x, the second by its y, each rotate-half within its half.
// Tables hold cos and sin [position, freqs], computed on the host in double.
// Grid: (2 * freqs, heads, tokens).
kernel void rope_axial(device float* x [[buffer(0)]],
                       const device int* xy [[buffer(1)]],
                       const device float* cos_table [[buffer(2)]],
                       const device float* sin_table [[buffer(3)]],
                       constant AxialRopeParams& p [[buffer(4)]],
                       uint3 gid [[thread_position_in_grid]]) {
  const int j = int(gid.x);
  const int head = int(gid.y);
  const int t = int(gid.z);
  if (j >= 2 * p.freqs || head >= p.heads) return;
  const int axis = j / p.freqs;
  const int i = j % p.freqs;
  const int pos = xy[2 * t + axis];
  device float* part = x + (t * p.heads + head) * p.dim + axis * (p.dim / 2);
  const float c = cos_table[pos * p.freqs + i];
  const float s = sin_table[pos * p.freqs + i];
  const float a = part[i];
  const float b = part[i + p.freqs];
  part[i] = a * c - b * s;
  part[i + p.freqs] = b * c + a * s;
}

struct PoolStandardizeParams {
  int cols;
  int patch_cols;   // patches per row
  int pooled_cols;  // pooled tokens per row
  int kernel_size;
  int tokens;
  float scale;      // sqrt(hidden)
  int standardize;
};

// out[token] = mean of the token's kernel x kernel patches, times scale, then
// (v - bias) * mul when standardizing. Grid: (cols, tokens).
kernel void pool_standardize(const device float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             const device float* bias [[buffer(2)]],
                             const device float* mul [[buffer(3)]],
                             constant PoolStandardizeParams& p [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int token = int(gid.y);
  if (c >= p.cols || token >= p.tokens) return;
  const int ty = token / p.pooled_cols;
  const int tx = token % p.pooled_cols;
  const int k = p.kernel_size;
  float sum = 0.0f;
  for (int ky = 0; ky < k; ++ky)
    for (int kx = 0; kx < k; ++kx) sum += in[((ty * k + ky) * p.patch_cols + tx * k + kx) * p.cols + c];
  float v = sum / float(k * k) * p.scale;
  if (p.standardize != 0) v = (v - bias[c]) * mul[c];
  out[token * p.cols + c] = v;
}
