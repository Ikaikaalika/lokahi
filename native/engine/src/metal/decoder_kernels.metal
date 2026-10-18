// Lokahi Metal kernels for decoders (Gemma 3, Gemma 4, the Llama family and
// DeepSeek).
//
// Activations are float32. Linear weights are MLX affine-quantized codes (2,
// 3, 4, 5, 6 or 8 bits) with BF16 or F16 per-group scales and biases, or (in
// the matrix-vector kernels) MXFP4: E2M1 codes with one E8M0 exponent per 32
// values. MLX packs each row as one little-endian bit stream, so 3-, 5- and
// 6-bit codes straddle words. Quantization parameters are function constants
// so each pipeline is specialized at load time.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// The Linux syntax check (tests/syntax_stubs) parses this file as C++, which
// does not accept attributes on explicit instantiations; it blanks this macro.
#ifndef LOKAHI_INSTANCE_ATTRS
#define LOKAHI_INSTANCE_ATTRS(name) [[host_name(name)]] [[kernel]]
#endif

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

// The two elements a lane holds of a float simdgroup_matrix<float, 8, 8>:
// row (lane / 4 & 4) + (lane / 2 % 4), columns (lane / 4 & 2) * 2 +
// (lane % 2) * 2 and the next. MSL declares the whole matrix's storage, so
// the lane's pair is reached through a cast. The layout is checked at load
// (simdgroup_layout_probe); MLX's kernels rely on it too.
#define FRAG_ELEMS(m) reinterpret_cast<thread float2&>((m).thread_elements())

// Whether a QBITS-bit code never straddles words (2, 4 and 8 bits).
inline bool whole_codes() { return 32 % QBITS == 0; }

// The codes from `units` little-endian `unit`-bit integers (bytes or
// halfwords) as one bit stream: count codes of QBITS bits, in order.
template <typename Unit, int Count>
inline void stream_codes(const device Unit* units, thread uint* codes) {
  constexpr int kUnitBits = 8 * int(sizeof(Unit));
  uint buffer = 0;
  int have = 0;
  int next = 0;
  for (int j = 0; j < Count; ++j) {
    if (have < QBITS) {
      buffer |= uint(units[next++]) << have;
      have += kUnitBits;
    }
    codes[j] = buffer & ((1u << QBITS) - 1u);
    buffer >>= QBITS;
    have -= QBITS;
  }
}

// Code c of a row of packed codes.
inline uint code_at(const device uint* row, int c) {
  const int bit = c * QBITS;
  uint value = row[bit >> 5] >> (bit & 31);
  if ((bit & 31) + QBITS > 32) value |= row[(bit >> 5) + 1] << (32 - (bit & 31));
  return value & ((1u << QBITS) - 1u);
}

// Dot product of a row's 16 codes from k0 (a multiple of 16) with x.
inline float dot16(const device uint* row, int k0, thread const float* x) {
  float acc = 0.0f;
  const device uint* w = row + k0 * QBITS / 32;
  if (!whole_codes()) {  // 16 codes fill QBITS halfwords
    uint codes[16];
    stream_codes<ushort, 16>(reinterpret_cast<const device ushort*>(row) + k0 * QBITS / 16, codes);
    for (int j = 0; j < 16; ++j) acc += float(codes[j]) * x[j];
  } else if (QBITS == 4) {
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
  const int words_per_row = p.cols * QBITS / 32;
  const int groups_per_row = p.cols / QGROUP;
  const int token = tokens[t];
  const uint code = code_at(w + token * words_per_row, c);
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

// x[t, :] += rows[index[t], :] wherever index[t] >= 0: a layer's deepstack
// rows (Qwen3-VL) onto the hidden states of the image tokens.
kernel void add_rows_at(const device float* rows [[buffer(0)]],
                        const device int* index [[buffer(1)]],
                        device float* x [[buffer(2)]],
                        constant ReplaceParams& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int t = int(gid.y);
  if (c >= p.cols || t >= p.n) return;
  const int r = index[t];
  if (r >= 0) x[t * p.cols + c] += rows[r * p.cols + c];
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

// Multi-token prediction: the head's input rows [norm_e(e) | norm_h(h)],
// out [rows, 2 * cols], from embedding rows e and the hidden states h before
// them. One threadgroup per row.
kernel void mtp_concat_norms(const device float* e [[buffer(0)]],
                             const device float* h [[buffer(1)]],
                             const device float* e_weight [[buffer(2)]],
                             const device float* h_weight [[buffer(3)]],
                             device float* out [[buffer(4)]],
                             constant NormParams& p [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const device float* x = e + row * p.cols;
  const device float* y = h + row * p.cols;
  float sx = 0.0f, sy = 0.0f;
  for (int i = int(tid); i < p.cols; i += int(tpg)) {
    sx += x[i] * x[i];
    sy += y[i] * y[i];
  }
  const uint simdgroups = (tpg + 31) / 32;
  const float ix = precise::rsqrt(threadgroup_sum(sx, scratch, sg, lane, simdgroups) / float(p.cols) + p.eps);
  const float iy = precise::rsqrt(threadgroup_sum(sy, scratch, sg, lane, simdgroups) / float(p.cols) + p.eps);
  device float* o = out + row * 2 * p.cols;
  for (int i = int(tid); i < p.cols; i += int(tpg)) {
    o[i] = x[i] * ix * (p.offset + e_weight[i]);
    o[p.cols + i] = y[i] * iy * (p.offset + h_weight[i]);
  }
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
// 4-bit affine weights (most checkpoints) take a faster layout instead: each
// half of the simdgroup owns two of the rows and strides them 16 codes per
// lane, so narrow matrices (Gemma 3 1B's 1152 columns, 2.25 steps of 512)
// keep every lane busy, and inputs are pre-divided by 16^j as in MLX so a
// code needs a mask rather than a shift. Callers still reduce each row with
// simd_sum: the other half's lanes contribute zero to it.
// Requires cols % 16 == 0 and QGROUP % 16 == 0.
// ---------------------------------------------------------------------------

constant constexpr int QMV_ROWS = 4;
constant constexpr int QMV_SIMDGROUPS = 2;

struct QmvParams {
  int rows;
  int cols;
};

// The 4-bit layout above: this lane's two rows, both returned in a[0..1].
inline void qmv_accumulate4(const device uint* w, const device ushort* scales, const device ushort* biases,
                            const device float* x, int row, int rows, int cols, int sub, thread float* a) {
  const int words_per_row = cols / 8;
  const int groups_per_row = cols / QGROUP;
  const int r0 = min(row, rows - 1), r1 = min(row + 1, rows - 1);  // a clamped row's sum is discarded
  for (int k0 = sub * 16; k0 < cols; k0 += 16 * 16) {
    float xv[16];
    float xsum = 0.0f;
    const device float4* x4 = reinterpret_cast<const device float4*>(x + k0);
    for (int j = 0; j < 4; ++j) {
      const float4 v = x4[j];
      xsum += v.x + v.y + v.z + v.w;
      xv[4 * j] = v.x;
      xv[4 * j + 1] = v.y * (1.0f / 16.0f);
      xv[4 * j + 2] = v.z * (1.0f / 256.0f);
      xv[4 * j + 3] = v.w * (1.0f / 4096.0f);
    }
    const int g = k0 / QGROUP;
    for (int rr = 0; rr < 2; ++rr) {
      const int r = rr == 0 ? r0 : r1;
      const device ushort* wr = reinterpret_cast<const device ushort*>(w + r * words_per_row + k0 / 8);
      float dot = 0.0f;
      for (int h = 0; h < 4; ++h) {
        const uint ww = wr[h];
        dot += xv[4 * h] * float(ww & 0x000Fu) + xv[4 * h + 1] * float(ww & 0x00F0u) +
               xv[4 * h + 2] * float(ww & 0x0F00u) + xv[4 * h + 3] * float(ww & 0xF000u);
      }
      const int gi = r * groups_per_row + g;
      a[rr] += scale_at(scales, gi) * dot + scale_at(biases, gi) * xsum;
    }
  }
}

inline void qmv_accumulate(const device uint* w, const device ushort* scales,
                           const device ushort* biases, const device float* x, int row0,
                           int rows, int cols, uint lane, thread float* acc) {
  if (QBITS == 4 && !QMXFP4) {
    const int half_ = int(lane) / 16;
    float a[2] = {0.0f, 0.0f};
    qmv_accumulate4(w, scales, biases, x, row0 + 2 * half_, rows, cols, int(lane) % 16, a);
    acc[0] += half_ == 0 ? a[0] : 0.0f;
    acc[1] += half_ == 0 ? a[1] : 0.0f;
    acc[2] += half_ == 1 ? a[0] : 0.0f;
    acc[3] += half_ == 1 ? a[1] : 0.0f;
    return;
  }
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
        const int gi = row * groups_per_row + g;
        acc[r] += scale_at(scales, gi) * dot16(w + row * words_per_row, k0, xv) + scale_at(biases, gi) * xsum;
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

// Quantized products for short passes of 2 to QMV_N_MAX tokens (speculative
// verification, batched decode): Y[n, rows] = X[n, cols] W^T with qmv's row
// and lane layout. Each lane unpacks its rows' 16 codes once per step and
// reads each token's 16 inputs once, applying them to all of its rows, so
// the weights are read once per pass rather than once per token and a pass
// costs a fraction of n decode steps instead of a whole qmm tile. Each
// token's sums take qmv's form. Not for MXFP4 (qmv's own path).
// ---------------------------------------------------------------------------

constant constexpr int QMV_N_MAX = 8;

struct QmvNParams {
  int rows;
  int cols;
  int n;
};

// The 16 codes of a row from code k0 (a multiple of 16), as floats.
inline void unpack16(const device uint* row, int k0, thread float* out) {
  if (!whole_codes()) {  // 16 codes fill QBITS halfwords
    uint codes[16];
    stream_codes<ushort, 16>(reinterpret_cast<const device ushort*>(row) + k0 * QBITS / 16, codes);
    for (int j = 0; j < 16; ++j) out[j] = float(codes[j]);
  } else if (QBITS == 4) {
    const uint2 ww = *reinterpret_cast<const device uint2*>(row + k0 / 8);
    for (int j = 0; j < 8; ++j) out[j] = float((ww.x >> (4 * j)) & 0xFu);
    for (int j = 0; j < 8; ++j) out[8 + j] = float((ww.y >> (4 * j)) & 0xFu);
  } else if (QBITS == 8) {
    const uint4 ww = *reinterpret_cast<const device uint4*>(row + k0 / 4);
    for (int j = 0; j < 4; ++j) out[j] = float((ww.x >> (8 * j)) & 0xFFu);
    for (int j = 0; j < 4; ++j) out[4 + j] = float((ww.y >> (8 * j)) & 0xFFu);
    for (int j = 0; j < 4; ++j) out[8 + j] = float((ww.z >> (8 * j)) & 0xFFu);
    for (int j = 0; j < 4; ++j) out[12 + j] = float((ww.w >> (8 * j)) & 0xFFu);
  } else {
    const uint ww = row[k0 / 16];
    for (int j = 0; j < 16; ++j) out[j] = float((ww >> (2 * j)) & 0x3u);
  }
}

// Fixed-trip loops over the accumulators unroll fully; a loop the compiler
// keeps would index them dynamically and spill them.
#define QMV_N_UNROLL _Pragma("clang loop unroll(full)")

kernel void qmv_n(const device uint* w [[buffer(0)]],
                  const device ushort* scales [[buffer(1)]],
                  const device ushort* biases [[buffer(2)]],
                  const device float* x [[buffer(3)]],
                  device float* y [[buffer(4)]],
                  constant QmvNParams& p [[buffer(5)]],
                  uint tg [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  const int row0 = (int(tg) * QMV_SIMDGROUPS + int(sg)) * QMV_ROWS;
  if (row0 >= p.rows) return;
  const int words_per_row = p.cols * QBITS / 32;
  const int groups_per_row = p.cols / QGROUP;
  float acc[QMV_N_MAX][QMV_ROWS];
  QMV_N_UNROLL for (int t = 0; t < QMV_N_MAX; ++t)
    QMV_N_UNROLL for (int r = 0; r < QMV_ROWS; ++r) acc[t][r] = 0.0f;
  for (int k0 = int(lane) * 16; k0 < p.cols; k0 += 32 * 16) {
    const int g = k0 / QGROUP;
    // Unpack the rows' codes once, then read each token's 16 inputs once
    // and apply them to every row.
    float codes[QMV_ROWS][16];
    float s[QMV_ROWS], b[QMV_ROWS];
    QMV_N_UNROLL for (int r = 0; r < QMV_ROWS; ++r) {
      const int row = min(row0 + r, p.rows - 1);  // a clamped row's sums are never stored
      unpack16(w + row * words_per_row, k0, codes[r]);
      const int gi = row * groups_per_row + g;
      s[r] = scale_at(scales, gi);
      b[r] = scale_at(biases, gi);
    }
    QMV_N_UNROLL for (int t = 0; t < QMV_N_MAX; ++t) {
      if (t < p.n) {  // Uniform across the simdgroup.
        const device float4* x4 = reinterpret_cast<const device float4*>(x + t * p.cols + k0);
        float xv[16];
        QMV_N_UNROLL for (int j = 0; j < 4; ++j) {
          const float4 v = x4[j];
          xv[4 * j] = v.x;
          xv[4 * j + 1] = v.y;
          xv[4 * j + 2] = v.z;
          xv[4 * j + 3] = v.w;
        }
        float xsum = 0.0f;
        QMV_N_UNROLL for (int j = 0; j < 16; ++j) xsum += xv[j];
        QMV_N_UNROLL for (int r = 0; r < QMV_ROWS; ++r) {
          float dot = 0.0f;
          QMV_N_UNROLL for (int j = 0; j < 16; ++j) dot += codes[r][j] * xv[j];
          acc[t][r] += s[r] * dot + b[r] * xsum;
        }
      }
    }
  }
  QMV_N_UNROLL for (int t = 0; t < QMV_N_MAX; ++t) {
    if (t < p.n) {
      QMV_N_UNROLL for (int r = 0; r < QMV_ROWS; ++r) {
        const float total = simd_sum(acc[t][r]);
        if (lane == 0 && row0 + r < p.rows) y[t * p.rows + row0 + r] = total;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The same short passes for 4-bit weights on simdgroup matrices: each weight
// is dequantized once, straight into the 8x8 fragments' lanes, and one
// multiply-accumulate applies it to all (up to 8) tokens. Y^T = W X^T: an
// accumulator is 8 weight rows by 8 tokens; A is 8 rows by 8 codes; B is
// those 8 codes by the tokens. A lane holds two adjacent elements of one
// fragment row (FRAG_ELEMS), and a sum over k may visit k in any order as
// long as A and B agree, so each lane fills its A elements from one 32-bit
// word: in k-block j of a 32-code step, lane (row fm, column fn) takes
// nibbles j and j + 4 of word fn / 2 of its row, which puts code
// k0 + 8 (c / 2) + 4 (c % 2) + j in column c, and B's row r holds the same
// code, so a lane's four k-blocks of x are one float4 per token. A nibble is
// masked in place (no shift) and its 16^p factor folded into the scale; a
// 32-code step lies in one group (groups of 32, 64 or 128). The
// threadgroup's simdgroups split K and add their partial sums at the end.
// Rows of x past p.n read as zero.
//
// Measured on M1 (rows x cols from 2048 x 2048 to 9728 x 2560): 8 tokens cost
// 1.4-1.7x one token's qmv (qmv_n: 2.7-3.5x), 4 tokens 1.4-1.6x (1.8-2.4x);
// at 2 tokens qmv_n is as fast or faster. Two 8-row blocks per simdgroup and
// eight simdgroups beat 1-4 blocks and 2-4 simdgroups; uint4 loads per lane
// and one accumulator per k-block did not help.
// ---------------------------------------------------------------------------

constant constexpr int QMV_MMA_RB = 2;  // 8-row blocks per simdgroup
constant constexpr int QMV_MMA_SG = 8;  // simdgroups splitting K
constant constexpr int QMV_MMA_ROWS = 8 * QMV_MMA_RB;

kernel void qmv_mma(const device uint* w [[buffer(0)]],
                    const device ushort* scales [[buffer(1)]],
                    const device ushort* biases [[buffer(2)]],
                    const device float* x [[buffer(3)]],
                    device float* y [[buffer(4)]],
                    constant QmvNParams& p [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[QMV_MMA_SG * QMV_MMA_RB * 64];
  const int fm = int((lane / 4) & 4) + int((lane / 2) % 4);
  const int fn = int((lane / 4) & 2) * 2 + int(lane % 2) * 2;
  const int row_base = int(tg) * QMV_MMA_ROWS;
  const int words_per_row = p.cols / 8;
  const int groups_per_row = p.cols / QGROUP;
  const int steps = p.cols / 32;
  const int per = (steps + QMV_MMA_SG - 1) / QMV_MMA_SG;
  const int s0 = int(sg) * per;
  const int s1 = min(steps, s0 + per);
  int row[QMV_MMA_RB];
  QMV_N_UNROLL for (int rb = 0; rb < QMV_MMA_RB; ++rb) row[rb] = min(row_base + 8 * rb + fm, p.rows - 1);
  // This lane's B elements: tokens fn and fn + 1 at codes x_at ... x_at + 3
  // of each step (k-blocks 0 to 3).
  const int x_at = 8 * (fm / 2) + 4 * (fm % 2);
  const bool t0 = fn < p.n, t1 = fn + 1 < p.n;
  const device float* x0 = x + (t0 ? fn : 0) * p.cols + x_at;
  const device float* x1 = x + (t1 ? fn + 1 : 0) * p.cols + x_at;
  simdgroup_float8x8 acc[QMV_MMA_RB];
  QMV_N_UNROLL for (int rb = 0; rb < QMV_MMA_RB; ++rb) acc[rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int st = s0; st < s1; ++st) {
    const int k0 = st * 32;
    const float4 xa = t0 ? *reinterpret_cast<const device float4*>(x0 + k0) : float4(0.0f);
    const float4 xb = t1 ? *reinterpret_cast<const device float4*>(x1 + k0) : float4(0.0f);
    simdgroup_float8x8 b[4];
    QMV_N_UNROLL for (int j = 0; j < 4; ++j) FRAG_ELEMS(b[j]) = float2(xa[j], xb[j]);
    // This lane's word: codes k0 + 8 (fn / 2) ... + 7, all in one group.
    const int word = k0 / 8 + fn / 2;
    const int group = (k0 + 8 * (fn / 2)) / QGROUP;
    QMV_N_UNROLL for (int rb = 0; rb < QMV_MMA_RB; ++rb) {
      const uint code = w[row[rb] * words_per_row + word];
      const int gi = row[rb] * groups_per_row + group;
      const float s = scale_at(scales, gi);
      const float bias = scale_at(biases, gi);
      QMV_N_UNROLL for (int j = 0; j < 4; ++j) {
        // Nibbles j and j + 4, each times 16^nibble, scaled back by s / 16^nibble.
        const float lo = float(code & (0xFu << (4 * j))) * (s * (1.0f / float(1u << (4 * j))));
        const float hi = float(code & (0xFu << (4 * j + 16))) * (s * (1.0f / float(1u << (4 * j + 16))));
        simdgroup_float8x8 a;
        FRAG_ELEMS(a) = float2(lo + bias, hi + bias);
        simdgroup_multiply_accumulate(acc[rb], a, b[j], acc[rb]);
      }
    }
  }
  // Sum the simdgroups' partial products: rows by tokens, 8 x 8 per block.
  QMV_N_UNROLL for (int rb = 0; rb < QMV_MMA_RB; ++rb)
    simdgroup_store(acc[rb], partial + (int(sg) * QMV_MMA_RB + rb) * 64, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < QMV_MMA_RB * 64; i += QMV_MMA_SG * 32) {
    const int rb = i / 64, r = (i % 64) / 8, t = i % 8;
    const int out_row = row_base + 8 * rb + r;
    if (t >= p.n || out_row >= p.rows) continue;
    float total = 0.0f;
    QMV_N_UNROLL for (int g = 0; g < QMV_MMA_SG; ++g) total += partial[(g * QMV_MMA_RB + rb) * 64 + r * 8 + t];
    y[t * p.rows + out_row] = total;
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
  int groups;         // > 1: only the topk_groups groups with the largest maxima compete (DeepSeek V2)
  int topk_groups;
  float scaling;      // every weight times this (DeepSeek's routed_scaling_factor)
};

// One simdgroup per token. Ties go to the lower expert index (and drop the
// lower group).
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
  // Experts already chosen or excluded, as a bit set over up to 512 experts.
  uint taken[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  if (p.groups > 1) {  // every lane drops the same groups
    const int per_group = p.experts / p.groups;
    float group_max[32];
    for (int g = 0; g < p.groups; ++g) {
      float best = -INFINITY;
      for (int e = g * per_group; e < (g + 1) * per_group; ++e) best = max(best, l[e]);
      group_max[g] = best;
    }
    uint dropped = 0;
    for (int j = 0; j < p.groups - p.topk_groups; ++j) {
      int worst = -1;
      for (int g = 0; g < p.groups; ++g) {
        if (!((dropped >> g) & 1u) && (worst < 0 || group_max[g] < group_max[worst])) worst = g;
      }
      dropped |= 1u << worst;
      for (int e = worst * per_group; e < (worst + 1) * per_group; ++e) taken[e >> 5] |= 1u << (e & 31);
    }
  }
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
  const float scale = ((p.topk_softmax != 0 || p.norm_topk != 0) ? 1.0f / kept : 1.0f / total) * p.scaling;
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
  int x_stride;     // floats between input rows; 0 for cols
  int y_stride;     // floats between output rows; 0 for rows
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
  const device float* xr = x + (p.input_rows != 0 ? pair / p.k : pair) * (p.x_stride != 0 ? p.x_stride : p.cols);
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
      y[pair * (p.y_stride != 0 ? p.y_stride : p.rows) + row] = glu(g, u, p.act, p.limit);
    }
  }
}

// y[pair, :] = W_e x[pair or pair / k] with e = chosen[pair]. Multi-head
// latent attention also runs its per-head absorptions and expansions
// through it, with e the pair's head and strided rows.
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
  const device float* xr = x + (p.input_rows != 0 ? pair / p.k : pair) * (p.x_stride != 0 ? p.x_stride : p.cols);
  device float* yr = y + pair * (p.y_stride != 0 ? p.y_stride : p.rows);
  float acc[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  qmv_accumulate(w + e * words, expert_scales(scales, e, groups), biases + e * groups, xr, row0, p.rows, p.cols, lane,
                 acc);
  for (int r = 0; r < QMV_ROWS; ++r) {
    const float total = simd_sum(acc[r]);
    const int row = row0 + r;
    if (lane == 0 && row < p.rows) yr[row] = total + (p.has_bias != 0 ? out_bias[e * p.rows + row] : 0.0f);
  }
}

// Multi-head latent attention's query absorption from kv_b_proj, whose rows
// are quantized along the latent rank: for pair (token, head h),
// y[pair * y_stride + r] = sum over d < nope of x[pair * x_stride + d] *
// W[h * head_rows + d, r]. One threadgroup per pair; each thread owns whole
// words of the latent (one group's scale and bias apply to a word) and walks
// the head's key rows, so neighbouring threads read neighbouring words.
struct AbsorbParams {
  int heads;
  int nope;       // key rows per head
  int head_rows;  // rows per head in kv_b_proj (nope + v)
  int cols;       // the latent rank
  int x_stride;
  int y_stride;
};

kernel void qmv_t_heads(const device uint* w [[buffer(0)]],
                        const device ushort* scales [[buffer(1)]],
                        const device ushort* biases [[buffer(2)]],
                        const device float* x [[buffer(3)]],
                        device float* y [[buffer(4)]],
                        constant AbsorbParams& p [[buffer(5)]],
                        uint pair [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
  // A thread's unit is a word, or 16 codes (QBITS halfwords) when codes
  // straddle words.
  const int per_unit = whole_codes() ? 32 / QBITS : 16;
  const uint mask = (1u << QBITS) - 1u;
  const int words_per_row = p.cols * QBITS / 32;
  const int units = p.cols / per_unit;
  const int groups_per_row = p.cols / QGROUP;
  const int h = int(pair) % p.heads;
  const device float* xr = x + int(pair) * p.x_stride;
  device float* yr = y + int(pair) * p.y_stride;
  for (int unit = int(tid); unit < units; unit += int(tpg)) {
    float acc[16];
    for (int j = 0; j < per_unit; ++j) acc[j] = 0.0f;
    const int g = unit * per_unit / QGROUP;
    for (int d = 0; d < p.nope; ++d) {
      const int row = h * p.head_rows + d;
      uint codes[16];
      if (whole_codes()) {
        const uint word = w[row * words_per_row + unit];
        for (int j = 0; j < per_unit; ++j) codes[j] = (word >> (QBITS * j)) & mask;
      } else {
        stream_codes<ushort, 16>(reinterpret_cast<const device ushort*>(w + row * words_per_row) + unit * QBITS, codes);
      }
      const float scale = scale_at(scales, row * groups_per_row + g);
      const float bias = scale_at(biases, row * groups_per_row + g);
      const float xv = xr[d];
      for (int j = 0; j < per_unit; ++j) acc[j] += xv * (scale * float(codes[j]) + bias);
    }
    for (int j = 0; j < per_unit; ++j) yr[unit * per_unit + j] = acc[j];
  }
}

// GLM-4.5 / DeepSeek V3 (noaux_tc) routing, one simdgroup per token:
// sigmoid scores plus the correction bias choose the top k within the best
// topk_groups groups (by each group's two best; the others' experts score 0,
// or are excluded with `exclude`); the chosen experts' raw scores,
// renormalized when norm_topk (and k > 1), times scaling, weight them. Ties
// go to the lower group or expert index. Lane 0 does the selection: at most
// 512 experts, so it is cheap next to the experts.
struct MoeGroupedParams {
  int experts;
  int k;
  int groups;
  int topk_groups;
  int norm_topk;
  float scaling;
  int exclude;
};

kernel void moe_route_grouped(const device float* logits [[buffer(0)]],
                              device int* chosen [[buffer(1)]],
                              device float* weights [[buffer(2)]],
                              constant MoeGroupedParams& p [[buffer(3)]],
                              const device float* correction [[buffer(4)]],
                              uint t [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float score[512];
  threadgroup float selection[512];
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
    const float dropped_score = p.exclude != 0 ? -INFINITY : 0.0f;
    for (int j = 0; j < p.groups - p.topk_groups; ++j) {  // zero or exclude the lowest groups
      int worst = -1;
      for (int g = 0; g < p.groups; ++g) {
        if (!((dropped >> g) & 1u) && (worst < 0 || group_score[g] < group_score[worst])) worst = g;
      }
      dropped |= 1u << worst;
      for (int e = worst * per_group; e < (worst + 1) * per_group; ++e) selection[e] = dropped_score;
    }
  }
  uint taken[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
// 64x64 output tiles (tokens x rows). Four simdgroups each own a 32x32
// quadrant, a 4x4 grid of 8x8 simdgroup matrices, so every K step of 32 does
// 64 multiply-accumulates per simdgroup between two barriers. X arrives as
// float4 loads; each thread dequantizes 16 consecutive codes of one row, which
// share one scale and bias because QGROUP is a multiple of 32. Whole tiles
// store straight to Y; edge tiles go through threadgroup memory, reusing the
// staging arrays. Requires cols % 32 == 0 and QGROUP % 32 == 0.
// ---------------------------------------------------------------------------

// dense_mm's 32x32 tiles (below); qmm's own tile is QMM_TILE_*.
constant constexpr int QMM_BM = 32;
constant constexpr int QMM_BN = 32;
constant constexpr int QMM_BK = 32;
constant constexpr int QMM_PAD = 4;

// Fixed-trip loops over simdgroup matrices must unroll fully, or the matrices
// are indexed dynamically and spill.
#define QMM_UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int QMM_TILE_M = 64;
constant constexpr int QMM_TILE_N = 64;
constant constexpr int QMM_TILE_K = 32;
constant constexpr int QMM_TILE_LD = QMM_TILE_K + 4;  // staging row stride, padded against bank conflicts
constant constexpr int QMM_TILE_OUT_LD = QMM_TILE_N + 4;

struct QmmParams {
  int n;
  int rows;
  int cols;
};

// 16 dequantized weights of one row from code k0 (a multiple of 16).
inline void dequant16(const device uint* row, int k0, float s, float b, thread float* out) {
  if (!whole_codes()) {  // 16 codes fill QBITS halfwords
    uint codes[16];
    stream_codes<ushort, 16>(reinterpret_cast<const device ushort*>(row) + k0 * QBITS / 16, codes);
    for (int j = 0; j < 16; ++j) out[j] = s * float(codes[j]) + b;
  } else if (QBITS == 4) {
    const uint2 ww = *reinterpret_cast<const device uint2*>(row + k0 / 8);
    for (int j = 0; j < 8; ++j) out[j] = s * float((ww.x >> (4 * j)) & 0xFu) + b;
    for (int j = 0; j < 8; ++j) out[8 + j] = s * float((ww.y >> (4 * j)) & 0xFu) + b;
  } else if (QBITS == 8) {
    const uint4 ww = *reinterpret_cast<const device uint4*>(row + k0 / 4);
    for (int j = 0; j < 4; ++j) out[j] = s * float((ww.x >> (8 * j)) & 0xFFu) + b;
    for (int j = 0; j < 4; ++j) out[4 + j] = s * float((ww.y >> (8 * j)) & 0xFFu) + b;
    for (int j = 0; j < 4; ++j) out[8 + j] = s * float((ww.z >> (8 * j)) & 0xFFu) + b;
    for (int j = 0; j < 4; ++j) out[12 + j] = s * float((ww.w >> (8 * j)) & 0xFFu) + b;
  } else {
    const uint ww = row[k0 / 16];
    for (int j = 0; j < 16; ++j) out[j] = s * float((ww >> (2 * j)) & 0x3u) + b;
  }
}

kernel void qmm(const device uint* w [[buffer(0)]],
                const device ushort* scales [[buffer(1)]],
                const device ushort* biases [[buffer(2)]],
                const device float* x [[buffer(3)]],
                device float* y [[buffer(4)]],
                constant QmmParams& p [[buffer(5)]],
                uint2 tg [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
  // xs and ws back to back; the edge-tile epilogue reuses them as one tile.
  threadgroup float stage[2 * QMM_TILE_M * QMM_TILE_LD];
  threadgroup float* xs = stage;
  threadgroup float* ws = stage + QMM_TILE_M * QMM_TILE_LD;

  const int m0 = int(tg.y) * QMM_TILE_M;  // first token of the tile
  const int n0 = int(tg.x) * QMM_TILE_N;  // first output row of the tile
  const int sm = int(sg / 2) * 32;
  const int sn = int(sg % 2) * 32;
  const int words_per_row = p.cols * QBITS / 32;
  const int groups_per_row = p.cols / QGROUP;

  simdgroup_float8x8 acc[4][4];
  QMM_UNROLL for (int i = 0; i < 4; ++i)
    QMM_UNROLL for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  // Each thread dequantizes half a row of the 64x32 W block: 16 codes.
  const int w_row = int(tid) / 2;
  const int w_col = (int(tid) % 2) * 16;
  const int row = n0 + w_row;
  const bool row_ok = row < p.rows;
  const device uint* wr = w + (row_ok ? row : 0) * words_per_row;

  for (int k0 = 0; k0 < p.cols; k0 += QMM_TILE_K) {
    // X block: 64 tokens x 32 columns = 512 float4, four per thread.
    QMM_UNROLL for (int i = int(tid); i < QMM_TILE_M * QMM_TILE_K / 4; i += 128) {
      const int r = i / (QMM_TILE_K / 4);
      const int c = (i % (QMM_TILE_K / 4)) * 4;
      const int token = m0 + r;
      float4 v = {0.0f, 0.0f, 0.0f, 0.0f};
      if (token < p.n) v = *reinterpret_cast<const device float4*>(x + token * p.cols + k0 + c);
      *reinterpret_cast<threadgroup float4*>(xs + r * QMM_TILE_LD + c) = v;
    }
    {
      float vals[16];
      if (row_ok) {
        const int gi = row * groups_per_row + (k0 + w_col) / QGROUP;
        dequant16(wr, k0 + w_col, scale_at(scales, gi), scale_at(biases, gi), vals);
      } else {
        for (int j = 0; j < 16; ++j) vals[j] = 0.0f;
      }
      threadgroup float* dst = ws + w_row * QMM_TILE_LD + w_col;
      QMM_UNROLL for (int j = 0; j < 16; j += 4) {
        const float4 v = {vals[j], vals[j + 1], vals[j + 2], vals[j + 3]};
        *reinterpret_cast<threadgroup float4*>(dst + j) = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    QMM_UNROLL for (int kk = 0; kk < QMM_TILE_K; kk += 8) {
      simdgroup_float8x8 a[4];
      simdgroup_float8x8 b[4];
      QMM_UNROLL for (int i = 0; i < 4; ++i) simdgroup_load(a[i], xs + (sm + 8 * i) * QMM_TILE_LD + kk, QMM_TILE_LD);
      QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_load(b[j], ws + (sn + 8 * j) * QMM_TILE_LD + kk, QMM_TILE_LD, ulong2(0, 0), true);
      QMM_UNROLL for (int i = 0; i < 4; ++i)
        QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (m0 + QMM_TILE_M <= p.n && n0 + QMM_TILE_N <= p.rows) {
    device float* out = y + (m0 + sm) * p.rows + n0 + sn;
    QMM_UNROLL for (int i = 0; i < 4; ++i)
      QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], out + 8 * i * p.rows + 8 * j, p.rows);
    return;
  }
  // The last loop iteration ended on a barrier, so the staging arrays are free.
  threadgroup float* tile = stage;
  QMM_UNROLL for (int i = 0; i < 4; ++i)
    QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], tile + (sm + 8 * i) * QMM_TILE_OUT_LD + sn + 8 * j, QMM_TILE_OUT_LD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < QMM_TILE_M * QMM_TILE_N; i += 128) {
    const int r = i / QMM_TILE_N;
    const int c = i % QMM_TILE_N;
    const int token = m0 + r;
    const int out_row = n0 + c;
    if (token < p.n && out_row < p.rows) y[token * p.rows + out_row] = tile[r * QMM_TILE_OUT_LD + c];
  }
}

// 4-bit affine qmm (most checkpoints), software-pipelined: each step's X
// float4s and W words load into registers before the previous step's
// multiply-accumulates, so device latency hides behind the math, and
// threadgroup memory holds only the staging (edge tiles leave one 8x8 block
// at a time through a per-simdgroup scratch in it), so more threadgroups fit
// on a core. T is the staging type: half (2x less staging, ~15% faster on an
// M1; for inputs a norm just produced, whose values half holds) or float.
// 64x64 tiles, 128 threads, as qmm. Requires cols % 32 == 0, QGROUP % 32 == 0.
template <typename T>
[[kernel]] void qmm4(const device uint* w [[buffer(0)]],
                     const device ushort* scales [[buffer(1)]],
                     const device ushort* biases [[buffer(2)]],
                     const device float* x [[buffer(3)]],
                     device float* y [[buffer(4)]],
                     constant QmmParams& p [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  constexpr int TM = QMM_TILE_M, TN = QMM_TILE_N, TK = QMM_TILE_K, LD = TK + 8;
  threadgroup float smem[2 * TM * LD * int(sizeof(T)) / 4];
  threadgroup T* xs = reinterpret_cast<threadgroup T*>(smem);
  threadgroup T* ws = xs + TM * LD;
  const int m0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
  const int sm = int(sg / 2) * 32, sn = int(sg % 2) * 32;
  const int words_per_row = p.cols / 8, groups_per_row = p.cols / QGROUP;
  // This thread's W: 16 codes (two words) of row tid / 2 from column (tid % 2) * 16.
  const int w_row = int(tid) / 2, w_col = (int(tid) % 2) * 16;
  const int row = n0 + w_row;
  const bool row_ok = row < p.rows;
  const int safe_row = row_ok ? row : 0;
  const device uint* wr = w + safe_row * words_per_row;
  float4 xr[4];  // X float4s (tid + 128 j): row / 8, column (% 8) * 4
  uint2 wv;
  float ws_scale, ws_bias;
#define QMM4_FETCH(K0)                                                                             \
  {                                                                                                \
    QMM_UNROLL for (int j = 0; j < 4; ++j) {                                                       \
      const int i = int(tid) + 128 * j;                                                            \
      const int r = i / 8, c = (i % 8) * 4;                                                        \
      xr[j] = {0.0f, 0.0f, 0.0f, 0.0f};                                                            \
      if (m0 + r < p.n) xr[j] = *reinterpret_cast<const device float4*>(x + (m0 + r) * p.cols + (K0) + c); \
    }                                                                                              \
    wv = {0u, 0u};                                                                                 \
    if (row_ok) wv = *reinterpret_cast<const device uint2*>(wr + ((K0) + w_col) / 8);              \
    const int gi = safe_row * groups_per_row + ((K0) + w_col) / QGROUP;                            \
    ws_scale = row_ok ? scale_at(scales, gi) : 0.0f;                                               \
    ws_bias = row_ok ? scale_at(biases, gi) : 0.0f;                                                \
  }
  simdgroup_float8x8 acc[4][4];
  QMM_UNROLL for (int i = 0; i < 4; ++i)
    QMM_UNROLL for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  QMM4_FETCH(0)
  for (int k0 = 0; k0 < p.cols; k0 += TK) {
    QMM_UNROLL for (int j = 0; j < 4; ++j) {
      const int i = int(tid) + 128 * j;
      threadgroup T* d = xs + (i / 8) * LD + (i % 8) * 4;
      d[0] = T(xr[j].x);
      d[1] = T(xr[j].y);
      d[2] = T(xr[j].z);
      d[3] = T(xr[j].w);
    }
    {
      threadgroup T* d = ws + w_row * LD + w_col;
      QMM_UNROLL for (int j = 0; j < 8; ++j) d[j] = T(ws_scale * float((wv.x >> (4 * j)) & 0xFu) + ws_bias);
      QMM_UNROLL for (int j = 0; j < 8; ++j) d[8 + j] = T(ws_scale * float((wv.y >> (4 * j)) & 0xFu) + ws_bias);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + TK < p.cols) QMM4_FETCH(k0 + TK)
    QMM_UNROLL for (int kk = 0; kk < TK; kk += 8) {
      simdgroup_matrix<T, 8, 8> a[4], b[4];
      QMM_UNROLL for (int i = 0; i < 4; ++i) simdgroup_load(a[i], xs + (sm + 8 * i) * LD + kk, LD);
      QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_load(b[j], ws + (sn + 8 * j) * LD + kk, LD, ulong2(0, 0), true);
      QMM_UNROLL for (int i = 0; i < 4; ++i)
        QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef QMM4_FETCH
  if (m0 + TM <= p.n && n0 + TN <= p.rows) {
    device float* out = y + (m0 + sm) * p.rows + n0 + sn;
    QMM_UNROLL for (int i = 0; i < 4; ++i)
      QMM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], out + 8 * i * p.rows + 8 * j, p.rows);
    return;
  }
  threadgroup float* scratch = smem + int(sg) * 64;  // the staging is free after the last barrier
  QMM_UNROLL for (int i = 0; i < 4; ++i)
    QMM_UNROLL for (int j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], scratch, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = int(lane); e < 64; e += 32) {
        const int r = m0 + sm + 8 * i + e / 8, c = n0 + sn + 8 * j + e % 8;
        if (r < p.n && c < p.rows) y[r * p.rows + c] = scratch[e];
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template LOKAHI_INSTANCE_ATTRS("qmm4_h") decltype(qmm4<half>) qmm4<half>;
template LOKAHI_INSTANCE_ATTRS("qmm4_f") decltype(qmm4<float>) qmm4<float>;

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
// Grid: threadgroups (heads + kv_heads, n_tokens), heads alone for layers
// that share another layer's keys and values, or kv_heads alone (from
// first_head = heads) to store keys and values only; head_dim / 2 threads
// each.
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
  // M-RoPE (Qwen3.5): frequency k reads the token's position on axis
  // rope_axes[k]. The pass's positions come from rope_pos [n, 3] when
  // rope_given, otherwise each is the KV position plus rope_offset.
  int mrope;
  int rope_given;
  int rope_offset;
  // The head the grid's first threadgroup column handles: 0, or heads to
  // store only keys and values (an MTP head following a pass).
  int first_head;
};

// A token's (temporal, row, column) RoPE positions.
struct RopePos {
  int t, r, c;
};

// The angle of rotary frequency k at a token's positions.
inline float rope_angle(int k, RopePos pos, int mrope, const device int* axes, const device float* inv_freq) {
  const int axis = mrope != 0 ? axes[k] : 0;
  return float(axis == 1 ? pos.r : axis == 2 ? pos.c : pos.t) * inv_freq[k];
}

// Classic partial RoPE for value j of a normed head in `head`: rotate-half
// within the first 2 * rh dimensions; the rest pass through.
inline float rotate_partial(threadgroup const float* head, int j, int rh, RopePos pos, int mrope,
                            const device int* axes, const device float* inv_freq) {
  if (j >= 2 * rh) return head[j];
  const int k = j < rh ? j : j - rh;
  const float angle = rope_angle(k, pos, mrope, axes, inv_freq);
  const float cs = precise::cos(angle);
  const float sn = precise::sin(angle);
  return j < rh ? head[j] * cs - head[j + rh] * sn : head[j] * cs + head[j - rh] * sn;
}

// KV is the cache's element type: half (qk_norm_rope_store_h) or float.
template <typename KV>
[[kernel]] void qk_norm_rope_store(device float* q [[buffer(0)]],
                               const device float* k_in [[buffer(1)]],
                               const device float* v_in [[buffer(2)]],
                               device KV* k_cache [[buffer(3)]],
                               device KV* v_cache [[buffer(4)]],
                               const device float* q_norm [[buffer(5)]],
                               const device float* k_norm [[buffer(6)]],
                               const device float* inv_freq [[buffer(7)]],
                               constant RopeParams& p [[buffer(8)]],
                               const device float* q_bias [[buffer(9)]],
                               const device float* k_bias [[buffer(10)]],
                               const device float* v_bias [[buffer(11)]],
                               const device int* rope_pos [[buffer(12)]],
                               const device int* rope_axes [[buffer(13)]],
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
  const int head = int(tgp.x) + p.first_head;
  const int t = int(tgp.y);
  const int half_dim = p.head_dim / 2;
  const int pos = p.start_pos + t;
  const RopePos rpos = p.rope_given != 0 ? RopePos{rope_pos[3 * t], rope_pos[3 * t + 1], rope_pos[3 * t + 2]}
                                         : RopePos{pos + p.rope_offset, pos + p.rope_offset, pos + p.rope_offset};
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
    const float angle = rope_angle(i, rpos, p.mrope, rope_axes, inv_freq);
    const float cs = precise::cos(angle);
    const float sn = precise::sin(angle);
    ra = na * cs - nb * sn;
    rb = nb * cs + na * sn;
  } else {  // Uniform across the threadgroup.
    normed[i] = na;
    normed[i + half_dim] = nb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ra = rotate_partial(normed, i, p.rotary / 2, rpos, p.mrope, rope_axes, inv_freq);
    rb = rotate_partial(normed, i + half_dim, p.rotary / 2, rpos, p.mrope, rope_axes, inv_freq);
  }
  if (is_q) {
    device float* dst = q + (t * p.heads + head) * p.head_dim;
    dst[i] = ra;
    dst[i + half_dim] = rb;
    return;
  }
  const int slot = pos % p.capacity;
  device KV* kd = k_cache + (kvh * p.capacity + slot) * p.head_dim;
  kd[i] = KV(ra);
  kd[i + half_dim] = KV(rb);
  const device float* vs = v_in + (t * p.kv_heads + kvh) * p.head_dim;
  device KV* vd = v_cache + (kvh * p.capacity + slot) * p.head_dim;
  const device float* vb = v_bias + kvh * p.head_dim;
  float v_lo = vs[i] + (p.qkv_bias != 0 ? vb[i] : 0.0f);
  float v_hi = vs[i + half_dim] + (p.qkv_bias != 0 ? vb[i + half_dim] : 0.0f);
  if (p.v_norm != 0) {  // Uniform across the threadgroup (every kv head).
    const float ss = threadgroup_sum(v_lo * v_lo + v_hi * v_hi, scratch, sg, lane, (tpg + 31) / 32);
    const float inv = precise::rsqrt(ss / float(p.head_dim) + p.eps);
    v_lo *= inv;
    v_hi *= inv;
  }
  vd[i] = KV(v_lo);
  vd[i + half_dim] = KV(v_hi);
}

template LOKAHI_INSTANCE_ATTRS("qk_norm_rope_store") decltype(qk_norm_rope_store<float>) qk_norm_rope_store<float>;
template LOKAHI_INSTANCE_ATTRS("qk_norm_rope_store_h") decltype(qk_norm_rope_store<half>) qk_norm_rope_store<half>;

// ---------------------------------------------------------------------------
// Multi-head latent attention's per-token RoPE and cache write. Threadgroup
// (g, t): g < heads rotates head g's rope query (q[t, g, nope:]) into
// latent_q[t, g, rank:]; g == heads writes the token's cache row: its latent
// (kv[t, :rank]) RMS-normed by kv_norm, then its rope key (kv[t, rank:])
// rotated. RoPE rotates interleaved pairs or halves, then scales by factor
// (YaRN's attention factor).
// ---------------------------------------------------------------------------

struct MlaRopeParams {
  int heads;
  int nope;
  int rope;
  int rank;
  int start_pos;
  float eps;
  int pairs;
  float factor;
};

inline void rope_latent(const device float* src, device float* dst, int dims, int pos, int pairs, float factor,
                        const device float* inv_freq, uint tid, uint tpg) {
  const int half_dims = dims / 2;
  for (int i = int(tid); i < half_dims; i += int(tpg)) {
    const float angle = float(pos) * inv_freq[i];
    const float cs = precise::cos(angle);
    const float sn = precise::sin(angle);
    const int ia = pairs != 0 ? 2 * i : i;
    const int ib = pairs != 0 ? 2 * i + 1 : i + half_dims;
    const float a = src[ia];
    const float b = src[ib];
    dst[ia] = (a * cs - b * sn) * factor;
    dst[ib] = (b * cs + a * sn) * factor;
  }
}

kernel void mla_rope_store(const device float* q [[buffer(0)]],
                           device float* latent_q [[buffer(1)]],
                           const device float* kv [[buffer(2)]],
                           device float* cache [[buffer(3)]],
                           const device float* kv_norm [[buffer(4)]],
                           const device float* inv_freq [[buffer(5)]],
                           constant MlaRopeParams& p [[buffer(6)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           uint2 tid2 [[thread_position_in_threadgroup]],
                           uint2 tpg2 [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const uint tid = tid2.x;
  const uint tpg = tpg2.x;
  const int g = int(tgp.x);
  const int t = int(tgp.y);
  const int pos = p.start_pos + t;
  const int width = p.rank + p.rope;
  if (g < p.heads) {  // uniform across the threadgroup
    const int row = t * p.heads + g;
    rope_latent(q + row * (p.nope + p.rope) + p.nope, latent_q + row * width + p.rank, p.rope, pos, p.pairs,
                p.factor, inv_freq, tid, tpg);
    return;
  }
  const device float* src = kv + t * width;
  device float* dst = cache + pos * width;
  float ss = 0.0f;
  for (int i = int(tid); i < p.rank; i += int(tpg)) ss += src[i] * src[i];
  const float total = threadgroup_sum(ss, scratch, sg, lane, (tpg + 31) / 32);
  const float inv = precise::rsqrt(total / float(p.rank) + p.eps);
  for (int i = int(tid); i < p.rank; i += int(tpg)) dst[i] = src[i] * inv * kv_norm[i];
  rope_latent(src + p.rank, dst + p.rank, p.rope, pos, p.pairs, p.factor, inv_freq, tid, tpg);
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (Qwen3.5, Qwen3-Next). Per pass of n tokens: the input
// projections (qmm) fill mixed [n, C] = [q | k | v] channels, z, and the raw
// b and a [n, Hv] (Qwen3-Next's grouped projections go through gdn_split
// first); gdn_conv runs the causal depthwise conv and SiLU over the channels;
// gdn_qk_norm L2-normalizes q and k per key head; gdn_gates turns b into beta
// and a into the decay exp(g); gdn_recurrence runs every value head's state
// over the pass; gated_rms_norm_rows applies rms(o) * w * silu(z) per value
// head before out_proj. The conv history [taps - 1, C] and the states
// [Hv, dv, dk] are double-buffered by position parity: a pass at start reads
// buffer (start + parity) % 2 and writes buffer (start + n + parity) % 2, so
// a decode step launched past a stop token, and then discarded, never
// overwrites the state the next step reads. Each thread (or simdgroup) owns
// its slice of the state, so a pass whose two buffers coincide updates it in
// place safely. A speculative verify pass always writes the other buffer,
// and gdn_history_replay and gdn_recurrence rebuild it for a kept prefix of
// the pass from the one it read.
// ---------------------------------------------------------------------------

struct GdnSplitParams {
  int key_heads;    // Hk
  int key_dim;      // dk
  int ratio;        // value heads per key head
  int value_dim;    // dv
  int n;
};

// Qwen3-Next: each key head's [q, k, v, z] rows of in_proj_qkvz and [b, a]
// rows of in_proj_ba into the [q | k | v] channels, z, b and a. Grid:
// (Hk * (2 dk + 2 r dv + 2 r), n).
kernel void gdn_split(const device float* qkvz [[buffer(0)]],
                      const device float* ba [[buffer(1)]],
                      device float* mixed [[buffer(2)]],
                      device float* z [[buffer(3)]],
                      device float* b [[buffer(4)]],
                      device float* a [[buffer(5)]],
                      constant GdnSplitParams& p [[buffer(6)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const int dk = p.key_dim, r = p.ratio, dv = p.value_dim, hk = p.key_heads;
  const int group = 2 * dk + 2 * r * dv, per_head = group + 2 * r;
  const int kh = int(gid.x) / per_head, j = int(gid.x) % per_head, t = int(gid.y);
  if (kh >= hk || t >= p.n) return;
  const int channels = 2 * hk * dk + hk * r * dv, values = hk * r * dv, hv = hk * r;
  if (j >= group) {  // [b, a]
    const int i = j - group;
    const float value = ba[(t * hk + kh) * 2 * r + i];
    if (i < r) b[t * hv + kh * r + i] = value;
    else a[t * hv + kh * r + i - r] = value;
    return;
  }
  const float value = qkvz[(t * hk + kh) * group + j];
  device float* row = mixed + t * channels;
  if (j < dk) row[kh * dk + j] = value;
  else if (j < 2 * dk) row[(hk + kh) * dk + j - dk] = value;
  else if (j < 2 * dk + r * dv) row[2 * hk * dk + kh * r * dv + j - 2 * dk] = value;
  else z[t * values + kh * r * dv + j - 2 * dk - r * dv] = value;
}

constant constexpr int GDN_MAX_TAPS = 8;

struct GdnConvParams {
  int channels;
  int taps;
  int n;
};

// mixed[t, c] = silu(sum_k taps[c, k] * input[t - (K - 1) + k, c]), in place,
// with the inputs before the pass from history_in; history_out gets the
// pass's last K - 1 inputs. One thread per channel.
kernel void gdn_conv(device float* mixed [[buffer(0)]],
                     const device float* taps [[buffer(1)]],
                     const device float* history_in [[buffer(2)]],
                     device float* history_out [[buffer(3)]],
                     constant GdnConvParams& p [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
  const int c = int(gid), C = p.channels, K = p.taps;
  if (c >= C) return;
  float window[GDN_MAX_TAPS];  // the previous K - 1 inputs, oldest first
  float w[GDN_MAX_TAPS];
  for (int k = 0; k < K; ++k) w[k] = taps[c * K + k];
  for (int k = 0; k < K - 1; ++k) window[k] = history_in[k * C + c];
  for (int t = 0; t < p.n; ++t) {
    const float x = mixed[t * C + c];
    float sum = w[K - 1] * x;
    for (int k = 0; k < K - 1; ++k) sum += w[k] * window[k];
    for (int k = 0; k + 1 < K - 1; ++k) window[k] = window[k + 1];
    if (K > 1) window[K - 2] = x;
    mixed[t * C + c] = silu(sum);
  }
  for (int k = 0; k < K - 1; ++k) history_out[k * C + c] = window[k];
}

struct GdnHistoryParams {
  int channels;
  int taps;
  int kept;
};

// Speculative decoding: the conv history after the first `kept` tokens of a
// verify pass, the last K - 1 rows of [history_in, raw], where raw holds the
// pass's conv inputs. history_in and history_out are different buffers.
// Grid: (C, K - 1).
kernel void gdn_history_replay(const device float* history_in [[buffer(0)]],
                               const device float* raw [[buffer(1)]],
                               device float* history_out [[buffer(2)]],
                               constant GdnHistoryParams& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x), k = int(gid.y), C = p.channels, H = p.taps - 1;
  if (c >= C || k >= H) return;
  const int at = p.kept + k;  // the row of the stream [history_in, raw]
  history_out[k * C + c] = at < H ? history_in[at * C + c] : raw[(at - H) * C + c];
}

struct GdnNormParams {
  int key_heads;
  int key_dim;
  int channels;
  float q_scale;
};

// q and k L2-normalized per key head, x * rsqrt(sum(x^2) + 1e-6), q also
// times q_scale (dk^-0.5). Grid: threadgroups (2 Hk, n) of dk threads.
kernel void gdn_qk_norm(device float* mixed [[buffer(0)]],
                        constant GdnNormParams& p [[buffer(1)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint2 tid2 [[thread_position_in_threadgroup]],
                        uint2 tpg2 [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const int head = int(tgp.x), t = int(tgp.y), i = int(tid2.x);
  device float* x = mixed + t * p.channels + head * p.key_dim;
  const float v = x[i];
  const float total = threadgroup_sum(v * v, scratch, sg, lane, (tpg2.x + 31) / 32);
  const float scale = head < p.key_heads ? p.q_scale : 1.0f;
  x[i] = v * precise::rsqrt(total + 1e-6f) * scale;
}

struct GdnGateParams {
  int value_heads;
  int count;  // n * Hv
};

// beta = sigmoid(b) and decay = exp(-A * softplus(a + dt_bias)), A = exp(A_log), in place.
kernel void gdn_gates(device float* b [[buffer(0)]],
                      device float* a [[buffer(1)]],
                      const device float* decay_rate [[buffer(2)]],
                      const device float* dt_bias [[buffer(3)]],
                      constant GdnGateParams& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.count) return;
  const int h = int(gid) % p.value_heads;
  b[gid] = 1.0f / (1.0f + precise::exp(-b[gid]));
  const float x = a[gid] + dt_bias[h];
  const float softplus = x > 0.0f ? x + precise::log(1.0f + precise::exp(-x)) : precise::log(1.0f + precise::exp(x));
  a[gid] = precise::exp(-decay_rate[h] * softplus);
}

constant constexpr int GDN_MAX_ROWS = 8;  // dk up to 256: dk / 32 state values per lane
constant constexpr int GDN_SIMDGROUPS = 4;

struct GdnRecurrenceParams {
  int key_heads;
  int value_heads;
  int key_dim;
  int value_dim;
  int channels;
  int n;
};

// One simdgroup per (value head h, value column j): its lanes hold column j
// of S (rows lane, lane + 32, ...), which per token decays by exp(g), takes
// k * (beta * (v_j - S_j . k)) and gives out[t, h, j] = S_j . q. The state is
// stored [Hv, dv, dk], so each column's rows are contiguous. Grid:
// threadgroups (ceil(dv / 4), Hv) of 4 simdgroups.
kernel void gdn_recurrence(const device float* mixed [[buffer(0)]],
                           const device float* beta [[buffer(1)]],
                           const device float* decay [[buffer(2)]],
                           const device float* state_in [[buffer(3)]],
                           device float* state_out [[buffer(4)]],
                           device float* out [[buffer(5)]],
                           constant GdnRecurrenceParams& p [[buffer(6)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const int j = int(tgp.x) * GDN_SIMDGROUPS + int(sg), h = int(tgp.y);
  if (j >= p.value_dim) return;  // uniform across the simdgroup
  const int dk = p.key_dim, hv = p.value_heads;
  const int kh = h / (hv / p.key_heads);
  const int rows = (dk - int(lane) + 31) / 32;  // this lane's rows: lane, lane + 32, ...
  const size_t column = (size_t(h) * p.value_dim + j) * dk;
  float s[GDN_MAX_ROWS];
  for (int r = 0; r < rows; ++r) s[r] = state_in[column + lane + 32 * r];
  for (int t = 0; t < p.n; ++t) {
    const device float* row = mixed + t * p.channels;
    const device float* q = row + kh * dk;
    const device float* k = row + (p.key_heads + kh) * dk;
    const float v = row[2 * p.key_heads * dk + h * p.value_dim + j];
    const float g = decay[t * hv + h], bt = beta[t * hv + h];
    float memory = 0.0f;
    for (int r = 0; r < rows; ++r) {
      s[r] *= g;
      memory += s[r] * k[lane + 32 * r];
    }
    const float delta = (v - simd_sum(memory)) * bt;
    float read = 0.0f;
    for (int r = 0; r < rows; ++r) {
      s[r] += k[lane + 32 * r] * delta;
      read += s[r] * q[lane + 32 * r];
    }
    read = simd_sum(read);
    if (lane == 0) out[(t * hv + h) * p.value_dim + j] = read;
  }
  for (int r = 0; r < rows; ++r) state_out[column + lane + 32 * r] = s[r];
}

// Gated DeltaNet's output norm: out = rms(o) * w * silu(z) per row of
// p.cols (one value head), with a plain weight. One threadgroup per row.
kernel void gated_rms_norm_rows(device float* o [[buffer(0)]],
                                const device float* weight [[buffer(1)]],
                                const device float* z [[buffer(2)]],
                                constant NormParams& p [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tpg [[threads_per_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device float* x = o + row * p.cols;
  const device float* gate = z + row * p.cols;
  float ss = 0.0f;
  for (int i = int(tid); i < p.cols; i += int(tpg)) ss += x[i] * x[i];
  const float total = threadgroup_sum(ss, scratch, sg, lane, (tpg + 31) / 32);
  const float inv = precise::rsqrt(total / float(p.cols) + p.eps);
  for (int i = int(tid); i < p.cols; i += int(tpg)) x[i] = x[i] * inv * weight[i] * silu(gate[i]);
}

// Zeroes Gated DeltaNet's conv histories and states, on load and reset.
kernel void zero_floats(device float* x [[buffer(0)]],
                        constant ElementwiseParams& p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) x[gid] = 0.0f;
}

struct SplitGateParams {
  int head_dim;
  int count;  // n * heads * head_dim
};

// Qwen3.5's gated attention: q_proj's [query | gate] rows per head into q and
// gate, both [n, heads, head_dim].
kernel void split_query_gate(const device float* both [[buffer(0)]],
                             device float* q [[buffer(1)]],
                             device float* gate [[buffer(2)]],
                             constant SplitGateParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.count) return;
  const int head = int(gid) / p.head_dim, d = int(gid) % p.head_dim;
  q[gid] = both[(2 * head) * p.head_dim + d];
  gate[gid] = both[(2 * head + 1) * p.head_dim + d];
}

struct SigmoidScaleParams {
  int cols;   // values per scale
  int count;  // n * rows * cols
};

// x[i] *= sigmoid(scales[i / cols]): the gated attention's output (cols 1)
// and the shared expert's gate (one scale per token, cols hidden).
kernel void mul_sigmoid(device float* x [[buffer(0)]],
                        const device float* scales [[buffer(1)]],
                        constant SigmoidScaleParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= p.count) return;
  x[gid] *= 1.0f / (1.0f + precise::exp(-scales[int(gid) / p.cols]));
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
  int groups_per_kv;  // threadgroups of G query heads per kv head: 1, or heads / G for latent attention
};

// KV is the cache's element type: half (the _h instances) or float.
template <int G, int DPL, typename KV>
[[kernel]] void attn_partial(const device float* q [[buffer(0)]],
                             const device KV* k_cache [[buffer(1)]],
                             const device KV* v_cache [[buffer(2)]],
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
  const int qg = int(tgp.y);  // this threadgroup's G query heads are qg * G ...
  const int kvh = qg / p.groups_per_kv;
  const int t = int(tgp.z);
  const int D = p.head_dim;
  const int pos = p.start_pos + t;
  const int lo = max(0, pos - p.window + 1);
  const int hi = p.has_last_key != 0 ? max(pos, last_key[t]) : pos;
  const int kb = lo + blk * p.block;
  const int ke = min(kb + p.block, hi + 1);

  float qv[G][DPL];
  for (int g = 0; g < G; ++g) {
    const device float* qh = q + (t * p.heads + qg * G + g) * D + int(lane) * DPL;
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
    const device KV* kp = k_cache + (kvh * p.capacity + slot) * D + int(lane) * DPL;
    const device KV* vp = v_cache + (kvh * p.capacity + slot) * D + int(lane) * DPL;
    float kv[DPL], vv[DPL];
    for (int j = 0; j < DPL; ++j) {
      kv[j] = float(kp[j]);
      vv[j] = float(vp[j]);
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
    const int head = qg * G + g;
    const int index = (t * p.heads + head) * p.n_blocks + blk;
    device float* o = part_o + index * D + int(lane) * DPL;
    for (int j = 0; j < DPL; ++j) o[j] = acc[g][j];
    if (lane == 0) {
      part_ml[2 * index] = m[g];
      part_ml[2 * index + 1] = l[g];
    }
  }
}

#define LOKAHI_ATTN_KV(G, DPL, KV, suffix)                                  \
  template LOKAHI_INSTANCE_ATTRS("attn_partial_g" #G "_d" #DPL suffix)      \
  decltype(attn_partial<G, DPL, KV>) attn_partial<G, DPL, KV>;
#define LOKAHI_ATTN(G, DPL) LOKAHI_ATTN_KV(G, DPL, float, "") LOKAHI_ATTN_KV(G, DPL, half, "_h")

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
// Multi-head latent attention: four query heads per threadgroup over the
// one latent key head, 576 wide in every released model (96 in the tiny
// fixtures). Its cache stays float32.
LOKAHI_ATTN_KV(4, 3, float, "")
LOKAHI_ATTN_KV(4, 18, float, "")

// ---------------------------------------------------------------------------
// Prefill attention on simdgroup matrices: scores S = Q K^T (scaled, soft-capped
// and masked as they are written), a row softmax (softmax_rows), then O = P V.
// attn_partial suits decode, where one query row spreads its keys over
// threadgroups; a long pass instead reuses every key and value tile across 64
// query rows. Both products take qmm's 64x64 tiles (four simdgroups of 4x4 8x8
// matrices, K in steps of 32). Keys are positions key_base ... key_base +
// keys - 1, read from their cache slots (position % capacity), so a sliding
// layer's ring is read in order. A sub-pass covers query rows row0 ... row0 +
// rows - 1 of the pass; S holds [heads, rows, keys]. Requires head_dim % 64 == 0.
// ---------------------------------------------------------------------------

struct AttnMmParams {
  int heads;
  int kv_heads;
  int head_dim;
  int start_pos;     // position of the pass's row 0
  int row0;          // this sub-pass's first query row within the pass
  int rows;          // query rows in this sub-pass
  int key_base;      // position of key 0
  int keys;          // keys per query row in S
  int window;        // keys allowed per query (sliding window or a large value)
  int capacity;      // KV slots for this layer
  float scale;
  float softcap;     // 0 disables
  int has_last_key;  // last_key[t] extends query t's keys (Gemma 3's image blocks)
};

#define ATT_MM_UNROLL _Pragma("clang loop unroll(full)")
constant constexpr int ATT_MM_T = 64;              // tile rows and columns
constant constexpr int ATT_MM_K = 32;              // reduction step
constant constexpr int ATT_MM_LD = ATT_MM_K + 4;   // staging stride
constant constexpr int ATT_MM_OUT_LD = ATT_MM_T + 4;

// The four simdgroups' 32x32 quadrants of a 64x64 tile over one K step.
inline void att_mm_step(threadgroup float* as, threadgroup float* bs, int sm, int sn,
                        thread simdgroup_float8x8 (&acc)[4][4]) {
  ATT_MM_UNROLL for (int kk = 0; kk < ATT_MM_K; kk += 8) {
    simdgroup_float8x8 a[4];
    simdgroup_float8x8 b[4];
    ATT_MM_UNROLL for (int i = 0; i < 4; ++i) simdgroup_load(a[i], as + (sm + 8 * i) * ATT_MM_LD + kk, ATT_MM_LD);
    ATT_MM_UNROLL for (int j = 0; j < 4; ++j)
      simdgroup_load(b[j], bs + (sn + 8 * j) * ATT_MM_LD + kk, ATT_MM_LD, ulong2(0, 0), true);
    ATT_MM_UNROLL for (int i = 0; i < 4; ++i)
      ATT_MM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
}

// Grid (ceil(keys / 64), ceil(rows / 64), heads), 128 threads.
template <typename KV>
[[kernel]] void attn_qk(const device float* q [[buffer(0)]],
                        const device KV* k_cache [[buffer(1)]],
                        device float* s [[buffer(2)]],
                        constant AttnMmParams& p [[buffer(3)]],
                        const device int* last_key [[buffer(4)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float stage[2 * ATT_MM_T * ATT_MM_LD];
  threadgroup float* as = stage;
  threadgroup float* bs = stage + ATT_MM_T * ATT_MM_LD;
  const int D = p.head_dim;
  const int h = int(tg.z);
  const int kvh = h / (p.heads / p.kv_heads);
  const int r0 = int(tg.y) * ATT_MM_T;  // first query row of the tile, within the sub-pass
  const int j0 = int(tg.x) * ATT_MM_T;  // first key of the tile
  const int sm = int(sg / 2) * 32;
  const int sn = int(sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  ATT_MM_UNROLL for (int i = 0; i < 4; ++i)
    ATT_MM_UNROLL for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int k0 = 0; k0 < D; k0 += ATT_MM_K) {
    // 64 rows x 32 dims of Q and of K: 512 float4 each, four per thread.
    ATT_MM_UNROLL for (int i = int(tid); i < ATT_MM_T * ATT_MM_K / 4; i += 128) {
      const int r = i / (ATT_MM_K / 4);
      const int c = (i % (ATT_MM_K / 4)) * 4;
      float4 qv = {0.0f, 0.0f, 0.0f, 0.0f};
      if (r0 + r < p.rows) qv = *reinterpret_cast<const device float4*>(q + ((p.row0 + r0 + r) * p.heads + h) * D + k0 + c);
      *reinterpret_cast<threadgroup float4*>(as + r * ATT_MM_LD + c) = qv;
      float4 kv = {0.0f, 0.0f, 0.0f, 0.0f};
      if (j0 + r < p.keys) {
        const int slot = (p.key_base + j0 + r) % p.capacity;
        const device KV* kp = k_cache + (kvh * p.capacity + slot) * D + k0 + c;
        kv = {float(kp[0]), float(kp[1]), float(kp[2]), float(kp[3])};
      }
      *reinterpret_cast<threadgroup float4*>(bs + r * ATT_MM_LD + c) = kv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    att_mm_step(as, bs, sm, sn, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* tile = stage;  // free after the last barrier
  ATT_MM_UNROLL for (int i = 0; i < 4; ++i)
    ATT_MM_UNROLL for (int j = 0; j < 4; ++j)
      simdgroup_store(acc[i][j], tile + (sm + 8 * i) * ATT_MM_OUT_LD + sn + 8 * j, ATT_MM_OUT_LD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < ATT_MM_T * ATT_MM_T; i += 128) {
    const int r = i / ATT_MM_T;
    const int c = i % ATT_MM_T;
    const int row = r0 + r, key = j0 + c;
    if (row >= p.rows || key >= p.keys) continue;
    const int t = p.row0 + row;  // query row within the pass
    const int qpos = p.start_pos + t;
    const int kpos = p.key_base + key;
    const int hi = p.has_last_key != 0 ? max(qpos, last_key[t]) : qpos;
    float v = tile[r * ATT_MM_OUT_LD + c] * p.scale;
    if (p.softcap > 0.0f) v = precise::tanh(v / p.softcap) * p.softcap;
    const bool allowed = kpos <= hi && kpos > qpos - p.window;
    s[(h * p.rows + row) * p.keys + key] = allowed ? v : -INFINITY;
  }
}

// Grid (head_dim / 64, ceil(rows / 64), heads), 128 threads.
template <typename KV>
[[kernel]] void attn_pv(const device float* s [[buffer(0)]],
                        const device KV* v_cache [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant AttnMmParams& p [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float stage[2 * ATT_MM_T * ATT_MM_LD];
  threadgroup float* as = stage;
  threadgroup float* bs = stage + ATT_MM_T * ATT_MM_LD;
  const int D = p.head_dim;
  const int h = int(tg.z);
  const int kvh = h / (p.heads / p.kv_heads);
  const int r0 = int(tg.y) * ATT_MM_T;
  const int d0 = int(tg.x) * ATT_MM_T;
  const int sm = int(sg / 2) * 32;
  const int sn = int(sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  ATT_MM_UNROLL for (int i = 0; i < 4; ++i)
    ATT_MM_UNROLL for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int k0 = 0; k0 < p.keys; k0 += ATT_MM_K) {
    // P: 64 rows x 32 keys; V: 32 keys x 64 dims, staged transposed as [dim][key].
    ATT_MM_UNROLL for (int i = int(tid); i < ATT_MM_T * ATT_MM_K; i += 128) {
      const int r = i / ATT_MM_K;
      const int c = i % ATT_MM_K;
      const int row = r0 + r, key = k0 + c;
      as[r * ATT_MM_LD + c] = row < p.rows && key < p.keys ? s[(h * p.rows + row) * p.keys + key] : 0.0f;
      const int kk = i / ATT_MM_T;  // consecutive threads read consecutive dims of one key
      const int d = i % ATT_MM_T;
      float vv = 0.0f;
      if (k0 + kk < p.keys) {
        const int slot = (p.key_base + k0 + kk) % p.capacity;
        vv = float(v_cache[(kvh * p.capacity + slot) * D + d0 + d]);
      }
      bs[d * ATT_MM_LD + kk] = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    att_mm_step(as, bs, sm, sn, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (r0 + ATT_MM_T <= p.rows) {
    device float* o = out + ((p.row0 + r0 + sm) * p.heads + h) * D + d0 + sn;
    const int ld = p.heads * D;  // consecutive query rows of one head
    ATT_MM_UNROLL for (int i = 0; i < 4; ++i)
      ATT_MM_UNROLL for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], o + 8 * i * ld + 8 * j, ld);
    return;
  }
  threadgroup float* tile = stage;
  ATT_MM_UNROLL for (int i = 0; i < 4; ++i)
    ATT_MM_UNROLL for (int j = 0; j < 4; ++j)
      simdgroup_store(acc[i][j], tile + (sm + 8 * i) * ATT_MM_OUT_LD + sn + 8 * j, ATT_MM_OUT_LD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < ATT_MM_T * ATT_MM_T; i += 128) {
    const int r = i / ATT_MM_T;
    const int c = i % ATT_MM_T;
    if (r0 + r < p.rows) out[((p.row0 + r0 + r) * p.heads + h) * D + d0 + c] = tile[r * ATT_MM_OUT_LD + c];
  }
}

template LOKAHI_INSTANCE_ATTRS("attn_qk") decltype(attn_qk<float>) attn_qk<float>;
template LOKAHI_INSTANCE_ATTRS("attn_qk_h") decltype(attn_qk<half>) attn_qk<half>;
template LOKAHI_INSTANCE_ATTRS("attn_pv") decltype(attn_pv<float>) attn_pv<float>;
template LOKAHI_INSTANCE_ATTRS("attn_pv_h") decltype(attn_pv<half>) attn_pv<half>;

// ---------------------------------------------------------------------------
// Fused prefill attention (flash attention). A threadgroup takes 32 query rows
// of one head, 8 per simdgroup, and streams the keys those rows can see in
// blocks of BK, keeping S = Q K^T, the online softmax and O = P V in
// registers: no scores in memory, and the key blocks past the tile's causal
// edge or before its window are never read, so a causal pass does about half
// of attn_qk + attn_pv's arithmetic. Each thread of a simdgroup_matrix<T, 8, 8>
// holds two adjacent elements of one row: row (lane / 4 & 4) + (lane / 2 % 4),
// columns (lane / 4 & 2) * 2 + (lane % 2) * 2 + {0, 1} (checked on M1, and
// MLX's steel attention relies on the same layout). Every fragment a thread
// holds is therefore in the same row, the row's four threads differ in lane
// bits 0 and 3, and the per-row max, sum and rescale take two shuffles each.
// Takes AttnMmParams with row0 = 0 and rows = the pass.
// ---------------------------------------------------------------------------

constant constexpr int ATT_FL_Q = 32;  // query rows per threadgroup

inline float att_fl_row_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}

inline float att_fl_row_sum(float v) {
  v += simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// Grid (ceil(rows / 32), heads), 128 * SPLIT threads. KV blocks of BK keys
// are staged as stored (half or float), BK * (D + pad) elements each for K and
// V. With SPLIT = 2 (512-wide heads, whose Q and O would not fit one
// simdgroup's registers) two simdgroups share each 8-row group, each holding
// half of the head's dimensions: each computes its half's share of the scores,
// they add the two through threadgroup memory, and both run the same softmax
// and accumulate their half of O.
template <typename KV, int D, int BK, int SPLIT>
[[kernel]] void attn_flash(const device float* q [[buffer(0)]],
                           const device KV* k_cache [[buffer(1)]],
                           const device KV* v_cache [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant AttnMmParams& p [[buffer(4)]],
                           const device int* last_key [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  constexpr int LD = D + 16 / int(sizeof(KV));  // 16 bytes of padding per staged key
  constexpr int DH = D / SPLIT;                  // dimensions per simdgroup
  constexpr int THREADS = 128 * SPLIT;
  threadgroup KV ks[BK * LD];
  threadgroup KV vs[BK * LD];
  threadgroup float halves[SPLIT > 1 ? 4 * SPLIT * (BK / 8) * 64 : 1];  // partial scores to exchange
  const int h = int(tg.y);
  const int kvh = h / (p.heads / p.kv_heads);
  const int t0 = int(tg.x) * ATT_FL_Q;
  const int t_last = min(t0 + ATT_FL_Q, p.rows) - 1;
  // This thread's row and first column within every 8x8 fragment.
  const int fm = int((lane / 4) & 4) + int((lane / 2) % 4);
  const int fn = int((lane / 4) & 2) * 2 + int(lane % 2) * 2;
  const int group = int(sg) % 4, half_index = int(sg) / 4;  // 8-row group; half of D (SPLIT = 2)
  const int d0 = half_index * DH;
  const int t = t0 + group * 8 + fm;  // query row within the pass
  const bool row_ok = t < p.rows;
  const int qpos = p.start_pos + t;
  const int lo = qpos - p.window + 1;  // first key the row may see
  const int hi = p.has_last_key != 0 && row_ok ? max(qpos, last_key[t]) : qpos;

  // The tile's key range: from its first row's window start to its furthest key.
  int tile_hi = p.start_pos + t_last;
  if (p.has_last_key != 0) tile_hi = max(tile_hi, simd_max(t0 + int(lane) <= t_last ? last_key[t0 + int(lane)] : 0));
  tile_hi = min(tile_hi, p.key_base + p.keys - 1);
  const int tile_lo = max(p.key_base, p.start_pos + t0 - p.window + 1);

  // Q, prescaled so the scores come out in log2 units (softcap applies the
  // log2 factor after its tanh instead).
  const float q_scale = p.softcap > 0.0f ? p.scale : p.scale * M_LOG2E_F;
  simdgroup_float8x8 qf[DH / 8];
  simdgroup_float8x8 o[DH / 8];
  const device float* qrow = q + (size_t(row_ok ? t : 0) * size_t(p.heads) + size_t(h)) * D + d0 + fn;
  ATT_MM_UNROLL for (int d = 0; d < DH / 8; ++d) {
    const float2 v = row_ok ? *reinterpret_cast<const device float2*>(qrow + 8 * d) : float2(0.0f);
    FRAG_ELEMS(qf[d]) = v * q_scale;
    o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float m = -INFINITY;  // running max (log2 units) and sum of this thread's row
  float l = 0.0f;

  for (int kb = tile_lo; kb <= tile_hi; kb += BK) {
    // Stage keys kb ... kb + BK - 1 from their ring slots; zeros past tile_hi.
    const int slot0 = kb % p.capacity;
    ATT_MM_UNROLL for (int i = int(tid); i < BK * D / 4; i += THREADS) {
      const int r = i / (D / 4);
      const int c = (i % (D / 4)) * 4;
      vec<KV, 4> kv4 = vec<KV, 4>(KV(0));
      vec<KV, 4> vv4 = vec<KV, 4>(KV(0));
      if (kb + r <= tile_hi) {
        int slot = slot0 + r;
        if (slot >= p.capacity) slot -= p.capacity;
        const size_t at = (size_t(kvh) * size_t(p.capacity) + size_t(slot)) * D + c;
        kv4 = *reinterpret_cast<const device vec<KV, 4>*>(k_cache + at);
        vv4 = *reinterpret_cast<const device vec<KV, 4>*>(v_cache + at);
      }
      *reinterpret_cast<threadgroup vec<KV, 4>*>(ks + r * LD + c) = kv4;
      *reinterpret_cast<threadgroup vec<KV, 4>*>(vs + r * LD + c) = vv4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 s[BK / 8];
    ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) s[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    ATT_MM_UNROLL for (int d = 0; d < DH / 8; ++d) {
      ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) {
        simdgroup_matrix<KV, 8, 8> kf;
        simdgroup_load(kf, ks + 8 * j * LD + d0 + 8 * d, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], qf[d], kf, s[j]);
      }
    }
    if (SPLIT > 1) {  // add the other half's share of the scores
      threadgroup float* mine = halves + ((group * SPLIT + half_index) * (BK / 8)) * 64;
      threadgroup float* other = halves + ((group * SPLIT + (1 - half_index)) * (BK / 8)) * 64;
      ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) simdgroup_store(s[j], mine + 64 * j, 8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) {
        simdgroup_float8x8 theirs;
        simdgroup_load(theirs, other + 64 * j, 8);
        FRAG_ELEMS(s[j]) += FRAG_ELEMS(theirs);
      }
    }

    // Soft-cap and mask, then the online softmax update of this row.
    // A block every row of the tile sees in full needs no mask.
    const bool full = kb + BK - 1 <= min(p.start_pos + t0, tile_hi) && kb >= p.start_pos + t_last - p.window + 1;
    float block_max = -INFINITY;
    ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) {
      thread float2& e = FRAG_ELEMS(s[j]);
      if (p.softcap > 0.0f) {
        e.x = precise::tanh(e.x / p.softcap) * p.softcap * M_LOG2E_F;
        e.y = precise::tanh(e.y / p.softcap) * p.softcap * M_LOG2E_F;
      }
      if (!full) {
        const int key = kb + 8 * j + fn;
        e.x = key >= lo && key <= hi ? e.x : -INFINITY;
        e.y = key + 1 >= lo && key + 1 <= hi ? e.y : -INFINITY;
      }
      block_max = max(block_max, max(e.x, e.y));
    }
    const float m_new = max(m, att_fl_row_max(block_max));
    const float m_ref = m_new == -INFINITY ? 0.0f : m_new;  // a row with no key yet keeps zeros
    const float factor = fast::exp2(m - m_ref);
    float block_sum = 0.0f;
    ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) {
      thread float2& e = FRAG_ELEMS(s[j]);
      e.x = fast::exp2(e.x - m_ref);
      e.y = fast::exp2(e.y - m_ref);
      block_sum += e.x + e.y;
    }
    l = l * factor + att_fl_row_sum(block_sum);
    m = m_new;
    ATT_MM_UNROLL for (int d = 0; d < DH / 8; ++d) FRAG_ELEMS(o[d]) *= factor;
    ATT_MM_UNROLL for (int j = 0; j < BK / 8; ++j) {
      ATT_MM_UNROLL for (int d = 0; d < DH / 8; ++d) {
        simdgroup_matrix<KV, 8, 8> vf;
        simdgroup_load(vf, vs + 8 * j * LD + d0 + 8 * d, LD);
        simdgroup_multiply_accumulate(o[d], s[j], vf, o[d]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (!row_ok) return;
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  device float* orow = out + (size_t(t) * size_t(p.heads) + size_t(h)) * D + d0 + fn;
  ATT_MM_UNROLL for (int d = 0; d < DH / 8; ++d)
    *reinterpret_cast<device float2*>(orow + 8 * d) = FRAG_ELEMS(o[d]) * inv;
}

#define LOKAHI_ATTN_FLASH(D, BK_H, BK_F)                                       \
  template LOKAHI_INSTANCE_ATTRS("attn_flash_d" #D "_h")                      \
  decltype(attn_flash<half, D, BK_H, 1>) attn_flash<half, D, BK_H, 1>;          \
  template LOKAHI_INSTANCE_ATTRS("attn_flash_d" #D)                           \
  decltype(attn_flash<float, D, BK_F, 1>) attn_flash<float, D, BK_F, 1>;
// Key blocks keep K and V staging within 32 KB of threadgroup memory; on M1,
// 16 keys beat 32 at 128-wide heads and 8 beat 16 at 256-wide heads, where
// Q and O already fill the registers.
LOKAHI_ATTN_FLASH(64, 32, 32)
LOKAHI_ATTN_FLASH(128, 16, 16)
LOKAHI_ATTN_FLASH(256, 8, 8)
// 512-wide heads (Gemma 4's full-attention layers): a half cache only, as 8
// float keys of 512 would not fit.
template LOKAHI_INSTANCE_ATTRS("attn_flash_d512_h") decltype(attn_flash<half, 512, 8, 2>) attn_flash<half, 512, 8, 2>;

// One simdgroup writes, per lane, the two elements thread_elements() holds of
// a loaded 8x8 matrix (element r * 8 + c holds r * 8 + c) and of its product
// with the identity, so the host can check the fragment layout attn_flash
// assumes before using it on this GPU.
kernel void simdgroup_layout_probe(device float* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float m[64];
  threadgroup float eye[64];
  for (uint i = lane; i < 64; i += 32) {
    m[i] = float(i);
    eye[i] = i / 8 == i % 8 ? 1.0f : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 a;
  simdgroup_float8x8 e;
  simdgroup_load(a, m, 8);
  simdgroup_load(e, eye, 8);
  simdgroup_float8x8 p = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_multiply_accumulate(p, e, a, p);
  const float2 la = FRAG_ELEMS(a);
  const float2 lp = FRAG_ELEMS(p);
  out[lane * 4 + 0] = la.x;
  out[lane * 4 + 1] = la.y;
  out[lane * 4 + 2] = lp.x;
  out[lane * 4 + 3] = lp.y;
}

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

// ---------------------------------------------------------------------------
// Qwen3.5's vision tower: rotate-half 2-D RoPE over whole heads, and the
// merger's exact GELU.
// ---------------------------------------------------------------------------

struct HalfRopeParams {
  int heads;
  int dim;     // head size; rotate-half over the whole head
  int stride;  // floats from one token's heads to the next's
};

// In place on heads of `dim` floats, token t's starting at x + t * stride:
// channels i and i + dim / 2 turn by row t, column i of the tables
// [tokens, dim / 2] (computed on the host in double). Grid: (dim / 2, heads, tokens).
kernel void rope_half_rows(device float* x [[buffer(0)]],
                           const device float* cos_table [[buffer(1)]],
                           const device float* sin_table [[buffer(2)]],
                           constant HalfRopeParams& p [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x);
  const int head = int(gid.y);
  const int t = int(gid.z);
  const int half_dim = p.dim / 2;
  if (i >= half_dim || head >= p.heads) return;
  device float* v = x + t * p.stride + head * p.dim;
  const float c = cos_table[t * half_dim + i];
  const float s = sin_table[t * half_dim + i];
  const float a = v[i];
  const float b = v[i + half_dim];
  v[i] = a * c - b * s;
  v[i + half_dim] = b * c + a * s;
}

// erf through Numerical Recipes' Chebyshev fit of erfc (fractional error
// below 1.2e-7); the Metal library has no erf.
inline float erf_approx(float x) {
  const float z = fabs(x);
  const float t = 1.0f / (1.0f + 0.5f * z);
  const float poly =
      -1.26551223f +
      t * (1.00002368f +
           t * (0.37409196f +
                t * (0.09678418f +
                     t * (-0.18628806f +
                          t * (0.27886807f +
                               t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
  const float erfc_z = t * precise::exp(-z * z + poly);
  return x >= 0.0f ? 1.0f - erfc_z : erfc_z - 1.0f;
}

// Qwen2-VL's QuickGELU, x * sigmoid(1.702 x).
kernel void quick_gelu_inplace(device float* x [[buffer(0)]],
                               constant ElementwiseParams& p [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) x[gid] = x[gid] / (1.0f + precise::exp(-1.702f * x[gid]));
}

kernel void gelu_erf_inplace(device float* x [[buffer(0)]],
                             constant ElementwiseParams& p [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
  if (int(gid) < p.count) x[gid] = 0.5f * x[gid] * (1.0f + erf_approx(x[gid] * 0.70710678f));
}

// ---------------------------------------------------------------------------
// Qwen2.5-VL's windowed attention. Patches arrive window by window, so each
// window's keys are consecutive rows. Threadgroup (head, t) attends patch
// t's query over rows window[2t] to window[2t + 1] of qkv (each row
// [q | k | v], `stride` floats) and writes out[t, head * dim ...]. A window
// holds at most kMaxWindowKeys patches (the host checks).
// ---------------------------------------------------------------------------

struct WindowAttentionParams {
  int dim;     // head size
  int stride;  // floats per qkv row
  int hidden;  // q's, k's and v's width, and out's
  float scale;
};

constant constexpr int kMaxWindowKeys = 1024;

kernel void window_attention(const device float* qkv [[buffer(0)]],
                             const device int* window [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant WindowAttentionParams& p [[buffer(3)]],
                             uint2 tgp [[threadgroup_position_in_grid]],
                             uint2 tid2 [[thread_position_in_threadgroup]],
                             uint2 tpg2 [[threads_per_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  threadgroup float scores[kMaxWindowKeys];
  const int head = int(tgp.x);
  const int t = int(tgp.y);
  const int tid = int(tid2.x);
  const int tpg = int(tpg2.x);
  const uint simdgroups = (uint(tpg) + 31) / 32;
  const int begin = window[2 * t];
  const int count = window[2 * t + 1] - begin;
  const device float* q = qkv + t * p.stride + head * p.dim;
  float m = -INFINITY;
  for (int j = tid; j < count; j += tpg) {
    const device float* k = qkv + (begin + j) * p.stride + p.hidden + head * p.dim;
    float dot = 0.0f;
    for (int d = 0; d < p.dim; ++d) dot += q[d] * k[d];
    scores[j] = dot * p.scale;
    m = max(m, scores[j]);
  }
  m = threadgroup_max(m, scratch, sg, lane, simdgroups);
  float sum = 0.0f;
  for (int j = tid; j < count; j += tpg) {
    const float e = precise::exp(scores[j] - m);
    scores[j] = e;
    sum += e;
  }
  // threadgroup_sum's barriers also publish every thread's scores.
  const float inv = 1.0f / threadgroup_sum(sum, scratch, sg, lane, simdgroups);
  const device float* v = qkv + begin * p.stride + 2 * p.hidden + head * p.dim;
  device float* o = out + t * p.hidden + head * p.dim;
  for (int d = tid; d < p.dim; d += tpg) {
    float acc = 0.0f;
    for (int j = 0; j < count; ++j) acc += scores[j] * v[j * p.stride + d];
    o[d] = acc * inv;
  }
}
