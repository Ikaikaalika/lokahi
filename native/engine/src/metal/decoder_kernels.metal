// Lokahi Metal kernels for dense decoders (Gemma 3 and the Llama family).
//
// Activations are float32. Linear weights are MLX affine-quantized codes with
// BF16 or F16 per-group scales and biases. Quantization parameters are
// function constants so each pipeline is specialized at load time.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant int QBITS [[function_constant(0)]];
constant int QGROUP [[function_constant(1)]];
constant bool QSCALE_BF16 [[function_constant(2)]];

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

// The gated MLP's activation: SiLU when `use_silu`, else tanh GELU.
inline float glu(float gate, float up, int use_silu) {
  return (use_silu != 0 ? silu(gate) : gelu_tanh(gate)) * up;
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

// x += r, with r first normalized by w_post when post_norm (sandwich norms);
// then, if has_next, h = norm(x) * (offset + w_next).
struct ResidualNormParams {
  int cols;
  float eps;
  float offset;
  int post_norm;
  int has_next;
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
      const float v = x[i] + r[i] * inv_r * (p.offset + w_post[i]);
      x[i] = v;
      xs += v * v;
    }
  } else {
    for (int i = int(tid); i < p.cols; i += int(tpg)) {
      const float v = x[i] + r[i];
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
    float xsum = 0.0f;
    for (int j = 0; j < 16; ++j) xsum += xv[j];
    const int g = k0 / QGROUP;
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
// and k, and KV-cache writes. Grid: threadgroups (heads + kv_heads,
// n_tokens); head_dim / 2 threads each.
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
};

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
  const float angle = float(pos) * inv_freq[i];
  const float cs = precise::cos(angle);
  const float sn = precise::sin(angle);
  const float ra = na * cs - nb * sn;
  const float rb = nb * cs + na * sn;
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
  vd[i] = vs[i] + (p.qkv_bias != 0 ? vb[i] : 0.0f);
  vd[i + half_dim] = vs[i + half_dim] + (p.qkv_bias != 0 ? vb[i + half_dim] : 0.0f);
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
};

template <int G, int DPL>
[[kernel]] void attn_partial(const device float* q [[buffer(0)]],
                             const device float* k_cache [[buffer(1)]],
                             const device float* v_cache [[buffer(2)]],
                             device float* part_o [[buffer(3)]],
                             device float* part_ml [[buffer(4)]],
                             constant AttnParams& p [[buffer(5)]],
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
  const int kb = lo + blk * p.block;
  const int ke = min(kb + p.block, pos + 1);

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
// (Qwen 2.5 uses 6 and 7, Qwen 3 14B uses 5); head_dim is 32 * DPL.
#define LOKAHI_ATTN_ALL_DPL(G) LOKAHI_ATTN(G, 2) LOKAHI_ATTN(G, 4) LOKAHI_ATTN(G, 8)
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
};

// Grid: threadgroups (heads, n_tokens), head_dim threads each.
kernel void attn_reduce(const device float* part_o [[buffer(0)]],
                        const device float* part_ml [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant ReduceParams& p [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint d = tid2.x;
  const int head = int(tgp.x);
  const int t = int(tgp.y);
  const int base = (t * p.heads + head) * p.n_blocks;
  float m_max = -INFINITY;
  for (int b = 0; b < p.n_blocks; ++b) {
    if (part_ml[2 * (base + b) + 1] > 0.0f) m_max = max(m_max, part_ml[2 * (base + b)]);
  }
  float numer = 0.0f;
  float denom = 0.0f;
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
