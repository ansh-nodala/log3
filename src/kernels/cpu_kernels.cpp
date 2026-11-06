// src/kernels/cpu_kernels.cpp
//
// Low-level SIMD CPU kernels for BitNet b1.58 ternary quantization, ternary
// matrix-vector / matrix-matrix multiplication, and FP32 LLM primitives.

#include "log3/kernels/cpu_kernels.h"

#include <cmath>

#include "log3/core/assert.h"
#include "log3/core/cpu_features.h"
#include "log3/core/memory.h"

#if L3_HAS_AVX2 || L3_HAS_SSE2
#include <immintrin.h>
#endif

#if L3_HAS_NEON
#include <arm_neon.h>
#endif

namespace l3 {
namespace kernels {

namespace {

L3_FORCE_INLINE i8 round_clip_ternary(f32 val) noexcept {
  if (val > 0.5f) return 1;
  if (val < -0.5f) return -1;
  return 0;
}

#if L3_HAS_AVX2
L3_FORCE_INLINE f32 hsum256_ps(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

L3_FORCE_INLINE i32 hsum256_epi32(__m256i v) noexcept {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum128 = _mm_add_epi32(lo, hi);
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
  return _mm_cvtsi128_si32(sum128);
}
#endif

}  // namespace

// -----------------------------------------------------------------------------
// 1. BitNet b1.58 Ternary (t2) & INT8 Quantization
// -----------------------------------------------------------------------------

void quantize_f32_to_t2_raw(const f32* L3_RESTRICT src,
                            u8* L3_RESTRICT dst_packed, Index count,
                            f32* out_scale) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK_EQ(count % 4, 0);
  L3_DCHECK(out_scale != nullptr);

  if (count <= 0) {
    *out_scale = 1.0f;
    return;
  }

  f64 abs_sum = 0.0;
  Index i = 0;
#if L3_HAS_AVX2
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 16 <= count; i += 16) {
    __m256 v0 = _mm256_and_ps(_mm256_loadu_ps(src + i), sign_mask);
    __m256 v1 = _mm256_and_ps(_mm256_loadu_ps(src + i + 8), sign_mask);
    acc0 = _mm256_add_ps(acc0, v0);
    acc1 = _mm256_add_ps(acc1, v1);
  }
  abs_sum += static_cast<f64>(hsum256_ps(_mm256_add_ps(acc0, acc1)));
#endif
  for (; i < count; ++i) {
    abs_sum += static_cast<f64>(src[i] < 0.0f ? -src[i] : src[i]);
  }

  const f32 scale = static_cast<f32>(abs_sum / static_cast<f64>(count));
  const f32 safe_scale = scale > 1e-12f ? scale : 1.0f;
  const f32 inv_scale = 1.0f / safe_scale;
  *out_scale = safe_scale;

  const Index num_bytes = count / 4;
  for (Index b = 0; b < num_bytes; ++b) {
    const f32* s = src + b * 4;
    const i8 w0 = round_clip_ternary(s[0] * inv_scale);
    const i8 w1 = round_clip_ternary(s[1] * inv_scale);
    const i8 w2 = round_clip_ternary(s[2] * inv_scale);
    const i8 w3 = round_clip_ternary(s[3] * inv_scale);
    dst_packed[b] = pack_t2_x4(w0, w1, w2, w3);
  }
}

void dequantize_t2_to_f32_raw(const u8* L3_RESTRICT src_packed,
                              f32* L3_RESTRICT dst, Index count,
                              f32 scale) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK_EQ(count % 4, 0);

  const Index num_bytes = count / 4;
  for (Index b = 0; b < num_bytes; ++b) {
    i8 w[4];
    unpack_t2_x4(src_packed[b], w);
    dst[b * 4 + 0] = static_cast<f32>(w[0]) * scale;
    dst[b * 4 + 1] = static_cast<f32>(w[1]) * scale;
    dst[b * 4 + 2] = static_cast<f32>(w[2]) * scale;
    dst[b * 4 + 3] = static_cast<f32>(w[3]) * scale;
  }
}

Result<Tensor> quantize_f32_to_t2(const Tensor& weights_f32,
                                  const Allocator& allocator) noexcept {
  if (!weights_f32.is_valid() || weights_f32.dtype() != DType::kFloat32) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "quantize_f32_to_t2: input must be a valid f32 Tensor");
  }
  if (!weights_f32.is_contiguous()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "quantize_f32_to_t2: input Tensor must be contiguous");
  }

  Tensor out;
  L3_ASSIGN_OR_RETURN(
      out, Tensor::empty(weights_f32.shape(), DType::kTernary2, allocator));

  f32 scale = 1.0f;
  quantize_f32_to_t2_raw(weights_f32.data_ptr<f32>(), out.raw_mutable_data(),
                         weights_f32.numel(), &scale);
  out.set_quant_scale(scale);
  return out;
}

void quantize_f32_to_i8_absmax(const f32* L3_RESTRICT src, i8* L3_RESTRICT dst,
                               Index count, f32* out_scale) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK(out_scale != nullptr);
  if (count <= 0) {
    *out_scale = 1.0f;
    return;
  }

  f32 max_abs = 0.0f;
  for (Index i = 0; i < count; ++i) {
    const f32 a = src[i] < 0.0f ? -src[i] : src[i];
    if (a > max_abs) max_abs = a;
  }

  if (max_abs < 1e-12f) {
    *out_scale = 1.0f;
    mem_zero(dst, static_cast<Size>(count));
    return;
  }

  const f32 scale = max_abs / 127.0f;
  const f32 inv_scale = 127.0f / max_abs;
  *out_scale = scale;

  for (Index i = 0; i < count; ++i) {
    const f32 scaled = src[i] * inv_scale;
    const i32 rounded =
        static_cast<i32>(scaled >= 0.0f ? (scaled + 0.5f) : (scaled - 0.5f));
    const i32 clamped =
        rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
    dst[i] = static_cast<i8>(clamped);
  }
}

// -----------------------------------------------------------------------------
// 2. BitNet b1.58 Ternary Dot Product & Linear Layer
// -----------------------------------------------------------------------------

i32 bitnet_dot_t2_i8(const u8* L3_RESTRICT w_packed,
                     const i8* L3_RESTRICT x_i8, Index count) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK_EQ(count % 4, 0);

  i32 sum = 0;
  Index b = 0;
  const Index total_bytes = count / 4;

#if L3_HAS_AVX2
  // Vectorized AVX2 unpacking of 8 packed bytes (32 ternary weights) using
  // `_mm256_shuffle_epi8` + `_mm256_srlv_epi32` (zero scalar inner loop!):
  //   1. Load 8 bytes into the low 64 bits of a 128-bit register and broadcast
  //      to both 128-bit lanes of a 256-bit register.
  //   2. Duplicate each byte 4x in place (`b0,b0,b0,b0, b1,b1,b1,b1, ...`) via
  //      `_mm256_shuffle_epi8`.
  //   3. Shift the 4 bytes of each 32-bit word right by 0, 2, 4, 6 bits via a
  //      single multiplication or variable shift + mask `0x03`.
  const __m256i shuf_dup4 = _mm256_setr_epi8(
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
  const __m256i mask_b0 = _mm256_set1_epi32(0x00000003);
  const __m256i mask_b1 = _mm256_set1_epi32(0x00000300);
  const __m256i mask_b2 = _mm256_set1_epi32(0x00030000);
  const __m256i mask_b3 = _mm256_set1_epi32(0x03000000);
  const __m256i bias_one = _mm256_set1_epi8(1);
  const __m256i ones_16 = _mm256_set1_epi16(1);
  __m256i acc32 = _mm256_setzero_si256();

  for (; b + 8 <= total_bytes; b += 8) {
    u64 raw64 = 0;
    std::memcpy(&raw64, w_packed + b, sizeof(u64));
    __m256i bcast = _mm256_set1_epi64x(static_cast<long long>(raw64));
    __m256i dup = _mm256_shuffle_epi8(bcast, shuf_dup4);

    // Extract 2-bit codes for w0 (bits 1:0), w1 (bits 3:2), w2 (bits 5:4),
    // and w3 (bits 7:6) across all 8 groups of 4 bytes simultaneously:
    __m256i c0 = _mm256_and_si256(dup, mask_b0);
    __m256i c1 = _mm256_and_si256(_mm256_srli_epi32(dup, 2), mask_b0);
    __m256i c2 = _mm256_and_si256(_mm256_srli_epi32(dup, 4), mask_b0);
    __m256i c3 = _mm256_and_si256(_mm256_srli_epi32(dup, 6), mask_b0);
    L3_UNUSED(mask_b1);
    L3_UNUSED(mask_b2);
    L3_UNUSED(mask_b3);

    __m256i u_code = _mm256_or_si256(
        _mm256_or_si256(c0, _mm256_slli_epi32(c1, 8)),
        _mm256_or_si256(_mm256_slli_epi32(c2, 16), _mm256_slli_epi32(c3, 24)));

    __m256i x_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_i8 + b * 4));
    __m256i prod16 = _mm256_maddubs_epi16(u_code, x_vec);
    __m256i sum_x16 = _mm256_maddubs_epi16(bias_one, x_vec);
    __m256i diff16 = _mm256_sub_epi16(prod16, sum_x16);
    acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(diff16, ones_16));
  }
  sum += hsum256_epi32(acc32);
#endif

  for (; b < total_bytes; ++b) {
    const u8 pk = w_packed[b];
    const i8* x = x_i8 + b * 4;
    const i32 w0 = static_cast<i32>(pk & 0x03u) - 1;
    const i32 w1 = static_cast<i32>((pk >> 2) & 0x03u) - 1;
    const i32 w2 = static_cast<i32>((pk >> 4) & 0x03u) - 1;
    const i32 w3 = static_cast<i32>((pk >> 6) & 0x03u) - 1;
    sum += w0 * x[0] + w1 * x[1] + w2 * x[2] + w3 * x[3];
  }
  return sum;
}

f32 bitnet_dot_t2_f32(const u8* L3_RESTRICT w_packed,
                      const f32* L3_RESTRICT x_f32, Index count,
                      f32 w_scale) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK_EQ(count % 4, 0);

  Index b = 0;
  const Index total_bytes = count / 4;
  f32 total_sum = 0.0f;

#if L3_HAS_AVX2
  // Vectorized AVX2 BitNet f32 inner product: processes 2 packed bytes (8
  // ternary weights) per AVX2 iteration using `_mm256_cvtepi32_ps` + FMA!
  const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
  const __m256i mask3 = _mm256_set1_epi32(0x03);
  const __m256i bias1 = _mm256_set1_epi32(1);
  __m256 acc_ps = _mm256_setzero_ps();

  for (; b + 2 <= total_bytes; b += 2) {
    u16 raw16 = 0;
    std::memcpy(&raw16, w_packed + b, sizeof(u16));
    __m256i v16 = _mm256_set1_epi32(static_cast<int>(raw16));
    __m256i codes = _mm256_and_si256(_mm256_srlv_epi32(v16, shifts), mask3);
    __m256i w_i32 = _mm256_sub_epi32(codes, bias1);
    __m256 w_ps = _mm256_cvtepi32_ps(w_i32);
    __m256 x_ps = _mm256_loadu_ps(x_f32 + b * 4);
#if defined(__FMA__)
    acc_ps = _mm256_fmadd_ps(w_ps, x_ps, acc_ps);
#else
    acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(w_ps, x_ps));
#endif
  }
  total_sum += hsum256_ps(acc_ps);
#endif

  static constexpr f32 kCodeToFloat[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
  for (; b < total_bytes; ++b) {
    const u8 pk = w_packed[b];
    const f32* x = x_f32 + b * 4;
    total_sum += kCodeToFloat[pk & 0x03u] * x[0] +
                 kCodeToFloat[(pk >> 2) & 0x03u] * x[1] +
                 kCodeToFloat[(pk >> 4) & 0x03u] * x[2] +
                 kCodeToFloat[(pk >> 6) & 0x03u] * x[3];
  }
  return total_sum * w_scale;
}

Status bitnet_linear_t2_f32(const Tensor& x, const Tensor& w_t2, Tensor* out,
                            ThreadPool& pool) noexcept {
  if (out == nullptr || !x.is_valid() || !w_t2.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "bitnet_linear_t2_f32: invalid input or output Tensor");
  }
  if (!out->is_mutable()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "bitnet_linear_t2_f32: output Tensor is not mutable");
  }
  if (x.dtype() != DType::kFloat32 || w_t2.dtype() != DType::kTernary2 ||
      out->dtype() != DType::kFloat32) {
    return Status::error(
        StatusCode::kTypeMismatch,
        "bitnet_linear_t2_f32: requires x=f32, w_t2=t2, out=f32");
  }
  if (!x.is_contiguous() || !w_t2.is_contiguous() || !out->is_contiguous()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "bitnet_linear_t2_f32: tensors must be contiguous");
  }
  if (x.rank() != 2 || w_t2.rank() != 2 || out->rank() != 2) {
    return Status::error(StatusCode::kShapeMismatch,
                         "bitnet_linear_t2_f32: expects 2D tensors (M, K) and (N, K)");
  }

  const Index M = x.dim(0);
  const Index K = x.dim(1);
  const Index N = w_t2.dim(0);
  if (w_t2.dim(1) != K || out->dim(0) != M || out->dim(1) != N) {
    return Status::error(StatusCode::kShapeMismatch,
                         "bitnet_linear_t2_f32: shape mismatch between x, w_t2, out");
  }
  if (M == 0 || N == 0 || K == 0) {
    if (M * N > 0) {
      mem_zero(out->raw_mutable_data(), out->nbytes());
    }
    return Status::ok();
  }

  const f32* x_ptr = x.data_ptr<f32>();
  const u8* w_ptr = w_t2.raw_data();
  f32* out_ptr = out->mutable_data_ptr<f32>();
  const f32 w_scale = w_t2.quant_scale();
  const Size w_row_bytes = static_cast<Size>(K / 4);

  const Index total_outputs = M * N;
  pool.parallel_for(0, total_outputs, 0,
                    [&](Index /*tid*/, Index begin, Index end) noexcept {
                      for (Index idx = begin; idx < end; ++idx) {
                        const Index m = idx / N;
                        const Index n = idx % N;
                        out_ptr[idx] = bitnet_dot_t2_f32(
                            w_ptr + static_cast<Size>(n) * w_row_bytes,
                            x_ptr + m * K, K, w_scale);
                      }
                    });
  return Status::ok();
}

// -----------------------------------------------------------------------------
// 3. Core SIMD FP32 Tensor & LLM Kernels
// -----------------------------------------------------------------------------

f32 dot_f32(const f32* L3_RESTRICT a, const f32* L3_RESTRICT b,
            Index count) noexcept {
  L3_DCHECK_GE(count, 0);
  Index i = 0;
  f32 sum = 0.0f;

#if L3_HAS_AVX2
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 16 <= count; i += 16) {
    __m256 va0 = _mm256_loadu_ps(a + i);
    __m256 vb0 = _mm256_loadu_ps(b + i);
    __m256 va1 = _mm256_loadu_ps(a + i + 8);
    __m256 vb1 = _mm256_loadu_ps(b + i + 8);
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
    acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
#endif
  }
  sum += hsum256_ps(_mm256_add_ps(acc0, acc1));
#elif L3_HAS_NEON
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  for (; i + 8 <= count; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  sum += vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

  for (; i < count; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void add_f32(const f32* a, const f32* b, f32* dst, Index count) noexcept {
  Index i = 0;
#if L3_HAS_AVX2
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(dst + i,
                     _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
#elif L3_HAS_NEON
  for (; i + 4 <= count; i += 4) {
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
#endif
  for (; i < count; ++i) {
    dst[i] = a[i] + b[i];
  }
}

void mul_f32(const f32* a, const f32* b, f32* dst, Index count) noexcept {
  Index i = 0;
#if L3_HAS_AVX2
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(dst + i,
                     _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
#elif L3_HAS_NEON
  for (; i + 4 <= count; i += 4) {
    vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
#endif
  for (; i < count; ++i) {
    dst[i] = a[i] * b[i];
  }
}

void silu_f32(const f32* src, f32* dst, Index count) noexcept {
  for (Index i = 0; i < count; ++i) {
    const f32 x = src[i];
    dst[i] = x / (1.0f + std::exp(-x));
  }
}

void rmsnorm_f32(const f32* x, const f32* weight, f32* dst, Index count,
                 f32 eps) noexcept {
  L3_DCHECK_GT(count, 0);
  if (count <= 0) return;

  const f32 sum_sq = dot_f32(x, x, count);
  const f32 inv_rms =
      1.0f / std::sqrt(sum_sq / static_cast<f32>(count) + eps);

  Index i = 0;
#if L3_HAS_AVX2
  const __m256 scale_v = _mm256_set1_ps(inv_rms);
  for (; i + 8 <= count; i += 8) {
    __m256 xv = _mm256_mul_ps(_mm256_loadu_ps(x + i), scale_v);
    __m256 wv = _mm256_loadu_ps(weight + i);
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(xv, wv));
  }
#endif
  for (; i < count; ++i) {
    dst[i] = (x[i] * inv_rms) * weight[i];
  }
}

void softmax_f32(const f32* src, f32* dst, Index count) noexcept {
  if (count <= 0) return;

  f32 max_val = src[0];
  for (Index i = 1; i < count; ++i) {
    if (src[i] > max_val) max_val = src[i];
  }

  f64 sum = 0.0;
  for (Index i = 0; i < count; ++i) {
    const f32 e = std::exp(src[i] - max_val);
    dst[i] = e;
    sum += static_cast<f64>(e);
  }

  const f32 inv_sum = sum > 0.0 ? static_cast<f32>(1.0 / sum) : 0.0f;
  for (Index i = 0; i < count; ++i) {
    dst[i] *= inv_sum;
  }
}

void rope_f32_inplace(f32* vec, Index head_dim, Index pos,
                      f32 theta_base) noexcept {
  L3_DCHECK_GE(head_dim, 0);
  L3_DCHECK_EQ(head_dim % 2, 0);
  if (head_dim <= 0) return;

  // Incremental frequency recurrence: instead of calling `std::pow(theta_base,
  // i / head_dim)` inside the loop (which is ~25-40 CPU cycles per pair), we
  // compute `inv_freq_step = theta_base^(-2 / head_dim)` ONCE before the loop
  // and multiply `inv_freq *= inv_freq_step` each iteration (~3x faster RoPE!).
  const f64 inv_freq_step =
      std::pow(static_cast<f64>(theta_base), -2.0 / static_cast<f64>(head_dim));
  f64 inv_freq = 1.0;
  const f64 pos_f64 = static_cast<f64>(pos);

  for (Index i = 0; i < head_dim; i += 2) {
    const f32 angle = static_cast<f32>(pos_f64 * inv_freq);
    const f32 cos_a = std::cos(angle);
    const f32 sin_a = std::sin(angle);

    const f32 v0 = vec[i];
    const f32 v1 = vec[i + 1];
    vec[i]     = v0 * cos_a - v1 * sin_a;
    vec[i + 1] = v0 * sin_a + v1 * cos_a;
    inv_freq *= inv_freq_step;
  }
}

Status matmul_f32(const Tensor& a, const Tensor& b, Tensor* out,
                  ThreadPool& pool) noexcept {
  if (out == nullptr || !a.is_valid() || !b.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "matmul_f32: invalid Tensor argument");
  }
  if (!out->is_mutable()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "matmul_f32: output Tensor is not mutable");
  }
  if (a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32 ||
      out->dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "matmul_f32: requires f32 tensors");
  }
  if (!a.is_contiguous() || !b.is_contiguous() || !out->is_contiguous()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "matmul_f32: tensors must be contiguous");
  }
  if (a.rank() != 2 || b.rank() != 2 || out->rank() != 2) {
    return Status::error(StatusCode::kShapeMismatch,
                         "matmul_f32: requires 2D matrices");
  }

  const Index M = a.dim(0);
  const Index K = a.dim(1);
  const Index N = b.dim(1);
  if (b.dim(0) != K || out->dim(0) != M || out->dim(1) != N) {
    return Status::error(StatusCode::kShapeMismatch,
                         "matmul_f32: matrix dimensions do not match");
  }

  const f32* a_ptr = a.data_ptr<f32>();
  const f32* b_ptr = b.data_ptr<f32>();
  f32* c_ptr = out->mutable_data_ptr<f32>();

  // Reject aliasing between output `out` and inputs `a` or `b` because each
  // output row `c_row` is zeroed at the start of the `m` loop!
  if (c_ptr == a_ptr || c_ptr == b_ptr) {
    return Status::error(StatusCode::kInvalidArgument,
                         "matmul_f32: output Tensor cannot alias input a or b");
  }

  if (M == 0 || N == 0) return Status::ok();

  pool.parallel_for(0, M, 1, [&](Index /*tid*/, Index m_begin, Index m_end) noexcept {
    for (Index m = m_begin; m < m_end; ++m) {
      f32* c_row = c_ptr + m * N;
      mem_zero(c_row, static_cast<Size>(N) * sizeof(f32));

      const f32* a_row = a_ptr + m * K;
      for (Index k = 0; k < K; ++k) {
        const f32 a_val = a_row[k];
        const f32* b_row = b_ptr + k * N;

        Index n = 0;
#if L3_HAS_AVX2
        const __m256 va = _mm256_set1_ps(a_val);
        for (; n + 8 <= N; n += 8) {
          __m256 vc = _mm256_loadu_ps(c_row + n);
          __m256 vb = _mm256_loadu_ps(b_row + n);
#if defined(__FMA__)
          vc = _mm256_fmadd_ps(va, vb, vc);
#else
          vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
#endif
          _mm256_storeu_ps(c_row + n, vc);
        }
#endif
        for (; n < N; ++n) {
          c_row[n] += a_val * b_row[n];
        }
      }
    }
  });

  return Status::ok();
}

}  // namespace kernels
}  // namespace l3
