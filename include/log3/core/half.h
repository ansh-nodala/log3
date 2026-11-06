// log3/core/half.h
//
// Half-precision floating-point conversions and utilities (f16 and bf16).
//
// Design notes:
//   * types.h defines f16 and bf16 as 2-byte storage structs so tensor headers
//     stay lightweight. All scalar conversions, constants, and batch helpers
//     live here.
//   * Pure integer bit-manipulation with round-to-nearest-even (IEEE 754
//     default). Does not depend on FPU rounding modes or flush-to-zero (FTZ)
//     control registers, guaranteeing bit-identical results across x86-64 and
//     ARM64 targets.
//   * Subnormals, signed zeros, infinities, and NaNs are handled accurately;
//     signaling NaNs are quieted on conversion.
//   * Vectorized bulk conversions (AVX2/AVX-512 F16C and ARM64 NEON FP16) live
//     in the SIMD kernel layer; the array helpers here provide the portable
//     baseline and tail-loop fallback.

#pragma once

#include <cstring>

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/types.h"

namespace l3 {

// -----------------------------------------------------------------------------
// Bit-cast helpers (compile to a single register move on GCC, Clang, and MSVC)
// -----------------------------------------------------------------------------
L3_FORCE_INLINE u32 f32_to_bits(f32 v) noexcept {
  u32 bits = 0;
  std::memcpy(&bits, &v, sizeof(u32));
  return bits;
}

L3_FORCE_INLINE f32 f32_from_bits(u32 bits) noexcept {
  f32 v = 0.0f;
  std::memcpy(&v, &bits, sizeof(f32));
  return v;
}

L3_FORCE_INLINE u64 f64_to_bits(f64 v) noexcept {
  u64 bits = 0;
  std::memcpy(&bits, &v, sizeof(u64));
  return bits;
}

L3_FORCE_INLINE f64 f64_from_bits(u64 bits) noexcept {
  f64 v = 0.0;
  std::memcpy(&v, &bits, sizeof(f64));
  return v;
}

// -----------------------------------------------------------------------------
// Useful constants
// -----------------------------------------------------------------------------
inline constexpr f16 kF16Zero = f16::from_bits(0x0000u);
inline constexpr f16 kF16NegZero = f16::from_bits(0x8000u);
inline constexpr f16 kF16One = f16::from_bits(0x3c00u);
inline constexpr f16 kF16NegOne = f16::from_bits(0xbc00u);
inline constexpr f16 kF16Max = f16::from_bits(0x7bffu);      // 65504.0
inline constexpr f16 kF16MinNorm = f16::from_bits(0x0400u);  // 2^-14
inline constexpr f16 kF16Infinity = f16::from_bits(0x7c00u);
inline constexpr f16 kF16NegInfinity = f16::from_bits(0xfc00u);
inline constexpr f16 kF16NaN = f16::from_bits(0x7e00u);

inline constexpr bf16 kBF16Zero = bf16::from_bits(0x0000u);
inline constexpr bf16 kBF16NegZero = bf16::from_bits(0x8000u);
inline constexpr bf16 kBF16One = bf16::from_bits(0x3f80u);
inline constexpr bf16 kBF16NegOne = bf16::from_bits(0xbf80u);
inline constexpr bf16 kBF16Max = bf16::from_bits(0x7f7fu);
inline constexpr bf16 kBF16MinNorm = bf16::from_bits(0x0080u);
inline constexpr bf16 kBF16Infinity = bf16::from_bits(0x7f80u);
inline constexpr bf16 kBF16NegInfinity = bf16::from_bits(0xff80u);
inline constexpr bf16 kBF16NaN = bf16::from_bits(0x7fc0u);

// -----------------------------------------------------------------------------
// Classification predicates (no <cmath> dependency)
// -----------------------------------------------------------------------------
L3_FORCE_INLINE bool is_nan(f32 v) noexcept {
  return (f32_to_bits(v) & 0x7fffffffu) > 0x7f800000u;
}

L3_FORCE_INLINE bool is_inf(f32 v) noexcept {
  return (f32_to_bits(v) & 0x7fffffffu) == 0x7f800000u;
}

L3_FORCE_INLINE bool is_finite(f32 v) noexcept {
  return (f32_to_bits(v) & 0x7f800000u) != 0x7f800000u;
}

constexpr bool is_nan(f16 h) noexcept {
  return (h.bits & 0x7fffu) > 0x7c00u;
}

constexpr bool is_inf(f16 h) noexcept {
  return (h.bits & 0x7fffu) == 0x7c00u;
}

constexpr bool is_finite(f16 h) noexcept {
  return (h.bits & 0x7c00u) != 0x7c00u;
}

constexpr bool is_zero(f16 h) noexcept {
  return (h.bits & 0x7fffu) == 0u;
}

constexpr bool is_nan(bf16 b) noexcept {
  return (b.bits & 0x7fffu) > 0x7f80u;
}

constexpr bool is_inf(bf16 b) noexcept {
  return (b.bits & 0x7fffu) == 0x7f80u;
}

constexpr bool is_finite(bf16 b) noexcept {
  return (b.bits & 0x7f80u) != 0x7f80u;
}

constexpr bool is_zero(bf16 b) noexcept {
  return (b.bits & 0x7fffu) == 0u;
}

// -----------------------------------------------------------------------------
// bf16 <-> f32 conversions
// -----------------------------------------------------------------------------

// Exact: bf16 is the top 16 bits of an IEEE 754 f32.
L3_FORCE_INLINE f32 bf16_to_f32(bf16 b) noexcept {
  return f32_from_bits(static_cast<u32>(b.bits) << 16);
}

// Round-to-nearest-even. Preserves NaNs (quieting signaling NaNs).
L3_FORCE_INLINE bf16 f32_to_bf16(f32 v) noexcept {
  u32 bits = f32_to_bits(v);
  if (L3_UNLIKELY((bits & 0x7fffffffu) > 0x7f800000u)) {
    // Force quiet NaN bit so payload truncation cannot turn a NaN into Inf.
    return bf16::from_bits(static_cast<u16>((bits >> 16) | 0x0040u));
  }
  const u32 lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return bf16::from_bits(static_cast<u16>(bits >> 16));
}

// Fast truncation (round toward zero); useful only when NaN is impossible.
L3_FORCE_INLINE bf16 f32_to_bf16_trunc(f32 v) noexcept {
  return bf16::from_bits(static_cast<u16>(f32_to_bits(v) >> 16));
}

// -----------------------------------------------------------------------------
// f16 <-> f32 conversions (IEEE 754 binary16, round-to-nearest-even)
// -----------------------------------------------------------------------------
inline f32 f16_to_f32(f16 h) noexcept {
  const u32 w = static_cast<u32>(h.bits);
  const u32 sign = (w & 0x8000u) << 16;
  u32 exp = (w >> 10) & 0x1fu;
  u32 mant = w & 0x03ffu;

  if (L3_LIKELY(exp != 0u && exp != 31u)) {
    // Normal number: rebias exponent from 15 to 127 (+112).
    return f32_from_bits(sign | ((exp + 112u) << 23) | (mant << 13));
  }

  if (exp == 0u) {
    if (mant == 0u) return f32_from_bits(sign);
    // Subnormal: shift mantissa until the hidden bit (0x0400) is set.
    exp = 113u;
    while ((mant & 0x0400u) == 0u) {
      mant <<= 1;
      --exp;
    }
    mant &= 0x03ffu;
    return f32_from_bits(sign | (exp << 23) | (mant << 13));
  }

  // Inf or NaN (exp == 31). Quiet any signaling NaN.
  const u32 nan_quiet = mant != 0u ? 0x00400000u : 0u;
  return f32_from_bits(sign | 0x7f800000u | (mant << 13) | nan_quiet);
}

inline f16 f32_to_f16(f32 v) noexcept {
  const u32 bits = f32_to_bits(v);
  const u32 sign = (bits >> 16) & 0x8000u;
  const u32 abs_bits = bits & 0x7fffffffu;

  // NaN or Inf in f32.
  if (L3_UNLIKELY(abs_bits >= 0x7f800000u)) {
    if (abs_bits > 0x7f800000u) {
      // NaN: keep high payload bits and set the quiet bit (0x0200).
      return f16::from_bits(
          static_cast<u16>(sign | 0x7e00u | ((abs_bits >> 13) & 0x01ffu)));
    }
    return f16::from_bits(static_cast<u16>(sign | 0x7c00u));
  }

  // Overflow: >= 65520.0 (0x477fe000) rounds up to infinity under RNE.
  if (L3_UNLIKELY(abs_bits >= 0x477fe000u)) {
    return f16::from_bits(static_cast<u16>(sign | 0x7c00u));
  }

  // Normal f16 range: >= 2^-14 (0x38800000).
  if (L3_LIKELY(abs_bits >= 0x38800000u)) {
    u32 biased = abs_bits - 0x38000000u;
    const u32 lsb = (biased >> 13) & 1u;
    biased += 0x0fffu + lsb;
    return f16::from_bits(static_cast<u16>(sign | (biased >> 13)));
  }

  // Too small even for subnormal f16 (< 2^-25): rounds to signed zero.
  if (abs_bits < 0x33000000u) {
    return f16::from_bits(static_cast<u16>(sign));
  }

  // Subnormal f16 (2^-25 <= |v| < 2^-14).
  const u32 exp = abs_bits >> 23;
  const u32 mant = (abs_bits & 0x007fffffu) | 0x00800000u;
  const u32 shift = 126u - exp;  // in [14, 24]
  const u32 lsb = (mant >> shift) & 1u;
  const u32 round_bias = (1u << (shift - 1u)) - 1u + lsb;
  const u16 sub = static_cast<u16>((mant + round_bias) >> shift);
  return f16::from_bits(static_cast<u16>(sign | sub));
}

// Direct f16 <-> bf16 goes through f32.
L3_FORCE_INLINE bf16 f16_to_bf16(f16 h) noexcept {
  return f32_to_bf16(f16_to_f32(h));
}

L3_FORCE_INLINE f16 bf16_to_f16(bf16 b) noexcept {
  return f32_to_f16(bf16_to_f32(b));
}

// -----------------------------------------------------------------------------
// Contiguous buffer conversions (portable baseline)
// -----------------------------------------------------------------------------
inline void convert_f32_to_f16(const f32* L3_RESTRICT src,
                               f16* L3_RESTRICT dst, Index count) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK(count == 0 || (src != nullptr && dst != nullptr));
  for (Index i = 0; i < count; ++i) {
    dst[i] = f32_to_f16(src[i]);
  }
}

inline void convert_f16_to_f32(const f16* L3_RESTRICT src,
                               f32* L3_RESTRICT dst, Index count) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK(count == 0 || (src != nullptr && dst != nullptr));
  for (Index i = 0; i < count; ++i) {
    dst[i] = f16_to_f32(src[i]);
  }
}

inline void convert_f32_to_bf16(const f32* L3_RESTRICT src,
                                bf16* L3_RESTRICT dst, Index count) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK(count == 0 || (src != nullptr && dst != nullptr));
  for (Index i = 0; i < count; ++i) {
    dst[i] = f32_to_bf16(src[i]);
  }
}

inline void convert_bf16_to_f32(const bf16* L3_RESTRICT src,
                                f32* L3_RESTRICT dst, Index count) noexcept {
  L3_DCHECK_GE(count, 0);
  L3_DCHECK(count == 0 || (src != nullptr && dst != nullptr));
  for (Index i = 0; i < count; ++i) {
    dst[i] = bf16_to_f32(src[i]);
  }
}

}  // namespace l3
