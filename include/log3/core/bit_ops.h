// log3/core/bit_ops.h
//
// Low-level bit manipulation, compiler intrinsics, endian byte-swapping, and
// unaligned little-endian memory load/store helpers.
//
// Why not <bit> (C++20) or manual loops:
//   * log3 targets C++17, where <bit> (std::popcount, std::countr_zero,
//     std::bit_ceil, std::byteswap) does not exist.
//   * Every function here maps to a single CPU instruction on x86-64 and ARM64
//     (TZCNT/BSF, LZCNT/BSR, POPCNT/CNT, BSWAP/REV).
//   * Model loaders (.gguf, .safetensors, .l3) and tokenizer files are stored
//     in little-endian byte order at arbitrary byte offsets inside MMapFile.
//     `read_le<T>` and `write_le<T>` perform safe, aliasing-clean (`memcpy`)
//     loads and stores that compile to a single unaligned `MOV` instruction on
//     x86-64 and ARM64.
//
// Compatibility:
//   * C++17, header-only, no exceptions, no RTTI.

#pragma once

#include <cstring>
#include <type_traits>

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/types.h"

#if L3_MSVC_ABI
#include <intrin.h>
#include <stdlib.h>
#endif

namespace l3 {

// -----------------------------------------------------------------------------
// Count trailing zeros (ctz)
//   Precondition: x != 0 (calling ctz on 0 is undefined in hardware intrinsics;
//   checked via L3_DCHECK in debug builds).
// -----------------------------------------------------------------------------
L3_FORCE_INLINE int ctz32(u32 x) noexcept {
  L3_DCHECK_NE(x, 0u);
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  unsigned long index = 0;
  ::_BitScanForward(&index, x);
  return static_cast<int>(index);
#else
  return __builtin_ctz(x);
#endif
}

L3_FORCE_INLINE int ctz64(u64 x) noexcept {
  L3_DCHECK_NE(x, 0ULL);
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
#if L3_64BIT
  unsigned long index = 0;
  ::_BitScanForward64(&index, x);
  return static_cast<int>(index);
#else
  const u32 lo = static_cast<u32>(x);
  if (lo != 0) return ctz32(lo);
  return 32 + ctz32(static_cast<u32>(x >> 32));
#endif
#else
  return __builtin_ctzll(x);
#endif
}

// -----------------------------------------------------------------------------
// Count leading zeros (clz)
//   Precondition: x != 0.
// -----------------------------------------------------------------------------
L3_FORCE_INLINE int clz32(u32 x) noexcept {
  L3_DCHECK_NE(x, 0u);
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  unsigned long index = 0;
  ::_BitScanReverse(&index, x);
  return 31 - static_cast<int>(index);
#else
  return __builtin_clz(x);
#endif
}

L3_FORCE_INLINE int clz64(u64 x) noexcept {
  L3_DCHECK_NE(x, 0ULL);
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
#if L3_64BIT
  unsigned long index = 0;
  ::_BitScanReverse64(&index, x);
  return 63 - static_cast<int>(index);
#else
  const u32 hi = static_cast<u32>(x >> 32);
  if (hi != 0) return clz32(hi);
  return 32 + clz32(static_cast<u32>(x));
#endif
#else
  return __builtin_clzll(x);
#endif
}

// -----------------------------------------------------------------------------
// Population count (number of set 1-bits)
// -----------------------------------------------------------------------------
L3_FORCE_INLINE int popcount32(u32 x) noexcept {
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  // Portable SWAR fallback on MSVC so it works even without /arch:AVX
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
#else
  return __builtin_popcount(x);
#endif
}

L3_FORCE_INLINE int popcount64(u64 x) noexcept {
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  x = x - ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  return static_cast<int>(
      (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#else
  return __builtin_popcountll(x);
#endif
}

// -----------------------------------------------------------------------------
// Power-of-two & log2 helpers (for hash tables, arenas, FFT/tiling blocks)
// -----------------------------------------------------------------------------

// Number of bits needed to represent x (0 -> 0, 1 -> 1, 2 -> 2, 7 -> 3).
L3_FORCE_INLINE int bit_width64(u64 x) noexcept {
  return x == 0 ? 0 : (64 - clz64(x));
}

// Floor(log2(x)) for x > 0.
L3_FORCE_INLINE int log2_floor64(u64 x) noexcept {
  L3_DCHECK_NE(x, 0ULL);
  return 63 - clz64(x);
}

// Smallest power of two >= x (for x == 0 or x == 1, returns 1).
// Returns 0 on overflow (when x > 2^63).
L3_FORCE_INLINE u64 next_power_of_two(u64 x) noexcept {
  if (x <= 1) return 1;
  if (L3_UNLIKELY(x > (1ULL << 63))) return 0;
  return 1ULL << (64 - clz64(x - 1));
}

// -----------------------------------------------------------------------------
// Byte-order reversal (BSWAP / REV)
// -----------------------------------------------------------------------------
L3_FORCE_INLINE u16 bswap16(u16 x) noexcept {
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  return ::_byteswap_ushort(x);
#else
  return __builtin_bswap16(x);
#endif
}

L3_FORCE_INLINE u32 bswap32(u32 x) noexcept {
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  return ::_byteswap_ulong(x);
#else
  return __builtin_bswap32(x);
#endif
}

L3_FORCE_INLINE u64 bswap64(u64 x) noexcept {
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
  return ::_byteswap_uint64(x);
#else
  return __builtin_bswap64(x);
#endif
}

// -----------------------------------------------------------------------------
// Unaligned little-endian memory load/store (zero UB, compiles to 1 MOV on
// x86-64 and ARM64). Used heavily by .gguf / .safetensors / .l3 parsers.
// -----------------------------------------------------------------------------
template <typename T>
L3_FORCE_INLINE T read_le(const void* src) noexcept {
  static_assert(std::is_trivially_copyable_v<T>,
                "read_le<T>: T must be trivially copyable.");
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                    sizeof(T) == 8,
                "read_le<T>: T must be 1, 2, 4, or 8 bytes wide.");
  T val{};
  std::memcpy(&val, src, sizeof(T));
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  if constexpr (sizeof(T) == 2) {
    u16 raw = 0;
    std::memcpy(&raw, &val, 2);
    raw = bswap16(raw);
    std::memcpy(&val, &raw, 2);
  } else if constexpr (sizeof(T) == 4) {
    u32 raw = 0;
    std::memcpy(&raw, &val, 4);
    raw = bswap32(raw);
    std::memcpy(&val, &raw, 4);
  } else if constexpr (sizeof(T) == 8) {
    u64 raw = 0;
    std::memcpy(&raw, &val, 8);
    raw = bswap64(raw);
    std::memcpy(&val, &raw, 8);
  }
#endif
  return val;
}

template <typename T>
L3_FORCE_INLINE void write_le(void* dst, T val) noexcept {
  static_assert(std::is_trivially_copyable_v<T>,
                "write_le<T>: T must be trivially copyable.");
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                    sizeof(T) == 8,
                "write_le<T>: T must be 1, 2, 4, or 8 bytes wide.");
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  if constexpr (sizeof(T) == 2) {
    u16 raw = 0;
    std::memcpy(&raw, &val, 2);
    raw = bswap16(raw);
    std::memcpy(&val, &raw, 2);
  } else if constexpr (sizeof(T) == 4) {
    u32 raw = 0;
    std::memcpy(&raw, &val, 4);
    raw = bswap32(raw);
    std::memcpy(&val, &raw, 4);
  } else if constexpr (sizeof(T) == 8) {
    u64 raw = 0;
    std::memcpy(&raw, &val, 8);
    raw = bswap64(raw);
    std::memcpy(&val, &raw, 8);
  }
#endif
  std::memcpy(dst, &val, sizeof(T));
}

}  // namespace l3
