
// log3/core/types.h
//
// Fundamental scalar types used across the entire log3 library.
//
// Design principles:
//   * Use standard fixed-width integer types instead of compiler-specific
//     primitive assumptions.
//   * Keep this header lightweight and independent of platform/runtime code.
//   * Use signed indexing for tensor dimensions, strides, and offsets.
//   * Use unsigned types for memory sizes and allocation byte counts.
//   * Keep f16 and bf16 as storage-only types. Arithmetic and conversions
//     belong in core/half.h and the numerical kernel implementations.
//   * Enforce the representation assumptions required by the CPU backend.
//
// Compatibility:
//   * C++17
//   * No dynamic allocation
//   * No exceptions
//   * No RTTI

#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace l3 {

// -----------------------------------------------------------------------------
// Platform representation requirements
// -----------------------------------------------------------------------------

// log3 currently assumes 8-bit bytes throughout its packed tensor formats,
// memory representation, and low-level CPU kernels.
static_assert(CHAR_BIT == 8,
              "log3 requires platforms with 8-bit bytes.");

// -----------------------------------------------------------------------------
// Fixed-width integer types
// -----------------------------------------------------------------------------
//
// These aliases use the standard exact-width integer types.
//
// Exact-width types are important for:
//   * Binary serialization.
//   * Tensor storage formats.
//   * Quantized weights.
//   * SIMD kernel interfaces.
//   * Cross-platform numerical consistency.
//

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Raw 8-bit storage unit.
//
// This remains an integer alias rather than std::byte to preserve convenient
// compatibility with packed tensor formats and existing log3 interfaces.
using byte = u8;

// -----------------------------------------------------------------------------
// Platform-sized integer types
// -----------------------------------------------------------------------------
//
// usize:
//   Represents the size of an object or memory allocation.
//   Equivalent to std::size_t.
//
// isize:
//   Represents pointer differences.
//   Equivalent to std::ptrdiff_t.
//
// These types are commonly pointer-sized, but their actual guarantees come
// from the C++ standard rather than an assumption that sizeof(pointer) matches.
//

using usize = std::size_t;
using isize = std::ptrdiff_t;

// -----------------------------------------------------------------------------
// Floating-point types
// -----------------------------------------------------------------------------

using f32 = float;
using f64 = double;

// log3 numerical kernels require IEEE-754 floating-point representations.
//
// Size assertions alone are insufficient because a floating-point type can
// have the expected size without having the expected representation.

static_assert(sizeof(f32) == 4,
              "log3 requires 32-bit f32 storage.");

static_assert(sizeof(f64) == 8,
              "log3 requires 64-bit f64 storage.");

static_assert(std::numeric_limits<f32>::is_iec559,
              "log3 requires IEEE-754 f32.");

static_assert(std::numeric_limits<f64>::is_iec559,
              "log3 requires IEEE-754 f64.");

// -----------------------------------------------------------------------------
// Tensor-facing integer types
// -----------------------------------------------------------------------------
//
// Index:
//   Signed integer used for tensor dimensions, strides, and element offsets.
//
//   Signed indexing allows negative strides and avoids unsigned underflow
//   during index arithmetic.
//
// Size:
//   Unsigned integer used for allocation sizes and byte counts.
//
//   Keep indexing arithmetic and allocation-size arithmetic conceptually
//   separate. Conversions between them must be validated.
//

using Index = i64;
using Size  = usize;

// Numeric limits for tensor index arithmetic.
//
// Prefer std::numeric_limits over implementation-provided integer macros.
// These expressions are constexpr and require no runtime initialization.

inline constexpr Index kMaxIndex =
    std::numeric_limits<Index>::max();

inline constexpr Index kMinIndex =
    std::numeric_limits<Index>::min();

// -----------------------------------------------------------------------------
// Half-precision storage types
// -----------------------------------------------------------------------------
//
// f16:
//   IEEE-754 binary16 storage representation.
//   1 sign bit, 5 exponent bits, 10 mantissa bits.
//
// bf16:
//   Brain floating-point 16-bit storage representation.
//   1 sign bit, 8 exponent bits, 7 mantissa bits.
//
// Both types are storage containers only.
//
// IMPORTANT:
//   These structures do not implement floating-point arithmetic.
//   Conversion and arithmetic must be implemented separately in core/half.h
//   and numerical kernels.
//
// The bits member stores the raw 16-bit representation.
//

struct f16 {
  u16 bits;

  // Construct a storage value from its raw binary16 representation.
  static constexpr f16 from_bits(u16 value) noexcept {
    return f16{value};
  }

  // Return the raw binary16 representation.
  constexpr u16 to_bits() const noexcept {
    return bits;
  }
};

struct bf16 {
  u16 bits;

  // Construct a storage value from its raw bfloat16 representation.
  static constexpr bf16 from_bits(u16 value) noexcept {
    return bf16{value};
  }

  // Return the raw bfloat16 representation.
  constexpr u16 to_bits() const noexcept {
    return bits;
  }
};

// -----------------------------------------------------------------------------
// Compile-time representation validation
// -----------------------------------------------------------------------------
//
// Integer aliases must retain their expected widths and signedness.
// These checks catch unsupported or unexpected platform configurations early.
//

static_assert(sizeof(i8) == 1 && sizeof(u8) == 1,
              "log3 requires 8-bit integer types.");

static_assert(sizeof(i16) == 2 && sizeof(u16) == 2,
              "log3 requires 16-bit integer types.");

static_assert(sizeof(i32) == 4 && sizeof(u32) == 4,
              "log3 requires 32-bit integer types.");

static_assert(sizeof(i64) == 8 && sizeof(u64) == 8,
              "log3 requires 64-bit integer types.");

static_assert(std::is_signed<i8>::value &&
                  std::is_signed<i16>::value &&
                  std::is_signed<i32>::value &&
                  std::is_signed<i64>::value,
              "log3 signed integer aliases must be signed.");

static_assert(!std::is_signed<u8>::value &&
                  !std::is_signed<u16>::value &&
                  !std::is_signed<u32>::value &&
                  !std::is_signed<u64>::value,
              "log3 unsigned integer aliases must be unsigned.");

// Tensor-facing types must preserve their intended signedness.

static_assert(std::is_signed<Index>::value,
              "log3 Index must be signed.");

static_assert(!std::is_signed<Size>::value,
              "log3 Size must be unsigned.");

static_assert(sizeof(Index) == 8,
              "log3 Index must be 64 bits.");

// -----------------------------------------------------------------------------
// Half-precision storage validation
// -----------------------------------------------------------------------------

static_assert(sizeof(f16) == 2,
              "log3 f16 storage must be exactly 2 bytes.");

static_assert(sizeof(bf16) == 2,
              "log3 bf16 storage must be exactly 2 bytes.");

static_assert(alignof(f16) == alignof(u16),
              "log3 f16 alignment must match u16.");

static_assert(alignof(bf16) == alignof(u16),
              "log3 bf16 alignment must match u16.");

static_assert(std::is_standard_layout<f16>::value &&
                  std::is_standard_layout<bf16>::value,
              "log3 half storage types must have standard layout.");

static_assert(std::is_trivially_copyable<f16>::value &&
                  std::is_trivially_copyable<bf16>::value,
              "log3 half storage types must be trivially copyable.");

}  // namespace l3
