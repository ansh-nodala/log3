// log3/core/dtype.h
//
// Data type descriptors, compile-time type traits, sub-byte packing helpers,
// and overflow-checked byte-size math for scalar and quantized tensors.
//
// Unified block model:
//   Every DType is described by (block_elements, block_bytes):
//     * Scalar types (f32, bf16, i8, ...) have block_elements == 1 and
//       block_bytes == sizeof(T).
//     * Sub-byte packed types have block_elements > 1 and block_bytes == 1:
//         - kTernary2 ("t2", BitNet b1.58 {-1, 0, +1}): 4 elements per 1 byte
//         - kInt4     ("i4", signed 4-bit [-8, 7])    : 2 elements per 1 byte
//         - kUInt4    ("u4", unsigned 4-bit [0, 15])  : 2 elements per 1 byte
//
//   Byte layout of kTernary2 ("t2") within one byte (4 weights w0..w3):
//     +---------+---------+---------+---------+
//     | bits 7:6| bits 5:4| bits 3:2| bits 1:0|
//     |   w3    |   w2    |   w1    |   w0    |
//     +---------+---------+---------+---------+
//     Encoding is biased by +1 (code = w + 1):
//       0b00 (0) -> -1
//       0b01 (1) ->  0
//       0b10 (2) -> +1
//       0b11 (3) -> reserved
//     Unpacking is a branchless mask-and-subtract: w = (code & 0x3) - 1.
//
//   Byte layout of kInt4 / kUInt4 within one byte (2 weights w0, w1):
//     +-------------------+-------------------+
//     |     bits 7:4      |     bits 3:0      |
//     |        w1         |        w0         |
//     +-------------------+-------------------+
//     kUInt4 stores [0, 15] directly; kInt4 uses biased +8 encoding
//     (code = w + 8, so w = (code & 0xF) - 8) for branchless SIMD unpacking.

#pragma once

#include <cstring>
#include <type_traits>

#include "log3/core/assert.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

// -----------------------------------------------------------------------------
// DType enumeration
// -----------------------------------------------------------------------------
enum class DType : u8 {
  kBool = 0,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
  // Sub-byte packed quantization types:
  kTernary2,  // 2-bit BitNet b1.58 ternary {-1, 0, +1}, 4 elements per byte
  kInt4,      // 4-bit signed integer [-8, 7], 2 elements per byte
  kUInt4,     // 4-bit unsigned integer [0, 15], 2 elements per byte
};

inline constexpr usize kNumDTypes = static_cast<usize>(DType::kUInt4) + 1;

// -----------------------------------------------------------------------------
// DTypeInfo metadata table (constexpr, zero runtime cost)
// -----------------------------------------------------------------------------
struct DTypeInfo {
  DType dtype;
  const char* name;     // short canonical name ("f32", "bf16", "t2", ...)
  u8 bits_per_element;  // 2, 4, 8, 16, 32, or 64
  u8 block_elements;    // elements per indivisible storage block (1, 2, or 4)
  u8 block_bytes;       // bytes per indivisible storage block (1, 2, 4, or 8)
  bool is_float;        // f16, bf16, f32, f64
  bool is_integer;      // i8..i64, u8..u64
  bool is_signed;       // signed integers, floats, t2, i4
  bool is_subbyte;      // t2, i4, u4 (block_elements > 1)
};

namespace detail {

inline constexpr DTypeInfo kDTypeTable[kNumDTypes] = {
    // dtype             name    bits  blk_el blk_B  flt    int    sgn    sub
    {DType::kBool,      "bool",   8,    1,     1,   false, false, false, false},
    {DType::kInt8,      "i8",     8,    1,     1,   false, true,  true,  false},
    {DType::kUInt8,     "u8",     8,    1,     1,   false, true,  false, false},
    {DType::kInt16,     "i16",   16,    1,     2,   false, true,  true,  false},
    {DType::kUInt16,    "u16",   16,    1,     2,   false, true,  false, false},
    {DType::kInt32,     "i32",   32,    1,     4,   false, true,  true,  false},
    {DType::kUInt32,    "u32",   32,    1,     4,   false, true,  false, false},
    {DType::kInt64,     "i64",   64,    1,     8,   false, true,  true,  false},
    {DType::kUInt64,    "u64",   64,    1,     8,   false, true,  false, false},
    {DType::kFloat16,   "f16",   16,    1,     2,   true,  false, true,  false},
    {DType::kBFloat16,  "bf16",  16,    1,     2,   true,  false, true,  false},
    {DType::kFloat32,   "f32",   32,    1,     4,   true,  false, true,  false},
    {DType::kFloat64,   "f64",   64,    1,     8,   true,  false, true,  false},
    {DType::kTernary2,  "t2",     2,    4,     1,   false, false, true,  true },
    {DType::kInt4,      "i4",     4,    2,     1,   false, false, true,  true },
    {DType::kUInt4,     "u4",     4,    2,     1,   false, false, false, true },
};

}  // namespace detail

constexpr bool is_valid_dtype(DType dt) noexcept {
  return static_cast<usize>(dt) < kNumDTypes;
}

constexpr const DTypeInfo& dtype_info(DType dt) noexcept {
  L3_DCHECK(is_valid_dtype(dt));
  return detail::kDTypeTable[static_cast<usize>(dt)];
}

constexpr const char* dtype_name(DType dt) noexcept {
  return is_valid_dtype(dt) ? detail::kDTypeTable[static_cast<usize>(dt)].name
                            : "unknown";
}

constexpr u8 dtype_bits(DType dt) noexcept {
  return dtype_info(dt).bits_per_element;
}

constexpr Index dtype_block_elements(DType dt) noexcept {
  return static_cast<Index>(dtype_info(dt).block_elements);
}

constexpr Size dtype_block_bytes(DType dt) noexcept {
  return static_cast<Size>(dtype_info(dt).block_bytes);
}

// Element size in bytes for non-sub-byte types. Asserts in debug builds if
// called on a sub-byte type (use dtype_block_bytes / checked_dtype_bytes).
constexpr Size dtype_element_size(DType dt) noexcept {
  L3_DCHECK(!dtype_info(dt).is_subbyte);
  return static_cast<Size>(dtype_info(dt).block_bytes);
}

constexpr bool is_float_dtype(DType dt) noexcept {
  return dtype_info(dt).is_float;
}

constexpr bool is_integer_dtype(DType dt) noexcept {
  return dtype_info(dt).is_integer;
}

constexpr bool is_signed_dtype(DType dt) noexcept {
  return dtype_info(dt).is_signed;
}

constexpr bool is_subbyte_dtype(DType dt) noexcept {
  return dtype_info(dt).is_subbyte;
}

// -----------------------------------------------------------------------------
// Compile-time mapping from C++ scalar types to DType
//   Sub-byte types (t2, i4, u4) have no 1-to-1 C++ scalar type, so DTypeOf<T>
//   is intentionally defined only for full-byte scalar types.
// -----------------------------------------------------------------------------
template <typename T>
struct DTypeTraits {
  static constexpr bool kSupported = false;
};

#define L3_DEFINE_DTYPE_TRAIT(cpp_type, enum_val)        \
  template <>                                            \
  struct DTypeTraits<cpp_type> {                         \
    static constexpr bool kSupported = true;             \
    static constexpr DType kValue = DType::enum_val;     \
  }

L3_DEFINE_DTYPE_TRAIT(bool, kBool);
L3_DEFINE_DTYPE_TRAIT(i8, kInt8);
L3_DEFINE_DTYPE_TRAIT(u8, kUInt8);
L3_DEFINE_DTYPE_TRAIT(i16, kInt16);
L3_DEFINE_DTYPE_TRAIT(u16, kUInt16);
L3_DEFINE_DTYPE_TRAIT(i32, kInt32);
L3_DEFINE_DTYPE_TRAIT(u32, kUInt32);
L3_DEFINE_DTYPE_TRAIT(i64, kInt64);
L3_DEFINE_DTYPE_TRAIT(u64, kUInt64);
L3_DEFINE_DTYPE_TRAIT(f16, kFloat16);
L3_DEFINE_DTYPE_TRAIT(bf16, kBFloat16);
L3_DEFINE_DTYPE_TRAIT(f32, kFloat32);
L3_DEFINE_DTYPE_TRAIT(f64, kFloat64);

#undef L3_DEFINE_DTYPE_TRAIT

template <typename T>
constexpr DType dtype_of() noexcept {
  using CleanT = std::remove_cv_t<T>;
  static_assert(DTypeTraits<CleanT>::kSupported,
                "dtype_of<T>: T is not a registered log3 scalar DType.");
  return DTypeTraits<CleanT>::kValue;
}

// -----------------------------------------------------------------------------
// Overflow-checked storage size calculation
// -----------------------------------------------------------------------------

// Computes the exact byte count for `numel` elements of type `dt`.
// Fails with:
//   * kInvalidArgument if `numel < 0` or `numel` is not divisible by
//     `dtype_block_elements(dt)` (e.g. 5 elements of kTernary2).
//   * kOutOfRange if the byte count overflows `Size`.
inline Status checked_dtype_bytes(DType dt, Index numel, Size* out_bytes) noexcept {
  L3_DCHECK(out_bytes != nullptr);
  if (L3_UNLIKELY(!is_valid_dtype(dt))) {
    return Status::error(StatusCode::kInvalidArgument, "invalid DType");
  }
  if (L3_UNLIKELY(numel < 0)) {
    return Status::errorf(StatusCode::kInvalidArgument,
                          "negative element count: %lld",
                          static_cast<long long>(numel));
  }
  const DTypeInfo& info = dtype_info(dt);
  const Index blk_elems = static_cast<Index>(info.block_elements);
  if (L3_UNLIKELY(numel % blk_elems != 0)) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "element count (%lld) for DType %s must be a multiple of %lld",
        static_cast<long long>(numel), info.name,
        static_cast<long long>(blk_elems));
  }
  const Size blocks = static_cast<Size>(numel / blk_elems);
  if (L3_UNLIKELY(!checked_mul(blocks, static_cast<Size>(info.block_bytes),
                               out_bytes))) {
    return Status::errorf(StatusCode::kOutOfRange,
                          "byte size overflow for %lld elements of %s",
                          static_cast<long long>(numel), info.name);
  }
  return Status::ok();
}

// Same as checked_dtype_bytes, but rounds `numel` UP to the nearest whole
// storage block for sub-byte types (useful when packing tail elements).
inline Status checked_dtype_bytes_padded(DType dt, Index numel,
                                         Size* out_bytes) noexcept {
  L3_DCHECK(out_bytes != nullptr);
  if (L3_UNLIKELY(!is_valid_dtype(dt))) {
    return Status::error(StatusCode::kInvalidArgument, "invalid DType");
  }
  if (L3_UNLIKELY(numel < 0)) {
    return Status::errorf(StatusCode::kInvalidArgument,
                          "negative element count: %lld",
                          static_cast<long long>(numel));
  }
  const DTypeInfo& info = dtype_info(dt);
  const Size blocks =
      div_ceil(static_cast<Size>(numel), static_cast<Size>(info.block_elements));
  if (L3_UNLIKELY(!checked_mul(blocks, static_cast<Size>(info.block_bytes),
                               out_bytes))) {
    return Status::errorf(StatusCode::kOutOfRange,
                          "byte size overflow for %lld elements of %s",
                          static_cast<long long>(numel), info.name);
  }
  return Status::ok();
}

// -----------------------------------------------------------------------------
// Name parsing (accepts canonical short names and common aliases)
// -----------------------------------------------------------------------------
inline Result<DType> dtype_from_name(const char* name) noexcept {
  if (L3_UNLIKELY(name == nullptr)) {
    return Status::error(StatusCode::kInvalidArgument, "dtype name is null");
  }
  for (usize i = 0; i < kNumDTypes; ++i) {
    if (std::strcmp(name, detail::kDTypeTable[i].name) == 0) {
      return static_cast<DType>(detail::kDTypeTable[i].dtype);
    }
  }
  // Common full-name aliases (for Python bindings / model config files).
  if (std::strcmp(name, "float32") == 0 || std::strcmp(name, "float") == 0) {
    return DType::kFloat32;
  }
  if (std::strcmp(name, "float16") == 0 || std::strcmp(name, "half") == 0) {
    return DType::kFloat16;
  }
  if (std::strcmp(name, "bfloat16") == 0) return DType::kBFloat16;
  if (std::strcmp(name, "float64") == 0 || std::strcmp(name, "double") == 0) {
    return DType::kFloat64;
  }
  if (std::strcmp(name, "int8") == 0) return DType::kInt8;
  if (std::strcmp(name, "uint8") == 0) return DType::kUInt8;
  if (std::strcmp(name, "int16") == 0) return DType::kInt16;
  if (std::strcmp(name, "uint16") == 0) return DType::kUInt16;
  if (std::strcmp(name, "int32") == 0) return DType::kInt32;
  if (std::strcmp(name, "uint32") == 0) return DType::kUInt32;
  if (std::strcmp(name, "int64") == 0) return DType::kInt64;
  if (std::strcmp(name, "uint64") == 0) return DType::kUInt64;
  if (std::strcmp(name, "ternary2") == 0 || std::strcmp(name, "i2_s") == 0 ||
      std::strcmp(name, "bitnet158") == 0) {
    return DType::kTernary2;
  }
  if (std::strcmp(name, "int4") == 0) return DType::kInt4;
  if (std::strcmp(name, "uint4") == 0) return DType::kUInt4;

  return Status::errorf(StatusCode::kInvalidArgument,
                        "unknown DType name: '%s'", name);
}

// -----------------------------------------------------------------------------
// BitNet kTernary2 ("t2") and 4-bit ("i4" / "u4") scalar pack/unpack helpers
// -----------------------------------------------------------------------------

// Clamp an integer to {-1, 0, +1} and encode as a 2-bit biased value {0, 1, 2}.
constexpr u8 encode_t2(i8 v) noexcept {
  const i8 clamped = v < -1 ? -1 : (v > 1 ? 1 : v);
  return static_cast<u8>(clamped + 1);
}

// Decode a 2-bit biased code {0, 1, 2} back to {-1, 0, +1}.
constexpr i8 decode_t2(u8 code) noexcept {
  return static_cast<i8>(static_cast<int>(code & 0x03u) - 1);
}

// Pack 4 ternary values {-1, 0, +1} into a single byte (w0 in bits 1:0,
// w1 in bits 3:2, w2 in bits 5:4, w3 in bits 7:6).
constexpr u8 pack_t2_x4(i8 w0, i8 w1, i8 w2, i8 w3) noexcept {
  return static_cast<u8>(encode_t2(w0) |
                         (encode_t2(w1) << 2) |
                         (encode_t2(w2) << 4) |
                         (encode_t2(w3) << 6));
}

// Unpack a single kTernary2 byte into 4 signed i8 values in {-1, 0, +1}.
inline void unpack_t2_x4(u8 packed, i8 out[4]) noexcept {
  out[0] = decode_t2(packed);
  out[1] = decode_t2(static_cast<u8>(packed >> 2));
  out[2] = decode_t2(static_cast<u8>(packed >> 4));
  out[3] = decode_t2(static_cast<u8>(packed >> 6));
}

// Pack 2 signed 4-bit integers in [-8, 7] into one byte (biased by +8).
constexpr u8 pack_i4_x2(i8 w0, i8 w1) noexcept {
  const i8 c0 = w0 < -8 ? -8 : (w0 > 7 ? 7 : w0);
  const i8 c1 = w1 < -8 ? -8 : (w1 > 7 ? 7 : w1);
  const u8 b0 = static_cast<u8>(c0 + 8) & 0x0fu;
  const u8 b1 = static_cast<u8>(c1 + 8) & 0x0fu;
  return static_cast<u8>(b0 | (b1 << 4));
}

inline void unpack_i4_x2(u8 packed, i8 out[2]) noexcept {
  out[0] = static_cast<i8>(static_cast<int>(packed & 0x0fu) - 8);
  out[1] = static_cast<i8>(static_cast<int>((packed >> 4) & 0x0fu) - 8);
}

// Pack 2 unsigned 4-bit integers in [0, 15] into one byte.
constexpr u8 pack_u4_x2(u8 w0, u8 w1) noexcept {
  const u8 b0 = (w0 > 15u ? 15u : w0) & 0x0fu;
  const u8 b1 = (w1 > 15u ? 15u : w1) & 0x0fu;
  return static_cast<u8>(b0 | (b1 << 4));
}

inline void unpack_u4_x2(u8 packed, u8 out[2]) noexcept {
  out[0] = packed & 0x0fu;
  out[1] = (packed >> 4) & 0x0fu;
}

}  // namespace l3
