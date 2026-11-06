// log3/core/assert.h
//
// Invariant checking. Built on log.h (fatal log + abort) and status.h.
//
// Two tiers, deliberately simple:
//
//   L3_CHECK*   Always on, in debug AND release. For conditions whose
//               violation means a bug or an unrecoverable state (corrupt
//               graph, impossible shape after validation, ...). On failure:
//               log a FATAL message with file:line and the failed expression,
//               then abort. Cheap enough to leave on: the failure path is
//               out-of-line and cold, the hot path is one predictable branch.
//
//   L3_DCHECK*  Debug builds only (L3_DEBUG). Use inside hot kernels and
//               inner loops. In release the condition is NOT evaluated and no
//               code is generated, but it is still type-checked, so it cannot
//               silently rot.
//
// What these are NOT for: validating user input. Bad arguments coming from a
// caller (wrong shape, wrong dtype, ...) are recoverable and must be reported
// with Status / Result<T>, never with CHECK.
//
//   L3_CHECK(cond)                     L3_CHECK_MSG(cond, fmt, ...)
//   L3_CHECK_EQ/NE/LT/LE/GT/GE(a, b)   prints both values on failure
//   L3_CHECK_OK(status_expr)           prints the Status code and message
//   L3_PANIC(fmt, ...)                 unconditional fatal error
//   L3_NOT_IMPLEMENTED()               unconditional fatal "not implemented"
//   L3_DCHECK_UNREACHABLE()            debug: panic, release: optimizer hint
//
// Comparison checks (L3_CHECK_EQ etc.) accept bool, integers, enums, floating
// point, pointers and nullptr. Mixed signed/unsigned integer comparisons are
// done correctly (-1 is never equal to UINT_MAX), which std only offers from
// C++20 (std::cmp_equal); we provide it for C++17.
//
// Tip: set -ffile-prefix-map=<src_dir>=. (GCC/Clang) or /pathmap (MSVC) so
// __FILE__ does not embed your build machine's absolute paths in the binary.

#pragma once

#include <type_traits>

#include "log3/core/log.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {
namespace detail {

// -----------------------------------------------------------------------------
// Value capture for failure messages
// -----------------------------------------------------------------------------
struct CheckValue {
  enum class Kind : u8 { kBool, kSigned, kUnsigned, kFloat, kPointer };

  Kind kind;
  union {
    i64 i;
    u64 u;
    f64 f;
    const void* p;
  };
};

template <typename T>
inline constexpr bool kAlwaysFalse = false;

template <typename T>
CheckValue make_check_value(const T& v) noexcept {
  CheckValue out{};
  if constexpr (std::is_same_v<T, bool>) {
    out.kind = CheckValue::Kind::kBool;
    out.u = v ? 1u : 0u;
  } else if constexpr (std::is_enum_v<T>) {
    return make_check_value(static_cast<std::underlying_type_t<T>>(v));
  } else if constexpr (std::is_integral_v<T>) {
    if constexpr (std::is_signed_v<T>) {
      out.kind = CheckValue::Kind::kSigned;
      out.i = static_cast<i64>(v);
    } else {
      out.kind = CheckValue::Kind::kUnsigned;
      out.u = static_cast<u64>(v);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    out.kind = CheckValue::Kind::kFloat;
    out.f = static_cast<f64>(v);
  } else if constexpr (std::is_pointer_v<T>) {
    out.kind = CheckValue::Kind::kPointer;
    out.p = static_cast<const void*>(v);
  } else if constexpr (std::is_null_pointer_v<T>) {
    out.kind = CheckValue::Kind::kPointer;
    out.p = nullptr;
  } else {
    static_assert(kAlwaysFalse<T>,
                  "L3_CHECK_<op> supports bool, integer, enum, floating point, "
                  "pointer and nullptr operands only.");
  }
  return out;
}

// -----------------------------------------------------------------------------
// Safe comparisons (mixed signed/unsigned integers handled correctly)
// -----------------------------------------------------------------------------
template <typename A, typename B>
inline constexpr bool kMixedSignInts =
    std::is_integral_v<A> && std::is_integral_v<B> &&
    !std::is_same_v<A, bool> && !std::is_same_v<B, bool> &&
    (std::is_signed_v<A> != std::is_signed_v<B>);

template <typename A, typename B>
constexpr bool cmp_eq(const A& a, const B& b) noexcept {
  if constexpr (kMixedSignInts<A, B>) {
    if constexpr (std::is_signed_v<A>) {
      return a >= 0 && static_cast<std::make_unsigned_t<A>>(a) == b;
    } else {
      return b >= 0 && a == static_cast<std::make_unsigned_t<B>>(b);
    }
  } else {
    return a == b;
  }
}

template <typename A, typename B>
constexpr bool cmp_ne(const A& a, const B& b) noexcept {
  return !cmp_eq(a, b);
}

template <typename A, typename B>
constexpr bool cmp_lt(const A& a, const B& b) noexcept {
  if constexpr (kMixedSignInts<A, B>) {
    if constexpr (std::is_signed_v<A>) {
      return a < 0 || static_cast<std::make_unsigned_t<A>>(a) < b;
    } else {
      return b >= 0 && a < static_cast<std::make_unsigned_t<B>>(b);
    }
  } else {
    return a < b;
  }
}

template <typename A, typename B>
constexpr bool cmp_gt(const A& a, const B& b) noexcept {
  return cmp_lt(b, a);
}

// For floating point, <= and >= must stay false when either side is NaN, so
// they are not written as !(a > b). Only the integer path uses negation.
template <typename A, typename B>
constexpr bool cmp_le(const A& a, const B& b) noexcept {
  if constexpr (kMixedSignInts<A, B>) {
    return !cmp_lt(b, a);
  } else {
    return a <= b;
  }
}

template <typename A, typename B>
constexpr bool cmp_ge(const A& a, const B& b) noexcept {
  if constexpr (kMixedSignInts<A, B>) {
    return !cmp_lt(a, b);
  } else {
    return a >= b;
  }
}

// -----------------------------------------------------------------------------
// Failure paths: out-of-line, cold, never return. Defined in
// src/core/assert.cpp. They log at FATAL level and call fatal_abort().
// -----------------------------------------------------------------------------
[[noreturn]] L3_COLD L3_API void check_failed(const char* file, int line,
                                              const char* func,
                                              const char* expr) noexcept;

[[noreturn]] L3_COLD L3_API void check_failed_msg(const char* file, int line,
                                                  const char* func,
                                                  const char* expr,
                                                  const char* fmt, ...) noexcept
    L3_PRINTF_FORMAT(5, 6);

[[noreturn]] L3_COLD L3_API void check_op_failed(const char* file, int line,
                                                 const char* func,
                                                 const char* expr,
                                                 CheckValue lhs,
                                                 CheckValue rhs) noexcept;

[[noreturn]] L3_COLD L3_API void check_status_failed(const char* file, int line,
                                                     const char* func,
                                                     const char* expr,
                                                     const Status& status) noexcept;

[[noreturn]] L3_COLD L3_API void panic_msg(const char* file, int line,
                                           const char* func,
                                           const char* fmt, ...) noexcept
    L3_PRINTF_FORMAT(4, 5);

}  // namespace detail
}  // namespace l3

// -----------------------------------------------------------------------------
// L3_CHECK family (always on)
// -----------------------------------------------------------------------------
#define L3_CHECK(cond)                                                      \
  do {                                                                      \
    if (L3_UNLIKELY(!(cond))) {                                             \
      ::l3::detail::check_failed(__FILE__, __LINE__, __func__, #cond);      \
    }                                                                       \
  } while (0)

// fmt is required; extra arguments are printf-style.
#define L3_CHECK_MSG(cond, ...)                                             \
  do {                                                                      \
    if (L3_UNLIKELY(!(cond))) {                                             \
      ::l3::detail::check_failed_msg(__FILE__, __LINE__, __func__, #cond,   \
                                     __VA_ARGS__);                          \
    }                                                                       \
  } while (0)

#define L3_CHECK_OP_IMPL(cmp_fn, expr_text, a, b)                           \
  do {                                                                      \
    const auto& L3_UNIQUE_NAME(l3_lhs_) = (a);                              \
    const auto& L3_UNIQUE_NAME(l3_rhs_) = (b);                              \
    if (L3_UNLIKELY(!::l3::detail::cmp_fn(L3_UNIQUE_NAME(l3_lhs_),          \
                                          L3_UNIQUE_NAME(l3_rhs_)))) {      \
      ::l3::detail::check_op_failed(                                        \
          __FILE__, __LINE__, __func__, expr_text,                          \
          ::l3::detail::make_check_value(L3_UNIQUE_NAME(l3_lhs_)),          \
          ::l3::detail::make_check_value(L3_UNIQUE_NAME(l3_rhs_)));         \
    }                                                                       \
  } while (0)

#define L3_CHECK_EQ(a, b) L3_CHECK_OP_IMPL(cmp_eq, #a " == " #b, a, b)
#define L3_CHECK_NE(a, b) L3_CHECK_OP_IMPL(cmp_ne, #a " != " #b, a, b)
#define L3_CHECK_LT(a, b) L3_CHECK_OP_IMPL(cmp_lt, #a " < " #b, a, b)
#define L3_CHECK_LE(a, b) L3_CHECK_OP_IMPL(cmp_le, #a " <= " #b, a, b)
#define L3_CHECK_GT(a, b) L3_CHECK_OP_IMPL(cmp_gt, #a " > " #b, a, b)
#define L3_CHECK_GE(a, b) L3_CHECK_OP_IMPL(cmp_ge, #a " >= " #b, a, b)

// expr may be a Status temporary or a const Status& (e.g. result.status()).
#define L3_CHECK_OK(expr)                                                   \
  do {                                                                      \
    const ::l3::Status& L3_UNIQUE_NAME(l3_cs_) = (expr);                    \
    if (L3_UNLIKELY(!L3_UNIQUE_NAME(l3_cs_).is_ok())) {                     \
      ::l3::detail::check_status_failed(__FILE__, __LINE__, __func__,       \
                                        #expr, L3_UNIQUE_NAME(l3_cs_));     \
    }                                                                       \
  } while (0)

// -----------------------------------------------------------------------------
// L3_DCHECK family (debug only). In release the branch is dead: nothing is
// evaluated, nothing is emitted, but the code is still type-checked.
// -----------------------------------------------------------------------------
#define L3_DCHECK(cond)         do { if (L3_DEBUG) { L3_CHECK(cond); } } while (0)
#define L3_DCHECK_MSG(cond, ...) \
  do { if (L3_DEBUG) { L3_CHECK_MSG(cond, __VA_ARGS__); } } while (0)

#define L3_DCHECK_EQ(a, b) do { if (L3_DEBUG) { L3_CHECK_EQ(a, b); } } while (0)
#define L3_DCHECK_NE(a, b) do { if (L3_DEBUG) { L3_CHECK_NE(a, b); } } while (0)
#define L3_DCHECK_LT(a, b) do { if (L3_DEBUG) { L3_CHECK_LT(a, b); } } while (0)
#define L3_DCHECK_LE(a, b) do { if (L3_DEBUG) { L3_CHECK_LE(a, b); } } while (0)
#define L3_DCHECK_GT(a, b) do { if (L3_DEBUG) { L3_CHECK_GT(a, b); } } while (0)
#define L3_DCHECK_GE(a, b) do { if (L3_DEBUG) { L3_CHECK_GE(a, b); } } while (0)
#define L3_DCHECK_OK(expr) do { if (L3_DEBUG) { L3_CHECK_OK(expr); } } while (0)

// -----------------------------------------------------------------------------
// Unconditional fatal errors
// -----------------------------------------------------------------------------
#define L3_PANIC(...) \
  ::l3::detail::panic_msg(__FILE__, __LINE__, __func__, __VA_ARGS__)

#define L3_NOT_IMPLEMENTED() \
  ::l3::detail::panic_msg(__FILE__, __LINE__, __func__, "not implemented")

// Debug: panic if reached. Release: tell the optimizer it cannot happen.
#if L3_DEBUG
#define L3_DCHECK_UNREACHABLE() \
  ::l3::detail::panic_msg(__FILE__, __LINE__, __func__, "unreachable code reached")
#else
#define L3_DCHECK_UNREACHABLE() L3_UNREACHABLE()
#endif
