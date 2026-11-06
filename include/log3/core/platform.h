// log3/core/platform.h
//
// Foundation header. Detects OS, compiler, CPU architecture and compile-time
// SIMD level, and exposes portable attribute macros.
//
// Rules for this file:
//   * Zero #includes (no std, no system headers). Everything here is macros
//     and constants, so it is safe to include from anywhere.
//   * Every detection macro is always defined to 0 or 1, so use `#if L3_X`,
//     never `#ifdef L3_X`.
//   * L3_HAS_* macros describe what THIS translation unit was compiled for.
//     Runtime CPU dispatch (choosing AVX2 vs AVX512 etc. on the user's
//     machine) lives in a separate cpu-features module, not here.

#pragma once

// -----------------------------------------------------------------------------
// Version
// -----------------------------------------------------------------------------
#define L3_VERSION_MAJOR 0
#define L3_VERSION_MINOR 1
#define L3_VERSION_PATCH 0

// -----------------------------------------------------------------------------
// C++ standard (MSVC reports it via _MSVC_LANG unless /Zc:__cplusplus is set)
// -----------------------------------------------------------------------------
#if defined(_MSVC_LANG)
#define L3_CPLUSPLUS _MSVC_LANG
#else
#define L3_CPLUSPLUS __cplusplus
#endif

#if L3_CPLUSPLUS < 201703L
#error "log3 requires C++17 or newer."
#endif

// -----------------------------------------------------------------------------
// Operating system
// -----------------------------------------------------------------------------
#if defined(_WIN32)
#define L3_OS_WINDOWS 1
#else
#define L3_OS_WINDOWS 0
#endif

// Android also defines __linux__, so it must be checked first.
#if defined(__ANDROID__)
#define L3_OS_ANDROID 1
#else
#define L3_OS_ANDROID 0
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#define L3_OS_LINUX 1
#else
#define L3_OS_LINUX 0
#endif

#if defined(__APPLE__)
#define L3_OS_APPLE 1
#else
#define L3_OS_APPLE 0
#endif

#if !(L3_OS_WINDOWS || L3_OS_ANDROID || L3_OS_LINUX || L3_OS_APPLE)
#error "log3: unsupported operating system."
#endif

// POSIX-like convenience (everything except Windows).
#if L3_OS_WINDOWS
#define L3_OS_POSIX 0
#else
#define L3_OS_POSIX 1
#endif

// Keep <windows.h> (if someone includes it) from defining min/max macros.
#if L3_OS_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// -----------------------------------------------------------------------------
// Compiler
// clang-cl defines both _MSC_VER and __clang__: it is Clang, but uses the
// MSVC ABI/driver, so L3_MSVC_ABI is the right switch for MSVC-style syntax.
// -----------------------------------------------------------------------------
#if defined(__clang__)
#define L3_COMPILER_CLANG 1
#else
#define L3_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define L3_COMPILER_GCC 1
#else
#define L3_COMPILER_GCC 0
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define L3_COMPILER_MSVC 1
#else
#define L3_COMPILER_MSVC 0
#endif

#if defined(_MSC_VER)
#define L3_MSVC_ABI 1
#else
#define L3_MSVC_ABI 0
#endif

#if !(L3_COMPILER_CLANG || L3_COMPILER_GCC || L3_COMPILER_MSVC)
#error "log3: unsupported compiler."
#endif

// -----------------------------------------------------------------------------
// CPU architecture
// -----------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
#define L3_ARCH_X64 1
#else
#define L3_ARCH_X64 0
#endif

#if (defined(__i386__) || defined(_M_IX86)) && !L3_ARCH_X64
#define L3_ARCH_X86 1
#else
#define L3_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define L3_ARCH_ARM64 1
#else
#define L3_ARCH_ARM64 0
#endif

#if (defined(__arm__) || defined(_M_ARM)) && !L3_ARCH_ARM64
#define L3_ARCH_ARM32 1
#else
#define L3_ARCH_ARM32 0
#endif

#define L3_ARCH_X86_FAMILY (L3_ARCH_X64 || L3_ARCH_X86)
#define L3_ARCH_ARM_FAMILY (L3_ARCH_ARM64 || L3_ARCH_ARM32)

#if !(L3_ARCH_X86_FAMILY || L3_ARCH_ARM_FAMILY)
#error "log3: unsupported CPU architecture."
#endif

// -----------------------------------------------------------------------------
// Compile-time SIMD level of this translation unit
// -----------------------------------------------------------------------------
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define L3_HAS_SSE2 1
#else
#define L3_HAS_SSE2 0
#endif

#if defined(__AVX2__)
#define L3_HAS_AVX2 1
#else
#define L3_HAS_AVX2 0
#endif

#if defined(__AVX512F__)
#define L3_HAS_AVX512F 1
#else
#define L3_HAS_AVX512F 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define L3_HAS_NEON 1
#else
#define L3_HAS_NEON 0
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#define L3_HAS_NEON_DOTPROD 1
#else
#define L3_HAS_NEON_DOTPROD 0
#endif

// -----------------------------------------------------------------------------
// Pointer width
// -----------------------------------------------------------------------------
static_assert(sizeof(void*) == 8 || sizeof(void*) == 4,
              "log3: only 32-bit and 64-bit targets are supported.");

#define L3_64BIT (sizeof(void*) == 8)

// -----------------------------------------------------------------------------
// Build mode
// -----------------------------------------------------------------------------
#if !defined(NDEBUG) || defined(L3_FORCE_DEBUG)
#define L3_DEBUG 1
#else
#define L3_DEBUG 0
#endif

// Exceptions are NOT used inside the core. This only reports whether the
// compiler currently allows them (the Python binding layer may enable them).
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#define L3_HAS_EXCEPTIONS 1
#else
#define L3_HAS_EXCEPTIONS 0
#endif

// ThreadSanitizer detection. TSan does not model standalone atomic fences, so
// lock-free code that uses a fence on its fast path switches to an equivalent
// (slightly slower) ordering when this is 1. Normal builds are unaffected.
#if defined(__SANITIZE_THREAD__)
#define L3_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define L3_TSAN 1
#endif
#endif
#ifndef L3_TSAN
#define L3_TSAN 0
#endif

// -----------------------------------------------------------------------------
// Symbol visibility / DLL export
//   L3_STATIC     : building or using log3 as a static library -> no decoration
//   L3_BUILDING   : defined by CMake while compiling log3 itself
// -----------------------------------------------------------------------------
#if defined(L3_STATIC)
#define L3_API
#define L3_LOCAL
#elif L3_OS_WINDOWS
#if defined(L3_BUILDING)
#define L3_API __declspec(dllexport)
#else
#define L3_API __declspec(dllimport)
#endif
#define L3_LOCAL
#else
#define L3_API __attribute__((visibility("default")))
#define L3_LOCAL __attribute__((visibility("hidden")))
#endif

// -----------------------------------------------------------------------------
// Inlining and branch hints
// -----------------------------------------------------------------------------
#if L3_MSVC_ABI
#define L3_FORCE_INLINE __forceinline
#define L3_NOINLINE __declspec(noinline)
#else
#define L3_FORCE_INLINE inline __attribute__((always_inline))
#define L3_NOINLINE __attribute__((noinline))
#endif

#if L3_MSVC_ABI && !L3_COMPILER_CLANG
#define L3_LIKELY(x) (x)
#define L3_UNLIKELY(x) (x)
#else
#define L3_LIKELY(x) (__builtin_expect(!!(x), 1))
#define L3_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Mark cold paths (error handling, logging slow path).
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
#define L3_COLD
#else
#define L3_COLD __attribute__((cold))
#endif

// Tell the optimizer a code path can never be reached.
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
#define L3_UNREACHABLE() __assume(0)
#else
#define L3_UNREACHABLE() __builtin_unreachable()
#endif

// Pointer aliasing promise for hot kernels.
#if L3_MSVC_ABI
#define L3_RESTRICT __restrict
#else
#define L3_RESTRICT __restrict__
#endif

// -----------------------------------------------------------------------------
// Standard attributes (C++17)
// -----------------------------------------------------------------------------
#define L3_NODISCARD [[nodiscard]]
#define L3_MAYBE_UNUSED [[maybe_unused]]
#define L3_FALLTHROUGH [[fallthrough]]

// printf-style format checking: a wrong format string or argument type becomes
// a compile warning. fmt_index / first_arg_index are 1-based parameter
// positions (for non-static member functions, `this` counts as position 1).
#if L3_COMPILER_GCC || L3_COMPILER_CLANG
#define L3_PRINTF_FORMAT(fmt_index, first_arg_index) \
  __attribute__((format(printf, fmt_index, first_arg_index)))
#else
#define L3_PRINTF_FORMAT(fmt_index, first_arg_index)
#endif

// -----------------------------------------------------------------------------
// Alignment
// -----------------------------------------------------------------------------
#define L3_ALIGNAS(n) alignas(n)

namespace l3 {

// Typical x86-64 and ARM64 cache line. Use to avoid false sharing.
inline constexpr unsigned kCacheLineSize = 64;

// Default tensor storage alignment: one cache line, also satisfies AVX-512.
inline constexpr unsigned kDefaultAlignment = 64;

}  // namespace l3

// -----------------------------------------------------------------------------
// Debug break
// -----------------------------------------------------------------------------
#if L3_MSVC_ABI
#define L3_DEBUG_BREAK() __debugbreak()
#elif L3_COMPILER_CLANG
#define L3_DEBUG_BREAK() __builtin_debugtrap()
#else
#define L3_DEBUG_BREAK() __builtin_trap()
#endif

// -----------------------------------------------------------------------------
// Function signature string (used by logging and error messages)
// -----------------------------------------------------------------------------
#if L3_MSVC_ABI
#define L3_FUNC_SIG __FUNCSIG__
#else
#define L3_FUNC_SIG __PRETTY_FUNCTION__
#endif

// -----------------------------------------------------------------------------
// Preprocessor helpers
// -----------------------------------------------------------------------------
#define L3_STRINGIFY_IMPL(x) #x
#define L3_STRINGIFY(x) L3_STRINGIFY_IMPL(x)

#define L3_CONCAT_IMPL(a, b) a##b
#define L3_CONCAT(a, b) L3_CONCAT_IMPL(a, b)

// Unique identifier per line, handy for scope guards and macros.
#define L3_UNIQUE_NAME(prefix) L3_CONCAT(prefix, __LINE__)

// Suppress "unused variable" for release builds where asserts vanish.
#define L3_UNUSED(x) static_cast<void>(sizeof(x))

// -----------------------------------------------------------------------------
// Local warning control
// -----------------------------------------------------------------------------
#if L3_MSVC_ABI && !L3_COMPILER_CLANG
#define L3_WARNING_PUSH __pragma(warning(push))
#define L3_WARNING_POP __pragma(warning(pop))
#elif L3_COMPILER_CLANG
#define L3_WARNING_PUSH _Pragma("clang diagnostic push")
#define L3_WARNING_POP _Pragma("clang diagnostic pop")
#else
#define L3_WARNING_PUSH _Pragma("GCC diagnostic push")
#define L3_WARNING_POP _Pragma("GCC diagnostic pop")
#endif
