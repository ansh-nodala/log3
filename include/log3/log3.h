// log3/log3.h
//
// Master umbrella header for the log3 C++17 BitNet b1.58, N-D Tensor, SIMD
// Kernels, Neural Modules, Zero-Copy Checkpoint I/O, and Reverse-Mode Autograd
// ecosystem.
//
// Including this single header exposes the entire log3 API:
//   #include "log3/log3.h"
//
// Platform & Compiler Compatibility:
//   * OS:       Linux, Android (NDK), Windows (Win32/Win64), macOS / Apple
//   * Arch:     x86_64 / x86 (SSE2 / AVX2 / FMA / AVX-512),
//               ARM64 / ARM32 (NEON / DOTPROD)
//   * Standard: C++17 (-std=c++17), 100% compatible with -fno-exceptions and
//               -fno-rtti, zero external dependencies.

#pragma once

#ifndef LOG3_UMBRELLA_LOG3_H_
#define LOG3_UMBRELLA_LOG3_H_

// =============================================================================
// Stage 1: Core Foundation (log3/core/*)
//   Ordered from lowest-level platform/type primitives up to concurrency & I/O
// =============================================================================
#include "log3/core/platform.h"
#include "log3/core/types.h"
#include "log3/core/bit_ops.h"
#include "log3/core/half.h"
#include "log3/core/dtype.h"
#include "log3/core/status.h"
#include "log3/core/log.h"
#include "log3/core/assert.h"
#include "log3/core/scope_guard.h"
#include "log3/core/memory.h"
#include "log3/core/arena.h"
#include "log3/core/ptr.h"
#include "log3/core/span.h"
#include "log3/core/str_view.h"
#include "log3/core/fn_ref.h"
#include "log3/core/small_vector.h"
#include "log3/core/flat_map.h"
#include "log3/core/cpu_features.h"
#include "log3/core/rng.h"
#include "log3/core/timer.h"
#include "log3/core/mmap.h"
#include "log3/core/thread_pool.h"

// =============================================================================
// Stage 2: N-D Tensor Engine & Low-Level SIMD Kernels
// =============================================================================
#include "log3/tensor/shape.h"
#include "log3/tensor/storage.h"
#include "log3/tensor/tensor.h"
#include "log3/kernels/cpu_kernels.h"

// =============================================================================
// Stage 3: BitNet b1.58 Neural Network Modules, Transformer & Zero-Copy I/O
// =============================================================================
#include "log3/io/checkpoint.h"
#include "log3/nn/module.h"
#include "log3/nn/layers.h"
#include "log3/nn/attention.h"
#include "log3/nn/transformer.h"

// =============================================================================
// Stage 4: Reverse-Mode Autograd, BitNet STE QAT & Optimizers
// =============================================================================
#include "log3/autograd/engine.h"
#include "log3/autograd/ops.h"
#include "log3/optim/optimizer.h"

// =============================================================================
// Composite Version Macros (Derived from log3/core/platform.h)
// =============================================================================
#ifndef L3_VERSION_HEX
#define L3_VERSION_HEX \
  ((L3_VERSION_MAJOR << 16) | (L3_VERSION_MINOR << 8) | (L3_VERSION_PATCH))
#endif

#ifndef L3_VERSION_STRING
#define L3_VERSION_STRING        \
  L3_STRINGIFY(L3_VERSION_MAJOR) \
  "." L3_STRINGIFY(L3_VERSION_MINOR) "." L3_STRINGIFY(L3_VERSION_PATCH)
#endif

// =============================================================================
// Top-Level Convenience Aliases & Runtime Helpers
// =============================================================================
namespace l3 {

namespace autograd {
// Convenience alias so both `l3::autograd::bitnet_linear_ste(...)` and
// `l3::autograd::ops::bitnet_linear_ste(...)` resolve cleanly.
namespace ops = ::l3::autograd;
}  // namespace autograd

// Returns the compile-time semantic version string (e.g., "0.1.0").
L3_NODISCARD L3_FORCE_INLINE constexpr const char* version_string() noexcept {
  return L3_VERSION_STRING;
}

// Returns the compile-time packed 24-bit version integer.
L3_NODISCARD L3_FORCE_INLINE constexpr u32 version_hex() noexcept {
  return static_cast<u32>(L3_VERSION_HEX);
}

// Returns a human-readable target OS name for runtime diagnostics.
L3_NODISCARD L3_FORCE_INLINE constexpr const char* platform_name() noexcept {
#if L3_OS_ANDROID
  return "Android";
#elif L3_OS_LINUX
  return "Linux";
#elif L3_OS_WINDOWS
  return "Windows";
#elif L3_OS_APPLE
  return "Apple (macOS/iOS)";
#else
  return "Unknown-OS";
#endif
}

// Returns a human-readable target CPU architecture name for runtime diagnostics.
L3_NODISCARD L3_FORCE_INLINE constexpr const char* arch_name() noexcept {
#if L3_ARCH_X64
  return "x86_64";
#elif L3_ARCH_X86
  return "x86 (32-bit)";
#elif L3_ARCH_ARM64
  return "ARM64 (AArch64)";
#elif L3_ARCH_ARM32
  return "ARM32";
#else
  return "Unknown-Arch";
#endif
}

// Initializes log3 runtime logging and warms up the CPU feature detector +
// global work-stealing thread pool so subsequent inference/training calls have
// zero cold-start overhead.
L3_FORCE_INLINE void init_runtime(LogLevel min_level = LogLevel::kInfo) noexcept {
  set_log_level(min_level);
  (void)cpu_features();
  (void)default_thread_pool();
}

}  // namespace l3

#endif  // LOG3_UMBRELLA_LOG3_H_
