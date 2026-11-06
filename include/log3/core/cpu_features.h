// log3/core/cpu_features.h
//
// Runtime CPU instruction set architecture (ISA) and cache topology detection.
//
// Design notes:
//   * Detected once at startup (or on first query) in a thread-safe manner and
//     cached in a global structure, so hot-path kernel dispatch costs zero overhead
//     (just a boolean check or bitfield lookup).
//   * Supports x86-64 (SSE2, AVX, AVX2, AVX-512F, AVX-512VNNI, F16C, FMA) and
//     ARM64 (NEON, FP16 arithmetic, DotProd, I8MM for sub-byte quantization).

#pragma once

#include "log3/core/platform.h"
#include "log3/core/types.h"

namespace l3 {

struct CpuFeatures {
  // x86 / x64 features
  bool has_sse2;
  bool has_sse41;
  bool has_sse42;
  bool has_avx;
  bool has_avx2;
  bool has_fma;
  bool has_f16c;
  bool has_avx512f;
  bool has_avx512bw;
  bool has_avx512dq;
  bool has_avx512_vnni;  // Vector Neural Network Instructions (int8/int4 dp)

  // ARM / ARM64 features
  bool has_neon;
  bool has_neon_fp16;
  bool has_neon_dotprod;  // UDOT/SDOT (8-bit dot products)
  bool has_neon_i8mm;     // Matmul extension (AArch64 v8.6+)

  // Cache & topology
  int logical_cores;
  int physical_cores;
  usize l1d_cache_size;  // L1 data cache size in bytes (0 if unknown)
  usize l2_cache_size;   // L2 cache size in bytes (0 if unknown)
  usize l3_cache_size;   // L3 cache size in bytes (0 if unknown)
  usize cache_line_size; // Typically 64 bytes
};

// Retrieve the singleton CpuFeatures structure (computed once, thread-safe).
L3_API const CpuFeatures& cpu_features() noexcept;

}  // namespace l3
