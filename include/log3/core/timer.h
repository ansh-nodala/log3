// log3/core/timer.h
//
// High-resolution monotonic clock, Stopwatch, and bandwidth/FLOP/s calculators
// for profiling BitNet kernels, model loading, and token/s generation speed.
//
// Why not <chrono>:
//   * Avoids pulling in <chrono> and <ratio> template machinery across every
//     translation unit.
//   * Uses direct OS monotonic hardware counters (`CLOCK_MONOTONIC` on
//     Linux/Android/macOS, `QueryPerformanceCounter` on Windows) that never
//     jump backward on NTP clock adjustments.
//   * Built-in helpers for effective memory bandwidth (GB/s), compute
//     throughput (GOPS / GFLOPS), and tokens/sec.
//
// Compatibility:
//   * C++17, header-only, no exceptions, no RTTI.

#pragma once

#include <ctime>

#include "log3/core/platform.h"
#include "log3/core/types.h"

#if L3_OS_WINDOWS
#include <windows.h>
#endif

namespace l3 {

// Monotonic timestamp in nanoseconds (for measuring elapsed intervals).
inline u64 monotonic_now_ns() noexcept {
#if L3_OS_WINDOWS
  static const LARGE_INTEGER s_freq = []() {
    LARGE_INTEGER f{};
    ::QueryPerformanceFrequency(&f);
    return f;
  }();
  LARGE_INTEGER counter{};
  ::QueryPerformanceCounter(&counter);
  if (s_freq.QuadPart <= 0) return 0;
  // Avoid u64 overflow by splitting quotient and remainder.
  const u64 whole_secs =
      static_cast<u64>(counter.QuadPart) / static_cast<u64>(s_freq.QuadPart);
  const u64 rem_ticks =
      static_cast<u64>(counter.QuadPart) % static_cast<u64>(s_freq.QuadPart);
  return whole_secs * 1000000000ULL +
         (rem_ticks * 1000000000ULL) / static_cast<u64>(s_freq.QuadPart);
#else
  struct timespec ts {};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<u64>(ts.tv_sec) * 1000000000ULL +
         static_cast<u64>(ts.tv_nsec);
#endif
}

// Lightweight stopwatch for kernel and layer benchmarking.
class Stopwatch {
 public:
  Stopwatch() noexcept : start_ns_(monotonic_now_ns()) {}

  void reset() noexcept { start_ns_ = monotonic_now_ns(); }

  u64 elapsed_ns() const noexcept {
    const u64 now = monotonic_now_ns();
    return now >= start_ns_ ? (now - start_ns_) : 0ULL;
  }

  f64 elapsed_us() const noexcept {
    return static_cast<f64>(elapsed_ns()) * 1e-3;
  }

  f64 elapsed_ms() const noexcept {
    return static_cast<f64>(elapsed_ns()) * 1e-6;
  }

  f64 elapsed_seconds() const noexcept {
    return static_cast<f64>(elapsed_ns()) * 1e-9;
  }

 private:
  u64 start_ns_;
};

// Compute effective memory bandwidth in Gigabytes/sec (10^9 B/s) given bytes
// transferred and elapsed nanoseconds.
constexpr f64 compute_gbps(Size bytes_transferred, u64 elapsed_ns) noexcept {
  return elapsed_ns > 0
             ? static_cast<f64>(bytes_transferred) / static_cast<f64>(elapsed_ns)
             : 0.0;
}

// Compute effective Giga-Operations/sec (GOPS / GFLOPS, 10^9 ops/s) given
// operation count and elapsed nanoseconds.
constexpr f64 compute_gops(u64 total_ops, u64 elapsed_ns) noexcept {
  return elapsed_ns > 0
             ? static_cast<f64>(total_ops) / static_cast<f64>(elapsed_ns)
             : 0.0;
}

}  // namespace l3
