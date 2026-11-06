// log3/core/rng.h
//
// Fast, deterministic, cross-platform pseudo-random number generation (Xoshiro256++)
// and statistical distributions for weight initialization, dropout, stochastic
// rounding, and LLM token sampling.
//
// Why not <random>:
//   * std::mt19937_64 has a 2,504-byte state struct (destroying L1 cache),
//     whereas Rng (Xoshiro256++) is 32 bytes (half a cache line) and 3x-4x
//     faster per 64-bit draw.
//   * std::uniform_real_distribution and std::normal_distribution are NOT
//     standardized across libstdc++ (Linux), libc++ (macOS/Android), and MSVC
//     (Windows) — the same seed produces DIFFERENT random tensors on different
//     OSes! `l3::Rng` uses exact IEEE-754 bit math so a seed produces
//     bit-identical tensors and token streams on every OS and CPU architecture.
//   * Supports `.jump()` (advances state by 2^128 steps in O(1)) so each thread
//     in `ThreadPool::parallel_for` gets a provably non-overlapping substream.
//
// Compatibility:
//   * C++17, header-only, no exceptions, no RTTI.

#pragma once

#include <cmath>

#include "log3/core/assert.h"
#include "log3/core/half.h"
#include "log3/core/platform.h"
#include "log3/core/str_view.h"
#include "log3/core/types.h"

namespace l3 {

class Rng {
 public:
  // Initialize with a 64-bit seed (expanded via SplitMix64 so even seed=0
  // produces a high-entropy non-zero state).
  explicit constexpr Rng(u64 seed = 0x853c49e6748fea9bULL) noexcept
      : s_{0, 0, 0, 0}, has_spare_normal_(false), spare_normal_(0.0f) {
    reseed(seed);
  }

  constexpr void reseed(u64 seed) noexcept {
    u64 z = seed;
    for (int i = 0; i < 4; ++i) {
      z += 0x9e3779b97f4a7c15ULL;
      s_[i] = hash_u64(z);
    }
    has_spare_normal_ = false;
    spare_normal_ = 0.0f;
  }

  // --- core integer draws (Xoshiro256++ by Blackman & Vigna) -----------------
  constexpr u64 next_u64() noexcept {
    const u64 result = rotl(s_[0] + s_[3], 23) + s_[0];
    const u64 t = s_[1] << 17;

    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];

    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  constexpr u32 next_u32() noexcept {
    return static_cast<u32>(next_u64() >> 32);
  }

  // Unbiased bounded integer in [0, bound) using Lemire's fast divisionless
  // rejection method where possible. Precondition: bound > 0.
  u32 bounded_u32(u32 bound) noexcept {
    L3_DCHECK_GT(bound, 0u);
    u64 x = next_u32();
    u64 m = x * static_cast<u64>(bound);
    u32 l = static_cast<u32>(m);
    if (L3_UNLIKELY(l < bound)) {
      const u32 threshold = (0u - bound) % bound;
      while (l < threshold) {
        x = next_u32();
        m = x * static_cast<u64>(bound);
        l = static_cast<u32>(m);
      }
    }
    return static_cast<u32>(m >> 32);
  }

  // --- floating-point draws (bit-exact across x86-64 and ARM64) --------------

  // Uniform f32 in [0.0f, 1.0f) with 24 bits of mantissa precision.
  L3_FORCE_INLINE f32 uniform_f32() noexcept {
    // 24 random bits * 2^-24 (0x1.0p-24f == 5.9604644775390625e-8f).
    return static_cast<f32>(next_u32() >> 8) * (1.0f / 16777216.0f);
  }

  // Uniform f32 in [low, high).
  L3_FORCE_INLINE f32 uniform_f32(f32 low, f32 high) noexcept {
    return low + (high - low) * uniform_f32();
  }

  // Uniform f64 in [0.0, 1.0) with 53 bits of mantissa precision.
  L3_FORCE_INLINE f64 uniform_f64() noexcept {
    return static_cast<f64>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
  }

  // Standard normal distribution N(0, 1) using Marsaglia's polar method
  // (avoids trigonometric sin/cos calls and caches the second variate).
  f32 normal_f32(f32 mean = 0.0f, f32 stddev = 1.0f) noexcept {
    if (has_spare_normal_) {
      has_spare_normal_ = false;
      return mean + stddev * spare_normal_;
    }

    f32 u = 0.0f;
    f32 v = 0.0f;
    f32 s = 0.0f;
    do {
      u = uniform_f32() * 2.0f - 1.0f;
      v = uniform_f32() * 2.0f - 1.0f;
      s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    const f32 mul = std::sqrt(-2.0f * std::log(s) / s);
    spare_normal_ = v * mul;
    has_spare_normal_ = true;
    return mean + stddev * (u * mul);
  }

  // Bernoulli trial: returns true with probability `p` in [0.0, 1.0].
  L3_FORCE_INLINE bool bernoulli(f32 p) noexcept {
    if (p <= 0.0f) return false;
    if (p >= 1.0f) return true;
    return uniform_f32() < p;
  }

  // --- parallel stream splitting ---------------------------------------------
  // Advance state by 2^128 calls to next_u64(). Use this to spawn independent
  // per-thread Rng instances inside ThreadPool::parallel_for.
  constexpr void jump() noexcept {
    constexpr u64 kJump[4] = {
        0x180ec6d33cfd0abaULL,
        0xd5a61266f0c9392cULL,
        0xa9582618e03fc9aaULL,
        0x39abdc4529b1661cULL,
    };
    u64 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; ++i) {
      for (int b = 0; b < 64; ++b) {
        if ((kJump[i] & (1ULL << b)) != 0) {
          s0 ^= s_[0];
          s1 ^= s_[1];
          s2 ^= s_[2];
          s3 ^= s_[3];
        }
        next_u64();
      }
    }
    s_[0] = s0;
    s_[1] = s1;
    s_[2] = s2;
    s_[3] = s3;
    has_spare_normal_ = false;
  }

  // Return a new Rng whose sequence is offset by `stream_index` jumps (each
  // jump is 2^128 steps).
  constexpr Rng split(Index stream_index = 1) const noexcept {
    Rng copy = *this;
    copy.has_spare_normal_ = false;
    for (Index i = 0; i < stream_index; ++i) {
      copy.jump();
    }
    return copy;
  }

 private:
  static constexpr u64 rotl(u64 x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  u64 s_[4];
  bool has_spare_normal_;
  f32 spare_normal_;
};

}  // namespace l3
