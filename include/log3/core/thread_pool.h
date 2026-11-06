// log3/core/thread_pool.h
//
// Low-latency data-parallel thread pool for BitNet ternary/INT4 GEMM, GEMV,
// attention heads, and elementwise tensor kernels.
//
// Why a custom ThreadPool instead of std::async / OpenMP:
//   * Zero per-dispatch heap allocation: tasks are dispatched via `FnRef`
//     (2 pointers wide) and an atomic work-stealing counter.
//   * Hybrid spin-then-sleep barrier: worker threads briefly spin with
//     CPU pause/yield instructions (`_mm_pause` / `yield`) between consecutive
//     layers of a transformer pass, avoiding 5-15 us OS futex/condvar wake-up
//     latency during token generation! If no new work arrives within the spin
//     window, workers sleep cleanly on an OS condition variable (0% idle CPU).
//   * Dynamic chunk stealing: `parallel_for(0, end, grain_size, fn)` divides
//      `[begin, end)` into chunks of `grain_size` and lets workers claim chunks
//     via a single cache-line-isolated `std::atomic<Index>`. Load imbalance
//     across P-cores and E-cores (Apple Silicon / Intel hybrid / Android
//     big.LITTLE) is automatically smoothed out.
//   * Caller participates as worker 0: `parallel_for` executes chunks on the
//     calling thread as well, so a 1-thread pool or small problem size pays
//     zero context-switch overhead.

#pragma once

#include "log3/core/fn_ref.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

// Signature of a range kernel invoked by ThreadPool::parallel_for:
//   thread_idx : 0 for the calling thread, 1..num_threads-1 for pool workers
//   begin, end : half-open sub-range [begin, end) assigned to this invocation
using ParallelRangeFn = FnRef<void(Index thread_idx, Index begin, Index end)>;

// Signature of a 1D index kernel invoked by ThreadPool::parallel_for_index:
using ParallelIndexFn = FnRef<void(Index thread_idx, Index i)>;

class L3_API ThreadPool {
 public:
  struct Impl;

  // Create a thread pool with `num_threads` total threads (including the
  // calling thread; i.e., `num_threads - 1` background worker threads are
  // spawned). If `num_threads <= 0`, defaults to `cpu_features().physical_cores`
  // (clamped to [1, 64]).
  static Result<ThreadPool> create(
      int num_threads = 0,
      const Allocator& allocator = default_allocator()) noexcept;

  ThreadPool() noexcept : impl_(nullptr), allocator_(nullptr) {}

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  ThreadPool(ThreadPool&& other) noexcept
      : impl_(other.impl_), allocator_(other.allocator_) {
    other.impl_ = nullptr;
    other.allocator_ = nullptr;
  }

  ThreadPool& operator=(ThreadPool&& other) noexcept {
    if (this != &other) {
      shutdown();
      impl_ = other.impl_;
      allocator_ = other.allocator_;
      other.impl_ = nullptr;
      other.allocator_ = nullptr;
    }
    return *this;
  }

  ~ThreadPool() noexcept { shutdown(); }

  // Total degree of parallelism (1 main thread + N-1 background workers).
  int num_threads() const noexcept;

  // Execute `fn(thread_idx, chunk_begin, chunk_end)` over `[begin, end)`.
  //   * If `end <= begin`, returns immediately.
  //   * If `grain_size <= 0`, automatically selects a balanced grain size
  //     (`div_ceil(end - begin, num_threads * 4)`).
  //   * Safe to call from multiple outer threads concurrently (serialized
  //     internally via a dispatch lock, or run inline if recursive).
  void parallel_for(Index begin, Index end, Index grain_size,
                    ParallelRangeFn fn) noexcept;

  // Convenience overload for per-index callbacks.
  void parallel_for_index(Index begin, Index end,
                          ParallelIndexFn fn) noexcept {
    parallel_for(begin, end, 0,
                 [&](Index tid, Index b, Index e) noexcept {
                   for (Index i = b; i < e; ++i) {
                     fn(tid, i);
                   }
                 });
  }

 private:
  void shutdown() noexcept;

  Impl* impl_;
  const Allocator* allocator_;
};

// Retrieve the global default ThreadPool (lazily initialized on first call,
// sized to `cpu_features().physical_cores`, thread-safe).
L3_API ThreadPool& default_thread_pool() noexcept;

// Top-level convenience wrapper using `default_thread_pool()`.
inline void parallel_for(Index begin, Index end, Index grain_size,
                         ParallelRangeFn fn) noexcept {
  default_thread_pool().parallel_for(begin, end, grain_size, fn);
}

}  // namespace l3
