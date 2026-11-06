// src/core/thread_pool.cpp
//
// Implementation of the low-latency data-parallel ThreadPool.
//
// Platform threading primitives:
//   * POSIX (Linux, Android, macOS): pthread_t, pthread_mutex_t, pthread_cond_t
//   * Windows: _beginthreadex / WaitForSingleObject, SRWLOCK, CONDITION_VARIABLE
//
// Using native OS primitives directly keeps the binary free of <thread>,
// <mutex>, and <condition_variable> exception-throwing wrappers.

#include "log3/core/thread_pool.h"

#include <atomic>

#include "log3/core/assert.h"
#include "log3/core/cpu_features.h"

#if L3_OS_WINDOWS
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace l3 {

namespace {

inline constexpr int kMaxPoolThreads = 64;
inline constexpr u32 kSpinIterationsBeforeSleep = 2000;

L3_FORCE_INLINE void cpu_relax() noexcept {
#if L3_ARCH_X64 || L3_ARCH_X86
#if L3_MSVC_ABI
  ::_mm_pause();
#else
  __builtin_ia32_pause();
#endif
#elif L3_ARCH_ARM64
  asm volatile("yield" ::: "memory");
#endif
}

// Per-thread TLS flag to detect nested `parallel_for` calls from inside a
// worker and run them inline without deadlocking.
thread_local bool g_in_parallel_for = false;

}  // namespace

struct ThreadPool::Impl {
  int total_threads = 1;
  int num_workers = 0;  // total_threads - 1

  // Serializes outer callers of parallel_for on the same ThreadPool instance.
  L3_ALIGNAS(kCacheLineSize) std::atomic_flag dispatch_lock = ATOMIC_FLAG_INIT;

  // Synchronizes worker sleep/wake when spinning times out.
#if L3_OS_WINDOWS
  SRWLOCK mutex = SRWLOCK_INIT;
  CONDITION_VARIABLE cv_work = CONDITION_VARIABLE_INIT;
  HANDLE workers[kMaxPoolThreads]{};
#else
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t cv_work = PTHREAD_COND_INITIALIZER;
  pthread_t workers[kMaxPoolThreads]{};
#endif

  // Current batch descriptor (written before bumping `generation`).
  L3_ALIGNAS(kCacheLineSize) Index range_begin = 0;
  Index range_end = 0;
  Index grain_size = 1;
  ParallelRangeFn active_fn{};

  // Cache-line-isolated atomic cursor for dynamic work stealing across chunks.
  L3_ALIGNAS(kCacheLineSize) std::atomic<Index> next_chunk{0};

  // Generation counter: bumped by the main thread on each dispatch.
  L3_ALIGNAS(kCacheLineSize) std::atomic<u64> generation{0};

  // Number of workers that have NOT yet finished the current generation.
  L3_ALIGNAS(kCacheLineSize) std::atomic<int> active_workers{0};

  // Number of workers currently blocked in pthread_cond_wait / SleepConditionVariableSRW.
  L3_ALIGNAS(kCacheLineSize) std::atomic<int> sleeping_workers{0};

  std::atomic<bool> stop{false};

  struct WorkerArg {
    Impl* pool;
    int worker_id;  // 1 .. num_workers
  };
  WorkerArg worker_args[kMaxPoolThreads]{};

  // Execute available chunks for the current task until the range is exhausted.
  void run_chunks(Index thread_idx) noexcept {
    const Index end = range_end;
    const Index step = grain_size;
    const ParallelRangeFn fn = active_fn;

    for (;;) {
      const Index start = next_chunk.fetch_add(step, std::memory_order_relaxed);
      if (start >= end) break;
      const Index chunk_end = (end - start < step) ? end : (start + step);
      fn(thread_idx, start, chunk_end);
    }
  }

  void worker_loop(int worker_id) noexcept {
    g_in_parallel_for = true;
    u64 last_gen = 0;

    for (;;) {
      // Phase 1: Hybrid short spin waiting for next generation or stop.
      u64 cur_gen = generation.load(std::memory_order_acquire);
      if (cur_gen == last_gen && !stop.load(std::memory_order_relaxed)) {
        for (u32 spin = 0; spin < kSpinIterationsBeforeSleep; ++spin) {
          cpu_relax();
          cur_gen = generation.load(std::memory_order_acquire);
          if (cur_gen != last_gen || stop.load(std::memory_order_relaxed)) {
            break;
          }
        }
      }

      // Phase 2: Fall back to OS condvar wait if still idle after spin window.
      if (cur_gen == last_gen && !stop.load(std::memory_order_relaxed)) {
#if L3_OS_WINDOWS
        ::AcquireSRWLockExclusive(&mutex);
        sleeping_workers.fetch_add(1, std::memory_order_relaxed);
        while ((cur_gen = generation.load(std::memory_order_acquire)) ==
                   last_gen &&
               !stop.load(std::memory_order_relaxed)) {
          ::SleepConditionVariableSRW(&cv_work, &mutex, INFINITE, 0);
        }
        sleeping_workers.fetch_sub(1, std::memory_order_relaxed);
        ::ReleaseSRWLockExclusive(&mutex);
#else
        ::pthread_mutex_lock(&mutex);
        sleeping_workers.fetch_add(1, std::memory_order_relaxed);
        while ((cur_gen = generation.load(std::memory_order_acquire)) ==
                   last_gen &&
               !stop.load(std::memory_order_relaxed)) {
          ::pthread_cond_wait(&cv_work, &mutex);
        }
        sleeping_workers.fetch_sub(1, std::memory_order_relaxed);
        ::pthread_mutex_unlock(&mutex);
#endif
      }

      if (stop.load(std::memory_order_acquire)) {
        return;
      }

      last_gen = cur_gen;

      // Participate in the current batch.
      run_chunks(static_cast<Index>(worker_id));

      // Signal completion of this worker for the current generation.
      active_workers.fetch_sub(1, std::memory_order_release);
    }
  }

#if L3_OS_WINDOWS
  static unsigned __stdcall worker_entry(void* arg) noexcept {
    auto* wa = static_cast<WorkerArg*>(arg);
    wa->pool->worker_loop(wa->worker_id);
    return 0;
  }
#else
  static void* worker_entry(void* arg) noexcept {
    auto* wa = static_cast<WorkerArg*>(arg);
    wa->pool->worker_loop(wa->worker_id);
    return nullptr;
  }
#endif
};

Result<ThreadPool> ThreadPool::create(int num_threads,
                                      const Allocator& allocator) noexcept {
  if (num_threads <= 0) {
    num_threads = cpu_features().physical_cores;
    if (num_threads <= 0) num_threads = 4;
  }
  if (num_threads > kMaxPoolThreads) {
    num_threads = kMaxPoolThreads;
  }

  Impl* impl = create_object<Impl>(allocator);
  if (L3_UNLIKELY(impl == nullptr)) {
    return Status::error(StatusCode::kOutOfMemory,
                         "ThreadPool::create: failed to allocate Impl");
  }

  impl->total_threads = num_threads;
  impl->num_workers = num_threads - 1;

#if !L3_OS_WINDOWS
  ::pthread_mutex_init(&impl->mutex, nullptr);
  ::pthread_cond_init(&impl->cv_work, nullptr);
#endif

  int spawned = 0;
  for (int i = 0; i < impl->num_workers; ++i) {
    impl->worker_args[i].pool = impl;
    impl->worker_args[i].worker_id = i + 1;

#if L3_OS_WINDOWS
    uintptr_t h = ::_beginthreadex(nullptr, 0, &Impl::worker_entry,
                                   &impl->worker_args[i], 0, nullptr);
    if (h == 0) break;
    impl->workers[i] = reinterpret_cast<HANDLE>(h);
#else
    if (::pthread_create(&impl->workers[i], nullptr, &Impl::worker_entry,
                         &impl->worker_args[i]) != 0) {
      break;
    }
#endif
    ++spawned;
  }

  if (spawned < impl->num_workers) {
    impl->num_workers = spawned;
    impl->total_threads = spawned + 1;
  }

  ThreadPool pool;
  pool.impl_ = impl;
  pool.allocator_ = &allocator;
  return pool;
}

void ThreadPool::shutdown() noexcept {
  if (impl_ == nullptr) return;

#if L3_OS_WINDOWS
  ::AcquireSRWLockExclusive(&impl_->mutex);
  impl_->stop.store(true, std::memory_order_release);
  ::WakeAllConditionVariable(&impl_->cv_work);
  ::ReleaseSRWLockExclusive(&impl_->mutex);

  for (int i = 0; i < impl_->num_workers; ++i) {
    ::WaitForSingleObject(impl_->workers[i], INFINITE);
    ::CloseHandle(impl_->workers[i]);
  }
#else
  ::pthread_mutex_lock(&impl_->mutex);
  impl_->stop.store(true, std::memory_order_release);
  ::pthread_cond_broadcast(&impl_->cv_work);
  ::pthread_mutex_unlock(&impl_->mutex);

  for (int i = 0; i < impl_->num_workers; ++i) {
    ::pthread_join(impl_->workers[i], nullptr);
  }
  ::pthread_cond_destroy(&impl_->cv_work);
  ::pthread_mutex_destroy(&impl_->mutex);
#endif

  const Allocator& alloc =
      allocator_ != nullptr ? *allocator_ : default_allocator();
  destroy_object(alloc, impl_);
  impl_ = nullptr;
  allocator_ = nullptr;
}

int ThreadPool::num_threads() const noexcept {
  return impl_ != nullptr ? impl_->total_threads : 1;
}

void ThreadPool::parallel_for(Index begin, Index end, Index grain_size,
                              ParallelRangeFn fn) noexcept {
  if (end <= begin || !fn.is_valid()) return;

  const Index total_items = end - begin;
  const int nthreads = num_threads();

  if (grain_size <= 0) {
    const Index target_chunks = static_cast<Index>(nthreads) * 4;
    grain_size =
        static_cast<Index>(div_ceil(static_cast<Size>(total_items),
                                    static_cast<Size>(target_chunks)));
    if (grain_size < 1) grain_size = 1;
  }

  // Fast single-thread path if pool has 1 thread, range fits in 1 chunk, or
  // called recursively from inside an active worker.
  if (nthreads <= 1 || total_items <= grain_size || g_in_parallel_for) {
    fn(0, begin, end);
    return;
  }

  // Acquire dispatch lock so concurrent outer calls serialize cleanly.
  while (impl_->dispatch_lock.test_and_set(std::memory_order_acquire)) {
    cpu_relax();
  }

  impl_->range_begin = begin;
  impl_->range_end = end;
  impl_->grain_size = grain_size;
  impl_->active_fn = fn;
  impl_->next_chunk.store(begin, std::memory_order_relaxed);
  impl_->active_workers.store(impl_->num_workers, std::memory_order_relaxed);

  // Publish new generation to wake spinning workers immediately.
  impl_->generation.fetch_add(1, std::memory_order_release);

  // If any workers are sleeping on the OS condvar, wake them up.
  if (impl_->sleeping_workers.load(std::memory_order_acquire) > 0) {
#if L3_OS_WINDOWS
    ::AcquireSRWLockExclusive(&impl_->mutex);
    ::WakeAllConditionVariable(&impl_->cv_work);
    ::ReleaseSRWLockExclusive(&impl_->mutex);
#else
    ::pthread_mutex_lock(&impl_->mutex);
    ::pthread_cond_broadcast(&impl_->cv_work);
    ::pthread_mutex_unlock(&impl_->mutex);
#endif
  }

  // Calling thread participates as thread 0.
  const bool prev_tls = g_in_parallel_for;
  g_in_parallel_for = true;
  impl_->run_chunks(0);
  g_in_parallel_for = prev_tls;

  // Wait for all workers to finish their final chunk.
  while (impl_->active_workers.load(std::memory_order_acquire) > 0) {
    cpu_relax();
  }

  impl_->dispatch_lock.clear(std::memory_order_release);
}

ThreadPool& default_thread_pool() noexcept {
  static ThreadPool s_pool = []() {
    auto res = ThreadPool::create(0, default_allocator());
    L3_CHECK_OK(res.status());
    return res.take_value();
  }();
  return s_pool;
}

}  // namespace l3
