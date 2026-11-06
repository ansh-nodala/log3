
// src/core/memory.cpp
//
// Aligned system memory allocation, default Allocator implementation, and
// atomic memory usage statistics.
//
// Platform details:
//   * Windows : _aligned_malloc / _aligned_free (<malloc.h>)
//   * POSIX   : posix_memalign / free (<cstdlib>), available across Linux,
//               macOS, and all Android NDK API levels.

#include "log3/core/memory.h"

#include <atomic>
#include <cstdlib>

#if L3_OS_WINDOWS
#include <malloc.h>
#endif

namespace l3 {

// -----------------------------------------------------------------------------
// Low-level aligned system allocation
// -----------------------------------------------------------------------------
void* alloc_aligned(Size size, Size alignment) noexcept {
  if (size == 0) return nullptr;
  L3_DCHECK(is_power_of_two(alignment));
  if (L3_UNLIKELY(!is_power_of_two(alignment))) return nullptr;

#if L3_OS_WINDOWS
  return ::_aligned_malloc(size, alignment);
#else
  // posix_memalign requires alignment to be a power of two AND a multiple of
  // sizeof(void*).
  const Size eff_align =
      alignment < sizeof(void*) ? sizeof(void*) : alignment;
  void* ptr = nullptr;
  if (::posix_memalign(&ptr, eff_align, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void free_aligned(void* ptr) noexcept {
  if (ptr == nullptr) return;
#if L3_OS_WINDOWS
  ::_aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

// -----------------------------------------------------------------------------
// Default allocator and statistics tracking
// -----------------------------------------------------------------------------
namespace {

#if L3_MEMORY_TRACKING
L3_ALIGNAS(kCacheLineSize) std::atomic<u64> g_current_bytes{0};
L3_ALIGNAS(kCacheLineSize) std::atomic<u64> g_peak_bytes{0};
L3_ALIGNAS(kCacheLineSize) std::atomic<u64> g_total_allocations{0};
L3_ALIGNAS(kCacheLineSize) std::atomic<u64> g_total_deallocations{0};
#endif

void* default_allocate(void* /*ctx*/, Size size, Size alignment) noexcept {
  void* ptr = alloc_aligned(size, alignment);
#if L3_MEMORY_TRACKING
  if (L3_LIKELY(ptr != nullptr)) {
    const u64 prev =
        g_current_bytes.fetch_add(static_cast<u64>(size), std::memory_order_relaxed);
    const u64 curr = prev + static_cast<u64>(size);

    u64 peak = g_peak_bytes.load(std::memory_order_relaxed);
    while (curr > peak &&
           !g_peak_bytes.compare_exchange_weak(
               peak, curr, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    g_total_allocations.fetch_add(1, std::memory_order_relaxed);
  }
#endif
  return ptr;
}

void default_deallocate(void* /*ctx*/, void* ptr, Size size,
                        Size /*alignment*/) noexcept {
  if (ptr == nullptr) return;
  free_aligned(ptr);
#if L3_MEMORY_TRACKING
  g_current_bytes.fetch_sub(static_cast<u64>(size), std::memory_order_relaxed);
  g_total_deallocations.fetch_add(1, std::memory_order_relaxed);
#endif
}

constexpr Allocator kDefaultAllocator = {
    &default_allocate,
    &default_deallocate,
    nullptr,
    "default",
};

}  // namespace

const Allocator& default_allocator() noexcept { return kDefaultAllocator; }

MemoryStats memory_stats() noexcept {
  MemoryStats stats{};
#if L3_MEMORY_TRACKING
  stats.current_bytes = g_current_bytes.load(std::memory_order_relaxed);
  stats.peak_bytes = g_peak_bytes.load(std::memory_order_relaxed);
  stats.total_allocations = g_total_allocations.load(std::memory_order_relaxed);
  stats.total_deallocations = g_total_deallocations.load(std::memory_order_relaxed);
#endif
  return stats;
}

void reset_peak_memory() noexcept {
#if L3_MEMORY_TRACKING
  const u64 current = g_current_bytes.load(std::memory_order_relaxed);
  g_peak_bytes.store(current, std::memory_order_relaxed);
#endif
}

}  // namespace l3
