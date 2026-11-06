// log3/core/memory.h
//
// Memory foundation: alignment math, overflow-checked size math, aligned
// allocation, the Allocator interface, memory statistics and small typed
// helpers. Tensor Storage, graph nodes, arenas and pools are all built on this.
//
// Conventions (the whole project follows them):
//   * Allocation failure is reported as nullptr, never by throwing. Callers
//     turn it into Status::error(StatusCode::kOutOfMemory, ...).
//   * Allocating 0 bytes returns nullptr, and deallocating nullptr is a no-op.
//   * alignment is always a power of two.
//   * Deallocation passes the original size and alignment ("sized
//     deallocation"). That lets pool/arena allocators pick the right size
//     class without storing a per-allocation header.
//   * Alignment defaults to kDefaultAlignment (64 bytes): one cache line, and
//     enough for aligned AVX-512 loads/stores on tensor data.
//
// Allocator is a plain struct of function pointers + context: no virtual
// calls through a vtable pointer chain, no std::function, no std::allocator,
// trivially copyable, constexpr-constructible, and cheap to pass by value.

#pragma once

#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/types.h"

// Track allocation statistics in the default allocator. Costs two relaxed
// atomic adds per allocation; allocations here are large (tensors), so this is
// negligible, and peak-memory numbers matter a lot on edge devices.
#ifndef L3_MEMORY_TRACKING
#define L3_MEMORY_TRACKING 1
#endif

namespace l3 {

// Smallest alignment handed out for ordinary objects (not tensor data).
inline constexpr Size kMinObjectAlignment = 16;

// -----------------------------------------------------------------------------
// Alignment helpers (constexpr, header-only)
//   Preconditions: alignment is a power of two; align_up must not overflow.
// -----------------------------------------------------------------------------
constexpr bool is_power_of_two(Size x) noexcept {
  return x != 0 && (x & (x - 1)) == 0;
}

constexpr Size align_up(Size value, Size alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

constexpr Size align_down(Size value, Size alignment) noexcept {
  return value & ~(alignment - 1);
}

inline bool is_aligned(const void* ptr, Size alignment) noexcept {
  return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ceil(a / b) without the (a + b - 1) overflow. b must be non-zero.
constexpr Size div_ceil(Size a, Size b) noexcept {
  return a / b + (a % b != 0 ? 1 : 0);
}

// -----------------------------------------------------------------------------
// Overflow-checked size math. Return false (and leave *out untouched) on
// overflow. Use these for every size computed from user-provided counts.
// -----------------------------------------------------------------------------
inline bool checked_add(Size a, Size b, Size* out) noexcept {
  if (b > static_cast<Size>(-1) - a) return false;
  *out = a + b;
  return true;
}

inline bool checked_mul(Size a, Size b, Size* out) noexcept {
  if (a != 0 && b > static_cast<Size>(-1) / a) return false;
  *out = a * b;
  return true;
}

// -----------------------------------------------------------------------------
// Raw byte operations (null/zero-size safe: memcpy(nullptr, nullptr, 0) is
// undefined behavior in C, these are not).
// -----------------------------------------------------------------------------
inline void mem_copy(void* dst, const void* src, Size n) noexcept {
  if (n != 0) std::memcpy(dst, src, n);
}

// dst and src may overlap.
inline void mem_move(void* dst, const void* src, Size n) noexcept {
  if (n != 0) std::memmove(dst, src, n);
}

inline void mem_zero(void* dst, Size n) noexcept {
  if (n != 0) std::memset(dst, 0, n);
}

inline void mem_fill(void* dst, u8 value, Size n) noexcept {
  if (n != 0) std::memset(dst, value, n);
}

// -----------------------------------------------------------------------------
// Aligned allocation from the system (thread-safe)
//   Windows : _aligned_malloc / _aligned_free
//   POSIX   : posix_memalign / free (also available on every Android API
//             level; C11 aligned_alloc only exists from Android API 28)
// Returns nullptr on failure or when size == 0. Memory is NOT zeroed.
// Memory from alloc_aligned must be released with free_aligned.
// -----------------------------------------------------------------------------
L3_API void* alloc_aligned(Size size, Size alignment = kDefaultAlignment) noexcept;
L3_API void free_aligned(void* ptr) noexcept;

// -----------------------------------------------------------------------------
// Allocator interface
//
//   allocate_fn   : returns memory of at least `size` bytes aligned to
//                   `alignment`, or nullptr (also for size == 0).
//   deallocate_fn : receives the SAME size and alignment that were passed to
//                   allocate_fn. Never called with a null pointer.
//   ctx           : opaque state for the implementation (arena, pool, ...).
//   name          : static string for logs and debugging.
//
// Implementations must be safe to call from any thread unless documented
// otherwise (an arena used by one thread can say so in its own header).
// -----------------------------------------------------------------------------
struct Allocator {
  using AllocateFn = void* (*)(void* ctx, Size size, Size alignment) noexcept;
  using DeallocateFn = void (*)(void* ctx, void* ptr, Size size,
                                Size alignment) noexcept;

  AllocateFn allocate_fn;
  DeallocateFn deallocate_fn;
  void* ctx;
  const char* name;

  void* allocate(Size size, Size alignment = kDefaultAlignment) const noexcept {
    L3_DCHECK(allocate_fn != nullptr);
    L3_DCHECK(is_power_of_two(alignment));
    return allocate_fn(ctx, size, alignment);
  }

  void deallocate(void* ptr, Size size,
                  Size alignment = kDefaultAlignment) const noexcept {
    if (ptr == nullptr) return;
    L3_DCHECK(deallocate_fn != nullptr);
    deallocate_fn(ctx, ptr, size, alignment);
  }

  bool is_valid() const noexcept {
    return allocate_fn != nullptr && deallocate_fn != nullptr;
  }
};

inline bool operator==(const Allocator& a, const Allocator& b) noexcept {
  return a.allocate_fn == b.allocate_fn && a.deallocate_fn == b.deallocate_fn &&
         a.ctx == b.ctx;
}
inline bool operator!=(const Allocator& a, const Allocator& b) noexcept {
  return !(a == b);
}

// System allocator (alloc_aligned/free_aligned) with statistics tracking.
// Never null, valid for the whole program lifetime, thread-safe.
L3_API const Allocator& default_allocator() noexcept;

// -----------------------------------------------------------------------------
// Memory statistics of the default allocator (all zeros if
// L3_MEMORY_TRACKING is 0).
// -----------------------------------------------------------------------------
struct MemoryStats {
  u64 current_bytes;        // bytes currently allocated
  u64 peak_bytes;           // highest current_bytes since start / last reset
  u64 total_allocations;    // number of successful allocations
  u64 total_deallocations;  // number of deallocations
};

L3_API MemoryStats memory_stats() noexcept;

// Set peak_bytes back to current_bytes (measure the peak of one phase).
L3_API void reset_peak_memory() noexcept;

// -----------------------------------------------------------------------------
// Typed helpers
// -----------------------------------------------------------------------------

// Uninitialized storage for `count` objects of trivially destructible type T.
// Returns nullptr on failure, on count == 0, or if count * sizeof(T) overflows.
// Pair with deallocate_array using the same count.
template <typename T>
T* allocate_array(const Allocator& allocator, Size count) noexcept {
  static_assert(std::is_trivially_destructible_v<T>,
                "allocate_array returns raw storage; T must be trivially "
                "destructible (use create_object for other types).");
  Size bytes = 0;
  if (!checked_mul(count, sizeof(T), &bytes)) return nullptr;
  constexpr Size kAlign =
      alignof(T) > kDefaultAlignment ? alignof(T) : kDefaultAlignment;
  return static_cast<T*>(allocator.allocate(bytes, kAlign));
}

template <typename T>
void deallocate_array(const Allocator& allocator, T* ptr, Size count) noexcept {
  constexpr Size kAlign =
      alignof(T) > kDefaultAlignment ? alignof(T) : kDefaultAlignment;
  allocator.deallocate(ptr, count * sizeof(T), kAlign);
}

// Allocate and construct one object. Returns nullptr if allocation fails.
// T's constructor must not throw (the core is built without exceptions).
template <typename T, typename... Args>
T* create_object(const Allocator& allocator, Args&&... args) noexcept {
  constexpr Size kAlign =
      alignof(T) > kMinObjectAlignment ? alignof(T) : kMinObjectAlignment;
  void* mem = allocator.allocate(sizeof(T), kAlign);
  if (mem == nullptr) return nullptr;
  return ::new (mem) T(std::forward<Args>(args)...);
}

// Destroy and free an object made by create_object with the same allocator.
// Null is allowed.
template <typename T>
void destroy_object(const Allocator& allocator, T* ptr) noexcept {
  if (ptr == nullptr) return;
  constexpr Size kAlign =
      alignof(T) > kMinObjectAlignment ? alignof(T) : kMinObjectAlignment;
  ptr->~T();
  allocator.deallocate(ptr, sizeof(T), kAlign);
}

}  // namespace l3
