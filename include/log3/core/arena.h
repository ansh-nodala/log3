// log3/core/arena.h
#pragma once

#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/types.h"

namespace l3 {

class L3_API Arena {
public:
  explicit Arena(Size chunk_size = 65536,
                 const Allocator& backing_allocator = default_allocator()) noexcept;

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  Arena(Arena&& other) noexcept;
  Arena& operator=(Arena&& other) noexcept;

  ~Arena();

  void* allocate(Size size, Size alignment = kDefaultAlignment) noexcept;
  void reset() noexcept;

  Size capacity_bytes() const noexcept { return total_capacity_bytes_; }
  Size allocated_bytes() const noexcept { return total_allocated_bytes_; }

  Allocator allocator() const noexcept {
    return Allocator{&Arena::allocate_trampoline, &Arena::deallocate_trampoline,
                     const_cast<Arena*>(this), "Arena"};
  }

private:
  struct Chunk;

  static void* allocate_trampoline(void* ctx, Size size, Size alignment) noexcept;
  static void deallocate_trampoline(void* ctx, void* ptr, Size size,
                                    Size alignment) noexcept;

  Status grow(Size min_size, Size alignment) noexcept;

  Size chunk_size_;
  Allocator backing_allocator_;
  Chunk* head_;
  Size total_capacity_bytes_;
  Size total_allocated_bytes_;
};

}  // namespace l3
