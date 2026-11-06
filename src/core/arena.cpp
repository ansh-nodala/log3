// log3/core/arena.cpp
#include "log3/core/arena.h"

#include <cstdlib>
#include <memory>
#include <new>

namespace l3 {

struct Arena::Chunk {
  Chunk* next;
  Size capacity;
  Size used;
  alignas(kDefaultAlignment) byte data[1];
};

Arena::Arena(Size chunk_size, const Allocator& backing_allocator) noexcept
    : chunk_size_(chunk_size < 4096 ? 4096 : chunk_size),
      backing_allocator_(backing_allocator),
      head_(nullptr),
      total_capacity_bytes_(0),
      total_allocated_bytes_(0) {}

Arena::Arena(Arena&& other) noexcept
    : chunk_size_(other.chunk_size_),
      backing_allocator_(other.backing_allocator_),
      head_(other.head_),
      total_capacity_bytes_(other.total_capacity_bytes_),
      total_allocated_bytes_(other.total_allocated_bytes_) {
  other.head_ = nullptr;
  other.total_capacity_bytes_ = 0;
  other.total_allocated_bytes_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
  if (this != &other) {
    reset();
    Chunk* curr = head_;
    while (curr != nullptr) {
      Chunk* next = curr->next;
      backing_allocator_.deallocate(curr, sizeof(Chunk) + curr->capacity,
                                    kDefaultAlignment);
      curr = next;
    }
    chunk_size_ = other.chunk_size_;
    backing_allocator_ = other.backing_allocator_;
    head_ = other.head_;
    total_capacity_bytes_ = other.total_capacity_bytes_;
    total_allocated_bytes_ = other.total_allocated_bytes_;

    other.head_ = nullptr;
    other.total_capacity_bytes_ = 0;
    other.total_allocated_bytes_ = 0;
  }
  return *this;
}

Arena::~Arena() {
  Chunk* curr = head_;
  while (curr != nullptr) {
    Chunk* next = curr->next;
    backing_allocator_.deallocate(curr, sizeof(Chunk) + curr->capacity,
                                  kDefaultAlignment);
    curr = next;
  }
}

void* Arena::allocate(Size size, Size alignment) noexcept {
  if (size == 0) return nullptr;
  L3_DCHECK(is_power_of_two(alignment));

  if (head_ != nullptr) {
    byte* raw_data = head_->data + head_->used;
    usize space = head_->capacity - head_->used;
    void* ptr = raw_data;
    if (std::align(alignment, size, ptr, space) != nullptr) {
      Size padding = static_cast<Size>(static_cast<byte*>(ptr) - raw_data);
      head_->used += padding + size;
      total_allocated_bytes_ += size;
      return ptr;
    }
  }

  Status grow_status = grow(size, alignment);
  if (L3_UNLIKELY(!grow_status.is_ok())) {
    return nullptr;
  }

  byte* raw_data = head_->data + head_->used;
  usize space = head_->capacity - head_->used;
  void* ptr = raw_data;
  void* aligned_ptr = std::align(alignment, size, ptr, space);
  L3_DCHECK(aligned_ptr != nullptr);
  Size padding = static_cast<Size>(static_cast<byte*>(aligned_ptr) - raw_data);
  head_->used += padding + size;
  total_allocated_bytes_ += size;
  return aligned_ptr;
}

void Arena::reset() noexcept {
  Chunk* curr = head_;
  while (curr != nullptr) {
    curr->used = 0;
    curr = curr->next;
  }
  total_allocated_bytes_ = 0;
}

Status Arena::grow(Size min_size, Size alignment) noexcept {
  Size needed = min_size + alignment + sizeof(Chunk);
  Size alloc_cap = chunk_size_;
  if (needed > alloc_cap) {
    alloc_cap = needed;
  }

  void* mem = backing_allocator_.allocate(sizeof(Chunk) + alloc_cap, kDefaultAlignment);
  if (L3_UNLIKELY(mem == nullptr)) {
    return Status::error(StatusCode::kOutOfMemory, "Arena: failed to allocate chunk");
  }

  auto* new_chunk = static_cast<Chunk*>(mem);
  new_chunk->next = head_;
  new_chunk->capacity = alloc_cap;
  new_chunk->used = 0;
  head_ = new_chunk;
  total_capacity_bytes_ += alloc_cap;
  return Status::ok();
}

void* Arena::allocate_trampoline(void* ctx, Size size, Size alignment) noexcept {
  return static_cast<Arena*>(ctx)->allocate(size, alignment);
}

void Arena::deallocate_trampoline(void* /*ctx*/, void* /*ptr*/, Size /*size*/,
                                  Size /*alignment*/) noexcept {
  // Arena deallocation is a no-op per individual block.
}

}  // namespace l3
