// log3/core/small_vector.h
//
// SmallVec<T, N>: a growable array that keeps its first N elements INSIDE the
// object (no heap allocation) and only touches the heap when it grows past N.
// Built for shapes and strides: a tensor with <= 8 dims never allocates.
//
//   SmallVec<Index, 8> shape;
//   L3_RETURN_IF_ERROR(shape.push_back(32));
//   L3_RETURN_IF_ERROR(shape.push_back(128));
//   Span<const Index> view(shape);          // converts to Span (data()/size())
//
// log3 conventions (this class is where they matter most):
//
//   * No exceptions. Every operation that may need memory returns Status
//     (kOutOfMemory, or kOutOfRange on size overflow). On failure the
//     container is left UNCHANGED (strong guarantee), so a failed push_back
//     never loses data.
//   * Move-only. Copying can allocate (and therefore fail), so there is no
//     copy constructor. Use clone() / assign() / from_span(), which report
//     errors. Moves are noexcept and never allocate.
//   * size()/capacity()/indices are signed Index, like Span and shapes.
//   * Allocator-aware: grows through log3's Allocator (default_allocator()
//     unless you pass one). The Allocator must outlive the SmallVec. A moved
//     SmallVec carries its allocator with it.
//   * T must be nothrow-move-constructible (the core never throws, so growth
//     cannot be allowed to fail halfway). Trivially copyable T (Index, f32,
//     ...) is relocated with a single memcpy.
//   * Aliasing is safe where it is cheap to be safe: v.push_back(v[0]) and
//     v.insert(i, v[j]) work even when they reallocate. assign()/append() with
//     a Span that points INTO the same vector is not supported.
//
// Bounds are checked with L3_DCHECK (debug only), like Span.

#pragma once

#include <new>
#include <type_traits>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

template <typename T, Index N>
class SmallVec {
  static_assert(N > 0, "SmallVec: inline capacity N must be at least 1.");
  static_assert(!std::is_const_v<T> && !std::is_reference_v<T>,
                "SmallVec<T>: T must be a non-const object type.");
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "SmallVec<T>: T must be nothrow move constructible.");
  static_assert(std::is_nothrow_destructible_v<T>,
                "SmallVec<T>: T must be nothrow destructible.");

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  static constexpr Index kInlineCapacity = N;

  // --- construction / destruction -------------------------------------------
  SmallVec() noexcept
      : data_(inline_data()), size_(0), capacity_(N), allocator_(nullptr) {}

  // `allocator` must outlive this SmallVec (and anything moved out of it).
  explicit SmallVec(const Allocator& allocator) noexcept
      : data_(inline_data()), size_(0), capacity_(N), allocator_(&allocator) {}

  SmallVec(SmallVec&& other) noexcept
      : data_(inline_data()),
        size_(0),
        capacity_(N),
        allocator_(other.allocator_) {
    take(other);
  }

  SmallVec& operator=(SmallVec&& other) noexcept {
    if (this != &other) {
      clear();
      release_heap();
      allocator_ = other.allocator_;
      take(other);
    }
    return *this;
  }

  SmallVec(const SmallVec&) = delete;
  SmallVec& operator=(const SmallVec&) = delete;

  ~SmallVec() {
    clear();
    release_heap();
  }

  // Explicit, fallible copies.
  Result<SmallVec> clone() const noexcept {
    SmallVec out(allocator());
    L3_RETURN_IF_ERROR(out.assign(span()));
    return out;
  }

  static Result<SmallVec> from_span(
      Span<const T> src, const Allocator& allocator = default_allocator()) noexcept {
    SmallVec out(allocator);
    L3_RETURN_IF_ERROR(out.assign(src));
    return out;
  }

  // --- observers --------------------------------------------------------------
  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  Index size() const noexcept { return size_; }
  Index capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }

  // True while the elements live inside the object (no heap block in use).
  bool is_inline() const noexcept { return data_ == inline_data(); }

  const Allocator& get_allocator() const noexcept { return allocator(); }

  Span<T> span() noexcept { return Span<T>(data_, size_); }
  Span<const T> span() const noexcept { return Span<const T>(data_, size_); }

  // --- element access -----------------------------------------------------------
  T& operator[](Index i) noexcept {
    L3_DCHECK_GE(i, 0);
    L3_DCHECK_LT(i, size_);
    return data_[i];
  }
  const T& operator[](Index i) const noexcept {
    L3_DCHECK_GE(i, 0);
    L3_DCHECK_LT(i, size_);
    return data_[i];
  }

  T& front() noexcept {
    L3_DCHECK_GT(size_, 0);
    return data_[0];
  }
  const T& front() const noexcept {
    L3_DCHECK_GT(size_, 0);
    return data_[0];
  }
  T& back() noexcept {
    L3_DCHECK_GT(size_, 0);
    return data_[size_ - 1];
  }
  const T& back() const noexcept {
    L3_DCHECK_GT(size_, 0);
    return data_[size_ - 1];
  }

  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept { return data_ + size_; }

  // --- capacity -------------------------------------------------------------------
  // Make room for at least n elements (exact, like std::vector::reserve).
  Status reserve(Index n) noexcept {
    L3_DCHECK_GE(n, 0);
    if (n <= capacity_) return Status::ok();
    return grow_to(n);
  }

  // --- modifiers --------------------------------------------------------------------
  void clear() noexcept {
    destroy_range(data_, size_);
    size_ = 0;
  }

  template <typename... Args>
  Status emplace_back(Args&&... args) noexcept {
    if (L3_LIKELY(size_ < capacity_)) {
      ::new (static_cast<void*>(data_ + size_)) T(std::forward<Args>(args)...);
      ++size_;
      return Status::ok();
    }
    return emplace_back_slow(std::forward<Args>(args)...);
  }

  Status push_back(const T& value) noexcept { return emplace_back(value); }
  Status push_back(T&& value) noexcept { return emplace_back(std::move(value)); }

  void pop_back() noexcept {
    L3_DCHECK_GT(size_, 0);
    --size_;
    data_[size_].~T();
  }

  // Grow with value-initialized elements (zeros for numbers) or shrink.
  Status resize(Index n) noexcept {
    L3_DCHECK_GE(n, 0);
    if (n <= size_) {
      destroy_range(data_ + n, size_ - n);
      size_ = n;
      return Status::ok();
    }
    L3_RETURN_IF_ERROR(ensure_capacity(n));
    for (Index i = size_; i < n; ++i) ::new (static_cast<void*>(data_ + i)) T();
    size_ = n;
    return Status::ok();
  }

  Status resize(Index n, const T& value) noexcept {
    L3_DCHECK_GE(n, 0);
    if (n <= size_) {
      destroy_range(data_ + n, size_ - n);
      size_ = n;
      return Status::ok();
    }
    T copy(value);  // `value` may live inside this vector
    L3_RETURN_IF_ERROR(ensure_capacity(n));
    for (Index i = size_; i < n; ++i) ::new (static_cast<void*>(data_ + i)) T(copy);
    size_ = n;
    return Status::ok();
  }

  // Replace the contents with a copy of src. src must not point into *this.
  Status assign(Span<const T> src) noexcept {
    L3_RETURN_IF_ERROR(ensure_capacity(src.size()));  // before touching data
    clear();
    copy_construct(data_, src.data(), src.size());
    size_ = src.size();
    return Status::ok();
  }

  // Append a copy of src. src must not point into *this.
  Status append(Span<const T> src) noexcept {
    Size total = 0;
    if (!checked_add(static_cast<Size>(size_), static_cast<Size>(src.size()), &total) ||
        total > static_cast<Size>(kMaxIndex)) {
      return Status::error(StatusCode::kOutOfRange, "SmallVec: size overflow");
    }
    L3_RETURN_IF_ERROR(ensure_capacity(static_cast<Index>(total)));
    copy_construct(data_ + size_, src.data(), src.size());
    size_ += src.size();
    return Status::ok();
  }

  // Insert value before position pos (0 <= pos <= size()). Later elements
  // shift right. `value` is taken by value so it can alias an element.
  Status insert(Index pos, T value) noexcept {
    L3_DCHECK_GE(pos, 0);
    L3_DCHECK_LE(pos, size_);
    L3_RETURN_IF_ERROR(ensure_capacity(size_ + 1));
    if (pos == size_) {
      ::new (static_cast<void*>(data_ + size_)) T(std::move(value));
    } else {
      ::new (static_cast<void*>(data_ + size_)) T(std::move(data_[size_ - 1]));
      for (Index i = size_ - 1; i > pos; --i) data_[i] = std::move(data_[i - 1]);
      data_[pos] = std::move(value);
    }
    ++size_;
    return Status::ok();
  }

  // Remove the element at pos; later elements shift left.
  void erase(Index pos) noexcept {
    L3_DCHECK_GE(pos, 0);
    L3_DCHECK_LT(pos, size_);
    erase(pos, pos + 1);
  }

  // Remove elements [first, last).
  void erase(Index first, Index last) noexcept {
    L3_DCHECK_GE(first, 0);
    L3_DCHECK_LE(first, last);
    L3_DCHECK_LE(last, size_);
    const Index count = last - first;
    if (count == 0) return;
    for (Index i = last; i < size_; ++i) data_[i - count] = std::move(data_[i]);
    destroy_range(data_ + (size_ - count), count);
    size_ -= count;
  }

 private:
  static constexpr Size kAlign =
      alignof(T) > kMinObjectAlignment ? alignof(T) : kMinObjectAlignment;

  T* inline_data() noexcept { return reinterpret_cast<T*>(inline_buf_); }
  const T* inline_data() const noexcept {
    return reinterpret_cast<const T*>(inline_buf_);
  }

  const Allocator& allocator() const noexcept {
    return allocator_ != nullptr ? *allocator_ : default_allocator();
  }

  // Move other's elements into *this, which must be empty and inline.
  void take(SmallVec& other) noexcept {
    if (other.is_inline()) {
      relocate(data_, other.data_, other.size_);
      size_ = other.size_;
      other.size_ = 0;
    } else {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.data_ = other.inline_data();
      other.size_ = 0;
      other.capacity_ = N;
    }
  }

  // Free the heap block (elements must already be destroyed or relocated) and
  // fall back to the inline buffer.
  void release_heap() noexcept {
    if (!is_inline()) {
      allocator().deallocate(data_, static_cast<Size>(capacity_) * sizeof(T), kAlign);
    }
    data_ = inline_data();
    capacity_ = N;
  }

  Status allocate_buffer(Index capacity, T** out) const noexcept {
    Size bytes = 0;
    if (!checked_mul(static_cast<Size>(capacity), sizeof(T), &bytes)) {
      return Status::error(StatusCode::kOutOfRange, "SmallVec: capacity overflow");
    }
    void* mem = allocator().allocate(bytes, kAlign);
    if (L3_UNLIKELY(mem == nullptr)) {
      return Status::errorf(StatusCode::kOutOfMemory,
                            "SmallVec: failed to allocate %zu bytes", bytes);
    }
    *out = static_cast<T*>(mem);
    return Status::ok();
  }

  Index next_capacity(Index min_needed) const noexcept {
    const Index doubled = capacity_ > kMaxIndex / 2 ? kMaxIndex : capacity_ * 2;
    return doubled < min_needed ? min_needed : doubled;
  }

  // Move to a bigger heap block of exactly new_cap elements (new_cap > capacity_).
  Status grow_to(Index new_cap) noexcept {
    T* buffer = nullptr;
    L3_RETURN_IF_ERROR(allocate_buffer(new_cap, &buffer));
    relocate(buffer, data_, size_);
    release_heap();
    data_ = buffer;
    capacity_ = new_cap;
    return Status::ok();
  }

  // Amortized growth: at least double.
  Status ensure_capacity(Index min_needed) noexcept {
    if (L3_LIKELY(min_needed <= capacity_)) return Status::ok();
    return grow_to(next_capacity(min_needed));
  }

  // Slow path of emplace_back. The new element is constructed in the new
  // buffer BEFORE the old one is released, so args may refer to old elements.
  template <typename... Args>
  L3_NOINLINE Status emplace_back_slow(Args&&... args) noexcept {
    const Index new_cap = next_capacity(size_ + 1);
    T* buffer = nullptr;
    L3_RETURN_IF_ERROR(allocate_buffer(new_cap, &buffer));
    ::new (static_cast<void*>(buffer + size_)) T(std::forward<Args>(args)...);
    relocate(buffer, data_, size_);
    release_heap();
    data_ = buffer;
    capacity_ = new_cap;
    ++size_;
    return Status::ok();
  }

  // Move n elements from src to uninitialized dst, destroying the sources.
  static void relocate(T* dst, T* src, Index n) noexcept {
    if constexpr (std::is_trivially_copyable_v<T>) {
      mem_copy(dst, src, static_cast<Size>(n) * sizeof(T));
    } else {
      for (Index i = 0; i < n; ++i) {
        ::new (static_cast<void*>(dst + i)) T(std::move(src[i]));
        src[i].~T();
      }
    }
  }

  static void copy_construct(T* dst, const T* src, Index n) noexcept {
    if constexpr (std::is_trivially_copyable_v<T>) {
      mem_copy(dst, src, static_cast<Size>(n) * sizeof(T));
    } else {
      for (Index i = 0; i < n; ++i) ::new (static_cast<void*>(dst + i)) T(src[i]);
    }
  }

  static void destroy_range(T* p, Index n) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (Index i = 0; i < n; ++i) p[i].~T();
    } else {
      (void)p;
      (void)n;
    }
  }

  T* data_;
  Index size_;
  Index capacity_;
  const Allocator* allocator_;  // nullptr = default_allocator()
  L3_ALIGNAS(alignof(T)) byte inline_buf_[sizeof(T) * static_cast<usize>(N)];
};

// Element-wise equality; the inline capacities may differ.
template <typename T, Index N, Index M>
bool operator==(const SmallVec<T, N>& a, const SmallVec<T, M>& b) noexcept {
  return equal(a.span(), b.span());
}

template <typename T, Index N, Index M>
bool operator!=(const SmallVec<T, N>& a, const SmallVec<T, M>& b) noexcept {
  return !(a == b);
}

}  // namespace l3
