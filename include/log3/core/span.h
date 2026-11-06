// log3/core/span.h
//
// Span<T>: a non-owning view of a contiguous run of T (pointer + length).
// A C++17 stand-in for std::span with log3 conventions:
//
//   * size() returns Index (SIGNED, i64), like every shape/stride/offset in
//     the library. `for (Index i = 0; i < s.size(); ++i)` has no sign-compare
//     warnings and no unsigned-underflow traps. size_bytes() returns Size.
//   * Bounds are checked with L3_DCHECK (debug builds); release builds pay
//     nothing. Validate user input with Status before building a Span.
//   * Dynamic extent only (no Span<T, N>): shapes and strides are runtime
//     sized, and a fixed extent would only add template bloat.
//   * Trivially copyable, two words wide, passed by value everywhere.
//   * Never owns, never allocates, never frees. The viewed memory must outlive
//     the Span (like a raw pointer).
//
//   int a[4] = {1, 2, 3, 4};
//   Span<int> s(a);                      // from array
//   Span<const int> c = s;               // Span<T> -> Span<const T>
//   Span<int> mid = s.subspan(1, 2);     // {2, 3}
//   Span s2(ptr, count);                 // class template argument deduction
//   for (int v : s) { ... }              // iterators are plain pointers
//
// Anything with data() and size() converts (SmallVec, Vec, arrays wrappers).
// Temporaries do not: a Span of a temporary container would dangle.

#pragma once

#include <type_traits>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/types.h"

namespace l3 {

template <typename T>
class Span;

namespace detail {

template <typename X>
struct IsSpan : std::false_type {};
template <typename X>
struct IsSpan<Span<X>> : std::true_type {};

// A container C is viewable as Span<T> if c.data() yields a pointer whose
// element type converts to T by qualification only (int -> const int is fine,
// Derived -> Base is rejected: that would be slicing) and c.size() is integral.
template <typename C, typename T, typename = void>
struct IsViewableAs : std::false_type {};

template <typename C, typename T>
struct IsViewableAs<
    C, T,
    std::void_t<decltype(std::declval<C&>().data()),
                decltype(std::declval<C&>().size())>>
    : std::bool_constant<
          std::is_convertible_v<
              std::remove_pointer_t<decltype(std::declval<C&>().data())> (*)[],
              T (*)[]> &&
          std::is_integral_v<decltype(std::declval<C&>().size())> &&
          !IsSpan<std::remove_cv_t<C>>::value> {};

}  // namespace detail

template <typename T>
class Span {
  static_assert(!std::is_reference_v<T>, "Span<T&> is not allowed.");
  static_assert(!std::is_void_v<T>,
                "Span<void> is not allowed; use Span<byte> for raw memory.");

 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using reference = T&;
  using iterator = T*;

  // --- construction ---------------------------------------------------------
  constexpr Span() noexcept : data_(nullptr), size_(0) {}

  constexpr Span(T* data, Index size) noexcept : data_(data), size_(size) {
    L3_DCHECK_GE(size, 0);
    L3_DCHECK(size == 0 || data != nullptr);
  }

  template <usize N>
  constexpr Span(T (&array)[N]) noexcept
      : data_(array), size_(static_cast<Index>(N)) {}

  template <typename C,
            typename = std::enable_if_t<detail::IsViewableAs<C, T>::value>>
  constexpr Span(C& container) noexcept
      : data_(container.data()), size_(static_cast<Index>(container.size())) {}

  // Span<U> -> Span<T> (e.g. Span<int> -> Span<const int>).
  template <typename U,
            typename = std::enable_if_t<
                std::is_convertible_v<U (*)[], T (*)[]>>>
  constexpr Span(const Span<U>& other) noexcept
      : data_(other.data()), size_(other.size()) {}

  // --- observers ------------------------------------------------------------
  constexpr T* data() const noexcept { return data_; }
  constexpr Index size() const noexcept { return size_; }
  constexpr Size size_bytes() const noexcept {
    return static_cast<Size>(size_) * sizeof(T);
  }
  constexpr bool empty() const noexcept { return size_ == 0; }

  // --- element access -------------------------------------------------------
  constexpr T& operator[](Index i) const noexcept {
    L3_DCHECK_GE(i, 0);
    L3_DCHECK_LT(i, size_);
    return data_[i];
  }

  constexpr T& front() const noexcept {
    L3_DCHECK_GT(size_, 0);
    return data_[0];
  }

  constexpr T& back() const noexcept {
    L3_DCHECK_GT(size_, 0);
    return data_[size_ - 1];
  }

  // --- iteration (raw pointers) ---------------------------------------------
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }

  // --- sub-views ------------------------------------------------------------
  // First n elements.
  constexpr Span first(Index n) const noexcept {
    L3_DCHECK_GE(n, 0);
    L3_DCHECK_LE(n, size_);
    return Span(data_, n);
  }

  // Last n elements.
  constexpr Span last(Index n) const noexcept {
    L3_DCHECK_GE(n, 0);
    L3_DCHECK_LE(n, size_);
    return Span(data_ + (size_ - n), n);
  }

  // Elements [offset, size()).
  constexpr Span subspan(Index offset) const noexcept {
    L3_DCHECK_GE(offset, 0);
    L3_DCHECK_LE(offset, size_);
    return Span(data_ + offset, size_ - offset);
  }

  // Elements [offset, offset + count).
  constexpr Span subspan(Index offset, Index count) const noexcept {
    L3_DCHECK_GE(offset, 0);
    L3_DCHECK_GE(count, 0);
    L3_DCHECK_LE(offset, size_);
    L3_DCHECK_LE(count, size_ - offset);
    return Span(data_ + offset, count);
  }

 private:
  T* data_;
  Index size_;
};

// --- class template argument deduction -----------------------------------------
template <typename T>
Span(T*, Index) -> Span<T>;

template <typename T, usize N>
Span(T (&)[N]) -> Span<T>;

template <typename C>
Span(C&) -> Span<std::remove_pointer_t<decltype(std::declval<C&>().data())>>;

// --- byte views ------------------------------------------------------------------
using ByteSpan = Span<byte>;
using ConstByteSpan = Span<const byte>;

template <typename T>
ConstByteSpan as_bytes(Span<T> s) noexcept {
  return ConstByteSpan(reinterpret_cast<const byte*>(s.data()),
                       static_cast<Index>(s.size_bytes()));
}

template <typename T>
ByteSpan as_writable_bytes(Span<T> s) noexcept {
  static_assert(!std::is_const_v<T>,
                "as_writable_bytes: cannot get writable bytes of a const Span.");
  return ByteSpan(reinterpret_cast<byte*>(s.data()),
                  static_cast<Index>(s.size_bytes()));
}

// --- comparison ------------------------------------------------------------------
// Element-wise equality (same length and a[i] == b[i] for all i).
template <typename A, typename B>
bool equal(Span<A> a, Span<B> b) noexcept {
  if (a.size() != b.size()) return false;
  for (Index i = 0; i < a.size(); ++i) {
    if (!(a[i] == b[i])) return false;
  }
  return true;
}

}  // namespace l3
