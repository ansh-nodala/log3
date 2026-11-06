// log3/tensor/shape.h
//
// Shape and Strides descriptors for N-dimensional tensors (up to kMaxTensorDims
// = 8 dimensions, stored completely inline with zero heap allocation).
//
// Why fixed inline storage (`kMaxTensorDims = 8`) instead of std::vector:
//   * Every modern deep learning workload (LLMs, BitNet, ViTs, CNNs, GNNs) uses
//     1D to 6D tensors. Capping tensor rank at 8 lets Shape and Strides live in
//     72 bytes on the stack/inside Tensor: TriviallyCopyable, passed by value,
//     zero heap allocations, zero destructor overhead, and 100% cache-friendly.
//   * Strides are expressed in ELEMENTS (not bytes) so sub-byte DTypes
//     (kTernary2 = 4 elems/byte, kInt4 = 2 elems/byte) and full-byte DTypes
//     share the exact same indexing math! For sub-byte tensors, the innermost
//     dimension (`dims.back()`) must be a multiple of `dtype_block_elements(dt)`
//     and contiguous (`strides.back() == 1`).
//
// Compatibility:
//   * C++17, no exceptions, no RTTI, zero heap allocations.

#pragma once

#include <cstring>
#include <initializer_list>

#include "log3/core/assert.h"
#include "log3/core/dtype.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

// Maximum supported tensor rank (dimensions).
inline constexpr Index kMaxTensorDims = 8;

// -----------------------------------------------------------------------------
// Shape
// -----------------------------------------------------------------------------
class L3_API Shape {
 public:
  // Default constructor creates an uninitialized / empty Shape (`rank() == 0`
  // and `numel() == 0`, so `empty() == true`). Use `Shape::scalar()` for a 0-D
  // scalar tensor with 1 element, or `Shape::from_dims({n})` for a 1-D vector.
  constexpr Shape() noexcept : dims_{0, 0, 0, 0, 0, 0, 0, 0}, rank_(0) {}

  // Checked creation from a Span of dimensions.
  // Fails with:
  //   * kInvalidArgument if rank < 0, rank > kMaxTensorDims, or any dim < 0.
  //   * kOutOfRange if the product of dimensions or any sub-product overflows
  //     Index (guarantees `Strides::contiguous_for` can never overflow).
  static Result<Shape> from_dims(Span<const Index> dims) noexcept;

  static Result<Shape> from_dims(std::initializer_list<Index> dims) noexcept {
    return from_dims(
        Span<const Index>(dims.begin(), static_cast<Index>(dims.size())));
  }

  // Create a 0-D scalar shape (rank = 0, numel = 1).
  // Distinguished internally by setting dims_[0] = 1 while rank_ = 0.
  static constexpr Shape scalar() noexcept {
    Shape s;
    s.dims_[0] = 1;
    s.rank_ = 0;
    return s;
  }

  // --- observers -------------------------------------------------------------
  constexpr Index rank() const noexcept { return rank_; }
  constexpr Index ndim() const noexcept { return rank_; }
  constexpr bool is_scalar() const noexcept {
    return rank_ == 0 && dims_[0] == 1;
  }
  constexpr bool empty() const noexcept {
    return rank_ == 0 && dims_[0] == 0;
  }

  constexpr const Index* data() const noexcept { return dims_; }
  constexpr Span<const Index> dims() const noexcept {
    return Span<const Index>(dims_, rank_);
  }

  constexpr Index dim(Index axis) const noexcept {
    L3_DCHECK_GE(axis, 0);
    L3_DCHECK_LT(axis, rank_);
    return dims_[axis];
  }

  constexpr Index operator[](Index axis) const noexcept { return dim(axis); }

  // Total number of elements across all dimensions.
  constexpr Index numel() const noexcept {
    if (rank_ == 0) return dims_[0];  // 0 for default empty, 1 for scalar()
    Index prod = 1;
    for (Index i = 0; i < rank_; ++i) {
      prod *= dims_[i];
    }
    return prod;
  }

  // Normalize a possibly-negative axis in [-rank, rank - 1] into [0, rank - 1].
  Result<Index> normalize_axis(Index axis) const noexcept {
    const Index actual = axis < 0 ? (axis + rank_) : axis;
    if (L3_UNLIKELY(actual < 0 || actual >= rank_)) {
      return Status::errorf(
          StatusCode::kOutOfRange,
          "Shape::normalize_axis: axis %lld out of range for rank %lld",
          static_cast<long long>(axis), static_cast<long long>(rank_));
    }
    return actual;
  }

  // Format human-readable shape string into `buf` (e.g., "(2, 512, 768)").
  // Returns the actual number of characters written into `buf` (excluding the
  // null terminator, never exceeding `buf_size - 1`).
  usize to_string(char* buf, usize buf_size) const noexcept;

 private:
  Index dims_[kMaxTensorDims];
  Index rank_;
};

constexpr bool operator==(const Shape& a, const Shape& b) noexcept {
  if (a.rank() != b.rank()) return false;
  if (a.rank() == 0) return a.numel() == b.numel();
  for (Index i = 0; i < a.rank(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

constexpr bool operator!=(const Shape& a, const Shape& b) noexcept {
  return !(a == b);
}

// -----------------------------------------------------------------------------
// Strides (expressed in elements, signed to support negative/zero strides)
// -----------------------------------------------------------------------------
class L3_API Strides {
 public:
  constexpr Strides() noexcept : strides_{0, 0, 0, 0, 0, 0, 0, 0}, rank_(0) {}

  // Compute standard row-major (C-contiguous) strides for `shape`.
  // Convention: stride[i] = product of shape[i+1 .. rank-1] (with 0-sized
  // dimensions treated as 1 for stride propagation so outer axes stay
  // monotonically ordered even on 0-element slices).
  static Strides contiguous_for(const Shape& shape) noexcept;

  static Result<Strides> from_strides(Span<const Index> strides) noexcept;

  static Result<Strides> from_strides(
      std::initializer_list<Index> strides) noexcept {
    return from_strides(
        Span<const Index>(strides.begin(), static_cast<Index>(strides.size())));
  }

  constexpr Index rank() const noexcept { return rank_; }
  constexpr const Index* data() const noexcept { return strides_; }
  constexpr Span<const Index> values() const noexcept {
    return Span<const Index>(strides_, rank_);
  }

  constexpr Index operator[](Index axis) const noexcept {
    L3_DCHECK_GE(axis, 0);
    L3_DCHECK_LT(axis, rank_);
    return strides_[axis];
  }

  constexpr Index& operator[](Index axis) noexcept {
    L3_DCHECK_GE(axis, 0);
    L3_DCHECK_LT(axis, rank_);
    return strides_[axis];
  }

 private:
  Index strides_[kMaxTensorDims];
  Index rank_;
};

constexpr bool operator==(const Strides& a, const Strides& b) noexcept {
  if (a.rank() != b.rank()) return false;
  for (Index i = 0; i < a.rank(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

constexpr bool operator!=(const Strides& a, const Strides& b) noexcept {
  return !(a == b);
}

// -----------------------------------------------------------------------------
// Layout, contiguity, broadcasting, and sub-byte block validation
// -----------------------------------------------------------------------------

// Returns true if `(shape, strides)` represents a dense, row-major contiguous
// layout with no gaps between elements.
L3_API bool is_contiguous_layout(const Shape& shape,
                                 const Strides& strides) noexcept;

// Validates that `(shape, strides)` is legal for `dtype`:
//   * Rejects negative strides on all axes.
//   * For sub-byte packed DTypes (`kTernary2`, `kInt4`, `kUInt4`), the last
//     dimension must be a multiple of `dtype_block_elements(dtype)` (even when
//     an outer dimension is 0), the innermost stride must be 1, and all outer
//     strides must be multiples of `dtype_block_elements(dtype)`.
L3_API Status validate_layout_for_dtype(const Shape& shape,
                                        const Strides& strides,
                                        DType dtype) noexcept;

// NumPy / PyTorch broadcast rule: computes the broadcast output shape of `a`
// and `b`, or returns `kShapeMismatch` if incompatible.
L3_API Result<Shape> broadcast_shapes(const Shape& a, const Shape& b) noexcept;

// Computes the effective broadcast strides (setting stride = 0 along axes where
// `src_shape` had size 1 and was stretched to `target_shape[axis] != 1`).
L3_API Result<Strides> broadcast_strides(const Shape& src_shape,
                                         const Strides& src_strides,
                                         const Shape& target_shape) noexcept;

}  // namespace l3
