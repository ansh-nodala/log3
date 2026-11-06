// src/tensor/shape.cpp
//
// Implementation of Shape, Strides, contiguity checks, broadcasting, and
// sub-byte layout validation.

#include "log3/tensor/shape.h"

#include <cstdio>

namespace l3 {

Result<Shape> Shape::from_dims(Span<const Index> dims) noexcept {
  if (L3_UNLIKELY(dims.size() < 0 || dims.size() > kMaxTensorDims)) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "Shape::from_dims: rank %lld exceeds maximum supported rank (%lld)",
        static_cast<long long>(dims.size()),
        static_cast<long long>(kMaxTensorDims));
  }

  Shape out;
  out.rank_ = dims.size();
  if (dims.empty()) {
    out.dims_[0] = 1;  // 0-D scalar
    return out;
  }

  // Always check overflow of the product of ALL non-zero dimensions even when
  // another axis is 0: `Strides::contiguous_for` multiplies all `d > 1`
  // dimensions together, so checking the full non-zero product guarantees
  // `Strides::contiguous_for(out)` can never suffer signed integer overflow!
  Size non_zero_prod = 1;
  for (Index i = 0; i < dims.size(); ++i) {
    const Index d = dims[i];
    if (L3_UNLIKELY(d < 0)) {
      return Status::errorf(
          StatusCode::kInvalidArgument,
          "Shape::from_dims: negative dimension %lld at axis %lld",
          static_cast<long long>(d), static_cast<long long>(i));
    }
    out.dims_[i] = d;
    if (d > 1) {
      if (L3_UNLIKELY(!checked_mul(non_zero_prod, static_cast<Size>(d),
                                   &non_zero_prod) ||
                      non_zero_prod > static_cast<Size>(kMaxIndex))) {
        return Status::error(StatusCode::kOutOfRange,
                             "Shape::from_dims: dimension product overflow");
      }
    }
  }
  return out;
}

usize Shape::to_string(char* buf, usize buf_size) const noexcept {
  if (buf == nullptr || buf_size == 0) return 0;

  usize pos = 0;
  auto append_fmt = [&](const char* fmt, long long val, bool has_val) noexcept {
    if (pos + 1 >= buf_size) return;
    const usize avail = buf_size - pos;
    const int written = has_val ? std::snprintf(buf + pos, avail, fmt, val)
                                : std::snprintf(buf + pos, avail, "%s", fmt);
    if (written > 0) {
      const usize w = static_cast<usize>(written);
      pos += (w < avail) ? w : (avail - 1);
    }
  };

  if (empty()) {
    append_fmt("(empty)", 0, false);
    return pos;
  }
  if (is_scalar()) {
    append_fmt("()", 0, false);
    return pos;
  }

  append_fmt("(", 0, false);
  for (Index i = 0; i < rank_; ++i) {
    if (i > 0) append_fmt(", ", 0, false);
    append_fmt("%lld", static_cast<long long>(dims_[i]), true);
  }
  append_fmt(")", 0, false);
  return pos;
}

Strides Strides::contiguous_for(const Shape& shape) noexcept {
  Strides out;
  const Index r = shape.rank();
  out.rank_ = r;
  if (r == 0) return out;

  Index running = 1;
  for (Index i = r - 1; i >= 0; --i) {
    out.strides_[i] = running;
    const Index d = shape[i];
    if (d > 1) {
      running *= d;
    }
  }
  return out;
}

Result<Strides> Strides::from_strides(Span<const Index> strides) noexcept {
  if (L3_UNLIKELY(strides.size() < 0 || strides.size() > kMaxTensorDims)) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "Strides::from_strides: rank %lld exceeds maximum (%lld)",
        static_cast<long long>(strides.size()),
        static_cast<long long>(kMaxTensorDims));
  }
  Strides out;
  out.rank_ = strides.size();
  for (Index i = 0; i < strides.size(); ++i) {
    out.strides_[i] = strides[i];
  }
  return out;
}

bool is_contiguous_layout(const Shape& shape, const Strides& strides) noexcept {
  if (shape.empty() || shape.rank() != strides.rank()) return false;
  if (shape.numel() <= 1) return true;

  Index expected_stride = 1;
  for (Index i = shape.rank() - 1; i >= 0; --i) {
    const Index d = shape[i];
    if (d == 1) continue;  // Size-1 axes do not affect contiguity
    if (strides[i] != expected_stride) return false;
    expected_stride *= d;
  }
  return true;
}

Status validate_layout_for_dtype(const Shape& shape, const Strides& strides,
                                 DType dtype) noexcept {
  if (!is_valid_dtype(dtype)) {
    return Status::error(StatusCode::kInvalidArgument,
                         "validate_layout_for_dtype: invalid DType");
  }
  if (shape.empty()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "validate_layout_for_dtype: shape is empty");
  }
  if (shape.rank() != strides.rank()) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "validate_layout_for_dtype: shape rank (%lld) != strides rank (%lld)",
        static_cast<long long>(shape.rank()),
        static_cast<long long>(strides.rank()));
  }

  for (Index i = 0; i < strides.rank(); ++i) {
    if (strides[i] < 0) {
      return Status::errorf(
          StatusCode::kInvalidArgument,
          "validate_layout_for_dtype: negative stride (%lld) at axis %lld",
          static_cast<long long>(strides[i]), static_cast<long long>(i));
    }
  }

  if (!is_subbyte_dtype(dtype)) {
    return Status::ok();
  }

  // Sub-byte packed types (t2, i4, u4) pack multiple elements along the
  // innermost dimension into each byte. Even if an outer dimension is 0, the
  // rank and innermost dimension alignment must still be valid.
  if (shape.rank() == 0) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "sub-byte DType '%s' cannot be used for a 0-D scalar tensor",
        dtype_name(dtype));
  }

  const Index blk_elems = dtype_block_elements(dtype);
  const Index last_dim = shape[shape.rank() - 1];
  const Index last_stride = strides[strides.rank() - 1];

  if (last_dim % blk_elems != 0) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "innermost dimension (%lld) for sub-byte DType '%s' must be a multiple "
        "of %lld",
        static_cast<long long>(last_dim), dtype_name(dtype),
        static_cast<long long>(blk_elems));
  }
  if (last_stride != 1) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "innermost stride (%lld) for sub-byte DType '%s' must be 1",
        static_cast<long long>(last_stride), dtype_name(dtype));
  }
  for (Index i = 0; i < shape.rank() - 1; ++i) {
    if (strides[i] % blk_elems != 0) {
      return Status::errorf(
          StatusCode::kInvalidArgument,
          "stride[%lld] = %lld for sub-byte DType '%s' must be a multiple of "
          "block size %lld",
          static_cast<long long>(i), static_cast<long long>(strides[i]),
          dtype_name(dtype), static_cast<long long>(blk_elems));
    }
  }
  return Status::ok();
}

Result<Shape> broadcast_shapes(const Shape& a, const Shape& b) noexcept {
  if (a.empty() || b.empty()) {
    return Status::error(StatusCode::kShapeMismatch,
                         "broadcast_shapes: cannot broadcast empty Shape");
  }
  const Index ra = a.rank();
  const Index rb = b.rank();
  const Index out_rank = ra > rb ? ra : rb;
  if (out_rank == 0) {
    return Shape::scalar();
  }

  Index out_dims[kMaxTensorDims]{};
  for (Index i = 0; i < out_rank; ++i) {
    const Index ia = ra - 1 - i;
    const Index ib = rb - 1 - i;
    const Index da = ia >= 0 ? a[ia] : 1;
    const Index db = ib >= 0 ? b[ib] : 1;

    if (da != db && da != 1 && db != 1) {
      char buf_a[64];
      char buf_b[64];
      a.to_string(buf_a, sizeof(buf_a));
      b.to_string(buf_b, sizeof(buf_b));
      return Status::errorf(StatusCode::kShapeMismatch,
                            "broadcast_shapes: incompatible shapes %s and %s",
                            buf_a, buf_b);
    }
    out_dims[out_rank - 1 - i] = (da == 0 || db == 0) ? 0 : (da > db ? da : db);
  }
  return Shape::from_dims(Span<const Index>(out_dims, out_rank));
}

Result<Strides> broadcast_strides(const Shape& src_shape,
                                  const Strides& src_strides,
                                  const Shape& target_shape) noexcept {
  if (src_shape.empty() || target_shape.empty()) {
    return Status::error(StatusCode::kShapeMismatch,
                         "broadcast_strides: source or target Shape is empty");
  }
  if (src_shape.rank() != src_strides.rank() ||
      src_shape.rank() > target_shape.rank()) {
    return Status::error(StatusCode::kShapeMismatch,
                         "broadcast_strides: invalid source or target rank");
  }

  const Index r_src = src_shape.rank();
  const Index r_tgt = target_shape.rank();
  const Index offset = r_tgt - r_src;

  Index out[kMaxTensorDims]{};
  for (Index i = 0; i < r_tgt; ++i) {
    if (i < offset) {
      out[i] = 0;  // Prepended broadcast dimension
    } else {
      const Index src_d = src_shape[i - offset];
      const Index tgt_d = target_shape[i];
      if (src_d == tgt_d) {
        out[i] = src_strides[i - offset];
      } else if (src_d == 1) {
        out[i] = 0;  // Broadcasted axis has 0 stride
      } else {
        return Status::error(
            StatusCode::kShapeMismatch,
            "broadcast_strides: source shape cannot broadcast to target");
      }
    }
  }
  return Strides::from_strides(Span<const Index>(out, r_tgt));
}

}  // namespace l3
