// src/tensor/tensor.cpp
//
// Implementation of Tensor creation, zero-copy view operations (reshape,
// transpose, permute, slice, squeeze, unsqueeze), strided-to-contiguous
// deep cloning, and intrusive AutogradMeta methods.

#include "log3/tensor/tensor.h"

#include "log3/autograd/engine.h"
#include "log3/kernels/cpu_kernels.h"

namespace l3 {

AutogradMeta::~AutogradMeta() noexcept = default;

namespace {

// Compute the minimum byte span required in Storage to hold all elements
// reachable by `(shape, strides)` starting at `byte_offset`, and verify it fits
// inside `storage_bytes`.
Status verify_storage_bounds(const Shape& shape, const Strides& strides,
                             DType dtype, Size byte_offset,
                             Size storage_bytes) noexcept {
  L3_RETURN_IF_ERROR(validate_layout_for_dtype(shape, strides, dtype));

  const Index count = shape.numel();
  if (count == 0) {
    if (byte_offset > storage_bytes) {
      return Status::error(StatusCode::kOutOfRange,
                           "Tensor byte_offset exceeds storage size");
    }
    return Status::ok();
  }

  Index max_elem_idx = 0;
  for (Index i = 0; i < shape.rank(); ++i) {
    if (shape[i] > 1 && strides[i] > 0) {
      Size term = 0;
      if (!checked_mul(static_cast<Size>(shape[i] - 1),
                       static_cast<Size>(strides[i]), &term) ||
          !checked_add(static_cast<Size>(max_elem_idx), term, &term) ||
          term >= static_cast<Size>(kMaxIndex)) {
        return Status::error(StatusCode::kOutOfRange,
                             "tensor stride span overflow");
      }
      max_elem_idx = static_cast<Index>(term);
    }
  }

  const Index span_elems = max_elem_idx + 1;
  Size needed_bytes = 0;
  L3_RETURN_IF_ERROR(
      checked_dtype_bytes_padded(dtype, span_elems, &needed_bytes));

  Size end_byte = 0;
  if (!checked_add(byte_offset, needed_bytes, &end_byte) ||
      end_byte > storage_bytes) {
    return Status::errorf(
        StatusCode::kOutOfRange,
        "Tensor view requires %zu bytes (offset %zu + span %zu), but Storage "
        "has %zu bytes",
        end_byte, byte_offset, needed_bytes, storage_bytes);
  }
  return Status::ok();
}

void copy_strided_recursive(const byte* src_base, byte* dst_base,
                            const Shape& shape, const Strides& src_strides,
                            Size elem_bytes, Index axis, Index src_elem_offset,
                            Index* dst_linear_idx) noexcept {
  if (axis == shape.rank()) {
    mem_copy(dst_base + static_cast<Size>(*dst_linear_idx) * elem_bytes,
             src_base + static_cast<Size>(src_elem_offset) * elem_bytes,
             elem_bytes);
    ++(*dst_linear_idx);
    return;
  }

  const Index d = shape[axis];
  const Index st = src_strides[axis];
  if (axis == shape.rank() - 1 && st == 1) {
    mem_copy(dst_base + static_cast<Size>(*dst_linear_idx) * elem_bytes,
             src_base + static_cast<Size>(src_elem_offset) * elem_bytes,
             static_cast<Size>(d) * elem_bytes);
    *dst_linear_idx += d;
    return;
  }

  for (Index i = 0; i < d; ++i) {
    copy_strided_recursive(src_base, dst_base, shape, src_strides, elem_bytes,
                           axis + 1, src_elem_offset + i * st, dst_linear_idx);
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// Autograd methods on Tensor
// -----------------------------------------------------------------------------

bool Tensor::requires_grad() const noexcept {
  return autograd_meta_ != nullptr && autograd_meta_->requires_grad;
}

Status Tensor::set_requires_grad(bool req) noexcept {
  if (!req) {
    if (autograd_meta_ != nullptr) {
      autograd_meta_->requires_grad = false;
    }
    return Status::ok();
  }
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::set_requires_grad: Tensor is not valid");
  }
  if (!is_float_dtype(dtype_)) {
    return Status::errorf(
        StatusCode::kTypeMismatch,
        "Tensor::set_requires_grad: only floating-point tensors can require "
        "gradients (got DType '%s')",
        dtype_name(dtype_));
  }
  if (autograd_meta_ == nullptr) {
    L3_ASSIGN_OR_RETURN(autograd_meta_, make_ptr<AutogradMeta>());
  }
  autograd_meta_->requires_grad = true;
  return Status::ok();
}

bool Tensor::is_leaf() const noexcept {
  return requires_grad() &&
         (autograd_meta_ == nullptr || autograd_meta_->grad_node == nullptr);
}

bool Tensor::has_grad() const noexcept {
  return autograd_meta_ != nullptr && autograd_meta_->grad.is_valid();
}

const Tensor* Tensor::grad() const noexcept {
  if (!has_grad()) return nullptr;
  return &autograd_meta_->grad;
}

Tensor* Tensor::mutable_grad() noexcept {
  if (!has_grad()) return nullptr;
  return &autograd_meta_->grad;
}

Status Tensor::accumulate_grad(const Tensor& incoming_grad) noexcept {
  if (!incoming_grad.is_valid() || incoming_grad.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::accumulate_grad: incoming_grad must be f32");
  }
  if (incoming_grad.shape() != shape_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "Tensor::accumulate_grad: gradient shape mismatch");
  }
  if (autograd_meta_ == nullptr) {
    L3_ASSIGN_OR_RETURN(autograd_meta_, make_ptr<AutogradMeta>());
  }
  if (!autograd_meta_->grad.is_valid()) {
    L3_ASSIGN_OR_RETURN(autograd_meta_->grad, incoming_grad.clone());
    return Status::ok();
  }

  Tensor contig_in;
  L3_ASSIGN_OR_RETURN(contig_in, incoming_grad.contiguous());
  kernels::add_f32(autograd_meta_->grad.data_ptr<f32>(),
                   contig_in.data_ptr<f32>(),
                   autograd_meta_->grad.mutable_data_ptr<f32>(), numel());
  return Status::ok();
}

void Tensor::zero_grad() noexcept {
  if (autograd_meta_ != nullptr && autograd_meta_->grad.is_valid()) {
    mem_zero(autograd_meta_->grad.raw_mutable_data(),
             autograd_meta_->grad.nbytes());
  }
}

Status Tensor::backward(const Tensor* grad_output) const noexcept {
  return autograd::run_backward(*this, grad_output);
}

autograd::GradNode* Tensor::grad_node() const noexcept {
  return autograd_meta_ != nullptr ? autograd_meta_->grad_node.get() : nullptr;
}

Status Tensor::set_grad_node(Ptr<autograd::GradNode> node) noexcept {
  if (autograd_meta_ == nullptr) {
    L3_ASSIGN_OR_RETURN(autograd_meta_, make_ptr<AutogradMeta>());
  }
  autograd_meta_->requires_grad = true;
  autograd_meta_->grad_node = std::move(node);
  return Status::ok();
}

// -----------------------------------------------------------------------------
// Factory & View methods
// -----------------------------------------------------------------------------

Result<Tensor> Tensor::empty(const Shape& shape, DType dtype,
                             const Allocator& allocator) noexcept {
  if (shape.empty()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::empty: shape is empty");
  }
  const Strides strides = Strides::contiguous_for(shape);
  L3_RETURN_IF_ERROR(validate_layout_for_dtype(shape, strides, dtype));

  Size bytes = 0;
  L3_RETURN_IF_ERROR(checked_dtype_bytes(dtype, shape.numel(), &bytes));

  Ptr<Storage> st;
  L3_ASSIGN_OR_RETURN(st,
                      Storage::allocate(bytes, /*zero_init=*/false, allocator));
  return from_storage(std::move(st), shape, strides, dtype, 0, 1.0f);
}

Result<Tensor> Tensor::zeros(const Shape& shape, DType dtype,
                             const Allocator& allocator) noexcept {
  if (shape.empty()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::zeros: shape is empty");
  }
  const Strides strides = Strides::contiguous_for(shape);
  L3_RETURN_IF_ERROR(validate_layout_for_dtype(shape, strides, dtype));

  Size bytes = 0;
  L3_RETURN_IF_ERROR(checked_dtype_bytes(dtype, shape.numel(), &bytes));

  Ptr<Storage> st;
  L3_ASSIGN_OR_RETURN(st,
                      Storage::allocate(bytes, /*zero_init=*/true, allocator));
  return from_storage(std::move(st), shape, strides, dtype, 0, 1.0f);
}

Result<Tensor> Tensor::full(const Shape& shape, f32 value, DType dtype,
                            const Allocator& allocator) noexcept {
  if (is_subbyte_dtype(dtype)) {
    return Status::errorf(
        StatusCode::kUnsupported,
        "Tensor::full does not support sub-byte DType '%s' directly",
        dtype_name(dtype));
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, empty(shape, dtype, allocator));
  const Index n = out.numel();

  switch (dtype) {
    case DType::kBool: {
      bool* p = out.mutable_data_ptr<bool>();
      const bool v = (value != 0.0f);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kFloat32: {
      f32* p = out.mutable_data_ptr<f32>();
      for (Index i = 0; i < n; ++i) p[i] = value;
      break;
    }
    case DType::kFloat64: {
      f64* p = out.mutable_data_ptr<f64>();
      const f64 v = static_cast<f64>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kFloat16: {
      f16* p = out.mutable_data_ptr<f16>();
      const f16 v = f32_to_f16(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kBFloat16: {
      bf16* p = out.mutable_data_ptr<bf16>();
      const bf16 v = f32_to_bf16(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kInt8: {
      i8* p = out.mutable_data_ptr<i8>();
      const i8 v = static_cast<i8>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kUInt8: {
      u8* p = out.mutable_data_ptr<u8>();
      const u8 v = static_cast<u8>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kInt16: {
      i16* p = out.mutable_data_ptr<i16>();
      const i16 v = static_cast<i16>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kUInt16: {
      u16* p = out.mutable_data_ptr<u16>();
      const u16 v = static_cast<u16>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kInt32: {
      i32* p = out.mutable_data_ptr<i32>();
      const i32 v = static_cast<i32>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kUInt32: {
      u32* p = out.mutable_data_ptr<u32>();
      const u32 v = static_cast<u32>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kInt64: {
      i64* p = out.mutable_data_ptr<i64>();
      const i64 v = static_cast<i64>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case DType::kUInt64: {
      u64* p = out.mutable_data_ptr<u64>();
      const u64 v = static_cast<u64>(value);
      for (Index i = 0; i < n; ++i) p[i] = v;
      break;
    }
    default:
      return Status::errorf(StatusCode::kUnsupported,
                            "Tensor::full: unsupported DType '%s'",
                            dtype_name(dtype));
  }
  return out;
}

Result<Tensor> Tensor::from_storage(Ptr<Storage> storage, const Shape& shape,
                                    const Strides& strides, DType dtype,
                                    Size byte_offset,
                                    f32 quant_scale) noexcept {
  if (L3_UNLIKELY(!storage)) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::from_storage: storage is null");
  }
  L3_RETURN_IF_ERROR(verify_storage_bounds(shape, strides, dtype, byte_offset,
                                           storage->size_bytes()));

  Tensor t;
  t.storage_ = std::move(storage);
  t.shape_ = shape;
  t.strides_ = strides;
  t.byte_offset_ = byte_offset;
  t.dtype_ = dtype;
  t.quant_scale_ = quant_scale;
  return t;
}

Result<Tensor> Tensor::from_blob(void* data, const Shape& shape, DType dtype,
                                 bool is_mutable,
                                 const Allocator& meta_allocator) noexcept {
  if (shape.empty()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::from_blob: shape is empty");
  }
  const Strides strides = Strides::contiguous_for(shape);
  L3_RETURN_IF_ERROR(validate_layout_for_dtype(shape, strides, dtype));

  Size bytes = 0;
  L3_RETURN_IF_ERROR(checked_dtype_bytes(dtype, shape.numel(), &bytes));

  Ptr<Storage> st;
  L3_ASSIGN_OR_RETURN(
      st, Storage::from_external(data, bytes, is_mutable, meta_allocator));
  return from_storage(std::move(st), shape, strides, dtype, 0, 1.0f);
}

Result<Tensor> Tensor::clone(const Allocator& allocator) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::clone: cannot clone invalid Tensor");
  }
  Tensor dst;
  L3_ASSIGN_OR_RETURN(dst, empty(shape_, dtype_, allocator));
  dst.set_quant_scale(quant_scale_);

  const Index count = numel();
  if (count == 0) return dst;

  if (is_contiguous()) {
    mem_copy(dst.raw_mutable_data(), raw_data(), dst.nbytes());
    return dst;
  }

  if (is_subbyte_dtype(dtype_)) {
    const Index blk = dtype_block_elements(dtype_);
    const Index last_dim = shape_[shape_.rank() - 1];
    const Size row_bytes = static_cast<Size>(last_dim / blk);
    const Index outer_rows = count / last_dim;

    for (Index r = 0; r < outer_rows; ++r) {
      Index rem = r;
      Index src_elem_off = 0;
      for (Index ax = shape_.rank() - 2; ax >= 0; --ax) {
        const Index coord = rem % shape_[ax];
        rem /= shape_[ax];
        src_elem_off += coord * strides_[ax];
      }
      const Size src_byte_off = static_cast<Size>(src_elem_off / blk);
      mem_copy(dst.raw_mutable_data() + static_cast<Size>(r) * row_bytes,
               raw_data() + src_byte_off, row_bytes);
    }
    return dst;
  }

  Index dst_idx = 0;
  copy_strided_recursive(raw_data(), dst.raw_mutable_data(), shape_, strides_,
                         dtype_element_size(dtype_), 0, 0, &dst_idx);
  return dst;
}

Result<Tensor> Tensor::reshape(const Shape& new_shape) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::reshape: invalid Tensor");
  }
  if (new_shape.empty()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::reshape: new_shape cannot be empty");
  }
  if (new_shape.numel() != numel()) {
    return Status::errorf(
        StatusCode::kShapeMismatch,
        "Tensor::reshape: cannot reshape %lld elements into %lld elements",
        static_cast<long long>(numel()),
        static_cast<long long>(new_shape.numel()));
  }
  if (!is_contiguous()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "Tensor::reshape: zero-copy reshape requires a contiguous tensor "
        "(call .contiguous() first)");
  }
  const Strides new_strides = Strides::contiguous_for(new_shape);
  return from_storage(storage_, new_shape, new_strides, dtype_, byte_offset_,
                      quant_scale_);
}

Result<Tensor> Tensor::transpose(Index dim0, Index dim1) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::transpose: invalid Tensor");
  }
  Index a0 = 0;
  Index a1 = 0;
  L3_ASSIGN_OR_RETURN(a0, shape_.normalize_axis(dim0));
  L3_ASSIGN_OR_RETURN(a1, shape_.normalize_axis(dim1));

  Index new_dims[kMaxTensorDims]{};
  Index new_st[kMaxTensorDims]{};
  for (Index i = 0; i < shape_.rank(); ++i) {
    new_dims[i] = shape_[i];
    new_st[i] = strides_[i];
  }
  std::swap(new_dims[a0], new_dims[a1]);
  std::swap(new_st[a0], new_st[a1]);

  Shape s;
  Strides st;
  L3_ASSIGN_OR_RETURN(s, Shape::from_dims(Span<const Index>(new_dims, rank())));
  L3_ASSIGN_OR_RETURN(st,
                      Strides::from_strides(Span<const Index>(new_st, rank())));
  return from_storage(storage_, s, st, dtype_, byte_offset_, quant_scale_);
}

Result<Tensor> Tensor::permute(Span<const Index> perm) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::permute: invalid Tensor");
  }
  if (perm.size() != rank()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Tensor::permute: permutation length != tensor rank");
  }

  bool seen[kMaxTensorDims]{};
  Index new_dims[kMaxTensorDims]{};
  Index new_st[kMaxTensorDims]{};

  for (Index i = 0; i < rank(); ++i) {
    Index ax = 0;
    L3_ASSIGN_OR_RETURN(ax, shape_.normalize_axis(perm[i]));
    if (seen[ax]) {
      return Status::error(StatusCode::kInvalidArgument,
                           "Tensor::permute: duplicate axis in permutation");
    }
    seen[ax] = true;
    new_dims[i] = shape_[ax];
    new_st[i] = strides_[ax];
  }

  Shape s;
  Strides st;
  L3_ASSIGN_OR_RETURN(s, Shape::from_dims(Span<const Index>(new_dims, rank())));
  L3_ASSIGN_OR_RETURN(st,
                      Strides::from_strides(Span<const Index>(new_st, rank())));
  return from_storage(storage_, s, st, dtype_, byte_offset_, quant_scale_);
}

Result<Tensor> Tensor::slice(Index axis, Index start,
                             Index end) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::slice: invalid Tensor");
  }
  Index ax = 0;
  L3_ASSIGN_OR_RETURN(ax, shape_.normalize_axis(axis));

  const Index d = shape_[ax];
  if (start < 0) start += d;
  if (end < 0) end += d;
  if (start < 0 || end < start || end > d) {
    return Status::errorf(
        StatusCode::kOutOfRange,
        "Tensor::slice: invalid slice [%lld, %lld) for dimension of size %lld",
        static_cast<long long>(start), static_cast<long long>(end),
        static_cast<long long>(d));
  }

  const Index elem_delta = start * strides_[ax];
  Size byte_delta = 0;
  L3_RETURN_IF_ERROR(checked_dtype_bytes(dtype_, elem_delta, &byte_delta));

  Index new_dims[kMaxTensorDims]{};
  for (Index i = 0; i < rank(); ++i) {
    new_dims[i] = (i == ax) ? (end - start) : shape_[i];
  }
  Shape s;
  L3_ASSIGN_OR_RETURN(s, Shape::from_dims(Span<const Index>(new_dims, rank())));
  return from_storage(storage_, s, strides_, dtype_, byte_offset_ + byte_delta,
                      quant_scale_);
}

Result<Tensor> Tensor::unsqueeze(Index axis) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::unsqueeze: invalid Tensor");
  }
  if (rank() >= kMaxTensorDims) {
    return Status::error(StatusCode::kOutOfRange,
                         "Tensor::unsqueeze: tensor already at kMaxTensorDims");
  }
  const Index new_rank = rank() + 1;
  const Index ax = axis < 0 ? (axis + new_rank) : axis;
  if (ax < 0 || ax >= new_rank) {
    return Status::error(StatusCode::kOutOfRange,
                         "Tensor::unsqueeze: axis out of bounds");
  }

  Index new_dims[kMaxTensorDims]{};
  Index new_st[kMaxTensorDims]{};
  for (Index i = 0, src = 0; i < new_rank; ++i) {
    if (i == ax) {
      new_dims[i] = 1;
      if (src < rank()) {
        const Index d = shape_[src] > 0 ? shape_[src] : 1;
        new_st[i] = d * strides_[src];
      } else {
        new_st[i] = 1;
      }
    } else {
      new_dims[i] = shape_[src];
      new_st[i] = strides_[src];
      ++src;
    }
  }

  Shape s;
  Strides st;
  L3_ASSIGN_OR_RETURN(s,
                      Shape::from_dims(Span<const Index>(new_dims, new_rank)));
  L3_ASSIGN_OR_RETURN(
      st, Strides::from_strides(Span<const Index>(new_st, new_rank)));
  return from_storage(storage_, s, st, dtype_, byte_offset_, quant_scale_);
}

Result<Tensor> Tensor::squeeze(Index axis) const noexcept {
  if (!is_valid()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "Tensor::squeeze: invalid Tensor");
  }

  if (axis == -999) {
    if (rank() == 0) return view();
    Index new_dims[kMaxTensorDims]{};
    Index new_st[kMaxTensorDims]{};
    Index dst = 0;
    for (Index i = 0; i < rank(); ++i) {
      if (shape_[i] == 1) continue;
      new_dims[dst] = shape_[i];
      new_st[dst] = strides_[i];
      ++dst;
    }
    if (dst == 0) {
      return from_storage(storage_, Shape::scalar(), Strides(), dtype_,
                          byte_offset_, quant_scale_);
    }
    Shape s;
    Strides st;
    L3_ASSIGN_OR_RETURN(s, Shape::from_dims(Span<const Index>(new_dims, dst)));
    L3_ASSIGN_OR_RETURN(st,
                        Strides::from_strides(Span<const Index>(new_st, dst)));
    return from_storage(storage_, s, st, dtype_, byte_offset_, quant_scale_);
  }

  Index ax = 0;
  L3_ASSIGN_OR_RETURN(ax, shape_.normalize_axis(axis));
  if (shape_[ax] != 1) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "Tensor::squeeze: dimension at axis %lld has size %lld (!= 1)",
        static_cast<long long>(ax), static_cast<long long>(shape_[ax]));
  }
  if (rank() == 1) {
    return from_storage(storage_, Shape::scalar(), Strides(), dtype_,
                        byte_offset_, quant_scale_);
  }

  Index new_dims[kMaxTensorDims]{};
  Index new_st[kMaxTensorDims]{};
  Index dst = 0;
  for (Index i = 0; i < rank(); ++i) {
    if (i == ax) continue;
    new_dims[dst] = shape_[i];
    new_st[dst] = strides_[i];
    ++dst;
  }

  Shape s;
  Strides st;
  L3_ASSIGN_OR_RETURN(s, Shape::from_dims(Span<const Index>(new_dims, dst)));
  L3_ASSIGN_OR_RETURN(st,
                      Strides::from_strides(Span<const Index>(new_st, dst)));
  return from_storage(storage_, s, st, dtype_, byte_offset_, quant_scale_);
}

}  // namespace l3
