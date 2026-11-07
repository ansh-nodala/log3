// log3/tensor/tensor.h
//
// Tensor: lightweight multi-dimensional array view over a ref-counted Storage,
// with first-class support for scalar DTypes (f32, f16, bf16, i8, i32, ...),
// BitNet sub-byte quantized DTypes (kTernary2 "t2", kInt4 "i4", kUInt4 "u4"),
// and optional zero-overhead reverse-mode Autograd tracking.
//
// Design principles:
//   * Copying a Tensor (via `.view()` or move) never copies underlying data:
//     multiple Tensors share `Ptr<Storage>` with independent `(shape, strides,
//     byte_offset)`.
//   * Move-only by default (matching `SmallVec` and `Status`) so accidental
//     deep vs shallow copy confusion is impossible: call `.view()` for an
//     explicit zero-copy alias (O(1) ref-count bump) or `.clone()` for an
//     explicit deep contiguous copy.
//   * Zero-overhead Autograd integration: `autograd_meta_` is a single
//     intrusive `Ptr<AutogradMeta>` (8 bytes, `nullptr` during inference).
//     Inference tensors never allocate `AutogradMeta`, paying 0 bytes of heap
//     overhead. When `set_requires_grad(true)` is called or a differentiable op
//     produces an output while `GradMode::is_enabled()`, `AutogradMeta` holds
//     the accumulated gradient `Tensor` and the creator `Ptr<GradNode>`.
//   * Built-in quantization scale metadata (`quant_scale()`): BitNet b1.58 and
//     INT4 weight tensors carry a per-tensor scalar scale (`f32`), so linear
//     layers and GEMM kernels have everything they need inside one Tensor.
//   * Zero exceptions, zero RTTI, 64-byte SIMD-aligned allocations by default.

#pragma once

#include <initializer_list>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/dtype.h"
#include "log3/core/half.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/ptr.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/tensor/shape.h"
#include "log3/tensor/storage.h"

namespace l3 {

namespace autograd {
class GradNode;
}  // namespace autograd

struct AutogradMeta;

class L3_API Tensor {
 public:
  // Default constructor creates an empty/uninitialized Tensor (`!is_valid()`).
  Tensor() noexcept
      : storage_(),
        autograd_meta_(),
        shape_(),
        strides_(),
        byte_offset_(0),
        dtype_(DType::kFloat32),
        quant_scale_(1.0f) {}

  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  // Explicit `.view()` or `.clone()` instead of implicit copy constructor.
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  ~Tensor() noexcept = default;

  // --- factory methods -------------------------------------------------------

  // Allocate a new contiguous uninitialized Tensor of `shape` and `dtype`.
  static Result<Tensor> empty(
      const Shape& shape, DType dtype = DType::kFloat32,
      const Allocator& allocator = default_allocator()) noexcept;

  static Result<Tensor> empty(
      std::initializer_list<Index> dims, DType dtype = DType::kFloat32,
      const Allocator& allocator = default_allocator()) noexcept {
    Shape s;
    L3_ASSIGN_OR_RETURN(s, Shape::from_dims(dims));
    return empty(s, dtype, allocator);
  }

  // Allocate a new contiguous zero-initialized Tensor of `shape` and `dtype`.
  static Result<Tensor> zeros(
      const Shape& shape, DType dtype = DType::kFloat32,
      const Allocator& allocator = default_allocator()) noexcept;

  static Result<Tensor> zeros(
      std::initializer_list<Index> dims, DType dtype = DType::kFloat32,
      const Allocator& allocator = default_allocator()) noexcept {
    Shape s;
    L3_ASSIGN_OR_RETURN(s, Shape::from_dims(dims));
    return zeros(s, dtype, allocator);
  }

  // Allocate a contiguous Tensor and fill all elements with `value` (supports
  // all non-sub-byte DTypes: bool, i8..i64, u8..u64, f16, bf16, f32, f64).
  static Result<Tensor> full(
      const Shape& shape, f32 value, DType dtype = DType::kFloat32,
      const Allocator& allocator = default_allocator()) noexcept;

  static Result<Tensor> full(
      std::initializer_list<Index> dims, f32 value,
      DType dtype = DType::kFloat32,
      const Allocator& allocator = default_allocator()) noexcept {
    Shape s;
    L3_ASSIGN_OR_RETURN(s, Shape::from_dims(dims));
    return full(s, value, dtype, allocator);
  }

  // Construct a Tensor view over an existing Storage with explicit shape,
  // strides, and byte offset. Validates bounds and sub-byte alignment.
  static Result<Tensor> from_storage(Ptr<Storage> storage, const Shape& shape,
                                     const Strides& strides, DType dtype,
                                     Size byte_offset = 0,
                                     f32 quant_scale = 1.0f) noexcept;

  // Wrap caller-owned contiguous memory as a zero-copy Tensor (no allocation).
  static Result<Tensor> from_blob(
      void* data, const Shape& shape, DType dtype = DType::kFloat32,
      bool is_mutable = true,
      const Allocator& meta_allocator = default_allocator()) noexcept;

  static Result<Tensor> from_blob_const(
      const void* data, const Shape& shape, DType dtype = DType::kFloat32,
      const Allocator& meta_allocator = default_allocator()) noexcept {
    return from_blob(const_cast<void*>(data), shape, dtype, /*is_mutable=*/false,
                     meta_allocator);
  }

  // --- explicit copy / clone / detach ----------------------------------------

  // O(1) shallow copy: shares the underlying `Storage` and `AutogradMeta`
  // with identical shape, strides, byte_offset, and quant_scale.
  Tensor view() const noexcept {
    Tensor out;
    out.storage_ = storage_;
    out.autograd_meta_ = autograd_meta_;
    out.shape_ = shape_;
    out.strides_ = strides_;
    out.byte_offset_ = byte_offset_;
    out.dtype_ = dtype_;
    out.quant_scale_ = quant_scale_;
    return out;
  }

  // O(1) shallow view detached from the autograd graph (`requires_grad() == false`,
  // `grad_node() == nullptr`). Shares `Storage` without tracking gradients.
  Tensor detach() const noexcept {
    Tensor out;
    out.storage_ = storage_;
    out.autograd_meta_ = nullptr;
    out.shape_ = shape_;
    out.strides_ = strides_;
    out.byte_offset_ = byte_offset_;
    out.dtype_ = dtype_;
    out.quant_scale_ = quant_scale_;
    return out;
  }

  // Deep copy into a newly allocated contiguous Tensor (detached from autograd
  // history). Works for both contiguous and non-contiguous tensors.
  Result<Tensor> clone(
      const Allocator& allocator = default_allocator()) const noexcept;

  // If this Tensor is already contiguous, returns `.view()` in O(1); otherwise
  // returns `.clone(allocator)`.
  Result<Tensor> contiguous(
      const Allocator& allocator = default_allocator()) const noexcept {
    if (is_contiguous()) return view();
    return clone(allocator);
  }

  // --- metadata observers ----------------------------------------------------
  bool is_valid() const noexcept {
    return storage_ != nullptr && !shape_.empty();
  }
  DType dtype() const noexcept { return dtype_; }
  const DTypeInfo& dtype_meta() const noexcept { return dtype_info(dtype_); }

  const Shape& shape() const noexcept { return shape_; }
  const Strides& strides() const noexcept { return strides_; }
  Index rank() const noexcept { return shape_.rank(); }
  Index ndim() const noexcept { return shape_.rank(); }
  Index dim(Index axis) const noexcept { return shape_.dim(axis); }
  Index stride(Index axis) const noexcept { return strides_[axis]; }
  Index numel() const noexcept { return shape_.numel(); }
  Size byte_offset() const noexcept { return byte_offset_; }

  bool is_contiguous() const noexcept {
    return is_contiguous_layout(shape_, strides_);
  }
  bool is_mutable() const noexcept {
    return storage_ != nullptr && storage_->is_mutable();
  }
  bool is_quantized() const noexcept { return is_subbyte_dtype(dtype_); }

  // Byte size of the logical elements (valid when contiguous).
  Size nbytes() const noexcept {
    Size out = 0;
    const Status st = checked_dtype_bytes(dtype_, shape_.numel(), &out);
    L3_DCHECK(st.is_ok());
    L3_UNUSED(st);
    return out;
  }

  f32 quant_scale() const noexcept { return quant_scale_; }
  void set_quant_scale(f32 scale) noexcept { quant_scale_ = scale; }

  const Ptr<Storage>& storage() const noexcept { return storage_; }

  // --- autograd interface ----------------------------------------------------

  // Returns true if gradients are tracked for this Tensor.
  bool requires_grad() const noexcept;

  // Enable or disable gradient tracking on this Tensor.
  // Only floating-point tensors (`f32`, `f16`, `bf16`, `f64`) may require grad.
  Status set_requires_grad(bool req = true) noexcept;

  // True if `requires_grad()` is true and this tensor was created by the user
  // (i.e. `grad_node() == nullptr`).
  bool is_leaf() const noexcept;

  // Returns true if a gradient tensor has been accumulated for this Tensor.
  bool has_grad() const noexcept;

  // Read-only or mutable access to the accumulated gradient Tensor.
  // Returns nullptr if `has_grad()` is false.
  const Tensor* grad() const noexcept;
  Tensor* mutable_grad() noexcept;

  // Accumulate `incoming_grad` (must match `shape()` and `kFloat32`) into this
  // Tensor's `.grad()`, allocating `.grad()` lazily on first accumulation.
  Status accumulate_grad(const Tensor& incoming_grad) noexcept;

  // Reset `.grad()` to zeros (if already allocated) or clear it.
  void zero_grad() noexcept;

  // Run reverse-mode automatic differentiation starting from this Tensor.
  // If `grad_output` is nullptr, defaults to `ones_like(*this)` (typically
  // called on a scalar loss tensor).
  Status backward(const Tensor* grad_output = nullptr) const noexcept;

  // Internal autograd graph plumbing (used by `log3/autograd/ops`).
  autograd::GradNode* grad_node() const noexcept;
  Status set_grad_node(Ptr<autograd::GradNode> node) noexcept;
  const Ptr<AutogradMeta>& autograd_meta() const noexcept {
    return autograd_meta_;
  }

  // --- raw & typed data pointers ---------------------------------------------
  const byte* raw_data() const noexcept {
    if (storage_ == nullptr || storage_->data() == nullptr) return nullptr;
    return storage_->data() + byte_offset_;
  }

  byte* raw_mutable_data() noexcept {
    L3_DCHECK(is_mutable());
    if (storage_ == nullptr || storage_->mutable_data() == nullptr) {
      return nullptr;
    }
    return storage_->mutable_data() + byte_offset_;
  }

  template <typename T>
  const T* data_ptr() const noexcept {
    L3_DCHECK_EQ(dtype_, dtype_of<T>());
    return reinterpret_cast<const T*>(raw_data());
  }

  template <typename T>
  T* mutable_data_ptr() noexcept {
    L3_DCHECK_EQ(dtype_, dtype_of<T>());
    L3_DCHECK(is_mutable());
    return reinterpret_cast<T*>(raw_mutable_data());
  }

  // Contiguous typed Span accessors (asserts contiguity and DType match in debug).
  template <typename T>
  Span<const T> span() const noexcept {
    L3_DCHECK(is_contiguous());
    return Span<const T>(data_ptr<T>(), numel());
  }

  template <typename T>
  Span<T> mutable_span() noexcept {
    L3_DCHECK(is_contiguous());
    return Span<T>(mutable_data_ptr<T>(), numel());
  }

  // --- zero-copy view transformations ----------------------------------------

  // Reshape a contiguous tensor to `new_shape` without copying data.
  // Fails with kShapeMismatch if `new_shape.numel() != numel()`, or
  // kFailedPrecondition if the tensor is non-contiguous.
  Result<Tensor> reshape(const Shape& new_shape) const noexcept;

  Result<Tensor> reshape(std::initializer_list<Index> new_dims) const noexcept {
    Shape s;
    L3_ASSIGN_OR_RETURN(s, Shape::from_dims(new_dims));
    return reshape(s);
  }

  // Swap two axes (`dim0` and `dim1`, negative indices supported) in O(1).
  Result<Tensor> transpose(Index dim0, Index dim1) const noexcept;

  // Permute all axes according to `perm` (length must equal `rank()`).
  Result<Tensor> permute(Span<const Index> perm) const noexcept;

  // Slice along `axis` in `[start, end)` with step 1 in O(1).
  Result<Tensor> slice(Index axis, Index start, Index end) const noexcept;

  // Insert a size-1 dimension at `axis` (in `[-rank - 1, rank]`).
  Result<Tensor> unsqueeze(Index axis) const noexcept;

  // Remove a size-1 dimension at `axis` (or remove ALL size-1 dimensions when
  // `axis == -999`).
  Result<Tensor> squeeze(Index axis = -999) const noexcept;

 private:
  Ptr<Storage> storage_;
  Ptr<AutogradMeta> autograd_meta_;
  Shape shape_;
  Strides strides_;
  Size byte_offset_;
  DType dtype_;
  f32 quant_scale_;
};

// -----------------------------------------------------------------------------
// AutogradMeta: Lazily allocated intrusive ref-counted gradient state
// -----------------------------------------------------------------------------
struct L3_API AutogradMeta final : public RefCounted {
  Tensor grad;                        // Accumulated gradient (empty until first backward)
  Ptr<autograd::GradNode> grad_node;  // Creator operation in the computational DAG
  bool requires_grad = false;

  AutogradMeta() noexcept = default;
  ~AutogradMeta() noexcept;
};

}  // namespace l3
