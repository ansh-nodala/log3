// src/nn/layers.cpp
//
// Implementations of RMSNorm, BitLinear, Linear, Embedding, and RotaryEmbedding.

#include "log3/nn/layers.h"

#include <cmath>

#include "log3/core/memory.h"
#include "log3/kernels/cpu_kernels.h"

namespace l3 {
namespace nn {

// -----------------------------------------------------------------------------
// RMSNorm
// -----------------------------------------------------------------------------

Result<RMSNorm> RMSNorm::create(Index dim, f32 eps,
                                const Allocator& allocator) noexcept {
  if (dim <= 0 || eps <= 0.0f) {
    return Status::error(StatusCode::kInvalidArgument,
                         "RMSNorm::create: dim and eps must be positive");
  }
  RMSNorm norm;
  norm.dim_ = dim;
  norm.eps_ = eps;
  L3_ASSIGN_OR_RETURN(
      norm.weight_, Tensor::full({dim}, 1.0f, DType::kFloat32, allocator));
  norm.register_parameter("weight", norm.weight_);
  return norm;
}

Status RMSNorm::forward(const Tensor& x, Tensor* out) const noexcept {
  if (out == nullptr || !x.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "RMSNorm::forward: invalid input or output");
  }
  if (!x.is_contiguous() || !out->is_contiguous() || !out->is_mutable()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "RMSNorm::forward: tensors must be contiguous and output mutable");
  }
  if (x.dtype() != DType::kFloat32 || out->dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "RMSNorm::forward: expects FP32 tensors");
  }
  if (x.rank() < 1 || x.dim(x.rank() - 1) != dim_ ||
      out->shape() != x.shape()) {
    return Status::error(StatusCode::kShapeMismatch,
                         "RMSNorm::forward: shape mismatch with dimension");
  }

  const Index rows = x.numel() / dim_;
  const f32* x_data = x.data_ptr<f32>();
  const f32* w_data = weight_.data_ptr<f32>();
  f32* out_data = out->mutable_data_ptr<f32>();

  for (Index r = 0; r < rows; ++r) {
    kernels::rmsnorm_f32(x_data + r * dim_, w_data, out_data + r * dim_, dim_,
                         eps_);
  }
  return Status::ok();
}

// -----------------------------------------------------------------------------
// BitLinear
// -----------------------------------------------------------------------------

Result<BitLinear> BitLinear::create(Index in_features, Index out_features,
                                    const Allocator& allocator) noexcept {
  if (in_features <= 0 || out_features <= 0 || in_features % 4 != 0) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "BitLinear::create: in_features (%lld) must be positive and divisible "
        "by 4, out_features (%lld) must be positive",
        static_cast<long long>(in_features),
        static_cast<long long>(out_features));
  }

  BitLinear layer;
  layer.in_features_ = in_features;
  layer.out_features_ = out_features;
  L3_ASSIGN_OR_RETURN(layer.weight_,
                      Tensor::zeros({out_features, in_features},
                                    DType::kTernary2, allocator));
  layer.weight_.set_quant_scale(1.0f);
  layer.register_parameter("weight", layer.weight_);
  return layer;
}

Status BitLinear::load_weights_from_f32(const Tensor& w_f32) noexcept {
  if (!w_f32.is_valid() || w_f32.dtype() != DType::kFloat32) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "BitLinear::load_weights_from_f32: requires f32 Tensor");
  }
  if (!w_f32.is_contiguous()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "BitLinear::load_weights_from_f32: w_f32 must be contiguous");
  }
  if (w_f32.rank() != 2 || w_f32.dim(0) != out_features_ ||
      w_f32.dim(1) != in_features_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "BitLinear::load_weights_from_f32: shape mismatch");
  }

  f32 scale = 1.0f;
  kernels::quantize_f32_to_t2_raw(w_f32.data_ptr<f32>(),
                                  weight_.raw_mutable_data(), w_f32.numel(),
                                  &scale);
  weight_.set_quant_scale(scale);
  // Keep the registered parameter metadata in sync with the new quant_scale:
  if (!named_tensors_.empty()) {
    named_tensors_[0].tensor.set_quant_scale(scale);
  }
  return Status::ok();
}

Status BitLinear::forward(const Tensor& x, Tensor* out,
                          ThreadPool& pool) const noexcept {
  if (out == nullptr || !x.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "BitLinear::forward: invalid input or output");
  }
  if (x.dtype() != DType::kFloat32 || out->dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "BitLinear::forward: requires f32 input and output");
  }
  if (!x.is_contiguous() || !out->is_contiguous() || !out->is_mutable()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "BitLinear::forward: tensors must be contiguous");
  }

  const Index r = x.rank();
  if (r < 1 || x.dim(r - 1) != in_features_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "BitLinear::forward: last dimension != in_features");
  }

  const Index M = x.numel() / in_features_;
  if (out->rank() != r || out->dim(r - 1) != out_features_ ||
      out->numel() != M * out_features_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "BitLinear::forward: output shape mismatch");
  }

  Shape shape_2d_x;
  Shape shape_2d_out;
  L3_ASSIGN_OR_RETURN(shape_2d_x, Shape::from_dims({M, in_features_}));
  L3_ASSIGN_OR_RETURN(shape_2d_out, Shape::from_dims({M, out_features_}));

  Tensor x_2d;
  Tensor out_2d;
  L3_ASSIGN_OR_RETURN(x_2d, x.reshape(shape_2d_x));
  L3_ASSIGN_OR_RETURN(out_2d, out->reshape(shape_2d_out));

  return kernels::bitnet_linear_t2_f32(x_2d, weight_, &out_2d, pool);
}

// -----------------------------------------------------------------------------
// Linear
// -----------------------------------------------------------------------------

Result<Linear> Linear::create(Index in_features, Index out_features,
                              bool has_bias,
                              const Allocator& allocator) noexcept {
  if (in_features <= 0 || out_features <= 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Linear::create: dimensions must be positive");
  }

  Linear layer;
  layer.in_features_ = in_features;
  layer.out_features_ = out_features;
  layer.has_bias_ = has_bias;

  L3_ASSIGN_OR_RETURN(layer.weight_,
                      Tensor::zeros({out_features, in_features},
                                    DType::kFloat32, allocator));
  layer.register_parameter("weight", layer.weight_);

  if (has_bias) {
    L3_ASSIGN_OR_RETURN(
        layer.bias_,
        Tensor::zeros({out_features}, DType::kFloat32, allocator));
    layer.register_parameter("bias", layer.bias_);
  }

  return layer;
}

Status Linear::forward(const Tensor& x, Tensor* out,
                       ThreadPool& pool) const noexcept {
  if (out == nullptr || !x.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Linear::forward: invalid input or output");
  }
  if (!x.is_contiguous() || !out->is_contiguous() || !out->is_mutable()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "Linear::forward: tensors must be contiguous and output mutable");
  }
  if (x.dtype() != DType::kFloat32 || out->dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "Linear::forward: requires f32 input and output");
  }
  if (x.rank() < 1 || x.dim(x.rank() - 1) != in_features_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "Linear::forward: last dimension != in_features");
  }
  const Index M = x.numel() / in_features_;
  if (out->rank() != x.rank() || out->dim(out->rank() - 1) != out_features_ ||
      out->numel() != M * out_features_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "Linear::forward: output shape mismatch");
  }

  // Zero-allocation row-wise dot product: since weight_ is stored in
  // (out_features, in_features) row-major order, both x[m, :] and weight_[n, :]
  // are contiguous vectors of length `in_features_`! Using `kernels::dot_f32`
  // directly avoids allocating and transposing a temporary copy of `weight_`
  // on every forward pass!
  const f32* x_ptr = x.data_ptr<f32>();
  const f32* w_ptr = weight_.data_ptr<f32>();
  const f32* b_ptr = has_bias_ ? bias_.data_ptr<f32>() : nullptr;
  f32* y_ptr = out->mutable_data_ptr<f32>();

  const Index total_outputs = M * out_features_;
  pool.parallel_for(0, total_outputs, 0,
                    [&](Index /*tid*/, Index begin, Index end) noexcept {
                      for (Index idx = begin; idx < end; ++idx) {
                        const Index m = idx / out_features_;
                        const Index n = idx % out_features_;
                        f32 acc = kernels::dot_f32(
                            x_ptr + m * in_features_,
                            w_ptr + n * in_features_, in_features_);
                        if (b_ptr != nullptr) acc += b_ptr[n];
                        y_ptr[idx] = acc;
                      }
                    });

  return Status::ok();
}

// -----------------------------------------------------------------------------
// Embedding
// -----------------------------------------------------------------------------

Result<Embedding> Embedding::create(Index vocab_size, Index embedding_dim,
                                    const Allocator& allocator) noexcept {
  if (vocab_size <= 0 || embedding_dim <= 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "Embedding::create: dimensions must be positive");
  }
  Embedding layer;
  layer.vocab_size_ = vocab_size;
  layer.embedding_dim_ = embedding_dim;
  L3_ASSIGN_OR_RETURN(
      layer.weight_,
      Tensor::zeros({vocab_size, embedding_dim}, DType::kFloat32, allocator));
  layer.register_parameter("weight", layer.weight_);
  return layer;
}

Status Embedding::forward(Span<const Index> tokens, Tensor* out) const noexcept {
  if (out == nullptr || !out->is_valid() || !out->is_contiguous() ||
      !out->is_mutable()) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "Embedding::forward: output must be a valid contiguous mutable Tensor");
  }
  if (out->dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "Embedding::forward: output must be f32");
  }
  const Index seq_len = tokens.size();
  if (out->numel() != seq_len * embedding_dim_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "Embedding::forward: output size mismatch");
  }

  const f32* table = weight_.data_ptr<f32>();
  f32* out_data = out->mutable_data_ptr<f32>();
  const Size row_bytes = static_cast<Size>(embedding_dim_) * sizeof(f32);

  for (Index i = 0; i < seq_len; ++i) {
    const Index tok = tokens[i];
    if (L3_UNLIKELY(tok < 0 || tok >= vocab_size_)) {
      return Status::errorf(
          StatusCode::kOutOfRange,
          "Embedding::forward: token ID %lld out of range [0, %lld)",
          static_cast<long long>(tok), static_cast<long long>(vocab_size_));
    }
    mem_copy(out_data + i * embedding_dim_, table + tok * embedding_dim_,
             row_bytes);
  }

  return Status::ok();
}

// -----------------------------------------------------------------------------
// RotaryEmbedding
// -----------------------------------------------------------------------------

Result<RotaryEmbedding> RotaryEmbedding::create(Index head_dim,
                                                f32 theta_base) noexcept {
  if (head_dim <= 0 || head_dim % 2 != 0 || theta_base <= 0.0f) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "RotaryEmbedding::create: head_dim must be positive/even and theta_base > 0");
  }
  RotaryEmbedding rope;
  rope.head_dim_ = head_dim;
  rope.theta_base_ = theta_base;
  return rope;
}

Status RotaryEmbedding::apply(Tensor* qk, Index start_pos) const noexcept {
  if (qk == nullptr || !qk->is_valid() || !qk->is_contiguous() ||
      !qk->is_mutable()) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "RotaryEmbedding::apply: input must be a valid contiguous mutable Tensor");
  }
  if (start_pos < 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "RotaryEmbedding::apply: start_pos cannot be negative");
  }
  if (qk->dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "RotaryEmbedding::apply: expects f32 Tensor");
  }
  if (qk->rank() < 2) {
    return Status::error(
        StatusCode::kShapeMismatch,
        "RotaryEmbedding::apply: expects rank >= 2 (..., num_heads, head_dim)");
  }
  if (qk->dim(qk->rank() - 1) != head_dim_) {
    return Status::error(StatusCode::kShapeMismatch,
                         "RotaryEmbedding::apply: last dimension != head_dim");
  }

  const Index head_dim = head_dim_;
  const Index num_heads = qk->dim(qk->rank() - 2);
  if (num_heads <= 0 || qk->numel() == 0) {
    return Status::ok();
  }
  const Index seq_len = qk->numel() / (num_heads * head_dim);

  f32* data = qk->mutable_data_ptr<f32>();
  for (Index s = 0; s < seq_len; ++s) {
    const Index pos = start_pos + s;
    for (Index h = 0; h < num_heads; ++h) {
      f32* head_vec = data + (s * num_heads + h) * head_dim;
      kernels::rope_f32_inplace(head_vec, head_dim, pos, theta_base_);
    }
  }

  return Status::ok();
}

}  // namespace nn
}  // namespace l3
