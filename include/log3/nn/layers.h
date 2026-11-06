// log3/nn/layers.h
//
// Core neural network layers:
//   * RMSNorm        : Root Mean Square Layer Normalization (Llama/BitNet)
//   * BitLinear      : 1.58-bit ternary linear projection layer
//   * Linear         : Standard FP32/FP16 dense layer
//   * Embedding      : Token lookup table
//   * RotaryEmbedding: Rotary Position Embeddings (RoPE) cache & operator
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/kernels/cpu_kernels.h"
#include "log3/nn/module.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace nn {

// -----------------------------------------------------------------------------
// RMSNorm Layer
// -----------------------------------------------------------------------------
class L3_API RMSNorm final : public Module {
 public:
  RMSNorm() noexcept : dim_(0), eps_(1e-5f) {}

  static Result<RMSNorm> create(
      Index dim, f32 eps = 1e-5f,
      const Allocator& allocator = default_allocator()) noexcept;

  Index dim() const noexcept { return dim_; }
  f32 eps() const noexcept { return eps_; }
  Tensor& weight() noexcept { return weight_; }
  const Tensor& weight() const noexcept { return weight_; }

  // Forward pass: y = rmsnorm(x, weight, eps).
  // x can be 2D (B, D) or 3D (B, S, D).
  Status forward(const Tensor& x, Tensor* out) const noexcept;

 private:
  Index dim_;
  f32 eps_;
  Tensor weight_;
};

// -----------------------------------------------------------------------------
// BitLinear Layer (BitNet b1.58 ternary weights {-1, 0, +1})
// -----------------------------------------------------------------------------
class L3_API BitLinear final : public Module {
 public:
  BitLinear() noexcept : in_features_(0), out_features_(0) {}

  static Result<BitLinear> create(
      Index in_features, Index out_features,
      const Allocator& allocator = default_allocator()) noexcept;

  Index in_features() const noexcept { return in_features_; }
  Index out_features() const noexcept { return out_features_; }
  Tensor& weight() noexcept { return weight_; }
  const Tensor& weight() const noexcept { return weight_; }

  // Load FP32 weights and quantize them into BitNet b1.58 format in-place:
  Status load_weights_from_f32(const Tensor& w_f32) noexcept;

  // Forward pass: out = x @ weight^T.
  // Supports 2D (B, in_feat) and 3D (B, S, in_feat) activations.
  Status forward(const Tensor& x, Tensor* out,
                 ThreadPool& pool = default_thread_pool()) const noexcept;

 private:
  Index in_features_;
  Index out_features_;
  Tensor weight_;  // DType::kTernary2, shape: (out_features, in_features)
};

// -----------------------------------------------------------------------------
// Linear Layer (Standard FP32 dense layer)
// -----------------------------------------------------------------------------
class L3_API Linear final : public Module {
 public:
  Linear() noexcept : in_features_(0), out_features_(0), has_bias_(false) {}

  static Result<Linear> create(
      Index in_features, Index out_features, bool has_bias = false,
      const Allocator& allocator = default_allocator()) noexcept;

  Index in_features() const noexcept { return in_features_; }
  Index out_features() const noexcept { return out_features_; }
  Tensor& weight() noexcept { return weight_; }
  const Tensor& weight() const noexcept { return weight_; }
  Tensor& bias() noexcept { return bias_; }
  const Tensor& bias() const noexcept { return bias_; }

  Status forward(const Tensor& x, Tensor* out,
                 ThreadPool& pool = default_thread_pool()) const noexcept;

 private:
  Index in_features_;
  Index out_features_;
  bool has_bias_;
  Tensor weight_;  // shape: (out_features, in_features)
  Tensor bias_;    // shape: (out_features)
};

// -----------------------------------------------------------------------------
// Embedding Layer (Token lookup table)
// -----------------------------------------------------------------------------
class L3_API Embedding final : public Module {
 public:
  Embedding() noexcept : vocab_size_(0), embedding_dim_(0) {}

  static Result<Embedding> create(
      Index vocab_size, Index embedding_dim,
      const Allocator& allocator = default_allocator()) noexcept;

  Index vocab_size() const noexcept { return vocab_size_; }
  Index embedding_dim() const noexcept { return embedding_dim_; }
  Tensor& weight() noexcept { return weight_; }
  const Tensor& weight() const noexcept { return weight_; }

  // Forward lookup: input token ids (Index array of length `seq_len`) ->
  // `out` of shape `(seq_len, embedding_dim)`.
  Status forward(Span<const Index> tokens, Tensor* out) const noexcept;

 private:
  Index vocab_size_;
  Index embedding_dim_;
  Tensor weight_;  // shape: (vocab_size, embedding_dim)
};

// -----------------------------------------------------------------------------
// Rotary Embedding (RoPE)
// -----------------------------------------------------------------------------
class L3_API RotaryEmbedding final : public Module {
 public:
  RotaryEmbedding() noexcept : head_dim_(0), theta_base_(10000.0f) {}

  static Result<RotaryEmbedding> create(Index head_dim,
                                        f32 theta_base = 10000.0f) noexcept;

  Index head_dim() const noexcept { return head_dim_; }

  // In-place rotary embedding applied to Q or K of shape (seq_len, num_heads, head_dim)
  // starting at sequence position `start_pos`.
  Status apply(Tensor* qk, Index start_pos = 0) const noexcept;

 private:
  Index head_dim_;
  f32 theta_base_;
};

}  // namespace nn
}  // namespace l3
