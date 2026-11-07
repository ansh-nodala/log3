// log3/autograd/ops.h
//
// Differentiable tensor operations and loss functions with automatic DAG
// construction when `GradMode::is_enabled()` and any input has
// `requires_grad() == true`.
//
// Includes BitNet b1.58 Quantization-Aware Training (QAT) via the Straight-
// Through Estimator (`bitnet_linear_ste`):
//   * Forward pass quantizes the latent FP32 master weights `w_master_f32`
//     on the fly using BitNet absmean ternary quantization (`{-1, 0, +1} * scale`)
//     and computes `Y = X @ W_t2^T * scale`.
//   * Backward pass propagates gradients directly to `x` (using the quantized
//     ternary weights) AND straight-through to the FP32 master weights
//     `dL/dW_master = dL/dY^T @ X`, exactly as specified in the BitNet b1.58
//     paper ("The Era of 1-bit LLMs").
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include "log3/autograd/engine.h"
#include "log3/core/platform.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace autograd {

// --- Elementwise & Binary Operations -----------------------------------------

// z = a + b (same shape, f32)
L3_API Result<Tensor> add(const Tensor& a, const Tensor& b) noexcept;

// z = a - b (same shape, f32)
L3_API Result<Tensor> sub(const Tensor& a, const Tensor& b) noexcept;

// z = a * b (elementwise Hadamard product, same shape, f32)
L3_API Result<Tensor> mul(const Tensor& a, const Tensor& b) noexcept;

// --- Activations & Normalization ---------------------------------------------

// y = silu(x) = x * sigmoid(x)
L3_API Result<Tensor> silu(const Tensor& x) noexcept;

// y = relu(x) = max(0, x)
L3_API Result<Tensor> relu(const Tensor& x) noexcept;

// y = rmsnorm(x, weight, eps) where x is (M, D) and weight is (D)
L3_API Result<Tensor> rmsnorm(const Tensor& x, const Tensor& weight,
                              f32 eps = 1e-5f) noexcept;

// --- Matrix Multiplication & BitNet STE Linear Projections -------------------

// C(M, N) = A(M, K) @ B(K, N)
L3_API Result<Tensor> matmul(const Tensor& a, const Tensor& b) noexcept;

// Standard dense linear projection: Y(M, N) = X(M, K) @ W(N, K)^T
L3_API Result<Tensor> linear(const Tensor& x, const Tensor& weight) noexcept;

// BitNet b1.58 Straight-Through Estimator (STE) Linear Projection for QAT:
//   * Quantizes `w_master_f32` (N, K) -> temporary `w_t2` (kTernary2) in the
//     forward pass and runs SIMD `bitnet_linear_t2_f32`.
//   * In the backward pass, computes `dL/dX = dL/dY @ W_dequant` and
//     `dL/dW_master = dL/dY^T @ X` (Straight-Through Estimator!).
L3_API Result<Tensor> bitnet_linear_ste(const Tensor& x,
                                        const Tensor& w_master_f32) noexcept;

// --- Reductions & Loss Functions ---------------------------------------------

// 0-D scalar sum of all elements in `x`.
L3_API Result<Tensor> sum(const Tensor& x) noexcept;

// 0-D scalar mean of all elements in `x`.
L3_API Result<Tensor> mean(const Tensor& x) noexcept;

// Mean Squared Error loss: `mean((pred - target)^2)` -> 0-D scalar Tensor.
L3_API Result<Tensor> mse_loss(const Tensor& pred,
                               const Tensor& target) noexcept;

// Numerically stable Cross-Entropy Loss over logits of shape `(N, vocab_size)`
// and integer target token IDs `targets[0..N)` -> 0-D scalar Tensor.
L3_API Result<Tensor> cross_entropy_loss(const Tensor& logits,
                                         Span<const Index> targets) noexcept;

}  // namespace autograd
}  // namespace l3
