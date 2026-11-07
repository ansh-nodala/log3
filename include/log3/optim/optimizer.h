// log3/optim/optimizer.h
//
// First-order and adaptive gradient optimizers (`SGD`, `AdamW`) and global
// L2 gradient norm clipping (`clip_grad_norm`) for training BitNet b1.58 and
// standard neural models in log3.
//
// Design notes:
//   * All parameter updates execute under `autograd::NoGradGuard` so weight
//     updates never record into the autograd graph.
//   * Uses `SmallVec` for parameter lists and moment buffers (`m`, `v`) — zero
//     `std::vector` or `std::unordered_map` overhead.
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include "log3/core/platform.h"
#include "log3/core/small_vector.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace optim {

// Clip the global L2 norm of all gradients across `params` to `max_norm`.
// Returns the total unclipped L2 norm.
L3_API Result<f32> clip_grad_norm(Span<Tensor> params, f32 max_norm) noexcept;

// -----------------------------------------------------------------------------
// SGD (Stochastic Gradient Descent with optional Momentum & Weight Decay)
// -----------------------------------------------------------------------------
class L3_API SGD {
 public:
  SGD() noexcept : lr_(1e-3f), momentum_(0.0f), weight_decay_(0.0f) {}

  static Result<SGD> create(Span<const Tensor> params, f32 lr = 1e-2f,
                            f32 momentum = 0.0f,
                            f32 weight_decay = 0.0f) noexcept;

  void zero_grad() noexcept;
  Status step() noexcept;

  f32 lr() const noexcept { return lr_; }
  void set_lr(f32 lr) noexcept { lr_ = lr; }

 private:
  f32 lr_;
  f32 momentum_;
  f32 weight_decay_;
  SmallVec<Tensor, 32> params_;
  SmallVec<Tensor, 32> velocities_;
};

// -----------------------------------------------------------------------------
// AdamW (Decoupled Weight Decay Regularization, Loshchilov & Hutter)
// -----------------------------------------------------------------------------
class L3_API AdamW {
 public:
  AdamW() noexcept
      : lr_(1e-3f),
        beta1_(0.9f),
        beta2_(0.999f),
        eps_(1e-8f),
        weight_decay_(0.01f),
        step_count_(0) {}

  static Result<AdamW> create(Span<const Tensor> params, f32 lr = 1e-3f,
                              f32 beta1 = 0.9f, f32 beta2 = 0.999f,
                              f32 eps = 1e-8f,
                              f32 weight_decay = 0.01f) noexcept;

  void zero_grad() noexcept;
  Status step() noexcept;

  f32 lr() const noexcept { return lr_; }
  void set_lr(f32 lr) noexcept { lr_ = lr; }
  Index step_count() const noexcept { return step_count_; }

 private:
  f32 lr_;
  f32 beta1_;
  f32 beta2_;
  f32 eps_;
  f32 weight_decay_;
  Index step_count_;
  SmallVec<Tensor, 32> params_;
  SmallVec<Tensor, 32> m_;  // 1st moment (mean)
  SmallVec<Tensor, 32> v_;  // 2nd moment (uncentered variance)
};

}  // namespace optim
}  // namespace l3
