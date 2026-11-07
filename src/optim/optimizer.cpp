// src/optim/optimizer.cpp
//
// Implementation of clip_grad_norm, SGD, and AdamW optimizers.

#include "log3/optim/optimizer.h"

#include <cmath>

#include "log3/autograd/engine.h"
#include "log3/kernels/cpu_kernels.h"

namespace l3 {
namespace optim {

Result<f32> clip_grad_norm(Span<Tensor> params, f32 max_norm) noexcept {
  if (max_norm <= 0.0f) {
    return Status::error(StatusCode::kInvalidArgument,
                         "clip_grad_norm: max_norm must be positive");
  }

  f64 total_sq = 0.0;
  for (Index i = 0; i < params.size(); ++i) {
    const Tensor* g = params[i].grad();
    if (g != nullptr && g->is_valid()) {
      const f32* gp = g->data_ptr<f32>();
      total_sq += static_cast<f64>(kernels::dot_f32(gp, gp, g->numel()));
    }
  }

  const f32 total_norm = static_cast<f32>(std::sqrt(total_sq));
  if (total_norm > max_norm) {
    const f32 scale = max_norm / (total_norm + 1e-6f);
    for (Index i = 0; i < params.size(); ++i) {
      Tensor* g = params[i].mutable_grad();
      if (g != nullptr && g->is_valid()) {
        f32* gp = g->mutable_data_ptr<f32>();
        const Index n = g->numel();
        for (Index j = 0; j < n; ++j) {
          gp[j] *= scale;
        }
      }
    }
  }
  return total_norm;
}

// -----------------------------------------------------------------------------
// SGD
// -----------------------------------------------------------------------------

Result<SGD> SGD::create(Span<const Tensor> params, f32 lr, f32 momentum,
                        f32 weight_decay) noexcept {
  if (lr <= 0.0f || momentum < 0.0f || weight_decay < 0.0f) {
    return Status::error(StatusCode::kInvalidArgument,
                         "SGD::create: invalid hyperparameters");
  }
  SGD opt;
  opt.lr_ = lr;
  opt.momentum_ = momentum;
  opt.weight_decay_ = weight_decay;

  for (Index i = 0; i < params.size(); ++i) {
    L3_RETURN_IF_ERROR(opt.params_.push_back(params[i].view()));
    if (momentum > 0.0f) {
      Tensor vel;
      L3_ASSIGN_OR_RETURN(
          vel, Tensor::zeros(params[i].shape(), DType::kFloat32));
      L3_RETURN_IF_ERROR(opt.velocities_.push_back(std::move(vel)));
    }
  }
  return opt;
}

void SGD::zero_grad() noexcept {
  for (Index i = 0; i < params_.size(); ++i) {
    params_[i].zero_grad();
  }
}

Status SGD::step() noexcept {
  autograd::NoGradGuard no_grad;

  for (Index i = 0; i < params_.size(); ++i) {
    Tensor& p = params_[i];
    const Tensor* g = p.grad();
    if (g == nullptr || !g->is_valid()) continue;

    f32* pw = p.mutable_data_ptr<f32>();
    const f32* gw = g->data_ptr<f32>();
    const Index n = p.numel();

    if (momentum_ > 0.0f) {
      f32* vw = velocities_[i].mutable_data_ptr<f32>();
      for (Index j = 0; j < n; ++j) {
        const f32 grad_wd = gw[j] + weight_decay_ * pw[j];
        vw[j] = momentum_ * vw[j] + grad_wd;
        pw[j] -= lr_ * vw[j];
      }
    } else {
      for (Index j = 0; j < n; ++j) {
        const f32 grad_wd = gw[j] + weight_decay_ * pw[j];
        pw[j] -= lr_ * grad_wd;
      }
    }
  }
  return Status::ok();
}

// -----------------------------------------------------------------------------
// AdamW
// -----------------------------------------------------------------------------

Result<AdamW> AdamW::create(Span<const Tensor> params, f32 lr, f32 beta1,
                            f32 beta2, f32 eps, f32 weight_decay) noexcept {
  if (lr <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f ||
      beta2 >= 1.0f || eps <= 0.0f || weight_decay < 0.0f) {
    return Status::error(StatusCode::kInvalidArgument,
                         "AdamW::create: invalid hyperparameters");
  }

  AdamW opt;
  opt.lr_ = lr;
  opt.beta1_ = beta1;
  opt.beta2_ = beta2;
  opt.eps_ = eps;
  opt.weight_decay_ = weight_decay;
  opt.step_count_ = 0;

  for (Index i = 0; i < params.size(); ++i) {
    L3_RETURN_IF_ERROR(opt.params_.push_back(params[i].view()));
    Tensor m_buf;
    Tensor v_buf;
    L3_ASSIGN_OR_RETURN(
        m_buf, Tensor::zeros(params[i].shape(), DType::kFloat32));
    L3_ASSIGN_OR_RETURN(
        v_buf, Tensor::zeros(params[i].shape(), DType::kFloat32));
    L3_RETURN_IF_ERROR(opt.m_.push_back(std::move(m_buf)));
    L3_RETURN_IF_ERROR(opt.v_.push_back(std::move(v_buf)));
  }
  return opt;
}

void AdamW::zero_grad() noexcept {
  for (Index i = 0; i < params_.size(); ++i) {
    params_[i].zero_grad();
  }
}

Status AdamW::step() noexcept {
  autograd::NoGradGuard no_grad;
  ++step_count_;

  const f32 bias_corr1 =
      1.0f - std::pow(beta1_, static_cast<f32>(step_count_));
  const f32 bias_corr2 =
      1.0f - std::pow(beta2_, static_cast<f32>(step_count_));
  const f32 step_size = lr_ / bias_corr1;
  const f32 inv_sqrt_bc2 = 1.0f / std::sqrt(bias_corr2);

  for (Index i = 0; i < params_.size(); ++i) {
    Tensor& p = params_[i];
    const Tensor* g = p.grad();
    if (g == nullptr || !g->is_valid()) continue;

    f32* pw = p.mutable_data_ptr<f32>();
    const f32* gw = g->data_ptr<f32>();
    f32* mw = m_[i].mutable_data_ptr<f32>();
    f32* vw = v_[i].mutable_data_ptr<f32>();
    const Index n = p.numel();

    for (Index j = 0; j < n; ++j) {
      // Decoupled weight decay
      pw[j] -= lr_ * weight_decay_ * pw[j];

      // Moment updates
      const f32 grad = gw[j];
      mw[j] = beta1_ * mw[j] + (1.0f - beta1_) * grad;
      vw[j] = beta2_ * vw[j] + (1.0f - beta2_) * (grad * grad);

      const f32 denom = std::sqrt(vw[j]) * inv_sqrt_bc2 + eps_;
      pw[j] -= step_size * (mw[j] / denom);
    }
  }
  return Status::ok();
}

}  // namespace optim
}  // namespace l3
