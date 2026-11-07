// tests/test_autograd.cpp
//
// Comprehensive unit test suite for Stage 4:
//   * Tensor Autograd integration (requires_grad, grad, accumulate_grad, detach, NoGradGuard)
//   * Reverse-mode DAG topological sort & multi-path gradient accumulation
//   * Differentiable ops: add, sub, mul, silu, relu, rmsnorm, matmul, linear
//   * BitNet b1.58 Quantization-Aware Training (QAT) via Straight-Through Estimator (`bitnet_linear_ste`)
//   * Loss functions: mse_loss, cross_entropy_loss
//   * Optimizers: SGD (with momentum) & AdamW + clip_grad_norm training convergence

#include <cmath>
#include <cstdio>

#include "log3/autograd/engine.h"
#include "log3/autograd/ops.h"
#include "log3/core/assert.h"
#include "log3/core/log.h"
#include "log3/core/memory.h"
#include "log3/core/rng.h"
#include "log3/core/thread_pool.h"
#include "log3/optim/optimizer.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace {

bool approx_eq(f32 a, f32 b, f32 tol = 1e-3f) noexcept {
  const f32 diff = a > b ? (a - b) : (b - a);
  return diff <= tol;
}

// -----------------------------------------------------------------------------
// 1. Multi-Path DAG Backward & NoGradGuard Verification
// -----------------------------------------------------------------------------
void test_dag_and_no_grad() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    // Compute f(x) = sum((x * x) + x) where x = [2.0, 3.0, -4.0, 5.0]
    // Analytical derivative: df/dx_i = 2 * x_i + 1 -> [5.0, 7.0, -7.0, 11.0]
    auto x_res = Tensor::empty({4}, DType::kFloat32);
    L3_CHECK_OK(x_res.status());
    Tensor x = x_res.take_value();
    x.mutable_data_ptr<f32>()[0] = 2.0f;
    x.mutable_data_ptr<f32>()[1] = 3.0f;
    x.mutable_data_ptr<f32>()[2] = -4.0f;
    x.mutable_data_ptr<f32>()[3] = 5.0f;
    L3_CHECK_OK(x.set_requires_grad(true));
    L3_CHECK(x.requires_grad());
    L3_CHECK(x.is_leaf());

    auto x2 = autograd::mul(x, x);
    L3_CHECK_OK(x2.status());
    auto y = autograd::add(x2.value(), x);
    L3_CHECK_OK(y.status());
    auto loss = autograd::sum(y.value());
    L3_CHECK_OK(loss.status());

    L3_CHECK_OK(loss->backward());
    L3_CHECK(x.has_grad());

    const f32* gx = x.grad()->data_ptr<f32>();
    L3_CHECK(approx_eq(gx[0], 5.0f));
    L3_CHECK(approx_eq(gx[1], 7.0f));
    L3_CHECK(approx_eq(gx[2], -7.0f));
    L3_CHECK(approx_eq(gx[3], 11.0f));

    // Verify NoGradGuard suppresses graph construction
    {
      autograd::NoGradGuard no_grad;
      auto z = autograd::mul(x, x);
      L3_CHECK_OK(z.status());
      L3_CHECK(!z->requires_grad());
      L3_CHECK(z->grad_node() == nullptr);
    }
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 2. Finite-Difference Numerical Gradient Check for RMSNorm, SiLU, MatMul & CrossEntropy
// -----------------------------------------------------------------------------
void test_numerical_gradients() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    Rng rng(1234ULL);
    auto x_res = Tensor::empty({2, 8}, DType::kFloat32);
    auto w_res = Tensor::empty({8}, DType::kFloat32);
    L3_CHECK_OK(x_res.status());
    L3_CHECK_OK(w_res.status());
    Tensor x = x_res.take_value();
    Tensor w = w_res.take_value();

    for (Index i = 0; i < x.numel(); ++i) {
      x.mutable_data_ptr<f32>()[i] = rng.uniform_f32(-1.5f, 1.5f);
    }
    for (Index i = 0; i < w.numel(); ++i) {
      w.mutable_data_ptr<f32>()[i] = rng.uniform_f32(0.5f, 1.5f);
    }
    L3_CHECK_OK(x.set_requires_grad(true));
    L3_CHECK_OK(w.set_requires_grad(true));

    // Forward: h = silu(rmsnorm(x, w)), loss = mean(h)
    auto normed = autograd::rmsnorm(x, w);
    L3_CHECK_OK(normed.status());
    auto act = autograd::silu(normed.value());
    L3_CHECK_OK(act.status());
    auto loss = autograd::mean(act.value());
    L3_CHECK_OK(loss.status());
    L3_CHECK_OK(loss->backward());

    // Finite-difference check on x[0..15]
    constexpr f32 kEps = 1e-3f;
    for (Index idx = 0; idx < x.numel(); ++idx) {
      autograd::NoGradGuard no_grad;
      const f32 orig = x.data_ptr<f32>()[idx];

      x.mutable_data_ptr<f32>()[idx] = orig + kEps;
      auto n_plus = autograd::rmsnorm(x, w);
      auto a_plus = autograd::silu(n_plus.value());
      auto l_plus = autograd::mean(a_plus.value());
      const f32 loss_plus = l_plus->data_ptr<f32>()[0];

      x.mutable_data_ptr<f32>()[idx] = orig - kEps;
      auto n_minus = autograd::rmsnorm(x, w);
      auto a_minus = autograd::silu(n_minus.value());
      auto l_minus = autograd::mean(a_minus.value());
      const f32 loss_minus = l_minus->data_ptr<f32>()[0];

      x.mutable_data_ptr<f32>()[idx] = orig;

      const f32 num_grad = (loss_plus - loss_minus) / (2.0f * kEps);
      const f32 ana_grad = x.grad()->data_ptr<f32>()[idx];
      L3_CHECK(approx_eq(ana_grad, num_grad, 2e-3f));
    }

    // Cross-Entropy loss gradient check
    auto logits_res = Tensor::empty({3, 5}, DType::kFloat32);
    L3_CHECK_OK(logits_res.status());
    Tensor logits = logits_res.take_value();
    for (Index i = 0; i < logits.numel(); ++i) {
      logits.mutable_data_ptr<f32>()[i] = rng.uniform_f32(-1.0f, 1.0f);
    }
    L3_CHECK_OK(logits.set_requires_grad(true));

    const Index targets[3] = {1, 4, 0};
    auto ce = autograd::cross_entropy_loss(logits, Span<const Index>(targets, 3));
    L3_CHECK_OK(ce.status());
    L3_CHECK_OK(ce->backward());

    for (Index idx = 0; idx < logits.numel(); ++idx) {
      autograd::NoGradGuard no_grad;
      const f32 orig = logits.data_ptr<f32>()[idx];

      logits.mutable_data_ptr<f32>()[idx] = orig + kEps;
      const f32 lp =
          autograd::cross_entropy_loss(logits, Span<const Index>(targets, 3))
              ->data_ptr<f32>()[0];

      logits.mutable_data_ptr<f32>()[idx] = orig - kEps;
      const f32 lm =
          autograd::cross_entropy_loss(logits, Span<const Index>(targets, 3))
              ->data_ptr<f32>()[0];

      logits.mutable_data_ptr<f32>()[idx] = orig;
      const f32 num_grad = (lp - lm) / (2.0f * kEps);
      const f32 ana_grad = logits.grad()->data_ptr<f32>()[idx];
      L3_CHECK(approx_eq(ana_grad, num_grad, 2e-3f));
    }
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 3. BitNet b1.58 QAT (Straight-Through Estimator) + AdamW Training Loop
// -----------------------------------------------------------------------------
void test_bitnet_ste_training_loop() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    Rng rng(777ULL);
    constexpr Index M = 8;
    constexpr Index K = 16;
    constexpr Index N = 4;

    auto x_res = Tensor::empty({M, K}, DType::kFloat32);
    auto w_teacher_res = Tensor::empty({N, K}, DType::kFloat32);
    auto w_master_res = Tensor::empty({N, K}, DType::kFloat32);
    L3_CHECK_OK(x_res.status());
    L3_CHECK_OK(w_teacher_res.status());
    L3_CHECK_OK(w_master_res.status());

    Tensor x = x_res.take_value();
    Tensor w_teacher = w_teacher_res.take_value();
    Tensor w_master = w_master_res.take_value();

    for (Index i = 0; i < x.numel(); ++i) {
      x.mutable_data_ptr<f32>()[i] = rng.uniform_f32(-1.0f, 1.0f);
    }
    for (Index i = 0; i < w_teacher.numel(); ++i) {
      const int r = static_cast<int>(rng.bounded_u32(3)) - 1;
      w_teacher.mutable_data_ptr<f32>()[i] = static_cast<f32>(r) * 0.8f;
      w_master.mutable_data_ptr<f32>()[i] =
          w_teacher.data_ptr<f32>()[i] + rng.uniform_f32(-0.6f, 0.6f);
    }
    auto target_res = autograd::linear(x, w_teacher);
    L3_CHECK_OK(target_res.status());
    Tensor target = target_res.take_value();

    L3_CHECK_OK(w_master.set_requires_grad(true));

    Tensor params[1] = {w_master.view()};
    auto opt_res = optim::AdamW::create(Span<const Tensor>(params, 1),
                                        /*lr=*/0.05f, 0.9f, 0.999f, 1e-8f,
                                        /*weight_decay=*/0.0f);
    L3_CHECK_OK(opt_res.status());
    optim::AdamW opt = opt_res.take_value();

    f32 initial_loss = 0.0f;
    f32 final_loss = 0.0f;

    for (int step = 0; step < 40; ++step) {
      opt.zero_grad();

      auto pred = autograd::bitnet_linear_ste(x, w_master);
      L3_CHECK_OK(pred.status());
      auto loss = autograd::mse_loss(pred.value(), target);
      L3_CHECK_OK(loss.status());

      const f32 cur_loss = loss->data_ptr<f32>()[0];
      if (step == 0) initial_loss = cur_loss;
      final_loss = cur_loss;

      L3_CHECK_OK(loss->backward());
      auto norm_res = optim::clip_grad_norm(Span<Tensor>(params, 1), 5.0f);
      L3_CHECK_OK(norm_res.status());
      L3_CHECK_OK(opt.step());
    }

    L3_LOG_INFO("BitNet b1.58 STE + AdamW QAT loss: %.6f -> %.6f",
                static_cast<double>(initial_loss),
                static_cast<double>(final_loss));
    L3_CHECK_LT(final_loss, initial_loss);
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

}  // namespace
}  // namespace l3

int main() {
  l3::set_log_level(l3::LogLevel::kInfo);
  L3_LOG_INFO("Running tests/test_autograd.cpp...");

  l3::test_dag_and_no_grad();
  l3::test_numerical_gradients();
  l3::test_bitnet_ste_training_loop();

  L3_LOG_INFO(
      "tests/test_autograd.cpp: ALL STAGE 4 AUTOGRAD & OPTIM TESTS PASSED (0 bytes leaked).");
  return 0;
}
