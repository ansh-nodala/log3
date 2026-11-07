// src/autograd/ops.cpp
//
// Implementation of differentiable tensor operations, backward GradNodes,
// BitNet b1.58 Straight-Through Estimator (STE), and loss functions.

#include "log3/autograd/ops.h"

#include <cmath>

#include "log3/core/memory.h"
#include "log3/kernels/cpu_kernels.h"

namespace l3 {
namespace autograd {

namespace {

bool should_record(const Tensor& a) noexcept {
  return GradMode::is_enabled() && a.requires_grad();
}

bool should_record(const Tensor& a, const Tensor& b) noexcept {
  return GradMode::is_enabled() && (a.requires_grad() || b.requires_grad());
}

// -----------------------------------------------------------------------------
// GradNodes for each operation
// -----------------------------------------------------------------------------

class AddBackward final : public GradNode {
 public:
  AddBackward() noexcept : GradNode("AddBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    if (inputs_[0].requires_grad()) {
      L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(grad_output));
    }
    if (inputs_[1].requires_grad()) {
      L3_RETURN_IF_ERROR(inputs_[1].accumulate_grad(grad_output));
    }
    return Status::ok();
  }
};

class SubBackward final : public GradNode {
 public:
  SubBackward() noexcept : GradNode("SubBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    if (inputs_[0].requires_grad()) {
      L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(grad_output));
    }
    if (inputs_[1].requires_grad()) {
      Tensor neg_grad;
      L3_ASSIGN_OR_RETURN(
          neg_grad, Tensor::empty(grad_output.shape(), DType::kFloat32));
      const f32* g = grad_output.data_ptr<f32>();
      f32* ng = neg_grad.mutable_data_ptr<f32>();
      const Index n = grad_output.numel();
      for (Index i = 0; i < n; ++i) ng[i] = -g[i];
      L3_RETURN_IF_ERROR(inputs_[1].accumulate_grad(neg_grad));
    }
    return Status::ok();
  }
};

class MulBackward final : public GradNode {
 public:
  MulBackward() noexcept : GradNode("MulBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    const Index n = grad_output.numel();
    if (inputs_[0].requires_grad()) {
      Tensor ga;
      L3_ASSIGN_OR_RETURN(ga,
                          Tensor::empty(grad_output.shape(), DType::kFloat32));
      kernels::mul_f32(grad_output.data_ptr<f32>(), inputs_[1].data_ptr<f32>(),
                       ga.mutable_data_ptr<f32>(), n);
      L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(ga));
    }
    if (inputs_[1].requires_grad()) {
      Tensor gb;
      L3_ASSIGN_OR_RETURN(gb,
                          Tensor::empty(grad_output.shape(), DType::kFloat32));
      kernels::mul_f32(grad_output.data_ptr<f32>(), inputs_[0].data_ptr<f32>(),
                       gb.mutable_data_ptr<f32>(), n);
      L3_RETURN_IF_ERROR(inputs_[1].accumulate_grad(gb));
    }
    return Status::ok();
  }
};

class SiluBackward final : public GradNode {
 public:
  SiluBackward() noexcept : GradNode("SiluBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    if (!inputs_[0].requires_grad()) return Status::ok();
    const Index n = grad_output.numel();
    Tensor gx;
    L3_ASSIGN_OR_RETURN(gx,
                        Tensor::empty(grad_output.shape(), DType::kFloat32));

    const f32* x = inputs_[0].data_ptr<f32>();
    const f32* dy = grad_output.data_ptr<f32>();
    f32* dx = gx.mutable_data_ptr<f32>();

    // d/dx [x * sig(x)] = sig(x) * (1 + x * (1 - sig(x)))
    for (Index i = 0; i < n; ++i) {
      const f32 sig = 1.0f / (1.0f + std::exp(-x[i]));
      dx[i] = dy[i] * sig * (1.0f + x[i] * (1.0f - sig));
    }
    return inputs_[0].accumulate_grad(gx);
  }
};

class ReluBackward final : public GradNode {
 public:
  ReluBackward() noexcept : GradNode("ReluBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    if (!inputs_[0].requires_grad()) return Status::ok();
    const Index n = grad_output.numel();
    Tensor gx;
    L3_ASSIGN_OR_RETURN(gx,
                        Tensor::empty(grad_output.shape(), DType::kFloat32));

    const f32* x = inputs_[0].data_ptr<f32>();
    const f32* dy = grad_output.data_ptr<f32>();
    f32* dx = gx.mutable_data_ptr<f32>();
    for (Index i = 0; i < n; ++i) {
      dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
    }
    return inputs_[0].accumulate_grad(gx);
  }
};

class RMSNormBackward final : public GradNode {
 public:
  explicit RMSNormBackward(f32 eps) noexcept
      : GradNode("RMSNormBackward"), eps_(eps) {}

  Status backward(const Tensor& grad_output) noexcept override {
    const Tensor& x = inputs_[0];
    const Tensor& w = inputs_[1];
    const Index D = w.numel();
    const Index M = x.numel() / D;

    const f32* x_ptr = x.data_ptr<f32>();
    const f32* w_ptr = w.data_ptr<f32>();
    const f32* dy_ptr = grad_output.data_ptr<f32>();

    Tensor dx;
    Tensor dw;
    if (x.requires_grad()) {
      L3_ASSIGN_OR_RETURN(dx, Tensor::zeros(x.shape(), DType::kFloat32));
    }
    if (w.requires_grad()) {
      L3_ASSIGN_OR_RETURN(dw, Tensor::zeros(w.shape(), DType::kFloat32));
    }

    for (Index m = 0; m < M; ++m) {
      const f32* xm = x_ptr + m * D;
      const f32* dym = dy_ptr + m * D;
      const f32 sum_sq = kernels::dot_f32(xm, xm, D);
      const f32 rms2 = sum_sq / static_cast<f32>(D) + eps_;
      const f32 inv_rms = 1.0f / std::sqrt(rms2);

      // dot_dy_w_x = sum_j (dy_j * w_j * x_j)
      f32 dot_dy_w_x = 0.0f;
      for (Index j = 0; j < D; ++j) {
        dot_dy_w_x += dym[j] * w_ptr[j] * xm[j];
      }

      if (w.requires_grad()) {
        f32* dw_ptr = dw.mutable_data_ptr<f32>();
        for (Index j = 0; j < D; ++j) {
          dw_ptr[j] += dym[j] * (xm[j] * inv_rms);
        }
      }
      if (x.requires_grad()) {
        f32* dxm = dx.mutable_data_ptr<f32>() + m * D;
        const f32 coeff =
            (inv_rms * inv_rms * inv_rms) / static_cast<f32>(D) * dot_dy_w_x;
        for (Index j = 0; j < D; ++j) {
          dxm[j] = inv_rms * (dym[j] * w_ptr[j]) - xm[j] * coeff;
        }
      }
    }

    if (x.requires_grad()) L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(dx));
    if (w.requires_grad()) L3_RETURN_IF_ERROR(inputs_[1].accumulate_grad(dw));
    return Status::ok();
  }

 private:
  f32 eps_;
};

class MatMulBackward final : public GradNode {
 public:
  MatMulBackward() noexcept : GradNode("MatMulBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    // C(M, N) = A(M, K) @ B(K, N)
    // dA(M, K) = dC(M, N) @ B^T(N, K)
    // dB(K, N) = A^T(K, M) @ dC(M, N)
    const Tensor& a = inputs_[0];
    const Tensor& b = inputs_[1];

    if (a.requires_grad()) {
      Tensor bt;
      Tensor bt_contig;
      Tensor da;
      L3_ASSIGN_OR_RETURN(bt, b.transpose(0, 1));
      L3_ASSIGN_OR_RETURN(bt_contig, bt.contiguous());
      L3_ASSIGN_OR_RETURN(da, Tensor::zeros(a.shape(), DType::kFloat32));
      L3_RETURN_IF_ERROR(kernels::matmul_f32(grad_output, bt_contig, &da));
      L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(da));
    }

    if (b.requires_grad()) {
      Tensor at;
      Tensor at_contig;
      Tensor db;
      L3_ASSIGN_OR_RETURN(at, a.transpose(0, 1));
      L3_ASSIGN_OR_RETURN(at_contig, at.contiguous());
      L3_ASSIGN_OR_RETURN(db, Tensor::zeros(b.shape(), DType::kFloat32));
      L3_RETURN_IF_ERROR(kernels::matmul_f32(at_contig, grad_output, &db));
      L3_RETURN_IF_ERROR(inputs_[1].accumulate_grad(db));
    }
    return Status::ok();
  }
};

// Backward for `Y(M, N) = X(M, K) @ W(N, K)^T` (used by both `linear` and
// `bitnet_linear_ste` Straight-Through Estimator!):
//   dX(M, K) = dY(M, N) @ W_eff(N, K)
//   dW(N, K) = dY(M, N)^T @ X(M, K)
class LinearBackward final : public GradNode {
 public:
  explicit LinearBackward(Tensor&& w_effective) noexcept
      : GradNode("LinearBackward"), w_effective_(std::move(w_effective)) {}

  Status backward(const Tensor& grad_output) noexcept override {
    const Tensor& x = inputs_[0];  // (M, K)
    const Tensor& w = inputs_[1];  // (N, K)

    if (x.requires_grad()) {
      Tensor dx;
      L3_ASSIGN_OR_RETURN(dx, Tensor::zeros(x.shape(), DType::kFloat32));
      L3_RETURN_IF_ERROR(kernels::matmul_f32(grad_output, w_effective_, &dx));
      L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(dx));
    }

    if (w.requires_grad()) {
      Tensor dyt;
      Tensor dyt_contig;
      Tensor dw;
      L3_ASSIGN_OR_RETURN(dyt, grad_output.transpose(0, 1));
      L3_ASSIGN_OR_RETURN(dyt_contig, dyt.contiguous());
      L3_ASSIGN_OR_RETURN(dw, Tensor::zeros(w.shape(), DType::kFloat32));
      L3_RETURN_IF_ERROR(kernels::matmul_f32(dyt_contig, x, &dw));
      L3_RETURN_IF_ERROR(inputs_[1].accumulate_grad(dw));
    }
    return Status::ok();
  }

 private:
  Tensor w_effective_;  // (N, K) FP32 or dequantized ternary weights for dX
};

class SumBackward final : public GradNode {
 public:
  explicit SumBackward(f32 scale) noexcept
      : GradNode("SumBackward"), scale_(scale) {}

  Status backward(const Tensor& grad_output) noexcept override {
    if (!inputs_[0].requires_grad()) return Status::ok();
    const f32 g0 = grad_output.data_ptr<f32>()[0] * scale_;
    Tensor dx;
    L3_ASSIGN_OR_RETURN(
        dx, Tensor::full(inputs_[0].shape(), g0, DType::kFloat32));
    return inputs_[0].accumulate_grad(dx);
  }

 private:
  f32 scale_;
};

class MSELossBackward final : public GradNode {
 public:
  MSELossBackward() noexcept : GradNode("MSELossBackward") {}

  Status backward(const Tensor& grad_output) noexcept override {
    const Tensor& pred = inputs_[0];
    const Tensor& target = inputs_[1];
    const Index n = pred.numel();
    const f32 scale =
        (2.0f / static_cast<f32>(n)) * grad_output.data_ptr<f32>()[0];

    if (pred.requires_grad()) {
      Tensor dpred;
      L3_ASSIGN_OR_RETURN(dpred, Tensor::empty(pred.shape(), DType::kFloat32));
      const f32* p = pred.data_ptr<f32>();
      const f32* t = target.data_ptr<f32>();
      f32* dp = dpred.mutable_data_ptr<f32>();
      for (Index i = 0; i < n; ++i) {
        dp[i] = scale * (p[i] - t[i]);
      }
      L3_RETURN_IF_ERROR(inputs_[0].accumulate_grad(dpred));
    }
    return Status::ok();
  }
};

class CrossEntropyBackward final : public GradNode {
 public:
  CrossEntropyBackward(Tensor&& probs, SmallVec<Index, 64>&& targets) noexcept
      : GradNode("CrossEntropyBackward"),
        probs_(std::move(probs)),
        targets_(std::move(targets)) {}

  Status backward(const Tensor& grad_output) noexcept override {
    if (!inputs_[0].requires_grad()) return Status::ok();
    const Index N = probs_.dim(0);
    const Index V = probs_.dim(1);
    const f32 upstream =
        grad_output.data_ptr<f32>()[0] / static_cast<f32>(N);

    Tensor dlogits;
    L3_ASSIGN_OR_RETURN(dlogits, probs_.clone());
    f32* dl = dlogits.mutable_data_ptr<f32>();

    for (Index i = 0; i < N; ++i) {
      dl[i * V + targets_[i]] -= 1.0f;
      for (Index v = 0; v < V; ++v) {
        dl[i * V + v] *= upstream;
      }
    }
    return inputs_[0].accumulate_grad(dlogits);
  }

 private:
  Tensor probs_;
  SmallVec<Index, 64> targets_;
};

}  // namespace

// -----------------------------------------------------------------------------
// Forward + Graph Construction Implementations
// -----------------------------------------------------------------------------

Result<Tensor> add(const Tensor& a, const Tensor& b) noexcept {
  if (!a.is_valid() || !b.is_valid() || a.shape() != b.shape() ||
      a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::add: expects matching f32 tensors");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(a.shape(), DType::kFloat32));
  kernels::add_f32(a.data_ptr<f32>(), b.data_ptr<f32>(),
                   out.mutable_data_ptr<f32>(), a.numel());

  if (should_record(a, b)) {
    Ptr<AddBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<AddBackward>());
    L3_RETURN_IF_ERROR(node->add_input(a));
    L3_RETURN_IF_ERROR(node->add_input(b));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> sub(const Tensor& a, const Tensor& b) noexcept {
  if (!a.is_valid() || !b.is_valid() || a.shape() != b.shape() ||
      a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::sub: expects matching f32 tensors");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(a.shape(), DType::kFloat32));
  const f32* pa = a.data_ptr<f32>();
  const f32* pb = b.data_ptr<f32>();
  f32* po = out.mutable_data_ptr<f32>();
  const Index n = a.numel();
  for (Index i = 0; i < n; ++i) po[i] = pa[i] - pb[i];

  if (should_record(a, b)) {
    Ptr<SubBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<SubBackward>());
    L3_RETURN_IF_ERROR(node->add_input(a));
    L3_RETURN_IF_ERROR(node->add_input(b));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> mul(const Tensor& a, const Tensor& b) noexcept {
  if (!a.is_valid() || !b.is_valid() || a.shape() != b.shape() ||
      a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::mul: expects matching f32 tensors");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(a.shape(), DType::kFloat32));
  kernels::mul_f32(a.data_ptr<f32>(), b.data_ptr<f32>(),
                   out.mutable_data_ptr<f32>(), a.numel());

  if (should_record(a, b)) {
    Ptr<MulBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<MulBackward>());
    L3_RETURN_IF_ERROR(node->add_input(a));
    L3_RETURN_IF_ERROR(node->add_input(b));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> silu(const Tensor& x) noexcept {
  if (!x.is_valid() || x.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::silu: expects valid f32 tensor");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(x.shape(), DType::kFloat32));
  kernels::silu_f32(x.data_ptr<f32>(), out.mutable_data_ptr<f32>(), x.numel());

  if (should_record(x)) {
    Ptr<SiluBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<SiluBackward>());
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> relu(const Tensor& x) noexcept {
  if (!x.is_valid() || x.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::relu: expects valid f32 tensor");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(x.shape(), DType::kFloat32));
  const f32* px = x.data_ptr<f32>();
  f32* po = out.mutable_data_ptr<f32>();
  const Index n = x.numel();
  for (Index i = 0; i < n; ++i) po[i] = px[i] > 0.0f ? px[i] : 0.0f;

  if (should_record(x)) {
    Ptr<ReluBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<ReluBackward>());
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> rmsnorm(const Tensor& x, const Tensor& weight,
                       f32 eps) noexcept {
  if (!x.is_valid() || !weight.is_valid() || x.dtype() != DType::kFloat32 ||
      weight.dtype() != DType::kFloat32 || x.rank() < 1 ||
      weight.rank() != 1 || x.dim(x.rank() - 1) != weight.dim(0)) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::rmsnorm: invalid x or weight shape/dtype");
  }

  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(x.shape(), DType::kFloat32));
  const Index D = weight.dim(0);
  const Index M = x.numel() / D;
  for (Index m = 0; m < M; ++m) {
    kernels::rmsnorm_f32(x.data_ptr<f32>() + m * D, weight.data_ptr<f32>(),
                         out.mutable_data_ptr<f32>() + m * D, D, eps);
  }

  if (should_record(x, weight)) {
    Ptr<RMSNormBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<RMSNormBackward>(eps));
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(node->add_input(weight));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> matmul(const Tensor& a, const Tensor& b) noexcept {
  if (!a.is_valid() || !b.is_valid() || a.rank() != 2 || b.rank() != 2 ||
      a.dim(1) != b.dim(0)) {
    return Status::error(StatusCode::kShapeMismatch,
                         "autograd::matmul: incompatible 2D shapes");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out,
                      Tensor::zeros({a.dim(0), b.dim(1)}, DType::kFloat32));
  L3_RETURN_IF_ERROR(kernels::matmul_f32(a, b, &out));

  if (should_record(a, b)) {
    Ptr<MatMulBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<MatMulBackward>());
    L3_RETURN_IF_ERROR(node->add_input(a));
    L3_RETURN_IF_ERROR(node->add_input(b));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> linear(const Tensor& x, const Tensor& weight) noexcept {
  if (!x.is_valid() || !weight.is_valid() || x.rank() != 2 ||
      weight.rank() != 2 || x.dim(1) != weight.dim(1)) {
    return Status::error(StatusCode::kShapeMismatch,
                         "autograd::linear: expects x=(M, K) and weight=(N, K)");
  }

  const Index M = x.dim(0);
  const Index K = x.dim(1);
  const Index N = weight.dim(0);

  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty({M, N}, DType::kFloat32));

  const f32* xp = x.data_ptr<f32>();
  const f32* wp = weight.data_ptr<f32>();
  f32* yp = out.mutable_data_ptr<f32>();

  for (Index m = 0; m < M; ++m) {
    for (Index n = 0; n < N; ++n) {
      yp[m * N + n] = kernels::dot_f32(xp + m * K, wp + n * K, K);
    }
  }

  if (should_record(x, weight)) {
    Ptr<LinearBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<LinearBackward>(weight.detach()));
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(node->add_input(weight));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> bitnet_linear_ste(const Tensor& x,
                                 const Tensor& w_master_f32) noexcept {
  if (!x.is_valid() || !w_master_f32.is_valid() || x.rank() != 2 ||
      w_master_f32.rank() != 2 || x.dim(1) != w_master_f32.dim(1) ||
      x.dim(1) % 4 != 0) {
    return Status::error(
        StatusCode::kShapeMismatch,
        "autograd::bitnet_linear_ste: expects x=(M, K) and w_master=(N, K) with K % 4 == 0");
  }

  // Step 1: Quantize latent FP32 master weights to BitNet b1.58 ternary `t2`
  Tensor w_t2;
  L3_ASSIGN_OR_RETURN(w_t2, kernels::quantize_f32_to_t2(w_master_f32));

  // Step 2: Run BitNet ternary forward projection
  Tensor out;
  L3_ASSIGN_OR_RETURN(
      out, Tensor::empty({x.dim(0), w_master_f32.dim(0)}, DType::kFloat32));
  L3_RETURN_IF_ERROR(kernels::bitnet_linear_t2_f32(x, w_t2, &out));

  // Step 3: Record STE backward node if training
  if (should_record(x, w_master_f32)) {
    Tensor w_deq;
    L3_ASSIGN_OR_RETURN(w_deq,
                        Tensor::empty(w_master_f32.shape(), DType::kFloat32));
    kernels::dequantize_t2_to_f32_raw(w_t2.raw_data(),
                                      w_deq.mutable_data_ptr<f32>(),
                                      w_deq.numel(), w_t2.quant_scale());

    Ptr<LinearBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<LinearBackward>(std::move(w_deq)));
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(node->add_input(w_master_f32));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> sum(const Tensor& x) noexcept {
  if (!x.is_valid() || x.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::sum: expects valid f32 tensor");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(Shape::scalar(), DType::kFloat32));

  const f32* px = x.data_ptr<f32>();
  const Index n = x.numel();
  f64 acc = 0.0;
  for (Index i = 0; i < n; ++i) acc += static_cast<f64>(px[i]);
  out.mutable_data_ptr<f32>()[0] = static_cast<f32>(acc);

  if (should_record(x)) {
    Ptr<SumBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<SumBackward>(1.0f));
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> mean(const Tensor& x) noexcept {
  if (!x.is_valid() || x.dtype() != DType::kFloat32 || x.numel() == 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::mean: expects non-empty f32 tensor");
  }
  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(Shape::scalar(), DType::kFloat32));

  const f32* px = x.data_ptr<f32>();
  const Index n = x.numel();
  f64 acc = 0.0;
  for (Index i = 0; i < n; ++i) acc += static_cast<f64>(px[i]);
  const f32 inv_n = 1.0f / static_cast<f32>(n);
  out.mutable_data_ptr<f32>()[0] = static_cast<f32>(acc) * inv_n;

  if (should_record(x)) {
    Ptr<SumBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<SumBackward>(inv_n));
    L3_RETURN_IF_ERROR(node->add_input(x));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> mse_loss(const Tensor& pred, const Tensor& target) noexcept {
  if (!pred.is_valid() || !target.is_valid() || pred.shape() != target.shape() ||
      pred.dtype() != DType::kFloat32 || target.dtype() != DType::kFloat32 ||
      pred.numel() == 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "autograd::mse_loss: expects matching non-empty f32 tensors");
  }

  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(Shape::scalar(), DType::kFloat32));

  const f32* p = pred.data_ptr<f32>();
  const f32* t = target.data_ptr<f32>();
  const Index n = pred.numel();
  f64 sq_sum = 0.0;
  for (Index i = 0; i < n; ++i) {
    const f64 diff = static_cast<f64>(p[i]) - static_cast<f64>(t[i]);
    sq_sum += diff * diff;
  }
  out.mutable_data_ptr<f32>()[0] =
      static_cast<f32>(sq_sum / static_cast<f64>(n));

  if (should_record(pred)) {
    Ptr<MSELossBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<MSELossBackward>());
    L3_RETURN_IF_ERROR(node->add_input(pred));
    L3_RETURN_IF_ERROR(node->add_input(target));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

Result<Tensor> cross_entropy_loss(const Tensor& logits,
                                  Span<const Index> targets) noexcept {
  if (!logits.is_valid() || logits.dtype() != DType::kFloat32 ||
      logits.rank() != 2 || logits.dim(0) != targets.size() ||
      targets.empty()) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "autograd::cross_entropy_loss: expects 2D logits (N, V) and N targets");
  }

  const Index N = logits.dim(0);
  const Index V = logits.dim(1);

  Tensor probs;
  L3_ASSIGN_OR_RETURN(probs, Tensor::empty(logits.shape(), DType::kFloat32));

  SmallVec<Index, 64> saved_targets;
  L3_RETURN_IF_ERROR(saved_targets.assign(targets));

  f64 total_nll = 0.0;
  for (Index i = 0; i < N; ++i) {
    const Index tgt = targets[i];
    if (tgt < 0 || tgt >= V) {
      return Status::error(StatusCode::kOutOfRange,
                           "autograd::cross_entropy_loss: target index out of bounds");
    }
    const f32* row_logits = logits.data_ptr<f32>() + i * V;
    f32* row_probs = probs.mutable_data_ptr<f32>() + i * V;
    kernels::softmax_f32(row_logits, row_probs, V);

    const f32 p_tgt = row_probs[tgt] > 1e-12f ? row_probs[tgt] : 1e-12f;
    total_nll += static_cast<f64>(-std::log(p_tgt));
  }

  Tensor out;
  L3_ASSIGN_OR_RETURN(out, Tensor::empty(Shape::scalar(), DType::kFloat32));
  out.mutable_data_ptr<f32>()[0] =
      static_cast<f32>(total_nll / static_cast<f64>(N));

  if (should_record(logits)) {
    Ptr<CrossEntropyBackward> node;
    L3_ASSIGN_OR_RETURN(node, make_ptr<CrossEntropyBackward>(
                                  std::move(probs), std::move(saved_targets)));
    L3_RETURN_IF_ERROR(node->add_input(logits));
    L3_RETURN_IF_ERROR(out.set_grad_node(std::move(node)));
  }
  return out;
}

}  // namespace autograd
}  // namespace l3
