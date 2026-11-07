// log3/autograd/engine.h
//
// Reverse-mode automatic differentiation engine, computational DAG nodes
// (`GradNode`), and thread-local gradient recording guards (`GradMode`,
// `NoGradGuard`).
//
// Why custom intrusive DAG nodes instead of std::shared_ptr / std::function:
//   * Every `GradNode` derives from `l3::RefCounted` (single allocation, 24B
//     intrusive header, zero weak_ptr control blocks).
//   * Input edges are stored inline inside `SmallVec<Tensor, 4>` so binary and
//     ternary ops (add, mul, matmul, linear, bitnet_linear_ste) never touch the
//     heap for their edge lists!
//   * Reverse-mode execution (`run_backward`) performs an iterative / bounded
//     DFS topological sort using `FlatMap<GradNode*, u8>` and `SmallVec`
//     (zero recursion stack-overflow risk, zero `std::unordered_set`).
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/ptr.h"
#include "log3/core/small_vector.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace autograd {

// -----------------------------------------------------------------------------
// Thread-local Gradient Recording Mode (`GradMode` & `NoGradGuard`)
// -----------------------------------------------------------------------------
class L3_API GradMode {
 public:
  static bool is_enabled() noexcept;
  static void set_enabled(bool enabled) noexcept;
};

// RAII scope guard that disables autograd graph construction for the duration
// of the scope (equivalent to `torch.no_grad()`):
//   {
//     l3::autograd::NoGradGuard no_grad;
//     ... inference or optimizer weight updates ...
//   }
class L3_API NoGradGuard {
 public:
  NoGradGuard() noexcept : prev_mode_(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
  }

  ~NoGradGuard() noexcept { GradMode::set_enabled(prev_mode_); }

  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool prev_mode_;
};

// -----------------------------------------------------------------------------
// GradNode: Base class for backward operations in the computational DAG
// -----------------------------------------------------------------------------
class L3_API GradNode : public RefCounted {
 public:
  explicit GradNode(const char* op_name = "GradNode") noexcept
      : op_name_(op_name) {}

  virtual ~GradNode() noexcept = default;

  const char* op_name() const noexcept { return op_name_; }

  // Connect an input Tensor to this operation so topological sort can traverse
  // upstream and `backward()` can accumulate into `inputs_[i]`.
  Status add_input(const Tensor& input) noexcept {
    return inputs_.push_back(input.view());
  }

  Index num_inputs() const noexcept { return inputs_.size(); }
  Tensor& input(Index i) noexcept { return inputs_[i]; }
  const Tensor& input(Index i) const noexcept { return inputs_[i]; }
  Span<Tensor> inputs() noexcept { return inputs_.span(); }
  Span<const Tensor> inputs() const noexcept { return inputs_.span(); }

  // Compute and accumulate gradients into `inputs_[0 .. num_inputs()-1]` given
  // the upstream gradient `grad_output` with respect to this node's output.
  virtual Status backward(const Tensor& grad_output) noexcept = 0;

 protected:
  const char* op_name_;
  SmallVec<Tensor, 4> inputs_;
};

// Execute reverse-mode automatic differentiation starting from `root`.
// If `grad_output` is nullptr, initializes root gradient to `1.0f` (for scalar
// losses, or all-ones for non-scalar roots).
L3_API Status run_backward(const Tensor& root,
                           const Tensor* grad_output = nullptr) noexcept;

}  // namespace autograd
}  // namespace l3
