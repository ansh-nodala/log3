// src/autograd/engine.cpp
//
// Implementation of GradMode, topological sort, and reverse-mode backward
// execution.

#include "log3/autograd/engine.h"

#include "log3/core/flat_map.h"
#include "log3/core/small_vector.h"

namespace l3 {
namespace autograd {

namespace {

// Per-thread gradient recording flag (true by default).
thread_local bool g_grad_mode_enabled = true;

// Build post-order topological sort of unique `Tensor` autograd nodes reachable
// from `root` so parents always execute before their children in reverse mode.
Status build_topo_order(const Tensor& current,
                        FlatMap<GradNode*, u8>* visited,
                        SmallVec<Tensor, 64>* order) noexcept {
  GradNode* node = current.grad_node();
  if (node == nullptr) {
    return Status::ok();
  }
  if (visited->contains(node)) {
    return Status::ok();
  }
  L3_RETURN_IF_ERROR(visited->put(node, 1u));

  for (Index i = 0; i < node->num_inputs(); ++i) {
    const Tensor& inp = node->input(i);
    if (inp.grad_node() != nullptr) {
      L3_RETURN_IF_ERROR(build_topo_order(inp, visited, order));
    }
  }

  return order->push_back(current.view());
}

}  // namespace

bool GradMode::is_enabled() noexcept { return g_grad_mode_enabled; }

void GradMode::set_enabled(bool enabled) noexcept {
  g_grad_mode_enabled = enabled;
}

Status run_backward(const Tensor& root, const Tensor* grad_output) noexcept {
  if (!root.is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "run_backward: root Tensor is invalid");
  }
  if (!root.requires_grad()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "run_backward: root Tensor does not have requires_grad=true");
  }

  // Disable gradient recording while computing backward passes so backward ops
  // themselves do not build new graph nodes.
  NoGradGuard no_grad;

  // Seed the root tensor's gradient.
  Tensor seed_grad;
  if (grad_output != nullptr) {
    if (!grad_output->is_valid() || grad_output->shape() != root.shape() ||
        grad_output->dtype() != DType::kFloat32) {
      return Status::error(
          StatusCode::kInvalidArgument,
          "run_backward: grad_output must be a valid f32 Tensor matching root shape");
    }
    seed_grad = grad_output->view();
  } else {
    L3_ASSIGN_OR_RETURN(seed_grad,
                        Tensor::full(root.shape(), 1.0f, DType::kFloat32));
  }

  Tensor mutable_root = root.view();
  L3_RETURN_IF_ERROR(mutable_root.accumulate_grad(seed_grad));

  if (root.grad_node() == nullptr) {
    // Root is a leaf tensor; nothing further to backpropagate.
    return Status::ok();
  }

  FlatMap<GradNode*, u8> visited;
  SmallVec<Tensor, 64> topo_order;
  L3_RETURN_IF_ERROR(build_topo_order(root, &visited, &topo_order));

  // Traverse in reverse topological order (from root down to leaves).
  for (Index i = topo_order.size() - 1; i >= 0; --i) {
    Tensor& t = topo_order[i];
    GradNode* node = t.grad_node();
    if (node == nullptr || !t.has_grad()) continue;

    L3_RETURN_IF_ERROR(node->backward(*t.grad()));
  }

  return Status::ok();
}

}  // namespace autograd
}  // namespace l3
