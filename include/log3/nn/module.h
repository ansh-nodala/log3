// log3/nn/module.h
#pragma once

#include "log3/core/assert.h"
#include "log3/core/fn_ref.h"
#include "log3/core/platform.h"
#include "log3/core/small_vector.h"
#include "log3/core/status.h"
#include "log3/core/str_view.h"
#include "log3/core/types.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace nn {

struct NamedTensor {
  const char* name;
  Tensor tensor;
  bool is_param;
};

class L3_API Module {
 public:
  Module() noexcept : training_(false) {}
  virtual ~Module() noexcept = default;

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  Module(Module&&) noexcept = default;
  Module& operator=(Module&&) noexcept = default;

  bool is_training() const noexcept { return training_; }
  virtual void train(bool mode = true) noexcept { training_ = mode; }
  void eval() noexcept { train(false); }

  void register_parameter(const char* name, const Tensor& tensor) noexcept {
    L3_DCHECK(name != nullptr);
    named_tensors_.push_back(NamedTensor{name, tensor.view(), true}).ignore();
  }

  void register_buffer(const char* name, const Tensor& tensor) noexcept {
    L3_DCHECK(name != nullptr);
    named_tensors_.push_back(NamedTensor{name, tensor.view(), false}).ignore();
  }

  void for_each_parameter(
      FnRef<void(StrView name, Tensor& param)> visitor) noexcept {
    for (Index i = 0; i < named_tensors_.size(); ++i) {
      if (named_tensors_[i].is_param && named_tensors_[i].tensor.is_valid()) {
        visitor(StrView(named_tensors_[i].name), named_tensors_[i].tensor);
      }
    }
  }

  void for_each_parameter(
      FnRef<void(StrView name, const Tensor& param)> visitor) const noexcept {
    for (Index i = 0; i < named_tensors_.size(); ++i) {
      if (named_tensors_[i].is_param && named_tensors_[i].tensor.is_valid()) {
        visitor(StrView(named_tensors_[i].name), named_tensors_[i].tensor);
      }
    }
  }

  Index num_parameters() const noexcept {
    Index total = 0;
    for (Index i = 0; i < named_tensors_.size(); ++i) {
      if (named_tensors_[i].is_param && named_tensors_[i].tensor.is_valid()) {
        total += named_tensors_[i].tensor.numel();
      }
    }
    return total;
  }

 protected:
  SmallVec<NamedTensor, 16> named_tensors_;
  bool training_;
};

}  // namespace nn
}  // namespace l3
