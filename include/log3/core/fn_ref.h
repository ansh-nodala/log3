// log3/core/fn_ref.h
//
// FnRef<R(Args...)>: lightweight, non-owning, zero-allocation reference to any
// callable (lambda, functor, or function pointer).
//
// Why not std::function:
//   * std::function may heap-allocate when captures exceed its small-buffer
//     size, pulls in <functional> and RTTI/exception machinery, and is heavy to
//     copy.
//   * FnRef is strictly 2 pointers wide (16 bytes on 64-bit): one opaque void*
//     pointing to the caller's lambda/callable (or holding a raw function
//     pointer directly) and one trampoline function pointer.
//   * Zero heap allocation, zero exceptions, zero RTTI, trivially copyable.
//
// Lifetime rule (same as Span and StrView):
//   FnRef does NOT own the callable object it references. Pass FnRef as a
//   function parameter (e.g., `parallel_for`, graph traversal visitors, tensor
//   init callbacks) where the temporary lambda stays alive for the duration of
//   the call, or bind it to a named lvalue callable. Function pointers (both
//   `R(*)(Args...)` and `R(*)(Args...) noexcept`) are stored directly by value
//   inside `obj_`, so they never dangle even when passed as a temporary.
//
// Usage:
//   void run_loop(Index n, FnRef<void(Index)> body) noexcept {
//     for (Index i = 0; i < n; ++i) body(i);
//   }
//   run_loop(100, [&](Index i) { out[i] = in[i] * scale; });

#pragma once

#include <type_traits>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/types.h"

namespace l3 {

template <typename Signature>
class FnRef;

template <typename R, typename... Args>
class FnRef<R(Args...)> {
 public:
  using CallFn = R (*)(void* obj, Args... args);

  constexpr FnRef() noexcept : obj_(nullptr), call_fn_(nullptr) {}
  constexpr FnRef(std::nullptr_t) noexcept : obj_(nullptr), call_fn_(nullptr) {}

  // Construct from any callable object F (lambda or functor, excluding raw
  // function pointers/references which are handled below by value).
  template <
      typename F,
      typename CleanF = std::remove_cv_t<std::remove_reference_t<F>>,
      typename = std::enable_if_t<
          !std::is_same_v<CleanF, FnRef> &&
          !std::is_same_v<CleanF, std::nullptr_t> &&
          !std::is_function_v<CleanF> &&
          !(std::is_pointer_v<CleanF> &&
            std::is_function_v<std::remove_pointer_t<CleanF>>) &&
          std::is_invocable_r_v<R, std::remove_reference_t<F>&, Args...>>>
  constexpr FnRef(F&& callable) noexcept
      : obj_(const_cast<void*>(
            static_cast<const void*>(std::addressof(callable)))),
        call_fn_(&trampoline_object<std::remove_reference_t<F>>) {}

  // Construct from a plain function pointer (stores the function pointer
  // directly in `obj_`, so temporary function pointers never dangle).
  constexpr FnRef(R (*fn)(Args...)) noexcept
      : obj_(reinterpret_cast<void*>(fn)),
        call_fn_(fn != nullptr ? &trampoline_fn_ptr : nullptr) {}

  // Construct from a `noexcept` function pointer (in C++17 `R(*)(Args...) noexcept`
  // is a distinct type from `R(*)(Args...)`).
  constexpr FnRef(R (*fn)(Args...) noexcept) noexcept
      : obj_(reinterpret_cast<void*>(fn)),
        call_fn_(fn != nullptr ? &trampoline_noexcept_fn_ptr : nullptr) {}

  constexpr FnRef(const FnRef&) noexcept = default;
  constexpr FnRef& operator=(const FnRef&) noexcept = default;

  R operator()(Args... args) const {
    L3_DCHECK(call_fn_ != nullptr);
    return call_fn_(obj_, std::forward<Args>(args)...);
  }

  constexpr explicit operator bool() const noexcept {
    return call_fn_ != nullptr;
  }

  constexpr bool is_valid() const noexcept { return call_fn_ != nullptr; }

 private:
  template <typename Callable>
  static R trampoline_object(void* obj, Args... args) {
    return (*static_cast<Callable*>(obj))(std::forward<Args>(args)...);
  }

  static R trampoline_fn_ptr(void* obj, Args... args) {
    auto fn = reinterpret_cast<R (*)(Args...)>(obj);
    return fn(std::forward<Args>(args)...);
  }

  static R trampoline_noexcept_fn_ptr(void* obj, Args... args) {
    auto fn = reinterpret_cast<R (*)(Args...) noexcept>(obj);
    return fn(std::forward<Args>(args)...);
  }

  void* obj_;
  CallFn call_fn_;
};

static_assert(sizeof(FnRef<void()>) == 2 * sizeof(void*),
              "log3: FnRef must be exactly two pointers wide.");

}  // namespace l3
