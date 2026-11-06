// log3/core/scope_guard.h
//
// Zero-overhead RAII scope exit guard (`ScopeGuard` and `L3_DEFER`).
//
// Why `L3_DEFER` is essential in a Status-based codebase:
//   * Because log3 disables exceptions and propagates errors explicitly via
//     `L3_RETURN_IF_ERROR(...)`, functions that acquire temporary buffers, OS
//     handles, or locks may have 5-10 early return points.
//   * `L3_DEFER { cleanup(); };` guarantees that `cleanup()` executes on every
//     return path unless explicitly `.dismiss()`-ed on the success path.
//   * Zero heap allocation, inlined directly into the caller's stack frame.
//
// Usage:
//   void* scratch = alloc_aligned(4096);
//   L3_DEFER { free_aligned(scratch); };
//   L3_RETURN_IF_ERROR(step_one(scratch));
//   L3_RETURN_IF_ERROR(step_two(scratch));

#pragma once

#include <type_traits>
#include <utility>

#include "log3/core/platform.h"

namespace l3 {

template <typename F>
class ScopeGuard {
 public:
  explicit constexpr ScopeGuard(F&& fn) noexcept
      : fn_(std::forward<F>(fn)), active_(true) {}

  ScopeGuard(ScopeGuard&& other) noexcept
      : fn_(std::move(other.fn_)), active_(other.active_) {
    other.active_ = false;
  }

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;

  ~ScopeGuard() noexcept {
    if (active_) {
      fn_();
    }
  }

  // Cancel the guard so the cleanup action is NOT run when leaving scope.
  constexpr void dismiss() noexcept { active_ = false; }

 private:
  F fn_;
  bool active_;
};

namespace detail {

struct DeferHelper {
  template <typename F>
  constexpr ScopeGuard<std::decay_t<F>> operator+(F&& fn) const noexcept {
    return ScopeGuard<std::decay_t<F>>(std::forward<F>(fn));
  }
};

}  // namespace detail

}  // namespace l3

// Execute a block unconditionally when the enclosing scope exits:
//   L3_DEFER { close_handle(h); };
#define L3_DEFER                                                        \
  auto L3_UNIQUE_NAME(l3_defer_guard_) = ::l3::detail::DeferHelper{} + \
                                         [&]() noexcept
