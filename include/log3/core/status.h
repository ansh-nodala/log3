
// log3/core/status.h
//
// Status and Result primitives used throughout the log3 library.
//
// Responsibilities:
//   * Represent successful and failed operations.
//   * Provide structured error codes and diagnostic messages.
//   * Support move-only error ownership.
//   * Provide value-or-error abstraction through Result<T>.
//   * Provide explicit error propagation macros.
//
// Compatibility:
//   * C++17
//   * No exceptions
//   * No RTTI

#pragma once

#include <new>
#include <type_traits>
#include <utility>

#include "log3/core/platform.h"
#include "log3/core/types.h"

namespace l3 {

// -----------------------------------------------------------------------------
// Status codes
// -----------------------------------------------------------------------------

enum class StatusCode : u8 {
  kOk = 0,
  kInvalidArgument,
  kOutOfRange,
  kShapeMismatch,
  kTypeMismatch,
  kNotFound,
  kAlreadyExists,
  kFailedPrecondition,
  kOutOfMemory,
  kUnimplemented,
  kUnsupported,
  kIoError,
  kInternal,
};

constexpr const char* status_code_name(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::kOk:
      return "Ok";
    case StatusCode::kInvalidArgument:
      return "InvalidArgument";
    case StatusCode::kOutOfRange:
      return "OutOfRange";
    case StatusCode::kShapeMismatch:
      return "ShapeMismatch";
    case StatusCode::kTypeMismatch:
      return "TypeMismatch";
    case StatusCode::kNotFound:
      return "NotFound";
    case StatusCode::kAlreadyExists:
      return "AlreadyExists";
    case StatusCode::kFailedPrecondition:
      return "FailedPrecondition";
    case StatusCode::kOutOfMemory:
      return "OutOfMemory";
    case StatusCode::kUnimplemented:
      return "Unimplemented";
    case StatusCode::kUnsupported:
      return "Unsupported";
    case StatusCode::kIoError:
      return "IoError";
    case StatusCode::kInternal:
      return "Internal";
  }

  return "Unknown";
}

// -----------------------------------------------------------------------------
// Status
// -----------------------------------------------------------------------------

// A successful Status owns no allocation.
// An error Status owns one internal State.
//
// Status is move-only. Moving transfers ownership and leaves the source OK.

class L3_NODISCARD Status {
public:
  struct State;

  Status() noexcept : state_(nullptr) {}

  Status(Status&& other) noexcept
      : state_(other.state_) {
    other.state_ = nullptr;
  }

  Status& operator=(Status&& other) noexcept {
    if (this != &other) {
      reset();

      state_ = other.state_;
      other.state_ = nullptr;
    }

    return *this;
  }

  Status(const Status&) = delete;
  Status& operator=(const Status&) = delete;

  ~Status() noexcept {
    reset();
  }

  static Status ok() noexcept {
    return Status();
  }

  L3_API static Status error(
      StatusCode code,
      const char* message) noexcept;

  L3_API static Status errorf(
      StatusCode code,
      const char* fmt,
      ...) noexcept L3_PRINTF_FORMAT(2, 3);

  bool is_ok() const noexcept {
    return state_ == nullptr;
  }

  L3_API StatusCode code() const noexcept;

  L3_API const char* message() const noexcept;

  bool is(StatusCode expected) const noexcept {
    return code() == expected;
  }

  L3_API Status clone() const noexcept;

  // Explicitly acknowledge an intentionally ignored status.
  void ignore() const noexcept {}

private:
  void reset() noexcept {
    if (state_ != nullptr) {
      destroy(state_);
      state_ = nullptr;
    }
  }

  L3_API static void destroy(State* state) noexcept;

  State* state_;
};

// -----------------------------------------------------------------------------
// Result access failure
// -----------------------------------------------------------------------------

namespace detail {

[[noreturn]] L3_API void result_access_failed(
    const Status& status) noexcept;

}  // namespace detail

// -----------------------------------------------------------------------------
// Result<T>
// -----------------------------------------------------------------------------

// Result<T> represents either:
//
//   1. A successfully constructed T.
//   2. A non-OK Status.
//
// Invariant:
//
//   has_value_ == true  => status_ is OK and T is alive.
//   has_value_ == false => status_ is non-OK and T is not alive.
//
// Moving a value leaves a valid moved-from T in the source Result.
// Moving an error leaves a FailedPrecondition Status in the source.
//
// T requirements:
//   * Object type.
//   * Not an array, reference, const, or volatile type.
//   * Not Status.
//   * Nothrow move-constructible.
//   * Nothrow destructible.

template <typename T>
class L3_NODISCARD Result {
  static_assert(
      std::is_object<T>::value,
      "Result<T> requires an object type.");

  static_assert(
      !std::is_array<T>::value,
      "Result<T> does not support arrays.");

  static_assert(
    !std::is_const<T>::value &&
    !std::is_volatile<T>::value,
    "Result<T> does not support cv-qualified types.");

  static_assert(
      !std::is_same<T, Status>::value,
      "Result<Status> is meaningless.");

  static_assert(
      std::is_nothrow_move_constructible<T>::value,
      "Result<T> requires nothrow move construction.");

  static_assert(
      std::is_nothrow_destructible<T>::value,
      "Result<T> requires nothrow destruction.");

public:
  Result(T&& value) noexcept
      : status_(),
        has_value_(true) {
    ::new (static_cast<void*>(storage_)) T(std::move(value));
  }

  // Only allow copying when copying cannot throw.
  template <
      typename U = T,
      typename std::enable_if<
          std::is_same<U, T>::value &&
          std::is_nothrow_copy_constructible<U>::value,
          int>::type = 0>
  Result(const T& value) noexcept
      : status_(),
        has_value_(true) {
    ::new (static_cast<void*>(storage_)) T(value);
  }

  // Construct an error Result.
  // Passing OK is converted into an internal error.
  Result(Status&& status) noexcept
      : status_(std::move(status)),
        has_value_(false) {
    if (L3_UNLIKELY(status_.is_ok())) {
      status_ = Status::error(
          StatusCode::kInternal,
          "Result<T> constructed from OK Status");
    }
  }

  // Move-construct a Result.
  Result(Result&& other) noexcept
      : status_(std::move(other.status_)),
        has_value_(other.has_value_) {

    if (has_value_) {
      // Keep the source's moved-from T alive.
      ::new (static_cast<void*>(storage_)) T(
          std::move(*other.ptr()));
    } else {
      // The source must remain a valid error Result.
      other.status_ = Status::error(
          StatusCode::kFailedPrecondition,
          "Result was moved from");
    }
  }

  Result& operator=(Result&& other) noexcept {
    if (this != &other) {
      destroy_value();

      status_ = std::move(other.status_);
      has_value_ = other.has_value_;

      if (has_value_) {
        // Source remains engaged with a valid moved-from T.
        ::new (static_cast<void*>(storage_)) T(
            std::move(*other.ptr()));
      } else {
        other.status_ = Status::error(
            StatusCode::kFailedPrecondition,
            "Result was moved from");
      }
    }

    return *this;
  }

  Result(const Result&) = delete;
  Result& operator=(const Result&) = delete;

  ~Result() noexcept {
    destroy_value();
  }

  bool is_ok() const noexcept {
    return has_value_;
  }

  const Status& status() const noexcept {
    return status_;
  }

  // Transfer the error status out.
  //
  // The Result remains valid after this operation.
  // Its new state is FailedPrecondition.
  Status take_status() noexcept {
    if (has_value_) {
      return Status::ok();
    }

    Status result(std::move(status_));

    status_ = Status::error(
        StatusCode::kFailedPrecondition,
        "Result status was moved out");

    return result;
  }

  T& value() & {
    check_value();
    return *ptr();
  }

  const T& value() const& {
    check_value();
    return *ptr();
  }

  T&& value() && {
    check_value();
    return std::move(*ptr());
  }

  // Move the value out.
  // The Result retains its moved-from T object.
  T take_value() {
    check_value();
    return std::move(*ptr());
  }

  T& operator*() & {
    return value();
  }

  const T& operator*() const& {
    return value();
  }

  T* operator->() {
    check_value();
    return ptr();
  }

  const T* operator->() const {
    check_value();
    return ptr();
  }

private:
  T* ptr() noexcept {
    return std::launder(
        reinterpret_cast<T*>(storage_));
  }

  const T* ptr() const noexcept {
    return std::launder(
        reinterpret_cast<const T*>(storage_));
  }

  void check_value() const {
    if (L3_UNLIKELY(!has_value_)) {
      detail::result_access_failed(status_);
    }
  }

  void destroy_value() noexcept {
    if (has_value_) {
      ptr()->~T();
      has_value_ = false;
    }
  }

  Status status_;
  bool has_value_;

  alignas(alignof(T)) byte storage_[sizeof(T)];
};

}  // namespace l3

// -----------------------------------------------------------------------------
// Error propagation macros
// -----------------------------------------------------------------------------

// L3_RETURN_IF_ERROR:
//   Evaluates a Status expression once.
//   Returns immediately when it contains an error.
//
// L3_ASSIGN_OR_RETURN:
//   Evaluates a Result expression once.
//   Returns its error when no value exists.
//   Otherwise moves its value into lhs.
//
// L3_ASSIGN_OR_RETURN intentionally expands to multiple statements
// so declarations remain visible in the caller's scope.
//
// Always use braces when placing it inside an if/else body.

#define L3_RETURN_IF_ERROR_IMPL(status_var, expr) \
  do { \
    ::l3::Status status_var = (expr); \
    if (L3_UNLIKELY(!status_var.is_ok())) { \
      return status_var; \
    } \
  } while (0)

#define L3_RETURN_IF_ERROR(expr) \
  L3_RETURN_IF_ERROR_IMPL( \
      L3_UNIQUE_NAME(l3_status_), expr)

#define L3_ASSIGN_OR_RETURN_IMPL(result_var, lhs, expr) \
  auto result_var = (expr); \
  if (L3_UNLIKELY(!result_var.is_ok())) { \
    return result_var.take_status(); \
  } \
  lhs = result_var.take_value()

#define L3_ASSIGN_OR_RETURN(lhs, expr) \
  L3_ASSIGN_OR_RETURN_IMPL( \
      L3_UNIQUE_NAME(l3_result_), lhs, expr)
