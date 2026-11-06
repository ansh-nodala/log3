
// src/core/status.cpp
//
// Out-of-line implementation of Status and Result failure handling.
//
// Design principles:
//   * Error metadata is independent of the tensor allocator.
//   * Error reporting does not modify memory-tracking statistics.
//   * Allocation failures use an immutable static OOM fallback.
//   * Messages are bounded to prevent excessive diagnostic allocations.
//   * Short messages use inline storage.
//   * Long messages use separately allocated storage.
//
// Compatibility:
//   * C++17
//   * No exceptions
//   * No RTTI

#include "log3/core/status.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "log3/core/log.h"

namespace l3 {

// -----------------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------------

struct Status::State {
  StatusCode code;
  u32 length;

  const char* message;
  char inline_message[16];
};

namespace {

constexpr usize kMaxStatusMessageLength = 1023;
constexpr usize kInlineMessageCapacity = 16;

constexpr char kOomMessage[] = "out of memory";

// Immutable fallback. Its message points to static storage.
const Status::State kOomState = {
    StatusCode::kOutOfMemory,
    static_cast<u32>(sizeof(kOomMessage) - 1),
    kOomMessage,
    {}
};

// -----------------------------------------------------------------------------
// Bounded string length
// -----------------------------------------------------------------------------

// Reads at most max_length characters.
//
// The caller must still provide a valid null-terminated C string.
// If its length exceeds max_length, the result is truncated.

usize bounded_string_length(
    const char* message,
    usize max_length) noexcept {

  usize length = 0;

  while (length < max_length && message[length] != '\0') {
    ++length;
  }

  return length;
}

// -----------------------------------------------------------------------------
// State allocation
// -----------------------------------------------------------------------------

Status::State* allocate_state(
    StatusCode code,
    const char* message,
    usize length) noexcept {

  if (length > kMaxStatusMessageLength) {
    length = kMaxStatusMessageLength;
  }

  // Allocate the State independently of tensor memory tracking.
  void* memory = std::malloc(sizeof(Status::State));

  if (L3_UNLIKELY(memory == nullptr)) {
    return const_cast<Status::State*>(&kOomState);
  }

  auto* state = ::new (memory) Status::State{};

  state->code = code;
  state->length = static_cast<u32>(length);

  // Inline storage includes the terminating null byte.
  if (length < kInlineMessageCapacity) {
    if (length != 0) {
      std::memcpy(
          state->inline_message,
          message,
          length);
    }

    state->inline_message[length] = '\0';
    state->message = state->inline_message;

    return state;
  }

  // Long messages use a separate allocation.
  // length is bounded, so length + 1 cannot overflow.
  auto* heap_message = static_cast<char*>(
      std::malloc(length + 1));

  if (L3_UNLIKELY(heap_message == nullptr)) {
    state->~State();
    std::free(state);

    return const_cast<Status::State*>(&kOomState);
  }

  std::memcpy(heap_message, message, length);
  heap_message[length] = '\0';

  state->message = heap_message;

  return state;
}

}  // namespace

// -----------------------------------------------------------------------------
// Destruction
// -----------------------------------------------------------------------------

void Status::destroy(State* state) noexcept {
  if (state == nullptr || state == &kOomState) {
    return;
  }

  // Free separately allocated messages only.
  if (state->message != state->inline_message) {
    std::free(const_cast<char*>(state->message));
  }

  state->~State();
  std::free(state);
}

// -----------------------------------------------------------------------------
// Status::error()
// -----------------------------------------------------------------------------

Status Status::error(
    StatusCode code,
    const char* message) noexcept {

  if (L3_UNLIKELY(code == StatusCode::kOk)) {
    return Status::ok();
  }

  const char* safe_message =
      message != nullptr ? message : "";

  const usize length = bounded_string_length(
      safe_message,
      kMaxStatusMessageLength);

  Status result;

  result.state_ = allocate_state(
      code,
      safe_message,
      length);

  return result;
}

// -----------------------------------------------------------------------------
// Status::errorf()
// -----------------------------------------------------------------------------

Status Status::errorf(
    StatusCode code,
    const char* fmt,
    ...) noexcept {

  if (L3_UNLIKELY(code == StatusCode::kOk)) {
    return Status::ok();
  }

  if (L3_UNLIKELY(fmt == nullptr)) {
    return Status::error(code, "");
  }

  char buffer[kMaxStatusMessageLength + 1];

  std::va_list args;
  va_start(args, fmt);

  const int written = std::vsnprintf(
      buffer,
      sizeof(buffer),
      fmt,
      args);

  va_end(args);

  if (L3_UNLIKELY(written < 0)) {
    return Status::error(
        code,
        "status message formatting failed");
  }

  const usize length =
      static_cast<usize>(written) > kMaxStatusMessageLength
          ? kMaxStatusMessageLength
          : static_cast<usize>(written);

  Status result;

  result.state_ = allocate_state(
      code,
      buffer,
      length);

  return result;
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

StatusCode Status::code() const noexcept {
  return state_ == nullptr
      ? StatusCode::kOk
      : state_->code;
}

const char* Status::message() const noexcept {
  return state_ == nullptr
      ? ""
      : state_->message;
}

// -----------------------------------------------------------------------------
// Status::clone()
// -----------------------------------------------------------------------------

Status Status::clone() const noexcept {
  if (state_ == nullptr) {
    return Status::ok();
  }

  if (state_ == &kOomState) {
    Status result;
    result.state_ = const_cast<State*>(&kOomState);
    return result;
  }

  Status result;

  result.state_ = allocate_state(
      state_->code,
      state_->message,
      state_->length);

  return result;
}

// -----------------------------------------------------------------------------
// Result<T> access failure
// -----------------------------------------------------------------------------

namespace detail {

[[noreturn]] void result_access_failed(
    const Status& status) noexcept {

  L3_LOG_FATAL(
      "Result<T>::value() called on error Status [%s]: %s",
      status_code_name(status.code()),
      status.message());

  std::abort();
}

}  // namespace detail

}  // namespace l3
