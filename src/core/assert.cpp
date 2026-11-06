// src/core/assert.cpp
//
// Cold, out-of-line failure handlers for L3_CHECK*, L3_DCHECK*, and L3_PANIC.
//
// Design notes:
//   * Each handler forwards the caller's __FILE__, __LINE__, and __func__
//     directly to log_message(LogLevel::kFatal, ...) so the log line points to
//     the failed check rather than to assert.cpp.
//   * CheckValue formatting uses a small stack buffer and handles bools,
//     signed/unsigned 64-bit integers, floats, and pointers without dynamic
//     allocation.

#include "log3/core/assert.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>

namespace l3 {
namespace detail {

namespace {

void format_check_value(CheckValue v, char* buf, usize buf_size) noexcept {
  switch (v.kind) {
    case CheckValue::Kind::kBool:
      std::snprintf(buf, buf_size, "%s", v.u != 0 ? "true" : "false");
      return;
    case CheckValue::Kind::kSigned:
      std::snprintf(buf, buf_size, "%" PRId64, v.i);
      return;
    case CheckValue::Kind::kUnsigned:
      std::snprintf(buf, buf_size, "%" PRIu64, v.u);
      return;
    case CheckValue::Kind::kFloat:
      std::snprintf(buf, buf_size, "%.9g", v.f);
      return;
    case CheckValue::Kind::kPointer:
      if (v.p == nullptr) {
        std::snprintf(buf, buf_size, "nullptr");
      } else {
        std::snprintf(buf, buf_size, "%p", v.p);
      }
      return;
  }
  std::snprintf(buf, buf_size, "?");
}

}  // namespace

[[noreturn]] L3_COLD void check_failed(const char* file, int line,
                                       const char* func,
                                       const char* expr) noexcept {
  log_message(LogLevel::kFatal, file, line, func,
              "Check failed: %s (in %s)", expr, func != nullptr ? func : "?");
  fatal_abort();
}

[[noreturn]] L3_COLD void check_failed_msg(const char* file, int line,
                                           const char* func, const char* expr,
                                           const char* fmt, ...) noexcept {
  char user_msg[512];
  user_msg[0] = '\0';
  if (fmt != nullptr) {
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(user_msg, sizeof(user_msg), fmt, args);
    va_end(args);
  }
  log_message(LogLevel::kFatal, file, line, func,
              "Check failed: %s — %s (in %s)", expr, user_msg,
              func != nullptr ? func : "?");
  fatal_abort();
}

[[noreturn]] L3_COLD void check_op_failed(const char* file, int line,
                                          const char* func, const char* expr,
                                          CheckValue lhs,
                                          CheckValue rhs) noexcept {
  char lhs_buf[64];
  char rhs_buf[64];
  format_check_value(lhs, lhs_buf, sizeof(lhs_buf));
  format_check_value(rhs, rhs_buf, sizeof(rhs_buf));

  log_message(LogLevel::kFatal, file, line, func,
              "Check failed: %s (%s vs. %s) (in %s)", expr, lhs_buf, rhs_buf,
              func != nullptr ? func : "?");
  fatal_abort();
}

[[noreturn]] L3_COLD void check_status_failed(const char* file, int line,
                                              const char* func,
                                              const char* expr,
                                              const Status& status) noexcept {
  log_message(LogLevel::kFatal, file, line, func,
              "Check OK failed: %s returned [%s]: %s (in %s)", expr,
              status_code_name(status.code()), status.message(),
              func != nullptr ? func : "?");
  fatal_abort();
}

[[noreturn]] L3_COLD void panic_msg(const char* file, int line,
                                    const char* func, const char* fmt,
                                    ...) noexcept {
  char user_msg[512];
  user_msg[0] = '\0';
  if (fmt != nullptr) {
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(user_msg, sizeof(user_msg), fmt, args);
    va_end(args);
  }
  log_message(LogLevel::kFatal, file, line, func, "Panic: %s (in %s)",
              user_msg, func != nullptr ? func : "?");
  fatal_abort();
}

}  // namespace detail
}  // namespace l3
