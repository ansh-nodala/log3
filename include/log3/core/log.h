// log3/core/log.h
//
// Leveled logging with pluggable sinks and compile-time stripping.
//
// Usage:
//   L3_LOG_INFO("loaded %d tensors in %.2f ms", count, ms);
//   L3_LOG_ERROR("matmul failed: %s", status.message());
//   l3::set_log_level(l3::LogLevel::kDebug);
//
// Design:
//   * Two filters. Compile-time: L3_LOG_MIN_LEVEL removes lower levels from the
//     binary completely (arguments are not even evaluated). Runtime: one
//     relaxed atomic load decides whether a message is built at all, so a
//     disabled log line costs a single compare-and-branch.
//   * Sinks are a plain function pointer + user_data (no virtual, no
//     std::function, no allocation). Built-in sinks: stderr and Android
//     logcat. The Python binding installs its own sink the same way.
//   * Sinks are invoked under an internal lock, so lines never interleave.
//     A sink must NOT log recursively (the implementation detects and drops
//     re-entrant calls).
//   * Logging never throws and never returns an error.
//   * L3_LOG_FATAL logs and then aborts the process.
//
// Compile-time level (set from CMake, e.g. -DL3_LOG_MIN_LEVEL=2):
//   0 trace, 1 debug, 2 info, 3 warn, 4 error, 5 fatal
//   default: 0 in debug builds, 2 (info) in release builds.

#pragma once

#include <atomic>

#include "log3/core/platform.h"
#include "log3/core/types.h"

#define L3_LOG_LEVEL_TRACE 0
#define L3_LOG_LEVEL_DEBUG 1
#define L3_LOG_LEVEL_INFO 2
#define L3_LOG_LEVEL_WARN 3
#define L3_LOG_LEVEL_ERROR 4
#define L3_LOG_LEVEL_FATAL 5

#ifndef L3_LOG_MIN_LEVEL
#if L3_DEBUG
#define L3_LOG_MIN_LEVEL L3_LOG_LEVEL_TRACE
#else
#define L3_LOG_MIN_LEVEL L3_LOG_LEVEL_INFO
#endif
#endif

namespace l3 {

// -----------------------------------------------------------------------------
// Levels
// -----------------------------------------------------------------------------
enum class LogLevel : u8 {
  kTrace = L3_LOG_LEVEL_TRACE,
  kDebug = L3_LOG_LEVEL_DEBUG,
  kInfo = L3_LOG_LEVEL_INFO,
  kWarn = L3_LOG_LEVEL_WARN,
  kError = L3_LOG_LEVEL_ERROR,
  kFatal = L3_LOG_LEVEL_FATAL,
  kOff = 6,  // only valid for set_log_level(): silences everything
};

constexpr const char* log_level_name(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::kTrace: return "TRACE";
    case LogLevel::kDebug: return "DEBUG";
    case LogLevel::kInfo: return "INFO";
    case LogLevel::kWarn: return "WARN";
    case LogLevel::kError: return "ERROR";
    case LogLevel::kFatal: return "FATAL";
    case LogLevel::kOff: return "OFF";
  }
  return "?";
}

// -----------------------------------------------------------------------------
// Log record and sinks
// -----------------------------------------------------------------------------

// Everything a sink needs. All pointers are valid only during the sink call.
struct LogRecord {
  LogLevel level;
  u32 thread_id;       // small stable id of the calling thread
  u64 timestamp_ns;    // nanoseconds since the Unix epoch (wall clock)
  const char* file;    // source file (basename only)
  int line;
  const char* func;
  const char* message;  // null-terminated, no trailing newline
  usize length;         // strlen(message)
};

// A sink receives fully built records. It must be fast and must not log.
using LogSinkFn = void (*)(const LogRecord& record, void* user_data) noexcept;

struct LogSink {
  LogSinkFn fn;
  void* user_data;
};

// Built-in sinks.
//   log_sink_stderr : "[12:30:05.123] [INFO ] [t1] file.cpp:42: message"
//   log_sink_logcat : Android logcat (tag "log3"); no-op on other platforms
//   log_sink_null   : drops everything
L3_API void log_sink_stderr(const LogRecord& record, void* user_data) noexcept;
L3_API void log_sink_logcat(const LogRecord& record, void* user_data) noexcept;
L3_API void log_sink_null(const LogRecord& record, void* user_data) noexcept;

// Install a custom sink (fn must not be null). Safe to call at any time.
L3_API void set_log_sink(LogSink sink) noexcept;

// Restore the platform default: logcat on Android, stderr everywhere else.
L3_API void reset_log_sink() noexcept;

// -----------------------------------------------------------------------------
// Runtime level
// -----------------------------------------------------------------------------
namespace detail {

// Defined in src/core/log.cpp. Starts at kInfo.
L3_API extern std::atomic<u8> g_log_level;

}  // namespace detail

L3_API void set_log_level(LogLevel level) noexcept;
L3_API LogLevel log_level() noexcept;

// Hot-path check used by the macros: one relaxed load + compare.
L3_FORCE_INLINE bool log_enabled(LogLevel level) noexcept {
  return static_cast<u8>(level) >=
         detail::g_log_level.load(std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Emission (normally used through the macros below)
// -----------------------------------------------------------------------------

// printf-style. The message is truncated to kLogMaxMessageLength bytes.
L3_API void log_message(LogLevel level, const char* file, int line,
                        const char* func, const char* fmt, ...) noexcept
    L3_PRINTF_FORMAT(5, 6);

// Pre-formatted message (no format parsing, no truncation by the formatter).
L3_API void log_write(LogLevel level, const char* file, int line,
                      const char* func, const char* message,
                      usize length) noexcept;

// Flush/terminate path used by L3_LOG_FATAL: breaks into the debugger in
// debug builds, then aborts the process.
[[noreturn]] L3_API void fatal_abort() noexcept;

inline constexpr usize kLogMaxMessageLength = 1024;

}  // namespace l3

// -----------------------------------------------------------------------------
// Macros
// -----------------------------------------------------------------------------
#define L3_LOG_IMPL(level, ...)                                           \
  do {                                                                    \
    if (::l3::log_enabled(level)) {                                       \
      ::l3::log_message(level, __FILE__, __LINE__, __func__, __VA_ARGS__);\
    }                                                                     \
  } while (0)

#if L3_LOG_MIN_LEVEL <= L3_LOG_LEVEL_TRACE
#define L3_LOG_TRACE(...) L3_LOG_IMPL(::l3::LogLevel::kTrace, __VA_ARGS__)
#else
#define L3_LOG_TRACE(...) ((void)0)
#endif

#if L3_LOG_MIN_LEVEL <= L3_LOG_LEVEL_DEBUG
#define L3_LOG_DEBUG(...) L3_LOG_IMPL(::l3::LogLevel::kDebug, __VA_ARGS__)
#else
#define L3_LOG_DEBUG(...) ((void)0)
#endif

#if L3_LOG_MIN_LEVEL <= L3_LOG_LEVEL_INFO
#define L3_LOG_INFO(...) L3_LOG_IMPL(::l3::LogLevel::kInfo, __VA_ARGS__)
#else
#define L3_LOG_INFO(...) ((void)0)
#endif

#if L3_LOG_MIN_LEVEL <= L3_LOG_LEVEL_WARN
#define L3_LOG_WARN(...) L3_LOG_IMPL(::l3::LogLevel::kWarn, __VA_ARGS__)
#else
#define L3_LOG_WARN(...) ((void)0)
#endif

#if L3_LOG_MIN_LEVEL <= L3_LOG_LEVEL_ERROR
#define L3_LOG_ERROR(...) L3_LOG_IMPL(::l3::LogLevel::kError, __VA_ARGS__)
#else
#define L3_LOG_ERROR(...) ((void)0)
#endif

// Fatal is never stripped and never filtered: it always logs, then aborts.
#define L3_LOG_FATAL(...)                                                   \
  do {                                                                      \
    ::l3::log_message(::l3::LogLevel::kFatal, __FILE__, __LINE__, __func__, \
                      __VA_ARGS__);                                         \
    ::l3::fatal_abort();                                                    \
  } while (0)
