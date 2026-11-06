// src/core/log.cpp
//
// Implementation of leveled logging, built-in sinks (stderr and Android
// logcat), thread-safe dispatch with re-entrancy protection, and fatal_abort().
//
// Design notes:
//   * Zero <mutex>, <chrono>, or <thread> dependencies. Locking uses a
//     cache-line-aligned atomic spinlock, and timestamps/thread IDs use
//     lightweight OS primitives plus thread_local state.
//   * Re-entrant calls from inside a custom sink are detected via a
//     thread-local flag and dropped silently so a buggy sink cannot deadlock.

#include "log3/core/log.h"

#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if L3_OS_WINDOWS
#include <windows.h>
#endif

#if L3_OS_ANDROID
#include <android/log.h>
#endif

namespace l3 {

namespace detail {

// Starts at kInfo by default.
L3_API std::atomic<u8> g_log_level{static_cast<u8>(LogLevel::kInfo)};

}  // namespace detail

namespace {

// -----------------------------------------------------------------------------
// Default sink selection
// -----------------------------------------------------------------------------
constexpr LogSinkFn default_sink_fn() noexcept {
#if L3_OS_ANDROID
  return &log_sink_logcat;
#else
  return &log_sink_stderr;
#endif
}

// -----------------------------------------------------------------------------
// Internal lock and active sink state
// -----------------------------------------------------------------------------
L3_ALIGNAS(kCacheLineSize) std::atomic_flag g_sink_lock = ATOMIC_FLAG_INIT;
LogSink g_active_sink = {default_sink_fn(), nullptr};

// Per-thread re-entrancy guard: true while this thread is executing a sink.
thread_local bool g_in_log_sink = false;

void lock_sink() noexcept {
  while (g_sink_lock.test_and_set(std::memory_order_acquire)) {
#if L3_ARCH_X64 || L3_ARCH_X86
#if L3_MSVC_ABI
    ::_mm_pause();
#else
    __builtin_ia32_pause();
#endif
#elif L3_ARCH_ARM64
    asm volatile("yield" ::: "memory");
#endif
  }
}

void unlock_sink() noexcept {
  g_sink_lock.clear(std::memory_order_release);
}

// -----------------------------------------------------------------------------
// Helpers: thread ID, wall-clock timestamp, and filename basename
// -----------------------------------------------------------------------------
u32 current_thread_id() noexcept {
  static std::atomic<u32> next_id{1};
  thread_local const u32 tid =
      next_id.fetch_add(1, std::memory_order_relaxed);
  return tid;
}

u64 wall_clock_ns() noexcept {
#if L3_OS_WINDOWS
  FILETIME ft;
  ::GetSystemTimeAsFileTime(&ft);
  ULARGE_INTEGER uli;
  uli.LowPart = ft.dwLowDateTime;
  uli.HighPart = ft.dwHighDateTime;
  // Convert 100-ns ticks since 1601-01-01 to Unix epoch (1970-01-01).
  constexpr u64 kEpochDiffTicks = 116444736000000000ULL;
  if (uli.QuadPart < kEpochDiffTicks) return 0;
  return (uli.QuadPart - kEpochDiffTicks) * 100ULL;
#else
  struct timespec ts {};
  if (::clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0;
  return static_cast<u64>(ts.tv_sec) * 1000000000ULL +
         static_cast<u64>(ts.tv_nsec);
#endif
}

const char* path_basename(const char* path) noexcept {
  if (path == nullptr) return "?";
  const char* base = path;
  for (const char* p = path; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\') {
      base = p + 1;
    }
  }
  return base;
}

const char* padded_level_name(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::kTrace: return "TRACE";
    case LogLevel::kDebug: return "DEBUG";
    case LogLevel::kInfo:  return "INFO ";
    case LogLevel::kWarn:  return "WARN ";
    case LogLevel::kError: return "ERROR";
    case LogLevel::kFatal: return "FATAL";
    case LogLevel::kOff:   return "OFF  ";
  }
  return "?    ";
}

}  // namespace

// -----------------------------------------------------------------------------
// Built-in sinks
// -----------------------------------------------------------------------------
void log_sink_stderr(const LogRecord& record, void* /*user_data*/) noexcept {
  // Format UTC/local time of day as [HH:MM:SS.mmm] from timestamp_ns.
  const u64 total_ms = record.timestamp_ns / 1000000ULL;
  const unsigned ms = static_cast<unsigned>(total_ms % 1000ULL);
  const std::time_t secs = static_cast<std::time_t>(total_ms / 1000ULL);

  struct std::tm tm_buf {};
#if L3_OS_WINDOWS
  ::localtime_s(&tm_buf, &secs);
#else
  ::localtime_r(&secs, &tm_buf);
#endif

  std::fprintf(stderr,
               "[%02d:%02d:%02d.%03u] [%s] [t%" PRIu32 "] %s:%d: %s\n",
               tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, ms,
               padded_level_name(record.level), record.thread_id,
               record.file != nullptr ? record.file : "?", record.line,
               record.message != nullptr ? record.message : "");
  if (record.level >= LogLevel::kError) {
    std::fflush(stderr);
  }
}

void log_sink_logcat(const LogRecord& record, void* /*user_data*/) noexcept {
#if L3_OS_ANDROID
  int prio = ANDROID_LOG_INFO;
  switch (record.level) {
    case LogLevel::kTrace: prio = ANDROID_LOG_VERBOSE; break;
    case LogLevel::kDebug: prio = ANDROID_LOG_DEBUG; break;
    case LogLevel::kInfo:  prio = ANDROID_LOG_INFO; break;
    case LogLevel::kWarn:  prio = ANDROID_LOG_WARN; break;
    case LogLevel::kError: prio = ANDROID_LOG_ERROR; break;
    case LogLevel::kFatal: prio = ANDROID_LOG_FATAL; break;
    case LogLevel::kOff:   return;
  }
  ::__android_log_print(prio, "log3", "[t%" PRIu32 "] %s:%d: %s",
                        record.thread_id,
                        record.file != nullptr ? record.file : "?",
                        record.line,
                        record.message != nullptr ? record.message : "");
#else
  L3_UNUSED(record);
#endif
}

void log_sink_null(const LogRecord& /*record*/, void* /*user_data*/) noexcept {}

// -----------------------------------------------------------------------------
// Sink and runtime level configuration
// -----------------------------------------------------------------------------
void set_log_sink(LogSink sink) noexcept {
  if (sink.fn == nullptr) return;
  lock_sink();
  g_active_sink = sink;
  unlock_sink();
}

void reset_log_sink() noexcept {
  lock_sink();
  g_active_sink = {default_sink_fn(), nullptr};
  unlock_sink();
}

void set_log_level(LogLevel level) noexcept {
  detail::g_log_level.store(static_cast<u8>(level), std::memory_order_relaxed);
}

LogLevel log_level() noexcept {
  return static_cast<LogLevel>(
      detail::g_log_level.load(std::memory_order_relaxed));
}

// -----------------------------------------------------------------------------
// Emission
// -----------------------------------------------------------------------------
void log_write(LogLevel level, const char* file, int line, const char* func,
               const char* message, usize length) noexcept {
  if (level != LogLevel::kFatal && !log_enabled(level)) {
    return;
  }
  if (L3_UNLIKELY(g_in_log_sink)) {
    return;  // Drop recursive log calls from inside a sink.
  }

  LogRecord record{};
  record.level = level;
  record.thread_id = current_thread_id();
  record.timestamp_ns = wall_clock_ns();
  record.file = path_basename(file);
  record.line = line;
  record.func = func != nullptr ? func : "?";
  record.message = message != nullptr ? message : "";
  record.length = length;

  lock_sink();
  g_in_log_sink = true;
  g_active_sink.fn(record, g_active_sink.user_data);
  g_in_log_sink = false;
  unlock_sink();
}

void log_message(LogLevel level, const char* file, int line, const char* func,
                 const char* fmt, ...) noexcept {
  if (level != LogLevel::kFatal && !log_enabled(level)) {
    return;
  }
  if (L3_UNLIKELY(fmt == nullptr)) {
    log_write(level, file, line, func, "", 0);
    return;
  }

  char buf[kLogMaxMessageLength];
  std::va_list args;
  va_start(args, fmt);
  const int written = std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  usize len = 0;
  if (written > 0) {
    len = static_cast<usize>(written) < sizeof(buf)
              ? static_cast<usize>(written)
              : sizeof(buf) - 1;
  }
  log_write(level, file, line, func, buf, len);
}

[[noreturn]] void fatal_abort() noexcept {
  std::fflush(stderr);
#if L3_DEBUG
  L3_DEBUG_BREAK();
#endif
  std::abort();
}

}  // namespace l3
