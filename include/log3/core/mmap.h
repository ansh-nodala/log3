// log3/core/mmap.h
//
// MMapFile: cross-platform read-only memory-mapped file for zero-copy model
// weight loading (.gguf, .safetensors, .l3).
//
// Why MMapFile belongs in log3/core:
//   * BitNet b1.58 and modern LLM checkpoints are hundreds of megabytes to
//     tens of gigabytes. Reading them via std::ifstream or fread() duplicates
//     the entire model in RAM and delays startup by seconds.
//   * MMapFile maps the file directly into the process address space via the
//     OS page cache (POSIX mmap / Windows CreateFileMappingW) in < 1 ms with
//     zero heap allocations.
//   * RefCounted-compatible or standalone move-only RAII object. Exposes
//     `ConstByteSpan`, `advise_sequential()`, `advise_willneed()`, and
//     checked sub-slice helpers for safe header/tensor parsing.
//
// Compatibility:
//   * Windows (CreateFileA / CreateFileMappingW / MapViewOfFile / VirtualLock)
//   * Linux, Android, macOS (open / fstat / mmap / madvise)
//   * C++17, no exceptions, no RTTI

#pragma once

#include "log3/core/platform.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

enum class MMapAdvice : u8 {
  kNormal = 0,      // Default OS paging behavior
  kSequential = 1,  // Aggressive read-ahead (scanning model headers/weights)
  kRandom = 2,      // Disable read-ahead (random access into large files)
  kWillNeed = 3,    // Prefetch pages into physical RAM asynchronously
  kDontNeed = 4,    // Hint that pages can be evicted from physical RAM
};

class L3_API MMapFile {
 public:
  MMapFile() noexcept
      : data_(nullptr), size_(0), os_file_handle_(-1), os_map_handle_(nullptr) {}

  MMapFile(const MMapFile&) = delete;
  MMapFile& operator=(const MMapFile&) = delete;

  MMapFile(MMapFile&& other) noexcept;
  MMapFile& operator=(MMapFile&& other) noexcept;

  ~MMapFile() noexcept { close(); }

  // Open `path` and map its entire contents read-only into virtual memory.
  // Fails with:
  //   * kInvalidArgument if path is null/empty or the file is 0 bytes.
  //   * kNotFound if the file does not exist.
  //   * kIoError if mapping fails.
  static Result<MMapFile> open(const char* path,
                               bool prefetch = false) noexcept;

  // Unmap the file and release OS descriptors. Safe to call multiple times.
  void close() noexcept;

  // Provide paging hints to the OS kernel (madvise / PrefetchVirtualMemory).
  Status advise(MMapAdvice advice, Size offset = 0,
                Size length = 0) const noexcept;

  // --- observers -------------------------------------------------------------
  bool is_open() const noexcept { return data_ != nullptr; }
  const byte* data() const noexcept { return data_; }
  Size size() const noexcept { return size_; }

  ConstByteSpan bytes() const noexcept {
    return ConstByteSpan(data_, static_cast<Index>(size_));
  }

  // Checked sub-slice of `[offset, offset + length)`.
  // Returns kOutOfRange if the slice exceeds the mapped file size.
  Result<ConstByteSpan> slice(Size offset, Size length) const noexcept;

 private:
  const byte* data_;
  Size size_;
  isize os_file_handle_;  // fd on POSIX, intptr_t HANDLE on Windows
  void* os_map_handle_;   // mapping HANDLE on Windows, unused on POSIX
};

}  // namespace l3
