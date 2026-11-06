// src/core/mmap.cpp
//
// Cross-platform read-only memory-mapped file implementation.

#include "log3/core/mmap.h"

#include <cerrno>
#include <cstring>

#include "log3/core/memory.h"

#if L3_OS_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace l3 {

MMapFile::MMapFile(MMapFile&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      os_file_handle_(other.os_file_handle_),
      os_map_handle_(other.os_map_handle_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.os_file_handle_ = -1;
  other.os_map_handle_ = nullptr;
}

MMapFile& MMapFile::operator=(MMapFile&& other) noexcept {
  if (this != &other) {
    close();
    data_ = other.data_;
    size_ = other.size_;
    os_file_handle_ = other.os_file_handle_;
    os_map_handle_ = other.os_map_handle_;

    other.data_ = nullptr;
    other.size_ = 0;
    other.os_file_handle_ = -1;
    other.os_map_handle_ = nullptr;
  }
  return *this;
}

Result<MMapFile> MMapFile::open(const char* path, bool prefetch) noexcept {
  if (L3_UNLIKELY(path == nullptr || path[0] == '\0')) {
    return Status::error(StatusCode::kInvalidArgument,
                         "MMapFile::open: path is null or empty");
  }

#if L3_OS_WINDOWS
  HANDLE file_h = ::CreateFileA(
      path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (file_h == INVALID_HANDLE_VALUE) {
    const DWORD err = ::GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      return Status::errorf(StatusCode::kNotFound,
                            "MMapFile::open: file not found: '%s'", path);
    }
    return Status::errorf(StatusCode::kIoError,
                          "MMapFile::open: CreateFileA failed for '%s' (err=%lu)",
                          path, static_cast<unsigned long>(err));
  }

  LARGE_INTEGER file_size{};
  if (!::GetFileSizeEx(file_h, &file_size) || file_size.QuadPart <= 0) {
    ::CloseHandle(file_h);
    return Status::errorf(StatusCode::kInvalidArgument,
                          "MMapFile::open: file is empty or unreadable: '%s'",
                          path);
  }

  HANDLE map_h =
      ::CreateFileMappingW(file_h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (map_h == nullptr) {
    const DWORD err = ::GetLastError();
    ::CloseHandle(file_h);
    return Status::errorf(
        StatusCode::kIoError,
        "MMapFile::open: CreateFileMapping failed for '%s' (err=%lu)", path,
        static_cast<unsigned long>(err));
  }

  const void* view = ::MapViewOfFile(map_h, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    const DWORD err = ::GetLastError();
    ::CloseHandle(map_h);
    ::CloseHandle(file_h);
    return Status::errorf(
        StatusCode::kIoError,
        "MMapFile::open: MapViewOfFile failed for '%s' (err=%lu)", path,
        static_cast<unsigned long>(err));
  }

  MMapFile out;
  out.data_ = static_cast<const byte*>(view);
  out.size_ = static_cast<Size>(file_size.QuadPart);
  out.os_file_handle_ = reinterpret_cast<isize>(file_h);
  out.os_map_handle_ = static_cast<void*>(map_h);

  if (prefetch) {
    out.advise(MMapAdvice::kWillNeed).ignore();
  }
  return out;

#else
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  const int fd = ::open(path, flags);
  if (fd < 0) {
    const int err = errno;
    if (err == ENOENT) {
      return Status::errorf(StatusCode::kNotFound,
                            "MMapFile::open: file not found: '%s'", path);
    }
    return Status::errorf(StatusCode::kIoError,
                          "MMapFile::open: open('%s') failed: %s", path,
                          std::strerror(err));
  }

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const int err = errno;
    ::close(fd);
    return Status::errorf(StatusCode::kIoError,
                          "MMapFile::open: fstat('%s') failed: %s", path,
                          std::strerror(err));
  }

  if (st.st_size <= 0) {
    ::close(fd);
    return Status::errorf(StatusCode::kInvalidArgument,
                          "MMapFile::open: cannot map empty file: '%s'", path);
  }

  const Size file_size = static_cast<Size>(st.st_size);
  void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    const int err = errno;
    ::close(fd);
    return Status::errorf(StatusCode::kIoError,
                          "MMapFile::open: mmap('%s', %zu bytes) failed: %s",
                          path, file_size, std::strerror(err));
  }

  // On POSIX, the file descriptor can be closed immediately after mmap succeeds
  // without invalidating the mapping.
  ::close(fd);

  MMapFile out;
  out.data_ = static_cast<const byte*>(mapped);
  out.size_ = file_size;
  out.os_file_handle_ = -1;
  out.os_map_handle_ = nullptr;

  if (prefetch) {
    out.advise(MMapAdvice::kWillNeed).ignore();
  }
  return out;
#endif
}

void MMapFile::close() noexcept {
  if (data_ == nullptr) return;

#if L3_OS_WINDOWS
  ::UnmapViewOfFile(static_cast<const void*>(data_));
  if (os_map_handle_ != nullptr) {
    ::CloseHandle(static_cast<HANDLE>(os_map_handle_));
  }
  if (os_file_handle_ != -1) {
    ::CloseHandle(reinterpret_cast<HANDLE>(os_file_handle_));
  }
#else
  ::munmap(const_cast<byte*>(data_), size_);
#endif

  data_ = nullptr;
  size_ = 0;
  os_file_handle_ = -1;
  os_map_handle_ = nullptr;
}

Status MMapFile::advise(MMapAdvice advice, Size offset,
                        Size length) const noexcept {
  if (!is_open()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "MMapFile::advise: file is not open");
  }
  if (offset > size_) {
    return Status::error(StatusCode::kOutOfRange,
                         "MMapFile::advise: offset exceeds file size");
  }
  const Size eff_len = (length == 0) ? (size_ - offset) : length;
  if (eff_len > size_ - offset) {
    return Status::error(StatusCode::kOutOfRange,
                         "MMapFile::advise: range exceeds file size");
  }
  if (eff_len == 0) return Status::ok();

#if L3_OS_POSIX
  // madvise requires page-aligned address on Linux/macOS.
  const long page_sz_raw = ::sysconf(_SC_PAGESIZE);
  const Size page_sz =
      page_sz_raw > 0 ? static_cast<Size>(page_sz_raw) : 4096u;
  const Size aligned_off = align_down(offset, page_sz);
  const Size delta = offset - aligned_off;
  const Size aligned_len = eff_len + delta;

  int posix_adv = MADV_NORMAL;
  switch (advice) {
    case MMapAdvice::kNormal:     posix_adv = MADV_NORMAL; break;
    case MMapAdvice::kSequential: posix_adv = MADV_SEQUENTIAL; break;
    case MMapAdvice::kRandom:     posix_adv = MADV_RANDOM; break;
    case MMapAdvice::kWillNeed:   posix_adv = MADV_WILLNEED; break;
    case MMapAdvice::kDontNeed:   posix_adv = MADV_DONTNEED; break;
  }

  void* addr = const_cast<byte*>(data_ + aligned_off);
  ::madvise(addr, aligned_len, posix_adv);
#else
  L3_UNUSED(advice);
#endif
  return Status::ok();
}

Result<ConstByteSpan> MMapFile::slice(Size offset,
                                      Size length) const noexcept {
  if (!is_open()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "MMapFile::slice: file is not open");
  }
  Size end = 0;
  if (!checked_add(offset, length, &end) || end > size_) {
    return Status::errorf(
        StatusCode::kOutOfRange,
        "MMapFile::slice: range [%zu, %zu + %zu) exceeds file size (%zu)",
        offset, offset, length, size_);
  }
  return ConstByteSpan(data_ + offset, static_cast<Index>(length));
}

}  // namespace l3
