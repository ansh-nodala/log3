// log3/tensor/storage.h
//
// Storage: intrusive reference-counted byte buffer backing one or more Tensors.
//
// Three ownership modes (unified under a single branchless data pointer):
//   1. kOwnedAllocator : Allocated from a `l3::Allocator` (default_allocator()
//                        or an `Arena`), aligned to `kDefaultAlignment` (64 B).
//                        Freed automatically when the last `Ptr<Storage>`
//                        releases.
//   2. kSharedMMap     : Zero-copy slice inside a shared `RefCounted` MMapFile
//                        holder (`MMapHolder`), keeping the OS file mapping
//                        alive for as long as any Tensor views its weights!
//   3. kExternalBorrowed: Non-owning view of caller-managed memory (e.g., a
//                        NumPy/PyTorch tensor buffer passed across the C API or
//                        a stack scratch buffer).
//
// Note on Allocator storage inside `Storage`:
//   `Storage` stores its `data_allocator_` by VALUE (not by pointer) because
//   `l3::Allocator` is a trivially copyable 32-byte struct (`Arena::allocator()`
//   returns a temporary `Allocator` by value). Only the `RefCounted` object
//   header itself is allocated via `default_allocator()`, which has static
//   program-wide lifetime, while the tensor payload buffer `data_` is allocated
//   and freed through `data_allocator_`.
//
// Compatibility:
//   * C++17, header-only, no exceptions, no RTTI.

#pragma once

#include <utility>

#include "log3/core/assert.h"
#include "log3/core/memory.h"
#include "log3/core/mmap.h"
#include "log3/core/platform.h"
#include "log3/core/ptr.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

// Intrusive RefCounted wrapper around MMapFile so multiple Tensor Storages can
// share sub-slices of the same memory-mapped model file safely.
class MMapHolder final : public RefCounted {
 public:
  explicit MMapHolder(MMapFile&& file) noexcept : file_(std::move(file)) {}

  const MMapFile& file() const noexcept { return file_; }
  MMapFile& file() noexcept { return file_; }

 private:
  MMapFile file_;
};

class Storage final : public RefCounted {
 public:
  enum class Kind : u8 {
    kOwnedAllocator = 0,
    kSharedMMap = 1,
    kExternalBorrowed = 2,
  };

  // --- factory methods -------------------------------------------------------

  // Allocate `size_bytes` of 64-byte-aligned memory from `allocator`.
  // If `zero_init` is true, zeros the buffer.
  static Result<Ptr<Storage>> allocate(
      Size size_bytes, bool zero_init = false,
      const Allocator& allocator = default_allocator()) noexcept {
    if (L3_UNLIKELY(!allocator.is_valid())) {
      return Status::error(StatusCode::kInvalidArgument,
                           "Storage::allocate: allocator is invalid");
    }

    byte* raw = nullptr;
    if (size_bytes > 0) {
      raw = static_cast<byte*>(
          allocator.allocate(size_bytes, kDefaultAlignment));
      if (L3_UNLIKELY(raw == nullptr)) {
        return Status::errorf(
            StatusCode::kOutOfMemory,
            "Storage::allocate: failed to allocate %zu bytes from '%s'",
            size_bytes, allocator.name != nullptr ? allocator.name : "?");
      }
      if (zero_init) {
        mem_zero(raw, size_bytes);
      }
    }

    // Allocate the small Storage control object via default_allocator() (which
    // has static lifetime, satisfying RefCounted's requirement that its header
    // allocator pointer outlive the object even when `allocator` was a
    // temporary struct returned by `arena.allocator()`).
    auto res = make_ptr_in<Storage>(default_allocator(), raw, size_bytes,
                                    Kind::kOwnedAllocator, /*mutable=*/true,
                                    &allocator, Ptr<MMapHolder>());
    if (L3_UNLIKELY(!res.is_ok())) {
      if (raw != nullptr) {
        allocator.deallocate(raw, size_bytes, kDefaultAlignment);
      }
      return res.take_status();
    }
    return res;
  }

  // Create a read-only zero-copy Storage backed by a slice `[offset, offset +
  // length)` of a shared `MMapHolder`.
  static Result<Ptr<Storage>> from_mmap(
      const Ptr<MMapHolder>& holder, Size offset, Size length,
      const Allocator& meta_allocator = default_allocator()) noexcept {
    if (L3_UNLIKELY(!holder || !holder->file().is_open())) {
      return Status::error(StatusCode::kInvalidArgument,
                           "Storage::from_mmap: MMapHolder is null or closed");
    }
    ConstByteSpan slice;
    L3_ASSIGN_OR_RETURN(slice, holder->file().slice(offset, length));

    return make_ptr_in<Storage>(
        default_allocator(), const_cast<byte*>(slice.data()), length,
        Kind::kSharedMMap, /*mutable=*/false, &meta_allocator, holder);
  }

  // Wrap an external mutable buffer without taking ownership.
  // The caller guarantees `data` outlives all Tensors referencing this Storage.
  static Result<Ptr<Storage>> from_external(
      void* data, Size size_bytes, bool is_mutable = true,
      const Allocator& meta_allocator = default_allocator()) noexcept {
    if (L3_UNLIKELY(size_bytes > 0 && data == nullptr)) {
      return Status::error(
          StatusCode::kInvalidArgument,
          "Storage::from_external: data is null with non-zero size");
    }
    return make_ptr_in<Storage>(
        default_allocator(), static_cast<byte*>(data), size_bytes,
        Kind::kExternalBorrowed, is_mutable, &meta_allocator,
        Ptr<MMapHolder>());
  }

  // Wrap an external read-only buffer without taking ownership.
  static Result<Ptr<Storage>> from_external_const(
      const void* data, Size size_bytes,
      const Allocator& meta_allocator = default_allocator()) noexcept {
    return from_external(const_cast<void*>(data), size_bytes,
                         /*is_mutable=*/false, meta_allocator);
  }

  // Public constructor for `make_ptr_in<Storage>` (prefer static factories).
  // Copies `*data_allocator` by value so temporary Allocator wrappers (such as
  // `arena.allocator()`) never dangle!
  Storage(byte* data, Size size_bytes, Kind kind, bool is_mutable,
          const Allocator* data_allocator, Ptr<MMapHolder> mmap_holder) noexcept
      : data_(data),
        size_bytes_(size_bytes),
        data_allocator_(data_allocator != nullptr ? *data_allocator
                                                  : default_allocator()),
        mmap_holder_(std::move(mmap_holder)),
        kind_(kind),
        is_mutable_(is_mutable) {}

  ~Storage() noexcept {
    if (kind_ == Kind::kOwnedAllocator && data_ != nullptr &&
        data_allocator_.is_valid()) {
      data_allocator_.deallocate(data_, size_bytes_, kDefaultAlignment);
      data_ = nullptr;
    }
  }

  // --- observers -------------------------------------------------------------
  const byte* data() const noexcept { return data_; }

  byte* mutable_data() noexcept {
    L3_DCHECK(is_mutable_);
    return data_;
  }

  Size size_bytes() const noexcept { return size_bytes_; }
  Kind kind() const noexcept { return kind_; }
  bool is_mutable() const noexcept { return is_mutable_; }

  ConstByteSpan bytes() const noexcept {
    return ConstByteSpan(data_, static_cast<Index>(size_bytes_));
  }

  ByteSpan writable_bytes() noexcept {
    L3_DCHECK(is_mutable_);
    return ByteSpan(data_, static_cast<Index>(size_bytes_));
  }

 private:
  byte* data_;
  Size size_bytes_;
  Allocator data_allocator_;  // Stored by value (32B) so temporary Allocators never dangle
  Ptr<MMapHolder> mmap_holder_;
  Kind kind_;
  bool is_mutable_;
};

}  // namespace l3
