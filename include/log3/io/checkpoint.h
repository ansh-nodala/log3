// log3/io/checkpoint.h
//
// Fast zero-copy model serialization and deserialization engine for the `.l3`
// binary checkpoint format.
//
// Binary `.l3` Checkpoint Format:
//   * Header (64 bytes):
//       - Magic bytes    : "LOG3_MODEL_V01\0\0" (16 bytes)
//       - Version        : u32 (1)
//       - Endian tag     : u32 (0x01020304)
//       - Tensor count   : u64
//       - Meta offset    : u64 (8-byte-aligned offset to TensorMeta table)
//       - Meta size      : u64 (byte length of TensorMeta directory table)
//       - Reserved       : 16 bytes (zeroed)
//   * Tensor Data Block  : 64-byte-aligned payload buffers (mapped via MMapFile)
//   * Tensor Meta Table  : 8-byte-aligned array of `TensorMeta` descriptors
//                          (name, dtype, rank, dims, quant_scale, offset, size)
//
// Zero-Copy Guarantee:
//   `ModelReader` memory-maps the file in O(1) via `MMapFile` and creates
//   Tensors using `Storage::from_mmap`. Gigabyte-scale model weights are mapped
//   into process virtual memory in < 1 ms with 0 bytes allocated in physical RAM.
//   Notice that `meta_map_` keys (`StrView`) point directly into the immutable
//   memory-mapped `TensorMeta::name` array (not into temporary stack copies),
//   so string keys never dangle when `meta_map_` rehashes or looks up entries!
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include <cstring>

#include "log3/core/flat_map.h"
#include "log3/core/memory.h"
#include "log3/core/mmap.h"
#include "log3/core/platform.h"
#include "log3/core/small_vector.h"
#include "log3/core/status.h"
#include "log3/core/str_view.h"
#include "log3/core/types.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace io {

struct TensorMeta {
  char name[128];
  DType dtype;
  u8 rank;
  Index dims[8];
  f32 quant_scale;
  Size data_offset;
  Size data_bytes;
};

class L3_API ModelWriter {
 public:
  ModelWriter() noexcept : entries_() {}

  // Register a contiguous tensor to be serialized into the `.l3` file.
  // Copies `name` into a fixed 128-byte inline buffer so caller-owned transient
  // strings never dangle before `save()` is called, and rejects duplicate names.
  Status add_tensor(const char* name, const Tensor& tensor) noexcept;

  // Serialize all added tensors into `filepath` with 64-byte payload alignment
  // and 8-byte metadata table alignment.
  Status save(const char* filepath) const noexcept;

 private:
  struct WriteEntry {
    char name[128];
    Tensor tensor;
  };
  SmallVec<WriteEntry, 256> entries_;
};

class L3_API ModelReader {
 public:
  ModelReader() noexcept : holder_(), meta_map_() {}

  // Open and memory-map an `.l3` model checkpoint from `filepath`.
  static Result<ModelReader> open(
      const char* filepath, bool prefetch = false,
      const Allocator& allocator = default_allocator()) noexcept;

  Index num_tensors() const noexcept { return meta_map_.size(); }
  bool has_tensor(StrView name) const noexcept { return meta_map_.contains(name); }
  Result<Tensor> get_tensor(StrView name) const noexcept;

 private:
  Ptr<MMapHolder> holder_;
  StrMap<TensorMeta> meta_map_;
};

}  // namespace io
}  // namespace l3
