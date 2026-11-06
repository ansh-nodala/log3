// src/io/checkpoint.cpp
//
// Implementation of ModelWriter and ModelReader for `.l3` binary serialization.

#include "log3/io/checkpoint.h"

#include <cstdio>
#include <cstring>

#include "log3/core/bit_ops.h"
#include "log3/core/scope_guard.h"

namespace l3 {
namespace io {

namespace {

constexpr char kMagicString[16] = "LOG3_MODEL_V01\0";
constexpr u32 kCurrentVersion = 1;
constexpr u32 kEndianTag = 0x01020304u;

struct Header {
  char magic[16];
  u32 version;
  u32 endian_tag;
  u64 tensor_count;
  u64 meta_offset;
  u64 meta_size;
  u8 reserved[16];
};

static_assert(sizeof(Header) == 64, "Header must be exactly 64 bytes");
static_assert(sizeof(TensorMeta) == 224, "TensorMeta must be 224 bytes");

// Bounded strlen that never reads past `max_cap` bytes if a corrupted file
// omits a null terminator inside `TensorMeta::name`.
usize safe_name_len(const char* s, usize max_cap) noexcept {
  usize n = 0;
  while (n < max_cap && s[n] != '\0') {
    ++n;
  }
  return n;
}

}  // namespace

// -----------------------------------------------------------------------------
// ModelWriter
// -----------------------------------------------------------------------------

Status ModelWriter::add_tensor(const char* name, const Tensor& tensor) noexcept {
  if (name == nullptr || name[0] == '\0' || !tensor.is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "ModelWriter::add_tensor: invalid name or Tensor");
  }
  const usize name_len = std::strlen(name);
  if (name_len >= sizeof(TensorMeta::name)) {
    return Status::errorf(
        StatusCode::kInvalidArgument,
        "ModelWriter::add_tensor: tensor name '%s' exceeds 127 characters",
        name);
  }
  if (!tensor.is_contiguous()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "ModelWriter::add_tensor: tensor must be contiguous");
  }

  // Reject duplicate tensor names upfront so a checkpoint never contains
  // shadowed entries.
  for (Index i = 0; i < entries_.size(); ++i) {
    if (std::strcmp(entries_[i].name, name) == 0) {
      return Status::errorf(
          StatusCode::kAlreadyExists,
          "ModelWriter::add_tensor: duplicate tensor name '%s'", name);
    }
  }

  WriteEntry entry{};
  std::memcpy(entry.name, name, name_len + 1);
  entry.tensor = tensor.view();
  return entries_.push_back(std::move(entry));
}

Status ModelWriter::save(const char* filepath) const noexcept {
  if (filepath == nullptr || filepath[0] == '\0') {
    return Status::error(StatusCode::kInvalidArgument,
                         "ModelWriter::save: invalid filepath");
  }

  std::FILE* fp = std::fopen(filepath, "wb");
  if (fp == nullptr) {
    return Status::errorf(
        StatusCode::kIoError,
        "ModelWriter::save: failed to open file '%s' for writing", filepath);
  }
  L3_DEFER { std::fclose(fp); };

  Header hdr{};
  std::memcpy(hdr.magic, kMagicString, 16);
  hdr.version = kCurrentVersion;
  hdr.endian_tag = kEndianTag;
  hdr.tensor_count = static_cast<u64>(entries_.size());
  hdr.meta_offset = 0;
  hdr.meta_size = 0;
  mem_zero(hdr.reserved, 16);

  if (std::fwrite(&hdr, sizeof(Header), 1, fp) != 1) {
    return Status::error(StatusCode::kIoError,
                         "ModelWriter::save: write header failed");
  }

  SmallVec<TensorMeta, 256> metas;
  Size current_pos = sizeof(Header);

  for (Index i = 0; i < entries_.size(); ++i) {
    const auto& entry = entries_[i];
    const Tensor& t = entry.tensor;

    // Align every tensor payload to 64 bytes (kDefaultAlignment) so SIMD AVX-512
    // and AVX2 loads on memory-mapped tensors are always cache-line aligned.
    const Size aligned_pos = align_up(current_pos, kDefaultAlignment);
    const Size pad_bytes = aligned_pos - current_pos;
    if (pad_bytes > 0) {
      byte zero_pad[64]{};
      if (std::fwrite(zero_pad, 1, pad_bytes, fp) != pad_bytes) {
        return Status::error(StatusCode::kIoError,
                             "ModelWriter::save: failed writing alignment pad");
      }
      current_pos = aligned_pos;
    }

    TensorMeta meta{};
    std::memcpy(meta.name, entry.name, sizeof(meta.name));
    meta.dtype = t.dtype();
    meta.rank = static_cast<u8>(t.rank());
    for (Index d = 0; d < t.rank(); ++d) {
      meta.dims[d] = t.dim(d);
    }
    meta.quant_scale = t.quant_scale();
    meta.data_offset = current_pos;
    meta.data_bytes = t.nbytes();

    if (t.nbytes() > 0) {
      if (std::fwrite(t.raw_data(), 1, t.nbytes(), fp) != t.nbytes()) {
        return Status::error(StatusCode::kIoError,
                             "ModelWriter::save: failed writing tensor payload");
      }
      current_pos += t.nbytes();
    }

    L3_RETURN_IF_ERROR(metas.push_back(meta));
  }

  // Align the metadata directory table to `alignof(TensorMeta)` (8 bytes) so
  // reading `TensorMeta` records from the memory-mapped file never triggers
  // misaligned struct access UB when the final tensor payload is not a multiple
  // of 8 bytes.
  const Size aligned_meta_pos = align_up(current_pos, alignof(TensorMeta));
  const Size meta_pad_bytes = aligned_meta_pos - current_pos;
  if (meta_pad_bytes > 0) {
    byte zero_pad[8]{};
    if (std::fwrite(zero_pad, 1, meta_pad_bytes, fp) != meta_pad_bytes) {
      return Status::error(StatusCode::kIoError,
                           "ModelWriter::save: failed writing meta padding");
    }
    current_pos = aligned_meta_pos;
  }

  const Size meta_offset = current_pos;
  const Size meta_bytes = static_cast<Size>(metas.size()) * sizeof(TensorMeta);

  if (meta_bytes > 0) {
    if (std::fwrite(metas.data(), sizeof(TensorMeta),
                    static_cast<usize>(metas.size()),
                    fp) != static_cast<usize>(metas.size())) {
      return Status::error(StatusCode::kIoError,
                           "ModelWriter::save: failed writing metadata table");
    }
  }

  hdr.meta_offset = meta_offset;
  hdr.meta_size = meta_bytes;
  if (std::fseek(fp, 0, SEEK_SET) != 0 ||
      std::fwrite(&hdr, sizeof(Header), 1, fp) != 1 || std::fflush(fp) != 0) {
    return Status::error(StatusCode::kIoError,
                         "ModelWriter::save: rewrite header failed");
  }

  return Status::ok();
}

// -----------------------------------------------------------------------------
// ModelReader
// -----------------------------------------------------------------------------

Result<ModelReader> ModelReader::open(const char* filepath, bool prefetch,
                                      const Allocator& allocator) noexcept {
  MMapFile mf;
  L3_ASSIGN_OR_RETURN(mf, MMapFile::open(filepath, prefetch));

  if (mf.size() < sizeof(Header)) {
    return Status::error(StatusCode::kInvalidArgument,
                         "ModelReader::open: file too small for log3 header");
  }

  Header hdr{};
  std::memcpy(&hdr, mf.data(), sizeof(Header));
  if (std::memcmp(hdr.magic, kMagicString, 16) != 0) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "ModelReader::open: invalid file magic (not an .l3 checkpoint)");
  }
  if (hdr.version != kCurrentVersion || hdr.endian_tag != kEndianTag) {
    return Status::error(StatusCode::kInvalidArgument,
                         "ModelReader::open: unsupported version or byte order");
  }

  // Overflow-checked validation of tensor_count, meta_size, and meta_offset.
  if (hdr.tensor_count > static_cast<u64>(kMaxIndex)) {
    return Status::error(StatusCode::kOutOfRange,
                         "ModelReader::open: tensor_count overflow");
  }
  Size expected_meta_bytes = 0;
  if (!checked_mul(static_cast<Size>(hdr.tensor_count), sizeof(TensorMeta),
                   &expected_meta_bytes) ||
      expected_meta_bytes != static_cast<Size>(hdr.meta_size)) {
    return Status::error(StatusCode::kInvalidArgument,
                         "ModelReader::open: corrupt meta_size in header");
  }
  Size meta_end = 0;
  if (!checked_add(static_cast<Size>(hdr.meta_offset), expected_meta_bytes,
                   &meta_end) ||
      meta_end > mf.size()) {
    return Status::error(StatusCode::kOutOfRange,
                         "ModelReader::open: metadata table exceeds file size");
  }

  // Allocate MMapHolder via default_allocator() so its RefCounted header is
  // always backed by a static-lifetime allocator.
  Ptr<MMapHolder> holder;
  L3_ASSIGN_OR_RETURN(holder,
                      make_ptr_in<MMapHolder>(default_allocator(), std::move(mf)));

  ModelReader reader;
  reader.holder_ = std::move(holder);
  L3_UNUSED(allocator);
  L3_RETURN_IF_ERROR(
      reader.meta_map_.reserve(static_cast<Index>(hdr.tensor_count)));

  const byte* raw_metas =
      reader.holder_->file().data() + static_cast<Size>(hdr.meta_offset);

  for (u64 i = 0; i < hdr.tensor_count; ++i) {
    const byte* record_ptr = raw_metas + static_cast<Size>(i) * sizeof(TensorMeta);
    TensorMeta m{};
    std::memcpy(&m, record_ptr, sizeof(TensorMeta));
    m.name[sizeof(m.name) - 1] = '\0';

    // CRITICAL FIX: Construct the StrView key pointing into the persistent
    // memory-mapped file (`record_ptr`), NOT into a temporary stack variable!
    // The MMapFile stays alive inside `reader.holder_` for the entire lifetime
    // of `ModelReader`.
    const char* mapped_name = reinterpret_cast<const char*>(record_ptr);
    const usize name_len = safe_name_len(mapped_name, sizeof(m.name) - 1);
    const StrView key(mapped_name, static_cast<Index>(name_len));

    L3_RETURN_IF_ERROR(reader.meta_map_.put(key, m));
  }

  return reader;
}

Result<Tensor> ModelReader::get_tensor(StrView name) const noexcept {
  const TensorMeta* meta = meta_map_.find(name);
  if (meta == nullptr) {
    return Status::errorf(StatusCode::kNotFound,
                          "ModelReader::get_tensor: tensor '%.*s' not found",
                          static_cast<int>(name.size()), name.data());
  }
  if (!is_valid_dtype(meta->dtype) || meta->rank > kMaxTensorDims) {
    return Status::error(StatusCode::kInvalidArgument,
                         "ModelReader::get_tensor: corrupt TensorMeta record");
  }

  Shape shape = (meta->rank == 0) ? Shape::scalar() : Shape();
  if (meta->rank > 0) {
    L3_ASSIGN_OR_RETURN(
        shape, Shape::from_dims(Span<const Index>(meta->dims, meta->rank)));
  }
  const Strides strides = Strides::contiguous_for(shape);

  Ptr<Storage> storage;
  L3_ASSIGN_OR_RETURN(
      storage,
      Storage::from_mmap(holder_, meta->data_offset, meta->data_bytes));

  return Tensor::from_storage(std::move(storage), shape, strides, meta->dtype, 0,
                              meta->quant_scale);
}

}  // namespace io
}  // namespace l3
