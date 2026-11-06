// main.cpp
//
// End-to-end verification and smoke-test suite for the entire log3/core
// foundation library (18 core headers + 8 implementation files).
//
// Modules verified in this file:
//   1. Logging & runtime levels       (log.h)
//   2. CPU ISA & cache topology       (cpu_features.h)
//   3. Status & Result<T> propagation (status.h, assert.h)
//   4. Aligned memory & Arena         (memory.h, arena.h)
//   5. Intrusive ref-counting Ptr<T>  (ptr.h)
//   6. Contiguous views & SmallVec    (span.h, small_vector.h)
//   7. Half-precision f16 & bf16      (half.h)
//   8. DType metadata & BitNet t2/i4  (dtype.h)
//   9. Zero-allocation StrView & hash (str_view.h)
//  10. Zero-allocation FnRef<Sig>     (fn_ref.h)
//  11. Open-addressed FlatMap/StrMap  (flat_map.h)
//  12. Zero-copy MMapFile             (mmap.h)
//  13. Hybrid-spin ThreadPool         (thread_pool.h)
//  14. Zero-leak memory audit         (memory_stats)

#include <cstdio>
#include <cstring>

#include "log3/core/arena.h"
#include "log3/core/assert.h"
#include "log3/core/cpu_features.h"
#include "log3/core/dtype.h"
#include "log3/core/flat_map.h"
#include "log3/core/fn_ref.h"
#include "log3/core/half.h"
#include "log3/core/log.h"
#include "log3/core/memory.h"
#include "log3/core/mmap.h"
#include "log3/core/platform.h"
#include "log3/core/ptr.h"
#include "log3/core/small_vector.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/str_view.h"
#include "log3/core/thread_pool.h"
#include "log3/core/types.h"

namespace {

// -----------------------------------------------------------------------------
// Helper class for intrusive RefCounted + Ptr<T> verification
// -----------------------------------------------------------------------------
class MockTensorBuffer final : public l3::RefCounted {
 public:
  explicit MockTensorBuffer(l3::Index numel, int* alive_counter) noexcept
      : numel_(numel), alive_counter_(alive_counter) {
    if (alive_counter_ != nullptr) {
      ++(*alive_counter_);
    }
  }

  ~MockTensorBuffer() noexcept {
    if (alive_counter_ != nullptr) {
      --(*alive_counter_);
    }
  }

  l3::Index numel() const noexcept { return numel_; }

 private:
  l3::Index numel_;
  int* alive_counter_;
};

// -----------------------------------------------------------------------------
// 1. Status & Result<T> propagation test
// -----------------------------------------------------------------------------
l3::Result<l3::Index> compute_checked_volume(
    l3::Span<const l3::Index> dims) noexcept {
  if (dims.empty()) {
    return l3::Status::error(l3::StatusCode::kInvalidArgument,
                             "shape cannot be empty");
  }
  l3::Size prod = 1;
  for (l3::Index d : dims) {
    if (d <= 0) {
      return l3::Status::errorf(l3::StatusCode::kInvalidArgument,
                                "dimension must be positive, got %lld",
                                static_cast<long long>(d));
    }
    if (!l3::checked_mul(prod, static_cast<l3::Size>(d), &prod) ||
        prod > static_cast<l3::Size>(l3::kMaxIndex)) {
      return l3::Status::error(l3::StatusCode::kOutOfRange,
                               "tensor volume overflow");
    }
  }
  return static_cast<l3::Index>(prod);
}

l3::Status test_status_and_result() noexcept {
  L3_LOG_INFO("[1/10] Testing Status & Result<T> propagation...");

  L3_RETURN_IF_ERROR(l3::Status::ok());

  const l3::Index dims[3] = {2, 512, 768};
  l3::Index volume = 0;
  L3_ASSIGN_OR_RETURN(volume, compute_checked_volume(l3::Span<const l3::Index>(dims)));
  L3_CHECK_EQ(volume, 2LL * 512LL * 768LL);

  // Verify error path returns kInvalidArgument cleanly
  const l3::Index bad_dims[2] = {64, -1};
  auto bad_res = compute_checked_volume(l3::Span<const l3::Index>(bad_dims));
  L3_CHECK(!bad_res.is_ok());
  L3_CHECK(bad_res.status().is(l3::StatusCode::kInvalidArgument));

  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 2. SmallVec & Span test
// -----------------------------------------------------------------------------
l3::Status test_small_vec_and_span() noexcept {
  L3_LOG_INFO("[2/10] Testing SmallVec<T, N> and Span<T>...");

  l3::SmallVec<l3::Index, 8> shape;
  L3_RETURN_IF_ERROR(shape.push_back(1));
  L3_RETURN_IF_ERROR(shape.push_back(512));
  L3_RETURN_IF_ERROR(shape.push_back(768));

  L3_CHECK(shape.is_inline());
  L3_CHECK_EQ(shape.size(), 3);
  L3_CHECK_EQ(shape[0], 1);
  L3_CHECK_EQ(shape[1], 512);
  L3_CHECK_EQ(shape[2], 768);

  // View as Span and slice subspan
  l3::Span<const l3::Index> view(shape);
  l3::Span<const l3::Index> tail = view.subspan(1, 2);
  L3_CHECK_EQ(tail.size(), 2);
  L3_CHECK_EQ(tail[0], 512);
  L3_CHECK_EQ(tail[1], 768);

  // Force transition from inline storage (N=8) to heap storage
  for (l3::Index i = 0; i < 16; ++i) {
    L3_RETURN_IF_ERROR(shape.push_back(i + 10));
  }
  L3_CHECK(!shape.is_inline());
  L3_CHECK_EQ(shape.size(), 19);

  l3::SmallVec<l3::Index, 8> cloned;
  L3_ASSIGN_OR_RETURN(cloned, shape.clone());
  L3_CHECK(cloned == shape);

  L3_LOG_INFO("       SmallVec rank=%lld (heap transition & clone verified)",
              static_cast<long long>(shape.size()));
  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 3. Intrusive Ptr<T> & RefCounted test
// -----------------------------------------------------------------------------
l3::Status test_intrusive_ptr() noexcept {
  L3_LOG_INFO("[3/10] Testing intrusive RefCounted & Ptr<T>...");

  int alive_objects = 0;
  {
    l3::Ptr<MockTensorBuffer> p1;
    L3_ASSIGN_OR_RETURN(p1, l3::make_ptr<MockTensorBuffer>(4096, &alive_objects));
    L3_CHECK_EQ(alive_objects, 1);
    L3_CHECK(p1->is_unique());
    L3_CHECK_EQ(p1.use_count(), 1u);
    L3_CHECK_EQ(p1->numel(), 4096);

    {
      l3::Ptr<MockTensorBuffer> p2 = p1;
      L3_CHECK(!p1->is_unique());
      L3_CHECK_EQ(p1.use_count(), 2u);
      L3_CHECK_EQ(alive_objects, 1);
    }

    L3_CHECK(p1->is_unique());
    L3_CHECK_EQ(p1.use_count(), 1u);
  }
  L3_CHECK_EQ(alive_objects, 0);

  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 4. Half-precision (f16 / bf16) & DType / BitNet packing test
// -----------------------------------------------------------------------------
l3::Status test_dtype_and_half() noexcept {
  L3_LOG_INFO("[4/10] Testing f16/bf16 conversions & BitNet (t2/i4/u4) packing...");

  // f16 and bf16 round-trip checks
  const l3::f32 sample_vals[4] = {0.0f, -1.0f, 3.5f, 42.25f};
  l3::f16 f16_buf[4]{};
  l3::bf16 bf16_buf[4]{};
  l3::f32 roundtrip_f16[4]{};
  l3::f32 roundtrip_bf16[4]{};

  l3::convert_f32_to_f16(sample_vals, f16_buf, 4);
  l3::convert_f16_to_f32(f16_buf, roundtrip_f16, 4);
  l3::convert_f32_to_bf16(sample_vals, bf16_buf, 4);
  l3::convert_bf16_to_f32(bf16_buf, roundtrip_bf16, 4);

  for (int i = 0; i < 4; ++i) {
    L3_CHECK_EQ(roundtrip_f16[i], sample_vals[i]);
    L3_CHECK_EQ(roundtrip_bf16[i], sample_vals[i]);
  }
  L3_CHECK(l3::is_nan(l3::kF16NaN));
  L3_CHECK(l3::is_inf(l3::kBF16Infinity));

  // BitNet b1.58 ternary (t2: 4 weights in 1 byte)
  const l3::i8 t2_orig[4] = {-1, 0, 1, -1};
  const l3::u8 t2_packed =
      l3::pack_t2_x4(t2_orig[0], t2_orig[1], t2_orig[2], t2_orig[3]);
  l3::i8 t2_unpacked[4]{};
  l3::unpack_t2_x4(t2_packed, t2_unpacked);

  for (int i = 0; i < 4; ++i) {
    L3_CHECK_EQ(t2_orig[i], t2_unpacked[i]);
  }

  // Signed 4-bit (i4: 2 weights in 1 byte) & Unsigned 4-bit (u4)
  const l3::u8 i4_packed = l3::pack_i4_x2(-8, 7);
  l3::i8 i4_unpacked[2]{};
  l3::unpack_i4_x2(i4_packed, i4_unpacked);
  L3_CHECK_EQ(i4_unpacked[0], -8);
  L3_CHECK_EQ(i4_unpacked[1], 7);

  const l3::u8 u4_packed = l3::pack_u4_x2(3u, 15u);
  l3::u8 u4_unpacked[2]{};
  l3::unpack_u4_x2(u4_packed, u4_unpacked);
  L3_CHECK_EQ(u4_unpacked[0], 3u);
  L3_CHECK_EQ(u4_unpacked[1], 15u);

  // Checked byte calculation for 1024 ternary weights -> 256 bytes
  l3::Size t2_bytes = 0;
  L3_RETURN_IF_ERROR(
      l3::checked_dtype_bytes(l3::DType::kTernary2, 1024, &t2_bytes));
  L3_CHECK_EQ(t2_bytes, 256u);

  l3::DType parsed_dt = l3::DType::kFloat32;
  L3_ASSIGN_OR_RETURN(parsed_dt, l3::dtype_from_name("bitnet158"));
  L3_CHECK_EQ(parsed_dt, l3::DType::kTernary2);

  L3_LOG_INFO(
      "       BitNet t2: [%d, %d, %d, %d] -> 0x%02X -> [%d, %d, %d, %d] (1024 weights = %zu B)",
      t2_orig[0], t2_orig[1], t2_orig[2], t2_orig[3],
      static_cast<unsigned int>(t2_packed),
      t2_unpacked[0], t2_unpacked[1], t2_unpacked[2], t2_unpacked[3],
      t2_bytes);
  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 5. Arena allocator test
// -----------------------------------------------------------------------------
l3::Status test_arena() noexcept {
  L3_LOG_INFO("[5/10] Testing Arena bump allocator...");

  l3::Arena arena(4096);
  void* block1 = arena.allocate(128, 64);
  void* block2 = arena.allocate(256, 64);
  L3_CHECK(block1 != nullptr);
  L3_CHECK(block2 != nullptr);
  L3_CHECK(l3::is_aligned(block1, 64));
  L3_CHECK(l3::is_aligned(block2, 64));
  L3_CHECK_EQ(arena.allocated_bytes(), 384u);

  L3_LOG_INFO("       Arena allocated blocks at %p and %p (used: %zu B)",
              block1, block2, arena.allocated_bytes());

  arena.reset();
  L3_CHECK_EQ(arena.allocated_bytes(), 0u);
  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 6. StrView & 64-bit Wyhash/SplitMix64 hashing test
// -----------------------------------------------------------------------------
l3::Status test_str_view() noexcept {
  L3_LOG_INFO("[6/10] Testing StrView & 64-bit fast hashing...");

  l3::StrView raw("   blk.0.attn_q.weight\r\n ");
  l3::StrView clean = raw.trim();
  L3_CHECK(clean == l3::StrView("blk.0.attn_q.weight"));
  L3_CHECK(clean.starts_with("blk.0."));
  L3_CHECK(clean.ends_with(".weight"));
  L3_CHECK(clean.contains("attn_q"));
  L3_CHECK_EQ(clean.find('.'), 3);
  L3_CHECK_EQ(clean.rfind('.'), 12);
  L3_CHECK(clean.substr(6, 6) == l3::StrView("attn_q"));

  const l3::u64 h1 = l3::hash_str(clean);
  const l3::u64 h2 = l3::hash_str("blk.0.attn_k.weight");
  L3_CHECK_NE(h1, h2);
  L3_CHECK_EQ(h1, l3::hash_str("blk.0.attn_q.weight"));

  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 7. FnRef<Sig> callable reference test
// -----------------------------------------------------------------------------
int scale_add_fn(int x, int y) noexcept { return x * 2 + y; }

l3::Status test_fn_ref() noexcept {
  L3_LOG_INFO("[7/10] Testing zero-allocation FnRef<R(Args...)>...");

  int bias = 5;
  auto lambda = [&](int a, int b) noexcept -> int { return a * b + bias; };

  l3::FnRef<int(int, int)> ref_lambda(lambda);
  L3_CHECK(ref_lambda.is_valid());
  L3_CHECK_EQ(ref_lambda(4, 6), 29);

  l3::FnRef<int(int, int)> ref_fn(&scale_add_fn);
  L3_CHECK(ref_fn.is_valid());
  L3_CHECK_EQ(ref_fn(4, 6), 14);

  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 8. FlatMap<K, V> & StrMap<V> open-addressed hash table test
// -----------------------------------------------------------------------------
l3::Status test_flat_map() noexcept {
  L3_LOG_INFO("[8/10] Testing FlatMap<K, V> & StrMap<V> (open addressing)...");

  l3::StrMap<l3::Index> tensor_map;
  L3_RETURN_IF_ERROR(tensor_map.reserve(32));

  bool inserted = false;
  L3_RETURN_IF_ERROR(tensor_map.put("blk.0.attn_q.weight", 0, &inserted));
  L3_CHECK(inserted);
  L3_RETURN_IF_ERROR(tensor_map.put("blk.0.attn_k.weight", 1, &inserted));
  L3_CHECK(inserted);
  L3_RETURN_IF_ERROR(tensor_map.put("blk.0.attn_v.weight", 2, &inserted));
  L3_CHECK(inserted);

  // Update existing key
  L3_RETURN_IF_ERROR(tensor_map.put("blk.0.attn_k.weight", 10, &inserted));
  L3_CHECK(!inserted);
  L3_CHECK_EQ(tensor_map.size(), 3);

  const l3::Index* found = tensor_map.find("blk.0.attn_k.weight");
  L3_CHECK(found != nullptr);
  L3_CHECK_EQ(*found, 10);

  // Backward-shift deletion stress test
  l3::FlatMap<l3::i64, l3::i64> id_map;
  for (l3::i64 i = 0; i < 512; ++i) {
    L3_RETURN_IF_ERROR(id_map.put(i * 13 + 7, i * 100));
  }
  L3_CHECK_EQ(id_map.size(), 512);

  for (l3::i64 i = 0; i < 512; i += 2) {
    L3_CHECK(id_map.erase(i * 13 + 7));
  }
  L3_CHECK_EQ(id_map.size(), 256);

  for (l3::i64 i = 0; i < 512; ++i) {
    const l3::i64* v = id_map.find(i * 13 + 7);
    if (i % 2 == 0) {
      L3_CHECK(v == nullptr);
    } else {
      L3_CHECK(v != nullptr);
      L3_CHECK_EQ(*v, i * 100);
    }
  }

  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 9. MMapFile zero-copy memory-mapped file test
// -----------------------------------------------------------------------------
l3::Status test_mmap_file() noexcept {
  L3_LOG_INFO("[9/10] Testing zero-copy MMapFile...");

  const char* tmp_file = "log3_test_weights.bin";
  {
    std::FILE* fp = std::fopen(tmp_file, "wb");
    if (fp == nullptr) {
      return l3::Status::error(l3::StatusCode::kIoError,
                               "failed to create temporary test file");
    }
    const char header[] = "L3GGUF_BITNET_B158_TENSOR_PAYLOAD_0123456789";
    std::fwrite(header, 1, sizeof(header) - 1, fp);
    std::fclose(fp);
  }

  l3::MMapFile mapped;
  L3_ASSIGN_OR_RETURN(mapped, l3::MMapFile::open(tmp_file, /*prefetch=*/true));
  L3_CHECK(mapped.is_open());
  L3_CHECK_EQ(mapped.size(), 44u);

  L3_RETURN_IF_ERROR(mapped.advise(l3::MMapAdvice::kSequential));

  l3::ConstByteSpan sub;
  L3_ASSIGN_OR_RETURN(sub, mapped.slice(7, 11));
  l3::StrView magic(reinterpret_cast<const char*>(sub.data()), sub.size());
  L3_CHECK(magic == l3::StrView("BITNET_B158"));

  mapped.close();
  std::remove(tmp_file);
  return l3::Status::ok();
}

// -----------------------------------------------------------------------------
// 10. ThreadPool parallel_for test
// -----------------------------------------------------------------------------
l3::Status test_thread_pool() noexcept {
  L3_LOG_INFO("[10/10] Testing hybrid-spin ThreadPool & parallel_for...");

  l3::ThreadPool pool;
  L3_ASSIGN_OR_RETURN(pool, l3::ThreadPool::create(4));
  L3_CHECK_EQ(pool.num_threads(), 4);

  constexpr l3::Index kCount = 8192;
  l3::SmallVec<l3::i64, 8> output;
  L3_RETURN_IF_ERROR(output.resize(kCount, 0));

  // Simulate parallel BitNet row dot-product across worker threads
  pool.parallel_for(0, kCount, 128,
                    [&](l3::Index tid, l3::Index begin, l3::Index end) noexcept {
                      L3_CHECK_GE(tid, 0);
                      L3_CHECK_LT(tid, 4);
                      for (l3::Index i = begin; i < end; ++i) {
                        output[i] = (i * 7) - 3;
                      }
                    });

  for (l3::Index i = 0; i < kCount; ++i) {
    L3_CHECK_EQ(output[i], (i * 7) - 3);
  }

  L3_LOG_INFO("        ThreadPool (%d threads) verified over %lld items",
              pool.num_threads(), static_cast<long long>(kCount));
  return l3::Status::ok();
}

l3::Status run_all_core_checks() noexcept {
  L3_RETURN_IF_ERROR(test_status_and_result());
  L3_RETURN_IF_ERROR(test_small_vec_and_span());
  L3_RETURN_IF_ERROR(test_intrusive_ptr());
  L3_RETURN_IF_ERROR(test_dtype_and_half());
  L3_RETURN_IF_ERROR(test_arena());
  L3_RETURN_IF_ERROR(test_str_view());
  L3_RETURN_IF_ERROR(test_fn_ref());
  L3_RETURN_IF_ERROR(test_flat_map());
  L3_RETURN_IF_ERROR(test_mmap_file());
  L3_RETURN_IF_ERROR(test_thread_pool());
  return l3::Status::ok();
}

}  // namespace

int main() {
  l3::set_log_level(l3::LogLevel::kTrace);
  l3::reset_peak_memory();

  const l3::u64 baseline_bytes = l3::memory_stats().current_bytes;

  L3_LOG_INFO("=== Starting log3 Core Foundation Test Suite ===");

  const auto& cpu = l3::cpu_features();
  L3_LOG_INFO(
      "CPU Info: logical_cores = %d, physical_cores = %d, cache_line = %zu B",
      cpu.logical_cores, cpu.physical_cores, cpu.cache_line_size);
  L3_LOG_INFO(
      "SIMD Support -> AVX2: %s, AVX-512F: %s, VNNI: %s, NEON: %s, DotProd: %s, I8MM: %s",
      cpu.has_avx2 ? "yes" : "no",
      cpu.has_avx512f ? "yes" : "no",
      cpu.has_avx512_vnni ? "yes" : "no",
      cpu.has_neon ? "yes" : "no",
      cpu.has_neon_dotprod ? "yes" : "no",
      cpu.has_neon_i8mm ? "yes" : "no");

  const l3::Status st = run_all_core_checks();
  if (!st.is_ok()) {
    L3_LOG_ERROR("Core test suite failed [%s]: %s",
                 l3::status_code_name(st.code()), st.message());
    return 1;
  }

  const auto stats = l3::memory_stats();
  const l3::u64 leaked_bytes = stats.current_bytes - baseline_bytes;

  L3_LOG_INFO(
      "Memory Stats -> current: %llu B (leaked: %llu B), peak: %llu B, "
      "allocs: %llu, deallocs: %llu",
      static_cast<unsigned long long>(stats.current_bytes),
      static_cast<unsigned long long>(leaked_bytes),
      static_cast<unsigned long long>(stats.peak_bytes),
      static_cast<unsigned long long>(stats.total_allocations),
      static_cast<unsigned long long>(stats.total_deallocations));

  L3_CHECK_EQ(leaked_bytes, 0ULL);

  L3_LOG_INFO("=== All 10 log3/core module suites passed with 0 bytes leaked! ===");
  return 0;
}
