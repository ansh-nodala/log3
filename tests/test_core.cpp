// tests/test_core.cpp
//
// Comprehensive unit tests for all 22 headers and 8 source files in log3/core.
// Built automatically by CMake when L3_BUILD_TESTS=ON and executed via `ctest`.

#include <cstdio>
#include <cstring>

#include "log3/core/arena.h"
#include "log3/core/assert.h"
#include "log3/core/bit_ops.h"
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
#include "log3/core/rng.h"
#include "log3/core/scope_guard.h"
#include "log3/core/small_vector.h"
#include "log3/core/span.h"
#include "log3/core/status.h"
#include "log3/core/str_view.h"
#include "log3/core/thread_pool.h"
#include "log3/core/timer.h"
#include "log3/core/types.h"

namespace l3 {
namespace {

// -----------------------------------------------------------------------------
// 1. Memory & Alignment Math Tests
// -----------------------------------------------------------------------------
void test_memory_and_alignment() {
  L3_CHECK(is_power_of_two(1));
  L3_CHECK(is_power_of_two(64));
  L3_CHECK(!is_power_of_two(0));
  L3_CHECK(!is_power_of_two(63));

  L3_CHECK_EQ(align_up(0, 64), 0u);
  L3_CHECK_EQ(align_up(1, 64), 64u);
  L3_CHECK_EQ(align_up(64, 64), 64u);
  L3_CHECK_EQ(align_up(65, 64), 128u);
  L3_CHECK_EQ(align_down(127, 64), 64u);
  L3_CHECK_EQ(div_ceil(100, 32), 4u);
  L3_CHECK_EQ(div_ceil(128, 32), 4u);

  Size out = 0;
  L3_CHECK(checked_add(100, 200, &out) && out == 300u);
  L3_CHECK(!checked_add(static_cast<Size>(-1), 1, &out));
  L3_CHECK(checked_mul(1024, 1024, &out) && out == 1048576u);
  L3_CHECK(!checked_mul(static_cast<Size>(-1), 2, &out));

  const u64 mem_before = memory_stats().current_bytes;
  f32* arr = allocate_array<f32>(default_allocator(), 256);
  L3_CHECK(arr != nullptr);
  L3_CHECK(is_aligned(arr, kDefaultAlignment));
  L3_CHECK_GE(memory_stats().current_bytes, mem_before + 256 * sizeof(f32));
  deallocate_array(default_allocator(), arr, 256);
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 2. Arena Tests
// -----------------------------------------------------------------------------
void test_arena() {
  const u64 mem_before = memory_stats().current_bytes;
  {
    Arena arena(4096);
    L3_CHECK_EQ(arena.allocated_bytes(), 0u);
    L3_CHECK(arena.allocate(0) == nullptr);

    void* p1 = arena.allocate(1024, 64);
    void* p2 = arena.allocate(2048, 128);
    void* p3 = arena.allocate(8192, 64);
    L3_CHECK(p1 != nullptr && is_aligned(p1, 64));
    L3_CHECK(p2 != nullptr && is_aligned(p2, 128));
    L3_CHECK(p3 != nullptr && is_aligned(p3, 64));
    L3_CHECK_EQ(arena.allocated_bytes(), 1024u + 2048u + 8192u);

    Arena moved_arena(std::move(arena));
    L3_CHECK_EQ(arena.allocated_bytes(), 0u);
    L3_CHECK_EQ(moved_arena.allocated_bytes(), 1024u + 2048u + 8192u);

    moved_arena.reset();
    L3_CHECK_EQ(moved_arena.allocated_bytes(), 0u);
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 3. Status & Result<T> Tests
// -----------------------------------------------------------------------------
void test_status_and_result() {
  Status ok = Status::ok();
  L3_CHECK(ok.is_ok());
  L3_CHECK_EQ(ok.code(), StatusCode::kOk);
  L3_CHECK_EQ(std::strcmp(ok.message(), ""), 0);

  Status short_err = Status::error(StatusCode::kNotFound, "missing key");
  L3_CHECK(!short_err.is_ok());
  L3_CHECK(short_err.is(StatusCode::kNotFound));
  L3_CHECK_EQ(std::strcmp(short_err.message(), "missing key"), 0);

  Status long_err = Status::errorf(
      StatusCode::kShapeMismatch,
      "matmul shape mismatch: lhs=(%d, %d) vs rhs=(%d, %d)", 128, 512, 256, 768);
  L3_CHECK(long_err.is(StatusCode::kShapeMismatch));
  Status cloned = long_err.clone();
  L3_CHECK(cloned.is(StatusCode::kShapeMismatch));
  L3_CHECK_EQ(std::strcmp(cloned.message(), long_err.message()), 0);

  Result<i64> r_val(42);
  L3_CHECK(r_val.is_ok());
  L3_CHECK_EQ(*r_val, 42);
  Result<i64> r_moved(std::move(r_val));
  L3_CHECK(r_moved.is_ok());
  L3_CHECK_EQ(r_moved.take_value(), 42);

  Result<i64> r_err(Status::error(StatusCode::kOutOfRange, "index out of bounds"));
  L3_CHECK(!r_err.is_ok());
  Status extracted = r_err.take_status();
  L3_CHECK(extracted.is(StatusCode::kOutOfRange));
}

// -----------------------------------------------------------------------------
// 4. Intrusive RefCounted & Ptr<T> Tests
// -----------------------------------------------------------------------------
struct BaseNode : public RefCounted {
  explicit BaseNode(int* dtor_count) noexcept : dtor_count(dtor_count) {}
  int* dtor_count;
};

struct DerivedNode : public BaseNode {
  DerivedNode(int* dtor_count, i64 payload) noexcept
      : BaseNode(dtor_count), payload(payload) {}
  ~DerivedNode() noexcept { ++(*dtor_count); }
  i64 payload;
};

void test_ptr() {
  const u64 mem_before = memory_stats().current_bytes;
  int destroyed = 0;
  {
    auto res = make_ptr<DerivedNode>(&destroyed, 999);
    L3_CHECK_OK(res.status());
    Ptr<DerivedNode> d = res.take_value();
    L3_CHECK_EQ(d.use_count(), 1u);
    L3_CHECK_EQ(d->payload, 999);

    Ptr<BaseNode> b = std::move(d);
    L3_CHECK(d == nullptr);
    L3_CHECK_EQ(b.use_count(), 1u);
  }
  L3_CHECK_EQ(destroyed, 1);
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 5. Span<T> & SmallVec<T, N> Tests
// -----------------------------------------------------------------------------
void test_span_and_small_vec() {
  const u64 mem_before = memory_stats().current_bytes;
  {
    SmallVec<i32, 4> vec;
    L3_CHECK(vec.is_inline());
    for (i32 i = 0; i < 4; ++i) {
      L3_CHECK_OK(vec.push_back(i * 10));
    }
    L3_CHECK(vec.is_inline());

    L3_CHECK_OK(vec.push_back(vec[1]));
    L3_CHECK(!vec.is_inline());
    L3_CHECK_EQ(vec.size(), 5);
    L3_CHECK_EQ(vec.back(), 10);

    L3_CHECK_OK(vec.insert(2, 99));
    L3_CHECK_EQ(vec[2], 99);
    vec.erase(2);
    L3_CHECK_EQ(vec[2], 20);

    Span<const i32> sp = vec.span();
    L3_CHECK_EQ(sp.first(2).size(), 2);
    L3_CHECK_EQ(sp.last(2)[1], 10);
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 6. Half-Precision (f16 / bf16) & DType / BitNet Sub-Byte Tests
// -----------------------------------------------------------------------------
void test_half_and_dtype() {
  L3_CHECK_EQ(f16_to_f32(kF16Zero), 0.0f);
  L3_CHECK_EQ(f16_to_f32(kF16One), 1.0f);
  L3_CHECK_EQ(f16_to_f32(kF16NegOne), -1.0f);
  L3_CHECK(is_inf(f16_to_f32(kF16Infinity)));
  L3_CHECK(is_nan(f16_to_f32(kF16NaN)));

  const f32 sub_f32 = f32_from_bits(0x35800000u);
  const f16 sub_f16 = f32_to_f16(sub_f32);
  L3_CHECK_EQ(f16_to_f32(sub_f16), sub_f32);

  const f32 snan = f32_from_bits(0x7f800001u);
  const bf16 qnan_bf16 = f32_to_bf16(snan);
  L3_CHECK(is_nan(qnan_bf16));

  L3_CHECK_EQ(dtype_of<f16>(), DType::kFloat16);
  L3_CHECK_EQ(dtype_of<bf16>(), DType::kBFloat16);
  L3_CHECK_EQ(dtype_of<i8>(), DType::kInt8);
  L3_CHECK(is_subbyte_dtype(DType::kTernary2));
  L3_CHECK(is_subbyte_dtype(DType::kInt4));
  L3_CHECK(is_subbyte_dtype(DType::kUInt4));

  Size bytes = 0;
  L3_CHECK(!checked_dtype_bytes(DType::kTernary2, 5, &bytes).is_ok());
  L3_CHECK_OK(checked_dtype_bytes_padded(DType::kTernary2, 5, &bytes));
  L3_CHECK_EQ(bytes, 2u);
}

// -----------------------------------------------------------------------------
// 7. StrView, FnRef, FlatMap, MMapFile, and ThreadPool Tests
// -----------------------------------------------------------------------------
void test_new_core_primitives() {
  const u64 mem_before = memory_stats().current_bytes;
  {
    StrView sv("  bitnet.b1_58.layer.0  ");
    L3_CHECK(sv.trim() == StrView("bitnet.b1_58.layer.0"));
    L3_CHECK_EQ(sv.trim().find("b1_58"), 7);

    int acc = 10;
    auto add_fn = [&](int v) noexcept { acc += v; };
    FnRef<void(int)> fn_ref(add_fn);
    fn_ref(15);
    L3_CHECK_EQ(acc, 25);

    StrMap<i32> sym_table;
    L3_CHECK_OK(sym_table.put("wq", 1));
    L3_CHECK_OK(sym_table.put("wk", 2));
    L3_CHECK_OK(sym_table.put("wv", 3));
    L3_CHECK_OK(sym_table.put("wo", 4));
    L3_CHECK(sym_table.erase("wk"));
    L3_CHECK(sym_table.find("wk") == nullptr);
    L3_CHECK_EQ(*sym_table.find("wv"), 3);

    const char* path = "log3_ctest_mmap.tmp";
    {
      std::FILE* f = std::fopen(path, "wb");
      L3_CHECK(f != nullptr);
      const char data[] = "0123456789ABCDEF";
      std::fwrite(data, 1, 16, f);
      std::fclose(f);
    }
    auto mres = MMapFile::open(path, true);
    L3_CHECK_OK(mres.status());
    MMapFile mf = mres.take_value();
    L3_CHECK_EQ(mf.size(), 16u);
    auto sres = mf.slice(10, 6);
    L3_CHECK_OK(sres.status());
    L3_CHECK(StrView(reinterpret_cast<const char*>(sres->data()), sres->size()) ==
             StrView("ABCDEF"));
    mf.close();
    std::remove(path);

    auto pres = ThreadPool::create(4);
    L3_CHECK_OK(pres.status());
    ThreadPool pool = pres.take_value();

    SmallVec<i32, 8> buf;
    L3_CHECK_OK(buf.resize(2048, 0));
    pool.parallel_for(0, 2048, 64, [&](Index /*tid*/, Index b, Index e) noexcept {
      pool.parallel_for(b, e, 16, [&](Index /*inner_tid*/, Index ib, Index ie) noexcept {
        for (Index i = ib; i < ie; ++i) {
          buf[i] = static_cast<i32>(i * 2);
        }
      });
    });

    for (Index i = 0; i < 2048; ++i) {
      L3_CHECK_EQ(buf[i], static_cast<i32>(i * 2));
    }
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 8. BitOps, Rng (Xoshiro256++), Timer, and ScopeGuard (L3_DEFER) Tests
// -----------------------------------------------------------------------------
void test_bitops_rng_timer_defer() {
  // BitOps
  L3_CHECK_EQ(ctz32(8u), 3);
  L3_CHECK_EQ(ctz64(1ULL << 42), 42);
  L3_CHECK_EQ(clz32(1u), 31);
  L3_CHECK_EQ(clz64(1ULL), 63);
  L3_CHECK_EQ(popcount32(0xF0F0F0F0u), 16);
  L3_CHECK_EQ(popcount64(0xAAAAAAAAAAAAAAAAULL), 32);
  L3_CHECK_EQ(next_power_of_two(0), 1ULL);
  L3_CHECK_EQ(next_power_of_two(17), 32ULL);
  L3_CHECK_EQ(next_power_of_two(64), 64ULL);
  L3_CHECK_EQ(bswap16(0x1234u), 0x3412u);
  L3_CHECK_EQ(bswap32(0x12345678u), 0x78563412u);

  // Unaligned little-endian read/write
  byte raw_buf[16]{};
  write_le<u32>(raw_buf + 3, 0xCAFEBABEu);
  write_le<u64>(raw_buf + 7, 0x0102030405060708ULL);
  L3_CHECK_EQ(read_le<u32>(raw_buf + 3), 0xCAFEBABEu);
  L3_CHECK_EQ(read_le<u64>(raw_buf + 7), 0x0102030405060708ULL);

  // Deterministic Rng (Xoshiro256++)
  Rng rng1(12345ULL);
  Rng rng2(12345ULL);
  for (int i = 0; i < 100; ++i) {
    L3_CHECK_EQ(rng1.next_u64(), rng2.next_u64());
  }
  for (int i = 0; i < 100; ++i) {
    const f32 u = rng1.uniform_f32();
    L3_CHECK_GE(u, 0.0f);
    L3_CHECK_LT(u, 1.0f);
    const u32 b = rng1.bounded_u32(37);
    L3_CHECK_LT(b, 37u);
  }
  const f32 g1 = rng1.normal_f32(0.0f, 1.0f);
  const f32 g2 = rng1.normal_f32(0.0f, 1.0f);
  L3_CHECK(is_finite(g1) && is_finite(g2));

  // Substream splitting via jump()
  Rng stream_a = rng1.split(1);
  Rng stream_b = rng1.split(2);
  L3_CHECK_NE(stream_a.next_u64(), stream_b.next_u64());

  // Stopwatch & monotonic timer
  Stopwatch sw;
  u64 t0 = monotonic_now_ns();
  u64 t1 = monotonic_now_ns();
  L3_CHECK_GE(t1, t0);
  L3_CHECK_GE(sw.elapsed_ns(), 0ULL);
  L3_CHECK_EQ(compute_gbps(1000000000ULL, 500000000ULL), 2.0);

  // ScopeGuard / L3_DEFER
  int cleanup_calls = 0;
  {
    L3_DEFER { ++cleanup_calls; };
    L3_CHECK_EQ(cleanup_calls, 0);
  }
  L3_CHECK_EQ(cleanup_calls, 1);

  {
    auto dismissable = ScopeGuard([&]() noexcept { ++cleanup_calls; });
    dismissable.dismiss();
  }
  L3_CHECK_EQ(cleanup_calls, 1);
}

}  // namespace
}  // namespace l3

int main() {
  l3::set_log_level(l3::LogLevel::kInfo);
  L3_LOG_INFO("Running tests/test_core.cpp...");

  l3::test_memory_and_alignment();
  l3::test_arena();
  l3::test_status_and_result();
  l3::test_ptr();
  l3::test_span_and_small_vec();
  l3::test_half_and_dtype();
  l3::test_new_core_primitives();
  l3::test_bitops_rng_timer_defer();

  L3_LOG_INFO("tests/test_core.cpp: ALL 8 SUITES PASSED (0 bytes leaked).");
  return 0;
}
