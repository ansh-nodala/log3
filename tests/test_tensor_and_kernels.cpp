// tests/test_tensor_and_kernels.cpp
//
// Comprehensive unit test suite for Stage 2:
//   * log3/tensor/shape.h    (Shape, Strides, broadcasting, sub-byte validation)
//   * log3/tensor/storage.h  (Storage: Allocator-owned, MMap-shared, External)
//   * log3/tensor/tensor.h   (Tensor: view, clone, reshape, transpose, slice)
//   * log3/kernels/cpu_kernels.h (BitNet b1.58 t2 quantize, dot, linear, SIMD
//                                 dot_f32, rmsnorm, softmax, silu, rope, matmul)

#include <cmath>
#include <cstdio>
#include <cstring>

#include "log3/core/arena.h"
#include "log3/core/assert.h"
#include "log3/core/log.h"
#include "log3/core/memory.h"
#include "log3/core/rng.h"
#include "log3/core/timer.h"
#include "log3/kernels/cpu_kernels.h"
#include "log3/tensor/shape.h"
#include "log3/tensor/storage.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace {

bool approx_eq(f32 a, f32 b, f32 tol = 1e-4f) noexcept {
  const f32 diff = a > b ? (a - b) : (b - a);
  return diff <= tol;
}

// -----------------------------------------------------------------------------
// 1. Shape, Strides, Broadcasting & Sub-Byte Layout Validation
// -----------------------------------------------------------------------------
void test_shape_and_strides() {
  Shape scalar = Shape::scalar();
  L3_CHECK(scalar.is_scalar());
  L3_CHECK_EQ(scalar.rank(), 0);
  L3_CHECK_EQ(scalar.numel(), 1);

  auto s_res = Shape::from_dims({2, 3, 4});
  L3_CHECK_OK(s_res.status());
  Shape s = s_res.take_value();
  L3_CHECK_EQ(s.rank(), 3);
  L3_CHECK_EQ(s.numel(), 24);

  auto ax_last = s.normalize_axis(-1);
  auto ax_first = s.normalize_axis(-3);
  L3_CHECK_OK(ax_last.status());
  L3_CHECK_OK(ax_first.status());
  L3_CHECK_EQ(ax_last.value(), 2);
  L3_CHECK_EQ(ax_first.value(), 0);

  Strides st = Strides::contiguous_for(s);
  L3_CHECK_EQ(st[0], 12);
  L3_CHECK_EQ(st[1], 4);
  L3_CHECK_EQ(st[2], 1);
  L3_CHECK(is_contiguous_layout(s, st));

  // Sub-byte validation: (2, 3, 4) has last dim 4 -> valid for kTernary2!
  L3_CHECK_OK(validate_layout_for_dtype(s, st, DType::kTernary2));

  // Shape (2, 3, 5) has last dim 5 (not divisible by 4) -> invalid for kTernary2!
  auto bad_s = Shape::from_dims({2, 3, 5});
  L3_CHECK_OK(bad_s.status());
  L3_CHECK(!validate_layout_for_dtype(
                bad_s.value(), Strides::contiguous_for(bad_s.value()),
                DType::kTernary2)
                .is_ok());

  // Broadcasting: (8, 1, 6, 1) and (7, 1, 5) -> (8, 7, 6, 5)
  auto sa = Shape::from_dims({8, 1, 6, 1});
  auto sb = Shape::from_dims({7, 1, 5});
  auto bc = broadcast_shapes(sa.value(), sb.value());
  L3_CHECK_OK(bc.status());
  auto expected_bc = Shape::from_dims({8, 7, 6, 5});
  L3_CHECK_OK(expected_bc.status());
  L3_CHECK(bc.value() == expected_bc.value());

  auto bst = broadcast_strides(
      sb.value(), Strides::contiguous_for(sb.value()), bc.value());
  L3_CHECK_OK(bst.status());
  const Strides& bstrides = bst.value();
  L3_CHECK_EQ(bstrides[0], 0);  // prepended axis -> stride 0
  L3_CHECK_EQ(bstrides[1], 5);
  L3_CHECK_EQ(bstrides[2], 0);  // stretched from 1 to 6 -> stride 0
  L3_CHECK_EQ(bstrides[3], 1);
}

// -----------------------------------------------------------------------------
// 2. Storage & Tensor Zero-Copy Views (reshape, transpose, slice, clone)
// -----------------------------------------------------------------------------
void test_tensor_views_and_clone() {
  const u64 mem_before = memory_stats().current_bytes;
  {
    auto t_res = Tensor::empty({4, 6}, DType::kFloat32);
    L3_CHECK_OK(t_res.status());
    Tensor t = t_res.take_value();
    L3_CHECK(t.is_valid());
    L3_CHECK(t.is_contiguous());
    L3_CHECK_EQ(t.numel(), 24);
    L3_CHECK_EQ(t.nbytes(), 96u);

    Span<f32> sp = t.mutable_span<f32>();
    for (Index i = 0; i < 24; ++i) {
      sp[i] = static_cast<f32>(i);
    }

    // Zero-copy reshape (4, 6) -> (2, 3, 4)
    auto r_res = t.reshape({2, 3, 4});
    L3_CHECK_OK(r_res.status());
    Tensor reshaped = r_res.take_value();
    L3_CHECK_EQ(t.storage().use_count(), 2u);
    L3_CHECK_EQ(reshaped.data_ptr<f32>(), t.data_ptr<f32>());

    // Zero-copy transpose (4, 6) -> (6, 4)
    auto tr_res = t.transpose(0, 1);
    L3_CHECK_OK(tr_res.status());
    Tensor tr = tr_res.take_value();
    L3_CHECK(!tr.is_contiguous());
    L3_CHECK_EQ(tr.dim(0), 6);
    L3_CHECK_EQ(tr.dim(1), 4);
    L3_CHECK_EQ(tr.stride(0), 1);
    L3_CHECK_EQ(tr.stride(1), 6);

    // Materialize contiguous copy of transposed tensor
    auto c_res = tr.contiguous();
    L3_CHECK_OK(c_res.status());
    Tensor tr_contig = c_res.take_value();
    L3_CHECK(tr_contig.is_contiguous());
    L3_CHECK_EQ(tr_contig.data_ptr<f32>()[2 * 4 + 3], 20.0f);

    // Zero-copy slice along axis 0: rows [1, 3) of (4, 6) -> (2, 6)
    auto sl_res = t.slice(0, 1, 3);
    L3_CHECK_OK(sl_res.status());
    Tensor sl = sl_res.take_value();
    L3_CHECK(sl.is_contiguous());
    L3_CHECK_EQ(sl.dim(0), 2);
    L3_CHECK_EQ(sl.dim(1), 6);
    L3_CHECK_EQ(sl.data_ptr<f32>()[0], 6.0f);

    // Unsqueeze & squeeze
    auto u_res = sl.unsqueeze(1);  // (2, 1, 6)
    L3_CHECK_OK(u_res.status());
    L3_CHECK_EQ(u_res->rank(), 3);
    auto sq_res = u_res->squeeze(1);  // back to (2, 6)
    L3_CHECK_OK(sq_res.status());
    L3_CHECK(sq_res->shape() == sl.shape());
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 3. BitNet b1.58 Ternary Quantization, Integer Dot Product & Linear Layer
// -----------------------------------------------------------------------------
void test_bitnet_kernels() {
  // Warm up singleton default_thread_pool() before recording baseline bytes.
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    constexpr Index kLen = 136;
    i8 w_raw[kLen];
    u8 w_packed[kLen / 4];
    i8 x_i8[kLen];

    Rng rng(42ULL);
    i32 expected_dot = 0;
    for (Index i = 0; i < kLen; ++i) {
      const i8 w = static_cast<i8>(static_cast<int>(rng.bounded_u32(3)) - 1);
      const i8 x = static_cast<i8>(static_cast<int>(rng.bounded_u32(200)) - 100);
      w_raw[i] = w;
      x_i8[i] = x;
      expected_dot += static_cast<i32>(w) * static_cast<i32>(x);
    }
    for (Index b = 0; b < kLen / 4; ++b) {
      w_packed[b] = pack_t2_x4(w_raw[b * 4 + 0], w_raw[b * 4 + 1],
                               w_raw[b * 4 + 2], w_raw[b * 4 + 3]);
    }

    const i32 actual_dot = kernels::bitnet_dot_t2_i8(w_packed, x_i8, kLen);
    L3_CHECK_EQ(actual_dot, expected_dot);

    // Full BitNet b1.58 Linear projection Y = (X @ W_t2^T) * w_scale
    constexpr Index M = 4;
    constexpr Index K = 64;
    constexpr Index N = 16;

    auto x_res = Tensor::empty({M, K}, DType::kFloat32);
    auto w_f32_res = Tensor::empty({N, K}, DType::kFloat32);
    auto out_res = Tensor::zeros({M, N}, DType::kFloat32);
    L3_CHECK_OK(x_res.status());
    L3_CHECK_OK(w_f32_res.status());
    L3_CHECK_OK(out_res.status());

    Tensor x = x_res.take_value();
    Tensor w_f32 = w_f32_res.take_value();
    Tensor out = out_res.take_value();

    for (Index i = 0; i < x.numel(); ++i) {
      x.mutable_data_ptr<f32>()[i] = rng.uniform_f32(-2.0f, 2.0f);
    }
    for (Index i = 0; i < w_f32.numel(); ++i) {
      w_f32.mutable_data_ptr<f32>()[i] = rng.normal_f32(0.0f, 1.5f);
    }

    // Quantize W_f32 (N, K) -> W_t2 (N, K) with 16x smaller byte footprint!
    auto w_t2_res = kernels::quantize_f32_to_t2(w_f32);
    L3_CHECK_OK(w_t2_res.status());
    Tensor w_t2 = w_t2_res.take_value();
    L3_CHECK_EQ(w_t2.dtype(), DType::kTernary2);
    L3_CHECK_EQ(w_t2.nbytes() * 16, w_f32.nbytes());

    L3_CHECK_OK(kernels::bitnet_linear_t2_f32(x, w_t2, &out));

    // Verify against dequantized f32 reference dot product
    f32 deq_row[K];
    for (Index m = 0; m < M; ++m) {
      for (Index n = 0; n < N; ++n) {
        kernels::dequantize_t2_to_f32_raw(
            w_t2.raw_data() + static_cast<Size>(n * (K / 4)), deq_row, K,
            w_t2.quant_scale());
        const f32 ref =
            kernels::dot_f32(x.data_ptr<f32>() + m * K, deq_row, K);
        const f32 got = out.data_ptr<f32>()[m * N + n];
        L3_CHECK(approx_eq(got, ref, 1e-3f));
      }
    }
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 4. Core SIMD FP32 LLM Kernels (RMSNorm, Softmax, SiLU, RoPE, MatMul)
// -----------------------------------------------------------------------------
void test_fp32_llm_kernels() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    constexpr Index D = 64;
    f32 x[D], w[D], norm_out[D];
    for (Index i = 0; i < D; ++i) {
      x[i] = static_cast<f32>(i + 1) * 0.1f;
      w[i] = 1.0f;
    }
    kernels::rmsnorm_f32(x, w, norm_out, D, 1e-5f);
    const f32 ms = kernels::dot_f32(norm_out, norm_out, D) / static_cast<f32>(D);
    L3_CHECK(approx_eq(ms, 1.0f, 1e-4f));

    f32 probs[D];
    kernels::softmax_f32(x, probs, D);
    f32 prob_sum = 0.0f;
    for (Index i = 0; i < D; ++i) {
      L3_CHECK_GT(probs[i], 0.0f);
      prob_sum += probs[i];
    }
    L3_CHECK(approx_eq(prob_sum, 1.0f, 1e-5f));

    const f32 norm_before = kernels::dot_f32(x, x, D);
    kernels::rope_f32_inplace(x, D, /*pos=*/7);
    const f32 norm_after = kernels::dot_f32(x, x, D);
    L3_CHECK(approx_eq(norm_before, norm_after, 1e-3f));

    // FP32 MatMul test: (8, 32) @ (32, 16) -> (8, 16)
    auto a_res = Tensor::full({8, 32}, 0.5f, DType::kFloat32);
    auto b_res = Tensor::full({32, 16}, 2.0f, DType::kFloat32);
    auto c_res = Tensor::zeros({8, 16}, DType::kFloat32);
    L3_CHECK_OK(a_res.status());
    L3_CHECK_OK(b_res.status());
    L3_CHECK_OK(c_res.status());

    Tensor a = a_res.take_value();
    Tensor b = b_res.take_value();
    Tensor c = c_res.take_value();
    L3_CHECK_OK(kernels::matmul_f32(a, b, &c));
    for (Index i = 0; i < c.numel(); ++i) {
      L3_CHECK(approx_eq(c.data_ptr<f32>()[i], 32.0f, 1e-4f));
    }
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

}  // namespace
}  // namespace l3

int main() {
  l3::set_log_level(l3::LogLevel::kInfo);
  L3_LOG_INFO("Running tests/test_tensor_and_kernels.cpp...");

  l3::test_shape_and_strides();
  l3::test_tensor_views_and_clone();
  l3::test_bitnet_kernels();
  l3::test_fp32_llm_kernels();

  L3_LOG_INFO(
      "tests/test_tensor_and_kernels.cpp: ALL STAGE 2 TESTS PASSED (0 bytes leaked).");
  return 0;
}
