// include/log3/kernels/cpu_kernels.h
#pragma once

#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/thread_pool.h"
#include "log3/core/types.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace kernels {

L3_API void quantize_f32_to_t2_raw(const f32* L3_RESTRICT src,
                                   u8* L3_RESTRICT dst_packed, Index count,
                                   f32* out_scale) noexcept;

L3_API void dequantize_t2_to_f32_raw(const u8* L3_RESTRICT src_packed,
                                     f32* L3_RESTRICT dst, Index count,
                                     f32 scale) noexcept;

L3_API Result<Tensor> quantize_f32_to_t2(
    const Tensor& weights_f32,
    const Allocator& allocator = default_allocator()) noexcept;

L3_API void quantize_f32_to_i8_absmax(const f32* L3_RESTRICT src,
                                      i8* L3_RESTRICT dst, Index count,
                                      f32* out_scale) noexcept;

L3_API i32 bitnet_dot_t2_i8(const u8* L3_RESTRICT w_packed,
                            const i8* L3_RESTRICT x_i8, Index count) noexcept;

L3_API f32 bitnet_dot_t2_f32(const u8* L3_RESTRICT w_packed,
                             const f32* L3_RESTRICT x_f32, Index count,
                             f32 w_scale) noexcept;

L3_API Status bitnet_linear_t2_f32(
    const Tensor& x, const Tensor& w_t2, Tensor* out,
    ThreadPool& pool = default_thread_pool()) noexcept;

L3_API f32 dot_f32(const f32* L3_RESTRICT a, const f32* L3_RESTRICT b,
                   Index count) noexcept;

// In-place friendly elementwise kernels (pointers without L3_RESTRICT so dst can alias a or b)
L3_API void add_f32(const f32* a, const f32* b, f32* dst,
                    Index count) noexcept;

L3_API void mul_f32(const f32* a, const f32* b, f32* dst,
                    Index count) noexcept;

L3_API void silu_f32(const f32* src, f32* dst, Index count) noexcept;

L3_API void rmsnorm_f32(const f32* x, const f32* weight, f32* dst,
                        Index count, f32 eps = 1e-5f) noexcept;

L3_API void softmax_f32(const f32* src, f32* dst, Index count) noexcept;

L3_API void rope_f32_inplace(f32* vec, Index head_dim, Index pos,
                             f32 theta_base = 10000.0f) noexcept;

L3_API Status matmul_f32(const Tensor& a, const Tensor& b, Tensor* out,
                         ThreadPool& pool = default_thread_pool()) noexcept;

}  // namespace kernels
}  // namespace l3
