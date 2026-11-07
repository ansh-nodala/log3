# log3 Technical Reference & C++17 Developer Manual

**Version:** 0.4.0 (Stages 1–4 Complete) | **Standard:** C++17 (`-fno-exceptions -fno-rtti`) | **Alignment:** 64-Byte Cache-Line Aligned

## Table of Contents
1. Executive Architecture & Build Guide
2. Stage 1 — Core Foundation (`log3/core`)
3. Stage 2 — N-D Tensor Engine & SIMD Kernels (`log3/tensor`, `log3/kernels`)
4. Stage 3 — BitNet Neural Modules, Transformer & Zero-Copy I/O (`log3/nn`, `log3/io`)
5. Stage 4 — Reverse-Mode Autograd, BitNet STE QAT & Optimizers (`log3/autograd`, `log3/optim`)
6. End-to-End Compilable C++17 Cookbooks

## Architectural Summary by Stage

| Stage | Namespace / Headers | Footprint | Core Capabilities |
| :--- | :--- | :--- | :--- |
| **Stage 1: Core Foundation** | `log3/core/*` | 22 headers, 8 .cpp | Zero-STL runtime, 64B aligned allocator, Arena, Status/Result&lt;T&gt;, Span, SmallVec, FlatMap, MMapFile, work-stealing ThreadPool, f16/bf16/Ternary2 DTypes, PCG/Xoshiro Rng, Timer, BitOps. |
| **Stage 2: Tensor & SIMD Kernels** | `log3/tensor/*, log3/kernels/*` | 4 headers, 3 .cpp | Inline 8-D Shape/Strides, reference-counted Storage, zero-copy views/slicing/broadcasting, sub-byte kTernary2 (4 weights/byte) & kInt4, AVX2/FMA/NEON BitNet b1.58 quantization & integer dot/GEMM kernels. |
| **Stage 3: Neural Layers & I/O** | `log3/nn/*, log3/io/*` | 5 headers, 4 .cpp | Module hierarchy, RMSNorm, BitLinear b1.58, FP32 Linear, Embedding, RoPE, KVCache, GroupedQueryAttention (GQA/MHA/MQA), SwiGLU BitNetFeedForward, BitNetTransformer, 64B-aligned zero-copy .l3 checkpoint I/O. |
| **Stage 4: Autograd & Optimizers** | `log3/autograd/*, log3/optim/*` | 3 headers, 3 .cpp | Reverse-mode topological DAG engine, GradMode/NoGradGuard, 13 differentiable ops, BitNet Straight-Through Estimator (bitnet_linear_ste) for QAT, global L2 clip_grad_norm, SGD with momentum, and decoupled AdamW. |

## Supported Data Types (`l3::DType`)

| Enumerator | Storage Size | Block Size | Description & Primary Role |
| :--- | :--- | :--- | :--- |
| `DType::kFloat32` | 32 bits (4 B) | 1 elem | IEEE-754 single-precision float; master weights, activations, and gradients. |
| `DType::kFloat16` | 16 bits (2 B) | 1 elem | IEEE-754 half-precision float (l3::f16) with software FP32 conversion. |
| `DType::kBFloat16` | 16 bits (2 B) | 1 elem | Brain floating-point (l3::bf16) preserving FP32 dynamic range. |
| `DType::kInt32` | 32 bits (4 B) | 1 elem | Signed 32-bit integer; token IDs and BitNet dot-product accumulators. |
| `DType::kInt16` | 16 bits (2 B) | 1 elem | Signed 16-bit integer. |
| `DType::kInt8` | 8 bits (1 B) | 1 elem | Signed 8-bit integer; absmax-quantized BitNet activations in [-127, 127]. |
| `DType::kUInt8` | 8 bits (1 B) | 1 elem | Unsigned 8-bit integer; raw byte buffers. |
| `DType::kBool` | 8 bits (1 B) | 1 elem | Boolean mask element. |
| `DType::kTernary2` | 2 bits (1 B / block) | 4 elems | BitNet b1.58 ternary weight block packing {-1, 0, +1} as {00, 01, 10} (4 weights/byte). |
| `DType::kInt4` | 4 bits (1 B / block) | 2 elems | Sub-byte 4-bit symmetric quantized weight block (2 weights/byte). |

## 1. Executive Architecture & Build Guide

log3 is a pure C++17, zero-dependency, low-level deep learning and BitNet b1.58 engine engineered to compile cleanly with -fno-exceptions and -fno-rtti. Every subsystem—from 64-byte cache-aligned memory arenas and work-stealing thread pools to 2-bit packed ternary SIMD kernels, zero-copy memory-mapped .l3 checkpoints, and Straight-Through Estimator (STE) Quantization-Aware Training—is built from first principles.

### Project Directory Structure & CMake / Compiler Configuration

- **Header:** `CMakeLists.txt / project/*`
- **Key Symbols / Signature:** `cmake -B build -DL3_ENABLE_NATIVE=ON -DL3_BUILD_TESTS=ON && cmake --build build -j$(nproc)`

The repository is organized into 7 modular header namespaces under include/log3/ (core, tensor, kernels, nn, io, autograd, optim) with corresponding implementation units in src/ and 4 comprehensive test suites in tests/. You can build log3 either with CMake or directly via g++/clang++ without external third-party dependencies.

```cpp
// Directory Hierarchy:
// include/log3/
//   core/      -> platform.h, types.h, status.h, log.h, assert.h, memory.h, arena.h,
//                 ptr.h, span.h, small_vector.h, half.h, dtype.h, cpu_features.h,
//                 str_view.h, fn_ref.h, flat_map.h, mmap.h, thread_pool.h,
//                 bit_ops.h, rng.h, timer.h, scope_guard.h
//   tensor/    -> shape.h, storage.h, tensor.h
//   kernels/   -> cpu_kernels.h
//   nn/        -> module.h, layers.h, attention.h, transformer.h
//   io/        -> checkpoint.h
//   autograd/  -> engine.h, ops.h
//   optim/     -> optimizer.h

// 1. Standard CMake Release Build (with host AVX2/NEON auto-tuning):
// $ cmake -B build -DCMAKE_BUILD_TYPE=Release -DL3_ENABLE_NATIVE=ON
// $ cmake --build build -j$(nproc)
// $ ctest --test-dir build --output-on-failure -V

// 2. Sanitizer Verification Build (AddressSanitizer + UndefinedBehaviorSanitizer):
// $ cmake -B build_asan -DCMAKE_BUILD_TYPE=Debug -DL3_ENABLE_ASAN=ON
// $ cmake --build build_asan -j$(nproc) && ctest --test-dir build_asan

// 3. Direct Single-Command g++ Compilation (Zero CMake Required):
// $ g++ -std=c++17 -O3 -march=native -fno-exceptions -fno-rtti -pthread \
//       -Iinclude src/core/*.cpp src/tensor/*.cpp src/kernels/*.cpp \
//       src/nn/*.cpp src/io/*.cpp src/autograd/*.cpp src/optim/*.cpp \
//       main.cpp -o log3_app
```

## 2. Stage 1 — Core Foundation (log3/core)

The log3/core module replaces bloated STL primitives with deterministic, cache-aligned, exception-free C++17 building blocks. Every allocation is 64-byte aligned for AVX-512/AVX2/NEON vectorization and tracked by atomic leak counters.

### Platform Macros, Primitive Types & Error Handling (Status / Result&lt;T&gt;)

- **Header:** `log3/core/platform.h, types.h, status.h`
- **Key Symbols / Signature:** `Status, Result<T>, L3_TRY(expr), L3_ASSIGN_OR_RETURN(lhs, expr)`

Defines fixed-width types (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Index, Size, kCacheLineBytes = 64) and zero-overhead error handling without C++ exceptions. Status encapsulates a StatusCode (kOk, kInvalidArgument, kOutOfMemory, kOutOfRange, kDimensionMismatch, kDTypeMismatch, kIoError, kCorruptedData, kUnimplemented, kInternal) plus source file/line metadata. Result&lt;T&gt; is a tagged union holding either a valid T or a Status.

```cpp
#include "log3/core/status.h"
#include "log3/core/types.h"

using namespace l3;

Result<Index> compute_stride_bytes(Index numel, Size elem_size) noexcept {
  if (numel < 0) {
    return L3_MAKE_STATUS(StatusCode::kInvalidArgument, "numel must be non-negative");
  }
  return static_cast<Index>(numel) * static_cast<Index>(elem_size);
}

Status validate_buffer(Index count) noexcept {
  // L3_ASSIGN_OR_RETURN unwraps Result<T> or returns the error Status immediately:
  L3_ASSIGN_OR_RETURN(Index total_bytes, compute_stride_bytes(count, sizeof(f32)));
  if (total_bytes > (1LL << 30)) {
    return L3_MAKE_STATUS(StatusCode::kOutOfMemory, "Requested buffer exceeds 1 GiB");
  }
  return Status::Ok();
}
```

### Structured Logging & Fail-Fast Assertions

- **Header:** `log3/core/log.h, assert.h`
- **Key Symbols / Signature:** `L3_LOG_INFO(...), L3_LOG_WARN(...), L3_ASSERT(cond), L3_CHECK(cond, fmt, ...)`

Thread-safe ANSI-colored structured logger with microsecond timestamps, thread IDs, and runtime/compile-time minimum log level filtering. L3_CHECK is always active in both Debug and Release builds, while L3_DCHECK and L3_ASSERT can be toggled.

```cpp
#include "log3/core/assert.h"
#include "log3/core/log.h"

void init_engine_logging() noexcept {
  l3::set_log_level(l3::LogLevel::kInfo);
  l3::set_log_color_enabled(true);

  L3_LOG_INFO("Initializing log3 runtime (Cache line = %zu bytes)", l3::kCacheLineBytes);
  int alignment = 64;
  L3_CHECK((alignment & (alignment - 1)) == 0, "Alignment %d must be a power of 2", alignment);
}
```

### 64-Byte Aligned Memory Allocator, Arena & Smart Pointers

- **Header:** `log3/core/memory.h, arena.h, ptr.h`
- **Key Symbols / Signature:** `Allocator, Arena, ArenaScope, UniquePtr<T>, RefCounted, Ptr<T>`

default_allocator() allocates 64-byte aligned buffers with atomic live-byte tracking (allocated_bytes(), allocation_count()). Arena provides O(1) pointer-bump allocation with RAII ArenaScope rollback and an Adapter (Arena::as_allocator()) that plugs directly into Tensor creation. UniquePtr&lt;T&gt; and intrusive atomic Ptr&lt;T&gt; manage ownership.

```cpp
#include "log3/core/arena.h"
#include "log3/core/memory.h"
#include "log3/core/ptr.h"

using namespace l3;

Status run_scratch_computation() noexcept {
  Arena scratch(4 * 1024 * 1024); // 4 MiB pre-allocated arena
  {
    ArenaScope temp_scope(scratch); // Automatically rewinds arena on scope exit
    f32* buf = scratch.alloc_array<f32>(1024);
    L3_CHECK(buf != nullptr, "Arena allocation failed");
    buf[0] = 42.0f;
    // Pass scratch.as_allocator() into any Tensor factory for zero-malloc tensors!
  }
  // scratch.used() is now back to 0 bytes
  return Status::Ok();
}
```

### Cache-Friendly Containers: Span&lt;T&gt;, SmallVec<T,N>, StrView, FnRef & FlatMap

- **Header:** `log3/core/span.h, small_vector.h, str_view.h, fn_ref.h, flat_map.h`
- **Key Symbols / Signature:** `Span<T>, SmallVec<T, N>, StrView, FnRef<R(Args...)>, FlatMap<K, V>, StrMap<V>`

Span&lt;T&gt; provides bounds-checked non-owning views. SmallVec<T, N> stores up to N elements on the stack before spilling to the 64B-aligned heap. StrView offers zero-copy string slicing and FNV-1a 64-bit hashing. FnRef&lt;Sig&gt; is a 16-byte type-erased callable reference that handles both regular and noexcept lambdas/functions in C++17. FlatMap<K, V> and StrMap&lt;V&gt; implement open-addressed Robin Hood linear probing with backward-shift deletion.

```cpp
#include "log3/core/flat_map.h"
#include "log3/core/fn_ref.h"
#include "log3/core/small_vector.h"
#include "log3/core/span.h"
#include "log3/core/str_view.h"

using namespace l3;

void demo_containers() noexcept {
  // 1. SmallVec lives completely on the stack for <= 8 elements:
  SmallVec<Index, 8> dims = {2, 4, 128};
  dims.push_back(64);
  Span<const Index> dim_span(dims.data(), dims.size());

  // 2. Open-addressing Robin Hood hash table:
  StrMap<f32> layer_scales;
  layer_scales.insert("blk.0.attn_q.scale", 0.03125f);
  if (const f32* scale = layer_scales.find("blk.0.attn_q.scale")) {
    // Found in O(1) cache-local probe
    (void)*scale;
  }

  // 3. Zero-allocation callable reference:
  auto apply_op = [](FnRef<f32(f32)> fn, f32 x) noexcept { return fn(x); };
  f32 res = apply_op([](f32 v) noexcept { return v * 2.0f; }, 3.5f);
  (void)dim_span; (void)res;
}
```

### Half-Precision Types (f16, bf16) & Sub-Byte BitNet Ternary DTypes

- **Header:** `log3/core/half.h, dtype.h`
- **Key Symbols / Signature:** `f16, bf16, DType, Ternary2Pack, Int4Pack, dtype_row_bytes(dt, cols)`

Provides IEEE-754 f16 and Brain Float bf16 conversions alongside DType metadata. Ternary2Pack encodes 4 BitNet b1.58 ternary weights {-1, 0, +1} into a single byte (2 bits per weight: 0 -> 00, +1 -> 01, -1 -> 10), achieving a 16x memory reduction compared to FP32 weights.

```cpp
#include "log3/core/dtype.h"
#include "log3/core/half.h"

using namespace l3;

void demo_dtypes() noexcept {
  // Pack 4 ternary weights {-1, 0, +1, -1} into 1 byte:
  u8 packed = Ternary2Pack::pack4(-1, 0, +1, -1);
  i8 w0, w1, w2, w3;
  Ternary2Pack::unpack4(packed, w0, w1, w2, w3); // w0=-1, w1=0, w2=1, w3=-1

  // Compute exact storage bytes for 4096 ternary weights (1024 bytes):
  Size bytes = dtype_row_bytes(DType::kTernary2, 4096);

  // Convert FP32 -> BF16 / FP16 and back:
  bf16 b = bf16::from_f32(3.1415926f);
  f16  h = f16::from_f32(-0.5f);
  f32  sum = b.to_f32() + h.to_f32();
  (void)bytes; (void)sum;
}
```

### Zero-Copy Memory-Mapped Files, Work-Stealing ThreadPool, SIMD Detection & Utilities

- **Header:** `log3/core/mmap.h, thread_pool.h, cpu_features.h, bit_ops.h, rng.h, timer.h, scope_guard.h`
- **Key Symbols / Signature:** `MMapFile::open_read, ThreadPool::global().parallel_for, cpu_features(), Rng, ScopedTimer, L3_DEFER`

MMapFile wraps POSIX mmap / Windows CreateFileMapping for instant zero-copy weight loading. ThreadPool implements per-worker Chase-Lev / mutex deque work-stealing with grain-sized parallel_for. cpu_features() queries CPUID / OS XGETBV for AVX2, FMA, AVX-512, and ARM NEON. Rng implements xoshiro256** with Box-Muller normal_f32() and Kaiming/Xavier initialization.

```cpp
#include "log3/core/bit_ops.h"
#include "log3/core/cpu_features.h"
#include "log3/core/rng.h"
#include "log3/core/scope_guard.h"
#include "log3/core/thread_pool.h"
#include "log3/core/timer.h"

using namespace l3;

void demo_system_runtime() noexcept {
  const CpuFeatures& cpu = cpu_features();
  L3_LOG_INFO("CPU: %s | AVX2=%d FMA=%d NEON=%d",
              cpu.brand, cpu.avx2, cpu.fma, cpu.neon);

  Rng rng(1337ULL);
  f32 sample = rng.normal_f32(0.0f, 1.0f);

  // Parallel loop across global work-stealing thread pool:
  f32 data[1024];
  ThreadPool::global().parallel_for(0, 1024, 64, [&](Index begin, Index end) noexcept {
    for (Index i = begin; i < end; ++i) {
      data[i] = static_cast<f32>(i) * 0.5f;
    }
  });
  (void)sample;
}
```

## 3. Stage 2 — N-D Tensor Engine & SIMD Kernels (log3/tensor, log3/kernels)

The Tensor subsystem stores Shape and Strides inline (up to 8 dimensions, 0 heap allocations for metadata) and expresses strides in elements so standard types (kFloat32, kInt8) and sub-byte packed types (kTernary2, kInt4) share a unified indexing model.

### Inline Shape, Strides & Reference-Counted Storage

- **Header:** `log3/tensor/shape.h, storage.h`
- **Key Symbols / Signature:** `Shape::make, Strides::contiguous_for, broadcast_shapes, Storage::allocate`

Shape holds up to kMaxTensorDims = 8 dimensions in 72 bytes on the stack. Strides computes row-major contiguous element strides, checks contiguity across arbitrary slices, and computes NumPy/PyTorch-compatible broadcasted shapes and 0-stride broadcast views. Storage manages 64-byte aligned buffers or zero-copy external blobs.

```cpp
#include "log3/tensor/shape.h"
#include "log3/tensor/storage.h"

using namespace l3;

Status demo_shape_and_broadcast() noexcept {
  L3_ASSIGN_OR_RETURN(Shape a, Shape::make({2, 1, 64}));
  L3_ASSIGN_OR_RETURN(Shape b, Shape::make({1, 8, 64}));
  L3_ASSIGN_OR_RETURN(Shape out, broadcast_shapes(a, b)); // {2, 8, 64}

  Strides contig = Strides::contiguous_for(out);
  L3_CHECK(contig.is_contiguous(out), "Must be row-major contiguous");
  return Status::Ok();
}
```

### Tensor Creation, Zero-Copy Views, Slicing, Permute & Contiguous Clone

- **Header:** `log3/tensor/tensor.h`
- **Key Symbols / Signature:** `Tensor::zeros, ones, full, arange, from_blob, reshape, transpose, permute, slice, select, expand, contiguous`

Tensor is a lightweight handle holding a Shape, Strides, element offset, DType, quant_scale, Ptr&lt;Storage&gt;, and optional Ptr&lt;AutogradMeta&gt;. View operations (reshape, transpose, permute, slice, select, squeeze, unsqueeze, expand) never copy underlying tensor memory—they return a new Tensor sharing the same Storage.

```cpp
#include "log3/tensor/tensor.h"

using namespace l3;

Status demo_tensor_views() noexcept {
  // Allocate a [4, 8, 16] FP32 tensor initialized to 1.0f:
  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::ones(Shape({4, 8, 16}), DType::kFloat32));

  // Zero-copy transpose of dimensions 1 and 2 -> shape [4, 16, 8]:
  L3_ASSIGN_OR_RETURN(Tensor xt, x.transpose(1, 2));
  L3_CHECK(!xt.is_contiguous(), "Transposed view is non-contiguous");

  // Materialize a contiguous copy when needed by flat SIMD kernels:
  L3_ASSIGN_OR_RETURN(Tensor xt_contig, xt.contiguous());
  L3_CHECK(xt_contig.is_contiguous(), "Contiguous tensor ready for SIMD");

  // Zero-copy slice along dim 1 from index 2 to 6 with step 1:
  L3_ASSIGN_OR_RETURN(Tensor sub, x.slice(1, 2, 6, 1)); // shape [4, 4, 16]
  (void)sub;
  return Status::Ok();
}
```

### BitNet b1.58 Quantization, Packed Ternary Dot Products & Parallel GEMM Kernels

- **Header:** `log3/kernels/cpu_kernels.h`
- **Key Symbols / Signature:** `bitnet_quantize_weights_t2, bitnet_quantize_activations_i8, bitnet_dot_t2_i8, bitnet_linear_t2_f32`

Implements the complete BitNet b1.58 mathematical pipeline with hardware-dispatched AVX2/FMA and ARM NEON intrinsics: (1) Absmean ternary weight quantization W_q = RoundClip(W / gamma, -1, +1) packed 4 weights/byte into kTernary2; (2) Per-token absmax INT8 activation quantization X_q = RoundClip(X * 127 / max(|X|), -127, 127); (3) Pure integer ternary-by-INT8 dot products using SIMD sign/zero masks without floating-point multiplies; (4) Multi-threaded bitnet_linear_t2_f32 across ThreadPool.

```cpp
#include "log3/kernels/cpu_kernels.h"
#include "log3/tensor/tensor.h"

using namespace l3;

Status demo_bitnet_kernels() noexcept {
  constexpr Index M = 4, N = 64, K = 128; // K must be a multiple of 4
  L3_ASSIGN_OR_RETURN(Tensor w_fp32, Tensor::full(Shape({N, K}), 0.4f, DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor w_t2,   Tensor::empty(Shape({N, K}), DType::kTernary2));

  // 1. Quantize FP32 weights -> 2-bit packed kTernary2 + return absmean scale gamma:
  f32 gamma = kernels::bitnet_quantize_weights_t2(
      w_fp32.data_ptr<f32>(), w_t2.data_ptr<u8>(), N * K);
  w_t2.set_quant_scale(gamma);

  // 2. Execute multi-threaded BitNet b1.58 linear projection Y = X @ W_t2^T * gamma:
  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::ones(Shape({M, K}), DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor y, Tensor::zeros(Shape({M, N}), DType::kFloat32));
  kernels::bitnet_linear_t2_f32(
      x.data_ptr<f32>(), w_t2.data_ptr<u8>(), /*bias=*/nullptr,
      y.data_ptr<f32>(), M, N, K, gamma);
  return Status::Ok();
}
```

### SIMD Floating-Point Kernels: RMSNorm, RoPE, SiLU, Softmax & Tiled MatMul

- **Header:** `log3/kernels/cpu_kernels.h`
- **Key Symbols / Signature:** `dot_f32, add_f32, mul_f32, silu_f32, rmsnorm_f32, softmax_f32, rope_f32_inplace, matmul_f32`

Vectorized FP32 kernels using 8-wide AVX2 FMA registers (with scalar tail cleanup): rmsnorm_f32 computes x * rsqrt(mean(x^2) + eps) * weight; rope_f32_inplace applies Rotary Position Embeddings across query and key heads; softmax_f32 computes numerically stable row-wise probabilities; matmul_f32 executes cache-tiled multi-threaded GEMM.

```cpp
#include "log3/kernels/cpu_kernels.h"

using namespace l3;

void demo_fp32_kernels() noexcept {
  alignas(64) f32 x[64], weight[64], out[64];
  for (int i = 0; i < 64; ++i) {
    x[i] = 0.1f * static_cast<f32>(i);
    weight[i] = 1.0f;
  }

  // Vectorized RMSNorm + SiLU + Numerically Stable Softmax:
  kernels::rmsnorm_f32(x, weight, out, 64, 1e-5f);
  kernels::silu_f32(out, out, 64);
  kernels::softmax_f32(out, out, 64);
}
```

## 4. Stage 3 — BitNet Neural Modules, Transformer & Zero-Copy I/O (log3/nn, log3/io)

Stage 3 assembles the low-level kernels into a complete modern LLM architecture featuring BitLinear b1.58 projections, Grouped-Query Attention (GQA) with pre-allocated KV-Cache, SwiGLU feed-forward blocks, and 64-byte aligned .l3 checkpoint serialization.

### Core Neural Layers: Module, RMSNorm, BitLinear, Linear, Embedding & RotaryEmbedding

- **Header:** `log3/nn/module.h, layers.h`
- **Key Symbols / Signature:** `RMSNorm, BitLinear, Linear, Embedding, RotaryEmbedding`

BitLinear stores weights in 2-bit kTernary2 format and can be initialized from FP32 master weights via init_from_fp32() (which quantizes and packs on the fly) or bound zero-copy to memory-mapped .l3 weights via init_from_packed(). Supports arbitrary 2D [M, K] or 3D [B, S, K] input activations.

```cpp
#include "log3/nn/layers.h"

using namespace l3;

Status demo_bitlinear_layer() noexcept {
  nn::BitLinear proj;
  L3_ASSIGN_OR_RETURN(Tensor w_master, Tensor::full(Shape({128, 256}), 0.25f, DType::kFloat32));
  L3_TRY(proj.init_from_fp32(w_master)); // Quantizes to 2-bit kTernary2 (8192 bytes!)

  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::ones(Shape({2, 16, 256}), DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor y, proj.forward(x)); // Output shape: [2, 16, 128]
  return Status::Ok();
}
```

### Grouped-Query Attention (GQA / MHA / MQA) & Pre-Allocated KVCache

- **Header:** `log3/nn/attention.h`
- **Key Symbols / Signature:** `KVCache::allocate, GroupedQueryAttention::init_random, GroupedQueryAttention::forward`

Supports Multi-Head Attention (num_kv_heads == num_q_heads), Grouped-Query Attention (num_q_heads % num_kv_heads == 0), and Multi-Query Attention (num_kv_heads == 1). All four projections (wq, wk, wv, wo) are 2-bit BitLinear b1.58 layers. KVCache pre-allocates contiguous [num_layers, max_seq_len, num_kv_heads, head_dim] buffers for zero-allocation autoregressive decoding.

```cpp
#include "log3/nn/attention.h"

using namespace l3;

Status demo_gqa_attention() noexcept {
  nn::AttentionConfig cfg{/*dim=*/128, /*num_q_heads=*/8, /*num_kv_heads=*/2,
                          /*head_dim=*/16, /*max_seq_len=*/512};
  nn::GroupedQueryAttention attn;
  Rng rng(42);
  L3_TRY(attn.init_random(cfg, rng));

  L3_ASSIGN_OR_RETURN(nn::KVCache cache, nn::KVCache::allocate(1, 512, 2, 16));
  nn::RotaryEmbedding rope;
  L3_TRY(rope.init(16, 512));

  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::ones(Shape({4, 128}), DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor out, attn.forward(x, rope, &cache, /*layer_idx=*/0, /*start_pos=*/0));
  (void)out;
  return Status::Ok();
}
```

### BitNet b1.58 Transformer & Autoregressive Token Generation

- **Header:** `log3/nn/transformer.h`
- **Key Symbols / Signature:** `TransformerConfig, BitNetFeedForward, TransformerBlock, BitNetTransformer::forward_tokens, generate_greedy`

Full decoder-only BitNet b1.58 Transformer combining token Embedding, N pre-norm TransformerBlocks (RMSNorm + GQA + RMSNorm + SwiGLU BitNetFeedForward with residual connections), final RMSNorm, and output language modeling head. Includes built-in KV-cached prefill + token-by-token generate_greedy().

```cpp
#include "log3/nn/transformer.h"

using namespace l3;

Status demo_transformer_generation() noexcept {
  nn::TransformerConfig cfg;
  cfg.vocab_size = 256;
  cfg.dim = 64;
  cfg.hidden_dim = 128;
  cfg.num_layers = 2;
  cfg.num_q_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.head_dim = 16;
  cfg.max_seq_len = 128;

  nn::BitNetTransformer model;
  Rng rng(2025);
  L3_TRY(model.init_random(cfg, rng));

  L3_ASSIGN_OR_RETURN(nn::KVCache kv, model.create_kv_cache());
  i32 prompt[3] = {1, 42, 99};
  i32 generated[8] = {0};

  // Prefills prompt and autoregressively decodes 5 new tokens using KV-cache:
  L3_ASSIGN_OR_RETURN(Index total, model.generate_greedy(
      Span<const i32>(prompt, 3), /*max_new_tokens=*/5, kv, Span<i32>(generated, 8)));
  (void)total;
  return Status::Ok();
}
```

### Zero-Copy 64-Byte Aligned .l3 Model Checkpoint Writer & Reader

- **Header:** `log3/io/checkpoint.h`
- **Key Symbols / Signature:** `ModelWriter::add_tensor, ModelWriter::save, ModelReader::open, ModelReader::get_tensor`

Custom binary container format (magic 'L3M\x01') that pads every tensor payload to a 64-byte cache-line boundary inside the file. When opened via ModelReader::open(), the file is memory-mapped via MMapFile and get_tensor() returns a Tensor backed directly by the OS page cache with zero RAM copies and instant startup.

```cpp
#include "log3/io/checkpoint.h"

using namespace l3;

Status demo_zero_copy_checkpoint(const nn::BitNetTransformer& model) noexcept {
  // 1. Save entire BitNetTransformer (2-bit packed weights + FP32 norms) to disk:
  L3_TRY(model.save_checkpoint("/tmp/bitnet_model.l3"));

  // 2. Memory-map checkpoint with zero heap copies:
  L3_ASSIGN_OR_RETURN(io::ModelReader reader, io::ModelReader::open("/tmp/bitnet_model.l3", /*prefetch=*/true));
  L3_ASSIGN_OR_RETURN(Tensor wq, reader.get_tensor("blocks.0.attn.wq.weight"));
  L3_CHECK(wq.dtype() == DType::kTernary2, "Loaded directly as 2-bit packed ternary tensor!");
  return Status::Ok();
}
```

## 5. Stage 4 — Reverse-Mode Autograd, BitNet STE QAT & Optimizers (log3/autograd, log3/optim)

Stage 4 adds dynamic reverse-mode automatic differentiation directly into Tensor via intrusive AutogradMeta and GradNode graphs, enabling both standard FP32 training and BitNet b1.58 Quantization-Aware Training (QAT) via the Straight-Through Estimator (STE).

### Autograd Engine, Topological Backward Pass & NoGradGuard

- **Header:** `log3/autograd/engine.h`
- **Key Symbols / Signature:** `GradMode::is_enabled, NoGradGuard, GradNode, Tensor::set_requires_grad, Tensor::backward`

When a Tensor has set_requires_grad(true) and GradMode::is_enabled() is true, operations in log3::autograd::ops record a lightweight GradNode DAG. Calling loss.backward() performs a Depth-First topological sort and executes reverse-mode vector-Jacobian products, accumulating gradients into each leaf tensor's .grad().

```cpp
#include "log3/autograd/engine.h"
#include "log3/autograd/ops.h"

using namespace l3;

Status demo_autograd_engine() noexcept {
  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::full(Shape({2, 4}), 3.0f, DType::kFloat32));
  L3_TRY(x.set_requires_grad(true));

  // Forward graph: y = mean(silu(x * x))
  L3_ASSIGN_OR_RETURN(Tensor sq,  autograd::ops::mul(x, x));
  L3_ASSIGN_OR_RETURN(Tensor act, autograd::ops::silu(sq));
  L3_ASSIGN_OR_RETURN(Tensor loss, autograd::ops::mean(act));

  // Reverse-mode backward pass:
  L3_TRY(loss.backward());
  const Tensor& dx = x.grad(); // Populated with exact d(loss)/dx!

  // Disable graph construction during inference or weight updates:
  {
    autograd::NoGradGuard no_grad;
    L3_CHECK(!autograd::GradMode::is_enabled(), "Autograd disabled inside scope");
  }
  (void)dx;
  return Status::Ok();
}
```

### Differentiable Operations & BitNet Straight-Through Estimator (bitnet_linear_ste)

- **Header:** `log3/autograd/ops.h`
- **Key Symbols / Signature:** `add, sub, mul, silu, relu, rmsnorm, matmul, linear, bitnet_linear_ste, sum, mean, mse_loss, cross_entropy_loss`

Provides analytical backward passes for all core neural network primitives, including fused stable log-softmax cross_entropy_loss and RMSNorm backward. Most importantly, bitnet_linear_ste(x, master_weight, bias) implements BitNet b1.58 Quantization-Aware Training: during the forward pass, it quantizes master_weight to ternary {-1, 0, +1} * gamma and activations to INT8; during the backward pass, gradients flow through the discrete quantizer via the Straight-Through Estimator (STE) to update the FP32 master weights.

```cpp
#include "log3/autograd/ops.h"

using namespace l3;

Status demo_bitnet_ste_qat() noexcept {
  // Master weights are kept in FP32 with requires_grad=true during QAT:
  L3_ASSIGN_OR_RETURN(Tensor w_master, Tensor::full(Shape({8, 16}), 0.15f, DType::kFloat32));
  L3_TRY(w_master.set_requires_grad(true));

  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::ones(Shape({4, 16}), DType::kFloat32));
  i32 targets_raw[4] = {1, 3, 0, 7};
  L3_ASSIGN_OR_RETURN(Tensor targets, Tensor::from_blob(targets_raw, Shape({4}), DType::kInt32));

  // Forward pass simulates exact BitNet b1.58 ternary weight + INT8 activation quantization:
  L3_ASSIGN_OR_RETURN(Tensor logits, autograd::ops::bitnet_linear_ste(x, w_master));
  L3_ASSIGN_OR_RETURN(Tensor loss,   autograd::ops::cross_entropy_loss(logits, targets));

  // Backward pass computes STE gradients into w_master.grad():
  L3_TRY(loss.backward());
  return Status::Ok();
}
```

### Gradient Clipping & Stateful Optimizers (SGD & Decoupled AdamW)

- **Header:** `log3/optim/optimizer.h`
- **Key Symbols / Signature:** `clip_grad_norm, SGD, AdamW`

clip_grad_norm computes the global L2 norm across all parameter gradients and scales them in-place if total_norm > max_norm. SGD supports momentum and L2 weight decay. AdamW implements decoupled weight decay with first-moment (m) and second-moment (v) running averages and exact bias correction (1 - beta1^t, 1 - beta2^t).

```cpp
#include "log3/optim/optimizer.h"

using namespace l3;

Status demo_adamw_step(Tensor& w_master) noexcept {
  Tensor* params[1] = {&w_master};
  optim::AdamWConfig cfg;
  cfg.lr = 1e-3f;
  cfg.weight_decay = 0.01f;

  optim::AdamW optimizer;
  L3_TRY(optimizer.init(Span<Tensor* const>(params, 1), cfg));

  // Clip global gradient norm to 1.0 and perform bias-corrected AdamW update:
  L3_ASSIGN_OR_RETURN(f32 grad_norm, optim::clip_grad_norm(Span<Tensor* const>(params, 1), 1.0f));
  L3_TRY(optimizer.step());
  optimizer.zero_grad();
  (void)grad_norm;
  return Status::Ok();
}
```

## 6. End-to-End Compilable C++17 Cookbooks

Three self-contained, production-ready C++17 programs demonstrating how all four stages of log3 work together in real-world pipelines.

### Cookbook 1: Zero-Heap Arena Pipeline with Parallel SIMD Kernels

- **Header:** `Full C++17 Program (Core + Tensor + Kernels)`
- **Key Symbols / Signature:** `Arena + Tensor::zeros(shape, dtype, arena.as_allocator()) + rmsnorm_f32`

Demonstrates allocating tensors inside a pre-allocated 64-byte aligned Arena so that an entire forward pass executes with zero OS heap calls (0 malloc/free overhead).

```cpp
#include "log3/core/arena.h"
#include "log3/core/log.h"
#include "log3/kernels/cpu_kernels.h"
#include "log3/tensor/tensor.h"

using namespace l3;

Status run_zero_malloc_pipeline() noexcept {
  Arena frame_arena(2 * 1024 * 1024); // 2 MiB reusable arena
  Allocator alloc = frame_arena.as_allocator();

  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::full(Shape({16, 128}), 0.5f, DType::kFloat32, alloc));
  L3_ASSIGN_OR_RETURN(Tensor w, Tensor::ones(Shape({128}), DType::kFloat32, alloc));
  L3_ASSIGN_OR_RETURN(Tensor y, Tensor::empty(Shape({16, 128}), DType::kFloat32, alloc));

  for (Index row = 0; row < 16; ++row) {
    kernels::rmsnorm_f32(
        x.data_ptr<f32>() + row * 128,
        w.data_ptr<f32>(),
        y.data_ptr<f32>() + row * 128,
        128, 1e-5f);
  }
  L3_LOG_INFO("Arena pipeline complete: %zu bytes used inside arena, 0 OS heap leaks.",
              frame_arena.used());
  return Status::Ok();
}
```

### Cookbook 2: BitNet b1.58 Transformer Checkpoint Save, Zero-Copy MMap Load & Generation

- **Header:** `Full C++17 Program (NN + IO + KVCache)`
- **Key Symbols / Signature:** `BitNetTransformer::save_checkpoint + load_checkpoint + generate_greedy`

Builds a BitNet b1.58 Transformer, serializes all 2-bit packed ternary weights and RMSNorm scales to a 64-byte aligned .l3 file, reloads it zero-copy via OS mmap, and runs KV-cached greedy token generation.

```cpp
#include "log3/core/log.h"
#include "log3/nn/transformer.h"

using namespace l3;

Status run_checkpoint_and_inference() noexcept {
  nn::TransformerConfig cfg{256, 64, 128, 2, 4, 2, 16, 64, 1e-5f, 10000.0f};
  nn::BitNetTransformer model_out;
  Rng rng(777);
  L3_TRY(model_out.init_random(cfg, rng));
  L3_TRY(model_out.save_checkpoint("/tmp/log3_demo.l3"));

  // Reload into a fresh BitNetTransformer directly backed by zero-copy mmap:
  nn::BitNetTransformer model_in;
  L3_TRY(model_in.load_checkpoint("/tmp/log3_demo.l3", cfg, /*prefetch=*/true));
  L3_ASSIGN_OR_RETURN(nn::KVCache cache, model_in.create_kv_cache());

  i32 prompt[4] = {1, 12, 34, 56};
  i32 output[10] = {0};
  L3_ASSIGN_OR_RETURN(Index n, model_in.generate_greedy(
      Span<const i32>(prompt, 4), /*max_new_tokens=*/6, cache, Span<i32>(output, 10)));
  L3_LOG_INFO("Generated %lld tokens via zero-copy mmap BitNet model!", static_cast<long long>(n));
  return Status::Ok();
}
```

### Cookbook 3: BitNet b1.58 Quantization-Aware Training (QAT) Loop with STE & AdamW

- **Header:** `Full C++17 Program (Autograd + STE + Optim)`
- **Key Symbols / Signature:** `bitnet_linear_ste + rmsnorm + silu + cross_entropy_loss + AdamW + init_from_fp32`

Trains a 2-layer BitNet b1.58 network using Straight-Through Estimator (STE) Quantization-Aware Training with AdamW and gradient clipping, then exports the trained FP32 master weights into hardware-packed 2-bit BitLinear inference layers.

```cpp
#include "log3/autograd/engine.h"
#include "log3/autograd/ops.h"
#include "log3/core/log.h"
#include "log3/core/rng.h"
#include "log3/nn/layers.h"
#include "log3/optim/optimizer.h"

using namespace l3;

Status run_bitnet_qat_training() noexcept {
  Rng rng(42);
  L3_ASSIGN_OR_RETURN(Tensor w1, Tensor::empty(Shape({32, 16}), DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor w2, Tensor::empty(Shape({8, 32}),  DType::kFloat32));
  rng.fill_kaiming_uniform(Span<f32>(w1.data_ptr<f32>(), w1.numel()), 16);
  rng.fill_kaiming_uniform(Span<f32>(w2.data_ptr<f32>(), w2.numel()), 32);
  L3_TRY(w1.set_requires_grad(true));
  L3_TRY(w2.set_requires_grad(true));

  Tensor* params[2] = {&w1, &w2};
  optim::AdamWConfig opt_cfg;
  opt_cfg.lr = 2e-2f;
  opt_cfg.weight_decay = 1e-2f;
  optim::AdamW opt;
  L3_TRY(opt.init(Span<Tensor* const>(params, 2), opt_cfg));

  L3_ASSIGN_OR_RETURN(Tensor x, Tensor::empty(Shape({4, 16}), DType::kFloat32));
  rng.fill_normal(Span<f32>(x.data_ptr<f32>(), x.numel()), 0.0f, 1.0f);
  i32 labels[4] = {2, 5, 1, 7};
  L3_ASSIGN_OR_RETURN(Tensor targets, Tensor::from_blob(labels, Shape({4}), DType::kInt32));

  for (int step = 0; step < 25; ++step) {
    opt.zero_grad();
    L3_ASSIGN_OR_RETURN(Tensor h,      autograd::ops::bitnet_linear_ste(x, w1));
    L3_ASSIGN_OR_RETURN(Tensor h_act,  autograd::ops::silu(h));
    L3_ASSIGN_OR_RETURN(Tensor logits, autograd::ops::bitnet_linear_ste(h_act, w2));
    L3_ASSIGN_OR_RETURN(Tensor loss,   autograd::ops::cross_entropy_loss(logits, targets));

    L3_TRY(loss.backward());
    L3_ASSIGN_OR_RETURN(f32 gnorm, optim::clip_grad_norm(Span<Tensor* const>(params, 2), 1.0f));
    L3_TRY(opt.step());
    (void)gnorm;
  }

  // Export trained FP32 master weights to 2-bit packed BitLinear layers for deployment:
  nn::BitLinear deploy_l1, deploy_l2;
  L3_TRY(deploy_l1.init_from_fp32(w1));
  L3_TRY(deploy_l2.init_from_fp32(w2));
  return Status::Ok();
}
```
