// log3/nn/attention.h
//
// High-performance BitNet Grouped-Query Attention (GQA / MHA / MQA) with
// zero-copy pre-allocated KV-Cache.
//
// Key Features:
//   * Supports Multi-Head Attention (num_kv_heads == num_q_heads), Multi-Query
//     Attention (num_kv_heads == 1), and Grouped-Query Attention (GQA).
//   * Projections (wq, wk, wv, wo) are BitLinear b1.58 layers (pure integer
//     additions/subtractions during GEMV/GEMM).
//   * Dynamic KVCache manages past key-value states token-by-token with zero
//     reallocations during autoregressive decoding.
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include "log3/core/assert.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/nn/layers.h"
#include "log3/nn/module.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace nn {

// -----------------------------------------------------------------------------
// KVCache: circular / sliding sequence buffer for Key and Value activations
// -----------------------------------------------------------------------------
class L3_API KVCache {
 public:
  KVCache() noexcept : max_seq_len_(0), num_kv_heads_(0), head_dim_(0), cur_len_(0) {}

  static Result<KVCache> create(
      Index max_seq_len, Index num_kv_heads, Index head_dim,
      const Allocator& allocator = default_allocator()) noexcept;

  void reset() noexcept { cur_len_ = 0; }

  Index current_length() const noexcept { return cur_len_; }
  Index max_length() const noexcept { return max_seq_len_; }
  Index num_kv_heads() const noexcept { return num_kv_heads_; }
  Index head_dim() const noexcept { return head_dim_; }

  Tensor& key_cache() noexcept { return k_cache_; }
  const Tensor& key_cache() const noexcept { return k_cache_; }
  Tensor& value_cache() noexcept { return v_cache_; }
  const Tensor& value_cache() const noexcept { return v_cache_; }

  // Append new key/value tokens of shape `(new_tokens, num_kv_heads, head_dim)`.
  Status append(const Tensor& new_k, const Tensor& new_v) noexcept;

 private:
  Index max_seq_len_;
  Index num_kv_heads_;
  Index head_dim_;
  Index cur_len_;
  Tensor k_cache_;  // shape: (max_seq_len, num_kv_heads, head_dim)
  Tensor v_cache_;  // shape: (max_seq_len, num_kv_heads, head_dim)
};

// -----------------------------------------------------------------------------
// GroupedQueryAttention (GQA) with BitLinear Projections
// -----------------------------------------------------------------------------
class L3_API GroupedQueryAttention final : public Module {
 public:
  GroupedQueryAttention() noexcept
      : dim_(0), num_heads_(0), num_kv_heads_(0), head_dim_(0) {}

  static Result<GroupedQueryAttention> create(
      Index dim, Index num_heads, Index num_kv_heads,
      const Allocator& allocator = default_allocator()) noexcept;

  Index dim() const noexcept { return dim_; }
  Index num_heads() const noexcept { return num_heads_; }
  Index num_kv_heads() const noexcept { return num_kv_heads_; }
  Index head_dim() const noexcept { return head_dim_; }

  BitLinear& wq() noexcept { return wq_; }
  BitLinear& wk() noexcept { return wk_; }
  BitLinear& wv() noexcept { return wv_; }
  BitLinear& wo() noexcept { return wo_; }

  RotaryEmbedding& rope() noexcept { return rope_; }

  // Forward pass:
  //   x      : (seq_len, dim)
  //   cache  : optional KVCache for autoregressive generation
  //   out    : (seq_len, dim)
  Status forward(const Tensor& x, Tensor* out, KVCache* cache = nullptr,
                 ThreadPool& pool = default_thread_pool()) const noexcept;

 private:
  Index dim_;
  Index num_heads_;
  Index num_kv_heads_;
  Index head_dim_;

  BitLinear wq_;  // (dim) -> (num_heads * head_dim)
  BitLinear wk_;  // (dim) -> (num_kv_heads * head_dim)
  BitLinear wv_;  // (dim) -> (num_kv_heads * head_dim)
  BitLinear wo_;  // (num_heads * head_dim) -> (dim)
  RotaryEmbedding rope_;
};

}  // namespace nn
}  // namespace l3
