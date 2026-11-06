// src/nn/attention.cpp
//
// Implementation of KVCache and GroupedQueryAttention (GQA) with multi-threaded
// head parallelism and SIMD value accumulation.

#include "log3/nn/attention.h"

#include <cmath>

#include "log3/core/memory.h"
#include "log3/kernels/cpu_kernels.h"

#if L3_HAS_AVX2
#include <immintrin.h>
#endif

namespace l3 {
namespace nn {

// -----------------------------------------------------------------------------
// KVCache
// -----------------------------------------------------------------------------

Result<KVCache> KVCache::create(Index max_seq_len, Index num_kv_heads,
                                Index head_dim,
                                const Allocator& allocator) noexcept {
  if (max_seq_len <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "KVCache::create: invalid cache dimensions");
  }

  KVCache cache;
  cache.max_seq_len_ = max_seq_len;
  cache.num_kv_heads_ = num_kv_heads;
  cache.head_dim_ = head_dim;
  cache.cur_len_ = 0;

  L3_ASSIGN_OR_RETURN(cache.k_cache_,
                      Tensor::zeros({max_seq_len, num_kv_heads, head_dim},
                                    DType::kFloat32, allocator));
  L3_ASSIGN_OR_RETURN(cache.v_cache_,
                      Tensor::zeros({max_seq_len, num_kv_heads, head_dim},
                                    DType::kFloat32, allocator));

  return cache;
}

Status KVCache::append(const Tensor& new_k, const Tensor& new_v) noexcept {
  if (!new_k.is_valid() || !new_v.is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "KVCache::append: invalid input Tensor");
  }
  if (!new_k.is_contiguous() || !new_v.is_contiguous()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "KVCache::append: new_k and new_v must be contiguous");
  }
  if (new_k.dtype() != DType::kFloat32 || new_v.dtype() != DType::kFloat32) {
    return Status::error(StatusCode::kTypeMismatch,
                         "KVCache::append: expects f32 tensors");
  }
  if (new_k.rank() < 1 || new_v.rank() < 1 || new_k.dim(0) != new_v.dim(0)) {
    return Status::error(StatusCode::kShapeMismatch,
                         "KVCache::append: token count mismatch between K and V");
  }

  const Index new_tokens = new_k.dim(0);
  const Index expected_elems = new_tokens * num_kv_heads_ * head_dim_;
  if (new_k.numel() != expected_elems || new_v.numel() != expected_elems) {
    return Status::error(
        StatusCode::kShapeMismatch,
        "KVCache::append: K/V dimensions do not match (num_kv_heads, head_dim)");
  }

  if (cur_len_ + new_tokens > max_seq_len_) {
    return Status::errorf(
        StatusCode::kOutOfRange,
        "KVCache::append: cache capacity exceeded (%lld + %lld > %lld)",
        static_cast<long long>(cur_len_),
        static_cast<long long>(new_tokens),
        static_cast<long long>(max_seq_len_));
  }

  const Size stride_bytes =
      static_cast<Size>(num_kv_heads_ * head_dim_) * sizeof(f32);
  const Size copy_bytes = static_cast<Size>(new_tokens) * stride_bytes;

  f32* dst_k =
      k_cache_.mutable_data_ptr<f32>() + cur_len_ * num_kv_heads_ * head_dim_;
  f32* dst_v =
      v_cache_.mutable_data_ptr<f32>() + cur_len_ * num_kv_heads_ * head_dim_;

  mem_copy(dst_k, new_k.raw_data(), copy_bytes);
  mem_copy(dst_v, new_v.raw_data(), copy_bytes);

  cur_len_ += new_tokens;
  return Status::ok();
}

// -----------------------------------------------------------------------------
// GroupedQueryAttention (GQA)
// -----------------------------------------------------------------------------

Result<GroupedQueryAttention> GroupedQueryAttention::create(
    Index dim, Index num_heads, Index num_kv_heads,
    const Allocator& allocator) noexcept {
  if (dim <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
      dim % num_heads != 0 || num_heads % num_kv_heads != 0 || dim % 4 != 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "GroupedQueryAttention::create: invalid configuration");
  }

  const Index head_dim = dim / num_heads;
  if (head_dim % 2 != 0 || (num_kv_heads * head_dim) % 4 != 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "GroupedQueryAttention::create: head_dim must be even "
                         "and KV projection divisible by 4");
  }

  GroupedQueryAttention gqa;
  gqa.dim_ = dim;
  gqa.num_heads_ = num_heads;
  gqa.num_kv_heads_ = num_kv_heads;
  gqa.head_dim_ = head_dim;

  L3_ASSIGN_OR_RETURN(gqa.wq_,
                      BitLinear::create(dim, num_heads * head_dim, allocator));
  L3_ASSIGN_OR_RETURN(
      gqa.wk_, BitLinear::create(dim, num_kv_heads * head_dim, allocator));
  L3_ASSIGN_OR_RETURN(
      gqa.wv_, BitLinear::create(dim, num_kv_heads * head_dim, allocator));
  L3_ASSIGN_OR_RETURN(gqa.wo_,
                      BitLinear::create(num_heads * head_dim, dim, allocator));
  L3_ASSIGN_OR_RETURN(gqa.rope_, RotaryEmbedding::create(head_dim));

  gqa.register_parameter("wq.weight", gqa.wq_.weight());
  gqa.register_parameter("wk.weight", gqa.wk_.weight());
  gqa.register_parameter("wv.weight", gqa.wv_.weight());
  gqa.register_parameter("wo.weight", gqa.wo_.weight());

  return gqa;
}

Status GroupedQueryAttention::forward(const Tensor& x, Tensor* out,
                                      KVCache* cache,
                                      ThreadPool& pool) const noexcept {
  if (out == nullptr || !x.is_valid() || !out->is_valid()) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "GroupedQueryAttention::forward: invalid input or output");
  }
  if (!x.is_contiguous() || !out->is_contiguous()) {
    return Status::error(
        StatusCode::kFailedPrecondition,
        "GroupedQueryAttention::forward: tensors must be contiguous");
  }
  if (x.dtype() != DType::kFloat32 || out->dtype() != DType::kFloat32) {
    return Status::error(
        StatusCode::kTypeMismatch,
        "GroupedQueryAttention::forward: requires f32 tensors");
  }
  if (x.rank() != 2 || x.dim(1) != dim_ || out->shape() != x.shape()) {
    return Status::error(
        StatusCode::kShapeMismatch,
        "GroupedQueryAttention::forward: expects 2D (seq_len, dim) tensors");
  }

  const Index seq_len = x.dim(0);

  // 1. Project Q, K, V via BitLinear
  Tensor q;
  Tensor k;
  Tensor v;
  L3_ASSIGN_OR_RETURN(q, Tensor::empty({seq_len, num_heads_ * head_dim_}));
  L3_ASSIGN_OR_RETURN(k, Tensor::empty({seq_len, num_kv_heads_ * head_dim_}));
  L3_ASSIGN_OR_RETURN(v, Tensor::empty({seq_len, num_kv_heads_ * head_dim_}));

  L3_RETURN_IF_ERROR(wq_.forward(x, &q, pool));
  L3_RETURN_IF_ERROR(wk_.forward(x, &k, pool));
  L3_RETURN_IF_ERROR(wv_.forward(x, &v, pool));

  // Reshape Q, K, and V to 3D (seq_len, heads, head_dim)
  Tensor q_3d;
  Tensor k_3d;
  Tensor v_3d;
  L3_ASSIGN_OR_RETURN(q_3d, q.reshape({seq_len, num_heads_, head_dim_}));
  L3_ASSIGN_OR_RETURN(k_3d, k.reshape({seq_len, num_kv_heads_, head_dim_}));
  L3_ASSIGN_OR_RETURN(v_3d, v.reshape({seq_len, num_kv_heads_, head_dim_}));

  // 2. Apply RoPE to Q and K
  const Index start_pos = (cache != nullptr) ? cache->current_length() : 0;
  L3_RETURN_IF_ERROR(rope_.apply(&q_3d, start_pos));
  L3_RETURN_IF_ERROR(rope_.apply(&k_3d, start_pos));

  // 3. Update or slice KV cache
  const f32* k_all_data = nullptr;
  const f32* v_all_data = nullptr;
  Index total_seq_len = seq_len;

  if (cache != nullptr) {
    L3_RETURN_IF_ERROR(cache->append(k_3d, v_3d));
    total_seq_len = cache->current_length();
    k_all_data = cache->key_cache().data_ptr<f32>();
    v_all_data = cache->value_cache().data_ptr<f32>();
  } else {
    k_all_data = k_3d.data_ptr<f32>();
    v_all_data = v_3d.data_ptr<f32>();
  }

  // 4. Multi-head Causal Attention: parallelized across (s, h) pairs via `pool`!
  Tensor attn_out;
  L3_ASSIGN_OR_RETURN(attn_out, Tensor::zeros({seq_len, num_heads_ * head_dim_}));

  const Index group_size = num_heads_ / num_kv_heads_;
  const f32 scale = 1.0f / std::sqrt(static_cast<f32>(head_dim_));

  // Allocate score matrix `(seq_len * num_heads_, total_seq_len)` so every
  // `(s, h)` task has its own conflict-free score slice during parallel_for.
  const Index total_heads_work = seq_len * num_heads_;
  Tensor score_buf;
  L3_ASSIGN_OR_RETURN(
      score_buf, Tensor::empty({total_heads_work, total_seq_len}, DType::kFloat32));
  f32* all_scores = score_buf.mutable_data_ptr<f32>();

  const f32* q_ptr = q_3d.data_ptr<f32>();
  f32* attn_ptr = attn_out.mutable_data_ptr<f32>();

  pool.parallel_for(
      0, total_heads_work, 1, [&](Index /*tid*/, Index begin, Index end) noexcept {
        for (Index work_idx = begin; work_idx < end; ++work_idx) {
          const Index s = work_idx / num_heads_;
          const Index h = work_idx % num_heads_;
          const Index cur_token_pos = start_pos + s;
          const Index kv_h = h / group_size;

          const f32* q_vec = q_ptr + (s * num_heads_ + h) * head_dim_;
          f32* scores = all_scores + work_idx * total_seq_len;

          // Only compute dot products over the causal window [0 .. cur_token_pos]
          const Index causal_len = cur_token_pos + 1;
          for (Index t = 0; t < causal_len; ++t) {
            const f32* k_vec =
                k_all_data + (t * num_kv_heads_ + kv_h) * head_dim_;
            scores[t] = kernels::dot_f32(q_vec, k_vec, head_dim_) * scale;
          }

          kernels::softmax_f32(scores, scores, causal_len);

          f32* out_vec = attn_ptr + (s * num_heads_ + h) * head_dim_;
          mem_zero(out_vec, static_cast<Size>(head_dim_) * sizeof(f32));

          for (Index t = 0; t < causal_len; ++t) {
            const f32 prob = scores[t];
            const f32* v_vec =
                v_all_data + (t * num_kv_heads_ + kv_h) * head_dim_;
            Index d = 0;
#if L3_HAS_AVX2
            const __m256 vprob = _mm256_set1_ps(prob);
            for (; d + 8 <= head_dim_; d += 8) {
              __m256 vo = _mm256_loadu_ps(out_vec + d);
              __m256 vv = _mm256_loadu_ps(v_vec + d);
#if defined(__FMA__)
              vo = _mm256_fmadd_ps(vprob, vv, vo);
#else
              vo = _mm256_add_ps(vo, _mm256_mul_ps(vprob, vv));
#endif
              _mm256_storeu_ps(out_vec + d, vo);
            }
#endif
            for (; d < head_dim_; ++d) {
              out_vec[d] += prob * v_vec[d];
            }
          }
        }
      });

  // 5. Final output projection via BitLinear (wo)
  return wo_.forward(attn_out, out, pool);
}

}  // namespace nn
}  // namespace l3
