// log3/nn/transformer.h
//
// Complete BitNet b1.58 Transformer model architecture:
//   * BitNetFeedForward : SwiGLU FFN with BitLinear w1 (gate), w2 (down), w3 (up)
//   * TransformerBlock  : Pre-RMSNorm -> GQA Attention -> Residual -> Pre-RMSNorm -> FFN -> Residual
//   * BitNetTransformer : Embedding -> N Blocks -> Final RMSNorm -> LM Head -> Token Generation
//
// Compatibility:
//   * C++17, no exceptions, no RTTI.

#pragma once

#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"
#include "log3/nn/attention.h"
#include "log3/nn/layers.h"
#include "log3/nn/module.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace nn {

struct TransformerConfig {
  Index vocab_size = 32000;
  Index dim = 512;
  Index hidden_dim = 1376;
  Index num_layers = 4;
  Index num_heads = 8;
  Index num_kv_heads = 4;
  Index max_seq_len = 2048;
  f32 norm_eps = 1e-5f;
  f32 rope_theta = 10000.0f;
};

// -----------------------------------------------------------------------------
// BitNet SwiGLU Feed-Forward Network (FFN)
// -----------------------------------------------------------------------------
class L3_API BitNetFeedForward final : public Module {
 public:
  BitNetFeedForward() noexcept : dim_(0), hidden_dim_(0) {}

  static Result<BitNetFeedForward> create(
      Index dim, Index hidden_dim,
      const Allocator& allocator = default_allocator()) noexcept;

  BitLinear& w1() noexcept { return w1_; }  // gate: dim -> hidden_dim
  BitLinear& w2() noexcept { return w2_; }  // down: hidden_dim -> dim
  BitLinear& w3() noexcept { return w3_; }  // up:   dim -> hidden_dim

  // Forward pass: y = w2(silu(w1(x)) * w3(x))
  Status forward(const Tensor& x, Tensor* out,
                 ThreadPool& pool = default_thread_pool()) const noexcept;

 private:
  Index dim_;
  Index hidden_dim_;
  BitLinear w1_;
  BitLinear w2_;
  BitLinear w3_;
};

// -----------------------------------------------------------------------------
// TransformerBlock
// -----------------------------------------------------------------------------
class L3_API TransformerBlock final : public Module {
 public:
  TransformerBlock() noexcept : layer_id_(0) {}

  static Result<TransformerBlock> create(
      Index layer_id, const TransformerConfig& cfg,
      const Allocator& allocator = default_allocator()) noexcept;

  RMSNorm& attn_norm() noexcept { return attn_norm_; }
  GroupedQueryAttention& attn() noexcept { return attn_; }
  RMSNorm& ffn_norm() noexcept { return ffn_norm_; }
  BitNetFeedForward& ffn() noexcept { return ffn_; }

  Status forward(const Tensor& x, Tensor* out, KVCache* cache = nullptr,
                 ThreadPool& pool = default_thread_pool()) const noexcept;

 private:
  Index layer_id_;
  RMSNorm attn_norm_;
  GroupedQueryAttention attn_;
  RMSNorm ffn_norm_;
  BitNetFeedForward ffn_;
};

// -----------------------------------------------------------------------------
// BitNetTransformer: End-to-end 1.58-bit Language Model
// -----------------------------------------------------------------------------
class L3_API BitNetTransformer final : public Module {
 public:
  BitNetTransformer() noexcept {}

  static Result<BitNetTransformer> create(
      const TransformerConfig& cfg,
      const Allocator& allocator = default_allocator()) noexcept;

  const TransformerConfig& config() const noexcept { return cfg_; }
  Embedding& tok_embeddings() noexcept { return tok_embeddings_; }
  TransformerBlock& block(Index i) noexcept { return blocks_[i]; }
  RMSNorm& norm() noexcept { return norm_; }
  BitLinear& output() noexcept { return output_; }

  // Forward pass: token sequence -> next token logits (shape: seq_len, vocab_size)
  Status forward(Span<const Index> tokens, Tensor* logits,
                 Span<KVCache> caches = {},
                 ThreadPool& pool = default_thread_pool()) const noexcept;

  // Single-token generation step: given prompt tokens, generates `max_new_tokens`
  Status generate(Span<const Index> prompt, Index max_new_tokens,
                  SmallVec<Index, 128>* out_tokens, f32 temperature = 0.0f,
                  ThreadPool& pool = default_thread_pool()) const noexcept;

 private:
  TransformerConfig cfg_;
  Embedding tok_embeddings_;
  SmallVec<TransformerBlock, 32> blocks_;
  RMSNorm norm_;
  BitLinear output_;
};

}  // namespace nn
}  // namespace l3
