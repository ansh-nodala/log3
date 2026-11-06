// src/nn/transformer.cpp
//
// Implementation of BitNetFeedForward, TransformerBlock, and BitNetTransformer.

#include "log3/nn/transformer.h"

#include <cmath>

#include "log3/core/rng.h"
#include "log3/kernels/cpu_kernels.h"

namespace l3 {
namespace nn {

// -----------------------------------------------------------------------------
// BitNetFeedForward (SwiGLU)
// -----------------------------------------------------------------------------

Result<BitNetFeedForward> BitNetFeedForward::create(
    Index dim, Index hidden_dim, const Allocator& allocator) noexcept {
  if (dim <= 0 || hidden_dim <= 0 || dim % 4 != 0 || hidden_dim % 4 != 0) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "BitNetFeedForward::create: dim and hidden_dim must be positive multiples of 4");
  }

  BitNetFeedForward ffn;
  ffn.dim_ = dim;
  ffn.hidden_dim_ = hidden_dim;

  L3_ASSIGN_OR_RETURN(ffn.w1_, BitLinear::create(dim, hidden_dim, allocator));
  L3_ASSIGN_OR_RETURN(ffn.w2_, BitLinear::create(hidden_dim, dim, allocator));
  L3_ASSIGN_OR_RETURN(ffn.w3_, BitLinear::create(dim, hidden_dim, allocator));

  ffn.register_parameter("w1.weight", ffn.w1_.weight());
  ffn.register_parameter("w2.weight", ffn.w2_.weight());
  ffn.register_parameter("w3.weight", ffn.w3_.weight());

  return ffn;
}

Status BitNetFeedForward::forward(const Tensor& x, Tensor* out,
                                  ThreadPool& pool) const noexcept {
  if (out == nullptr || !x.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "BitNetFeedForward::forward: invalid input or output");
  }
  if (!x.is_contiguous() || !out->is_contiguous() || !out->is_mutable()) {
    return Status::error(StatusCode::kFailedPrecondition,
                         "BitNetFeedForward::forward: tensors must be contiguous");
  }
  if (x.rank() != 2 || x.dim(1) != dim_ || out->shape() != x.shape()) {
    return Status::error(StatusCode::kShapeMismatch,
                         "BitNetFeedForward::forward: shape mismatch");
  }

  const Index seq_len = x.dim(0);
  Tensor gate;
  Tensor up;
  L3_ASSIGN_OR_RETURN(gate, Tensor::empty({seq_len, hidden_dim_}));
  L3_ASSIGN_OR_RETURN(up, Tensor::empty({seq_len, hidden_dim_}));

  L3_RETURN_IF_ERROR(w1_.forward(x, &gate, pool));
  L3_RETURN_IF_ERROR(w3_.forward(x, &up, pool));

  // SwiGLU: silu(gate) * up
  const Index n = gate.numel();
  f32* g = gate.mutable_data_ptr<f32>();
  const f32* u = up.data_ptr<f32>();

  kernels::silu_f32(g, g, n);
  kernels::mul_f32(g, u, g, n);

  // Down projection
  return w2_.forward(gate, out, pool);
}

// -----------------------------------------------------------------------------
// TransformerBlock
// -----------------------------------------------------------------------------

Result<TransformerBlock> TransformerBlock::create(
    Index layer_id, const TransformerConfig& cfg,
    const Allocator& allocator) noexcept {
  TransformerBlock block;
  block.layer_id_ = layer_id;

  L3_ASSIGN_OR_RETURN(block.attn_norm_,
                      RMSNorm::create(cfg.dim, cfg.norm_eps, allocator));
  L3_ASSIGN_OR_RETURN(
      block.attn_,
      GroupedQueryAttention::create(cfg.dim, cfg.num_heads, cfg.num_kv_heads,
                                    allocator));
  L3_ASSIGN_OR_RETURN(block.ffn_norm_,
                      RMSNorm::create(cfg.dim, cfg.norm_eps, allocator));
  L3_ASSIGN_OR_RETURN(
      block.ffn_, BitNetFeedForward::create(cfg.dim, cfg.hidden_dim, allocator));

  // Register sub-layer parameters on TransformerBlock so `block.num_parameters()`
  // and `block.for_each_parameter()` report all weights of the block!
  block.register_parameter("attn_norm.weight", block.attn_norm_.weight());
  block.register_parameter("attn.wq.weight", block.attn_.wq().weight());
  block.register_parameter("attn.wk.weight", block.attn_.wk().weight());
  block.register_parameter("attn.wv.weight", block.attn_.wv().weight());
  block.register_parameter("attn.wo.weight", block.attn_.wo().weight());
  block.register_parameter("ffn_norm.weight", block.ffn_norm_.weight());
  block.register_parameter("ffn.w1.weight", block.ffn_.w1().weight());
  block.register_parameter("ffn.w2.weight", block.ffn_.w2().weight());
  block.register_parameter("ffn.w3.weight", block.ffn_.w3().weight());

  return block;
}

Status TransformerBlock::forward(const Tensor& x, Tensor* out, KVCache* cache,
                                 ThreadPool& pool) const noexcept {
  if (out == nullptr || !x.is_valid() || !out->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "TransformerBlock::forward: invalid input or output");
  }
  if (x.rank() != 2 || out->shape() != x.shape()) {
    return Status::error(StatusCode::kShapeMismatch,
                         "TransformerBlock::forward: shape mismatch");
  }

  const Index seq_len = x.dim(0);
  const Index dim = x.dim(1);

  Tensor h;
  Tensor norm_h;
  L3_ASSIGN_OR_RETURN(h, Tensor::empty({seq_len, dim}));
  L3_ASSIGN_OR_RETURN(norm_h, Tensor::empty({seq_len, dim}));

  // 1. Attention with pre-RMSNorm and residual
  L3_RETURN_IF_ERROR(attn_norm_.forward(x, &norm_h));
  L3_RETURN_IF_ERROR(attn_.forward(norm_h, &h, cache, pool));
  kernels::add_f32(x.data_ptr<f32>(), h.data_ptr<f32>(),
                   h.mutable_data_ptr<f32>(), x.numel());

  // 2. FFN with pre-RMSNorm and residual
  L3_RETURN_IF_ERROR(ffn_norm_.forward(h, &norm_h));
  L3_RETURN_IF_ERROR(ffn_.forward(norm_h, out, pool));
  kernels::add_f32(h.data_ptr<f32>(), out->data_ptr<f32>(),
                   out->mutable_data_ptr<f32>(), x.numel());

  return Status::ok();
}

// -----------------------------------------------------------------------------
// BitNetTransformer
// -----------------------------------------------------------------------------

Result<BitNetTransformer> BitNetTransformer::create(
    const TransformerConfig& cfg, const Allocator& allocator) noexcept {
  if (cfg.vocab_size <= 0 || cfg.dim <= 0 || cfg.hidden_dim <= 0 ||
      cfg.num_layers <= 0 || cfg.num_heads <= 0 || cfg.num_kv_heads <= 0 ||
      cfg.max_seq_len <= 0) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "BitNetTransformer::create: all config dimensions must be positive");
  }

  BitNetTransformer model;
  model.cfg_ = cfg;

  L3_ASSIGN_OR_RETURN(
      model.tok_embeddings_,
      Embedding::create(cfg.vocab_size, cfg.dim, allocator));
  model.register_parameter("tok_embeddings.weight",
                           model.tok_embeddings_.weight());

  for (Index i = 0; i < cfg.num_layers; ++i) {
    TransformerBlock blk;
    L3_ASSIGN_OR_RETURN(blk, TransformerBlock::create(i, cfg, allocator));
    L3_RETURN_IF_ERROR(model.blocks_.push_back(std::move(blk)));
  }

  // Register every block's parameters into the top-level model after `blocks_`
  // is populated.
  for (Index i = 0; i < cfg.num_layers; ++i) {
    model.blocks_[i].for_each_parameter(
        [&](StrView name, Tensor& param) noexcept {
          model.register_parameter(name.data(), param);
        });
  }

  L3_ASSIGN_OR_RETURN(model.norm_,
                      RMSNorm::create(cfg.dim, cfg.norm_eps, allocator));
  L3_ASSIGN_OR_RETURN(model.output_,
                      BitLinear::create(cfg.dim, cfg.vocab_size, allocator));

  model.register_parameter("norm.weight", model.norm_.weight());
  model.register_parameter("output.weight", model.output_.weight());

  return model;
}

Status BitNetTransformer::forward(Span<const Index> tokens, Tensor* logits,
                                  Span<KVCache> caches,
                                  ThreadPool& pool) const noexcept {
  if (logits == nullptr || !logits->is_valid()) {
    return Status::error(StatusCode::kInvalidArgument,
                         "BitNetTransformer::forward: logits is null or invalid");
  }
  const Index seq_len = tokens.size();
  if (seq_len <= 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "BitNetTransformer::forward: empty token sequence");
  }
  if (!caches.empty() && caches.size() < cfg_.num_layers) {
    return Status::error(
        StatusCode::kInvalidArgument,
        "BitNetTransformer::forward: caches.size() < num_layers");
  }

  // 1. Token Embeddings
  Tensor h;
  L3_ASSIGN_OR_RETURN(h, Tensor::empty({seq_len, cfg_.dim}));
  L3_RETURN_IF_ERROR(tok_embeddings_.forward(tokens, &h));

  // 2. Transformer Blocks (ping-pong between two pre-allocated buffers `h` and
  // `next_h` via `std::swap` instead of reallocating `next_h` on every layer!)
  Tensor next_h;
  L3_ASSIGN_OR_RETURN(next_h, Tensor::empty({seq_len, cfg_.dim}));

  for (Index l = 0; l < cfg_.num_layers; ++l) {
    KVCache* c = (caches.size() > l) ? &caches[l] : nullptr;
    L3_RETURN_IF_ERROR(blocks_[l].forward(h, &next_h, c, pool));
    std::swap(h, next_h);
  }

  // 3. Final RMSNorm (in-place into `next_h`) & LM Head projection
  L3_RETURN_IF_ERROR(norm_.forward(h, &next_h));
  return output_.forward(next_h, logits, pool);
}

Status BitNetTransformer::generate(Span<const Index> prompt,
                                   Index max_new_tokens,
                                   SmallVec<Index, 128>* out_tokens,
                                   f32 temperature,
                                   ThreadPool& pool) const noexcept {
  if (out_tokens == nullptr || prompt.empty() || max_new_tokens <= 0) {
    return Status::error(StatusCode::kInvalidArgument,
                         "BitNetTransformer::generate: invalid arguments");
  }
  if (prompt.size() + max_new_tokens > cfg_.max_seq_len) {
    return Status::errorf(
        StatusCode::kOutOfRange,
        "BitNetTransformer::generate: prompt (%lld) + max_new_tokens (%lld) "
        "exceeds max_seq_len (%lld)",
        static_cast<long long>(prompt.size()),
        static_cast<long long>(max_new_tokens),
        static_cast<long long>(cfg_.max_seq_len));
  }

  out_tokens->clear();
  for (Index i = 0; i < prompt.size(); ++i) {
    L3_RETURN_IF_ERROR(out_tokens->push_back(prompt[i]));
  }

  // Pre-allocate KVCache per layer
  SmallVec<KVCache, 32> caches;
  for (Index l = 0; l < cfg_.num_layers; ++l) {
    KVCache c;
    L3_ASSIGN_OR_RETURN(
        c, KVCache::create(cfg_.max_seq_len, cfg_.num_kv_heads,
                           cfg_.dim / cfg_.num_heads));
    L3_RETURN_IF_ERROR(caches.push_back(std::move(c)));
  }

  Tensor logits;
  L3_ASSIGN_OR_RETURN(logits, Tensor::empty({1, cfg_.vocab_size}));

  // Prefill prompt
  Tensor prefill_logits;
  L3_ASSIGN_OR_RETURN(prefill_logits,
                      Tensor::empty({prompt.size(), cfg_.vocab_size}));
  L3_RETURN_IF_ERROR(forward(prompt, &prefill_logits, caches.span(), pool));

  // Scratch buffer for temperature-scaled softmax categorical sampling
  SmallVec<f32, 256> probs;
  if (temperature > 0.0f) {
    L3_RETURN_IF_ERROR(probs.resize(cfg_.vocab_size, 0.0f));
  }

  Rng rng(42ULL);

  // Generate tokens step by step
  for (Index step = 0; step < max_new_tokens; ++step) {
    Index next_token = 0;
    const f32* last_logits =
        (step == 0) ? (prefill_logits.data_ptr<f32>() +
                       (prompt.size() - 1) * cfg_.vocab_size)
                    : logits.data_ptr<f32>();

    if (temperature <= 0.0f) {
      // Greedy argmax decoding
      f32 best = last_logits[0];
      for (Index v = 1; v < cfg_.vocab_size; ++v) {
        if (last_logits[v] > best) {
          best = last_logits[v];
          next_token = v;
        }
      }
    } else {
      // Proper temperature-scaled softmax multinomial sampling:
      //   p_v = softmax(logits / temperature)_v
      const f32 inv_temp = 1.0f / temperature;
      for (Index v = 0; v < cfg_.vocab_size; ++v) {
        probs[v] = last_logits[v] * inv_temp;
      }
      kernels::softmax_f32(probs.data(), probs.data(), cfg_.vocab_size);

      const f32 u = rng.uniform_f32();
      f32 cdf = 0.0f;
      next_token = cfg_.vocab_size - 1;
      for (Index v = 0; v < cfg_.vocab_size; ++v) {
        cdf += probs[v];
        if (u <= cdf) {
          next_token = v;
          break;
        }
      }
    }

    L3_RETURN_IF_ERROR(out_tokens->push_back(next_token));

    // Only run forward decode if another step remains
    if (step + 1 < max_new_tokens) {
      const Index tok_arr[1] = {next_token};
      L3_RETURN_IF_ERROR(forward(Span<const Index>(tok_arr, 1), &logits,
                                 caches.span(), pool));
    }
  }

  return Status::ok();
}

}  // namespace nn
}  // namespace l3
