// tests/test_nn_and_io.cpp
//
// Comprehensive unit test suite for Stage 3:
//   * log3/nn/module.h       (Module base, parameter registration & visitation)
//   * log3/nn/layers.h       (RMSNorm, BitLinear, Linear, Embedding, RoPE)
//   * log3/nn/attention.h    (KVCache, GroupedQueryAttention with BitLinear)
//   * log3/nn/transformer.h  (BitNetFeedForward SwiGLU, TransformerBlock, BitNetTransformer, generation)
//   * log3/io/checkpoint.h   (ModelWriter & ModelReader zero-copy MMap serialization)

#include <cmath>
#include <cstdio>
#include <cstring>

#include "log3/core/arena.h"
#include "log3/core/assert.h"
#include "log3/core/log.h"
#include "log3/core/memory.h"
#include "log3/core/rng.h"
#include "log3/core/timer.h"
#include "log3/io/checkpoint.h"
#include "log3/nn/attention.h"
#include "log3/nn/layers.h"
#include "log3/nn/module.h"
#include "log3/nn/transformer.h"
#include "log3/tensor/tensor.h"

namespace l3 {
namespace {

bool approx_eq(f32 a, f32 b, f32 tol = 1e-4f) noexcept {
  const f32 diff = a > b ? (a - b) : (b - a);
  return diff <= tol;
}

// -----------------------------------------------------------------------------
// 1. RMSNorm, BitLinear, Linear, Embedding, and RoPE Layers
// -----------------------------------------------------------------------------
void test_nn_layers() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    // Test RMSNorm
    auto norm_res = nn::RMSNorm::create(64);
    L3_CHECK_OK(norm_res.status());
    nn::RMSNorm norm = norm_res.take_value();
    L3_CHECK_EQ(norm.dim(), 64);
    L3_CHECK_EQ(norm.num_parameters(), 64);

    auto x_res = Tensor::full({2, 64}, 2.0f, DType::kFloat32);
    auto y_res = Tensor::zeros({2, 64}, DType::kFloat32);
    L3_CHECK_OK(x_res.status());
    L3_CHECK_OK(y_res.status());
    Tensor x = x_res.take_value();
    Tensor y = y_res.take_value();

    L3_CHECK_OK(norm.forward(x, &y));
    for (Index i = 0; i < y.numel(); ++i) {
      L3_CHECK(approx_eq(y.data_ptr<f32>()[i], 1.0f, 1e-4f));
    }

    // Test Embedding
    auto emb_res = nn::Embedding::create(100, 32);
    L3_CHECK_OK(emb_res.status());
    nn::Embedding emb = emb_res.take_value();
    L3_CHECK_EQ(emb.num_parameters(), 3200);

    for (Index d = 0; d < 32; ++d) {
      emb.weight().mutable_data_ptr<f32>()[7 * 32 + d] = 42.0f;
    }

    const Index tok_seq[3] = {7, 0, 7};
    auto emb_out_res = Tensor::zeros({3, 32}, DType::kFloat32);
    L3_CHECK_OK(emb_out_res.status());
    Tensor emb_out = emb_out_res.take_value();
    L3_CHECK_OK(emb.forward(Span<const Index>(tok_seq, 3), &emb_out));
    L3_CHECK_EQ(emb_out.data_ptr<f32>()[0 * 32 + 0], 42.0f);
    L3_CHECK_EQ(emb_out.data_ptr<f32>()[1 * 32 + 0], 0.0f);
    L3_CHECK_EQ(emb_out.data_ptr<f32>()[2 * 32 + 0], 42.0f);

    // Test BitLinear
    auto bl_res = nn::BitLinear::create(64, 32);
    L3_CHECK_OK(bl_res.status());
    nn::BitLinear bl = bl_res.take_value();
    L3_CHECK_EQ(bl.in_features(), 64);
    L3_CHECK_EQ(bl.out_features(), 32);

    auto bl_out_res = Tensor::zeros({2, 32}, DType::kFloat32);
    L3_CHECK_OK(bl_out_res.status());
    Tensor bl_out = bl_out_res.take_value();
    L3_CHECK_OK(bl.forward(x, &bl_out));
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 2. KVCache & GroupedQueryAttention (GQA)
// -----------------------------------------------------------------------------
void test_attention_and_kv_cache() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    constexpr Index kDim = 64;
    constexpr Index kHeads = 4;
    constexpr Index kKVHeads = 2;
    constexpr Index kHeadDim = 16;
    constexpr Index kMaxSeq = 128;

    auto cache_res = nn::KVCache::create(kMaxSeq, kKVHeads, kHeadDim);
    L3_CHECK_OK(cache_res.status());
    nn::KVCache cache = cache_res.take_value();
    L3_CHECK_EQ(cache.current_length(), 0);

    auto gqa_res = nn::GroupedQueryAttention::create(kDim, kHeads, kKVHeads);
    L3_CHECK_OK(gqa_res.status());
    nn::GroupedQueryAttention gqa = gqa_res.take_value();

    // Prefill 4 tokens: (4, 64)
    auto x_res = Tensor::full({4, kDim}, 0.5f, DType::kFloat32);
    auto out_res = Tensor::zeros({4, kDim}, DType::kFloat32);
    L3_CHECK_OK(x_res.status());
    L3_CHECK_OK(out_res.status());
    Tensor x = x_res.take_value();
    Tensor out = out_res.take_value();

    L3_CHECK_OK(gqa.forward(x, &out, &cache));
    L3_CHECK_EQ(cache.current_length(), 4);

    // Decode single next token (1, 64)
    auto next_x_res = Tensor::full({1, kDim}, 0.5f, DType::kFloat32);
    auto next_out_res = Tensor::zeros({1, kDim}, DType::kFloat32);
    L3_CHECK_OK(next_x_res.status());
    L3_CHECK_OK(next_out_res.status());
    Tensor next_x = next_x_res.take_value();
    Tensor next_out = next_out_res.take_value();

    L3_CHECK_OK(gqa.forward(next_x, &next_out, &cache));
    L3_CHECK_EQ(cache.current_length(), 5);
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 3. BitNetTransformer Architecture & Token Generation
// -----------------------------------------------------------------------------
void test_transformer_generation() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  {
    nn::TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.dim = 32;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.max_seq_len = 64;

    auto model_res = nn::BitNetTransformer::create(cfg);
    L3_CHECK_OK(model_res.status());
    nn::BitNetTransformer model = model_res.take_value();

    const Index prompt[3] = {1, 5, 12};
    SmallVec<Index, 128> generated;
    L3_CHECK_OK(model.generate(Span<const Index>(prompt, 3), 4, &generated));
    L3_CHECK_EQ(generated.size(), 7);
    L3_CHECK_EQ(generated[0], 1);
    L3_CHECK_EQ(generated[1], 5);
    L3_CHECK_EQ(generated[2], 12);
    for (Index i = 3; i < 7; ++i) {
      L3_CHECK_GE(generated[i], 0);
      L3_CHECK_LT(generated[i], cfg.vocab_size);
    }
  }
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

// -----------------------------------------------------------------------------
// 4. IO Checkpoint Serialization & Zero-Copy Deserialization
// -----------------------------------------------------------------------------
void test_checkpoint_io() {
  (void)default_thread_pool();
  const u64 mem_before = memory_stats().current_bytes;
  const char* checkpoint_path = "log3_test_model.l3";
  {
    io::ModelWriter writer;

    auto t1_res = Tensor::full({4, 16}, 3.14f, DType::kFloat32);
    auto t2_res = Tensor::empty({8, 16}, DType::kTernary2);
    L3_CHECK_OK(t1_res.status());
    L3_CHECK_OK(t2_res.status());
    Tensor t1 = t1_res.take_value();
    Tensor t2 = t2_res.take_value();
    t2.set_quant_scale(0.25f);
    mem_fill(t2.raw_mutable_data(), 0xAAu, t2.nbytes());

    L3_CHECK_OK(writer.add_tensor("model.layers.0.attn.wq.weight", t1));
    L3_CHECK_OK(writer.add_tensor("model.layers.0.attn.wk.weight", t2));
    L3_CHECK_OK(writer.save(checkpoint_path));
  }

  {
    auto reader_res = io::ModelReader::open(checkpoint_path, /*prefetch=*/true);
    L3_CHECK_OK(reader_res.status());
    io::ModelReader reader = reader_res.take_value();

    L3_CHECK_EQ(reader.num_tensors(), 2);
    L3_CHECK(reader.has_tensor("model.layers.0.attn.wq.weight"));
    L3_CHECK(reader.has_tensor("model.layers.0.attn.wk.weight"));
    L3_CHECK(!reader.has_tensor("nonexistent.tensor"));

    auto loaded_t1_res = reader.get_tensor("model.layers.0.attn.wq.weight");
    auto loaded_t2_res = reader.get_tensor("model.layers.0.attn.wk.weight");
    L3_CHECK_OK(loaded_t1_res.status());
    L3_CHECK_OK(loaded_t2_res.status());
    Tensor loaded_t1 = loaded_t1_res.take_value();
    Tensor loaded_t2 = loaded_t2_res.take_value();

    L3_CHECK_EQ(loaded_t1.dtype(), DType::kFloat32);
    L3_CHECK(loaded_t1.shape() == *Shape::from_dims({4, 16}));
    L3_CHECK_EQ(loaded_t1.data_ptr<f32>()[0], 3.14f);

    L3_CHECK_EQ(loaded_t2.dtype(), DType::kTernary2);
    L3_CHECK(loaded_t2.shape() == *Shape::from_dims({8, 16}));
    L3_CHECK_EQ(loaded_t2.quant_scale(), 0.25f);
    L3_CHECK_EQ(loaded_t2.raw_data()[0], 0xAAu);
  }

  std::remove(checkpoint_path);
  L3_CHECK_EQ(memory_stats().current_bytes, mem_before);
}

}  // namespace
}  // namespace l3

int main() {
  l3::set_log_level(l3::LogLevel::kInfo);
  L3_LOG_INFO("Running tests/test_nn_and_io.cpp...");

  l3::test_nn_layers();
  l3::test_attention_and_kv_cache();
  l3::test_transformer_generation();
  l3::test_checkpoint_io();

  L3_LOG_INFO(
      "tests/test_nn_and_io.cpp: ALL STAGE 3 TESTS PASSED (0 bytes leaked).");
  return 0;
}
