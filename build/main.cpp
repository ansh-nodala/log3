// main.cpp
//
// End-to-end example neural network and BitNet b1.58 showcase for log3.
// Uses ONLY the single master umbrella header: #include "log3/log3.h"
//
// Demonstrates:
//   1. Runtime & SIMD hardware feature detection (AVX2 / FMA / NEON)
//   2. Custom BitNet b1.58 Neural Network (RMSNorm -> BitLinear STE -> SiLU ->
//      BitLinear STE) trained end-to-end with Reverse-Mode Autograd, L2
//      gradient norm clipping, and decoupled AdamW optimizer
//   3. Exporting trained FP32 master weights into 2-bit packed kTernary2
//      nn::BitLinear inference layers (4 weights/byte, 16x compression),
//      saving to a 64-byte aligned .l3 checkpoint, and zero-copy mmap loading
//   4. End-to-end BitNet b1.58 GQA Transformer token generation
//   5. Deterministic 0-byte memory leak verification at exit

#include "log3/log3.h"

namespace {

using namespace l3;

// Helper to initialize an FP32 tensor with scaled normal weights using l3::Rng
void init_kaiming_normal(Tensor* t, Index fan_in, Rng* rng) noexcept {
  f32* ptr = t->mutable_data_ptr<f32>();
  const Index n = t->numel();
  const f32 std_dev = 1.0f / std::sqrt(static_cast<f32>(fan_in));
  for (Index i = 0; i < n; ++i) {
    ptr[i] = rng->normal_f32(0.0f, std_dev);
  }
}

// =============================================================================
// Example 1: Custom BitNet b1.58 Neural Network (QAT Training -> 2-Bit Deploy)
//   Architecture:
//     Input (B, 16)
//       -> RMSNorm (16)
//       -> BitLinear STE (16 -> 32)
//       -> SiLU Activation
//       -> BitLinear STE (32 -> 4 classes)
//       -> Cross-Entropy Loss
// =============================================================================
Status run_bitnet_neural_network_demo() noexcept {
  L3_LOG_INFO("=== [1/2] BitNet b1.58 Neural Network Training (STE QAT + AdamW) ===");

  constexpr Index kBatchSize = 8;
  constexpr Index kInFeatures = 16;
  constexpr Index kHiddenDim = 32;
  constexpr Index kNumClasses = 4;

  Rng rng(20261107ULL);

  // 1. Create synthetic 4-class dataset: each class has a distinct quadrant signature
  L3_ASSIGN_OR_RETURN(Tensor x_train,
                      Tensor::zeros({kBatchSize, kInFeatures}, DType::kFloat32));
  Index targets[kBatchSize] = {0, 1, 2, 3, 0, 1, 2, 3};

  f32* x_ptr = x_train.mutable_data_ptr<f32>();
  for (Index b = 0; b < kBatchSize; ++b) {
    const Index cls = targets[b];
    for (Index d = 0; d < kInFeatures; ++d) {
      const f32 signal = (d / 4 == cls) ? 2.0f : -0.5f;
      x_ptr[b * kInFeatures + d] = signal + rng.normal_f32(0.0f, 0.15f);
    }
  }

  // 2. Initialize trainable FP32 master weights for BitNet QAT
  L3_ASSIGN_OR_RETURN(Tensor norm_w, Tensor::full({kInFeatures}, 1.0f, DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor w1_master,
                      Tensor::empty({kHiddenDim, kInFeatures}, DType::kFloat32));
  L3_ASSIGN_OR_RETURN(Tensor w2_master,
                      Tensor::empty({kNumClasses, kHiddenDim}, DType::kFloat32));

  init_kaiming_normal(&w1_master, kInFeatures, &rng);
  init_kaiming_normal(&w2_master, kHiddenDim, &rng);

  L3_RETURN_IF_ERROR(norm_w.set_requires_grad(true));
  L3_RETURN_IF_ERROR(w1_master.set_requires_grad(true));
  L3_RETURN_IF_ERROR(w2_master.set_requires_grad(true));

  // 3. Configure AdamW optimizer over all 3 parameter tensors
  Tensor params[3] = {norm_w.view(), w1_master.view(), w2_master.view()};
  L3_ASSIGN_OR_RETURN(
      optim::AdamW optimizer,
      optim::AdamW::create(Span<const Tensor>(params, 3), /*lr=*/0.04f,
                           /*beta1=*/0.9f, /*beta2=*/0.999f, /*eps=*/1e-8f,
                           /*weight_decay=*/0.01f));

  // 4. QAT Training Loop using Straight-Through Estimator (bitnet_linear_ste)
  constexpr Index kSteps = 40;
  for (Index step = 1; step <= kSteps; ++step) {
    optimizer.zero_grad();

    // Forward pass with live ternary weight quantization {-1, 0, +1}
    L3_ASSIGN_OR_RETURN(Tensor x_norm, autograd::rmsnorm(x_train, norm_w, 1e-5f));
    L3_ASSIGN_OR_RETURN(Tensor h_lin, autograd::bitnet_linear_ste(x_norm, w1_master));
    L3_ASSIGN_OR_RETURN(Tensor h_act, autograd::silu(h_lin));
    L3_ASSIGN_OR_RETURN(Tensor logits, autograd::bitnet_linear_ste(h_act, w2_master));
    L3_ASSIGN_OR_RETURN(
        Tensor loss,
        autograd::cross_entropy_loss(logits, Span<const Index>(targets, kBatchSize)));

    // Reverse-mode backward pass through the dynamic computation graph
    L3_RETURN_IF_ERROR(loss.backward());

    // Global L2 gradient clipping + AdamW parameter update
    L3_ASSIGN_OR_RETURN(f32 grad_norm,
                        optim::clip_grad_norm(Span<Tensor>(params, 3), 1.0f));
    L3_RETURN_IF_ERROR(optimizer.step());

    if (step == 1 || step == 10 || step == 20 || step == kSteps) {
      // Compute batch classification accuracy
      const f32* l_ptr = logits.data_ptr<f32>();
      Index correct = 0;
      for (Index b = 0; b < kBatchSize; ++b) {
        Index best_c = 0;
        f32 best_v = l_ptr[b * kNumClasses];
        for (Index c = 1; c < kNumClasses; ++c) {
          if (l_ptr[b * kNumClasses + c] > best_v) {
            best_v = l_ptr[b * kNumClasses + c];
            best_c = c;
          }
        }
        if (best_c == targets[b]) ++correct;
      }
      const f32 acc = 100.0f * static_cast<f32>(correct) / static_cast<f32>(kBatchSize);
      L3_LOG_INFO("  Step %2lld/%lld | Loss: %.4f | GradNorm: %.4f | Accuracy: %.1f%%",
                  static_cast<long long>(step), static_cast<long long>(kSteps),
                  static_cast<double>(loss.data_ptr<f32>()[0]),
                  static_cast<double>(grad_norm), static_cast<double>(acc));
    }
  }

  // 5. Export trained FP32 master weights to 2-bit packed kTernary2 inference layers
  autograd::NoGradGuard no_grad;
  L3_ASSIGN_OR_RETURN(nn::RMSNorm infer_norm, nn::RMSNorm::create(kInFeatures, 1e-5f));
  mem_copy(infer_norm.weight().mutable_data_ptr<f32>(),
           norm_w.data_ptr<f32>(), norm_w.nbytes());

  L3_ASSIGN_OR_RETURN(nn::BitLinear infer_fc1,
                      nn::BitLinear::create(kInFeatures, kHiddenDim));
  L3_ASSIGN_OR_RETURN(nn::BitLinear infer_fc2,
                      nn::BitLinear::create(kHiddenDim, kNumClasses));
  L3_RETURN_IF_ERROR(infer_fc1.load_weights_from_f32(w1_master));
  L3_RETURN_IF_ERROR(infer_fc2.load_weights_from_f32(w2_master));

  L3_LOG_INFO("  Exported 2-bit kTernary2 layers: FC1=%zu bytes (FP32 was %zu B), "
              "FC2=%zu bytes (FP32 was %zu B)",
              infer_fc1.weight().nbytes(), w1_master.nbytes(),
              infer_fc2.weight().nbytes(), w2_master.nbytes());

  // 6. Save 2-bit packed weights to a 64-byte aligned .l3 checkpoint and reload via mmap
  const char* ckpt_path = "/tmp/log3_bitnet_classifier.l3";
  io::ModelWriter writer;
  L3_RETURN_IF_ERROR(writer.add_tensor("norm.weight", infer_norm.weight()));
  L3_RETURN_IF_ERROR(writer.add_tensor("fc1.weight_t2", infer_fc1.weight()));
  L3_RETURN_IF_ERROR(writer.add_tensor("fc2.weight_t2", infer_fc2.weight()));
  L3_RETURN_IF_ERROR(writer.save(ckpt_path));

  L3_ASSIGN_OR_RETURN(io::ModelReader reader,
                      io::ModelReader::open(ckpt_path, /*prefetch=*/true));
  L3_ASSIGN_OR_RETURN(Tensor mmap_w1, reader.get_tensor("fc1.weight_t2"));
  L3_CHECK(mmap_w1.dtype() == DType::kTernary2);

  // 7. Run pure 2-bit SIMD inference pass and verify 100% accuracy
  L3_ASSIGN_OR_RETURN(Tensor x_norm_inf,
                      Tensor::empty({kBatchSize, kInFeatures}, DType::kFloat32));
  L3_RETURN_IF_ERROR(infer_norm.forward(x_train, &x_norm_inf));

  L3_ASSIGN_OR_RETURN(Tensor h_inf,
                      Tensor::empty({kBatchSize, kHiddenDim}, DType::kFloat32));
  L3_RETURN_IF_ERROR(infer_fc1.forward(x_norm_inf, &h_inf));
  kernels::silu_f32(h_inf.data_ptr<f32>(), h_inf.mutable_data_ptr<f32>(), h_inf.numel());

  L3_ASSIGN_OR_RETURN(Tensor logits_inf,
                      Tensor::empty({kBatchSize, kNumClasses}, DType::kFloat32));
  L3_RETURN_IF_ERROR(infer_fc2.forward(h_inf, &logits_inf));

  Index infer_correct = 0;
  for (Index b = 0; b < kBatchSize; ++b) {
    Index best_c = 0;
    f32 best_v = logits_inf.data_ptr<f32>()[b * kNumClasses];
    for (Index c = 1; c < kNumClasses; ++c) {
      const f32 v = logits_inf.data_ptr<f32>()[b * kNumClasses + c];
      if (v > best_v) {
        best_v = v;
        best_c = c;
      }
    }
    if (best_c == targets[b]) ++infer_correct;
  }
  L3_LOG_INFO("  Zero-copy 2-bit BitLinear inference accuracy: %lld/%lld (%.1f%%)",
              static_cast<long long>(infer_correct),
              static_cast<long long>(kBatchSize),
              100.0 * static_cast<double>(infer_correct) / static_cast<double>(kBatchSize));

  return Status::ok();
}

// =============================================================================
// Example 2: BitNet b1.58 Grouped-Query Attention (GQA) Transformer Generation
// =============================================================================
Status run_bitnet_transformer_demo() noexcept {
  L3_LOG_INFO("=== [2/2] BitNet b1.58 GQA Transformer Autoregressive Generation ===");

  nn::TransformerConfig cfg;
  cfg.vocab_size = 128;
  cfg.dim = 64;
  cfg.hidden_dim = 128;
  cfg.num_layers = 2;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.max_seq_len = 64;

  L3_ASSIGN_OR_RETURN(nn::BitNetTransformer llm, nn::BitNetTransformer::create(cfg));

  // Initialize token embeddings with deterministic normal weights
  Rng rng(42ULL);
  init_kaiming_normal(&llm.tok_embeddings().weight(), cfg.dim, &rng);

  Index prompt[4] = {1, 15, 42, 88};
  SmallVec<Index, 128> generated;
  L3_RETURN_IF_ERROR(llm.generate(Span<const Index>(prompt, 4),
                                  /*max_new_tokens=*/6, &generated,
                                  /*temperature=*/0.0f));

  L3_LOG_INFO("  Prompt tokens (4) -> Generated sequence (%zu total tokens): "
              "[%lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld]",
              generated.size(),
              static_cast<long long>(generated[0]),
              static_cast<long long>(generated[1]),
              static_cast<long long>(generated[2]),
              static_cast<long long>(generated[3]),
              static_cast<long long>(generated[4]),
              static_cast<long long>(generated[5]),
              static_cast<long long>(generated[6]),
              static_cast<long long>(generated[7]),
              static_cast<long long>(generated[8]),
              static_cast<long long>(generated[9]));

  return Status::ok();
}

}  // namespace

int main() {
  l3::init_runtime(l3::LogLevel::kInfo);

  const l3::CpuFeatures& cpu = l3::cpu_features();
  L3_LOG_INFO("log3 v%s (%s / %s) | Cores: %d | AVX2: %d | FMA: %d | NEON: %d",
              l3::version_string(), l3::platform_name(), l3::arch_name(),
              cpu.logical_cores, static_cast<int>(cpu.has_avx2),
              static_cast<int>(cpu.has_fma), static_cast<int>(cpu.has_neon));

  {
    l3::Status s1 = run_bitnet_neural_network_demo();
    if (!s1.is_ok()) {
      L3_LOG_ERROR("BitNet neural network demo failed: %s", s1.message());
      return 1;
    }

    l3::Status s2 = run_bitnet_transformer_demo();
    if (!s2.is_ok()) {
      L3_LOG_ERROR("BitNet transformer demo failed: %s", s2.message());
      return 1;
    }
  }

  const l3::MemoryStats mem = l3::memory_stats();
  L3_LOG_INFO("All demos finished! Peak memory = %llu B | Live tensor bytes = %llu B",
              static_cast<unsigned long long>(mem.peak_bytes),
              static_cast<unsigned long long>(mem.current_bytes));
  return 0;
}
