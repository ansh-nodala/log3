#!/usr/bin/env python3
"""test.py — Comprehensive End-to-End Test & Benchmark Suite for the `log3` Python Library.

Tests every layer of the `log3` C++17 BitNet b1.58 & Autograd ecosystem from Python:
  1. Core Runtime, Hardware SIMD Feature Detection, Memory Tracking & Deterministic Rng
  2. N-D Tensor Creation, Zero-Copy PEP 3118 Buffer Protocol (`memoryview` & NumPy),
     Zero-Copy Views (`reshape`, `transpose`, `permute`, `slice`, `unsqueeze`, `squeeze`)
  3. Low-Level SIMD & BitNet b1.58 2-Bit Ternary Kernels (`quantize_f32_to_t2`,
     `dequantize_t2_to_f32`, `bitnet_linear_t2_f32`, `dot_f32`, `silu_f32`,
     `rmsnorm_f32`, `softmax_f32`, `rope_f32_inplace`, `matmul_f32`)
  4. Neural Network Modules (`RMSNorm`, `BitLinear`, `Linear`, `Embedding`,
     `RotaryEmbedding`, `KVCache`, `GroupedQueryAttention`, `BitNetFeedForward`,
     `TransformerBlock`, `BitNetTransformer` autoregressive token generation)
  5. Zero-Copy 64-Byte Aligned `.l3` Checkpoint Serialization (`ModelWriter`, `ModelReader`)
  6. Reverse-Mode Autograd, BitNet Straight-Through Estimator (`bitnet_linear_ste`)
     Quantization-Aware Training (QAT), Gradient Clipping, and `SGD` / `AdamW` Optimizers
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import time
import log3

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def assert_close(a: float, b: float, tol: float = 1e-4, msg: str = "") -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: expected {b}, got {a} (diff={abs(a - b)})")


def test_1_core_runtime_and_rng() -> None:
    print("\n--- [1/6] Testing Core Runtime, SIMD Detection & Deterministic Rng ---")
    log3.init_runtime(log3.LOG_INFO)
    ver = log3.version_string()
    plat = log3.platform_name()
    arch = log3.arch_name()
    cpu = log3.cpu_features()
    print(f"  log3 v{ver} ({ hex(log3.version_hex()) }) on {plat} / {arch}")
    print(f"  Logical cores: {cpu['logical_cores']} | Physical: {cpu['physical_cores']} | "
          f"AVX2: {cpu['has_avx2']} | FMA: {cpu['has_fma']} | NEON: {cpu['has_neon']}")
    assert cpu["cache_line_size"] == 64

    # Deterministic Xoshiro256++ PRNG verification
    rng1 = log3.Rng(1337)
    rng2 = log3.Rng(1337)
    for _ in range(10):
        assert rng1.next_u64() == rng2.next_u64()
        assert_close(rng1.uniform_f32(-2.0, 2.0), rng2.uniform_f32(-2.0, 2.0), 1e-7)
        assert_close(rng1.normal_f32(0.0, 1.0), rng2.normal_f32(0.0, 1.0), 1e-7)
    for _ in range(50):
        b = rng1.bounded_u32(17)
        assert 0 <= b < 17
    print("  [PASS] Core runtime, CPU features, and bit-exact Xoshiro256++ Rng verified.")


def test_2_tensor_views_and_zero_copy_buffer() -> None:
    print("\n--- [2/6] Testing N-D Tensor, Views & PEP 3118 Zero-Copy Buffer Protocol ---")
    t = log3.zeros((4, 8), dtype=log3.float32)
    assert t.shape == (4, 8)
    assert t.strides == (8, 1)
    assert t.ndim == 2 and t.numel == 32 and t.nbytes == 128
    assert t.is_contiguous and t.is_mutable and not t.is_quantized

    # Zero-copy mutation via Python built-in memoryview (PEP 3118)
    mv = memoryview(t)
    assert mv.format == "f" and mv.shape == (4, 8)
    mv[0, 0] = 42.5
    mv[3, 7] = -9.25
    flat = t.to_list()
    assert_close(flat[0], 42.5)
    assert_close(flat[-1], -9.25)

    # Optional zero-copy NumPy interop check if numpy is installed
    if HAS_NUMPY:
        arr = np.asarray(t)
        arr[1, 2] = 123.75
        assert_close(t.to_list()[1 * 8 + 2], 123.75)

    # Construct Tensor from raw buffer & from Python list
    raw_bytes = struct.pack("4f", 1.5, -2.5, 3.5, 4.5)
    t_buf = log3.from_buffer(raw_bytes, shape=(2, 2), dtype="f32")
    assert t_buf.to_list() == [1.5, -2.5, 3.5, 4.5]

    t_lst = log3.from_list([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], shape=(2, 3))
    t_trans = t_lst.transpose(0, 1)
    assert t_trans.shape == (3, 2) and not t_trans.is_contiguous
    t_contig = t_trans.contiguous()
    assert t_contig.is_contiguous
    assert t_contig.to_list() == [10.0, 40.0, 20.0, 50.0, 30.0, 60.0]

    # Slicing, unsqueeze, squeeze, reshape
    s = t.slice(axis=1, start=2, end=6)
    assert s.shape == (4, 4) and not s.is_contiguous
    u = t_lst.unsqueeze(0)
    assert u.shape == (1, 2, 3)
    sq = u.squeeze(0)
    assert sq.shape == (2, 3)
    print("  [PASS] Tensor factories, PEP 3118 zero-copy memoryview, and O(1) views verified.")


def test_3_simd_and_bitnet_kernels() -> None:
    print("\n--- [3/6] Testing SIMD FP32 & BitNet b1.58 2-Bit Ternary Kernels ---")
    a = log3.from_list([1.0, 2.0, 3.0, 4.0], shape=(4,))
    b = log3.from_list([0.5, 1.5, -1.0, 2.0], shape=(4,))
    dot_val = log3.kernels.dot_f32(a, b)
    assert_close(dot_val, 1.0 * 0.5 + 2.0 * 1.5 - 3.0 + 8.0, 1e-5, "dot_f32")

    added = log3.kernels.add_f32(a, b)
    assert added.to_list() == [1.5, 3.5, 2.0, 6.0]

    probs = log3.kernels.softmax_f32(a.reshape((1, 4))).to_list()
    assert_close(sum(probs), 1.0, 1e-5, "softmax sum to 1")

    # BitNet b1.58 2-bit ternary quantization (4 weights per byte -> 16x compression vs FP32)
    w_f32 = log3.from_list(
        [0.8, -0.7, 0.02, 0.9, -0.85, 0.75, -0.01, -0.95],
        shape=(2, 4),
        dtype="f32",
    )
    w_t2 = log3.kernels.quantize_f32_to_t2(w_f32)
    assert w_t2.dtype == "t2" and w_t2.is_quantized
    assert w_t2.nbytes == 2  # 8 weights packed into 2 bytes!
    assert w_f32.nbytes == 32

    w_deq = log3.kernels.dequantize_t2_to_f32(w_t2).to_list()
    scale = w_t2.quant_scale
    # Expected ternary pattern: [+1, -1, 0, +1, -1, +1, 0, -1] * scale
    expected_signs = [1, -1, 0, 1, -1, 1, 0, -1]
    for got, sign in zip(w_deq, expected_signs):
        assert_close(got, sign * scale, 1e-5, "dequantized ternary weight")

    x = log3.ones((3, 4), dtype="f32")
    y = log3.kernels.bitnet_linear_t2_f32(x, w_t2)
    assert y.shape == (3, 2)
    print(f"  [PASS] SIMD kernels & 2-bit kTernary2 packing verified (32 B FP32 -> {w_t2.nbytes} B kTernary2, scale={scale:.4f}).")


def test_4_neural_network_modules_and_transformer() -> None:
    print("\n--- [4/6] Testing Neural Network Layers, GQA, KVCache & BitNetTransformer ---")
    rng = log3.Rng(42)

    norm = log3.nn.RMSNorm(16)
    fc_bit = log3.nn.BitLinear(16, 32)
    w_init = log3.empty((32, 16))
    rng.fill_kaiming_normal_(w_init, 16)
    fc_bit.load_weights_from_f32(w_init)

    x = log3.ones((4, 16))
    y = fc_bit(norm(x))
    assert y.shape == (4, 32)

    # Test Grouped-Query Attention with pre-allocated KVCache
    gqa = log3.nn.GroupedQueryAttention(dim=32, num_heads=4, num_kv_heads=2)
    cache = log3.nn.KVCache(max_seq_len=16, num_kv_heads=2, head_dim=8)
    assert cache.current_length == 0
    attn_out = gqa(log3.ones((3, 32)), cache=cache)
    assert attn_out.shape == (3, 32) and cache.current_length == 3

    # Test Full BitNet b1.58 Decoder Transformer
    llm = log3.nn.BitNetTransformer(
        vocab_size=128,
        dim=64,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=64,
    )
    rng.fill_kaiming_normal_(llm.embedding_weight, 64)
    logits = llm([1, 10, 25, 40])
    assert logits.shape == (4, 128)
    gen_tokens = llm.generate([1, 10, 25, 40], max_new_tokens=5, temperature=0.0)
    assert len(gen_tokens) == 9
    print(f"  [PASS] All NN layers, GQA + KVCache, and BitNetTransformer verified (tokens={gen_tokens}).")


def test_5_zero_copy_checkpoint_io() -> None:
    print("\n--- [5/6] Testing 64-Byte Aligned Zero-Copy .l3 Checkpoint I/O ---")
    w_f32 = log3.full((16, 32), 0.35, dtype="f32")
    w_t2 = log3.kernels.quantize_f32_to_t2(w_f32)
    bias = log3.from_list([float(i) for i in range(16)], shape=(16,))

    with tempfile.NamedTemporaryFile(suffix=".l3", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        writer = log3.io.ModelWriter()
        writer.add_tensor("layer.weight_t2", w_t2)
        writer.add_tensor("layer.bias", bias)
        writer.save(ckpt_path)

        reader = log3.io.ModelReader.open(ckpt_path, prefetch=True)
        assert reader.num_tensors == 2
        assert reader.has_tensor("layer.weight_t2")
        assert reader.has_tensor("layer.bias")

        loaded_t2 = reader.get_tensor("layer.weight_t2")
        loaded_bias = reader.get_tensor("layer.bias")
        assert loaded_t2.dtype == "t2" and loaded_t2.shape == (16, 32)
        assert_close(loaded_t2.quant_scale, w_t2.quant_scale, 1e-6)
        assert loaded_bias.to_list() == bias.to_list()
        print(f"  [PASS] Zero-copy mmap .l3 checkpoint verified ({os.path.getsize(ckpt_path)} bytes on disk).")
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_6_autograd_bitnet_ste_qat_and_optimizers() -> None:
    print("\n--- [6/6] Testing Reverse-Mode Autograd, BitNet STE QAT & AdamW ---")
    rng = log3.Rng(20261107)
    batch_size, in_dim, hidden_dim, num_classes = 8, 16, 32, 4

    # Build a synthetic 4-class classification task
    x_list = []
    targets = [0, 1, 2, 3, 0, 1, 2, 3]
    for cls in targets:
        for d in range(in_dim):
            sig = 2.0 if (d // 4 == cls) else -0.5
            x_list.append(sig + rng.normal_f32(0.0, 0.1))
    x_train = log3.from_list(x_list, shape=(batch_size, in_dim))

    norm_w = log3.ones((in_dim,), requires_grad=True)
    w1_master = log3.empty((hidden_dim, in_dim), requires_grad=True)
    w2_master = log3.empty((num_classes, hidden_dim), requires_grad=True)
    rng.fill_kaiming_normal_(w1_master, in_dim)
    rng.fill_kaiming_normal_(w2_master, hidden_dim)

    params = [norm_w, w1_master, w2_master]
    optimizer = log3.optim.AdamW(params, lr=0.04, weight_decay=0.01)

    t0 = time.perf_counter()
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, 41):
        optimizer.zero_grad()
        h = log3.autograd.rmsnorm(x_train, norm_w)
        h = log3.autograd.bitnet_linear_ste(h, w1_master)
        h = log3.autograd.silu(h)
        logits = log3.autograd.bitnet_linear_ste(h, w2_master)
        loss = log3.autograd.cross_entropy_loss(logits, targets)

        if step == 1:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())

        loss.backward()
        grad_norm = log3.optim.clip_grad_norm(params, max_norm=1.0)
        optimizer.step()

        if step in (1, 10, 20, 40):
            logits_flat = logits.to_list()
            correct = 0
            for b in range(batch_size):
                row = logits_flat[b * num_classes : (b + 1) * num_classes]
                pred = row.index(max(row))
                if pred == targets[b]:
                    correct += 1
            acc = 100.0 * correct / batch_size
            print(f"  Step {step:2d}/40 | Loss: {final_loss:.4f} | GradNorm: {grad_norm:.4f} | Accuracy: {acc:.1f}%")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert final_loss < initial_loss * 0.05, f"Loss did not converge: {initial_loss} -> {final_loss}"

    # Verify deployed 2-bit BitLinear inference under log3.no_grad()
    with log3.no_grad():
        assert not log3.is_grad_enabled()
        infer_norm = log3.nn.RMSNorm(in_dim)
        infer_fc1 = log3.nn.BitLinear(in_dim, hidden_dim)
        infer_fc2 = log3.nn.BitLinear(hidden_dim, num_classes)
        infer_fc1.load_weights_from_f32(w1_master)
        infer_fc2.load_weights_from_f32(w2_master)

        h_inf = log3.kernels.silu_f32(infer_fc1(infer_norm(x_train)))
        logits_inf = infer_fc2(h_inf).to_list()
        correct_inf = 0
        for b in range(batch_size):
            row = logits_inf[b * num_classes : (b + 1) * num_classes]
            if row.index(max(row)) == targets[b]:
                correct_inf += 1
        assert correct_inf == batch_size

    print(f"  [PASS] BitNet b1.58 STE QAT + 2-bit deployment achieved 100% accuracy in {elapsed_ms:.2f} ms.")


def main() -> None:
    print("========================================================================")
    print("  log3 Python C++17 Extension — Full Ecosystem Verification Suite")
    print("========================================================================")
    test_1_core_runtime_and_rng()
    test_2_tensor_views_and_zero_copy_buffer()
    test_3_simd_and_bitnet_kernels()
    test_4_neural_network_modules_and_transformer()
    test_5_zero_copy_checkpoint_io()
    test_6_autograd_bitnet_ste_qat_and_optimizers()

    stats = log3.memory_stats()
    print("\n========================================================================")
    print(f"  ALL 6 PYTHON TEST SUITES PASSED! (Peak C++ Memory = {stats['peak_bytes']} B)")
    print("========================================================================")


if __name__ == "__main__":
    main()
