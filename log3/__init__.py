# log3/__init__.py
"""log3: High-Performance C++17 BitNet b1.58 (2-Bit Ternary), N-D Tensor,
SIMD Kernels, GQA Transformer, Zero-Copy .l3 I/O, and Reverse-Mode Autograd Library.

Exposes both flat top-level symbols (`log3.Tensor`, `log3.zeros`, `log3.BitLinear`,
`log3.AdamW`, ...) and structured sub-namespaces (`log3.nn`, `log3.autograd`,
`log3.kernels`, `log3.io`, `log3.optim`) backed by the native `log3._C` C++17
extension module.
"""

from __future__ import annotations

import types
from . import _C
from ._C import (
    # Constants
    LOG_TRACE,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_FATAL,
    LOG_OFF,
    # Core & Runtime
    init_runtime,
    set_log_level,
    log_level,
    version_string,
    version_hex,
    platform_name,
    arch_name,
    cpu_features,
    memory_stats,
    reset_peak_memory,
    Rng,
    # Tensor & Factories
    Tensor,
    empty,
    zeros,
    ones,
    full,
    from_list,
    from_buffer,
    # Low-Level SIMD & BitNet Kernels
    quantize_f32_to_t2,
    dequantize_t2_to_f32,
    bitnet_linear_t2_f32,
    dot_f32,
    add_f32,
    mul_f32,
    silu_f32,
    rmsnorm_f32,
    softmax_f32,
    rope_f32_inplace,
    matmul_f32,
    # Reverse-Mode Autograd & BitNet STE QAT
    NoGradGuard,
    is_grad_enabled,
    set_grad_enabled,
    add,
    sub,
    mul,
    silu,
    relu,
    rmsnorm,
    matmul,
    linear,
    bitnet_linear_ste,
    sum,
    mean,
    mse_loss,
    cross_entropy_loss,
    # Neural Network Layers & Transformer
    RMSNorm,
    BitLinear,
    Linear,
    Embedding,
    RotaryEmbedding,
    KVCache,
    GroupedQueryAttention,
    BitNetFeedForward,
    TransformerBlock,
    BitNetTransformer,
    # Zero-Copy .l3 Checkpoint I/O
    ModelWriter,
    ModelReader,
    # Optimizers & Gradient Clipping
    clip_grad_norm,
    SGD,
    AdamW,
)

__version__ = version_string()

# Canonical DType string constants
float32 = "f32"
float16 = "f16"
bfloat16 = "bf16"
float64 = "f64"
int32 = "i32"
int64 = "i64"
int16 = "i16"
int8 = "i8"
uint8 = "u8"
bool_ = "bool"
ternary2 = "t2"
int4 = "i4"
uint4 = "u4"


def no_grad() -> NoGradGuard:
    """Return a context manager that disables reverse-mode Autograd graph construction.

    Example:
        >>> with log3.no_grad():
        ...     y = model(x)
    """
    return NoGradGuard()


# Structured sub-namespaces (`log3.nn`, `log3.autograd`, `log3.kernels`, `log3.io`, `log3.optim`)
nn = types.SimpleNamespace(
    RMSNorm=RMSNorm,
    BitLinear=BitLinear,
    Linear=Linear,
    Embedding=Embedding,
    RotaryEmbedding=RotaryEmbedding,
    KVCache=KVCache,
    GroupedQueryAttention=GroupedQueryAttention,
    BitNetFeedForward=BitNetFeedForward,
    TransformerBlock=TransformerBlock,
    BitNetTransformer=BitNetTransformer,
)

autograd = types.SimpleNamespace(
    NoGradGuard=NoGradGuard,
    no_grad=no_grad,
    is_grad_enabled=is_grad_enabled,
    set_grad_enabled=set_grad_enabled,
    add=add,
    sub=sub,
    mul=mul,
    silu=silu,
    relu=relu,
    rmsnorm=rmsnorm,
    matmul=matmul,
    linear=linear,
    bitnet_linear_ste=bitnet_linear_ste,
    sum=sum,
    mean=mean,
    mse_loss=mse_loss,
    cross_entropy_loss=cross_entropy_loss,
)

kernels = types.SimpleNamespace(
    quantize_f32_to_t2=quantize_f32_to_t2,
    dequantize_t2_to_f32=dequantize_t2_to_f32,
    bitnet_linear_t2_f32=bitnet_linear_t2_f32,
    dot_f32=dot_f32,
    add_f32=add_f32,
    mul_f32=mul_f32,
    silu_f32=silu_f32,
    rmsnorm_f32=rmsnorm_f32,
    softmax_f32=softmax_f32,
    rope_f32_inplace=rope_f32_inplace,
    matmul_f32=matmul_f32,
)

io = types.SimpleNamespace(
    ModelWriter=ModelWriter,
    ModelReader=ModelReader,
)

optim = types.SimpleNamespace(
    clip_grad_norm=clip_grad_norm,
    SGD=SGD,
    AdamW=AdamW,
)

__all__ = [
    "__version__",
    "LOG_TRACE",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARN",
    "LOG_ERROR",
    "LOG_FATAL",
    "LOG_OFF",
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "int32",
    "int64",
    "int16",
    "int8",
    "uint8",
    "bool_",
    "ternary2",
    "int4",
    "uint4",
    "init_runtime",
    "set_log_level",
    "log_level",
    "version_string",
    "version_hex",
    "platform_name",
    "arch_name",
    "cpu_features",
    "memory_stats",
    "reset_peak_memory",
    "Rng",
    "Tensor",
    "empty",
    "zeros",
    "ones",
    "full",
    "from_list",
    "from_buffer",
    "quantize_f32_to_t2",
    "dequantize_t2_to_f32",
    "bitnet_linear_t2_f32",
    "dot_f32",
    "add_f32",
    "mul_f32",
    "silu_f32",
    "rmsnorm_f32",
    "softmax_f32",
    "rope_f32_inplace",
    "matmul_f32",
    "NoGradGuard",
    "no_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "add",
    "sub",
    "mul",
    "silu",
    "relu",
    "rmsnorm",
    "matmul",
    "linear",
    "bitnet_linear_ste",
    "sum",
    "mean",
    "mse_loss",
    "cross_entropy_loss",
    "RMSNorm",
    "BitLinear",
    "Linear",
    "Embedding",
    "RotaryEmbedding",
    "KVCache",
    "GroupedQueryAttention",
    "BitNetFeedForward",
    "TransformerBlock",
    "BitNetTransformer",
    "ModelWriter",
    "ModelReader",
    "clip_grad_norm",
    "SGD",
    "AdamW",
    "nn",
    "autograd",
    "kernels",
    "io",
    "optim",
]
