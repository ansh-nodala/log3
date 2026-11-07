import numpy as np
import revquant

weights = np.array(
    [0.5, 1.0, -0.5, -1.0, 2.0],
    dtype=np.float32
)

q = revquant.quantize(
    input=weights,
    scale=0.5,
    qmin=-8,
    qmax=7
)

restored = revquant.dequantize(
    input=q,
    scale=0.5
)
print("original :", weights)
print("Quantized:", q)
print("Restored:", restored)
