
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ------------------------------------------------------------
// Validation
// ------------------------------------------------------------

static void validate_scale(float scale) {
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument(
            "scale must be finite and greater than zero"
        );
    }
}

static void validate_range(int8_t qmin, int8_t qmax) {
    if (qmin > qmax) {
        throw std::invalid_argument(
            "qmin must be less than or equal to qmax"
        );
    }
}

// ------------------------------------------------------------
// Quantization
// ------------------------------------------------------------

std::vector<int8_t> quantize(
    const std::vector<float>& X,
    float scale,
    int8_t qmin,
    int8_t qmax
) {
    validate_scale(scale);
    validate_range(qmin, qmax);

    std::vector<int8_t> Q(X.size());

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X[i])) {
            throw std::invalid_argument(
                "Input contains NaN or infinity"
            );
        }

        const float normalized = X[i] / scale;

        int q;

        if (normalized <= static_cast<float>(qmin)) {
            q = qmin;
        } else if (normalized >= static_cast<float>(qmax)) {
            q = qmax;
        } else {
            q = static_cast<int>(std::round(normalized));
        }

        Q[i] = static_cast<int8_t>(q);
    }

    return Q;
}

// ------------------------------------------------------------
// Dequantization
// ------------------------------------------------------------

std::vector<float> dequantize(
    const std::vector<int8_t>& Q,
    float scale
) {
    validate_scale(scale);

    std::vector<float> X(Q.size());

    for (std::size_t i = 0; i < Q.size(); ++i) {
        X[i] = static_cast<float>(Q[i]) * scale;
    }

    return X;
}

// ------------------------------------------------------------
// AVX2 placeholder
// ------------------------------------------------------------

#ifdef USE_AVX2
#include <immintrin.h>

// The original AVX2 matmul was incomplete.
// No incorrect or non-functional SIMD implementation
// is exposed as a working API.
#endif

// ------------------------------------------------------------
// Python bindings
// ------------------------------------------------------------

PYBIND11_MODULE(revquant, m) {

    m.doc() = "C++ low-bit quantization research module";

    // Float32 -> Int8

    m.def(
        "quantize",
        [](py::array_t<float, py::array::forcecast> input,
           float scale,
           int qmin,
           int qmax) {

            if (qmin < -128 || qmin > 127 ||
                qmax < -128 || qmax > 127) {
                throw std::invalid_argument(
                    "qmin and qmax must fit in int8"
                );
            }

            if (qmin > qmax) {
                throw std::invalid_argument(
                    "qmin must be <= qmax"
                );
            }

            auto buf = input.request();

            if (buf.ndim != 1) {
                throw std::invalid_argument(
                    "input must be a 1D array"
                );
            }

            const auto* data =
                static_cast<const float*>(buf.ptr);

            const std::size_t N =
                static_cast<std::size_t>(buf.shape[0]);

            std::vector<float> X(data, data + N);

            auto Q = quantize(
                X,
                scale,
                static_cast<int8_t>(qmin),
                static_cast<int8_t>(qmax)
            );

            py::array_t<int8_t> output(N);

            if (N > 0) {
                std::memcpy(
                    output.mutable_data(),
                    Q.data(),
                    N * sizeof(int8_t)
                );
            }

            return output;
        },
        py::arg("input"),
        py::arg("scale"),
        py::arg("qmin") = -128,
        py::arg("qmax") = 127
    );

    // Int8 -> Float32

    m.def(
        "dequantize",
        [](py::array_t<int8_t, py::array::forcecast> input,
           float scale) {

            auto buf = input.request();

            if (buf.ndim != 1) {
                throw std::invalid_argument(
                    "input must be a 1D array"
                );
            }

            const auto* data =
                static_cast<const int8_t*>(buf.ptr);

            const std::size_t N =
                static_cast<std::size_t>(buf.shape[0]);

            std::vector<int8_t> Q(data, data + N);

            auto X = dequantize(Q, scale);

            py::array_t<float> output(N);

            if (N > 0) {
                std::memcpy(
                    output.mutable_data(),
                    X.data(),
                    N * sizeof(float)
                );
            }

            return output;
        },
        py::arg("input"),
        py::arg("scale")
    );
}
