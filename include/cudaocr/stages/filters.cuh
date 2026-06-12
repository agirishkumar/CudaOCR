#pragma once

#include <cmath>
#include <vector>

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// All filters: 1-channel 8u in/out, replicate-clamped borders, integer arithmetic
// (byte-exact against the CPU oracles by construction).

// Local mean over blockSize x blockSize (odd), separable two-pass u32 sums.
// mean = (sum + n/2) / n with n = blockSize^2.
Status boxMean(const DeviceImage& in, DeviceImage& outMean, int blockSize,
               cudaStream_t stream = nullptr);

// Median over k x k, k in {3, 5}. In-register insertion sort.
Status medianFilter(const DeviceImage& in, DeviceImage& out, int k, cudaStream_t stream = nullptr);

// Separable Gaussian with fixed-point weights (sum 65536); each pass rounds back
// to 8u via (acc + 32768) >> 16. Intermediate quantization is part of the spec.
Status gaussianBlur(const DeviceImage& in, DeviceImage& out, int ksize, double sigma,
                    cudaStream_t stream = nullptr);

// Shared weight generator (used by both the kernel wrapper and the test oracle).
// Quantizes a sampled Gaussian to ints summing exactly to 65536 (residual on center tap).
inline std::vector<int> gaussianFixedWeights(int ksize, double sigma) {
    const int r = ksize / 2;
    if (sigma <= 0) sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;  // OpenCV's default rule
    std::vector<double> w(ksize);
    double total = 0;
    for (int i = 0; i < ksize; ++i) {
        const double d = i - r;
        w[i] = std::exp(-(d * d) / (2 * sigma * sigma));
        total += w[i];
    }
    std::vector<int> q(ksize);
    long long sum = 0;
    for (int i = 0; i < ksize; ++i) {
        q[i] = static_cast<int>(std::lround(w[i] / total * 65536.0));
        sum += q[i];
    }
    q[r] += static_cast<int>(65536 - sum);
    return q;
}

}  // namespace cudaocr
