#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/stages/deskew.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cudaocr {
namespace {

constexpr int kMaxTheta = 256;
__constant__ float c_cosTheta[kMaxTheta];
__constant__ float c_sinTheta[kMaxTheta];

// Votes: for each baseline pixel and each candidate theta,
// r = round(x*cos + y*sin) + rhoOffset; accumulator index r * numTheta + t.
// Bounds: 0 <= r < rhoBins (the legacy kernel's r < 2*rho check wrote OOB).
// Only bottom-edge dark pixels (dark with light below) vote: voting on all ink
// is degenerate — diagonals cutting through several thick text rows can out-vote
// the true baselines, biasing the angle by several degrees.
__global__ void houghKernel(const std::uint8_t* __restrict__ img, size_t pitch, int w, int h,
                            int* __restrict__ acc, int rhoBins, int rhoOffset, int numTheta) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (img[static_cast<size_t>(y) * pitch + x] >= 128) return;  // background
    if (y + 1 < h && img[static_cast<size_t>(y + 1) * pitch + x] < 128) return;  // not a baseline
    for (int t = 0; t < numTheta; ++t) {
        const int r = __float2int_rn(x * c_cosTheta[t] + y * c_sinTheta[t]) + rhoOffset;
        if (r >= 0 && r < rhoBins) atomicAdd(&acc[static_cast<size_t>(r) * numTheta + t], 1);
    }
}

__global__ void rotateKernel(const std::uint8_t* __restrict__ in, size_t inPitch, int w, int h,
                             std::uint8_t* __restrict__ out, size_t outPitch, double cosA,
                             double sinA, double cx, double cy, std::uint8_t border) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    // Inverse mapping: rotate the output pixel back by -angle to find the source.
    const double dx = x - cx;
    const double dy = y - cy;
    const double sx = cosA * dx + sinA * dy + cx;
    const double sy = -sinA * dx + cosA * dy + cy;

    std::uint8_t result = border;
    const int x0 = static_cast<int>(floor(sx));
    const int y0 = static_cast<int>(floor(sy));
    if (x0 >= -1 && x0 < w && y0 >= -1 && y0 < h) {
        const double fx = sx - x0;
        const double fy = sy - y0;
        auto sample = [&](int xx, int yy) -> double {
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) return border;
            return in[static_cast<size_t>(yy) * inPitch + xx];
        };
        const double v = (1 - fx) * (1 - fy) * sample(x0, y0) + fx * (1 - fy) * sample(x0 + 1, y0) +
                         (1 - fx) * fy * sample(x0, y0 + 1) + fx * fy * sample(x0 + 1, y0 + 1);
        result = static_cast<std::uint8_t>(min(255.0, max(0.0, nearbyint(v))));
    }
    out[static_cast<size_t>(y) * outPitch + x] = result;
}

}  // namespace

Result<float> detectSkewAngle(const DeviceImage& bin, float maxAngleDeg, float stepDeg,
                              cudaStream_t stream) {
    if (bin.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, "detectSkewAngle: 1-channel only");
    const int numTheta = 2 * static_cast<int>(std::round(maxAngleDeg / stepDeg)) + 1;
    if (numTheta < 3 || numTheta > kMaxTheta)
        return Status::Error(ErrorCode::InvalidArgument,
                             "detectSkewAngle: maxAngleDeg/stepDeg yields " +
                                 std::to_string(numTheta) + " steps (max " +
                                 std::to_string(kMaxTheta) + ")");

    // Line orientation theta = 90deg + skew; a page skewed by +a peaks at t for skew_t == a.
    std::vector<float> cosT(numTheta), sinT(numTheta), skewDeg(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        skewDeg[t] = -maxAngleDeg + static_cast<float>(t) * stepDeg;
        const double th = (90.0 + skewDeg[t]) * M_PI / 180.0;
        cosT[t] = static_cast<float>(std::cos(th));
        sinT[t] = static_cast<float>(std::sin(th));
    }
    CUDAOCR_CHECK_R(cudaMemcpyToSymbolAsync(c_cosTheta, cosT.data(), numTheta * sizeof(float), 0,
                                            cudaMemcpyHostToDevice, stream));
    CUDAOCR_CHECK_R(cudaMemcpyToSymbolAsync(c_sinTheta, sinT.data(), numTheta * sizeof(float), 0,
                                            cudaMemcpyHostToDevice, stream));

    const int diag = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(bin.width()) * bin.width() +
                            static_cast<double>(bin.height()) * bin.height())));
    const int rhoBins = 2 * diag + 1;
    const int rhoOffset = diag;

    auto acc = DeviceBuffer<int>::create(static_cast<size_t>(rhoBins) * numTheta);
    if (!acc.ok()) return acc.status();
    Status s = acc.value().zero();
    if (!s.ok()) return s;

    const dim3 block(32, 8);
    const dim3 grid((bin.width() + block.x - 1) / block.x,
                    (bin.height() + block.y - 1) / block.y);
    houghKernel<<<grid, block, 0, stream>>>(bin.data(), bin.pitch(), bin.width(), bin.height(),
                                            acc.value().data(), rhoBins, rhoOffset, numTheta);
    CUDAOCR_CHECK_R(cudaGetLastError());
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));

    std::vector<int> h_acc;
    s = acc.value().download(h_acc);
    if (!s.ok()) return s;

    const auto it = std::max_element(h_acc.begin(), h_acc.end());
    if (it == h_acc.end() || *it == 0) return 0.0f;  // blank page: no skew evidence
    const size_t idx = static_cast<size_t>(it - h_acc.begin());
    return skewDeg[idx % numTheta];
}

Status rotate(const DeviceImage& in, DeviceImage& out, float angleDeg, std::uint8_t border,
              cudaStream_t stream) {
    if (in.channels() != 1 || out.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, "rotate: 1-channel only");
    if (in.width() != out.width() || in.height() != out.height())
        return Status::Error(ErrorCode::InvalidArgument, "rotate: shape mismatch");
    const double a = angleDeg * M_PI / 180.0;
    const dim3 block(32, 8);
    const dim3 grid((in.width() + block.x - 1) / block.x, (in.height() + block.y - 1) / block.y);
    rotateKernel<<<grid, block, 0, stream>>>(in.data(), in.pitch(), in.width(), in.height(),
                                             out.data(), out.pitch(), std::cos(a), std::sin(a),
                                             (in.width() - 1) / 2.0, (in.height() - 1) / 2.0,
                                             border);
    CUDAOCR_CHECK_LAUNCH("rotateKernel");
    return Status::Good();
}

}  // namespace cudaocr
