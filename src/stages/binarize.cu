#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/stages/binarize.cuh"
#include "cudaocr/stages/filters.cuh"

#include <cstdint>

namespace cudaocr {
namespace {

__global__ void adaptiveKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                               const std::uint8_t* __restrict__ mean, size_t meanPitch,
                               std::uint8_t* __restrict__ out, size_t outPitch, int w, int h,
                               int C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int v = in[static_cast<size_t>(y) * inPitch + x];
    const int m = mean[static_cast<size_t>(y) * meanPitch + x];
    out[static_cast<size_t>(y) * outPitch + x] = (v > m - C) ? 255 : 0;
}

__global__ void thresholdKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                                std::uint8_t* __restrict__ out, size_t outPitch, int w, int h,
                                int t) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out[static_cast<size_t>(y) * outPitch + x] =
        (in[static_cast<size_t>(y) * inPitch + x] > t) ? 255 : 0;
}

__global__ void histogramKernel(const std::uint8_t* __restrict__ in, size_t pitch, int w, int h,
                                unsigned int* __restrict__ hist) {
    __shared__ unsigned int sh[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 256) sh[tid] = 0;
    __syncthreads();
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) atomicAdd(&sh[in[static_cast<size_t>(y) * pitch + x]], 1u);
    __syncthreads();
    if (tid < 256 && sh[tid] != 0) atomicAdd(&hist[tid], sh[tid]);
}

}  // namespace

Result<std::vector<unsigned int>> histogram256(const DeviceImage& gray, cudaStream_t stream) {
    if (gray.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, "histogram256: 1-channel only");
    auto d_hist = DeviceBuffer<unsigned int>::create(256);
    if (!d_hist.ok()) return d_hist.status();
    Status s = d_hist.value().zero(stream);
    if (!s.ok()) return s;

    const dim3 block(32, 8);
    const dim3 grid((gray.width() + block.x - 1) / block.x,
                    (gray.height() + block.y - 1) / block.y);
    histogramKernel<<<grid, block, 0, stream>>>(gray.data(), gray.pitch(), gray.width(),
                                                gray.height(), d_hist.value().data());
    CUDAOCR_CHECK_LAUNCH("histogramKernel");
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));
    std::vector<unsigned int> hist;
    s = d_hist.value().download(hist, stream);
    if (!s.ok()) return s;
    return hist;
}

Status adaptiveThreshold(const DeviceImage& gray, DeviceImage& bin, int blockSize, int C,
                         cudaStream_t stream) {
    auto mean = DeviceImage::create(gray.width(), gray.height(), 1);
    if (!mean.ok()) return mean.status();
    Status s = boxMean(gray, mean.value(), blockSize, stream);
    if (!s.ok()) return s;

    const dim3 block(32, 8);
    const dim3 grid((gray.width() + block.x - 1) / block.x,
                    (gray.height() + block.y - 1) / block.y);
    adaptiveKernel<<<grid, block, 0, stream>>>(gray.data(), gray.pitch(), mean.value().data(),
                                               mean.value().pitch(), bin.data(), bin.pitch(),
                                               gray.width(), gray.height(), C);
    CUDAOCR_CHECK_LAUNCH("adaptiveKernel");
    CUDAOCR_CHECK(cudaStreamSynchronize(stream));  // mean freed on return
    return Status::Good();
}

Result<int> otsuThreshold(const DeviceImage& gray, cudaStream_t stream) {
    auto histRes = histogram256(gray, stream);
    if (!histRes.ok()) return histRes.status();
    const auto& hist = histRes.value();

    // Standard Otsu: maximize between-class variance over candidate thresholds.
    double total = 0, sumAll = 0;
    for (int i = 0; i < 256; ++i) {
        total += hist[i];
        sumAll += static_cast<double>(i) * hist[i];
    }
    double sumB = 0, wB = 0, best = -1.0;
    int bestT = 0;
    for (int t = 0; t < 256; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        const double wF = total - wB;
        if (wF == 0) break;
        sumB += static_cast<double>(t) * hist[t];
        const double mB = sumB / wB;
        const double mF = (sumAll - sumB) / wF;
        const double between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) {
            best = between;
            bestT = t;
        }
    }
    return bestT;
}

Status applyThreshold(const DeviceImage& gray, DeviceImage& bin, int t, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((gray.width() + block.x - 1) / block.x,
                    (gray.height() + block.y - 1) / block.y);
    thresholdKernel<<<grid, block, 0, stream>>>(gray.data(), gray.pitch(), bin.data(), bin.pitch(),
                                                gray.width(), gray.height(), t);
    CUDAOCR_CHECK_LAUNCH("thresholdKernel");
    return Status::Good();
}

}  // namespace cudaocr
