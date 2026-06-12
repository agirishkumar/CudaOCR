#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/stages/binarize.cuh"
#include "cudaocr/stages/enhance.cuh"

#include <cmath>
#include <cstdint>

namespace cudaocr {
namespace {

__constant__ std::uint8_t c_lut[256];

__global__ void lutKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                          std::uint8_t* __restrict__ out, size_t outPitch, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out[static_cast<size_t>(y) * outPitch + x] = c_lut[in[static_cast<size_t>(y) * inPitch + x]];
}

__global__ void invertKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                             std::uint8_t* __restrict__ out, size_t outPitch, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out[static_cast<size_t>(y) * outPitch + x] =
        static_cast<std::uint8_t>(255 - in[static_cast<size_t>(y) * inPitch + x]);
}

dim3 gridFor(const DeviceImage& img, dim3 block) {
    return dim3((img.width() + block.x - 1) / block.x, (img.height() + block.y - 1) / block.y);
}

}  // namespace

Status histogramEqualize(const DeviceImage& gray, DeviceImage& out, cudaStream_t stream) {
    auto histRes = histogram256(gray, stream);
    if (!histRes.ok()) return histRes.status();
    const auto& hist = histRes.value();

    const std::uint64_t total = static_cast<std::uint64_t>(gray.width()) * gray.height();
    std::uint64_t cdf = 0, cdfMin = 0;
    std::uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        cdf += hist[i];
        if (cdfMin == 0 && cdf != 0) cdfMin = cdf;
        if (total == cdfMin) {
            lut[i] = static_cast<std::uint8_t>(i);  // constant image: identity
        } else {
            const double v = 255.0 * (static_cast<double>(cdf) - static_cast<double>(cdfMin)) /
                             (static_cast<double>(total) - static_cast<double>(cdfMin));
            lut[i] = static_cast<std::uint8_t>(std::lround(std::max(0.0, v)));
        }
    }
    CUDAOCR_CHECK(cudaMemcpyToSymbolAsync(c_lut, lut, sizeof(lut), 0, cudaMemcpyHostToDevice,
                                          stream));
    const dim3 block(32, 8);
    lutKernel<<<gridFor(gray, block), block, 0, stream>>>(gray.data(), gray.pitch(), out.data(),
                                                          out.pitch(), gray.width(),
                                                          gray.height());
    CUDAOCR_CHECK_LAUNCH("lutKernel");
    return Status::Good();
}

Status invert(const DeviceImage& in, DeviceImage& out, cudaStream_t stream) {
    const dim3 block(32, 8);
    invertKernel<<<gridFor(in, block), block, 0, stream>>>(in.data(), in.pitch(), out.data(),
                                                           out.pitch(), in.width(), in.height());
    CUDAOCR_CHECK_LAUNCH("invertKernel");
    return Status::Good();
}

Result<bool> isInverted(const DeviceImage& bin, cudaStream_t stream) {
    auto histRes = histogram256(bin, stream);
    if (!histRes.ok()) return histRes.status();
    std::uint64_t dark = 0, light = 0;
    for (int i = 0; i < 128; ++i) dark += histRes.value()[i];
    for (int i = 128; i < 256; ++i) light += histRes.value()[i];
    return dark > light;
}

}  // namespace cudaocr
