#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/stages/grayscale.cuh"

namespace cudaocr {
namespace {

__global__ void grayscaleKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                                std::uint8_t* __restrict__ out, size_t outPitch, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const std::uint8_t* p = in + static_cast<size_t>(y) * inPitch + 3 * x;
    out[static_cast<size_t>(y) * outPitch + x] =
        static_cast<std::uint8_t>((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
}

}  // namespace

Status grayscale(const DeviceImage& rgb, DeviceImage& gray, cudaStream_t stream) {
    if (rgb.channels() != 3)
        return Status::Error(ErrorCode::InvalidArgument, "grayscale: input must be 3-channel");
    if (gray.width() != rgb.width() || gray.height() != rgb.height() || gray.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, "grayscale: bad output shape");

    const dim3 block(32, 8);
    const dim3 grid((rgb.width() + block.x - 1) / block.x, (rgb.height() + block.y - 1) / block.y);
    grayscaleKernel<<<grid, block, 0, stream>>>(rgb.data(), rgb.pitch(), gray.data(), gray.pitch(),
                                                rgb.width(), rgb.height());
    CUDAOCR_CHECK_LAUNCH("grayscaleKernel");
    return Status::Good();
}

}  // namespace cudaocr
