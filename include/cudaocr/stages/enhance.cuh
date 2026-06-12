#pragma once

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Global histogram equalization: LUT built on host from the GPU histogram,
// lut[i] = round(255 * (cdf[i] - cdfMin) / (N - cdfMin)), applied by a kernel.
Status histogramEqualize(const DeviceImage& gray, DeviceImage& out, cudaStream_t stream = nullptr);

// out = 255 - in.
Status invert(const DeviceImage& in, DeviceImage& out, cudaStream_t stream = nullptr);

// True if dark pixels (< 128) form the majority — i.e. light-on-dark text that
// should be inverted so text is black on white.
Result<bool> isInverted(const DeviceImage& bin, cudaStream_t stream = nullptr);

}  // namespace cudaocr
