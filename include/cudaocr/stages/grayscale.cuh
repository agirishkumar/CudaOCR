#pragma once

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// RGB -> gray, fixed-point BT.601: y = (77*r + 150*g + 29*b + 128) >> 8.
// Integer arithmetic so the GPU result is byte-exact against the CPU oracle.
Status grayscale(const DeviceImage& rgb, DeviceImage& gray, cudaStream_t stream = nullptr);

}  // namespace cudaocr
