#pragma once

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// out = (in > mean(blockSize) - C) ? 255 : 0  — text comes out black (0) on white (255).
Status adaptiveThreshold(const DeviceImage& gray, DeviceImage& bin, int blockSize, int C,
                         cudaStream_t stream = nullptr);

// Global Otsu threshold from a GPU 256-bin histogram (argmax of between-class
// variance computed on the host). Returns the threshold value t in [0,255].
Result<int> otsuThreshold(const DeviceImage& gray, cudaStream_t stream = nullptr);

// out = (in > t) ? 255 : 0.
Status applyThreshold(const DeviceImage& gray, DeviceImage& bin, int t,
                      cudaStream_t stream = nullptr);

// GPU 256-bin histogram of a 1-channel image.
Result<std::vector<unsigned int>> histogram256(const DeviceImage& gray,
                                               cudaStream_t stream = nullptr);

}  // namespace cudaocr
