#pragma once

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Hough-based skew estimate on a binarized image (text = dark pixels < 128).
// Searches line orientations theta = 90deg + skew for skew in
// [-maxAngleDeg, +maxAngleDeg] at stepDeg resolution; returns the skew angle in
// degrees (positive = page content rotated counter-clockwise).
// The sin/cos tables are built on the host and shared with the kernel so the
// accumulator is deterministic. Accumulator bounds: r in [0, rhoBins),
// index r * numTheta + t  (regression: the legacy kernel checked r < 2*rho and
// wrote out of bounds).
Result<float> detectSkewAngle(const DeviceImage& bin, float maxAngleDeg = 15.0f,
                              float stepDeg = 0.25f, cudaStream_t stream = nullptr);

// Rotate around the image center by angleDeg (degrees, CCW positive), bilinear
// sampling, out-of-range pixels filled with `border`.
Status rotate(const DeviceImage& in, DeviceImage& out, float angleDeg, std::uint8_t border,
              cudaStream_t stream = nullptr);

}  // namespace cudaocr
