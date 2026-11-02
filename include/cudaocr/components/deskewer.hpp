#pragma once

#include <cstdint>
#include <string_view>

#include "cudaocr/components/stage_context.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Skew is estimated once (from the binary image) and applied to several images
// (binary for detection, gray for recognition), so it is two calls, not one
// ImageStage.
class Deskewer {
  public:
    virtual ~Deskewer() = default;
    virtual std::string_view name() const = 0;
    // Degrees, counter-clockwise positive; the angle to undo is -estimate.
    virtual Result<float> estimate(const DeviceImage& binary, StageContext& ctx) = 0;
    // Rotates `in` by `angleDeg` into `out` (allocated as needed); uncovered
    // pixels get `border`.
    virtual Status apply(float angleDeg, const DeviceImage& in, DeviceImage& out,
                         std::uint8_t border, StageContext& ctx) = 0;
};

}  // namespace cudaocr
