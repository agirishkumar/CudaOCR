#pragma once

#include <memory>

#include "cudaocr/components/deskewer.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

// "hough": baseline-pixel Hough skew estimate (stages/deskew.cuh) + bilinear
// GPU rotation. Params: maxAngle in (0, 45] degrees (default 15), step > 0 and
// <= maxAngle (default 0.25). |angle| < step/2 is below the estimator's
// resolution, so apply() copies instead of resampling.
class HoughDeskewer final : public Deskewer {
  public:
    static Result<std::unique_ptr<HoughDeskewer>> create(const Params& p);
    std::string_view name() const override { return "hough"; }
    Result<float> estimate(const DeviceImage& binary, StageContext& ctx) override;
    Status apply(float angleDeg, const DeviceImage& in, DeviceImage& out, std::uint8_t border,
                 StageContext& ctx) override;

  private:
    HoughDeskewer(float maxAngle, float step) : maxAngle_(maxAngle), step_(step) {}
    float maxAngle_, step_;
};

}  // namespace cudaocr
