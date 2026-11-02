#include "cudaocr/components/hough_deskewer.hpp"

#include <cmath>

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/deskew.cuh"

namespace cudaocr {

Result<std::unique_ptr<HoughDeskewer>> HoughDeskewer::create(const Params& p) {
    Status s = p.requireKnown({"maxAngle", "step"}, "hough");
    if (!s.ok()) return s;
    auto maxAngle = p.get<float>("maxAngle", 15.0f);
    if (!maxAngle.ok()) return maxAngle.status();
    auto step = p.get<float>("step", 0.25f);
    if (!step.ok()) return step.status();
    if (!(maxAngle.value() > 0 && maxAngle.value() <= 45))
        return Status::Error(ErrorCode::InvalidArgument, "hough: maxAngle must be in (0, 45]");
    if (!(step.value() > 0 && step.value() <= maxAngle.value()))
        return Status::Error(ErrorCode::InvalidArgument, "hough: step must be in (0, maxAngle]");
    return std::unique_ptr<HoughDeskewer>(new HoughDeskewer(maxAngle.value(), step.value()));
}

Result<float> HoughDeskewer::estimate(const DeviceImage& binary, StageContext& ctx) {
    return detectSkewAngle(binary, maxAngle_, step_, ctx.stream);
}

Status HoughDeskewer::apply(float angleDeg, const DeviceImage& in, DeviceImage& out,
                            std::uint8_t border, StageContext& ctx) {
    if (std::abs(angleDeg) < step_ / 2) return in.copyTo(out, ctx.stream);
    Status s = out.ensure(in.width(), in.height(), in.channels());
    if (!s.ok()) return s;
    return rotate(in, out, angleDeg, border, ctx.stream);
}

}  // namespace cudaocr
