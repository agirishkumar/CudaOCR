#include "cudaocr/components/projection_detector.hpp"

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/segment.cuh"

namespace cudaocr {

Result<std::unique_ptr<ProjectionDetector>> ProjectionDetector::create(const Params& p) {
    Status s = p.requireKnown({"minLineHeight"}, "projection");
    if (!s.ok()) return s;
    auto h = p.get<int>("minLineHeight", 3);
    if (!h.ok()) return h.status();
    if (h.value() < 1)
        return Status::Error(ErrorCode::InvalidArgument, "projection: minLineHeight must be >= 1");
    return std::unique_ptr<ProjectionDetector>(new ProjectionDetector(h.value()));
}

Result<std::vector<LineResult>> ProjectionDetector::detect(const DeviceImage& binary,
                                                           StageContext& ctx) {
    auto rows = rowProfile(binary, ctx.stream);
    if (!rows.ok()) return rows.status();
    std::vector<TextLine> lines = detectLines(rows.value(), binary.width(), minLineHeight_);
    for (auto& line : lines) {
        Status s = segmentLine(binary, line, ctx.stream);
        if (!s.ok()) return s;
    }
    return toLineResults(lines, binary.width());
}

}  // namespace cudaocr
