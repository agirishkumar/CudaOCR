#pragma once

#include <memory>

#include "cudaocr/components/params.hpp"
#include "cudaocr/components/text_detector.hpp"

namespace cudaocr {

// "projection": row-profile line bands + column-profile word boxes
// (stages/segment.cuh). Assumes a single text column. Params: minLineHeight >= 1
// (default 3) — shorter ink bands are dropped as noise.
class ProjectionDetector final : public TextDetector {
  public:
    static Result<std::unique_ptr<ProjectionDetector>> create(const Params& p);
    std::string_view name() const override { return "projection"; }
    Result<std::vector<LineResult>> detect(const DeviceImage& binary, StageContext& ctx) override;

  private:
    explicit ProjectionDetector(int minLineHeight) : minLineHeight_(minLineHeight) {}
    int minLineHeight_;
};

}  // namespace cudaocr
