#pragma once

#include <string_view>
#include <vector>

#include "cudaocr/components/stage_context.hpp"
#include "cudaocr/core/status.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

// Finds text lines on the (deskewed) binary page. Returns geometry only — line
// boxes in reading order, optionally with word boxes; text fields stay empty.
class TextDetector {
  public:
    virtual ~TextDetector() = default;
    virtual std::string_view name() const = 0;
    virtual Result<std::vector<LineResult>> detect(const DeviceImage& binary,
                                                   StageContext& ctx) = 0;
};

}  // namespace cudaocr
