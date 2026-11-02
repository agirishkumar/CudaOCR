#pragma once

#include <string_view>

#include "cudaocr/components/stage_context.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Image -> image transform: grayscale conversion, denoise, equalize, polarity
// normalization, binarization. Contract: `out` may be empty or a different size;
// the stage (re)allocates it as needed. `in` and `out` must be distinct objects.
class ImageStage {
  public:
    virtual ~ImageStage() = default;
    virtual std::string_view name() const = 0;
    virtual Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) = 0;
};

}  // namespace cudaocr
