#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "cudaocr/components/ctc.hpp"
#include "cudaocr/components/stage_context.hpp"
#include "cudaocr/core/status.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

// Reads each line region of the deskewed gray page. Output i corresponds to
// lines[i]. Implementations: ONNX Runtime (M3), hand-written CUDA (M5).
class LineRecognizer {
  public:
    virtual ~LineRecognizer() = default;
    virtual std::string_view name() const = 0;
    virtual const Charset& charset() const = 0;
    virtual Result<std::vector<CtcProbs>> recognize(const DeviceImage& gray,
                                                    std::span<const LineResult> lines,
                                                    StageContext& ctx) = 0;
};

}  // namespace cudaocr
