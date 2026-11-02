#pragma once

#include <string>
#include <vector>

#include "cudaocr/core/host_image.hpp"
#include "cudaocr/core/status.hpp"
#include "cudaocr/stages/segment_types.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

struct PipelineConfig {
    int medianK = 3;        // 0 disables denoising; otherwise 3 or 5
    bool useOtsu = false;   // false = adaptive threshold
    int blockSize = 31;     // adaptive threshold window (odd)
    int C = 10;             // adaptive threshold offset
    bool equalize = false;  // histogram equalization before binarization
    bool deskew = true;
    float maxSkewDeg = 15.0f;
    float skewStepDeg = 0.25f;
    int minLineHeight = 3;
};

struct PipelineResult {
    float skewAngleDeg = 0.0f;
    int otsuThreshold = -1;  // set when useOtsu
    bool wasInverted = false;
    std::vector<TextLine> lines;
    int wordCount = 0;
    int charCount = 0;
    std::vector<StageTime> timings;
    // Filled when keepStages: intermediate images for inspection.
    HostImage gray, binary, deskewed;
};

Result<PipelineResult> runPipeline(const HostImage& input, const PipelineConfig& cfg,
                                   bool keepStages = false);

}  // namespace cudaocr
