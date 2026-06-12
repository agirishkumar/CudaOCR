#pragma once

#include <vector>

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

struct CharBox {
    int x0, y0, x1, y1;  // inclusive
};
struct WordBox {
    int x0, y0, x1, y1;  // inclusive
    std::vector<CharBox> chars;
};
struct TextLine {
    int y0, y1;  // inclusive band
    std::vector<WordBox> words;
};

// Dark-pixel (< 128) count per row / per column-range. Exact integer counts.
Result<std::vector<int>> rowProfile(const DeviceImage& bin, cudaStream_t stream = nullptr);
Result<std::vector<int>> colProfile(const DeviceImage& bin, int y0, int y1,
                                    cudaStream_t stream = nullptr);

// Host-side analysis of the profiles (deterministic, unit-testable without a GPU).
// Rows with count >= max(2, width/200) are "ink"; runs shorter than minHeight are dropped.
std::vector<TextLine> detectLines(const std::vector<int>& rowCounts, int width,
                                  int minHeight = 3);

// Fills line.words (and word.chars) from column profiles of the line band.
// Word gap >= max(2, lineHeight/3) blank columns; char gap >= 1 blank column.
Status segmentLine(const DeviceImage& bin, TextLine& line, cudaStream_t stream = nullptr);

}  // namespace cudaocr
