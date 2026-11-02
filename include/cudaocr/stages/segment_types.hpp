#pragma once

// Geometry produced by the projection-profile segmenter (segment.cuh), plus its
// bridge to the generic result types. Host-only.

#include <vector>

#include "cudaocr/types.hpp"

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

// Line box is tight around its words; a line with no words spans the page width.
LineResult toLineResult(const TextLine& line, int pageWidth);
std::vector<LineResult> toLineResults(const std::vector<TextLine>& lines, int pageWidth);

}  // namespace cudaocr
