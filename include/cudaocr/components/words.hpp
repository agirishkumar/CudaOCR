#pragma once

#include <vector>

#include "cudaocr/components/ctc.hpp"
#include "cudaocr/components/decoder.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

// Splits a decoded line into words at space symbols (charset entries " ").
// Word box: page columns [x0 + (t_first - 0.5) * step, x0 + (t_last + 1.5) * step)
// (CTC emits near a glyph's centre, hence the half-step padding), clamped to
// `lineBox`, with the line's vertical extent. conf = mean probability of its symbols.
// stepWidth <= 0 (geometry unknown): words carry text/conf and an empty box.
std::vector<WordResult> wordsFromCtc(const DecodedLine& line, const Charset& charset, float x0,
                                     float stepWidth, const BBox& lineBox);

}  // namespace cudaocr
