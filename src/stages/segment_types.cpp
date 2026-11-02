#include "cudaocr/stages/segment_types.hpp"

namespace cudaocr {

LineResult toLineResult(const TextLine& line, int pageWidth) {
    LineResult out;
    for (const auto& w : line.words) {
        WordResult word;
        word.box = {w.x0, w.y0, w.x1, w.y1};
        for (const auto& c : w.chars) word.glyphs.push_back({c.x0, c.y0, c.x1, c.y1});
        out.box = BBox::unite(out.box, word.box);
        out.words.push_back(std::move(word));
    }
    if (out.words.empty()) out.box = {0, line.y0, pageWidth - 1, line.y1};
    return out;
}

std::vector<LineResult> toLineResults(const std::vector<TextLine>& lines, int pageWidth) {
    std::vector<LineResult> out;
    out.reserve(lines.size());
    for (const auto& l : lines) out.push_back(toLineResult(l, pageWidth));
    return out;
}

}  // namespace cudaocr
