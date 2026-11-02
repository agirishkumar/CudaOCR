#pragma once

// Pipeline result types shared by every component. Host-only: no CUDA headers,
// so formatters, the CLI and tests can use them from plain C++ TUs.

#include <algorithm>
#include <string>
#include <vector>

namespace cudaocr {

// Axis-aligned box in page pixels, inclusive on both ends. Default is empty.
struct BBox {
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;

    int width() const { return x1 - x0 + 1; }
    int height() const { return y1 - y0 + 1; }
    bool empty() const { return x1 < x0 || y1 < y0; }

    // Smallest box covering both; an empty box is the identity.
    static BBox unite(const BBox& a, const BBox& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;
        return {std::min(a.x0, b.x0), std::min(a.y0, b.y0), std::max(a.x1, b.x1),
                std::max(a.y1, b.y1)};
    }
    bool operator==(const BBox&) const = default;
};

struct WordResult {
    BBox box;
    std::string text;          // UTF-8; empty until a recognizer runs
    float conf = 0.0f;         // [0, 1]
    std::vector<BBox> glyphs;  // optional per-glyph boxes (debug overlays)
};

struct LineResult {
    BBox box;
    std::vector<WordResult> words;
    std::string text;   // UTF-8; empty until a recognizer runs
    float conf = 0.0f;  // [0, 1]
};

struct StageTime {
    std::string name;
    float ms;
};

struct PageResult {
    int width = 0, height = 0;
    float skewDeg = 0.0f;
    bool inverted = false;          // input was light-on-dark and got normalized
    std::vector<LineResult> lines;  // reading order
    std::vector<StageTime> timings;

    // Line texts joined with '\n'.
    std::string text() const {
        std::string out;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i) out += '\n';
            out += lines[i].text;
        }
        return out;
    }
    int wordCount() const {
        int n = 0;
        for (const auto& l : lines) n += static_cast<int>(l.words.size());
        return n;
    }
};

}  // namespace cudaocr
