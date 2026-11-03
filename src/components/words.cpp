#include "cudaocr/components/words.hpp"

#include <algorithm>
#include <cmath>

namespace cudaocr {

std::vector<WordResult> wordsFromCtc(const DecodedLine& line, const Charset& charset, float x0,
                                     float stepWidth, const BBox& lineBox) {
    std::vector<WordResult> words;
    const size_t n = std::min({line.classes.size(), line.timesteps.size(), line.probs.size()});
    size_t i = 0;
    while (i < n) {
        const auto isSpace = [&](size_t k) {
            const int c = line.classes[k];
            return c >= 0 && c < static_cast<int>(charset.symbols.size()) &&
                   charset.symbols[c] == " ";
        };
        if (isSpace(i)) {
            ++i;
            continue;
        }
        size_t end = i;
        WordResult w;
        double confSum = 0;
        while (end < n && !isSpace(end)) {
            const int c = line.classes[end];
            if (c >= 0 && c < static_cast<int>(charset.symbols.size()))
                w.text += charset.symbols[c];
            confSum += line.probs[end];
            ++end;
        }
        w.conf = static_cast<float>(confSum / static_cast<double>(end - i));
        if (stepWidth > 0) {
            // CTC emits a symbol near its centre, not its left edge: measured on
            // the synthetic pages, unpadded boxes started ~0.65 steps late and
            // ended ~0.5 steps early, so extend half a step on each side.
            const float left = x0 + (static_cast<float>(line.timesteps[i]) - 0.5f) * stepWidth;
            const float right =
                x0 + (static_cast<float>(line.timesteps[end - 1]) + 1.5f) * stepWidth;
            w.box = {std::clamp(static_cast<int>(std::floor(left)), lineBox.x0, lineBox.x1),
                     lineBox.y0,
                     std::clamp(static_cast<int>(std::ceil(right)) - 1, lineBox.x0, lineBox.x1),
                     lineBox.y1};
        }
        words.push_back(std::move(w));
        i = end;
    }
    return words;
}

}  // namespace cudaocr
