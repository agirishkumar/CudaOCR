#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "cudaocr/components/ctc.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

struct DecodedLine {
    std::string text;            // UTF-8
    float conf = 0.0f;           // [0, 1]
    std::vector<int> timesteps;  // timestep of each emitted symbol (word-box alignment)
};

// CTC probabilities -> text. Implementations: greedy (M3), beam search (M6).
class Decoder {
  public:
    virtual ~Decoder() = default;
    virtual std::string_view name() const = 0;
    virtual Result<DecodedLine> decode(const CtcProbs& probs, const Charset& charset) = 0;
};

}  // namespace cudaocr
