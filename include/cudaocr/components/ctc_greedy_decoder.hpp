#pragma once

#include <memory>

#include "cudaocr/components/decoder.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

// "ctc-greedy": best class per timestep, repeats collapsed, blanks dropped.
// conf = mean probability of each emitted symbol at the timestep it was emitted
// (PaddleOCR's convention); 0 for an empty result. No params.
class CtcGreedyDecoder final : public Decoder {
  public:
    static Result<std::unique_ptr<CtcGreedyDecoder>> create(const Params& p);
    std::string_view name() const override { return "ctc-greedy"; }
    // InvalidArgument if probs.classes() != charset.symbols.size() or the blank
    // index is out of range.
    Result<DecodedLine> decode(const CtcProbs& probs, const Charset& charset) override;
};

}  // namespace cudaocr
