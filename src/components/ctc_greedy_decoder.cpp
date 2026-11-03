#include "cudaocr/components/ctc_greedy_decoder.hpp"

#include <string>

namespace cudaocr {

Result<std::unique_ptr<CtcGreedyDecoder>> CtcGreedyDecoder::create(const Params& p) {
    Status s = p.requireKnown({}, "ctc-greedy");
    if (!s.ok()) return s;
    return std::make_unique<CtcGreedyDecoder>();
}

Result<DecodedLine> CtcGreedyDecoder::decode(const CtcProbs& probs, const Charset& charset) {
    const int classes = probs.classes();
    if (probs.timesteps() > 0 && classes != static_cast<int>(charset.symbols.size()))
        return Status::Error(ErrorCode::InvalidArgument,
                             "ctc-greedy: " + std::to_string(classes) +
                                 " classes but charset has " +
                                 std::to_string(charset.symbols.size()) + " symbols");
    if (charset.blank < 0 || charset.blank >= static_cast<int>(charset.symbols.size()))
        return Status::Error(ErrorCode::InvalidArgument, "ctc-greedy: blank index out of range");

    DecodedLine out;
    double confSum = 0;
    int prev = -1;
    for (int t = 0; t < probs.timesteps(); ++t) {
        int best = 0;
        for (int c = 1; c < classes; ++c)
            if (probs.at(t, c) > probs.at(t, best)) best = c;
        if (best != charset.blank && best != prev) {
            out.text += charset.symbols[best];
            out.timesteps.push_back(t);
            out.classes.push_back(best);
            out.probs.push_back(probs.at(t, best));
            confSum += probs.at(t, best);
        }
        prev = best;
    }
    out.conf = out.timesteps.empty() ? 0.0f : static_cast<float>(confSum / out.timesteps.size());
    return out;
}

}  // namespace cudaocr
