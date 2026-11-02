#include <gtest/gtest.h>

#include "cudaocr/components/ctc_greedy_decoder.hpp"
#include "cudaocr/registry.hpp"

namespace cudaocr {
namespace {

const Charset kCs{{"<b>", "a", "b", "c", " "}, 0};

// Builds [T x C] probabilities where timestep t puts `p` on class seq[t] and
// spreads the rest uniformly.
CtcProbs probs(const std::vector<int>& seq, float p = 0.9f, int classes = 5) {
    CtcProbs out(static_cast<int>(seq.size()), classes);
    for (int t = 0; t < out.timesteps(); ++t)
        for (int c = 0; c < classes; ++c)
            out.at(t, c) = c == seq[t] ? p : (1.0f - p) / (classes - 1);
    return out;
}

DecodedLine decode(const CtcProbs& p, const Charset& cs = kCs) {
    CtcGreedyDecoder d;
    auto r = d.decode(p, cs);
    EXPECT_TRUE(r.ok()) << r.status().message;
    return r.ok() ? r.value() : DecodedLine{};
}

TEST(CtcGreedy, CollapsesRepeatsAndDropsBlanks) {
    const auto d = decode(probs({0, 1, 1, 0, 2, 2, 2, 3, 0}));
    EXPECT_EQ(d.text, "abc");
    EXPECT_EQ(d.timesteps, (std::vector<int>{1, 4, 7}));  // first step of each run
}

TEST(CtcGreedy, BlankBetweenRepeatsKeepsDoubleLetters) {
    EXPECT_EQ(decode(probs({1, 0, 1})).text, "aa");
    EXPECT_EQ(decode(probs({1, 1, 1})).text, "a");
    EXPECT_EQ(decode(probs({3, 4, 1, 4, 4, 2})).text, "c a b");  // space is a class
}

TEST(CtcGreedy, AllBlankIsEmptyWithZeroConfidence) {
    const auto d = decode(probs({0, 0, 0, 0}));
    EXPECT_EQ(d.text, "");
    EXPECT_FLOAT_EQ(d.conf, 0.0f);
    EXPECT_TRUE(d.timesteps.empty());
    EXPECT_EQ(decode(CtcProbs(0, 5)).text, "");  // zero timesteps
}

TEST(CtcGreedy, ConfidenceIsMeanProbabilityOfEmittedSymbols) {
    CtcProbs p = probs({1, 1, 0, 2});
    p.at(0, 1) = 0.8f;  // emitted 'a' at t=0 with 0.8 (t=1 is a repeat, not counted)
    p.at(3, 2) = 0.6f;  // emitted 'b' at t=3 with 0.6
    const auto d = decode(p);
    EXPECT_EQ(d.text, "ab");
    EXPECT_NEAR(d.conf, 0.7f, 1e-6);
}

TEST(CtcGreedy, MultiByteSymbolsAreConcatenated) {
    const Charset cs{{"", "\xC3\xA9", "\xE2\x82\xAC"}, 0};  // é, €
    EXPECT_EQ(decode(probs({1, 0, 2}, 0.9f, 3), cs).text, "\xC3\xA9\xE2\x82\xAC");
}

TEST(CtcGreedy, ClassCountMustMatchCharset) {
    CtcGreedyDecoder d;
    auto r = d.decode(probs({1, 5}, 0.9f, 6), kCs);  // index 5 has no symbol
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.status().code, ErrorCode::InvalidArgument);
    const Charset badBlank{{"a", "b"}, 7};
    EXPECT_FALSE(d.decode(probs({0, 1}, 0.9f, 2), badBlank).ok());
}

TEST(CtcGreedy, RegisteredAsCtcGreedy) {
    auto reg = ComponentRegistry::withBuiltins();
    auto dec = reg.make<Decoder>({"ctc-greedy", {}});
    ASSERT_TRUE(dec.ok()) << dec.status().message;
    EXPECT_EQ(dec.value()->name(), "ctc-greedy");
    EXPECT_FALSE(reg.make<Decoder>({"ctc-greedy", Params().set("beam", 5)}).ok());
}

}  // namespace
}  // namespace cudaocr
