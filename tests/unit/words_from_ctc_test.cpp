// M4.1: words and word boxes from a decoded CTC line.
#include <gtest/gtest.h>

#include "cudaocr/components/words.hpp"

namespace cudaocr {
namespace {

const Charset kCs{{"", "a", "b", "c", " "}, 0};

// "ab ca": symbols a b ' ' c a at timesteps 1 2 4 6 7.
DecodedLine line() {
    DecodedLine d;
    d.text = "ab ca";
    d.timesteps = {1, 2, 4, 6, 7};
    d.classes = {1, 2, 4, 3, 1};
    d.probs = {0.9f, 0.7f, 0.99f, 0.6f, 0.8f};
    return d;
}

TEST(WordsFromCtc, SplitsAtSpacesWithBoxesAndConfidence) {
    const BBox lineBox{100, 20, 400, 50};
    const auto words = wordsFromCtc(line(), kCs, /*x0=*/90.0f, /*stepWidth=*/10.0f, lineBox);
    ASSERT_EQ(words.size(), 2u);
    EXPECT_EQ(words[0].text, "ab");
    EXPECT_EQ(words[1].text, "ca");
    // Boxes span [x0 + (t_first - 0.5) * step, x0 + (t_last + 1.5) * step):
    // "ab": timesteps 1..2 -> [95, 125) -> columns 95..124, clamped to the line: 100..124
    EXPECT_EQ(words[0].box, (BBox{100, 20, 124, 50}));
    // "ca": timesteps 6..7 -> [145, 175) -> 145..174
    EXPECT_EQ(words[1].box, (BBox{145, 20, 174, 50}));
    EXPECT_NEAR(words[0].conf, 0.8f, 1e-6);  // mean(0.9, 0.7)
    EXPECT_NEAR(words[1].conf, 0.7f, 1e-6);  // mean(0.6, 0.8); space not counted
}

TEST(WordsFromCtc, ClampsToLineBox) {
    const BBox lineBox{100, 20, 160, 50};
    const auto words = wordsFromCtc(line(), kCs, 0.0f, 25.0f, lineBox);
    ASSERT_EQ(words.size(), 2u);
    EXPECT_EQ(words[0].box.x0, 100);  // 12 clamped up to the line start
    EXPECT_EQ(words[1].box.x1, 160);  // 212 clamped to the line end
}

TEST(WordsFromCtc, LeadingTrailingAndRepeatedSpacesMakeNoEmptyWords) {
    DecodedLine d;
    d.text = " a  b ";
    d.timesteps = {0, 1, 2, 3, 4, 5};
    d.classes = {4, 1, 4, 4, 2, 4};
    d.probs = {1, 1, 1, 1, 1, 1};
    const auto words = wordsFromCtc(d, kCs, 0.0f, 8.0f, BBox{0, 0, 100, 10});
    ASSERT_EQ(words.size(), 2u);
    EXPECT_EQ(words[0].text, "a");
    EXPECT_EQ(words[1].text, "b");
}

TEST(WordsFromCtc, UnknownGeometryGivesTextOnlyWords) {
    const auto words = wordsFromCtc(line(), kCs, 0.0f, /*stepWidth=*/0.0f, BBox{0, 0, 9, 9});
    ASSERT_EQ(words.size(), 2u);
    EXPECT_TRUE(words[0].box.empty());
    EXPECT_EQ(words[1].text, "ca");
}

TEST(WordsFromCtc, EmptyLine) {
    EXPECT_TRUE(wordsFromCtc(DecodedLine{}, kCs, 0.0f, 8.0f, BBox{0, 0, 9, 9}).empty());
}

}  // namespace
}  // namespace cudaocr
