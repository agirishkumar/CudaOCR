#include <gtest/gtest.h>

#include "cudaocr/stages/segment.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

TEST(Profiles, RowProfileMatchesOracle) {
    const auto page = oracle::makeSyntheticPage();
    auto dev = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(dev.ok());
    auto rows = rowProfile(dev.value());
    ASSERT_TRUE(rows.ok()) << rows.status().message;
    EXPECT_EQ(rows.value(), oracle::rowProfile(page.image));
}

TEST(Profiles, ColProfileMatchesOracle) {
    const auto page = oracle::makeSyntheticPage();
    auto dev = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(dev.ok());
    auto cols = colProfile(dev.value(), 30, 47);  // first line band
    ASSERT_TRUE(cols.ok()) << cols.status().message;
    EXPECT_EQ(cols.value(), oracle::colProfile(page.image, 30, 47));
}

TEST(Profiles, ColProfileRejectsBadBand) {
    auto dev = DeviceImage::create(32, 32, 1);
    ASSERT_TRUE(dev.ok());
    EXPECT_FALSE(colProfile(dev.value(), 10, 5).ok());
    EXPECT_FALSE(colProfile(dev.value(), 0, 32).ok());
}

TEST(DetectLines, FindsSyntheticLines) {
    const auto page = oracle::makeSyntheticPage();
    const auto lines = detectLines(oracle::rowProfile(page.image), page.image.width);
    ASSERT_EQ(static_cast<int>(lines.size()), page.lineCount);
    EXPECT_EQ(lines[0].y0, 30);
    EXPECT_EQ(lines[0].y1, 47);
}

TEST(DetectLines, IgnoresShortRunsAndBlankPages) {
    std::vector<int> counts(100, 0);
    counts[50] = 99;  // single-row speckle, below minHeight
    EXPECT_TRUE(detectLines(counts, 400, 3).empty());
    EXPECT_TRUE(detectLines(std::vector<int>(100, 0), 400).empty());
}

TEST(SegmentLine, FindsWordsAndChars) {
    const auto page = oracle::makeSyntheticPage();
    auto dev = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(dev.ok());
    auto lines = detectLines(oracle::rowProfile(page.image), page.image.width);
    ASSERT_EQ(static_cast<int>(lines.size()), page.lineCount);
    for (int li = 0; li < page.lineCount; ++li) {
        Status s = segmentLine(dev.value(), lines[li]);
        ASSERT_TRUE(s.ok()) << s.message;
        EXPECT_EQ(static_cast<int>(lines[li].words.size()), page.wordsPerLine[li]) << "line " << li;
        int chars = 0;
        for (const auto& w : lines[li].words) chars += static_cast<int>(w.chars.size());
        EXPECT_EQ(chars, page.charsPerLine[li]) << "line " << li;
    }
}

TEST(SegmentLine, WordBoxesAreOrderedAndDisjoint) {
    const auto page = oracle::makeSyntheticPage();
    auto dev = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(dev.ok());
    auto lines = detectLines(oracle::rowProfile(page.image), page.image.width);
    ASSERT_FALSE(lines.empty());
    ASSERT_TRUE(segmentLine(dev.value(), lines[0]).ok());
    for (size_t i = 1; i < lines[0].words.size(); ++i)
        EXPECT_GT(lines[0].words[i].x0, lines[0].words[i - 1].x1);
}

}  // namespace
}  // namespace cudaocr
