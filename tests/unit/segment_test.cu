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

// Regression (found by M3.1's synthetic fixtures): just below the baseline only
// a few thin descender strokes carry ink, so the row count can dip under the
// ink threshold and the descender bottoms (g, y, p) became a separate 5-6 px
// "line" — 10 lines detected on an 8-line page. Slivers much thinner than the
// typical line and close to a neighbour are merged into the nearest line.
std::vector<int> bands(int h, std::initializer_list<std::pair<int, int>> inkRows) {
    std::vector<int> counts(h, 0);
    for (auto [y0, y1] : inkRows)
        for (int y = y0; y <= y1; ++y) counts[y] = 50;
    return counts;
}

TEST(DetectLines, MergesDescenderSliverIntoLineAbove) {
    // line 10-44, gap of 5 rows, sliver 50-54, then lines 80-114 and 150-184.
    const auto lines = detectLines(bands(200, {{10, 44}, {50, 54}, {80, 114}, {150, 184}}), 400);
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0].y0, 10);
    EXPECT_EQ(lines[0].y1, 54);  // sliver absorbed
    EXPECT_EQ(lines[1].y0, 80);
}

TEST(DetectLines, MergesTittleSliverIntoLineBelow) {
    // i/j dots above a line with no ascenders: sliver 20-24, gap 3, line 28-55.
    const auto lines = detectLines(bands(120, {{20, 24}, {28, 55}, {85, 112}}), 400);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0].y0, 20);
    EXPECT_EQ(lines[0].y1, 55);
}

TEST(DetectLines, KeepsSmallBandsFarFromAnyLine) {
    // A short isolated band (e.g. a rule or a page number) stays its own line.
    const auto lines = detectLines(bands(300, {{10, 44}, {150, 155}, {250, 284}}), 400);
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[1].y0, 150);
}

TEST(DetectLines, DoesNotSwallowShortLineBetweenNoiseBlobs) {
    // Bands measured on resources/noisy_text.png (adaptive threshold): a real
    // 24-row text line between noise blobs must survive (median height is 55).
    const auto lines = detectLines(bands(343, {{5, 59}, {66, 89}, {92, 337}}), 727);
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[1].y0, 66);
    EXPECT_EQ(lines[1].y1, 89);
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
