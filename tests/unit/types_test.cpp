// Host-only TU (compiled by the C++ compiler, not nvcc): proves types.hpp and
// segment_types.hpp carry no CUDA dependency.
#include <gtest/gtest.h>

#include "cudaocr/stages/segment_types.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {
namespace {

TEST(BBox, InclusiveExtentsAndUnion) {
    const BBox a{2, 3, 5, 7};
    EXPECT_EQ(a.width(), 4);
    EXPECT_EQ(a.height(), 5);
    EXPECT_FALSE(a.empty());
    EXPECT_TRUE(BBox{}.empty());

    const BBox u = BBox::unite(a, BBox{0, 4, 3, 9});
    EXPECT_EQ(u, (BBox{0, 3, 5, 9}));
    EXPECT_EQ(BBox::unite(BBox{}, a), a);  // empty is the identity
    EXPECT_EQ(BBox::unite(a, BBox{}), a);
}

TEST(PageResult, DefaultsAreUnrecognized) {
    const PageResult page;
    EXPECT_TRUE(page.lines.empty());
    EXPECT_EQ(page.wordCount(), 0);
    EXPECT_EQ(page.text(), "");
    EXPECT_FLOAT_EQ(page.skewDeg, 0.0f);
    EXPECT_FALSE(page.inverted);

    const WordResult w;
    EXPECT_TRUE(w.text.empty());
    EXPECT_FLOAT_EQ(w.conf, 0.0f);
}

TEST(PageResult, TextJoinsLinesAndCountsWords) {
    PageResult page;
    LineResult l1;
    l1.text = "hello world";
    l1.words.resize(2);
    LineResult l2;
    l2.text = "bye";
    l2.words.resize(1);
    page.lines = {l1, l2};
    EXPECT_EQ(page.text(), "hello world\nbye");
    EXPECT_EQ(page.wordCount(), 3);
}

TEST(SegmentTypes, TextLineConvertsToLineResult) {
    TextLine line{10, 20, {}};
    line.words.push_back(WordBox{5, 10, 30, 20, {CharBox{5, 10, 12, 20}, CharBox{15, 10, 30, 20}}});
    line.words.push_back(WordBox{40, 10, 55, 20, {CharBox{40, 10, 55, 20}}});

    const LineResult r = toLineResult(line, /*pageWidth=*/100);
    EXPECT_EQ(r.box, (BBox{5, 10, 55, 20}));  // tight around the words
    ASSERT_EQ(r.words.size(), 2u);
    EXPECT_EQ(r.words[0].box, (BBox{5, 10, 30, 20}));
    ASSERT_EQ(r.words[0].glyphs.size(), 2u);
    EXPECT_EQ(r.words[0].glyphs[1], (BBox{15, 10, 30, 20}));
    EXPECT_EQ(r.words[1].box, (BBox{40, 10, 55, 20}));
    EXPECT_TRUE(r.text.empty());  // geometry only; text comes from a recognizer
}

TEST(SegmentTypes, LineWithoutWordsSpansPageWidth) {
    const LineResult r = toLineResult(TextLine{3, 8, {}}, /*pageWidth=*/50);
    EXPECT_EQ(r.box, (BBox{0, 3, 49, 8}));
    EXPECT_TRUE(r.words.empty());
}

TEST(SegmentTypes, ConvertsAllLinesInOrder) {
    const std::vector<TextLine> lines{TextLine{0, 4, {}}, TextLine{10, 14, {}}};
    const auto rs = toLineResults(lines, 20);
    ASSERT_EQ(rs.size(), 2u);
    EXPECT_EQ(rs[0].box.y0, 0);
    EXPECT_EQ(rs[1].box.y0, 10);
}

}  // namespace
}  // namespace cudaocr
