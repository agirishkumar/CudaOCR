// M3.6: line crop + resize + normalize kernel vs its CPU oracle.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include "cudaocr/pipeline.hpp"
#include "cudaocr/stages/line_prep.cuh"
#include "oracles/oracles.hpp"

namespace cudaocr {
namespace {

namespace fs = std::filesystem;

TEST(PlanLineCrops, PadsByFractionOfHeightAndKeepsAspect) {
    LinePrepParams p;                                   // height 48, padFrac 0.1
    const std::vector<BBox> boxes{{100, 50, 399, 79}};  // 300 x 30
    auto plan = planLineCrops(boxes, 1000, 500, p);
    ASSERT_TRUE(plan.ok()) << plan.status().message;
    const LineCrop& c = plan.value()[0];
    // pad = round(0.1 * 30) = 3 on every side -> 306 x 36 starting at (97, 47)
    EXPECT_EQ(c.x0, 97);
    EXPECT_EQ(c.y0, 47);
    EXPECT_EQ(c.w, 306);
    EXPECT_EQ(c.h, 36);
    EXPECT_EQ(c.outW, 408);  // round(306 * 48 / 36)
}

TEST(PlanLineCrops, OffPagePaddingAndMinWidth) {
    LinePrepParams p;
    p.padFrac = 0.5f;
    // 2 x 40 box at the page corner; pad = 20 -> crop 42 x 80 from (-20, -20).
    const std::vector<BBox> boxes{{0, 0, 1, 39}};
    auto plan = planLineCrops(boxes, 100, 100, p);
    ASSERT_TRUE(plan.ok()) << plan.status().message;
    const LineCrop& c = plan.value()[0];
    EXPECT_EQ(c.x0, -20);  // padding may extend off-page (sampled as white)
    EXPECT_EQ(c.y0, -20);
    EXPECT_EQ(c.w, 42);
    EXPECT_EQ(c.h, 80);
    EXPECT_EQ(c.outW, 25);  // round(42 * 48 / 80) = 25.2 -> 25

    p.padFrac = 0.0f;  // 2 x 40 unpadded -> round(2 * 48 / 40) = 2 -> clamped to minWidth 8
    auto narrow = planLineCrops(boxes, 100, 100, p);
    ASSERT_TRUE(narrow.ok());
    EXPECT_EQ(narrow.value()[0].outW, 8);
}

TEST(PlanLineCrops, RejectsBadInput) {
    LinePrepParams p;
    auto bad = [&](std::vector<BBox> b, int w = 100, int h = 100) {
        return !planLineCrops(b, w, h, p).ok();
    };
    EXPECT_TRUE(bad({BBox{}}));                // empty box
    EXPECT_TRUE(bad({{10, 10, 5, 20}}));       // inverted
    EXPECT_TRUE(bad({{200, 10, 220, 20}}));    // entirely off-page
    EXPECT_TRUE(bad({{0, 0, 10, 10}}, 0, 0));  // empty page
    p.height = 0;
    EXPECT_TRUE(bad({{0, 0, 10, 10}}));
}

void expectMatchesOracle(const HostImage& gray, const std::vector<BBox>& boxes,
                         const LinePrepParams& p) {
    auto d = DeviceImage::fromHost(gray);
    ASSERT_TRUE(d.ok());
    auto batch = prepareLines(d.value(), boxes, p);
    ASSERT_TRUE(batch.ok()) << batch.status().message;
    const LineBatch& b = batch.value();
    ASSERT_EQ(b.count, static_cast<int>(boxes.size()));
    EXPECT_EQ(b.channels, p.channels);
    EXPECT_EQ(b.height, p.height);
    int maxW = 0;
    for (const auto& c : b.crops) maxW = std::max(maxW, c.outW);
    EXPECT_EQ(b.width, maxW);

    std::vector<float> got;
    ASSERT_TRUE(b.data.download(got).ok());
    const std::vector<float> want = oracle::prepareLines(gray, b.crops, p, b.width);
    ASSERT_EQ(got.size(), want.size());
    double maxDiff = 0;
    for (size_t i = 0; i < got.size(); ++i)
        maxDiff = std::max(maxDiff, std::abs(double(got[i]) - double(want[i])));
    EXPECT_LE(maxDiff, 1e-4);
}

TEST(PrepareLines, MatchesOracleOnDetectedLines) {
    // Real geometry: the lines the pipeline detects on the deskewed fixtures.
    auto pipe = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(pipe.ok());
    for (const char* f : {"page_sans_clean.png", "page_serif_skew_m4.png", "line_punct_mono.png"}) {
        SCOPED_TRACE(f);
        auto img = loadImage((fs::path(CUDAOCR_SYNTH_DIR) / f).string());
        ASSERT_TRUE(img.ok());
        StageImages keep;
        auto r = pipe.value().run(img.value(), &keep);
        ASSERT_TRUE(r.ok());
        std::vector<BBox> boxes;
        for (const auto& l : r.value().lines) boxes.push_back(l.box);
        ASSERT_FALSE(boxes.empty());
        expectMatchesOracle(keep.deskewedGray, boxes, LinePrepParams{});
    }
}

TEST(PrepareLines, MatchesOracleOnRandomImageAndOddParams) {
    const HostImage img = oracle::randomImage(331, 97, 1, /*seed=*/9);
    LinePrepParams p;
    p.height = 32;
    p.channels = 1;
    p.padFrac = 0.25f;
    p.mean = 0.0f;
    p.scale = 1.0f;
    // Includes boxes hugging every page edge so padding samples off-page.
    expectMatchesOracle(
        img, {{0, 0, 120, 20}, {200, 70, 330, 96}, {50, 30, 52, 60}, {5, 40, 325, 55}}, p);
}

TEST(PrepareLines, PaddingAndOffPageAreWhite) {
    const HostImage black = HostImage::create(64, 32, 1, 0);
    auto d = DeviceImage::fromHost(black);
    ASSERT_TRUE(d.ok());
    LinePrepParams p;
    p.padFrac = 1.0f;  // pad 16 px around a 64x16 box in a 64x32 page -> lots off-page
    auto batch = prepareLines(d.value(), std::vector<BBox>{{0, 8, 63, 23}, {10, 8, 20, 23}}, p);
    ASSERT_TRUE(batch.ok()) << batch.status().message;
    std::vector<float> v;
    ASSERT_TRUE(batch.value().data.download(v).ok());
    const LineBatch& b = batch.value();
    auto at = [&](int n, int c, int y, int x) {
        return v[((static_cast<size_t>(n) * b.channels + c) * b.height + y) * b.width + x];
    };
    EXPECT_FLOAT_EQ(at(0, 0, 0, 0), 1.0f);                        // off-page corner: white
    EXPECT_FLOAT_EQ(at(0, 2, b.height / 2, b.width / 2), -1.0f);  // page center: black
    ASSERT_LT(b.crops[1].outW, b.width);
    EXPECT_FLOAT_EQ(at(1, 1, b.height / 2, b.width - 1), 1.0f);  // right padding: white
}

TEST(PrepareLines, EmptyBoxListIsEmptyBatch) {
    auto d = DeviceImage::create(10, 10, 1);
    ASSERT_TRUE(d.ok());
    auto batch = prepareLines(d.value(), std::vector<BBox>{}, LinePrepParams{});
    ASSERT_TRUE(batch.ok());
    EXPECT_EQ(batch.value().count, 0);
}

TEST(PrepareLines, RejectsColorInput) {
    auto d = DeviceImage::create(10, 10, 3);
    ASSERT_TRUE(d.ok());
    EXPECT_FALSE(prepareLines(d.value(), std::vector<BBox>{{0, 0, 5, 5}}, LinePrepParams{}).ok());
}

}  // namespace
}  // namespace cudaocr
