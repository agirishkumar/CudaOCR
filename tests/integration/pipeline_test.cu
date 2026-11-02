#include <gtest/gtest.h>

#include <cmath>

#include "cudaocr/legacy_pipeline.hpp"
#include "oracles/oracles.hpp"

namespace cudaocr {
namespace {

TEST(Pipeline, EndToEndOnSkewedSyntheticPage) {
    const auto page = oracle::makeSyntheticPage();
    const float trueAngle = 2.5f;
    const HostImage skewed = oracle::rotate(page.image, trueAngle, 255);

    LegacyPipelineConfig cfg;
    cfg.medianK = 0;  // synthetic page is noise-free; keep glyph edges crisp
    auto res = runLegacyPipeline(skewed, cfg, /*keepStages=*/true);
    ASSERT_TRUE(res.ok()) << res.status().message;
    const LegacyPipelineResult& r = res.value();

    EXPECT_NEAR(r.skewAngleDeg, trueAngle, 0.5f);
    ASSERT_EQ(static_cast<int>(r.lines.size()), page.lineCount);
    for (int li = 0; li < page.lineCount; ++li)
        EXPECT_EQ(static_cast<int>(r.lines[li].words.size()), page.wordsPerLine[li])
            << "line " << li;
    EXPECT_FALSE(r.wasInverted);
    EXPECT_FALSE(r.binary.empty());
    EXPECT_FALSE(r.deskewed.empty());
    EXPECT_FALSE(r.timings.empty());
}

TEST(Pipeline, HandlesInvertedInput) {
    const auto page = oracle::makeSyntheticPage();
    const HostImage inverted = oracle::invert(page.image);
    LegacyPipelineConfig cfg;
    cfg.medianK = 0;
    cfg.deskew = false;
    auto res = runLegacyPipeline(inverted, cfg);
    ASSERT_TRUE(res.ok()) << res.status().message;
    EXPECT_TRUE(res.value().wasInverted);
    EXPECT_EQ(static_cast<int>(res.value().lines.size()), page.lineCount);
}

TEST(Pipeline, OtsuVariantWorks) {
    const auto page = oracle::makeSyntheticPage();
    LegacyPipelineConfig cfg;
    cfg.useOtsu = true;
    cfg.medianK = 0;
    cfg.deskew = false;
    auto res = runLegacyPipeline(page.image, cfg);
    ASSERT_TRUE(res.ok()) << res.status().message;
    EXPECT_GE(res.value().otsuThreshold, 0);
    EXPECT_EQ(static_cast<int>(res.value().lines.size()), page.lineCount);
}

TEST(Pipeline, RunsOnRealSampleFixture) {
    auto img = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/sample.png");
    ASSERT_TRUE(img.ok()) << img.status().message;
    auto res = runLegacyPipeline(img.value(), LegacyPipelineConfig{});
    ASSERT_TRUE(res.ok()) << res.status().message;
    EXPECT_GE(res.value().lines.size(), 1u);  // real scan: at least one text line
}

TEST(Pipeline, RejectsEmptyInput) {
    EXPECT_FALSE(runLegacyPipeline(HostImage{}, LegacyPipelineConfig{}).ok());
}

}  // namespace
}  // namespace cudaocr
