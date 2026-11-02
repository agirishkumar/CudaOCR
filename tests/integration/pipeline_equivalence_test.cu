// MOD.4: the component-based Pipeline must reproduce the legacy hard-wired
// pipeline exactly (skew, polarity, every line/word/glyph box).
#include <gtest/gtest.h>

#include "cudaocr/legacy_pipeline.hpp"
#include "cudaocr/pipeline.hpp"
#include "oracles/oracles.hpp"

namespace cudaocr {
namespace {

PipelineConfig fromLegacy(const LegacyPipelineConfig& c) {
    PipelineConfig cfg;
    cfg.preprocess.clear();
    if (c.equalize) cfg.preprocess.push_back({"equalize", {}});
    if (c.medianK > 0) cfg.preprocess.push_back({"median", Params().set("k", c.medianK)});
    cfg.preprocess.push_back({"polarity", {}});
    cfg.binarizer =
        c.useOtsu ? ComponentSpec{"otsu", {}}
                  : ComponentSpec{"adaptive", Params().set("blockSize", c.blockSize).set("C", c.C)};
    if (c.deskew)
        cfg.deskewer = ComponentSpec{
            "hough", Params().set("maxAngle", c.maxSkewDeg).set("step", c.skewStepDeg)};
    else
        cfg.deskewer.reset();
    cfg.detector = {"projection", Params().set("minLineHeight", c.minLineHeight)};
    return cfg;
}

void expectSamePage(const LegacyPipelineResult& old, const PageResult& now, int width) {
    EXPECT_EQ(now.skewDeg, old.skewAngleDeg);
    EXPECT_EQ(now.inverted, old.wasInverted);
    const auto want = toLineResults(old.lines, width);
    ASSERT_EQ(now.lines.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(now.lines[i].box, want[i].box) << "line " << i;
        ASSERT_EQ(now.lines[i].words.size(), want[i].words.size()) << "line " << i;
        for (size_t w = 0; w < want[i].words.size(); ++w) {
            EXPECT_EQ(now.lines[i].words[w].box, want[i].words[w].box);
            EXPECT_EQ(now.lines[i].words[w].glyphs, want[i].words[w].glyphs);
        }
    }
}

struct Case {
    std::string name;
    HostImage image;
    LegacyPipelineConfig cfg;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    const auto page = oracle::makeSyntheticPage();
    LegacyPipelineConfig crisp;
    crisp.medianK = 0;
    out.push_back({"synthetic_skewed", oracle::rotate(page.image, 2.5f, 255), crisp});
    LegacyPipelineConfig inv = crisp;
    inv.deskew = false;
    out.push_back({"synthetic_inverted", oracle::invert(page.image), inv});
    LegacyPipelineConfig otsu = inv;
    otsu.useOtsu = true;
    out.push_back({"synthetic_otsu", page.image, otsu});

    LegacyPipelineConfig def, otsuNoDeskew, eqMedian5;
    otsuNoDeskew.useOtsu = true;
    otsuNoDeskew.deskew = false;
    eqMedian5.equalize = true;
    eqMedian5.medianK = 5;
    eqMedian5.blockSize = 15;
    eqMedian5.C = 4;
    for (const char* f :
         {"sample.png", "sample.jpg", "noisy_text.png", "noisy_text2.png", "vigorous-before.png"}) {
        auto img = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/" + f);
        EXPECT_TRUE(img.ok()) << f;
        if (!img.ok()) continue;
        out.push_back({std::string(f) + "/default", img.value(), def});
        out.push_back({std::string(f) + "/otsu_nodeskew", img.value(), otsuNoDeskew});
        out.push_back({std::string(f) + "/equalize_median5", img.value(), eqMedian5});
    }
    return out;
}

TEST(PipelineEquivalence, MatchesLegacyOnAllFixtures) {
    for (const auto& c : cases()) {
        SCOPED_TRACE(c.name);
        auto old = runLegacyPipeline(c.image, c.cfg);
        ASSERT_TRUE(old.ok()) << old.status().message;
        auto pipe = Pipeline::create(fromLegacy(c.cfg));
        ASSERT_TRUE(pipe.ok()) << pipe.status().message;
        auto now = pipe.value().run(c.image);
        ASSERT_TRUE(now.ok()) << now.status().message;
        expectSamePage(old.value(), now.value(), c.image.width);
    }
}

TEST(PipelineEquivalence, DefaultConfigIsLegacyDefault) {
    const auto img = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/sample.png");
    ASSERT_TRUE(img.ok());
    auto old = runLegacyPipeline(img.value(), LegacyPipelineConfig{});
    auto pipe = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(old.ok() && pipe.ok());
    auto now = pipe.value().run(img.value());
    ASSERT_TRUE(now.ok());
    expectSamePage(old.value(), now.value(), img.value().width);
}

}  // namespace
}  // namespace cudaocr
