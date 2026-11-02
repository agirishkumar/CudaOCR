// Behavior of the component-based Pipeline beyond legacy equivalence.
#include <gtest/gtest.h>

#include "cudaocr/pipeline.hpp"
#include "oracles/oracles.hpp"

namespace cudaocr {
namespace {

TEST(Pipeline, UnknownComponentFailsAtCreateNamingSlot) {
    PipelineConfig cfg;
    cfg.preprocess.push_back({"sharpen", {}});
    auto p = Pipeline::create(cfg);
    ASSERT_FALSE(p.ok());
    EXPECT_EQ(p.status().code, ErrorCode::InvalidArgument);
    EXPECT_NE(p.status().message.find("'sharpen'"), std::string::npos) << p.status().message;
    EXPECT_NE(p.status().message.find("preprocess[2]"), std::string::npos) << p.status().message;
}

TEST(Pipeline, BadParamFailsAtCreate) {
    PipelineConfig cfg;
    cfg.binarizer = {"adaptive", Params().set("blockSize", 8)};
    auto p = Pipeline::create(cfg);
    ASSERT_FALSE(p.ok());
    EXPECT_NE(p.status().message.find("binarizer"), std::string::npos) << p.status().message;
}

TEST(Pipeline, RejectsEmptyInput) {
    auto p = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(p.ok());
    EXPECT_FALSE(p.value().run(HostImage{}).ok());
}

TEST(Pipeline, ReusableAcrossPagesOfDifferentSizes) {
    auto p = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(p.ok());
    const auto big = oracle::makeSyntheticPage(640, 240);
    const auto small = oracle::makeSyntheticPage(320, 200);
    for (int rep = 0; rep < 2; ++rep) {
        for (const auto* page : {&big, &small}) {
            auto r = p.value().run(page->image);
            ASSERT_TRUE(r.ok()) << r.status().message;
            EXPECT_EQ(static_cast<int>(r.value().lines.size()), page->lineCount);
            EXPECT_EQ(r.value().width, page->image.width);
        }
    }
}

TEST(Pipeline, RecordsTimingsMetricsAndStageImages) {
    PipelineConfig cfg;
    cfg.binarizer = {"otsu", {}};
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok());
    const auto page = oracle::makeSyntheticPage();
    StageImages keep;
    auto r = p.value().run(page.image, &keep);
    ASSERT_TRUE(r.ok()) << r.status().message;
    std::vector<std::string> names;
    for (const auto& t : r.value().timings) names.push_back(t.name);
    EXPECT_EQ(names,
              (std::vector<std::string>{"median", "polarity", "otsu", "hough", "projection"}));
    ASSERT_TRUE(r.value().metrics.count("otsu.threshold"));
    EXPECT_GE(r.value().metrics.at("otsu.threshold"), 0.0);
    EXPECT_FALSE(keep.gray.empty());
    EXPECT_FALSE(keep.binary.empty());
    EXPECT_FALSE(keep.deskewed.empty());
}

TEST(Pipeline, DeskewCanBeDisabled) {
    PipelineConfig cfg;
    cfg.deskewer.reset();
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok());
    const auto page = oracle::makeSyntheticPage();
    auto r = p.value().run(oracle::rotate(page.image, 3.0f, 255));
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().skewDeg, 0.0f);
}

}  // namespace
}  // namespace cudaocr
