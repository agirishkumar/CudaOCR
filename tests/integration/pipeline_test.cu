// Component-based Pipeline: end-to-end behavior on synthetic and real pages.
// Exact geometry vs the original pipeline is pinned in pipeline_golden_test.cu.
#include <gtest/gtest.h>

#include "cudaocr/pipeline.hpp"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

PipelineConfig crisp() {
    PipelineConfig cfg;
    cfg.preprocess = {{"polarity", {}}};  // synthetic page is noise-free; no median
    return cfg;
}

TEST(Pipeline, EndToEndOnSkewedSyntheticPage) {
    const auto page = oracle::makeSyntheticPage();
    const float trueAngle = 2.5f;
    const HostImage skewed = oracle::rotate(page.image, trueAngle, 255);

    auto p = Pipeline::create(crisp());
    ASSERT_TRUE(p.ok()) << p.status().message;
    StageImages keep;
    auto res = p.value().run(skewed, &keep);
    ASSERT_TRUE(res.ok()) << res.status().message;
    const PageResult& r = res.value();

    EXPECT_NEAR(r.skewDeg, trueAngle, 0.5f);
    ASSERT_EQ(static_cast<int>(r.lines.size()), page.lineCount);
    for (int li = 0; li < page.lineCount; ++li)
        EXPECT_EQ(static_cast<int>(r.lines[li].words.size()), page.wordsPerLine[li])
            << "line " << li;
    EXPECT_FALSE(r.inverted);
    EXPECT_FALSE(keep.binary.empty());
    EXPECT_FALSE(keep.deskewed.empty());
    EXPECT_FALSE(r.timings.empty());
}

TEST(Pipeline, HandlesInvertedInput) {
    const auto page = oracle::makeSyntheticPage();
    PipelineConfig cfg = crisp();
    cfg.deskewer.reset();
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok());
    auto res = p.value().run(oracle::invert(page.image));
    ASSERT_TRUE(res.ok()) << res.status().message;
    EXPECT_TRUE(res.value().inverted);
    EXPECT_EQ(static_cast<int>(res.value().lines.size()), page.lineCount);
}

TEST(Pipeline, OtsuVariantWorks) {
    const auto page = oracle::makeSyntheticPage();
    PipelineConfig cfg = crisp();
    cfg.binarizer = {"otsu", {}};
    cfg.deskewer.reset();
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok());
    auto res = p.value().run(page.image);
    ASSERT_TRUE(res.ok()) << res.status().message;
    EXPECT_GE(res.value().metrics.at("otsu.threshold"), 0.0);
    EXPECT_EQ(static_cast<int>(res.value().lines.size()), page.lineCount);
}

TEST(Pipeline, RunsOnRealSampleFixture) {
    auto img = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/sherlock.png");
    ASSERT_TRUE(img.ok()) << img.status().message;
    auto p = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(p.ok());
    auto res = p.value().run(img.value());
    ASSERT_TRUE(res.ok()) << res.status().message;
    EXPECT_GE(res.value().lines.size(), 1u);  // real scan: at least one text line
}

TEST(Pipeline, RejectsEmptyInput) {
    auto p = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(p.ok());
    EXPECT_FALSE(p.value().run(HostImage{}).ok());
}

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

namespace cudaocr {
namespace {

// M3.11: denoising that helps binarization (median) blurs thin gaps the
// recognizer needs (it appeared to merge doubled letters on one scan), so binarizePrep
// stages feed only the binarizer; the gray page the recognizer reads (and
// StageImages::gray) skips them.
TEST(Pipeline, BinarizePrepDoesNotTouchTheRecognizerGray) {
    const auto page = oracle::makeSyntheticPage();
    PipelineConfig cfg;
    cfg.preprocess = {{"polarity", {}}};
    cfg.binarizePrep = {{"median", Params().set("k", 5)}};
    cfg.deskewer.reset();
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok()) << p.status().message;
    const HostImage noisy = oracle::randomImage(page.image.width, page.image.height, 1, 3);
    StageImages keep;
    ASSERT_TRUE(p.value().run(noisy, &keep).ok());
    EXPECT_TRUE(testutil::imagesEqual(keep.gray, noisy));  // polarity no-op, no median
    EXPECT_TRUE(testutil::imagesEqual(keep.deskewedGray, noisy));
    // ...while the binarizer did see the median-filtered page.
    PipelineConfig noMedian = cfg;
    noMedian.binarizePrep.clear();
    auto q = Pipeline::create(noMedian);
    StageImages keep2;
    ASSERT_TRUE(q.value().run(noisy, &keep2).ok());
    EXPECT_FALSE(testutil::imagesEqual(keep.binary, keep2.binary));
}

TEST(Pipeline, BinarizePrepErrorsNameTheSlot) {
    PipelineConfig cfg;
    cfg.binarizePrep = {{"median", Params().set("k", 4)}};
    auto p = Pipeline::create(cfg);
    ASSERT_FALSE(p.ok());
    EXPECT_NE(p.status().message.find("binarizePrep[0]"), std::string::npos) << p.status().message;
}

}  // namespace
}  // namespace cudaocr
