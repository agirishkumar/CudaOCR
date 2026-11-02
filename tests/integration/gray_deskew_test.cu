// M3.3: the recognizer reads the gray page, so the pipeline must deskew the gray
// image with exactly the angle it applied to the binary one. Alignment is
// measured as IoU of the dark-pixel masks (< 128) of the deskewed gray and the
// deskewed binary image.
#include <gtest/gtest.h>

#include <filesystem>

#include "cudaocr/pipeline.hpp"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

namespace fs = std::filesystem;

double darkIoU(const HostImage& a, const HostImage& b) {
    EXPECT_EQ(a.width, b.width);
    EXPECT_EQ(a.height, b.height);
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const bool da = a.data[i] < 128, db = b.data[i] < 128;
        inter += da && db;
        uni += da || db;
    }
    return uni ? static_cast<double>(inter) / static_cast<double>(uni) : 1.0;
}

StageImages runKeep(const std::string& fixture, const PipelineConfig& cfg = {}) {
    auto img = loadImage((fs::path(CUDAOCR_SYNTH_DIR) / fixture).string());
    EXPECT_TRUE(img.ok()) << img.status().message;
    auto p = Pipeline::create(cfg);
    EXPECT_TRUE(p.ok()) << p.status().message;
    StageImages keep;
    auto r = p.value().run(img.value(), &keep);
    EXPECT_TRUE(r.ok()) << r.status().message;
    EXPECT_NE(r.value().skewDeg, 0.0f);
    return keep;
}

// Measured (2026-11-02): aligned IoU 0.872 (sans +3°) / 0.796 (serif -4°); against
// the un-deskewed binary 0.097 / 0.053. Aligned is < 1 because adaptive-threshold
// strokes and the gray glyphs' dark (< 128) core differ in width at the edges.
// Tolerance: aligned >= 0.75 and >= 5x the unaligned IoU.
TEST(GrayDeskew, GrayAndBinaryAreRotatedByTheSameAngle) {
    for (const char* f : {"page_sans_skew_p3.png", "page_serif_skew_m4.png"}) {
        SCOPED_TRACE(f);
        const StageImages k = runKeep(f);
        ASSERT_FALSE(k.deskewedGray.empty());
        const double aligned = darkIoU(k.deskewedGray, k.deskewed);
        const double unaligned = darkIoU(k.deskewedGray, k.binary);
        RecordProperty(std::string(f) + ".alignedIoU", std::to_string(aligned));
        EXPECT_GE(aligned, 0.75) << "unaligned IoU " << unaligned;
        EXPECT_GE(aligned, 5.0 * unaligned);
        // The gray page really was rotated (measured 0.092 / 0.048 vs the input gray).
        EXPECT_LT(darkIoU(k.deskewedGray, k.gray), 0.2);
    }
}

TEST(GrayDeskew, DeskewOffLeavesGrayUntouched) {
    PipelineConfig cfg;
    cfg.deskewer.reset();
    auto img = loadImage((fs::path(CUDAOCR_SYNTH_DIR) / "page_sans_skew_p3.png").string());
    ASSERT_TRUE(img.ok());
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok());
    StageImages keep;
    ASSERT_TRUE(p.value().run(img.value(), &keep).ok());
    EXPECT_TRUE(testutil::imagesEqual(keep.deskewedGray, keep.gray));
}

TEST(GrayDeskew, BorderFillIsWhiteOnInvertedInput) {
    // Polarity runs before deskew, so uncovered corners must be paper (white)
    // even when the scan was light-on-dark.
    const auto page = oracle::makeSyntheticPage();
    const HostImage skewedDark = oracle::invert(oracle::rotate(page.image, 3.0f, 255));
    PipelineConfig cfg;
    cfg.preprocess = {{"polarity", {}}};
    auto p = Pipeline::create(cfg);
    ASSERT_TRUE(p.ok());
    StageImages keep;
    auto r = p.value().run(skewedDark, &keep);
    ASSERT_TRUE(r.ok());
    EXPECT_TRUE(r.value().inverted);
    EXPECT_EQ(keep.deskewedGray.at(0, 0), 255);
    EXPECT_EQ(keep.deskewedGray.at(keep.deskewedGray.width - 1, keep.deskewedGray.height - 1), 255);
}

}  // namespace
}  // namespace cudaocr
