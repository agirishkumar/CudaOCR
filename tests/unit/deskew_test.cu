#include <gtest/gtest.h>

#include <cmath>

#include "cudaocr/stages/deskew.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

TEST(Rotate, MatchesOracleWithinTolerance) {
    const auto page = oracle::makeSyntheticPage();
    auto dev = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(dev.ok());
    for (float angle : {2.5f, -7.0f, 12.25f}) {
        auto out = DeviceImage::create(page.image.width, page.image.height, 1);
        ASSERT_TRUE(out.ok());
        Status s = rotate(dev.value(), out.value(), angle, 255);
        ASSERT_TRUE(s.ok()) << s.message;
        HostImage got;
        ASSERT_TRUE(out.value().download(got).ok());
        // Binary input: an FMA-induced half-ulp flip at a glyph edge changes a pixel
        // by 255, so compare by outlier fraction rather than max diff.
        EXPECT_TRUE(testutil::imagesClose(got, oracle::rotate(page.image, angle, 255),
                                          /*tol=*/2, /*maxOutlierFrac=*/0.005))
            << "angle=" << angle;
    }
}

TEST(Rotate, ZeroAngleIsIdentity) {
    const HostImage img = oracle::randomImage(80, 50, 1, /*seed=*/37);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto out = DeviceImage::create(img.width, img.height, 1);
    ASSERT_TRUE(out.ok());
    ASSERT_TRUE(rotate(dev.value(), out.value(), 0.0f, 255).ok());
    HostImage got;
    ASSERT_TRUE(out.value().download(got).ok());
    EXPECT_TRUE(testutil::imagesEqual(got, img));
}

// Regression for the legacy hough kernel's out-of-bounds accumulator write
// (bounds check was r < 2*rho against an allocation of rho*theta ints; see
// docs/product.md §6). Wide aspect ratio maximizes |rho| relative to the bins.
TEST(DetectSkew, HoughAccumulatorBoundsRegression) {
    HostImage wide = HostImage::create(2048, 64, 1, 255);
    for (int x = 0; x < wide.width; ++x)
        for (int y = 30; y < 34; ++y) wide.at(x, y) = 0;
    auto dev = DeviceImage::fromHost(wide);
    ASSERT_TRUE(dev.ok());
    auto angle = detectSkewAngle(dev.value());
    ASSERT_TRUE(angle.ok()) << angle.status().message;  // legacy kernel crashed here
    EXPECT_NEAR(angle.value(), 0.0f, 0.25f);
}

TEST(DetectSkew, RecoversKnownRotation) {
    const auto page = oracle::makeSyntheticPage();
    for (float trueAngle : {0.0f, 2.5f, -4.75f, 9.0f}) {
        const HostImage rotated = oracle::rotate(page.image, trueAngle, 255);
        auto dev = DeviceImage::fromHost(rotated);
        ASSERT_TRUE(dev.ok());
        auto angle = detectSkewAngle(dev.value());
        ASSERT_TRUE(angle.ok()) << angle.status().message;
        EXPECT_NEAR(angle.value(), trueAngle, 0.5f) << "trueAngle=" << trueAngle;
    }
}

TEST(DetectSkew, BlankPageReturnsZero) {
    const HostImage blank = HostImage::create(200, 100, 1, 255);
    auto dev = DeviceImage::fromHost(blank);
    ASSERT_TRUE(dev.ok());
    auto angle = detectSkewAngle(dev.value());
    ASSERT_TRUE(angle.ok());
    EXPECT_EQ(angle.value(), 0.0f);
}

TEST(DetectSkew, RejectsAbsurdResolution) {
    auto dev = DeviceImage::create(32, 32, 1);
    ASSERT_TRUE(dev.ok());
    EXPECT_FALSE(detectSkewAngle(dev.value(), 45.0f, 0.01f).ok());  // > kMaxTheta steps
}

TEST(DeskewRoundTrip, RotateThenCorrectStraightens) {
    const auto page = oracle::makeSyntheticPage();
    const float trueAngle = 3.5f;
    const HostImage rotated = oracle::rotate(page.image, trueAngle, 255);
    auto dev = DeviceImage::fromHost(rotated);
    ASSERT_TRUE(dev.ok());
    auto angle = detectSkewAngle(dev.value());
    ASSERT_TRUE(angle.ok());
    auto straight = DeviceImage::create(rotated.width, rotated.height, 1);
    ASSERT_TRUE(straight.ok());
    ASSERT_TRUE(rotate(dev.value(), straight.value(), -angle.value(), 255).ok());
    auto after = detectSkewAngle(straight.value());
    ASSERT_TRUE(after.ok());
    EXPECT_NEAR(after.value(), 0.0f, 0.5f);
}

}  // namespace
}  // namespace cudaocr
