#include <gtest/gtest.h>

#include "cudaocr/stages/enhance.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

TEST(Equalize, MatchesOracle) {
    // Low-contrast image: values clustered in [100, 140].
    HostImage img = oracle::randomImage(180, 120, 1, /*seed=*/23);
    for (auto& p : img.data) p = static_cast<std::uint8_t>(100 + p % 40);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto out = DeviceImage::create(img.width, img.height, 1);
    ASSERT_TRUE(out.ok());
    Status s = histogramEqualize(dev.value(), out.value());
    ASSERT_TRUE(s.ok()) << s.message;
    HostImage got;
    ASSERT_TRUE(out.value().download(got).ok());
    EXPECT_TRUE(testutil::imagesEqual(got, oracle::equalize(img)));
}

TEST(Equalize, StretchesContrast) {
    HostImage img = oracle::randomImage(128, 128, 1, /*seed=*/29);
    for (auto& p : img.data) p = static_cast<std::uint8_t>(110 + p % 20);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto out = DeviceImage::create(img.width, img.height, 1);
    ASSERT_TRUE(out.ok());
    ASSERT_TRUE(histogramEqualize(dev.value(), out.value()).ok());
    HostImage got;
    ASSERT_TRUE(out.value().download(got).ok());
    int lo = 255, hi = 0;
    for (auto p : got.data) {
        lo = std::min(lo, int(p));
        hi = std::max(hi, int(p));
    }
    EXPECT_LE(lo, 10);
    EXPECT_GE(hi, 245);
}

TEST(Invert, MatchesOracleAndIsInvolution) {
    const HostImage img = oracle::randomImage(90, 60, 1, /*seed=*/31);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto once = DeviceImage::create(img.width, img.height, 1);
    auto twice = DeviceImage::create(img.width, img.height, 1);
    ASSERT_TRUE(once.ok() && twice.ok());
    ASSERT_TRUE(invert(dev.value(), once.value()).ok());
    HostImage gotOnce;
    ASSERT_TRUE(once.value().download(gotOnce).ok());
    EXPECT_TRUE(testutil::imagesEqual(gotOnce, oracle::invert(img)));
    ASSERT_TRUE(invert(once.value(), twice.value()).ok());
    HostImage gotTwice;
    ASSERT_TRUE(twice.value().download(gotTwice).ok());
    EXPECT_TRUE(testutil::imagesEqual(gotTwice, img));
}

TEST(IsInverted, DetectsPolarity) {
    const auto page = oracle::makeSyntheticPage();  // black text on white: not inverted
    auto normal = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(normal.ok());
    auto r1 = isInverted(normal.value());
    ASSERT_TRUE(r1.ok());
    EXPECT_FALSE(r1.value());

    auto flipped = DeviceImage::fromHost(oracle::invert(page.image));
    ASSERT_TRUE(flipped.ok());
    auto r2 = isInverted(flipped.value());
    ASSERT_TRUE(r2.ok());
    EXPECT_TRUE(r2.value());
}

}  // namespace
}  // namespace cudaocr
