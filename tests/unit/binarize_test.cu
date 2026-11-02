#include <gtest/gtest.h>

#include "cudaocr/stages/binarize.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

TEST(Histogram, MatchesOracle) {
    const HostImage img = oracle::randomImage(307, 211, 1, /*seed=*/3);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto hist = histogram256(dev.value());
    ASSERT_TRUE(hist.ok()) << hist.status().message;
    EXPECT_EQ(hist.value(), oracle::histogram256(img));
}

TEST(AdaptiveThreshold, MatchesOracle) {
    const HostImage img = oracle::randomImage(220, 140, 1, /*seed=*/9);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    for (auto [blockSize, C] : {std::pair<int, int>{3, 2}, {31, 10}, {15, 5}}) {
        auto out = DeviceImage::create(img.width, img.height, 1);
        ASSERT_TRUE(out.ok());
        Status s = adaptiveThreshold(dev.value(), out.value(), blockSize, C);
        ASSERT_TRUE(s.ok()) << s.message;
        HostImage got;
        ASSERT_TRUE(out.value().download(got).ok());
        EXPECT_TRUE(testutil::imagesEqual(got, oracle::adaptiveThreshold(img, blockSize, C)))
            << "blockSize=" << blockSize << " C=" << C;
    }
}

TEST(AdaptiveThreshold, OutputIsBinary) {
    const auto page = oracle::makeSyntheticPage();
    auto dev = DeviceImage::fromHost(page.image);
    ASSERT_TRUE(dev.ok());
    auto out = DeviceImage::create(page.image.width, page.image.height, 1);
    ASSERT_TRUE(out.ok());
    ASSERT_TRUE(adaptiveThreshold(dev.value(), out.value(), 31, 10).ok());
    HostImage got;
    ASSERT_TRUE(out.value().download(got).ok());
    for (auto p : got.data) EXPECT_TRUE(p == 0 || p == 255);
}

TEST(Otsu, MatchesOracleThreshold) {
    // Bimodal image: two flat regions plus noise.
    HostImage img = oracle::randomImage(200, 100, 1, /*seed=*/21);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
            img.at(x, y) =
                (x < img.width / 2) ? (40 + img.at(x, y) % 30) : (190 + img.at(x, y) % 30);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto t = otsuThreshold(dev.value());
    ASSERT_TRUE(t.ok()) << t.status().message;
    EXPECT_EQ(t.value(), oracle::otsuThreshold(oracle::histogram256(img)));
    EXPECT_GT(t.value(), 60);
    EXPECT_LT(t.value(), 195);
}

TEST(ApplyThreshold, MatchesOracle) {
    const HostImage img = oracle::randomImage(99, 77, 1, /*seed=*/17);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto out = DeviceImage::create(img.width, img.height, 1);
    ASSERT_TRUE(out.ok());
    ASSERT_TRUE(applyThreshold(dev.value(), out.value(), 128).ok());
    HostImage got;
    ASSERT_TRUE(out.value().download(got).ok());
    EXPECT_TRUE(testutil::imagesEqual(got, oracle::applyThreshold(img, 128)));
}

TEST(ApplyThreshold, IsIdempotent) {
    const HostImage img = oracle::randomImage(64, 64, 1, /*seed=*/19);
    auto dev = DeviceImage::fromHost(img);
    ASSERT_TRUE(dev.ok());
    auto once = DeviceImage::create(img.width, img.height, 1);
    auto twice = DeviceImage::create(img.width, img.height, 1);
    ASSERT_TRUE(once.ok() && twice.ok());
    ASSERT_TRUE(applyThreshold(dev.value(), once.value(), 100).ok());
    ASSERT_TRUE(applyThreshold(once.value(), twice.value(), 100).ok());
    HostImage a, b;
    ASSERT_TRUE(once.value().download(a).ok());
    ASSERT_TRUE(twice.value().download(b).ok());
    EXPECT_TRUE(testutil::imagesEqual(a, b));
}

}  // namespace
}  // namespace cudaocr
