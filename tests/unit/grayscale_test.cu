#include <gtest/gtest.h>

#include "cudaocr/stages/grayscale.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

HostImage runGrayscale(const HostImage& rgb) {
    auto in = DeviceImage::fromHost(rgb);
    EXPECT_TRUE(in.ok()) << in.status().message;
    auto out = DeviceImage::create(rgb.width, rgb.height, 1);
    EXPECT_TRUE(out.ok());
    Status s = grayscale(in.value(), out.value());
    EXPECT_TRUE(s.ok()) << s.message;
    HostImage host;
    EXPECT_TRUE(out.value().download(host).ok());
    return host;
}

TEST(Grayscale, MatchesOracleOnRandomImage) {
    const HostImage rgb = oracle::randomImage(321, 199, 3, /*seed=*/42);
    EXPECT_TRUE(testutil::imagesEqual(runGrayscale(rgb), oracle::grayscale(rgb)));
}

TEST(Grayscale, MatchesOracleOnSampleFixture) {
    // Real page (RGBA PNG loaded as 3 channels); 1-channel inputs are replicated.
    auto loaded = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/sherlock.png");
    ASSERT_TRUE(loaded.ok()) << loaded.status().message;
    const HostImage& src = loaded.value();
    HostImage rgb = HostImage::create(src.width, src.height, 3);
    for (int y = 0; y < src.height; ++y)
        for (int x = 0; x < src.width; ++x)
            for (int c = 0; c < 3; ++c) rgb.at(x, y, c) = src.at(x, y, src.channels == 3 ? c : 0);
    EXPECT_TRUE(testutil::imagesEqual(runGrayscale(rgb), oracle::grayscale(rgb)));
}

TEST(Grayscale, PureColorsUseBt601Weights) {
    HostImage rgb = HostImage::create(3, 1, 3);
    rgb.at(0, 0, 0) = 255;  // pure red
    rgb.at(1, 0, 1) = 255;  // pure green
    rgb.at(2, 0, 2) = 255;  // pure blue
    const HostImage gray = runGrayscale(rgb);
    EXPECT_EQ(gray.at(0, 0), (77 * 255 + 128) >> 8);
    EXPECT_EQ(gray.at(1, 0), (150 * 255 + 128) >> 8);
    EXPECT_EQ(gray.at(2, 0), (29 * 255 + 128) >> 8);
}

TEST(Grayscale, RejectsSingleChannelInput) {
    auto in = DeviceImage::create(8, 8, 1);
    auto out = DeviceImage::create(8, 8, 1);
    ASSERT_TRUE(in.ok() && out.ok());
    EXPECT_FALSE(grayscale(in.value(), out.value()).ok());
}

}  // namespace
}  // namespace cudaocr
