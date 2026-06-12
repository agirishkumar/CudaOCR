#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>

#include "cudaocr/core/host_image.hpp"
#include "oracles/oracles.hpp"

namespace cudaocr {
namespace {

TEST(HostImage, CreateAndIndex) {
    HostImage img = HostImage::create(4, 3, 3, 7);
    EXPECT_EQ(img.data.size(), 4u * 3 * 3);
    EXPECT_EQ(img.at(2, 1, 1), 7);
    img.at(2, 1, 1) = 99;
    EXPECT_EQ(img.at(2, 1, 1), 99);
}

TEST(HostImage, PngRoundTrip) {
    const HostImage img = oracle::randomImage(31, 17, 3, /*seed=*/5);
    const auto path = std::filesystem::temp_directory_path() / "cudaocr_roundtrip.png";
    ASSERT_TRUE(saveImage(path.string(), img).ok());
    auto loaded = loadImage(path.string());
    ASSERT_TRUE(loaded.ok()) << loaded.status().message;
    EXPECT_EQ(loaded.value().width, img.width);
    EXPECT_EQ(loaded.value().height, img.height);
    EXPECT_EQ(loaded.value().channels, 3);
    EXPECT_EQ(loaded.value().data, img.data);  // PNG is lossless
    std::filesystem::remove(path);
}

TEST(HostImage, LoadFixtureSample) {
    auto img = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/sample.png");
    ASSERT_TRUE(img.ok()) << img.status().message;
    EXPECT_GT(img.value().width, 0);
    EXPECT_GT(img.value().height, 0);
}

TEST(HostImage, LoadMissingFileFails) {
    auto img = loadImage("/nonexistent/nope.png");
    ASSERT_FALSE(img.ok());
    EXPECT_EQ(img.status().code, ErrorCode::IoError);
}

}  // namespace
}  // namespace cudaocr
