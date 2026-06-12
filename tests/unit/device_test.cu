#include <gtest/gtest.h>

#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/core/device_image.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

TEST(DeviceBuffer, RoundTrip) {
    auto buf = DeviceBuffer<int>::create(1024);
    ASSERT_TRUE(buf.ok()) << buf.status().message;
    std::vector<int> host(1024);
    for (int i = 0; i < 1024; ++i) host[i] = i * 3 - 7;
    ASSERT_TRUE(buf.value().upload(host).ok());
    std::vector<int> back;
    ASSERT_TRUE(buf.value().download(back).ok());
    EXPECT_EQ(back, host);
}

TEST(DeviceBuffer, ZeroFills) {
    auto buf = DeviceBuffer<unsigned int>::create(64);
    ASSERT_TRUE(buf.ok());
    ASSERT_TRUE(buf.value().zero().ok());
    std::vector<unsigned int> back;
    ASSERT_TRUE(buf.value().download(back).ok());
    for (auto v : back) EXPECT_EQ(v, 0u);
}

TEST(DeviceBuffer, RejectsZeroSize) {
    auto buf = DeviceBuffer<int>::create(0);
    EXPECT_FALSE(buf.ok());
}

TEST(DeviceImage, RoundTripPreservesPixels) {
    const HostImage host = oracle::randomImage(123, 77, 3, /*seed=*/1);
    auto dev = DeviceImage::fromHost(host);
    ASSERT_TRUE(dev.ok()) << dev.status().message;
    EXPECT_GE(dev.value().pitch(), host.rowBytes());
    HostImage back;
    ASSERT_TRUE(dev.value().download(back).ok());
    EXPECT_TRUE(testutil::imagesEqual(host, back));
}

TEST(DeviceImage, MoveTransfersOwnership) {
    auto a = DeviceImage::create(16, 16, 1);
    ASSERT_TRUE(a.ok());
    std::uint8_t* ptr = a.value().data();
    DeviceImage b = std::move(a.value());
    EXPECT_EQ(b.data(), ptr);
    EXPECT_TRUE(a.value().empty());  // NOLINT(bugprone-use-after-move) intentional check
}

TEST(DeviceImage, RejectsBadShapes) {
    EXPECT_FALSE(DeviceImage::create(0, 5, 1).ok());
    EXPECT_FALSE(DeviceImage::create(5, 5, 2).ok());
}

TEST(DeviceImage, UploadShapeMismatchFails) {
    auto dev = DeviceImage::create(8, 8, 1);
    ASSERT_TRUE(dev.ok());
    const HostImage wrong = HostImage::create(9, 8, 1);
    EXPECT_FALSE(dev.value().upload(wrong).ok());
}

}  // namespace
}  // namespace cudaocr
