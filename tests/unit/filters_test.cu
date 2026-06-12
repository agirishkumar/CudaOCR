#include <gtest/gtest.h>

#include "cudaocr/stages/filters.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

using Runner = Status (*)(const DeviceImage&, DeviceImage&, cudaStream_t);

HostImage runFilter(const HostImage& gray,
                    const std::function<Status(const DeviceImage&, DeviceImage&)>& fn) {
    auto in = DeviceImage::fromHost(gray);
    EXPECT_TRUE(in.ok()) << in.status().message;
    auto out = DeviceImage::create(gray.width, gray.height, 1);
    EXPECT_TRUE(out.ok());
    Status s = fn(in.value(), out.value());
    EXPECT_TRUE(s.ok()) << s.message;
    HostImage host;
    EXPECT_TRUE(out.value().download(host).ok());
    return host;
}

TEST(BoxMean, MatchesOracle) {
    const HostImage img = oracle::randomImage(257, 131, 1, /*seed=*/7);
    for (int blockSize : {3, 15, 31}) {
        const HostImage got = runFilter(
            img, [&](const DeviceImage& i, DeviceImage& o) { return boxMean(i, o, blockSize); });
        EXPECT_TRUE(testutil::imagesEqual(got, oracle::boxMean(img, blockSize)))
            << "blockSize=" << blockSize;
    }
}

TEST(BoxMean, RejectsEvenBlockSize) {
    auto in = DeviceImage::create(8, 8, 1);
    auto out = DeviceImage::create(8, 8, 1);
    ASSERT_TRUE(in.ok() && out.ok());
    EXPECT_FALSE(boxMean(in.value(), out.value(), 4).ok());
}

TEST(Median, MatchesOracle3and5) {
    const HostImage img = oracle::randomImage(201, 144, 1, /*seed=*/11);
    for (int k : {3, 5}) {
        const HostImage got = runFilter(
            img, [&](const DeviceImage& i, DeviceImage& o) { return medianFilter(i, o, k); });
        EXPECT_TRUE(testutil::imagesEqual(got, oracle::median(img, k))) << "k=" << k;
    }
}

TEST(Median, RemovesSaltAndPepper) {
    HostImage img = HostImage::create(64, 64, 1, 200);
    img.at(30, 30) = 0;  // lone pepper pixel
    const HostImage got = runFilter(
        img, [&](const DeviceImage& i, DeviceImage& o) { return medianFilter(i, o, 3); });
    EXPECT_EQ(got.at(30, 30), 200);
}

TEST(Median, RejectsUnsupportedKernel) {
    auto in = DeviceImage::create(8, 8, 1);
    auto out = DeviceImage::create(8, 8, 1);
    ASSERT_TRUE(in.ok() && out.ok());
    EXPECT_FALSE(medianFilter(in.value(), out.value(), 7).ok());
}

TEST(Gaussian, MatchesOracle) {
    const HostImage img = oracle::randomImage(190, 122, 1, /*seed=*/13);
    for (auto [k, sigma] : {std::pair<int, double>{3, 0.8}, {5, 1.1}, {7, 0.0}}) {
        const HostImage got = runFilter(img, [&](const DeviceImage& i, DeviceImage& o) {
            return gaussianBlur(i, o, k, sigma);
        });
        EXPECT_TRUE(testutil::imagesEqual(got, oracle::gaussian(img, k, sigma)))
            << "k=" << k << " sigma=" << sigma;
    }
}

TEST(Gaussian, WeightsSumTo65536) {
    for (auto [k, sigma] : {std::pair<int, double>{3, 0.5}, {5, 1.0}, {31, 5.0}}) {
        const auto w = gaussianFixedWeights(k, sigma);
        long long sum = 0;
        for (int v : w) sum += v;
        EXPECT_EQ(sum, 65536) << "k=" << k;
    }
}

}  // namespace
}  // namespace cudaocr
