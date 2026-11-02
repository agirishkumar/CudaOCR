// Regression (found by MOD.4's legacy-equivalence test): pipeline streams are
// cudaStreamNonBlocking and do NOT synchronize with the legacy default stream.
// DeviceImage::upload (pageable cudaMemcpy2D, returns before the DMA lands) and
// DeviceBuffer::zero (cudaMemset) ran on the default stream, so a histogram
// kernel on the pipeline stream read a half-uploaded image and its counts were
// then wiped by the late memset: the second pipeline run in a process saw an
// all-zero histogram and misjudged page polarity. All transfers and fills now
// take the caller's stream.
#include <gtest/gtest.h>

#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/binarize.cuh"
#include "cudaocr/stages/segment.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

// The race is timing-dependent (~1 in 3 runs on an RTX 3060 with a page-sized
// image), so each test repeats the exact pipeline pattern — fresh non-blocking
// stream, upload, then a zero-initialised reduction on that stream — 50 times.
constexpr int kReps = 50;

const HostImage& pageImage() {
    static const HostImage img = oracle::invert(oracle::makeSyntheticPage().image);
    return img;
}

TEST(StreamOrdering, UploadThenHistogramOnCallerStream) {
    const auto want = oracle::histogram256(pageImage());
    for (int rep = 0; rep < kReps; ++rep) {
        auto stream = Stream::create();
        ASSERT_TRUE(stream.ok());
        auto d = DeviceImage::fromHost(pageImage(), stream.value().get());
        ASSERT_TRUE(d.ok());
        auto hist = histogram256(d.value(), stream.value().get());
        ASSERT_TRUE(hist.ok()) << hist.status().message;
        ASSERT_EQ(hist.value(), want) << "rep " << rep << ": histogram raced its zero-fill";
    }
}

TEST(StreamOrdering, UploadThenRowProfileOnCallerStream) {
    const auto want = oracle::rowProfile(pageImage());
    for (int rep = 0; rep < kReps; ++rep) {
        auto stream = Stream::create();
        ASSERT_TRUE(stream.ok());
        auto d = DeviceImage::fromHost(pageImage(), stream.value().get());
        ASSERT_TRUE(d.ok());
        auto rows = rowProfile(d.value(), stream.value().get());
        ASSERT_TRUE(rows.ok());
        ASSERT_EQ(rows.value(), want) << "rep " << rep << ": profile raced its zero-fill";
    }
}

__global__ void slowFillKernel(std::uint8_t* p, size_t pitch, int w, int h, std::uint8_t v) {
    const long long start = clock64();
    while (clock64() - start < 50'000'000) {
    }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) p[static_cast<size_t>(y) * pitch + x] = v;
}

TEST(StreamOrdering, ImageDownloadWaitsForCallerStream) {
    auto stream = Stream::create();
    ASSERT_TRUE(stream.ok());
    auto d = DeviceImage::fromHost(HostImage::create(64, 32, 1), stream.value().get());
    ASSERT_TRUE(d.ok());
    slowFillKernel<<<1, 1, 0, stream.value().get()>>>(d.value().data(), d.value().pitch(), 64, 32,
                                                      7);
    HostImage back;
    ASSERT_TRUE(d.value().download(back, stream.value().get()).ok());
    for (auto v : back.data) ASSERT_EQ(v, 7) << "download raced the producing kernel";
}

}  // namespace
}  // namespace cudaocr
