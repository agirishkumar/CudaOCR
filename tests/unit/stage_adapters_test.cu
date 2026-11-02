// MOD.3: each component adapter must produce exactly what the free function it
// wraps produces, and reject bad parameters at construction.
#include <gtest/gtest.h>

#include "cudaocr/components/hough_deskewer.hpp"
#include "cudaocr/components/preprocess.hpp"
#include "cudaocr/components/projection_detector.hpp"
#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/binarize.cuh"
#include "cudaocr/stages/deskew.cuh"
#include "cudaocr/stages/enhance.cuh"
#include "cudaocr/stages/filters.cuh"
#include "cudaocr/stages/grayscale.cuh"
#include "cudaocr/stages/segment.cuh"
#include "oracles/oracles.hpp"
#include "test_util.hpp"

namespace cudaocr {
namespace {

using testutil::imagesEqual;

DeviceImage up(const HostImage& h) {
    auto d = DeviceImage::fromHost(h);
    EXPECT_TRUE(d.ok()) << d.status().message;
    return d.take();
}

HostImage down(const DeviceImage& d) {
    HostImage h;
    EXPECT_TRUE(d.download(h).ok());
    return h;
}

// Runs a stage through the interface only.
HostImage runStage(ImageStage& stage, const HostImage& in, PageResult* page = nullptr) {
    DeviceImage din = up(in), dout;
    StageContext ctx;
    ctx.page = page;
    Status s = stage.run(din, dout, ctx);
    EXPECT_TRUE(s.ok()) << stage.name() << ": " << s.message;
    return down(dout);
}

// Calls a free function with a preallocated output of the input's shape.
template <typename F>
HostImage runDirect(const HostImage& in, int outChannels, F&& fn) {
    DeviceImage din = up(in);
    auto out = DeviceImage::create(in.width, in.height, outChannels);
    EXPECT_TRUE(out.ok());
    Status s = fn(din, out.value());
    EXPECT_TRUE(s.ok()) << s.message;
    return down(out.value());
}

template <typename T>
std::unique_ptr<T> make(const Params& p = {}) {
    auto r = T::create(p);
    EXPECT_TRUE(r.ok()) << r.status().message;
    return r.ok() ? r.take() : nullptr;
}

template <typename T>
void expectRejected(const Params& p, const char* why) {
    auto r = T::create(p);
    ASSERT_FALSE(r.ok()) << why;
    EXPECT_EQ(r.status().code, ErrorCode::InvalidArgument) << why;
}

const HostImage& gray() {
    static const HostImage img = oracle::randomImage(97, 61, 1, /*seed=*/11);
    return img;
}

TEST(GrayscaleStage, MatchesFreeFunctionAndPassesThroughGray) {
    const HostImage rgb = oracle::randomImage(97, 61, 3, 3);
    auto st = make<GrayscaleStage>();
    EXPECT_TRUE(imagesEqual(runStage(*st, rgb),
                            runDirect(rgb, 1, [](auto& i, auto& o) { return grayscale(i, o); })));
    EXPECT_TRUE(imagesEqual(runStage(*st, gray()), gray()));  // 1-channel input: copy
    EXPECT_EQ(st->name(), "grayscale");
}

TEST(EqualizeStage, MatchesFreeFunction) {
    auto st = make<EqualizeStage>();
    EXPECT_TRUE(imagesEqual(runStage(*st, gray()), runDirect(gray(), 1, [](auto& i, auto& o) {
                                return histogramEqualize(i, o);
                            })));
}

TEST(MedianStage, MatchesFreeFunctionAndValidatesK) {
    for (int k : {3, 5}) {
        auto st = make<MedianStage>(Params().set("k", k));
        EXPECT_TRUE(imagesEqual(
            runStage(*st, gray()),
            runDirect(gray(), 1, [k](auto& i, auto& o) { return medianFilter(i, o, k); })))
            << "k=" << k;
    }
    expectRejected<MedianStage>(Params().set("k", 4), "even k");
    expectRejected<MedianStage>(Params().set("k", 0), "k=0 (omit the stage instead)");
    expectRejected<MedianStage>(Params().set("size", 3), "unknown key");
}

TEST(GaussianStage, MatchesFreeFunctionAndValidates) {
    auto st = make<GaussianStage>(Params().set("ksize", 5).set("sigma", 1.2));
    EXPECT_TRUE(imagesEqual(runStage(*st, gray()), runDirect(gray(), 1, [](auto& i, auto& o) {
                                return gaussianBlur(i, o, 5, 1.2);
                            })));
    expectRejected<GaussianStage>(Params().set("ksize", 4), "even ksize");
    expectRejected<GaussianStage>(Params().set("sigma", -1), "negative sigma");
}

TEST(PolarityStage, InvertsLightOnDarkAndRecordsIt) {
    const auto page = oracle::makeSyntheticPage();
    const HostImage dark = oracle::invert(page.image);
    auto st = make<PolarityStage>();

    PageResult meta;
    EXPECT_TRUE(imagesEqual(runStage(*st, dark, &meta), page.image));
    EXPECT_TRUE(meta.inverted);

    PageResult meta2;
    EXPECT_TRUE(imagesEqual(runStage(*st, page.image, &meta2), page.image));  // untouched
    EXPECT_FALSE(meta2.inverted);
}

TEST(AdaptiveBinarizer, MatchesFreeFunctionAndValidates) {
    auto st = make<AdaptiveBinarizer>(Params().set("blockSize", 15).set("C", 4));
    EXPECT_TRUE(imagesEqual(runStage(*st, gray()), runDirect(gray(), 1, [](auto& i, auto& o) {
                                return adaptiveThreshold(i, o, 15, 4);
                            })));
    auto defaults = make<AdaptiveBinarizer>();  // documented defaults: 31, 10
    EXPECT_TRUE(imagesEqual(runStage(*defaults, gray()), runDirect(gray(), 1, [](auto& i, auto& o) {
                                return adaptiveThreshold(i, o, 31, 10);
                            })));
    expectRejected<AdaptiveBinarizer>(Params().set("blockSize", 16), "even blockSize");
    expectRejected<AdaptiveBinarizer>(Params().set("blockSize", 1), "blockSize < 3");
    expectRejected<AdaptiveBinarizer>(Params().set("C", "x"), "non-int C");
}

TEST(OtsuBinarizer, MatchesFreeFunctionsAndExposesThreshold) {
    auto st = make<OtsuBinarizer>();
    const HostImage got = runStage(*st, gray());
    DeviceImage din = up(gray());
    auto t = otsuThreshold(din);
    ASSERT_TRUE(t.ok());
    EXPECT_EQ(st->lastThreshold(), t.value());
    EXPECT_TRUE(imagesEqual(got, runDirect(gray(), 1, [&](auto& i, auto& o) {
                                return applyThreshold(i, o, t.value());
                            })));
}

TEST(HoughDeskewer, MatchesFreeFunctions) {
    const auto page = oracle::makeSyntheticPage();
    const HostImage skewed = oracle::rotate(page.image, 3.0f, 255);
    auto dk = make<HoughDeskewer>(Params().set("maxAngle", 10).set("step", 0.5));
    DeviceImage bin = up(skewed);
    StageContext ctx;

    auto est = dk->estimate(bin, ctx);
    auto ref = detectSkewAngle(bin, 10.0f, 0.5f);
    ASSERT_TRUE(est.ok() && ref.ok());
    EXPECT_EQ(est.value(), ref.value());

    DeviceImage out;
    ASSERT_TRUE(dk->apply(-est.value(), bin, out, 255, ctx).ok());
    EXPECT_TRUE(imagesEqual(down(out), runDirect(skewed, 1, [&](auto& i, auto& o) {
                                return rotate(i, o, -ref.value(), 255);
                            })));
}

TEST(HoughDeskewer, AnglesBelowResolutionAreACopy) {
    auto dk = make<HoughDeskewer>(Params().set("step", 0.5));
    DeviceImage in = up(gray()), out;
    StageContext ctx;
    ASSERT_TRUE(dk->apply(0.2f, in, out, 255, ctx).ok());  // < step/2
    EXPECT_TRUE(imagesEqual(down(out), gray()));
    expectRejected<HoughDeskewer>(Params().set("step", 0), "zero step");
    expectRejected<HoughDeskewer>(Params().set("maxAngle", 50), "maxAngle > 45");
}

TEST(ProjectionDetector, MatchesFreeFunctions) {
    const auto page = oracle::makeSyntheticPage();
    auto det = make<ProjectionDetector>(Params().set("minLineHeight", 3));
    DeviceImage bin = up(page.image);
    StageContext ctx;
    auto got = det->detect(bin, ctx);
    ASSERT_TRUE(got.ok()) << got.status().message;

    auto rows = rowProfile(bin);
    ASSERT_TRUE(rows.ok());
    auto lines = detectLines(rows.value(), bin.width(), 3);
    for (auto& l : lines) ASSERT_TRUE(segmentLine(bin, l).ok());
    const auto want = toLineResults(lines, bin.width());

    ASSERT_EQ(got.value().size(), want.size());
    ASSERT_EQ(static_cast<int>(want.size()), page.lineCount);
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(got.value()[i].box, want[i].box) << "line " << i;
        ASSERT_EQ(got.value()[i].words.size(), want[i].words.size()) << "line " << i;
        for (size_t w = 0; w < want[i].words.size(); ++w)
            EXPECT_EQ(got.value()[i].words[w].box, want[i].words[w].box);
    }
    expectRejected<ProjectionDetector>(Params().set("minLineHeight", 0), "minLineHeight < 1");
}

}  // namespace
}  // namespace cudaocr
