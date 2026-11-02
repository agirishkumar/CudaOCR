// Host-only TU: every component interface must be includable and implementable
// without nvcc. The fakes here double as minimal reference implementations.
#include <gtest/gtest.h>

#include "cudaocr/components/decoder.hpp"
#include "cudaocr/components/deskewer.hpp"
#include "cudaocr/components/formatter.hpp"
#include "cudaocr/components/image_stage.hpp"
#include "cudaocr/components/line_recognizer.hpp"
#include "cudaocr/components/text_detector.hpp"

#include <type_traits>

namespace cudaocr {
namespace {

// The interface headers above must only forward-declare DeviceImage (no CUDA
// headers). Checked before device_image.cuh is included below.
template <typename T, typename = void>
struct IsComplete : std::false_type {};
template <typename T>
struct IsComplete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};
static_assert(!IsComplete<DeviceImage>::value,
              "component headers must not include device_image.cuh");

}  // namespace
}  // namespace cudaocr

#include "cudaocr/core/device_image.cuh"

static_assert(std::is_same_v<cudaStream_t, CUstream_st*>,
              "StageContext::stream must be layout-identical to cudaStream_t");

namespace cudaocr {
namespace {

struct FakeStage : ImageStage {
    std::string_view name() const override { return "fake"; }
    Status run(const DeviceImage&, DeviceImage&, StageContext& ctx) override {
        if (ctx.page) ctx.page->inverted = true;  // stages may annotate the page
        return Status::Good();
    }
};

struct FakeDeskewer : Deskewer {
    std::string_view name() const override { return "fake"; }
    Result<float> estimate(const DeviceImage&, StageContext&) override { return 1.5f; }
    Status apply(float, const DeviceImage&, DeviceImage&, std::uint8_t, StageContext&) override {
        return Status::Good();
    }
};

struct FakeDetector : TextDetector {
    std::string_view name() const override { return "fake"; }
    Result<std::vector<LineResult>> detect(const DeviceImage&, StageContext&) override {
        LineResult l;
        l.box = {0, 0, 9, 4};
        return std::vector<LineResult>{l};
    }
};

struct FakeRecognizer : LineRecognizer {
    Charset cs{{"<blank>", "h", "i"}, 0};
    std::string_view name() const override { return "fake"; }
    const Charset& charset() const override { return cs; }
    Result<std::vector<CtcProbs>> recognize(const DeviceImage&, std::span<const LineResult> lines,
                                            StageContext&) override {
        std::vector<CtcProbs> out;
        for (size_t i = 0; i < lines.size(); ++i) {
            CtcProbs p(2, 3);
            p.at(0, 1) = 1.0f;  // "h"
            p.at(1, 2) = 1.0f;  // "i"
            out.push_back(std::move(p));
        }
        return out;
    }
};

struct ArgmaxDecoder : Decoder {
    std::string_view name() const override { return "argmax"; }
    Result<DecodedLine> decode(const CtcProbs& probs, const Charset& cs) override {
        DecodedLine d;
        for (int t = 0; t < probs.timesteps(); ++t) {
            int best = 0;
            for (int c = 1; c < probs.classes(); ++c)
                if (probs.at(t, c) > probs.at(t, best)) best = c;
            if (best != cs.blank) {
                d.text += cs.symbols[best];
                d.timesteps.push_back(t);
            }
        }
        d.conf = 1.0f;
        return d;
    }
};

struct TextFormatter : Formatter {
    std::string_view name() const override { return "txt"; }
    std::string_view extension() const override { return "txt"; }
    Result<std::string> format(const PageResult& page) override { return page.text(); }
};

TEST(CtcProbs, ShapeAndRowMajorAccess) {
    CtcProbs p(4, 3);
    EXPECT_EQ(p.timesteps(), 4);
    EXPECT_EQ(p.classes(), 3);
    EXPECT_EQ(p.data().size(), 12u);
    p.at(2, 1) = 0.5f;
    EXPECT_FLOAT_EQ(p.data()[2 * 3 + 1], 0.5f);
}

TEST(Components, FakesComposeThroughInterfacesOnly) {
    // Composes the way Pipeline will: only via base-class references.
    PageResult page;
    StageContext ctx;
    ctx.page = &page;
    FakeStage stage;
    FakeDeskewer deskew;
    FakeDetector det;
    FakeRecognizer rec;
    ArgmaxDecoder dec;
    TextFormatter fmt;
    ImageStage& s = stage;
    Deskewer& dk = deskew;
    TextDetector& d = det;
    LineRecognizer& r = rec;
    Decoder& de = dec;
    Formatter& f = fmt;

    const DeviceImage img;  // fakes never touch pixels
    DeviceImage out;

    ASSERT_TRUE(s.run(img, out, ctx).ok());
    EXPECT_TRUE(page.inverted);
    page.skewDeg = dk.estimate(img, ctx).value();
    auto lines = d.detect(img, ctx);
    ASSERT_TRUE(lines.ok());
    page.lines = lines.take();
    auto probs = r.recognize(img, page.lines, ctx);
    ASSERT_TRUE(probs.ok());
    for (size_t i = 0; i < page.lines.size(); ++i) {
        auto line = de.decode(probs.value()[i], r.charset());
        ASSERT_TRUE(line.ok());
        page.lines[i].text = line.value().text;
        page.lines[i].conf = line.value().conf;
    }
    EXPECT_EQ(f.format(page).value(), "hi");
    EXPECT_FLOAT_EQ(page.skewDeg, 1.5f);
    EXPECT_EQ(s.name(), "fake");
    EXPECT_EQ(f.extension(), "txt");
}

}  // namespace
}  // namespace cudaocr
