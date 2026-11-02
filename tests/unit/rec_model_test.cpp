// M3.5: the fetched recognition model (scripts/fetch_models.sh) matches the I/O
// contract documented in models/README.md. Skipped when models/ is empty.
#include <gtest/gtest.h>

#include <onnxruntime_cxx_api.h>

#include "cudaocr/core/host_image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
const fs::path kModel = fs::path(CUDAOCR_MODEL_DIR) / "rec.onnx";
const fs::path kDict = fs::path(CUDAOCR_MODEL_DIR) / "dict.txt";

// Documented contract (models/README.md).
constexpr int64_t kHeight = 48;
constexpr int64_t kDictEntries = 436;
constexpr int64_t kClasses = kDictEntries + 2;  // blank (index 0) + dict + space (last)

#define REQUIRE_MODEL()                            \
    if (!fs::exists(kModel) || !fs::exists(kDict)) \
    GTEST_SKIP() << "models not fetched; run scripts/fetch_models.sh (expects " << kModel << ")"

// Process-lifetime singletons, leaked on purpose (see ort_smoke_test.cpp).
Ort::Env& env() {
    static Ort::Env* e = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "cudaocr-rec-test");
    return *e;
}

Ort::Session& session() {
    static Ort::Session* s = [] {
        Ort::SessionOptions opts;
        return new Ort::Session(env(), kModel.c_str(), opts);
    }();
    return *s;
}

// Runs a uniform-gray [1, 3, 48, width] input; returns {T, C, data}.
struct Out {
    int64_t t, c;
    std::vector<float> data;
};
Out runBlank(int64_t width) {
    std::vector<float> x(static_cast<size_t>(3 * kHeight * width), 1.0f);  // white after norm
    const std::array<int64_t, 4> shape{1, 3, kHeight, width};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto in = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), 4);
    const char* inName = "x";
    const char* outName = "fetch_name_0";
    auto out = session().Run(Ort::RunOptions{nullptr}, &inName, &in, 1, &outName, 1);
    const auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    EXPECT_EQ(oshape.size(), 3u);
    const float* p = out[0].GetTensorData<float>();
    const size_t n = static_cast<size_t>(oshape[0] * oshape[1] * oshape[2]);
    return {oshape[1], oshape[2], std::vector<float>(p, p + n)};
}

TEST(RecModel, DictionaryHasDocumentedSize) {
    REQUIRE_MODEL();
    std::ifstream in(kDict);
    int64_t n = 0;
    for (std::string line; std::getline(in, line);) ++n;
    EXPECT_EQ(n, kDictEntries);
}

TEST(RecModel, InputOutputSignature) {
    REQUIRE_MODEL();
    Ort::Session& s = session();
    ASSERT_EQ(s.GetInputCount(), 1u);
    ASSERT_EQ(s.GetOutputCount(), 1u);
    Ort::AllocatorWithDefaultOptions alloc;
    EXPECT_EQ(std::string(s.GetInputNameAllocated(0, alloc).get()), "x");
    EXPECT_EQ(std::string(s.GetOutputNameAllocated(0, alloc).get()), "fetch_name_0");
    // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() is a view into it.
    const Ort::TypeInfo inInfo = s.GetInputTypeInfo(0);
    const auto in = inInfo.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(in.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    EXPECT_EQ(in.GetShape(), (std::vector<int64_t>{-1, 3, kHeight, -1}));  // N, C, H, W
    const Ort::TypeInfo outInfo = s.GetOutputTypeInfo(0);
    const auto out = outInfo.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(out.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    EXPECT_EQ(out.GetShape(), (std::vector<int64_t>{-1, -1, kClasses}));  // N, T, C
}

// Measured 2026-11-02: T = floor(W / 8) for W in {48, 96, 100, 160, 320, 321, 640, 1000}.
TEST(RecModel, TimestepsAreWidthOverEight) {
    REQUIRE_MODEL();
    for (int64_t w : {48, 100, 321, 1000}) {
        const Out o = runBlank(w);
        EXPECT_EQ(o.t, w / 8) << "W=" << w;
        EXPECT_EQ(o.c, kClasses);
    }
}

// The output is already softmax-normalized: each timestep is a distribution.
TEST(RecModel, OutputRowsAreProbabilities) {
    REQUIRE_MODEL();
    const Out o = runBlank(320);
    for (int64_t t = 0; t < o.t; ++t) {
        double sum = 0;
        for (int64_t c = 0; c < o.c; ++c) {
            const float p = o.data[static_cast<size_t>(t * o.c + c)];
            ASSERT_GE(p, 0.0f);
            ASSERT_LE(p, 1.0f);
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0, 1e-4) << "t=" << t;
    }
}

// End-to-end check of the documented preprocessing (CPU reference): gray line
// cropped tight to the ink, bilinear resize to height 48 keeping aspect, replicate to 3 channels,
// x/255 -> (x - 0.5) / 0.5; greedy CTC with blank 0, dict 1..436, space 437.
std::string recognizeLineFixture(const std::string& name) {
    auto img = cudaocr::loadImage(std::string(CUDAOCR_SYNTH_DIR) + "/" + name + ".png");
    EXPECT_TRUE(img.ok()) << img.status().message;
    if (!img.ok() || img.value().channels != 1) return "<bad fixture>";
    // Tight crop to the ink (as the detector's line box will be) + 4 px padding;
    // the fixture's one-font-size margins leave the text ~18 px tall at H=48.
    const cudaocr::HostImage& full = img.value();
    int x0i = full.width, y0i = full.height, x1i = -1, y1i = -1;
    for (int y = 0; y < full.height; ++y)
        for (int xx = 0; xx < full.width; ++xx)
            if (full.at(xx, y) < 128) {
                x0i = std::min(x0i, xx), x1i = std::max(x1i, xx);
                y0i = std::min(y0i, y), y1i = std::max(y1i, y);
            }
    if (x1i < 0) return "<no ink>";
    constexpr int kPad = 4;
    x0i = std::max(0, x0i - kPad), y0i = std::max(0, y0i - kPad);
    x1i = std::min(full.width - 1, x1i + kPad), y1i = std::min(full.height - 1, y1i + kPad);
    cudaocr::HostImage g = cudaocr::HostImage::create(x1i - x0i + 1, y1i - y0i + 1, 1);
    for (int y = 0; y < g.height; ++y)
        for (int xx = 0; xx < g.width; ++xx) g.at(xx, y) = full.at(x0i + xx, y0i + y);
    const int64_t w = std::max<int64_t>(8, std::lround(double(g.width) * kHeight / g.height));
    std::vector<float> x(static_cast<size_t>(3 * kHeight * w));
    for (int64_t y = 0; y < kHeight; ++y)
        for (int64_t xo = 0; xo < w; ++xo) {
            const double sy = std::clamp((y + 0.5) * g.height / kHeight - 0.5, 0.0, g.height - 1.0);
            const double sx =
                std::clamp((xo + 0.5) * g.width / double(w) - 0.5, 0.0, g.width - 1.0);
            const int y0 = int(sy), x0 = int(sx);
            const int y1 = std::min(y0 + 1, g.height - 1), x1 = std::min(x0 + 1, g.width - 1);
            const double fy = sy - y0, fx = sx - x0;
            const double v = (1 - fy) * ((1 - fx) * g.at(x0, y0) + fx * g.at(x1, y0)) +
                             fy * ((1 - fx) * g.at(x0, y1) + fx * g.at(x1, y1));
            const float n = static_cast<float>((v / 255.0 - 0.5) / 0.5);
            for (int c = 0; c < 3; ++c) x[static_cast<size_t>((c * kHeight + y) * w + xo)] = n;
        }
    const std::array<int64_t, 4> shape{1, 3, kHeight, w};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto in = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), 4);
    const char* inName = "x";
    const char* outName = "fetch_name_0";
    auto out = session().Run(Ort::RunOptions{nullptr}, &inName, &in, 1, &outName, 1);
    const auto os = out[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* p = out[0].GetTensorData<float>();

    std::vector<std::string> symbols{""};  // index 0: blank
    std::ifstream dict(kDict);
    for (std::string line; std::getline(dict, line);) symbols.push_back(line);
    symbols.push_back(" ");  // last class: space
    EXPECT_EQ(static_cast<int64_t>(symbols.size()), kClasses);

    std::string text;
    int prev = -1;
    for (int64_t t = 0; t < os[1]; ++t) {
        const float* row = p + t * os[2];
        const int best = static_cast<int>(std::max_element(row, row + os[2]) - row);
        if (best != 0 && best != prev) text += symbols[best];
        prev = best;
    }
    return text;
}

// Measured 2026-11-02: all three M3.1 line fixtures (sans/serif/mono, digits,
// brackets, & and punctuation) are read exactly.
TEST(RecModel, DocumentedPreprocessingReadsLineFixturesExactly) {
    REQUIRE_MODEL();
    for (const char* name : {"line_pangram_sans", "line_digits_serif", "line_punct_mono"}) {
        std::ifstream gt(std::string(CUDAOCR_SYNTH_DIR) + "/" + name + ".gt.txt");
        std::string want;
        std::getline(gt, want);
        EXPECT_EQ(recognizeLineFixture(name), want) << name;
    }
}

}  // namespace
