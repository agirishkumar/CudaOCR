// M3.9: image in, text out. Requires the fetched model (skips otherwise).
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "cudaocr/components/onnx_line_recognizer.hpp"
#include "cudaocr/eval/metrics.hpp"
#include "cudaocr/pipeline.hpp"

namespace cudaocr {
namespace {

namespace fs = std::filesystem;
const fs::path kModelDir = CUDAOCR_MODEL_DIR;

#define REQUIRE_MODEL()                      \
    if (!fs::exists(kModelDir / "rec.onnx")) \
    GTEST_SKIP() << "models not fetched; run scripts/fetch_models.sh"

PipelineConfig ocrConfig(const std::string& ep = "cuda") {
    PipelineConfig cfg;
    cfg.recognizer = ComponentSpec{
        "onnx", Params().set("model", (kModelDir / "rec.onnx").string()).set("ep", ep)};
    return cfg;
}

std::string readFile(const fs::path& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string ocr(Pipeline& p, const fs::path& image) {
    auto img = loadImage(image.string());
    EXPECT_TRUE(img.ok()) << img.status().message;
    auto r = p.run(img.value());
    EXPECT_TRUE(r.ok()) << r.status().message;
    if (!r.ok()) return {};
    auto fmt = ComponentRegistry::withBuiltins().make<Formatter>({"txt", {}});
    EXPECT_TRUE(fmt.ok());
    auto text = fmt.value()->format(r.value());
    EXPECT_TRUE(text.ok());
    return text.value();
}

TEST(OcrEndToEnd, CleanSyntheticPagesUnderFivePercentCer) {
    REQUIRE_MODEL();
    auto pipe = Pipeline::create(ocrConfig());
    ASSERT_TRUE(pipe.ok()) << pipe.status().message;
    eval::ErrorCounts total;
    for (const char* f : {"page_sans_clean", "page_serif_clean", "page_mono_clean"}) {
        SCOPED_TRACE(f);
        const fs::path dir = CUDAOCR_SYNTH_DIR;
        const auto e = eval::charErrors(readFile(dir / (std::string(f) + ".gt.txt")),
                                        ocr(pipe.value(), dir / (std::string(f) + ".png")));
        EXPECT_LE(e.rate(), 0.05);
        total += e;
    }
    RecordProperty("cleanCer", std::to_string(total.rate()));
    EXPECT_LE(total.rate(), 0.05);
}

TEST(OcrEndToEnd, LinesCarryTextAndConfidence) {
    REQUIRE_MODEL();
    auto pipe = Pipeline::create(ocrConfig());
    ASSERT_TRUE(pipe.ok());
    auto img = loadImage((fs::path(CUDAOCR_SYNTH_DIR) / "line_pangram_sans.png").string());
    ASSERT_TRUE(img.ok());
    auto r = pipe.value().run(img.value());
    ASSERT_TRUE(r.ok()) << r.status().message;
    ASSERT_EQ(r.value().lines.size(), 1u);
    EXPECT_EQ(r.value().lines[0].text, "The quick brown fox jumps over the lazy dog.");
    EXPECT_GT(r.value().lines[0].conf, 0.9f);
    EXPECT_LE(r.value().lines[0].conf, 1.0f);
    std::vector<std::string> stages;
    for (const auto& t : r.value().timings) stages.push_back(t.name);
    EXPECT_NE(std::find(stages.begin(), stages.end(), "onnx"), stages.end());
    EXPECT_NE(std::find(stages.begin(), stages.end(), "ctc-greedy"), stages.end());
}

TEST(OcrEndToEnd, RealSampleProducesText) {
    REQUIRE_MODEL();
    auto pipe = Pipeline::create(ocrConfig());
    ASSERT_TRUE(pipe.ok());
    const std::string text = ocr(pipe.value(), fs::path(CUDAOCR_FIXTURE_DIR) / "sample.png");
    EXPECT_GT(text.size(), 200u) << text;  // 35 lines of prose
}

TEST(OcrEndToEnd, CpuAndCudaGiveTheSameText) {
    REQUIRE_MODEL();
    auto cpu = Pipeline::create(ocrConfig("cpu"));
    auto cuda = Pipeline::create(ocrConfig("cuda"));
    ASSERT_TRUE(cpu.ok() && cuda.ok());
    const fs::path page = fs::path(CUDAOCR_SYNTH_DIR) / "page_serif_skew_m4.png";
    EXPECT_EQ(ocr(cpu.value(), page), ocr(cuda.value(), page));
}

TEST(OcrEndToEnd, UnknownRecognizerFailsAtCreate) {
    PipelineConfig cfg;
    cfg.recognizer = ComponentSpec{"tesseract", {}};
    auto p = Pipeline::create(cfg);
    ASSERT_FALSE(p.ok());
    EXPECT_NE(p.status().message.find("recognizer"), std::string::npos) << p.status().message;
}

}  // namespace
}  // namespace cudaocr
